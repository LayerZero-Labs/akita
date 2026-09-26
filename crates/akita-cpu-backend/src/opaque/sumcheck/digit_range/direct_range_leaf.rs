//! Small-basis direct digit-range sum-check prover for the Akita PCS.
//!
//! The committed witness is a Boolean table
//! `w : {0,1}^{col_bits} x {0,1}^{ring_bits} -> {-half, ..., half-1}` with
//! `half = basis/2`. Define the virtual table
//! `range_image(z) = w(z) * (w(z) + 1)`. For an honest witness every entry of
//! `w` is a valid digit, so `range_image(z)` lies in the
//! set `{k(k+1) : k = 0, ..., half-1}`. The range-check polynomial
//!
//! `Q(r) = prod_{k=0}^{half-1} (r - k(k+1))`
//!
//! has degree `basis/2` and vanishes on exactly that set. The sumcheck proves
//!
//! `0 = sum_z eq(tau0, z) * Q(range_image(z))`,
//!
//! where the input claim is `0` (an honest prover makes every summand vanish).
//! Stage 1 uses the generic eq-factored sumcheck path: each round writes the
//! full polynomial as `p(X) = l(X) * q(X)`, where `l` is the linear eq factor
//! for the current round and `q` has degree `basis/2`. The proof sends the
//! headerless `q` message with its linear term omitted, rather than the full
//! degree-`basis/2 + 1` product polynomial. After all rounds, at `stage1_point`, the
//! verifier checks
//!
//! `eq(tau0, stage1_point) * Q(range_image_eval)`
//!
//! where
//!
//! `range_image_eval = sum_z eq(stage1_point, z) * range_image(z)`
//!
//! is the carried multilinear-table claim passed into Stage 2. It is not generally
//! `w(stage1_point) * (w(stage1_point) + 1)` away from Boolean points. The wire field
//! retains its legacy `range_image_evaluation` name until the scheduled wire-vocabulary
//! cutover.
//!
//! ## `basis = 8` specialization
//!
//! With `half = 4` the roots are `{0, 2, 6, 12}`, giving
//!
//! `Q(r) = r * (r - 2) * (r - 6) * (r - 12)`,
//!
//! degree 4, so round polynomials have degree 5.

use crate::opaque::sumcheck::fold_prefix_pair_with_zero_padding;
use crate::opaque::sumcheck::two_round_prefix::{
    build_stage1_prefix_cache, range_polynomial_eval, Stage1PrefixCache,
};
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_sumcheck::{fold_evals_in_place, EqFactoredSumcheckInstanceProver};
use akita_types::DigitRangePlan;
use jolt_field::solinas::parallel::*;
use jolt_field::{Field, Ring, Zero};
use jolt_field::{Fold, Unreduced};
use jolt_poly::OmittedConstantPoly;
use std::ops::Range;

use crate::sources::packed_digits::PackedSignedDigits;

/// Nonconstant round coefficients `q_1..q_d` of the range polynomial. The
/// eq-factored driver recovers `q_0` from the running claim, so no kernel
/// computes it.
const MAX_DIRECT_RANGE_COEFFICIENTS: usize = 4;

/// What the first of a round's per-pair sums holds; every later sum is a
/// Taylor term divided by its integer factor.
#[derive(Clone, Copy)]
enum LinearSum {
    /// The linear Taylor term, divided by its factor like the others.
    Taylor,
    /// `Q(right) - Q(left)`, the sum of all nonconstant coefficients, which
    /// table-driven rounds read without computing the linear term.
    RangeDifference,
}

#[derive(Clone)]
struct RangePolynomialPrecomputation {
    degree_q: usize,
}

impl RangePolynomialPrecomputation {
    fn new(basis: usize) -> Self {
        assert!(
            matches!(basis, 4 | 8),
            "direct range prover requires basis 4 or 8"
        );
        Self {
            degree_q: basis / 2,
        }
    }

    /// Number of nonconstant round coefficients each pair contributes.
    fn num_coefficients(&self) -> usize {
        self.degree_q
    }

    /// Integer factor of each nonconstant Taylor coefficient of
    /// `Q(left + X * delta)`, which the per-pair kernels leave out and
    /// [`round_poly_from_sums`](Self::round_poly_from_sums) applies once.
    fn taylor_factors(&self) -> &'static [u64] {
        match self.degree_q {
            2 => &[2, 1],
            4 => &[4, 6, 4, 1],
            _ => unreachable!("direct range leaf only supports quadratic and quartic checks"),
        }
    }

    /// Reduce a round's accumulated per-pair sums into its nonconstant
    /// coefficients.
    fn round_poly_from_sums<E: Field + Ring + Unreduced>(
        &self,
        sums: &TaylorSums<E>,
        linear: LinearSum,
    ) -> OmittedConstantPoly<E> {
        let mut coefficients: Vec<E> = sums
            .iter()
            .zip(self.taylor_factors())
            .map(|(&sum, &factor)| E::from_u64(factor) * E::reduce_product(sum))
            .collect();
        if let LinearSum::RangeDifference = linear {
            let higher: E = coefficients[1..].iter().copied().sum();
            coefficients[0] = E::reduce_product(sums[0]) - higher;
        }
        OmittedConstantPoly::new(coefficients)
    }
}

/// Nonconstant Taylor sums of a round, before
/// [`RangePolynomialPrecomputation::round_poly_from_sums`] scales them.
type TaylorSums<E> = [<E as Unreduced>::Product; MAX_DIRECT_RANGE_COEFFICIENTS];

/// Add `weight` times the nonconstant Taylor terms of `Q(left + X * delta)`,
/// each divided by its integer factor from
/// [`RangePolynomialPrecomputation::taylor_factors`], to `sums`.
///
/// With `b = left - 5`, the quartic range polynomial is
/// `Q = b^4 - 42 b^2 - 64 b + 105`, so its terms at `left` are
/// `4((b^2 - 21) b - 16) delta`, `6(b^2 - 7) delta^2`, `4 b delta^3` and
/// `delta^4`. Every term carries a power of `delta`, so the weight rides on
/// the powers `weight * delta^k`: five multiplications, one squaring and four
/// unreduced products. The quadratic terms are `2(left - 1) delta` and
/// `delta^2`.
#[inline(always)]
fn accumulate_entry_terms<E: Field + Ring + Unreduced>(
    sums: &mut TaylorSums<E>,
    precomp: &RangePolynomialPrecomputation,
    left_range_image: E,
    range_image_delta: E,
    weight: E,
) {
    let weighted_delta = weight * range_image_delta;
    match precomp.degree_q {
        2 => {
            sums[0] += (left_range_image - E::one()).mul_unreduced(weighted_delta);
            sums[1] += range_image_delta.mul_unreduced(weighted_delta);
        }
        4 => {
            let shifted = left_range_image - E::from_u64(5);
            let shifted_squared = shifted.square();
            let weighted_delta_squared = weighted_delta * range_image_delta;
            let weighted_delta_cubed = weighted_delta_squared * range_image_delta;
            sums[0] += ((shifted_squared - E::from_u64(21)) * shifted - E::from_u64(16))
                .mul_unreduced(weighted_delta);
            sums[1] += (shifted_squared - E::from_u64(7)).mul_unreduced(weighted_delta_squared);
            sums[2] += shifted.mul_unreduced(weighted_delta_cubed);
            sums[3] += range_image_delta.mul_unreduced(weighted_delta_cubed);
        }
        _ => unreachable!("direct range leaf only supports quadratic and quartic checks"),
    }
}

fn compute_range_round_polynomial_from_range_image<E: Field + Ring + Unreduced>(
    split_eq: &GruenSplitEq<E>,
    polynomial_precomputation: &RangePolynomialPrecomputation,
    range_image_pair: impl Fn(usize) -> (E, E) + Sync,
) -> OmittedConstantPoly<E> {
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();
    let num_coeffs_q = polynomial_precomputation.num_coefficients();

    let sums = cfg_fold_reduce!(
        0..e_second.len(),
        || [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS],
        |mut outer_accum, j_high| {
            let mut inner_accum = [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
            for (j_low, &e_in) in e_first.iter().enumerate() {
                let (left_range_image, right_range_image) =
                    range_image_pair(j_high * num_first + j_low);
                accumulate_entry_terms(
                    &mut inner_accum,
                    polynomial_precomputation,
                    left_range_image,
                    right_range_image - left_range_image,
                    e_in,
                );
            }

            let e_out = e_second[j_high];
            for (outer, inner) in outer_accum.iter_mut().zip(inner_accum).take(num_coeffs_q) {
                *outer += e_out.mul_unreduced(E::reduce_product(inner));
            }
            outer_accum
        },
        |mut a, b| {
            for (ai, bi) in a.iter_mut().zip(b) {
                *ai += bi;
            }
            a
        }
    );
    polynomial_precomputation.round_poly_from_sums(&sums, LinearSum::Taylor)
}

enum LowBasisRangeImageStorage<E: Field> {
    Compact(PackedSignedDigits),
    Materialized(Vec<E>),
}

#[inline]
fn range_image_from_digit(w: i8) -> i16 {
    let w = i32::from(w);
    let range_image = w * (w + 1);
    debug_assert!(range_image >= 0);
    range_image as i16
}

#[cfg(test)]
fn build_compact_range_image(digit_witness: &[i8]) -> Vec<i16> {
    digit_witness
        .iter()
        .copied()
        .map(range_image_from_digit)
        .collect()
}

/// Compact-table state for rounds 0 through 3, which bind the three digit
/// positions inside an octet and then pair adjacent octets.
struct DirectRangePrefixState<E: Field> {
    cache: Stage1PrefixCache<E>,
    /// Sum of the round-2 equality weight `eq(tau0[3..], octet)` over the live
    /// octets of each octet class.
    octet_class_weights: Vec<E>,
    /// Sum of the round-3 equality weight `eq(tau0[4..], pair)` over the live
    /// octet pairs whose even (`[0]`) or odd (`[1]`) octet has each class.
    octet_pair_class_weights: Vec<[E; 2]>,
    first_challenge: Option<E>,
    /// The folded value of each quad class after round 1.
    quad_values: Vec<E>,
    /// The round-3 terms of each octet class after round 2.
    octet_terms: Vec<octet_prefix::OctetClassTerms<E>>,
}

/// Direct leaf state over `range_image(x) = w(x)(w(x)+1)`.
///
/// For `basis <= 8` this is the complete stage-1 instance prover: it
/// implements [`EqFactoredSumcheckInstanceProver`], so callers that batch
/// several stage-1 instances over one shared equality point can drive its
/// rounds directly instead of going through the single-instance
/// `DigitRangeProver` wrapper.
pub struct LowBasisRangeCheckProver<E: Field> {
    range_image: LowBasisRangeImageStorage<E>,
    split_eq: GruenSplitEq<E>,
    polynomial_precomputation: RangePolynomialPrecomputation,
    live_x_cols: usize,
    col_bits: usize,
    num_vars: usize,
    basis: usize,
    prefix_tau: Option<Vec<E>>,
    initial_round_prefix: Option<DirectRangePrefixState<E>>,
    cached_round_poly: Option<OmittedConstantPoly<E>>,
    rounds_completed: usize,
}

mod live_prefix;
mod octet_prefix;
mod rounds;
mod state;

#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use rounds::pad_compact_witness;
