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

use self::prefix_cache::{build_stage1_prefix_cache, Stage1PrefixCache};
use super::range_poly::{
    LinearSum, OctetClassTerms, RangePoly, TaylorSums, MAX_DIRECT_RANGE_COEFFICIENTS,
};
use crate::opaque::sumcheck::fold_prefix_pair_with_zero_padding;
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

fn compute_range_round_polynomial_from_range_image<E: Field + Ring + Unreduced>(
    split_eq: &GruenSplitEq<E>,
    range_poly: &RangePoly,
    range_image_pair: impl Fn(usize) -> (E, E) + Sync,
) -> OmittedConstantPoly<E> {
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();
    let num_coeffs_q = range_poly.num_coefficients();

    let sums = cfg_fold_reduce!(
        0..e_second.len(),
        || [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS],
        |mut outer_accum, j_high| {
            let mut inner_accum = [E::Product::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
            for (j_low, &e_in) in e_first.iter().enumerate() {
                let (left_range_image, right_range_image) =
                    range_image_pair(j_high * num_first + j_low);
                range_poly.accumulate_entry_terms(
                    &mut inner_accum,
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
    range_poly.round_poly_from_sums(&sums, LinearSum::Taylor)
}

enum LowBasisRangeImageStorage<E: Field> {
    OctetPrefix(OctetPrefix<E>),
    Materialized(Vec<E>),
}

struct OctetPrefix<E: Field> {
    digits: PackedSignedDigits,
    tau: Vec<E>,
    state: Option<DirectRangePrefixState<E>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoundKernel {
    OctetPrefix,
    LivePrefix,
    Dense,
}

#[inline]
fn range_image_from_digit(w: i8) -> i16 {
    let w = i32::from(w);
    let range_image = w * (w + 1);
    debug_assert!(range_image >= 0);
    range_image as i16
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
    octet_terms: Vec<OctetClassTerms<E>>,
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
    range_poly: RangePoly,
    live_x_cols: usize,
    col_bits: usize,
    num_vars: usize,
    basis: usize,
    cached_round_poly: Option<OmittedConstantPoly<E>>,
    rounds_completed: usize,
}

mod live_prefix;
mod octet_prefix;
mod prefix_cache;
mod rounds;
mod state;

#[cfg(test)]
mod kernel_tests;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
pub(crate) use tests::pad_compact_witness;
