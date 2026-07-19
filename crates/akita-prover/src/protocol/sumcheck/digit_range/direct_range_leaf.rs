//! Stage-1 norm sumcheck prover/verifier for the Akita PCS.
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
//! where `range_image_eval = range_image(stage1_point)` is the carried virtual
//! claim passed into stage 2. The wire field retains its legacy `range_image_evaluation` name
//! until the scheduled wire-vocabulary cutover.
//!
//! ## `basis = 8` specialization
//!
//! With `half = 4` the roots are `{0, 2, 6, 12}`, giving
//!
//! `Q(r) = r * (r - 2) * (r - 6) * (r - 12)`,
//!
//! degree 4, so round polynomials have degree 5.

use super::super::fold_full_prefix_pair;
use super::super::two_round_prefix::{
    build_stage1_bivariate_skip_proof_from_compact_range_image, can_use_stage1_two_round_prefix,
    stage1_b4_digit_from_compact_range_image, stage1_b8_digit_from_compact_range_image,
    Stage1BivariateSkipState,
};
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, Zero};
use akita_sumcheck::{
    fold_evals_in_place, CompactPairFoldLut, EqFactoredSumcheckInstanceProver, EqFactoredUniPoly,
};
use akita_types::DigitRangePlan;

const MAX_DIRECT_RANGE_COEFFICIENTS: usize = 5;

#[derive(Clone, Copy, Debug, Default)]
struct CompactCoeffEntry {
    abs_coeff: u64,
    is_neg: bool,
}

fn polynomial_coefficients_from_integer_roots(roots: &[i128]) -> Vec<i128> {
    let mut coeffs = vec![1i128];
    for &root in roots {
        let mut next = vec![0i128; coeffs.len() + 1];
        for (idx, &coeff) in coeffs.iter().enumerate() {
            next[idx] -= coeff * root;
            next[idx + 1] += coeff;
        }
        coeffs = next;
    }
    coeffs
}

#[derive(Clone)]
struct RangePolynomialPrecomputation<E: FieldCore> {
    dense_coeffs: Vec<E>,
    dense_row_offsets: Vec<usize>,
    degree_q: usize,
    compact_coeff_lut: Vec<CompactCoeffEntry>,
    /// Maps a raw range-image integer (offset by `minimum_range_image`) to a compact index into the
    /// `basis/2`-element valid-value set `{k(k+1) : k = 0..half-1}`.
    range_image_to_index: Vec<u8>,
    valid_range_image_count: usize,
    minimum_range_image: i16,
}

impl<E: FieldCore + FromPrimitiveInt> RangePolynomialPrecomputation<E> {
    fn new(basis: usize) -> Self {
        assert!(
            matches!(basis, 4 | 8),
            "direct range prover requires basis 4 or 8"
        );
        let half = (basis / 2) as i128;
        let pair_offsets: Vec<i128> = (0..half).map(|k| k * (k + 1)).collect();
        let range_coeffs = polynomial_coefficients_from_integer_roots(&pair_offsets);
        let degree_q = range_coeffs.len() - 1;
        let num_rows = degree_q + 1;

        let total_elems = num_rows * (num_rows + 1) / 2;
        let mut dense_int = Vec::with_capacity(total_elems);
        let mut dense_row_offsets = Vec::with_capacity(num_rows + 1);

        for i in 0..num_rows {
            dense_row_offsets.push(dense_int.len());
            let row_len = degree_q - i + 1;
            let mut binom: i128 = 1;
            for k in 0..row_len {
                let m = i + k;
                let coeff = range_coeffs[m] * binom;
                dense_int.push(coeff);
                if k + 1 < row_len {
                    binom = binom * (m as i128 + 1) / (k as i128 + 1);
                }
            }
        }
        dense_row_offsets.push(dense_int.len());
        let dense_coeffs = dense_int.iter().copied().map(E::from_i128).collect();

        let minimum_range_image = 0i16;
        let maximum_range_image_i128 = half * (half - 1);
        assert!(
            maximum_range_image_i128 <= i16::MAX as i128,
            "compact range-image values exceed i16 for basis={basis}"
        );
        let maximum_range_image = maximum_range_image_i128 as i16;
        let raw_range =
            (i32::from(maximum_range_image) - i32::from(minimum_range_image) + 1) as usize;
        let valid_range_image_count = half as usize;

        let mut range_image_to_index = vec![u8::MAX; raw_range];
        for (compact_idx, &range_image_value) in pair_offsets.iter().enumerate() {
            range_image_to_index[(range_image_value as i16 - minimum_range_image) as usize] =
                compact_idx as u8;
        }

        let mut valid_range_image_lut_int = vec![0i128; valid_range_image_count * num_rows];
        for (compact_idx, &range_image_value) in pair_offsets.iter().enumerate() {
            for i in 0..num_rows {
                let row = &dense_int[dense_row_offsets[i]..dense_row_offsets[i + 1]];
                let mut h: i128 = 0;
                for &c in row.iter().rev() {
                    h = h * range_image_value + c;
                }
                valid_range_image_lut_int[compact_idx * num_rows + i] = h;
            }
        }

        let mut compact_coeff_lut =
            Vec::with_capacity(valid_range_image_count * valid_range_image_count * num_rows);
        for (left_index, &left_value) in pair_offsets.iter().enumerate() {
            let h_base = left_index * num_rows;
            for &right_value in &pair_offsets {
                let delta = right_value - left_value;
                let mut delta_pow = 1i128;
                for &h_i in &valid_range_image_lut_int[h_base..h_base + num_rows] {
                    let coeff = h_i
                        .checked_mul(delta_pow)
                        .expect("compact affine coefficient overflow");
                    let abs_coeff = coeff.unsigned_abs();
                    assert!(
                        abs_coeff <= u64::MAX as u128,
                        "compact affine coefficient exceeds u64"
                    );
                    compact_coeff_lut.push(CompactCoeffEntry {
                        abs_coeff: abs_coeff as u64,
                        is_neg: coeff < 0,
                    });
                    delta_pow = delta_pow
                        .checked_mul(delta)
                        .expect("compact affine power overflow");
                }
            }
        }

        Self {
            dense_coeffs,
            dense_row_offsets,
            degree_q,
            compact_coeff_lut,
            range_image_to_index,
            valid_range_image_count,
            minimum_range_image,
        }
    }
}

impl<E: FieldCore> RangePolynomialPrecomputation<E> {
    #[inline]
    fn compact_index(&self, range_image_integer: i16) -> usize {
        let raw = (range_image_integer - self.minimum_range_image) as usize;
        debug_assert!(raw < self.range_image_to_index.len());
        let ci = self.range_image_to_index[raw];
        debug_assert_ne!(
            ci,
            u8::MAX,
            "range_image={range_image_integer} is not a valid w*(w+1) value"
        );
        ci as usize
    }

    fn num_rows(&self) -> usize {
        self.degree_q + 1
    }

    #[inline]
    fn dense_row(&self, i: usize) -> &[E] {
        &self.dense_coeffs[self.dense_row_offsets[i]..self.dense_row_offsets[i + 1]]
    }

    #[inline]
    fn pair_coeff_lut_start(
        &self,
        left_range_image_integer: i16,
        right_range_image_integer: i16,
    ) -> usize {
        let pair_idx = self.compact_index(left_range_image_integer) * self.valid_range_image_count
            + self.compact_index(right_range_image_integer);
        pair_idx * self.num_rows()
    }

    #[inline]
    fn compact_coeffs_lut(
        &self,
        left_range_image_integer: i16,
        right_range_image_integer: i16,
    ) -> &[CompactCoeffEntry] {
        let num_rows = self.num_rows();
        let start = self.pair_coeff_lut_start(left_range_image_integer, right_range_image_integer);
        &self.compact_coeff_lut[start..start + num_rows]
    }
}

#[inline]
fn accumulate_compact_coeff_slot<E: FieldCore + HasUnreducedOps>(
    pos_accum: &mut [E::MulU64Accum],
    neg_accum: &mut [E::MulU64Accum],
    slot: usize,
    e_in: E,
    coeff: &CompactCoeffEntry,
) {
    if coeff.abs_coeff == 0 {
        return;
    }
    let prod = e_in.mul_u64_unreduced(coeff.abs_coeff);
    if coeff.is_neg {
        neg_accum[slot] += prod;
    } else {
        pos_accum[slot] += prod;
    }
}

#[inline]
fn accumulate_compact_coeffs<E: FieldCore + HasUnreducedOps>(
    pos_accum: &mut [E::MulU64Accum],
    neg_accum: &mut [E::MulU64Accum],
    e_in: E,
    coeffs: &[CompactCoeffEntry],
) {
    debug_assert_eq!(pos_accum.len(), neg_accum.len());
    debug_assert!(pos_accum.len() >= coeffs.len());

    for (idx, coeff) in coeffs.iter().enumerate().take(pos_accum.len()) {
        accumulate_compact_coeff_slot(pos_accum, neg_accum, idx, e_in, coeff);
    }
}

#[inline]
fn reduce_small_coeff_accum<E: FieldCore + HasUnreducedOps>(
    pos: E::MulU64Accum,
    neg: E::MulU64Accum,
) -> E {
    E::reduce_mul_u64_accum(pos) - E::reduce_mul_u64_accum(neg)
}

#[inline]
fn accumulate_dense_entry_coeffs<E: FieldCore + HasUnreducedOps>(
    accum: &mut [E::ProductAccum],
    entry_coeffs: &[E],
    e_in: E,
) {
    if accum.is_empty() {
        return;
    }

    for (acc, &entry) in accum.iter_mut().zip(entry_coeffs.iter()) {
        *acc += e_in.mul_to_product_accum(entry);
    }
}

#[inline]
fn compute_entry_coefficients<E: FieldCore + HasUnreducedOps>(
    out: &mut [E],
    precomp: &RangePolynomialPrecomputation<E>,
    left_range_image: E,
    range_image_delta: E,
) {
    let num_rows = precomp.num_rows();
    debug_assert!(out.len() >= num_rows);

    let mut a_pow = E::one();
    for (i, out_i) in out.iter_mut().enumerate().take(num_rows) {
        let mut h_i = E::zero();
        for &coeff in precomp.dense_row(i).iter().rev() {
            h_i = h_i * left_range_image + coeff;
        }
        *out_i = a_pow * h_i;
        a_pow *= range_image_delta;
    }
}

#[inline]
fn compute_entry_coefficients_x4<E: FieldCore + HasUnreducedOps>(
    out: &mut [[E; MAX_DIRECT_RANGE_COEFFICIENTS]; 4],
    precomp: &RangePolynomialPrecomputation<E>,
    left_range_image: [E; 4],
    range_image_delta: [E; 4],
) {
    let num_rows = precomp.num_rows();

    let mut ap = [E::one(); 4];
    let [out0, out1, out2, out3] = out.each_mut();
    for (i, (((out0_i, out1_i), out2_i), out3_i)) in out0
        .iter_mut()
        .zip(out1.iter_mut())
        .zip(out2.iter_mut())
        .zip(out3.iter_mut())
        .take(num_rows)
        .enumerate()
    {
        let mut h0 = E::zero();
        let mut h1 = E::zero();
        let mut h2 = E::zero();
        let mut h3 = E::zero();
        for &coeff in precomp.dense_row(i).iter().rev() {
            h0 = h0 * left_range_image[0] + coeff;
            h1 = h1 * left_range_image[1] + coeff;
            h2 = h2 * left_range_image[2] + coeff;
            h3 = h3 * left_range_image[3] + coeff;
        }

        *out0_i = ap[0] * h0;
        *out1_i = ap[1] * h1;
        *out2_i = ap[2] * h2;
        *out3_i = ap[3] * h3;

        ap[0] *= range_image_delta[0];
        ap[1] *= range_image_delta[1];
        ap[2] *= range_image_delta[2];
        ap[3] *= range_image_delta[3];
    }
}

fn compute_range_round_polynomial_from_range_image<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
>(
    split_eq: &GruenSplitEq<E>,
    polynomial_precomputation: &RangePolynomialPrecomputation<E>,
    range_image_pair: impl Fn(usize) -> (E, E) + Sync,
) -> EqFactoredUniPoly<E> {
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();
    let full_num_coeffs_q = polynomial_precomputation.degree_q + 1;
    let num_coeffs_q = full_num_coeffs_q;

    let q_coeffs = cfg_fold_reduce!(
        0..e_second.len(),
        || vec![E::ProductAccum::zero(); num_coeffs_q],
        |mut outer_accum, j_high| {
            debug_assert!(full_num_coeffs_q <= MAX_DIRECT_RANGE_COEFFICIENTS);
            let mut inner_accum = [E::ProductAccum::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
            let base_j = j_high * num_first;
            let full_chunks = e_first.len() / 4;
            let mut batch_out = [[E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS]; 4];

            for chunk in 0..full_chunks {
                let jl = chunk * 4;
                let pairs = [
                    range_image_pair(base_j + jl),
                    range_image_pair(base_j + jl + 1),
                    range_image_pair(base_j + jl + 2),
                    range_image_pair(base_j + jl + 3),
                ];
                compute_entry_coefficients_x4(
                    &mut batch_out,
                    polynomial_precomputation,
                    [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0],
                    [
                        pairs[0].1 - pairs[0].0,
                        pairs[1].1 - pairs[1].0,
                        pairs[2].1 - pairs[2].0,
                        pairs[3].1 - pairs[3].0,
                    ],
                );
                for (b_idx, bo) in batch_out.iter().enumerate() {
                    let e_in = e_first[jl + b_idx];
                    accumulate_dense_entry_coeffs(
                        &mut inner_accum[..num_coeffs_q],
                        &bo[..full_num_coeffs_q],
                        e_in,
                    );
                }
            }

            let mut entry_buf = [E::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
            for (tail_idx, &e_in) in e_first[full_chunks * 4..].iter().enumerate() {
                let j = base_j + full_chunks * 4 + tail_idx;
                let (left_range_image, right_range_image) = range_image_pair(j);
                compute_entry_coefficients(
                    &mut entry_buf,
                    polynomial_precomputation,
                    left_range_image,
                    right_range_image - left_range_image,
                );
                accumulate_dense_entry_coeffs(
                    &mut inner_accum[..num_coeffs_q],
                    &entry_buf[..full_num_coeffs_q],
                    e_in,
                );
            }

            let e_out = e_second[j_high];
            for k in 0..num_coeffs_q {
                let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
            }
            outer_accum
        },
        |mut a, b_vec| {
            for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                *ai += *bi;
            }
            a
        }
    )
    .into_iter()
    .map(E::reduce_product_accum)
    .collect::<Vec<_>>();

    let _ = split_eq;
    EqFactoredUniPoly::from_q_coeffs(q_coeffs)
}

fn compute_range_round_polynomial_from_compact_image_pairs<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
>(
    split_eq: &GruenSplitEq<E>,
    polynomial_precomputation: &RangePolynomialPrecomputation<E>,
    range_image_pair: impl Fn(usize) -> (i16, i16) + Sync,
) -> EqFactoredUniPoly<E> {
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();

    let full_num_coeffs_q = polynomial_precomputation.degree_q + 1;
    let num_coeffs_q = full_num_coeffs_q;

    let q_coeffs = cfg_fold_reduce!(
        0..e_second.len(),
        || vec![E::ProductAccum::zero(); num_coeffs_q],
        |mut outer_accum, j_high| {
            debug_assert!(full_num_coeffs_q <= MAX_DIRECT_RANGE_COEFFICIENTS);
            let mut inner_pos = [E::MulU64Accum::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
            let mut inner_neg = [E::MulU64Accum::zero(); MAX_DIRECT_RANGE_COEFFICIENTS];
            for (j_low, &e_in) in e_first.iter().enumerate() {
                let j = j_high * num_first + j_low;
                let (left_range_image_integer, right_range_image_integer) = range_image_pair(j);
                let coeffs = polynomial_precomputation
                    .compact_coeffs_lut(left_range_image_integer, right_range_image_integer);
                accumulate_compact_coeffs(
                    &mut inner_pos[..num_coeffs_q],
                    &mut inner_neg[..num_coeffs_q],
                    e_in,
                    coeffs,
                );
            }
            let e_out = e_second[j_high];
            for k in 0..num_coeffs_q {
                let inner_reduced = reduce_small_coeff_accum(inner_pos[k], inner_neg[k]);
                outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
            }
            outer_accum
        },
        |mut a, b_vec| {
            for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                *ai += *bi;
            }
            a
        }
    )
    .into_iter()
    .map(E::reduce_product_accum)
    .collect::<Vec<_>>();

    let _ = split_eq;
    EqFactoredUniPoly::from_q_coeffs(q_coeffs)
}

fn compute_range_round_polynomial_from_compact_image<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
    V: CompactRangeImageValue,
>(
    split_eq: &GruenSplitEq<E>,
    compact_range_image: &[V],
    polynomial_precomputation: &RangePolynomialPrecomputation<E>,
) -> EqFactoredUniPoly<E> {
    compute_range_round_polynomial_from_compact_image_pairs(
        split_eq,
        polynomial_precomputation,
        |j| {
            (
                compact_range_image[2 * j].range_image_value(),
                compact_range_image[2 * j + 1].range_image_value(),
            )
        },
    )
}

enum LowBasisRangeImageStorage<E: FieldCore> {
    Compact(std::sync::Arc<[i8]>),
    Materialized(Vec<E>),
}

pub(crate) trait CompactRangeImageValue: Copy + Send + Sync {
    fn range_image_value(self) -> i16;
}

impl CompactRangeImageValue for i16 {
    #[inline(always)]
    fn range_image_value(self) -> i16 {
        self
    }
}

impl CompactRangeImageValue for i8 {
    #[inline(always)]
    fn range_image_value(self) -> i16 {
        range_image_from_digit(self)
    }
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

struct DirectRangePrefixState<E: FieldCore> {
    skip_state: Stage1BivariateSkipState<E>,
    first_challenge: Option<E>,
    second_challenge: Option<E>,
}

/// Direct leaf state over `range_image(x) = w(x)(w(x)+1)`.
pub(crate) struct LowBasisRangeCheckProver<E: FieldCore> {
    range_image: LowBasisRangeImageStorage<E>,
    split_eq: GruenSplitEq<E>,
    polynomial_precomputation: RangePolynomialPrecomputation<E>,
    live_x_cols: usize,
    col_bits: usize,
    num_vars: usize,
    basis: usize,
    prefix_tau: Option<Vec<E>>,
    initial_round_prefix: Option<DirectRangePrefixState<E>>,
    cached_round_poly: Option<EqFactoredUniPoly<E>>,
    rounds_completed: usize,
}

mod initial_round_deferral;
mod live_prefix;
mod rounds;
mod sparse_low_variables;
mod state;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use rounds::pad_compact_witness;
