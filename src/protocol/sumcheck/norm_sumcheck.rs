//! Norm (range-check) sumcheck instance (F_0).
//!
//! **F_{0,τ₀}(x, y)** = ẽq(τ₀,(x,y)) · w̃(x,y) · (w̃−1)(w̃+1)···(w̃−b+1)(w̃+b−1)
//!
//! Proves that all entries of w̃ lie in {−(b−1), …, b−1}; the sum over the
//! boolean hypercube should equal zero when the range constraint holds.

use super::eq_poly::EqPolynomial;
use super::split_eq::GruenSplitEq;
use super::{fold_evals_in_place, multilinear_eval, range_check_eval};
use super::{SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly};
use crate::algebra::fields::HasUnreducedOps;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::{cfg_fold_reduce, AdditiveGroup, CanonicalField, FieldCore, FromSmallInt};

/// Max number of affine coefficient rows (degree_q + 1) for `b <= 8`.
pub(crate) const MAX_AFFINE_COEFFS: usize = 16;

/// Which kernel to use for the norm sumcheck accumulation loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormRoundKernel {
    /// Evaluate the range-check polynomial at `degree_q+1` points, then interpolate.
    PointEvalInterpolation,
    /// Directly accumulate polynomial coefficients via affine substitution.
    AffineCoeffComposition,
}

/// Select the norm kernel for a given `b`.
///
/// Override with env var `HACHI_NORM_KERNEL=point_eval` or `affine_coeff`.
pub fn choose_round_kernel(b: usize) -> NormRoundKernel {
    if let Ok(v) = std::env::var("HACHI_NORM_KERNEL") {
        match v.as_str() {
            "point_eval" => return NormRoundKernel::PointEvalInterpolation,
            "affine_coeff" => return NormRoundKernel::AffineCoeffComposition,
            _ => {}
        }
    }
    if b <= 8 {
        NormRoundKernel::AffineCoeffComposition
    } else {
        NormRoundKernel::PointEvalInterpolation
    }
}

/// A nonzero coefficient entry in the affine decomposition polynomial.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseCoeffEntry {
    /// Power index: which `w_0^k` this coefficient multiplies.
    pub k: u8,
    /// Absolute value of the mixed coefficient (fits u64 for b <= 8).
    pub abs_coeff: u64,
    /// Sign: true if the coefficient is negative.
    pub is_neg: bool,
}

#[derive(Clone)]
pub(crate) struct RangeAffinePrecomp<E: FieldCore> {
    /// Flat storage of nonzero `coeff_mix[i][k]` entries.
    sparse_entries: Vec<SparseCoeffEntry>,
    /// `sparse_row_offsets[i]..sparse_row_offsets[i+1]` indexes into `sparse_entries`.
    sparse_row_offsets: Vec<usize>,
    pub(crate) degree_q: usize,
    /// Precomputed `h_i(w_0)` for all small-integer `w_0 ∈ {-(b-1),...,b-1}`.
    /// Indexed as `small_w_lut[(w_0 + b - 1) * num_rows + i]`.
    small_w_lut: Vec<E>,
    b: usize,
}

/// Integer version of `range_check_coeffs`: returns the polynomial coefficients
/// of `R(w) = w * Π_{k=1}^{b-1}(w² - k²)` as exact i64 values.
fn range_check_coeffs_int(b: usize) -> Vec<i64> {
    assert!(b >= 1, "b must be at least 1");
    let mut coeffs: Vec<i64> = vec![0, 1];
    for k in 1..b as i64 {
        let k_sq = k * k;
        let mut next = vec![0i64; coeffs.len() + 2];
        for (idx, &c) in coeffs.iter().enumerate() {
            next[idx] -= c * k_sq;
            next[idx + 2] += c;
        }
        coeffs = next;
    }
    coeffs
}

impl<E: FieldCore + FromSmallInt> RangeAffinePrecomp<E> {
    pub(crate) fn new(b: usize) -> Self {
        assert!(b >= 1, "b must be at least 1");

        let range_coeffs = range_check_coeffs_int(b);
        let degree_q = range_coeffs.len() - 1;
        let num_rows = degree_q + 1;

        // Build dense integer coeff_mix and sparse entries simultaneously.
        let total_elems = num_rows * (num_rows + 1) / 2;
        let mut dense_int = Vec::with_capacity(total_elems);
        let mut dense_row_offsets = Vec::with_capacity(num_rows + 1);
        let mut sparse_entries = Vec::new();
        let mut sparse_row_offsets = Vec::with_capacity(num_rows + 1);

        for i in 0..num_rows {
            dense_row_offsets.push(dense_int.len());
            sparse_row_offsets.push(sparse_entries.len());
            let row_len = degree_q - i + 1;
            let mut binom: i64 = 1; // binom(i, i) = 1
            for k in 0..row_len {
                let m = i + k;
                let coeff = range_coeffs[m] * binom;
                dense_int.push(coeff);
                if coeff != 0 {
                    sparse_entries.push(SparseCoeffEntry {
                        k: k as u8,
                        abs_coeff: coeff.unsigned_abs(),
                        is_neg: coeff < 0,
                    });
                }
                if k + 1 < row_len {
                    binom = binom * (m as i64 + 1) / (k as i64 + 1);
                }
            }
        }
        dense_row_offsets.push(dense_int.len());
        sparse_row_offsets.push(sparse_entries.len());

        // Precompute LUT using i128 integer Horner.
        let num_w_vals = 2 * b - 1;
        let mut small_w_lut = vec![E::zero(); num_w_vals * num_rows];
        for (w_idx, w_0_int) in (-(b as i64 - 1)..=(b as i64 - 1)).enumerate() {
            for i in 0..num_rows {
                let row = &dense_int[dense_row_offsets[i]..dense_row_offsets[i + 1]];
                let mut h: i128 = 0;
                for &c in row.iter().rev() {
                    h = h * w_0_int as i128 + c as i128;
                }
                small_w_lut[w_idx * num_rows + i] = E::from_i128(h);
            }
        }

        Self {
            sparse_entries,
            sparse_row_offsets,
            degree_q,
            small_w_lut,
            b,
        }
    }
}

impl<E: FieldCore> RangeAffinePrecomp<E> {
    #[inline]
    pub(crate) fn sparse_row(&self, i: usize) -> &[SparseCoeffEntry] {
        &self.sparse_entries[self.sparse_row_offsets[i]..self.sparse_row_offsets[i + 1]]
    }

    pub(crate) fn num_rows(&self) -> usize {
        self.degree_q + 1
    }

    #[inline]
    pub(crate) fn h_i_lut(&self, w_0_int: i8, i: usize) -> E {
        let w_idx = (w_0_int as i16 + self.b as i16 - 1) as usize;
        self.small_w_lut[w_idx * self.num_rows() + i]
    }
}

#[derive(Clone)]
pub(crate) struct PointEvalPrecomp<E: FieldCore> {
    pub(crate) range_offsets_sq: Vec<E>,
}

impl<E: FieldCore + FromSmallInt> PointEvalPrecomp<E> {
    pub(crate) fn new(b: usize) -> Self {
        assert!(b >= 1, "b must be at least 1");
        let range_offsets_sq = (1..b)
            .map(|k| {
                let k_e = E::from_u64(k as u64);
                k_e * k_e
            })
            .collect();
        Self { range_offsets_sq }
    }
}

/// Evaluate `R(w) = w · Π_{k=1}^{b-1}(w² - k²)` in native `i128` arithmetic.
///
/// Only valid for `b <= 10` (intermediates fit i128; verified up to ~2^117 for b=8).
/// Panics in debug mode if an intermediate overflows.
#[inline]
pub(crate) fn range_check_eval_i128(w: i32, b: usize) -> i128 {
    debug_assert!(b <= 10, "i128 range-check only valid for b <= 10");
    let s = (w as i128) * (w as i128);
    let mut acc = w as i128;
    for k in 1..b as i128 {
        acc = acc
            .checked_mul(s - k * k)
            .expect("i128 overflow in range-check");
    }
    acc
}

/// Convert an `i128` to a field element via `CanonicalField::from_canonical_u128_reduced`.
#[inline]
pub(crate) fn field_from_i128<E: CanonicalField>(val: i128) -> E {
    if val >= 0 {
        E::from_canonical_u128_reduced(val as u128)
    } else {
        -E::from_canonical_u128_reduced(val.unsigned_abs())
    }
}

pub(crate) fn range_check_eval_precomputed<E: FieldCore>(w: E, offsets_sq: &[E]) -> E {
    let s = w * w;
    let mut acc = w;
    for &k_sq in offsets_sq {
        acc = acc * (s - k_sq);
    }
    acc
}

/// Compute per-entry affine range-check coefficients using power-table +
/// sparse unreduced dot product. Writes `a^i · h_i(w_0)` into `out[i]`
/// for `i ∈ 0..precomp.num_rows()`.
///
/// `w_pows` is a caller-provided scratch buffer of length >= `degree_q + 1`.
#[inline]
pub(crate) fn compute_entry_coeffs<E: FieldCore + HasUnreducedOps>(
    out: &mut [E],
    w_pows: &mut [E],
    precomp: &RangeAffinePrecomp<E>,
    w_0: E,
    a: E,
) {
    let deg = precomp.degree_q;
    let num_rows = precomp.num_rows();
    debug_assert!(out.len() >= num_rows);
    debug_assert!(w_pows.len() > deg);

    w_pows[0] = E::one();
    for k in 1..=deg {
        w_pows[k] = w_pows[k - 1] * w_0;
    }

    let mut a_pow = E::one();
    for (i, out_i) in out.iter_mut().enumerate().take(num_rows) {
        let entries = precomp.sparse_row(i);
        let mut pos = E::MulU64Accum::ZERO;
        let mut neg = E::MulU64Accum::ZERO;
        for entry in entries {
            let prod = w_pows[entry.k as usize].mul_u64_unreduced(entry.abs_coeff);
            if entry.is_neg {
                neg += prod;
            } else {
                pos += prod;
            }
        }
        let h_i = E::reduce_mul_u64_accum(pos) - E::reduce_mul_u64_accum(neg);
        *out_i = a_pow * h_i;
        a_pow = a_pow * a;
    }
}

/// Batched version: processes 4 entries simultaneously to expose ILP across
/// independent power-table and sparse-dot-product chains.
#[inline]
pub(crate) fn compute_entry_coeffs_x4<E: FieldCore + HasUnreducedOps>(
    out: &mut [[E; MAX_AFFINE_COEFFS]; 4],
    precomp: &RangeAffinePrecomp<E>,
    w_0: [E; 4],
    a: [E; 4],
) {
    let deg = precomp.degree_q;
    let num_rows = precomp.num_rows();

    let mut pw = [[E::zero(); MAX_AFFINE_COEFFS]; 4];
    for p in &mut pw {
        p[0] = E::one();
    }
    for k in 1..=deg {
        pw[0][k] = pw[0][k - 1] * w_0[0];
        pw[1][k] = pw[1][k - 1] * w_0[1];
        pw[2][k] = pw[2][k - 1] * w_0[2];
        pw[3][k] = pw[3][k - 1] * w_0[3];
    }

    let mut ap = [E::one(); 4];
    for i in 0..num_rows {
        let entries = precomp.sparse_row(i);

        let mut pos0 = E::MulU64Accum::ZERO;
        let mut neg0 = E::MulU64Accum::ZERO;
        let mut pos1 = E::MulU64Accum::ZERO;
        let mut neg1 = E::MulU64Accum::ZERO;
        let mut pos2 = E::MulU64Accum::ZERO;
        let mut neg2 = E::MulU64Accum::ZERO;
        let mut pos3 = E::MulU64Accum::ZERO;
        let mut neg3 = E::MulU64Accum::ZERO;

        for entry in entries {
            let k = entry.k as usize;
            let c = entry.abs_coeff;
            let p0 = pw[0][k].mul_u64_unreduced(c);
            let p1 = pw[1][k].mul_u64_unreduced(c);
            let p2 = pw[2][k].mul_u64_unreduced(c);
            let p3 = pw[3][k].mul_u64_unreduced(c);
            if entry.is_neg {
                neg0 += p0;
                neg1 += p1;
                neg2 += p2;
                neg3 += p3;
            } else {
                pos0 += p0;
                pos1 += p1;
                pos2 += p2;
                pos3 += p3;
            }
        }

        let h0 = E::reduce_mul_u64_accum(pos0) - E::reduce_mul_u64_accum(neg0);
        let h1 = E::reduce_mul_u64_accum(pos1) - E::reduce_mul_u64_accum(neg1);
        let h2 = E::reduce_mul_u64_accum(pos2) - E::reduce_mul_u64_accum(neg2);
        let h3 = E::reduce_mul_u64_accum(pos3) - E::reduce_mul_u64_accum(neg3);

        out[0][i] = ap[0] * h0;
        out[1][i] = ap[1] * h1;
        out[2][i] = ap[2] * h2;
        out[3][i] = ap[3] * h3;

        ap[0] = ap[0] * a[0];
        ap[1] = ap[1] * a[1];
        ap[2] = ap[2] * a[2];
        ap[3] = ap[3] * a[3];
    }
}

pub(crate) fn trim_trailing_zeros<E: FieldCore>(coeffs: &mut Vec<E>) {
    while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
        coeffs.pop();
    }
}

/// Centralized norm round polynomial computation (full field-element path).
///
/// Both `NormSumcheckProver` and `HachiSumcheckProver` delegate here.
pub(crate) fn compute_norm_round_poly<E: FieldCore + FromSmallInt + HasUnreducedOps>(
    split_eq: &GruenSplitEq<E>,
    half: usize,
    b: usize,
    round_kernel: NormRoundKernel,
    point_precomp: Option<&PointEvalPrecomp<E>>,
    range_precomp: Option<&RangeAffinePrecomp<E>>,
    w_pair: impl Fn(usize) -> (E, E) + Sync,
) -> UniPoly<E> {
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();
    let first_bits = num_first.trailing_zeros();

    match round_kernel {
        NormRoundKernel::PointEvalInterpolation => {
            let degree_q = 2 * b - 1;
            let num_points_q = degree_q + 1;
            let offsets_sq = &point_precomp.unwrap().range_offsets_sq;

            let q_evals = {
                let _span = tracing::info_span!("norm_accumulate", kernel = "point_eval").entered();
                cfg_fold_reduce!(
                    0..half,
                    || vec![E::zero(); num_points_q],
                    |mut evals, j| {
                        let j_low = j & (num_first - 1);
                        let j_high = j >> first_bits;
                        let eq_rem = e_first[j_low] * e_second[j_high];
                        let (w_0, w_1) = w_pair(j);
                        let delta = w_1 - w_0;
                        let mut w_t = w_0;
                        for eval in evals.iter_mut() {
                            *eval += eq_rem * range_check_eval_precomputed(w_t, offsets_sq);
                            w_t += delta;
                        }
                        evals
                    },
                    |mut a, b_vec| {
                        for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                            *ai += *bi;
                        }
                        a
                    }
                )
            };

            let q_poly = UniPoly::from_evals(&q_evals);
            split_eq.gruen_mul(&q_poly)
        }
        NormRoundKernel::AffineCoeffComposition => {
            let rp = range_precomp.unwrap();
            let num_coeffs_q = rp.degree_q + 1;

            let mut q_coeffs = {
                let _span =
                    tracing::info_span!("norm_accumulate", kernel = "affine_coeff").entered();

                cfg_fold_reduce!(
                    0..e_second.len(),
                    || vec![E::ProductAccum::ZERO; num_coeffs_q],
                    |mut outer_accum, j_high| {
                        debug_assert!(num_coeffs_q <= MAX_AFFINE_COEFFS);
                        let mut inner_accum = [E::ProductAccum::ZERO; MAX_AFFINE_COEFFS];
                        let base_j = j_high * num_first;
                        let full_chunks = e_first.len() / 4;
                        let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];

                        for chunk in 0..full_chunks {
                            let jl = chunk * 4;
                            let pairs = [
                                w_pair(base_j + jl),
                                w_pair(base_j + jl + 1),
                                w_pair(base_j + jl + 2),
                                w_pair(base_j + jl + 3),
                            ];
                            compute_entry_coeffs_x4(
                                &mut batch_out,
                                rp,
                                [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0],
                                [
                                    pairs[0].1 - pairs[0].0,
                                    pairs[1].1 - pairs[1].0,
                                    pairs[2].1 - pairs[2].0,
                                    pairs[3].1 - pairs[3].0,
                                ],
                            );
                            for (b, bo) in batch_out.iter().enumerate() {
                                let e_in = e_first[jl + b];
                                for (acc, &entry) in inner_accum[..num_coeffs_q]
                                    .iter_mut()
                                    .zip(bo[..num_coeffs_q].iter())
                                {
                                    *acc += e_in.mul_to_product_accum(entry);
                                }
                            }
                        }

                        let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                        let mut w_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];
                        for (tail_idx, &e_in) in e_first[full_chunks * 4..].iter().enumerate() {
                            let j = base_j + full_chunks * 4 + tail_idx;
                            let (w_0, w_1) = w_pair(j);
                            compute_entry_coeffs(
                                &mut entry_buf,
                                &mut w_pows_buf,
                                rp,
                                w_0,
                                w_1 - w_0,
                            );
                            for (acc, &entry) in inner_accum[..num_coeffs_q]
                                .iter_mut()
                                .zip(entry_buf[..num_coeffs_q].iter())
                            {
                                *acc += e_in.mul_to_product_accum(entry);
                            }
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
            }
            .into_iter()
            .map(E::reduce_product_accum)
            .collect::<Vec<_>>();

            trim_trailing_zeros(&mut q_coeffs);
            let q_poly = UniPoly::from_coeffs(q_coeffs);
            split_eq.gruen_mul(&q_poly)
        }
    }
}

/// Compact round-0 variant: uses native i128 arithmetic (point-eval, b<=10)
/// or precomputed LUT (affine-coeff) when w values are small integers.
pub(crate) fn compute_norm_round_poly_compact<
    E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps,
>(
    split_eq: &GruenSplitEq<E>,
    w_compact: &[i8],
    b: usize,
    round_kernel: NormRoundKernel,
    point_precomp: Option<&PointEvalPrecomp<E>>,
    range_precomp: Option<&RangeAffinePrecomp<E>>,
) -> UniPoly<E> {
    let half = w_compact.len() / 2;
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();
    let first_bits = num_first.trailing_zeros();

    match round_kernel {
        NormRoundKernel::PointEvalInterpolation if b <= 10 => {
            let degree_q = 2 * b - 1;
            let num_points_q = degree_q + 1;

            let q_evals = {
                let _span =
                    tracing::info_span!("norm_accumulate", kernel = "point_eval_i128").entered();
                cfg_fold_reduce!(
                    0..half,
                    || vec![E::zero(); num_points_q],
                    |mut evals, j| {
                        let j_low = j & (num_first - 1);
                        let j_high = j >> first_bits;
                        let eq_rem = e_first[j_low] * e_second[j_high];
                        let w0_i = w_compact[2 * j] as i32;
                        let delta_i = w_compact[2 * j + 1] as i32 - w0_i;
                        let mut w_t_i = w0_i;
                        for eval in evals.iter_mut() {
                            let rc = range_check_eval_i128(w_t_i, b);
                            *eval += eq_rem * field_from_i128::<E>(rc);
                            w_t_i += delta_i;
                        }
                        evals
                    },
                    |mut a, b_vec| {
                        for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                            *ai += *bi;
                        }
                        a
                    }
                )
            };

            let q_poly = UniPoly::from_evals(&q_evals);
            split_eq.gruen_mul(&q_poly)
        }
        NormRoundKernel::AffineCoeffComposition => {
            let rp = range_precomp.unwrap();
            let num_coeffs_q = rp.degree_q + 1;

            let mut q_coeffs = {
                let _span =
                    tracing::info_span!("norm_accumulate", kernel = "affine_coeff_lut").entered();

                cfg_fold_reduce!(
                    0..e_second.len(),
                    || vec![E::ProductAccum::ZERO; num_coeffs_q],
                    |mut outer_accum, j_high| {
                        debug_assert!(num_coeffs_q <= MAX_AFFINE_COEFFS);
                        let mut inner_accum = [E::ProductAccum::ZERO; MAX_AFFINE_COEFFS];
                        for (j_low, &e_in) in e_first.iter().enumerate() {
                            let j = j_high * num_first + j_low;
                            let w_0_int = w_compact[2 * j];
                            let w_1 = E::from_i64(w_compact[2 * j + 1] as i64);
                            let a = w_1 - E::from_i64(w_0_int as i64);
                            let mut a_pow = E::one();
                            for (i, acc) in inner_accum[..num_coeffs_q].iter_mut().enumerate() {
                                let h_i_w0 = rp.h_i_lut(w_0_int, i);
                                let val = a_pow * h_i_w0;
                                *acc += e_in.mul_to_product_accum(val);
                                a_pow = a_pow * a;
                            }
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
            }
            .into_iter()
            .map(E::reduce_product_accum)
            .collect::<Vec<_>>();

            trim_trailing_zeros(&mut q_coeffs);
            let q_poly = UniPoly::from_coeffs(q_coeffs);
            split_eq.gruen_mul(&q_poly)
        }
        _ => {
            // b > 10 with point-eval: fall back to field-element path
            let pair = |j: usize| {
                (
                    E::from_i64(w_compact[2 * j] as i64),
                    E::from_i64(w_compact[2 * j + 1] as i64),
                )
            };
            compute_norm_round_poly(
                split_eq,
                half,
                b,
                round_kernel,
                point_precomp,
                range_precomp,
                pair,
            )
        }
    }
}

/// Prover for `F_{0,τ₀}(x,y) = ẽq(τ₀,(x,y)) · w̃(x,y) · range_check(w̃(x,y), b)`.
///
/// Uses the Gruen/Dao-Thaler optimization: the eq polynomial is factored into
/// a running scalar and split tables instead of being stored as a full table
/// and folded each round. The round polynomial is computed as `l(X) · q(X)`
/// where `l(X)` is the linear eq factor and `q(X)` is the inner sum without
/// the current-variable eq contribution.
pub struct NormSumcheckProver<E: FieldCore> {
    split_eq: GruenSplitEq<E>,
    w_table: Vec<E>,
    round_kernel: NormRoundKernel,
    point_precomp: Option<PointEvalPrecomp<E>>,
    range_precomp: Option<RangeAffinePrecomp<E>>,
    num_vars: usize,
    b: usize,
}

impl<E: FieldCore + FromSmallInt + HasUnreducedOps> NormSumcheckProver<E> {
    /// Create a new norm (range-check) sumcheck prover.
    ///
    /// # Panics
    ///
    /// Panics if `w_evals.len() != 2^tau.len()`.
    pub fn new(tau: &[E], w_evals: Vec<E>, b: usize) -> Self {
        Self::new_with_kernel(tau, w_evals, b, choose_round_kernel(b))
    }

    fn new_with_kernel(
        tau: &[E],
        w_evals: Vec<E>,
        b: usize,
        round_kernel: NormRoundKernel,
    ) -> Self {
        assert!(b >= 1, "b must be at least 1");
        let num_vars = tau.len();
        assert_eq!(w_evals.len(), 1 << num_vars);
        let point_precomp = match round_kernel {
            NormRoundKernel::PointEvalInterpolation => Some(PointEvalPrecomp::new(b)),
            NormRoundKernel::AffineCoeffComposition => None,
        };
        let range_precomp = match round_kernel {
            NormRoundKernel::PointEvalInterpolation => None,
            NormRoundKernel::AffineCoeffComposition => Some(RangeAffinePrecomp::new(b)),
        };
        Self {
            split_eq: GruenSplitEq::new(tau),
            w_table: w_evals,
            round_kernel,
            point_precomp,
            range_precomp,
            num_vars,
            b,
        }
    }
}

impl<E: FieldCore + FromSmallInt + HasUnreducedOps> SumcheckInstanceProver<E>
    for NormSumcheckProver<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2 * self.b
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let half = self.w_table.len() / 2;
        let w_table = &self.w_table;
        compute_norm_round_poly(
            &self.split_eq,
            half,
            self.b,
            self.round_kernel,
            self.point_precomp.as_ref(),
            self.range_precomp.as_ref(),
            |j| (w_table[2 * j], w_table[2 * j + 1]),
        )
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        self.split_eq.bind(r);
        fold_evals_in_place(&mut self.w_table, r);
    }
}

/// Verifier for the norm (range-check) sumcheck `F_{0,τ₀}`.
pub struct NormSumcheckVerifier<E> {
    tau: Vec<E>,
    w_evals: Vec<E>,
    num_vars: usize,
    b: usize,
}

impl<E: FieldCore + FromSmallInt> NormSumcheckVerifier<E> {
    /// Create a new norm (range-check) sumcheck verifier.
    ///
    /// # Panics
    ///
    /// Panics if `w_evals.len() != 2^tau.len()`.
    pub fn new(tau: Vec<E>, w_evals: Vec<E>, b: usize) -> Self {
        let num_vars = tau.len();
        assert_eq!(w_evals.len(), 1 << num_vars);
        Self {
            tau,
            w_evals,
            num_vars,
            b,
        }
    }
}

impl<E: FieldCore + FromSmallInt> SumcheckInstanceVerifier<E> for NormSumcheckVerifier<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2 * self.b
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, HachiError> {
        let eq_val = EqPolynomial::mle(&self.tau, challenges);
        let w_val = multilinear_eval(&self.w_evals, challenges)?;
        Ok(eq_val * range_check_eval(w_val, self.b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::ext::Ext2;
    use crate::algebra::fields::lift::LiftBase;
    use crate::algebra::ring::CyclotomicRing;
    use crate::algebra::Fp64;
    use crate::protocol::hachi_poly_ops::DensePoly;
    use crate::protocol::opening_point::BasisMode;
    use crate::protocol::ring_switch::r_decomp_levels;
    use crate::protocol::sumcheck::eq_poly::EqPolynomial;
    use crate::protocol::sumcheck::multilinear_eval;
    use crate::protocol::transcript::labels;
    use crate::protocol::{
        prove_sumcheck, verify_sumcheck, Blake2bTranscript, CommitmentConfig, CommitmentScheme,
        HachiCommitmentScheme, SmallTestCommitmentConfig, Transcript,
    };
    use crate::{FieldCore, FromSmallInt};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::array::from_fn;

    type F = Fp64<4294967197>;
    type Cfg = SmallTestCommitmentConfig;
    const D: usize = Cfg::D;
    type Scheme = HachiCommitmentScheme<D, Cfg>;

    struct PointEvalReferenceNormSumcheckProver<E: FieldCore> {
        split_eq: GruenSplitEq<E>,
        w_table: Vec<E>,
        num_vars: usize,
        b: usize,
    }

    impl<E: FieldCore + FromSmallInt> PointEvalReferenceNormSumcheckProver<E> {
        fn new(tau: &[E], w_evals: Vec<E>, b: usize) -> Self {
            let num_vars = tau.len();
            assert_eq!(w_evals.len(), 1 << num_vars);
            Self {
                split_eq: GruenSplitEq::new(tau),
                w_table: w_evals,
                num_vars,
                b,
            }
        }
    }

    impl<E: FieldCore + FromSmallInt> SumcheckInstanceProver<E>
        for PointEvalReferenceNormSumcheckProver<E>
    {
        fn num_rounds(&self) -> usize {
            self.num_vars
        }

        fn degree_bound(&self) -> usize {
            2 * self.b
        }

        fn input_claim(&self) -> E {
            E::zero()
        }

        fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
            let half = self.w_table.len() / 2;
            let degree_q = 2 * self.b - 1;
            let num_points_q = degree_q + 1;

            let (e_first, e_second) = self.split_eq.remaining_eq_tables();
            let num_first = e_first.len();
            let first_bits = num_first.trailing_zeros();
            let b = self.b;

            let mut q_evals = vec![E::zero(); num_points_q];
            for j in 0..half {
                let j_low = j & (num_first - 1);
                let j_high = j >> first_bits;
                let eq_rem = e_first[j_low] * e_second[j_high];
                let w_0 = self.w_table[2 * j];
                let w_1 = self.w_table[2 * j + 1];
                for (t, eval) in q_evals.iter_mut().enumerate() {
                    let t_e = E::from_u64(t as u64);
                    let w_t = w_0 + t_e * (w_1 - w_0);
                    *eval += eq_rem * range_check_eval(w_t, b);
                }
            }

            let q_poly = UniPoly::from_evals(&q_evals);
            self.split_eq.gruen_mul(&q_poly)
        }

        fn ingest_challenge(&mut self, _round: usize, r: E) {
            self.split_eq.bind(r);
            fold_evals_in_place(&mut self.w_table, r);
        }
    }

    fn ring_with_small_coeff(value: u64) -> CyclotomicRing<F, D> {
        let coeffs = from_fn(|_| F::from_u64(value));
        CyclotomicRing::from_coefficients(coeffs)
    }

    #[test]
    fn norm_sumcheck_runtime_dispatch_matches_reference_kernels() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        for (case_idx, b) in [4usize, 8, 16].into_iter().enumerate() {
            let case_idx = case_idx as u64;
            let num_vars = 6;
            let n = 1usize << num_vars;
            let w_evals: Vec<F> = (0..n)
                .map(|i| F::from_u64((i as u64 * 31 + case_idx * 17) % b as u64))
                .collect();
            let tau: Vec<F> = (0..num_vars)
                .map(|_| F::from_u64(rand::Rng::gen_range(&mut rng, 1u64..=257)))
                .collect();

            let mut dispatched = NormSumcheckProver::new(&tau, w_evals.clone(), b);
            let mut point_eval = NormSumcheckProver::new_with_kernel(
                &tau,
                w_evals.clone(),
                b,
                NormRoundKernel::PointEvalInterpolation,
            );
            let use_affine = b <= 8;
            let mut affine_coeff = if use_affine {
                Some(NormSumcheckProver::new_with_kernel(
                    &tau,
                    w_evals.clone(),
                    b,
                    NormRoundKernel::AffineCoeffComposition,
                ))
            } else {
                None
            };
            let mut reference = PointEvalReferenceNormSumcheckProver::new(&tau, w_evals, b);

            let mut claim_dispatched = F::zero();
            let mut claim_point = F::zero();
            let mut claim_affine = F::zero();
            let mut claim_reference = F::zero();
            for round in 0..num_vars {
                let g_dispatch = dispatched.compute_round_univariate(round, claim_dispatched);
                let g_point = point_eval.compute_round_univariate(round, claim_point);
                let g_affine = affine_coeff
                    .as_mut()
                    .map(|p| p.compute_round_univariate(round, claim_affine));
                let g_ref = reference.compute_round_univariate(round, claim_reference);

                assert_eq!(
                    g_point, g_ref,
                    "point-eval mismatch for case {case_idx} round {round}"
                );
                if let Some(ref ga) = g_affine {
                    assert_eq!(
                        *ga, g_ref,
                        "affine-coeff mismatch for case {case_idx} round {round}"
                    );
                }
                match choose_round_kernel(b) {
                    NormRoundKernel::PointEvalInterpolation => {
                        assert_eq!(
                            g_dispatch, g_point,
                            "dispatch mismatch for case {case_idx} round {round}"
                        );
                    }
                    NormRoundKernel::AffineCoeffComposition => {
                        assert_eq!(
                            g_dispatch,
                            g_affine.as_ref().unwrap().clone(),
                            "dispatch mismatch for case {case_idx} round {round}"
                        );
                    }
                }

                assert_eq!(
                    g_dispatch.evaluate(&F::zero()) + g_dispatch.evaluate(&F::one()),
                    claim_dispatched,
                    "dispatched hint mismatch for case {case_idx} round {round}"
                );
                assert_eq!(
                    g_ref.evaluate(&F::zero()) + g_ref.evaluate(&F::one()),
                    claim_reference,
                    "reference hint mismatch for case {case_idx} round {round}"
                );

                let r = F::from_u64(rand::Rng::gen_range(&mut rng, 1u64..=257));
                claim_dispatched = g_dispatch.evaluate(&r);
                claim_point = g_point.evaluate(&r);
                if let Some(ref ga) = g_affine {
                    claim_affine = ga.evaluate(&r);
                }
                claim_reference = g_ref.evaluate(&r);
                dispatched.ingest_challenge(round, r);
                point_eval.ingest_challenge(round, r);
                if let Some(ref mut p) = affine_coeff {
                    p.ingest_challenge(round, r);
                }
                reference.ingest_challenge(round, r);
            }
            assert_eq!(
                claim_dispatched, claim_reference,
                "final dispatched claim mismatch for case {case_idx}"
            );
            assert_eq!(
                claim_point, claim_reference,
                "final point claim mismatch for case {case_idx}"
            );
            if use_affine {
                assert_eq!(
                    claim_affine, claim_reference,
                    "final affine claim mismatch for case {case_idx}"
                );
            }
        }
    }

    #[test]
    fn norm_sumcheck_uses_commitment_w_evals() {
        let z = [
            ring_with_small_coeff(1),
            ring_with_small_coeff(2),
            ring_with_small_coeff(3),
        ];
        let r = [ring_with_small_coeff(0), ring_with_small_coeff(1)];
        let log_basis = SmallTestCommitmentConfig::decomposition().log_basis;
        let levels = r_decomp_levels::<F>(log_basis);
        let r_hat: Vec<CyclotomicRing<F, D>> = r
            .iter()
            .flat_map(|ri| ri.balanced_decompose_pow2(levels, log_basis))
            .collect();
        let mut w_evals: Vec<F> = z
            .iter()
            .chain(r_hat.iter())
            .flat_map(|elem| elem.coefficients().iter().copied())
            .collect();

        let target_len = w_evals.len().next_power_of_two();
        w_evals.resize(target_len, F::zero());
        let num_vars = target_len.trailing_zeros() as usize;
        let tau: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let b = 1usize << SmallTestCommitmentConfig::decomposition().log_basis;

        let eq_table = EqPolynomial::evals(&tau);
        let _claim: F = (0..w_evals.len())
            .map(|i| eq_table[i] * range_check_eval(w_evals[i], b))
            .fold(F::zero(), |a, v| a + v);

        let mut prover = NormSumcheckProver::new(&tau, w_evals.clone(), b);
        let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let (proof, prover_challenges, final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut pt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        let oracle = EqPolynomial::mle(&tau, &prover_challenges)
            * range_check_eval(multilinear_eval(&w_evals, &prover_challenges).unwrap(), b);
        assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

        let verifier = NormSumcheckVerifier::new(tau, w_evals, b);
        let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let verifier_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut vt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }

    #[test]
    fn norm_sumcheck_uses_prove_w_evals() {
        let alpha = SmallTestCommitmentConfig::D.trailing_zeros() as usize;
        let layout = SmallTestCommitmentConfig::commitment_layout(8).unwrap();
        let num_vars = layout.m_vars + layout.r_vars + alpha;
        let len = 1usize << num_vars;
        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();

        let setup = Scheme::setup_prover(num_vars);
        let (commitment, hint) = Scheme::commit(&poly, &setup, &layout).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let proof = Scheme::prove(
            &setup,
            &poly,
            &opening_point,
            hint,
            &mut prover_transcript,
            &commitment,
            BasisMode::Lagrange,
            &layout,
        )
        .unwrap();

        let mut w_evals: Vec<F> = proof.final_w.to_field_elems();
        let target_len = w_evals.len().next_power_of_two();
        w_evals.resize(target_len, F::zero());
        let num_sumcheck_vars = target_len.trailing_zeros() as usize;
        let tau: Vec<F> = (0..num_sumcheck_vars)
            .map(|i| F::from_u64((i + 3) as u64))
            .collect();
        let b = 1usize << SmallTestCommitmentConfig::decomposition().log_basis;

        let eq_table = EqPolynomial::evals(&tau);
        let _claim: F = (0..w_evals.len())
            .map(|i| eq_table[i] * range_check_eval(w_evals[i], b))
            .fold(F::zero(), |a, v| a + v);

        let mut prover = NormSumcheckProver::new(&tau, w_evals.clone(), b);
        let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let (proof_sc, prover_challenges, final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut pt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        let oracle = EqPolynomial::mle(&tau, &prover_challenges)
            * range_check_eval(multilinear_eval(&w_evals, &prover_challenges).unwrap(), b);
        assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

        let verifier = NormSumcheckVerifier::new(tau, w_evals, b);
        let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let verifier_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof_sc, &verifier, &mut vt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }

    #[test]
    fn norm_sumcheck_over_ext2() {
        type E2 = Ext2<F>;

        let num_vars = 3;
        let n = 1usize << num_vars;
        let b = 2;
        let w_evals_f: Vec<F> = (0..n).map(|i| F::from_u64(i as u64 % b as u64)).collect();
        let tau_f: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();

        let w_evals_e: Vec<E2> = w_evals_f.iter().map(|&f| E2::lift_base(f)).collect();
        let tau_e: Vec<E2> = tau_f.iter().map(|&f| E2::lift_base(f)).collect();

        let mut prover = NormSumcheckProver::new(&tau_e, w_evals_e.clone(), b);

        let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let (proof, prover_challenges, final_claim) =
            prove_sumcheck::<F, _, E2, _, _>(&mut prover, &mut pt, |tr| {
                E2::lift_base(tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND))
            })
            .unwrap();

        let oracle = EqPolynomial::mle(&tau_e, &prover_challenges)
            * range_check_eval(multilinear_eval(&w_evals_e, &prover_challenges).unwrap(), b);
        assert_eq!(final_claim, oracle, "E2 prover final claim != oracle eval");

        let verifier = NormSumcheckVerifier::new(tau_e, w_evals_e, b);
        let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let verifier_challenges =
            verify_sumcheck::<F, _, E2, _, _>(&proof, &verifier, &mut vt, |tr| {
                E2::lift_base(tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND))
            })
            .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }

    #[test]
    fn range_check_eval_i128_matches_field() {
        for b in [2, 4, 8, 10] {
            for w in -(b as i32 - 1)..=(b as i32 - 1) {
                let i128_val = range_check_eval_i128(w, b);
                let field_val: F = range_check_eval(F::from_i64(w as i64), b);
                let field_from_i128_val: F = field_from_i128(i128_val);
                assert_eq!(
                    field_from_i128_val, field_val,
                    "i128 range-check mismatch for b={b}, w={w}: \
                     i128={i128_val}, field_from_i128={field_from_i128_val:?}, field={field_val:?}"
                );
            }
        }
    }
}
