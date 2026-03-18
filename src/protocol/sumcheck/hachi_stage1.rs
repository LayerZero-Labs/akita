//! Stage-1 norm sumcheck prover/verifier for the Hachi PCS.
//!
//! This stage works over the virtual table `S(x) = w(x)(w(x)+1)` and carries
//! the resulting `s_claim = S(r_stage1)` into stage 2.

use super::eq_poly::EqPolynomial;
use super::split_eq::GruenSplitEq;
use super::{
    fold_evals_in_place, trim_trailing_zeros, SumcheckInstanceProver, SumcheckInstanceVerifier,
    UniPoly,
};
use crate::algebra::fields::HasUnreducedOps;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::{
    cfg_fold_reduce, cfg_into_iter, AdditiveGroup, CanonicalField, FieldCore, FromSmallInt,
};
use std::time::Instant;

const MAX_AFFINE_COEFFS: usize = 17;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NormRoundKernel {
    PointEvalInterpolation,
    AffineCoeffComposition,
}

fn choose_round_kernel(b: usize) -> NormRoundKernel {
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

#[derive(Clone, Copy, Debug)]
struct SparseCoeffEntry {
    k: u8,
    abs_coeff: u64,
    is_neg: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompactCoeffEntry {
    abs_coeff: u64,
    is_neg: bool,
}

fn poly_coeffs_from_roots_int(roots: &[i64]) -> Vec<i64> {
    let mut coeffs = vec![1i64];
    for &root in roots {
        let mut next = vec![0i64; coeffs.len() + 1];
        for (idx, &coeff) in coeffs.iter().enumerate() {
            next[idx] -= coeff * root;
            next[idx + 1] += coeff;
        }
        coeffs = next;
    }
    coeffs
}

#[derive(Clone)]
struct RangeAffineFromSPrecomp<E: FieldCore> {
    sparse_entries: Vec<SparseCoeffEntry>,
    sparse_row_offsets: Vec<usize>,
    degree_q: usize,
    small_s_lut: Vec<E>,
    compact_coeff_lut: Option<Vec<CompactCoeffEntry>>,
    min_s: i32,
    s_count: usize,
}

impl<E: FieldCore + FromSmallInt> RangeAffineFromSPrecomp<E> {
    fn new(b: usize) -> Self {
        assert!(b >= 2, "b must be at least 2");
        let half = (b / 2) as i64;
        let pair_offsets: Vec<i64> = (0..half).map(|k| k * (k + 1)).collect();
        let range_coeffs = poly_coeffs_from_roots_int(&pair_offsets);
        let degree_q = range_coeffs.len() - 1;
        let num_rows = degree_q + 1;

        let total_elems = num_rows * (num_rows + 1) / 2;
        let mut dense_int = Vec::with_capacity(total_elems);
        let mut dense_row_offsets = Vec::with_capacity(num_rows + 1);
        let mut sparse_entries = Vec::new();
        let mut sparse_row_offsets = Vec::with_capacity(num_rows + 1);

        for i in 0..num_rows {
            dense_row_offsets.push(dense_int.len());
            sparse_row_offsets.push(sparse_entries.len());
            let row_len = degree_q - i + 1;
            let mut binom: i64 = 1;
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

        let min_s = 0i32;
        let max_s = (half * (half - 1)) as i32;
        let s_count = (max_s - min_s + 1) as usize;
        let mut small_s_lut = vec![E::zero(); s_count * num_rows];
        let mut small_s_lut_int = vec![0i128; s_count * num_rows];
        for (s_idx, s_0_int) in (min_s..=max_s).enumerate() {
            for i in 0..num_rows {
                let row = &dense_int[dense_row_offsets[i]..dense_row_offsets[i + 1]];
                let mut h: i128 = 0;
                for &c in row.iter().rev() {
                    h = h * s_0_int as i128 + c as i128;
                }
                small_s_lut_int[s_idx * num_rows + i] = h;
                small_s_lut[s_idx * num_rows + i] = E::from_i128(h);
            }
        }

        let compact_coeff_lut = if b <= 8 {
            let mut lut = Vec::with_capacity(s_count * s_count * num_rows);
            for s0_idx in 0..s_count {
                let s_0_int = min_s as i64 + s0_idx as i64;
                let h_base = s0_idx * num_rows;
                for s_1_int in min_s as i64..=max_s as i64 {
                    let delta = (s_1_int - s_0_int) as i128;
                    let mut delta_pow = 1i128;
                    for &h_i in &small_s_lut_int[h_base..h_base + num_rows] {
                        let coeff = h_i
                            .checked_mul(delta_pow)
                            .expect("compact affine coefficient overflow");
                        let abs_coeff = coeff.unsigned_abs();
                        assert!(
                            abs_coeff <= u64::MAX as u128,
                            "compact affine coefficient exceeds u64"
                        );
                        lut.push(CompactCoeffEntry {
                            abs_coeff: abs_coeff as u64,
                            is_neg: coeff < 0,
                        });
                        delta_pow = delta_pow
                            .checked_mul(delta)
                            .expect("compact affine power overflow");
                    }
                }
            }
            Some(lut)
        } else {
            None
        };

        Self {
            sparse_entries,
            sparse_row_offsets,
            degree_q,
            small_s_lut,
            compact_coeff_lut,
            min_s,
            s_count,
        }
    }
}

impl<E: FieldCore> RangeAffineFromSPrecomp<E> {
    #[inline]
    fn value_index(&self, s_0_int: i32) -> usize {
        debug_assert!(s_0_int >= self.min_s);
        debug_assert!(s_0_int < self.min_s + self.s_count as i32);
        (s_0_int - self.min_s) as usize
    }

    #[inline]
    fn sparse_row(&self, i: usize) -> &[SparseCoeffEntry] {
        &self.sparse_entries[self.sparse_row_offsets[i]..self.sparse_row_offsets[i + 1]]
    }

    fn num_rows(&self) -> usize {
        self.degree_q + 1
    }

    #[inline]
    fn h_i_lut(&self, s_0_int: i32, i: usize) -> E {
        let s_idx = self.value_index(s_0_int);
        self.small_s_lut[s_idx * self.num_rows() + i]
    }

    #[inline]
    fn compact_coeffs_lut(&self, s_0_int: i32, s_1_int: i32) -> Option<&[CompactCoeffEntry]> {
        let lut = self.compact_coeff_lut.as_ref()?;
        let num_rows = self.num_rows();
        let pair_idx = self.value_index(s_0_int) * self.s_count + self.value_index(s_1_int);
        let start = pair_idx * num_rows;
        Some(&lut[start..start + num_rows])
    }
}

#[derive(Clone)]
struct PointEvalPrecomp<E: FieldCore> {
    pair_offsets: Vec<E>,
}

impl<E: FieldCore + FromSmallInt> PointEvalPrecomp<E> {
    fn new(b: usize) -> Self {
        assert!(b >= 2, "b must be at least 2");
        let half = (b / 2) as i64;
        let pair_offsets = (0..half).map(|k| E::from_i64(k * (k + 1))).collect();
        Self { pair_offsets }
    }
}

#[inline]
fn field_from_i128<E: CanonicalField>(val: i128) -> E {
    if val >= 0 {
        E::from_canonical_u128_reduced(val as u128)
    } else {
        -E::from_canonical_u128_reduced(val.unsigned_abs())
    }
}

#[inline]
fn range_check_eval_from_s_precomputed<E: FieldCore>(s: E, pair_offsets: &[E]) -> E {
    let mut acc = E::one();
    for &offset in pair_offsets {
        acc = acc * (s - offset);
    }
    acc
}

#[inline]
pub(crate) fn range_check_eval_from_s<E: FieldCore + FromSmallInt>(s: E, b: usize) -> E {
    let half = (b / 2) as i64;
    let mut acc = E::one();
    for k in 0..half {
        acc = acc * (s - E::from_i64(k * (k + 1)));
    }
    acc
}

#[inline]
fn range_check_eval_from_s_i128(s: i64, b: usize) -> i128 {
    let half = (b / 2) as i128;
    let mut acc = 1i128;
    let mut offset = 0i128;
    for k in 0..half {
        acc = acc
            .checked_mul(s as i128 - offset)
            .expect("i128 overflow in range-check from s");
        offset += 2 * k + 2;
    }
    acc
}

#[inline]
fn accumulate_compact_coeffs<E: FieldCore + HasUnreducedOps>(
    pos_accum: &mut [E::MulU64Accum],
    neg_accum: &mut [E::MulU64Accum],
    e_in: E,
    coeffs: &[CompactCoeffEntry],
) {
    debug_assert!(pos_accum.len() >= coeffs.len());
    debug_assert!(neg_accum.len() >= coeffs.len());
    for (idx, coeff) in coeffs.iter().enumerate() {
        if coeff.abs_coeff == 0 {
            continue;
        }
        let prod = e_in.mul_u64_unreduced(coeff.abs_coeff);
        if coeff.is_neg {
            neg_accum[idx] += prod;
        } else {
            pos_accum[idx] += prod;
        }
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
fn compute_entry_coeffs_from_s<E: FieldCore + HasUnreducedOps>(
    out: &mut [E],
    s_pows: &mut [E],
    precomp: &RangeAffineFromSPrecomp<E>,
    s_0: E,
    a: E,
) {
    let deg = precomp.degree_q;
    let num_rows = precomp.num_rows();
    debug_assert!(out.len() >= num_rows);
    debug_assert!(s_pows.len() > deg);

    s_pows[0] = E::one();
    for k in 1..=deg {
        s_pows[k] = s_pows[k - 1] * s_0;
    }

    let mut a_pow = E::one();
    for (i, out_i) in out.iter_mut().enumerate().take(num_rows) {
        let entries = precomp.sparse_row(i);
        let mut pos = E::MulU64Accum::ZERO;
        let mut neg = E::MulU64Accum::ZERO;
        for entry in entries {
            let prod = s_pows[entry.k as usize].mul_u64_unreduced(entry.abs_coeff);
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

#[inline]
fn compute_entry_coeffs_from_s_x4<E: FieldCore + HasUnreducedOps>(
    out: &mut [[E; MAX_AFFINE_COEFFS]; 4],
    precomp: &RangeAffineFromSPrecomp<E>,
    s_0: [E; 4],
    a: [E; 4],
) {
    let deg = precomp.degree_q;
    let num_rows = precomp.num_rows();

    let mut pw = [[E::zero(); MAX_AFFINE_COEFFS]; 4];
    for p in &mut pw {
        p[0] = E::one();
    }
    for k in 1..=deg {
        pw[0][k] = pw[0][k - 1] * s_0[0];
        pw[1][k] = pw[1][k - 1] * s_0[1];
        pw[2][k] = pw[2][k - 1] * s_0[2];
        pw[3][k] = pw[3][k - 1] * s_0[3];
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

fn compute_norm_round_poly_from_s<E: FieldCore + FromSmallInt + HasUnreducedOps>(
    split_eq: &GruenSplitEq<E>,
    half: usize,
    b: usize,
    round_kernel: NormRoundKernel,
    point_precomp: Option<&PointEvalPrecomp<E>>,
    range_precomp: Option<&RangeAffineFromSPrecomp<E>>,
    s_pair: impl Fn(usize) -> (E, E) + Sync,
) -> UniPoly<E> {
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();
    let first_bits = num_first.trailing_zeros();
    let degree_q = b / 2;

    match round_kernel {
        NormRoundKernel::PointEvalInterpolation => {
            let num_points_q = degree_q + 1;
            let pair_offsets = &point_precomp.unwrap().pair_offsets;

            let q_evals = cfg_fold_reduce!(
                0..half,
                || vec![E::zero(); num_points_q],
                |mut evals, j| {
                    let j_low = j & (num_first - 1);
                    let j_high = j >> first_bits;
                    let eq_rem = e_first[j_low] * e_second[j_high];
                    let (s_0, s_1) = s_pair(j);
                    let delta = s_1 - s_0;
                    let mut s_t = s_0;
                    for eval in &mut evals {
                        *eval += eq_rem * range_check_eval_from_s_precomputed(s_t, pair_offsets);
                        s_t += delta;
                    }
                    evals
                },
                |mut a, b_vec| {
                    for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                        *ai += *bi;
                    }
                    a
                }
            );

            let q_poly = UniPoly::from_evals(&q_evals);
            split_eq.gruen_mul(&q_poly)
        }
        NormRoundKernel::AffineCoeffComposition => {
            let rp = range_precomp.unwrap();
            let num_coeffs_q = rp.degree_q + 1;

            let mut q_coeffs = cfg_fold_reduce!(
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
                            s_pair(base_j + jl),
                            s_pair(base_j + jl + 1),
                            s_pair(base_j + jl + 2),
                            s_pair(base_j + jl + 3),
                        ];
                        compute_entry_coeffs_from_s_x4(
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
                        for (b_idx, bo) in batch_out.iter().enumerate() {
                            let e_in = e_first[jl + b_idx];
                            for (acc, &entry) in inner_accum[..num_coeffs_q]
                                .iter_mut()
                                .zip(bo[..num_coeffs_q].iter())
                            {
                                *acc += e_in.mul_to_product_accum(entry);
                            }
                        }
                    }

                    let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                    let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];
                    for (tail_idx, &e_in) in e_first[full_chunks * 4..].iter().enumerate() {
                        let j = base_j + full_chunks * 4 + tail_idx;
                        let (s_0, s_1) = s_pair(j);
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            rp,
                            s_0,
                            s_1 - s_0,
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
            .into_iter()
            .map(E::reduce_product_accum)
            .collect::<Vec<_>>();

            trim_trailing_zeros(&mut q_coeffs);
            let q_poly = UniPoly::from_coeffs(q_coeffs);
            split_eq.gruen_mul(&q_poly)
        }
    }
}

fn compute_norm_round_poly_from_s_compact<
    E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps,
>(
    split_eq: &GruenSplitEq<E>,
    s_compact: &[i32],
    b: usize,
    round_kernel: NormRoundKernel,
    point_precomp: Option<&PointEvalPrecomp<E>>,
    range_precomp: Option<&RangeAffineFromSPrecomp<E>>,
) -> UniPoly<E> {
    let half = s_compact.len() / 2;
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();
    let first_bits = num_first.trailing_zeros();
    let degree_q = b / 2;

    match round_kernel {
        NormRoundKernel::PointEvalInterpolation if b <= 16 => {
            let num_points_q = degree_q + 1;

            let q_evals = cfg_fold_reduce!(
                0..half,
                || vec![E::zero(); num_points_q],
                |mut evals, j| {
                    let j_low = j & (num_first - 1);
                    let j_high = j >> first_bits;
                    let eq_rem = e_first[j_low] * e_second[j_high];
                    let s0_i = s_compact[2 * j] as i64;
                    let delta_i = s_compact[2 * j + 1] as i64 - s0_i;
                    let mut s_t_i = s0_i;
                    for eval in &mut evals {
                        *eval +=
                            eq_rem * field_from_i128::<E>(range_check_eval_from_s_i128(s_t_i, b));
                        s_t_i += delta_i;
                    }
                    evals
                },
                |mut a, b_vec| {
                    for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                        *ai += *bi;
                    }
                    a
                }
            );

            let q_poly = UniPoly::from_evals(&q_evals);
            split_eq.gruen_mul(&q_poly)
        }
        NormRoundKernel::AffineCoeffComposition => {
            let rp = range_precomp.unwrap();
            let num_coeffs_q = rp.degree_q + 1;

            let mut q_coeffs = if rp.compact_coeffs_lut(0, 0).is_some() {
                cfg_fold_reduce!(
                    0..e_second.len(),
                    || vec![E::ProductAccum::ZERO; num_coeffs_q],
                    |mut outer_accum, j_high| {
                        debug_assert!(num_coeffs_q <= MAX_AFFINE_COEFFS);
                        let mut inner_pos = [E::MulU64Accum::ZERO; MAX_AFFINE_COEFFS];
                        let mut inner_neg = [E::MulU64Accum::ZERO; MAX_AFFINE_COEFFS];
                        for (j_low, &e_in) in e_first.iter().enumerate() {
                            let j = j_high * num_first + j_low;
                            let s_0_int = s_compact[2 * j];
                            let s_1_int = s_compact[2 * j + 1];
                            let coeffs = rp
                                .compact_coeffs_lut(s_0_int, s_1_int)
                                .expect("missing compact coefficient LUT");
                            accumulate_compact_coeffs(
                                &mut inner_pos[..num_coeffs_q],
                                &mut inner_neg[..num_coeffs_q],
                                e_in,
                                coeffs,
                            );
                        }
                        let e_out = e_second[j_high];
                        for k in 0..num_coeffs_q {
                            let inner_reduced =
                                reduce_small_coeff_accum(inner_pos[k], inner_neg[k]);
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
                .collect::<Vec<_>>()
            } else {
                cfg_fold_reduce!(
                    0..e_second.len(),
                    || vec![E::ProductAccum::ZERO; num_coeffs_q],
                    |mut outer_accum, j_high| {
                        debug_assert!(num_coeffs_q <= MAX_AFFINE_COEFFS);
                        let mut inner_accum = [E::ProductAccum::ZERO; MAX_AFFINE_COEFFS];
                        for (j_low, &e_in) in e_first.iter().enumerate() {
                            let j = j_high * num_first + j_low;
                            let s_0_int = s_compact[2 * j];
                            let s_1 = E::from_i64(s_compact[2 * j + 1] as i64);
                            let a = s_1 - E::from_i64(s_0_int as i64);
                            let mut a_pow = E::one();
                            for (i, acc) in inner_accum[..num_coeffs_q].iter_mut().enumerate() {
                                let h_i_s0 = rp.h_i_lut(s_0_int, i);
                                let val = a_pow * h_i_s0;
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
                .into_iter()
                .map(E::reduce_product_accum)
                .collect::<Vec<_>>()
            };

            trim_trailing_zeros(&mut q_coeffs);
            let q_poly = UniPoly::from_coeffs(q_coeffs);
            split_eq.gruen_mul(&q_poly)
        }
        _ => {
            let s_pair = |j: usize| {
                (
                    E::from_i64(s_compact[2 * j] as i64),
                    E::from_i64(s_compact[2 * j + 1] as i64),
                )
            };
            compute_norm_round_poly_from_s(
                split_eq,
                half,
                b,
                round_kernel,
                point_precomp,
                range_precomp,
                s_pair,
            )
        }
    }
}

enum STable<E: FieldCore> {
    Compact(Vec<i32>),
    Full(Vec<E>),
}

/// Stage-1 norm sumcheck prover over the virtual table `S(x) = w(x)(w(x)+1)`.
pub struct HachiStage1Prover<E: FieldCore> {
    s_table: STable<E>,
    split_eq: GruenSplitEq<E>,
    round_kernel: NormRoundKernel,
    point_precomp: Option<PointEvalPrecomp<E>>,
    range_precomp: Option<RangeAffineFromSPrecomp<E>>,
    live_x_cols: usize,
    num_u: usize,
    num_vars: usize,
    b: usize,
    prefix_time_total: f64,
    dense_time_total: f64,
    fold_time_total: f64,
    rounds_completed: usize,
}

impl<E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps> HachiStage1Prover<E> {
    /// Build the stage-1 prover from the compact witness table.
    #[tracing::instrument(skip_all, name = "HachiStage1Prover::new")]
    pub fn new(
        w_evals_compact: &[i8],
        tau0: &[E],
        b: usize,
        live_x_cols: usize,
        num_u: usize,
        num_l: usize,
    ) -> Self {
        assert!(b >= 2, "b must be at least 2");
        let num_vars = num_u + num_l;
        let y_len = 1usize << num_l;
        assert_eq!(w_evals_compact.len(), live_x_cols * y_len);
        assert_eq!(tau0.len(), num_vars);

        let s_table = w_evals_compact
            .iter()
            .map(|&w| {
                let w = w as i32;
                w * (w + 1)
            })
            .collect();
        let round_kernel = choose_round_kernel(b / 2);
        let point_precomp = match round_kernel {
            NormRoundKernel::PointEvalInterpolation => Some(PointEvalPrecomp::new(b)),
            NormRoundKernel::AffineCoeffComposition => None,
        };
        let range_precomp = match round_kernel {
            NormRoundKernel::PointEvalInterpolation => None,
            NormRoundKernel::AffineCoeffComposition => Some(RangeAffineFromSPrecomp::new(b)),
        };

        Self {
            s_table: STable::Compact(s_table),
            split_eq: GruenSplitEq::new(tau0),
            round_kernel,
            point_precomp,
            range_precomp,
            live_x_cols,
            num_u,
            num_vars,
            b,
            prefix_time_total: 0.0,
            dense_time_total: 0.0,
            fold_time_total: 0.0,
            rounds_completed: 0,
        }
    }

    /// Return the fully folded virtual-polynomial claim `S(r_stage1)`.
    ///
    /// # Panics
    ///
    /// Panics if called before the virtual table has been fully folded to a
    /// single field element.
    pub fn final_s_claim(&self) -> E {
        match &self.s_table {
            STable::Full(s_full) => {
                assert_eq!(s_full.len(), 1, "s_table not fully folded");
                s_full[0]
            }
            STable::Compact(_) => panic!("s_table remained compact after final fold"),
        }
    }

    #[inline]
    fn current_x_width(&self) -> usize {
        self.num_u.saturating_sub(self.rounds_completed)
    }

    #[inline]
    fn current_x_len(&self) -> usize {
        1usize << self.current_x_width()
    }

    #[inline]
    fn use_prefix_x_round(&self) -> bool {
        self.rounds_completed < self.num_u && self.live_x_cols < self.current_x_len()
    }

    #[tracing::instrument(skip_all, name = "HachiStage1Prover::compute_round_compact_prefix_x")]
    fn compute_round_compact_prefix_x(&self, s_compact: &[i32]) -> UniPoly<E> {
        debug_assert!(self.rounds_completed < self.num_u);
        debug_assert_eq!(
            s_compact.len(),
            self.live_x_cols * (1usize << (self.num_vars - self.num_u))
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let degree_q = self.b / 2;

        match self.round_kernel {
            NormRoundKernel::PointEvalInterpolation => {
                let num_points_q = degree_q + 1;
                let q_evals = cfg_fold_reduce!(
                    0..(1usize << (self.num_vars - self.num_u)),
                    || vec![E::zero(); num_points_q],
                    |mut norm_evals, y| {
                        let row_start = y * self.live_x_cols;
                        let row = &s_compact[row_start..row_start + self.live_x_cols];
                        for pair_x in 0..live_pairs {
                            let j = y * current_x_half + pair_x;
                            let j_low = j & (num_first - 1);
                            let j_high = j >> first_bits;
                            let eq_rem = e_first[j_low] * e_second[j_high];

                            let left = 2 * pair_x;
                            let s0_i = row[left] as i64;
                            let s1_i = if left + 1 < self.live_x_cols {
                                row[left + 1] as i64
                            } else {
                                0
                            };
                            let delta_i = s1_i - s0_i;
                            let mut s_t_i = s0_i;
                            for eval in &mut norm_evals {
                                *eval += eq_rem
                                    * field_from_i128::<E>(range_check_eval_from_s_i128(
                                        s_t_i, self.b,
                                    ));
                                s_t_i += delta_i;
                            }
                        }
                        norm_evals
                    },
                    |mut a, b_vec| {
                        for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                            *ai += *bi;
                        }
                        a
                    }
                );
                let q_poly = UniPoly::from_evals(&q_evals);
                self.split_eq.gruen_mul(&q_poly)
            }
            NormRoundKernel::AffineCoeffComposition => {
                let rp = self.range_precomp.as_ref().unwrap();
                let num_coeffs_q = rp.degree_q + 1;
                let mut q_coeffs = if rp.compact_coeffs_lut(0, 0).is_some() {
                    let (pos_accum, neg_accum) = cfg_fold_reduce!(
                        0..(1usize << (self.num_vars - self.num_u)),
                        || (
                            vec![E::MulU64Accum::ZERO; num_coeffs_q],
                            vec![E::MulU64Accum::ZERO; num_coeffs_q],
                        ),
                        |(mut pos_accum, mut neg_accum), y| {
                            let row_start = y * self.live_x_cols;
                            let row = &s_compact[row_start..row_start + self.live_x_cols];
                            for pair_x in 0..live_pairs {
                                let j = y * current_x_half + pair_x;
                                let j_low = j & (num_first - 1);
                                let j_high = j >> first_bits;
                                let eq_rem = e_first[j_low] * e_second[j_high];

                                let left = 2 * pair_x;
                                let s0_i = row[left];
                                let s1_i = if left + 1 < self.live_x_cols {
                                    row[left + 1]
                                } else {
                                    0
                                };
                                let coeffs = rp
                                    .compact_coeffs_lut(s0_i, s1_i)
                                    .expect("missing compact coefficient LUT");
                                accumulate_compact_coeffs(
                                    &mut pos_accum,
                                    &mut neg_accum,
                                    eq_rem,
                                    coeffs,
                                );
                            }
                            (pos_accum, neg_accum)
                        },
                        |(mut pa, mut na), (pb, nb)| {
                            for (ai, bi) in pa.iter_mut().zip(pb.iter()) {
                                *ai += *bi;
                            }
                            for (ai, bi) in na.iter_mut().zip(nb.iter()) {
                                *ai += *bi;
                            }
                            (pa, na)
                        }
                    );
                    pos_accum
                        .into_iter()
                        .zip(neg_accum)
                        .map(|(pos, neg)| reduce_small_coeff_accum(pos, neg))
                        .collect()
                } else {
                    cfg_fold_reduce!(
                        0..(1usize << (self.num_vars - self.num_u)),
                        || vec![E::ProductAccum::ZERO; num_coeffs_q],
                        |mut q_coeffs, y| {
                            let row_start = y * self.live_x_cols;
                            let row = &s_compact[row_start..row_start + self.live_x_cols];
                            let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                            let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];
                            for pair_x in 0..live_pairs {
                                let j = y * current_x_half + pair_x;
                                let j_low = j & (num_first - 1);
                                let j_high = j >> first_bits;
                                let eq_rem = e_first[j_low] * e_second[j_high];

                                let left = 2 * pair_x;
                                let s0_i = row[left];
                                let s1_i = if left + 1 < self.live_x_cols {
                                    row[left + 1]
                                } else {
                                    0
                                };
                                compute_entry_coeffs_from_s(
                                    &mut entry_buf,
                                    &mut s_pows_buf,
                                    rp,
                                    E::from_i64(s0_i as i64),
                                    E::from_i64((s1_i as i64) - (s0_i as i64)),
                                );
                                for (acc, &entry) in
                                    q_coeffs.iter_mut().zip(entry_buf[..num_coeffs_q].iter())
                                {
                                    *acc += eq_rem.mul_to_product_accum(entry);
                                }
                            }
                            q_coeffs
                        },
                        |mut ca, cb| {
                            for (ai, bi) in ca.iter_mut().zip(cb.iter()) {
                                *ai += *bi;
                            }
                            ca
                        }
                    )
                    .into_iter()
                    .map(E::reduce_product_accum)
                    .collect()
                };

                trim_trailing_zeros(&mut q_coeffs);
                let q_poly = UniPoly::from_coeffs(q_coeffs);
                self.split_eq.gruen_mul(&q_poly)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "HachiStage1Prover::compute_round_full_prefix_x")]
    fn compute_round_full_prefix_x(&self, s_full: &[E]) -> UniPoly<E> {
        debug_assert!(self.rounds_completed < self.num_u);
        let y_len = s_full.len() / self.live_x_cols;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let degree_q = self.b / 2;

        match self.round_kernel {
            NormRoundKernel::PointEvalInterpolation => {
                let num_points_q = degree_q + 1;
                let pair_offsets = &self.point_precomp.as_ref().unwrap().pair_offsets;
                let q_evals = cfg_fold_reduce!(
                    0..y_len,
                    || vec![E::zero(); num_points_q],
                    |mut norm_evals, y| {
                        let row_start = y * self.live_x_cols;
                        let row = &s_full[row_start..row_start + self.live_x_cols];
                        for pair_x in 0..live_pairs {
                            let j = y * current_x_half + pair_x;
                            let j_low = j & (num_first - 1);
                            let j_high = j >> first_bits;
                            let eq_rem = e_first[j_low] * e_second[j_high];

                            let left = 2 * pair_x;
                            let s_0 = row[left];
                            let s_1 = if left + 1 < self.live_x_cols {
                                row[left + 1]
                            } else {
                                E::zero()
                            };
                            let delta = s_1 - s_0;
                            let mut s_t = s_0;
                            for eval in &mut norm_evals {
                                *eval +=
                                    eq_rem * range_check_eval_from_s_precomputed(s_t, pair_offsets);
                                s_t += delta;
                            }
                        }
                        norm_evals
                    },
                    |mut a, b_vec| {
                        for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                            *ai += *bi;
                        }
                        a
                    }
                );
                let q_poly = UniPoly::from_evals(&q_evals);
                self.split_eq.gruen_mul(&q_poly)
            }
            NormRoundKernel::AffineCoeffComposition => {
                let range_pc = self.range_precomp.as_ref().unwrap();
                let num_coeffs_q = range_pc.degree_q + 1;
                let mut q_coeffs = cfg_fold_reduce!(
                    0..y_len,
                    || vec![E::ProductAccum::ZERO; num_coeffs_q],
                    |mut q_coeffs, y| {
                        debug_assert!(num_coeffs_q <= MAX_AFFINE_COEFFS);
                        let row_start = y * self.live_x_cols;
                        let row = &s_full[row_start..row_start + self.live_x_cols];
                        let base_j = y * current_x_half;
                        let full_chunks = live_pairs / 4;
                        let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];

                        for chunk in 0..full_chunks {
                            let pair_base = chunk * 4;
                            let mut pairs = [(E::zero(), E::zero()); 4];
                            for (slot, pair_x) in (pair_base..pair_base + 4).enumerate() {
                                let left = 2 * pair_x;
                                let s_0 = row[left];
                                let s_1 = if left + 1 < self.live_x_cols {
                                    row[left + 1]
                                } else {
                                    E::zero()
                                };
                                pairs[slot] = (s_0, s_1);
                            }

                            compute_entry_coeffs_from_s_x4(
                                &mut batch_out,
                                range_pc,
                                [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0],
                                [
                                    pairs[0].1 - pairs[0].0,
                                    pairs[1].1 - pairs[1].0,
                                    pairs[2].1 - pairs[2].0,
                                    pairs[3].1 - pairs[3].0,
                                ],
                            );

                            for (slot, _) in pairs.iter().enumerate() {
                                let pair_x = pair_base + slot;
                                let j = base_j + pair_x;
                                let j_low = j & (num_first - 1);
                                let j_high = j >> first_bits;
                                let eq_rem = e_first[j_low] * e_second[j_high];
                                for (acc, &entry) in q_coeffs
                                    .iter_mut()
                                    .zip(batch_out[slot][..num_coeffs_q].iter())
                                {
                                    *acc += eq_rem.mul_to_product_accum(entry);
                                }
                            }
                        }

                        let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                        let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];
                        for pair_x in full_chunks * 4..live_pairs {
                            let left = 2 * pair_x;
                            let s_0 = row[left];
                            let s_1 = if left + 1 < self.live_x_cols {
                                row[left + 1]
                            } else {
                                E::zero()
                            };
                            compute_entry_coeffs_from_s(
                                &mut entry_buf,
                                &mut s_pows_buf,
                                range_pc,
                                s_0,
                                s_1 - s_0,
                            );

                            let j = base_j + pair_x;
                            let j_low = j & (num_first - 1);
                            let j_high = j >> first_bits;
                            let eq_rem = e_first[j_low] * e_second[j_high];
                            for (acc, &entry) in
                                q_coeffs.iter_mut().zip(entry_buf[..num_coeffs_q].iter())
                            {
                                *acc += eq_rem.mul_to_product_accum(entry);
                            }
                        }

                        q_coeffs
                    },
                    |mut ca, cb| {
                        for (ai, bi) in ca.iter_mut().zip(cb.iter()) {
                            *ai += *bi;
                        }
                        ca
                    }
                );

                let mut q_coeffs: Vec<E> =
                    q_coeffs.drain(..).map(E::reduce_product_accum).collect();
                trim_trailing_zeros(&mut q_coeffs);
                let q_poly = UniPoly::from_coeffs(q_coeffs);
                self.split_eq.gruen_mul(&q_poly)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "HachiStage1Prover::fold_s_compact_prefix_x")]
    fn fold_s_compact_prefix_x(
        s_compact: &[i32],
        live_x_cols: usize,
        y_len: usize,
        r: E,
    ) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_x_cols;
                let row = &s_compact[row_start..row_start + live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let s_0 = E::from_i64(row[left] as i64);
                    let s_1 = if left + 1 < live_x_cols {
                        E::from_i64(row[left + 1] as i64)
                    } else {
                        E::zero()
                    };
                    *dst = s_0 + r * (s_1 - s_0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row_start = y * live_x_cols;
            let row = &s_compact[row_start..row_start + live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let s_0 = E::from_i64(row[left] as i64);
                let s_1 = if left + 1 < live_x_cols {
                    E::from_i64(row[left + 1] as i64)
                } else {
                    E::zero()
                };
                *dst = s_0 + r * (s_1 - s_0);
            }
        }

        out
    }

    #[tracing::instrument(skip_all, name = "HachiStage1Prover::fold_s_full_prefix_x")]
    fn fold_s_full_prefix_x(s_full: &[E], live_x_cols: usize, y_len: usize, r: E) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_x_cols;
                let row = &s_full[row_start..row_start + live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let s_0 = row[left];
                    let s_1 = if left + 1 < live_x_cols {
                        row[left + 1]
                    } else {
                        E::zero()
                    };
                    *dst = s_0 + r * (s_1 - s_0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row_start = y * live_x_cols;
            let row = &s_full[row_start..row_start + live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let s_0 = row[left];
                let s_1 = if left + 1 < live_x_cols {
                    row[left + 1]
                } else {
                    E::zero()
                };
                *dst = s_0 + r * (s_1 - s_0);
            }
        }

        out
    }

    #[tracing::instrument(skip_all, name = "HachiStage1Prover::fold_s_compact_to_full")]
    fn fold_s_compact_to_full(s_compact: &[i32], r: E) -> Vec<E> {
        cfg_into_iter!(0..s_compact.len() / 2)
            .map(|j| {
                let s_0 = E::from_i64(s_compact[2 * j] as i64);
                let s_1 = E::from_i64(s_compact[2 * j + 1] as i64);
                s_0 + r * (s_1 - s_0)
            })
            .collect()
    }
}

impl<E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps> SumcheckInstanceProver<E>
    for HachiStage1Prover<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        self.b / 2 + 1
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let use_prefix_x_round = self.use_prefix_x_round();
        let t_round = Instant::now();
        let poly = match &self.s_table {
            STable::Compact(s_compact) => {
                if use_prefix_x_round {
                    self.compute_round_compact_prefix_x(s_compact)
                } else {
                    compute_norm_round_poly_from_s_compact(
                        &self.split_eq,
                        s_compact,
                        self.b,
                        self.round_kernel,
                        self.point_precomp.as_ref(),
                        self.range_precomp.as_ref(),
                    )
                }
            }
            STable::Full(s_full) => {
                if use_prefix_x_round {
                    self.compute_round_full_prefix_x(s_full)
                } else {
                    let half = s_full.len() / 2;
                    let s_table = s_full;
                    compute_norm_round_poly_from_s(
                        &self.split_eq,
                        half,
                        self.b,
                        self.round_kernel,
                        self.point_precomp.as_ref(),
                        self.range_precomp.as_ref(),
                        |j| (s_table[2 * j], s_table[2 * j + 1]),
                    )
                }
            }
        };

        if use_prefix_x_round {
            self.prefix_time_total += t_round.elapsed().as_secs_f64();
        } else {
            self.dense_time_total += t_round.elapsed().as_secs_f64();
        }

        poly
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let t_fold = Instant::now();
        let _span = tracing::info_span!("HachiStage1Prover::fold_round").entered();
        self.split_eq.bind(r);
        let use_prefix_x_round = self.use_prefix_x_round();
        let y_len = match &self.s_table {
            STable::Compact(s_compact) => s_compact.len() / self.live_x_cols,
            STable::Full(s_full) => s_full.len() / self.live_x_cols,
        };

        self.s_table = match std::mem::replace(&mut self.s_table, STable::Full(Vec::new())) {
            STable::Compact(s_compact) => {
                let s_full = if use_prefix_x_round {
                    Self::fold_s_compact_prefix_x(&s_compact, self.live_x_cols, y_len, r)
                } else {
                    Self::fold_s_compact_to_full(&s_compact, r)
                };
                STable::Full(s_full)
            }
            STable::Full(mut s_full) => {
                if use_prefix_x_round {
                    s_full = Self::fold_s_full_prefix_x(&s_full, self.live_x_cols, y_len, r);
                } else {
                    fold_evals_in_place(&mut s_full, r);
                }
                STable::Full(s_full)
            }
        };

        if self.rounds_completed < self.num_u {
            self.live_x_cols = self.live_x_cols.div_ceil(2);
        }
        self.rounds_completed += 1;
        drop(_span);
        self.fold_time_total += t_fold.elapsed().as_secs_f64();
    }

    fn finalize(&mut self) {
        tracing::debug!(
            rounds = self.num_vars,
            prefix_s = self.prefix_time_total,
            dense_s = self.dense_time_total,
            fold_s = self.fold_time_total,
            "stage1 sumcheck rounds complete"
        );
    }
}

/// Verifier for the stage-1 norm sumcheck over the virtual table `S`.
pub struct HachiStage1Verifier<F: FieldCore> {
    tau0: Vec<F>,
    s_claim: F,
    b: usize,
}

impl<F: FieldCore + FromSmallInt> HachiStage1Verifier<F> {
    /// Construct the stage-1 verifier from `tau0`, the carried `s_claim`, and `b`.
    pub fn new(tau0: Vec<F>, s_claim: F, b: usize) -> Self {
        Self { tau0, s_claim, b }
    }
}

impl<F: FieldCore + FromSmallInt> SumcheckInstanceVerifier<F> for HachiStage1Verifier<F> {
    fn num_rounds(&self) -> usize {
        self.tau0.len()
    }

    fn degree_bound(&self) -> usize {
        self.b / 2 + 1
    }

    fn input_claim(&self) -> F {
        F::zero()
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, HachiError> {
        let eq_val = EqPolynomial::mle(&self.tau0, challenges);
        Ok(eq_val * range_check_eval_from_s(self.s_claim, self.b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Prime128M8M4M1M0;
    use crate::protocol::sumcheck::multilinear_eval;

    type F = Prime128M8M4M1M0;

    fn pad_compact_rows(
        w_prefix: &[i8],
        live_x_cols: usize,
        num_u: usize,
        num_l: usize,
    ) -> Vec<i8> {
        let x_len = 1usize << num_u;
        let y_len = 1usize << num_l;
        let mut padded = vec![0i8; x_len * y_len];
        for y in 0..y_len {
            let src_start = y * live_x_cols;
            let dst_start = y * x_len;
            padded[dst_start..dst_start + live_x_cols]
                .copy_from_slice(&w_prefix[src_start..src_start + live_x_cols]);
        }
        padded
    }

    #[test]
    fn stage1_round0_matches_dense_reference() {
        let num_u = 3usize;
        let num_l = 2usize;
        let b = 8usize;
        let n = 1usize << (num_u + num_l);
        let half = (b / 2) as i8;
        let w_compact: Vec<i8> = (0..n).map(|i| ((i * 5 + 3) % b) as i8 - half).collect();
        let tau0: Vec<F> = (0..(num_u + num_l))
            .map(|i| F::from_u64((i as u64) + 2))
            .collect();

        let mut prover =
            HachiStage1Prover::new(&w_compact, &tau0, b, 1usize << num_u, num_u, num_l);
        let stage1_poly = prover.compute_round_univariate(0, F::zero());
        let s_compact: Vec<i32> = w_compact
            .iter()
            .map(|&w| {
                let w = w as i32;
                w * (w + 1)
            })
            .collect();
        let reference = compute_norm_round_poly_from_s_compact(
            &prover.split_eq,
            &s_compact,
            b,
            prover.round_kernel,
            prover.point_precomp.as_ref(),
            prover.range_precomp.as_ref(),
        );

        assert_eq!(stage1_poly, reference);
    }

    #[test]
    fn stage1_prefix_aware_rounds_match_explicit_zero_padding() {
        let num_l = 2usize;
        let b = 8usize;
        let half = (b / 2) as i8;

        for live_x_cols in [5usize, 6usize] {
            let num_u = live_x_cols.next_power_of_two().trailing_zeros() as usize;
            let y_len = 1usize << num_l;
            let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
                .map(|i| ((i * 7 + 5) % b) as i8 - half)
                .collect();
            let w_padded = pad_compact_rows(&w_prefix, live_x_cols, num_u, num_l);
            let tau0: Vec<F> = (0..(num_u + num_l))
                .map(|i| F::from_u64((i as u64) + 19))
                .collect();
            let mut prefix_prover =
                HachiStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, num_u, num_l);
            let mut padded_prover =
                HachiStage1Prover::new(&w_padded, &tau0, b, 1usize << num_u, num_u, num_l);
            let mut challenges = Vec::new();

            for round in 0..(num_u + num_l) {
                let prefix_poly = prefix_prover.compute_round_univariate(round, F::zero());
                let padded_poly = padded_prover.compute_round_univariate(round, F::zero());
                assert_eq!(
                    prefix_poly, padded_poly,
                    "round {round} polynomial mismatch live_x_cols={live_x_cols}"
                );

                let challenge = F::from_u64((round as u64) + 29);
                challenges.push(challenge);
                prefix_prover.ingest_challenge(round, challenge);
                padded_prover.ingest_challenge(round, challenge);
            }

            assert_eq!(prefix_prover.final_s_claim(), padded_prover.final_s_claim());
            let s_padded: Vec<F> = w_padded
                .iter()
                .map(|&w| {
                    let w = F::from_i64(w as i64);
                    w * (w + F::one())
                })
                .collect();
            assert_eq!(
                prefix_prover.final_s_claim(),
                multilinear_eval(&s_padded, &challenges).unwrap(),
                "final s-claim mismatch live_x_cols={live_x_cols}"
            );
        }
    }
}
