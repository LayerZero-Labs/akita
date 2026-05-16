//! Stage-1 norm sumcheck prover/verifier for the Akita PCS.
//!
//! The committed witness is a Boolean table
//! `w : {0,1}^{col_bits} x {0,1}^{ring_bits} -> {-half, ..., half-1}` with
//! `half = b/2`. Define the virtual table `S(z) = w(z) * (w(z) + 1)`. For an
//! honest witness every entry of `w` is a valid digit, so `S(z)` lies in the
//! set `{k(k+1) : k = 0, ..., half-1}`. The range-check polynomial
//!
//! `Q(s) = prod_{k=0}^{half-1} (s - k(k+1))`
//!
//! has degree `b/2` and vanishes on exactly that set. The sumcheck proves
//!
//! `0 = sum_z eq(tau0, z) * Q(S(z))`,
//!
//! where the input claim is `0` (an honest prover makes every summand vanish).
//! Stage 1 uses the generic eq-factored sumcheck path: each round writes the
//! full polynomial as `s(X) = l(X) * q(X)`, where `l` is the linear eq factor
//! for the current round and `q` has degree `b/2`. The proof sends the
//! headerless `q` message with its linear term omitted, rather than the full
//! degree-`b/2 + 1` product polynomial. After all rounds, at `r_stage1`, the
//! verifier checks
//!
//! `eq(tau0, r_stage1) * Q(s_claim)`
//!
//! where `s_claim = S(r_stage1) = w(r_stage1) * (w(r_stage1) + 1)` is the
//! carried virtual claim passed into stage 2.
//!
//! ## `b = 8` specialization
//!
//! With `half = 4` the roots are `{0, 2, 6, 12}`, giving
//!
//! `Q(s) = s * (s - 2) * (s - 6) * (s - 12)`,
//!
//! degree 4, so round polynomials have degree 5.

use super::fold_full_prefix_pair;
use super::two_round_prefix::{
    build_stage1_bivariate_skip_proof_from_s_compact, can_use_stage1_two_round_prefix,
    stage1_b4_s_digit_from_compact_s, stage1_b8_s_digit_from_compact_s, Stage1BivariateSkipState,
};
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::fields::HasUnreducedOps;
use akita_field::parallel::*;
use akita_field::{FieldCore, FromPrimitiveInt, Zero};
use akita_sumcheck::{
    fold_evals_in_place, CompactPairFoldLut, EqFactoredSumcheckInstanceProver, EqFactoredUniPoly,
};
use std::time::Instant;

const MAX_AFFINE_COEFFS: usize = 17;
const MAX_COMPACT_COEFF_LUT_B: usize = 16;
const MAX_FIELD_COEFF_LUT_B: usize = 32;

#[derive(Clone, Copy, Debug, Default)]
struct CompactCoeffEntry {
    abs_coeff: u64,
    is_neg: bool,
}

fn poly_coeffs_from_roots_int(roots: &[i128]) -> Vec<i128> {
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
struct RangeAffineFromSPrecomp<E: FieldCore> {
    dense_coeffs: Vec<E>,
    dense_row_offsets: Vec<usize>,
    degree_q: usize,
    /// `h_i(s_0)` for each valid `s_0` and coefficient index `i`.
    /// Indexed as `compact_idx * num_rows + i`, where `compact_idx` is
    /// obtained from `s_to_compact`.
    small_s_lut: Vec<E>,
    compact_coeff_lut: Option<Vec<CompactCoeffEntry>>,
    field_coeff_lut: Option<Vec<E>>,
    /// Maps raw `s` integer (offset by `min_s`) to a compact index into the
    /// `b/2`-element valid-value set `{k(k+1) : k = 0..half-1}`.
    s_to_compact: Vec<u8>,
    num_valid_s: usize,
    min_s: i16,
}

impl<E: FieldCore + FromPrimitiveInt> RangeAffineFromSPrecomp<E> {
    fn new(b: usize) -> Self {
        assert!(b >= 2, "b must be at least 2");
        let half = (b / 2) as i128;
        let pair_offsets: Vec<i128> = (0..half).map(|k| k * (k + 1)).collect();
        let range_coeffs = poly_coeffs_from_roots_int(&pair_offsets);
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

        let min_s = 0i16;
        let max_s_i128 = half * (half - 1);
        assert!(
            max_s_i128 <= i16::MAX as i128,
            "compact s range exceeds i16 for b={b}"
        );
        let max_s = max_s_i128 as i16;
        let raw_range = (i32::from(max_s) - i32::from(min_s) + 1) as usize;
        let num_valid_s = half as usize;

        let mut s_to_compact = vec![u8::MAX; raw_range];
        for (compact_idx, &s_val) in pair_offsets.iter().enumerate() {
            s_to_compact[(s_val as i16 - min_s) as usize] = compact_idx as u8;
        }

        let mut small_s_lut = vec![E::zero(); num_valid_s * num_rows];
        let mut small_s_lut_int = vec![0i128; num_valid_s * num_rows];
        for (compact_idx, &s_val) in pair_offsets.iter().enumerate() {
            for i in 0..num_rows {
                let row = &dense_int[dense_row_offsets[i]..dense_row_offsets[i + 1]];
                let mut h: i128 = 0;
                for &c in row.iter().rev() {
                    h = h * s_val + c;
                }
                small_s_lut_int[compact_idx * num_rows + i] = h;
                small_s_lut[compact_idx * num_rows + i] = E::from_i128(h);
            }
        }

        let compact_coeff_lut = if b <= MAX_COMPACT_COEFF_LUT_B {
            let mut lut = Vec::with_capacity(num_valid_s * num_valid_s * num_rows);
            for (s0_ci, &s0_val) in pair_offsets.iter().enumerate() {
                let h_base = s0_ci * num_rows;
                for &s1_val in &pair_offsets {
                    let delta = s1_val - s0_val;
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
        let field_coeff_lut = if b > MAX_COMPACT_COEFF_LUT_B && b <= MAX_FIELD_COEFF_LUT_B {
            let mut lut = Vec::with_capacity(num_valid_s * num_valid_s * num_rows);
            for (s0_ci, &s0_val) in pair_offsets.iter().enumerate() {
                let h_base = s0_ci * num_rows;
                for &s1_val in &pair_offsets {
                    let delta = E::from_i128(s1_val - s0_val);
                    let mut delta_pow = E::one();
                    for &h_i in &small_s_lut[h_base..h_base + num_rows] {
                        lut.push(h_i * delta_pow);
                        delta_pow *= delta;
                    }
                }
            }
            Some(lut)
        } else {
            None
        };

        Self {
            dense_coeffs,
            dense_row_offsets,
            degree_q,
            small_s_lut,
            compact_coeff_lut,
            field_coeff_lut,
            s_to_compact,
            num_valid_s,
            min_s,
        }
    }
}

impl<E: FieldCore> RangeAffineFromSPrecomp<E> {
    #[inline]
    fn compact_index(&self, s_int: i16) -> usize {
        let raw = (s_int - self.min_s) as usize;
        debug_assert!(raw < self.s_to_compact.len());
        let ci = self.s_to_compact[raw];
        debug_assert_ne!(ci, u8::MAX, "s={s_int} is not a valid w*(w+1) value");
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
    fn h_i_lut(&self, s_0_int: i16, i: usize) -> E {
        let ci = self.compact_index(s_0_int);
        self.small_s_lut[ci * self.num_rows() + i]
    }

    #[inline]
    fn pair_coeff_lut_start(&self, s_0_int: i16, s_1_int: i16) -> usize {
        let pair_idx = self.compact_index(s_0_int) * self.num_valid_s + self.compact_index(s_1_int);
        pair_idx * self.num_rows()
    }

    #[inline]
    fn compact_coeffs_lut(&self, s_0_int: i16, s_1_int: i16) -> Option<&[CompactCoeffEntry]> {
        let lut = self.compact_coeff_lut.as_ref()?;
        let num_rows = self.num_rows();
        let start = self.pair_coeff_lut_start(s_0_int, s_1_int);
        Some(&lut[start..start + num_rows])
    }

    #[inline]
    fn field_coeffs_lut(&self, s_0_int: i16, s_1_int: i16) -> Option<&[E]> {
        let lut = self.field_coeff_lut.as_ref()?;
        let num_rows = self.num_rows();
        let start = self.pair_coeff_lut_start(s_0_int, s_1_int);
        Some(&lut[start..start + num_rows])
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
fn compute_entry_coeffs_from_s<E: FieldCore + HasUnreducedOps>(
    out: &mut [E],
    _s_pows: &mut [E],
    precomp: &RangeAffineFromSPrecomp<E>,
    s_0: E,
    a: E,
) {
    let num_rows = precomp.num_rows();
    debug_assert!(out.len() >= num_rows);

    let mut a_pow = E::one();
    for (i, out_i) in out.iter_mut().enumerate().take(num_rows) {
        let mut h_i = E::zero();
        for &coeff in precomp.dense_row(i).iter().rev() {
            h_i = h_i * s_0 + coeff;
        }
        *out_i = a_pow * h_i;
        a_pow *= a;
    }
}

#[inline]
fn compute_entry_coeffs_from_s_x4<E: FieldCore + HasUnreducedOps>(
    out: &mut [[E; MAX_AFFINE_COEFFS]; 4],
    precomp: &RangeAffineFromSPrecomp<E>,
    s_0: [E; 4],
    a: [E; 4],
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
            h0 = h0 * s_0[0] + coeff;
            h1 = h1 * s_0[1] + coeff;
            h2 = h2 * s_0[2] + coeff;
            h3 = h3 * s_0[3] + coeff;
        }

        *out0_i = ap[0] * h0;
        *out1_i = ap[1] * h1;
        *out2_i = ap[2] * h2;
        *out3_i = ap[3] * h3;

        ap[0] *= a[0];
        ap[1] *= a[1];
        ap[2] *= a[2];
        ap[3] *= a[3];
    }
}

fn compute_norm_round_eq_poly_from_s<E: FieldCore + FromPrimitiveInt + HasUnreducedOps>(
    split_eq: &GruenSplitEq<E>,
    range_precomp: &RangeAffineFromSPrecomp<E>,
    s_pair: impl Fn(usize) -> (E, E) + Sync,
) -> EqFactoredUniPoly<E> {
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();
    let rp = range_precomp;
    let full_num_coeffs_q = rp.degree_q + 1;
    let num_coeffs_q = full_num_coeffs_q;

    let q_coeffs = cfg_fold_reduce!(
        0..e_second.len(),
        || vec![E::ProductAccum::zero(); num_coeffs_q],
        |mut outer_accum, j_high| {
            debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
            let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
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
                    accumulate_dense_entry_coeffs(
                        &mut inner_accum[..num_coeffs_q],
                        &bo[..full_num_coeffs_q],
                        e_in,
                    );
                }
            }

            let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
            let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];
            for (tail_idx, &e_in) in e_first[full_chunks * 4..].iter().enumerate() {
                let j = base_j + full_chunks * 4 + tail_idx;
                let (s_0, s_1) = s_pair(j);
                compute_entry_coeffs_from_s(&mut entry_buf, &mut s_pows_buf, rp, s_0, s_1 - s_0);
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

fn compute_norm_round_eq_poly_from_s_compact_with_pairs<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
>(
    split_eq: &GruenSplitEq<E>,
    range_precomp: &RangeAffineFromSPrecomp<E>,
    s_pair: impl Fn(usize) -> (i16, i16) + Sync,
) -> EqFactoredUniPoly<E> {
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();

    let rp = range_precomp;
    let full_num_coeffs_q = rp.degree_q + 1;
    let num_coeffs_q = full_num_coeffs_q;

    let q_coeffs = if rp.compact_coeffs_lut(0, 0).is_some() {
        cfg_fold_reduce!(
            0..e_second.len(),
            || vec![E::ProductAccum::zero(); num_coeffs_q],
            |mut outer_accum, j_high| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let mut inner_pos = [E::MulU64Accum::zero(); MAX_AFFINE_COEFFS];
                let mut inner_neg = [E::MulU64Accum::zero(); MAX_AFFINE_COEFFS];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = j_high * num_first + j_low;
                    let (s_0_int, s_1_int) = s_pair(j);
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
        .collect::<Vec<_>>()
    } else if rp.field_coeffs_lut(0, 0).is_some() {
        cfg_fold_reduce!(
            0..e_second.len(),
            || vec![E::ProductAccum::zero(); num_coeffs_q],
            |mut outer_accum, j_high| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = j_high * num_first + j_low;
                    let (s_0_int, s_1_int) = s_pair(j);
                    let coeffs = rp
                        .field_coeffs_lut(s_0_int, s_1_int)
                        .expect("missing field coefficient LUT");
                    accumulate_dense_entry_coeffs(&mut inner_accum[..num_coeffs_q], coeffs, e_in);
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
    } else {
        cfg_fold_reduce!(
            0..e_second.len(),
            || vec![E::ProductAccum::zero(); num_coeffs_q],
            |mut outer_accum, j_high| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                for (j_low, &e_in) in e_first.iter().enumerate() {
                    let j = j_high * num_first + j_low;
                    let (s_0_int, s_1_int) = s_pair(j);
                    let s_1 = E::from_i64(i64::from(s_1_int));
                    let a = s_1 - E::from_i64(i64::from(s_0_int));
                    let mut a_pow = E::one();
                    for (coeff_idx, coeff_accum) in
                        inner_accum.iter_mut().take(full_num_coeffs_q).enumerate()
                    {
                        let h_i_s0 = rp.h_i_lut(s_0_int, coeff_idx);
                        let val = a_pow * h_i_s0;
                        *coeff_accum += e_in.mul_to_product_accum(val);
                        a_pow *= a;
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

    let _ = split_eq;
    EqFactoredUniPoly::from_q_coeffs(q_coeffs)
}

fn compute_norm_round_eq_poly_from_s_compact<E: FieldCore + FromPrimitiveInt + HasUnreducedOps>(
    split_eq: &GruenSplitEq<E>,
    s_compact: &[i16],
    range_precomp: &RangeAffineFromSPrecomp<E>,
) -> EqFactoredUniPoly<E> {
    compute_norm_round_eq_poly_from_s_compact_with_pairs(split_eq, range_precomp, |j| {
        (s_compact[2 * j], s_compact[2 * j + 1])
    })
}

enum STable<E: FieldCore> {
    Compact(Vec<i16>),
    Full(Vec<E>),
}

#[inline]
fn compact_s_from_w(w: i8) -> i16 {
    let w = i32::from(w);
    let s = w * (w + 1);
    debug_assert!(s >= 0);
    s as i16
}

fn build_compact_s_table(w_evals_compact: &[i8]) -> Vec<i16> {
    w_evals_compact
        .iter()
        .copied()
        .map(compact_s_from_w)
        .collect()
}

struct Stage1TwoRoundPrefix<E: FieldCore> {
    skip_state: Stage1BivariateSkipState<E>,
    first_challenge: Option<E>,
}

/// Stage-1 norm sumcheck prover over the virtual table `S(x) = w(x)(w(x)+1)`.
pub struct AkitaStage1Prover<E: FieldCore> {
    s_table: STable<E>,
    split_eq: GruenSplitEq<E>,
    range_precomp: RangeAffineFromSPrecomp<E>,
    live_x_cols: usize,
    col_bits: usize,
    num_vars: usize,
    b: usize,
    prefix_tau: Option<Vec<E>>,
    two_round_prefix: Option<Stage1TwoRoundPrefix<E>>,
    cached_round_poly: Option<EqFactoredUniPoly<E>>,
    prefix_time_total: f64,
    dense_time_total: f64,
    fold_time_total: f64,
    rounds_completed: usize,
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage1Prover<E> {
    /// Build the stage-1 prover from the compact witness table.
    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::new")]
    pub fn new(
        w_evals_compact: &[i8],
        tau0: &[E],
        b: usize,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
    ) -> Self {
        assert!(b >= 2, "b must be at least 2");
        let num_vars = col_bits + ring_bits;
        let y_len = 1usize << ring_bits;
        assert_eq!(w_evals_compact.len(), live_x_cols * y_len);
        assert_eq!(tau0.len(), num_vars);
        let s_table = build_compact_s_table(w_evals_compact);

        Self {
            s_table: STable::Compact(s_table),
            split_eq: GruenSplitEq::new(tau0).expect("valid prover stage-1 challenge shape"),
            range_precomp: RangeAffineFromSPrecomp::new(b),
            live_x_cols,
            col_bits,
            num_vars,
            b,
            prefix_tau: can_use_stage1_two_round_prefix(ring_bits, b).then(|| tau0.to_vec()),
            two_round_prefix: None,
            cached_round_poly: None,
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
    fn ring_bits(&self) -> usize {
        self.num_vars - self.col_bits
    }

    #[inline]
    fn in_x_phase(&self) -> bool {
        self.rounds_completed >= self.ring_bits()
    }

    #[inline]
    fn current_x_width(&self) -> usize {
        debug_assert!(self.in_x_phase());
        self.num_vars.saturating_sub(self.rounds_completed)
    }

    #[inline]
    fn current_x_len(&self) -> usize {
        1usize << self.current_x_width()
    }

    #[inline]
    fn use_prefix_x_round(&self) -> bool {
        self.in_x_phase() && self.live_x_cols < self.current_x_len()
    }

    #[inline]
    fn next_use_prefix_x_round_after_current(&self) -> bool {
        self.in_x_phase()
            && self.rounds_completed + 1 < self.num_vars
            && self.live_x_cols.div_ceil(2) < (self.current_x_len() / 2)
    }

    #[inline]
    fn next_use_sparse_x_y_round_after_current(&self) -> bool {
        !self.in_x_phase() && self.rounds_completed + 1 < self.ring_bits()
    }

    #[inline]
    pub(crate) fn can_use_two_round_prefix(&self) -> bool {
        self.prefix_tau.is_some()
    }

    #[inline]
    fn using_two_round_prefix(&self) -> bool {
        self.rounds_completed < 2 && self.can_use_two_round_prefix()
    }

    #[inline]
    fn compact_s_values(b: usize) -> Vec<i16> {
        let half = (b / 2) as i16;
        (0..half).map(|k| k * (k + 1)).collect()
    }

    #[inline]
    fn build_compact_s_fold_lut(b: usize, r: E) -> CompactPairFoldLut<E> {
        let valid_s = Self::compact_s_values(b);
        CompactPairFoldLut::from_allowed_values(&valid_s, r)
    }

    fn ensure_two_round_prefix(&mut self) -> &mut Stage1TwoRoundPrefix<E> {
        if self.two_round_prefix.is_none() {
            let tau0 = self
                .prefix_tau
                .clone()
                .expect("two-round prefix requested without cached tau");
            let ring_bits = self.num_vars - self.col_bits;
            let s_compact = match &self.s_table {
                STable::Compact(s_compact) => s_compact,
                STable::Full(_) => panic!("two-round prefix can only build from compact table"),
            };
            let proof = build_stage1_bivariate_skip_proof_from_s_compact(
                s_compact,
                &tau0,
                self.b,
                self.live_x_cols,
                self.col_bits,
                ring_bits,
            )
            .expect("two-round prefix should be available");
            let skip_state = Stage1BivariateSkipState::new(&proof, &tau0, self.b)
                .expect("valid bivariate-skip state");
            self.two_round_prefix = Some(Stage1TwoRoundPrefix {
                skip_state,
                first_challenge: None,
            });
        }
        self.two_round_prefix
            .as_mut()
            .expect("two-round prefix should be initialized")
    }

    #[inline]
    fn direct_fold_s_quad_to_round2(s00: i16, s10: i16, s01: i16, s11: i16, r0: E, r1: E) -> E {
        let s00 = E::from_i64(i64::from(s00));
        let s10 = E::from_i64(i64::from(s10));
        let s01 = E::from_i64(i64::from(s01));
        let s11 = E::from_i64(i64::from(s11));
        let x0 = s00 + r0 * (s10 - s00);
        let x1 = s01 + r0 * (s11 - s01);
        x0 + r1 * (x1 - x0)
    }

    #[inline(always)]
    fn stage1_b4_quad_lookup_index_from_row(row: &[i16], base: usize) -> usize {
        let d0 = row
            .get(base)
            .copied()
            .map(stage1_b4_s_digit_from_compact_s)
            .unwrap_or(0);
        let d1 = row
            .get(base + 1)
            .copied()
            .map(stage1_b4_s_digit_from_compact_s)
            .unwrap_or(0);
        let d2 = row
            .get(base + 2)
            .copied()
            .map(stage1_b4_s_digit_from_compact_s)
            .unwrap_or(0);
        let d3 = row
            .get(base + 3)
            .copied()
            .map(stage1_b4_s_digit_from_compact_s)
            .unwrap_or(0);
        d0 | (d1 << 1) | (d2 << 2) | (d3 << 3)
    }

    fn build_round2_s_lookup_b4(r0: E, r1: E) -> Vec<E> {
        const S_VALUES: [i16; 2] = [0, 2];
        (0..16usize)
            .map(|idx| {
                let d0 = idx & 0b1;
                let d1 = (idx >> 1) & 0b1;
                let d2 = (idx >> 2) & 0b1;
                let d3 = (idx >> 3) & 0b1;
                Self::direct_fold_s_quad_to_round2(
                    S_VALUES[d0],
                    S_VALUES[d1],
                    S_VALUES[d2],
                    S_VALUES[d3],
                    r0,
                    r1,
                )
            })
            .collect()
    }

    #[inline(always)]
    fn stage1_b8_quad_lookup_index_from_row(row: &[i16], base: usize) -> usize {
        let d0 = row
            .get(base)
            .copied()
            .map(stage1_b8_s_digit_from_compact_s)
            .unwrap_or(0);
        let d1 = row
            .get(base + 1)
            .copied()
            .map(stage1_b8_s_digit_from_compact_s)
            .unwrap_or(0);
        let d2 = row
            .get(base + 2)
            .copied()
            .map(stage1_b8_s_digit_from_compact_s)
            .unwrap_or(0);
        let d3 = row
            .get(base + 3)
            .copied()
            .map(stage1_b8_s_digit_from_compact_s)
            .unwrap_or(0);
        d0 | (d1 << 2) | (d2 << 4) | (d3 << 6)
    }

    fn build_round2_s_lookup_b8(r0: E, r1: E) -> Vec<E> {
        const S_VALUES: [i16; 4] = [0, 2, 6, 12];
        (0..256usize)
            .map(|idx| {
                let d0 = idx & 0b11;
                let d1 = (idx >> 2) & 0b11;
                let d2 = (idx >> 4) & 0b11;
                let d3 = (idx >> 6) & 0b11;
                Self::direct_fold_s_quad_to_round2(
                    S_VALUES[d0],
                    S_VALUES[d1],
                    S_VALUES[d2],
                    S_VALUES[d3],
                    r0,
                    r1,
                )
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::fold_s_compact_to_round2")]
    fn fold_s_compact_to_round2(
        s_compact: &[i16],
        live_x_cols: usize,
        y_len: usize,
        r0: E,
        r1: E,
    ) -> Vec<E> {
        debug_assert_eq!(y_len % 4, 0);
        let next_y_len = y_len / 4;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];
        for (x, col_out) in out.chunks_mut(next_y_len).enumerate() {
            let col = &s_compact[x * y_len..(x + 1) * y_len];
            for (quad_y, dst) in col_out.iter_mut().enumerate() {
                let base = 4 * quad_y;
                *dst = Self::direct_fold_s_quad_to_round2(
                    col[base],
                    col[base + 1],
                    col[base + 2],
                    col[base + 3],
                    r0,
                    r1,
                );
            }
        }
        out
    }

    #[tracing::instrument(
        skip_all,
        name = "AkitaStage1Prover::fuse_compact_to_round2_and_compute_round"
    )]
    fn fuse_compact_to_round2_and_compute_round(
        &self,
        s_compact: &[i16],
        r0: E,
        r1: E,
    ) -> (Vec<E>, EqFactoredUniPoly<E>) {
        debug_assert!(self.ring_bits() > 2);
        let live_x_cols = self.live_x_cols;
        let y_len = s_compact.len() / live_x_cols;
        debug_assert_eq!(y_len % 4, 0);
        let next_y_len = y_len / 4;
        let live_pairs = next_y_len / 2;
        let current_y_half = next_y_len / 2;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let block_size = num_first.min(live_pairs);
        let quad_fold_lut = match self.b {
            4 => Self::build_round2_s_lookup_b4(r0, r1),
            _ => Self::build_round2_s_lookup_b8(r0, r1),
        };
        let quad_index_fn: fn(&[i16], usize) -> usize = match self.b {
            4 => Self::stage1_b4_quad_lookup_index_from_row,
            _ => Self::stage1_b8_quad_lookup_index_from_row,
        };

        let range_pc = &self.range_precomp;
        let full_num_coeffs_q = range_pc.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];

        #[cfg(feature = "parallel")]
        let q_coeffs = out
            .par_chunks_mut(next_y_len)
            .enumerate()
            .map(|(x, col_out)| {
                let col = &s_compact[x * y_len..(x + 1) * y_len];
                let j_base = x * current_y_half;
                let mut outer_accum = vec![E::ProductAccum::zero(); num_coeffs_q];
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];

                    for pair_y in blk..blk_end {
                        let j_low = (j_base + pair_y) & (num_first - 1);
                        let e_in = e_first[j_low];
                        let top_y = 2 * pair_y;
                        let top_base = 8 * pair_y;
                        let s0 = quad_fold_lut[quad_index_fn(col, top_base)];
                        let s1 = quad_fold_lut[quad_index_fn(col, top_base + 4)];
                        col_out[top_y] = s0;
                        col_out[top_y + 1] = s1;
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s0,
                            s1 - s0,
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
                    blk = blk_end;
                }
                outer_accum
            })
            .reduce(
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += *bi;
                    }
                    a
                },
            )
            .into_iter()
            .map(E::reduce_product_accum)
            .collect::<Vec<_>>();

        #[cfg(not(feature = "parallel"))]
        let q_coeffs = {
            let mut outer = vec![E::ProductAccum::zero(); num_coeffs_q];
            for (x, col_out) in out.chunks_mut(next_y_len).enumerate() {
                let col = &s_compact[x * y_len..(x + 1) * y_len];
                let j_base = x * current_y_half;
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];

                    for pair_y in blk..blk_end {
                        let j_low = (j_base + pair_y) & (num_first - 1);
                        let e_in = e_first[j_low];
                        let top_y = 2 * pair_y;
                        let top_base = 8 * pair_y;
                        let s0 = quad_fold_lut[quad_index_fn(col, top_base)];
                        let s1 = quad_fold_lut[quad_index_fn(col, top_base + 4)];
                        col_out[top_y] = s0;
                        col_out[top_y + 1] = s1;
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s0,
                            s1 - s0,
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
                        outer[k] += e_out.mul_to_product_accum(inner_reduced);
                    }
                    blk = blk_end;
                }
            }
            outer
                .into_iter()
                .map(E::reduce_product_accum)
                .collect::<Vec<_>>()
        };

        let poly = EqFactoredUniPoly::from_q_coeffs(q_coeffs);
        (out, poly)
    }

    #[tracing::instrument(
        skip_all,
        name = "AkitaStage1Prover::fuse_full_prefix_x_and_compute_round"
    )]
    fn fuse_full_prefix_x_and_compute_round(
        &self,
        s_full: &[E],
        r: E,
    ) -> (Vec<E>, EqFactoredUniPoly<E>) {
        debug_assert!(self.next_use_prefix_x_round_after_current());
        debug_assert!(self.current_x_width() >= 2);

        let old_live_x_cols = self.live_x_cols;
        let next_live_x_cols = old_live_x_cols.div_ceil(2);
        let y_len = s_full.len() / old_live_x_cols;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let next_current_x_half = 1usize << (self.current_x_width() - 2);
        let live_pairs = next_live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        let range_pc = &self.range_precomp;
        let full_num_coeffs_q = range_pc.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        #[cfg(feature = "parallel")]
        let q_coeffs = out
            .par_chunks_mut(next_live_x_cols)
            .enumerate()
            .map(|(y, row_out)| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let row = &s_full[y * old_live_x_cols..(y + 1) * old_live_x_cols];
                let j_base = y * next_current_x_half;
                let mut outer_accum = vec![E::ProductAccum::zero(); num_coeffs_q];
                let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                    let blk_len = blk_end - blk;
                    let full_chunks = blk_len / 4;

                    for chunk in 0..full_chunks {
                        let pair_base = blk + chunk * 4;
                        let mut pairs = [(E::zero(), E::zero()); 4];
                        for (slot, pair_x) in (pair_base..pair_base + 4).enumerate() {
                            let left_next = 2 * pair_x;
                            let left_old = 4 * pair_x;
                            let s0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = s0;
                            let s1 = if left_next + 1 < next_live_x_cols {
                                let s1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = s1;
                                s1
                            } else {
                                E::zero()
                            };
                            pairs[slot] = (s0, s1);
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
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_x in blk + full_chunks * 4..blk_end {
                        let left_next = 2 * pair_x;
                        let left_old = 4 * pair_x;
                        let s_0 = fold_full_prefix_pair(row, left_old, r);
                        row_out[left_next] = s_0;
                        let s_1 = if left_next + 1 < next_live_x_cols {
                            let s_1 = fold_full_prefix_pair(row, left_old + 2, r);
                            row_out[left_next + 1] = s_1;
                            s_1
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
                        let j_low = (j_base + pair_x) & (num_first - 1);
                        let e_in = e_first[j_low];
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
                    blk = blk_end;
                }

                outer_accum
            })
            .reduce(
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += *bi;
                    }
                    a
                },
            )
            .into_iter()
            .map(E::reduce_product_accum)
            .collect::<Vec<_>>();

        #[cfg(not(feature = "parallel"))]
        let q_coeffs = {
            let mut outer_accum = vec![E::ProductAccum::zero(); num_coeffs_q];
            for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let row = &s_full[y * old_live_x_cols..(y + 1) * old_live_x_cols];
                let j_base = y * next_current_x_half;
                let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                    let blk_len = blk_end - blk;
                    let full_chunks = blk_len / 4;

                    for chunk in 0..full_chunks {
                        let pair_base = blk + chunk * 4;
                        let mut pairs = [(E::zero(), E::zero()); 4];
                        for (slot, pair_x) in (pair_base..pair_base + 4).enumerate() {
                            let left_next = 2 * pair_x;
                            let left_old = 4 * pair_x;
                            let s0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = s0;
                            let s1 = if left_next + 1 < next_live_x_cols {
                                let s1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = s1;
                                s1
                            } else {
                                E::zero()
                            };
                            pairs[slot] = (s0, s1);
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
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_x in blk + full_chunks * 4..blk_end {
                        let left_next = 2 * pair_x;
                        let left_old = 4 * pair_x;
                        let s_0 = fold_full_prefix_pair(row, left_old, r);
                        row_out[left_next] = s_0;
                        let s_1 = if left_next + 1 < next_live_x_cols {
                            let s_1 = fold_full_prefix_pair(row, left_old + 2, r);
                            row_out[left_next + 1] = s_1;
                            s_1
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
                        let j_low = (j_base + pair_x) & (num_first - 1);
                        let e_in = e_first[j_low];
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
                    blk = blk_end;
                }
            }

            outer_accum
                .into_iter()
                .map(E::reduce_product_accum)
                .collect::<Vec<_>>()
        };

        let poly = EqFactoredUniPoly::from_q_coeffs(q_coeffs);
        (out, poly)
    }

    #[inline]
    fn use_sparse_x_y_round(&self) -> bool {
        !self.in_x_phase() && self.live_x_cols < (1usize << self.col_bits)
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::compute_round_compact_sparse_x_y")]
    fn compute_round_compact_sparse_x_y(&self, s_compact: &[i16]) -> EqFactoredUniPoly<E> {
        debug_assert!(self.use_sparse_x_y_round());
        let y_len = s_compact.len() / self.live_x_cols;
        let y_pairs = y_len / 2;
        compute_norm_round_eq_poly_from_s_compact_with_pairs(
            &self.split_eq,
            &self.range_precomp,
            |j| {
                let x = j / y_pairs;
                if x >= self.live_x_cols {
                    return (0, 0);
                }
                let y_pair = j % y_pairs;
                let top = x * y_len + 2 * y_pair;
                (s_compact[top], s_compact[top + 1])
            },
        )
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::compute_round_full_sparse_x_y")]
    fn compute_round_full_sparse_x_y(&self, s_full: &[E]) -> EqFactoredUniPoly<E> {
        debug_assert!(self.use_sparse_x_y_round());
        let y_len = s_full.len() / self.live_x_cols;
        let y_pairs = y_len / 2;
        compute_norm_round_eq_poly_from_s(&self.split_eq, &self.range_precomp, |j| {
            let x = j / y_pairs;
            if x >= self.live_x_cols {
                return (E::zero(), E::zero());
            }
            let y_pair = j % y_pairs;
            let top = x * y_len + 2 * y_pair;
            (s_full[top], s_full[top + 1])
        })
    }

    #[tracing::instrument(
        skip_all,
        name = "AkitaStage1Prover::fuse_full_sparse_x_y_and_compute_round"
    )]
    fn fuse_full_sparse_x_y_and_compute_round(
        &self,
        s_full: &[E],
        r: E,
    ) -> (Vec<E>, EqFactoredUniPoly<E>) {
        debug_assert!(self.use_sparse_x_y_round());
        debug_assert!(self.next_use_sparse_x_y_round_after_current());
        let live_x_cols = self.live_x_cols;
        let y_len = s_full.len() / live_x_cols;
        debug_assert_eq!(y_len % 4, 0);
        let next_y_len = y_len / 2;
        let live_pairs = next_y_len / 2;
        let current_y_half = next_y_len / 2;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let block_size = num_first.min(live_pairs);
        let range_pc = &self.range_precomp;
        let full_num_coeffs_q = range_pc.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];

        #[cfg(feature = "parallel")]
        let q_coeffs = out
            .par_chunks_mut(next_y_len)
            .enumerate()
            .map(|(x, col_out)| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let col = &s_full[x * y_len..(x + 1) * y_len];
                let j_base = x * current_y_half;
                let mut outer_accum = vec![E::ProductAccum::zero(); num_coeffs_q];
                let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                    let blk_len = blk_end - blk;
                    let full_chunks = blk_len / 4;

                    for chunk in 0..full_chunks {
                        let pair_base = blk + chunk * 4;
                        let mut pairs = [(E::zero(), E::zero()); 4];
                        for (slot, pair_y) in (pair_base..pair_base + 4).enumerate() {
                            let top_y = 2 * pair_y;
                            let top = 4 * pair_y;
                            let s0 = col[top] + r * (col[top + 1] - col[top]);
                            let s1 = col[top + 2] + r * (col[top + 3] - col[top + 2]);
                            col_out[top_y] = s0;
                            col_out[top_y + 1] = s1;
                            pairs[slot] = (s0, s1);
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
                            let pair_y = pair_base + slot;
                            let j_low = (j_base + pair_y) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_y in blk + full_chunks * 4..blk_end {
                        let top_y = 2 * pair_y;
                        let top = 4 * pair_y;
                        let s0 = col[top] + r * (col[top + 1] - col[top]);
                        let s1 = col[top + 2] + r * (col[top + 3] - col[top + 2]);
                        col_out[top_y] = s0;
                        col_out[top_y + 1] = s1;
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s0,
                            s1 - s0,
                        );
                        let j_low = (j_base + pair_y) & (num_first - 1);
                        let e_in = e_first[j_low];
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
                    blk = blk_end;
                }

                outer_accum
            })
            .reduce(
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += *bi;
                    }
                    a
                },
            )
            .into_iter()
            .map(E::reduce_product_accum)
            .collect::<Vec<_>>();

        #[cfg(not(feature = "parallel"))]
        let q_coeffs = {
            let mut outer = vec![E::ProductAccum::zero(); num_coeffs_q];
            for (x, col_out) in out.chunks_mut(next_y_len).enumerate() {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let col = &s_full[x * y_len..(x + 1) * y_len];
                let j_base = x * current_y_half;
                let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                    let blk_len = blk_end - blk;
                    let full_chunks = blk_len / 4;

                    for chunk in 0..full_chunks {
                        let pair_base = blk + chunk * 4;
                        let mut pairs = [(E::zero(), E::zero()); 4];
                        for (slot, pair_y) in (pair_base..pair_base + 4).enumerate() {
                            let top_y = 2 * pair_y;
                            let top = 4 * pair_y;
                            let s0 = col[top] + r * (col[top + 1] - col[top]);
                            let s1 = col[top + 2] + r * (col[top + 3] - col[top + 2]);
                            col_out[top_y] = s0;
                            col_out[top_y + 1] = s1;
                            pairs[slot] = (s0, s1);
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
                            let pair_y = pair_base + slot;
                            let j_low = (j_base + pair_y) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_y in blk + full_chunks * 4..blk_end {
                        let top_y = 2 * pair_y;
                        let top = 4 * pair_y;
                        let s0 = col[top] + r * (col[top + 1] - col[top]);
                        let s1 = col[top + 2] + r * (col[top + 3] - col[top + 2]);
                        col_out[top_y] = s0;
                        col_out[top_y + 1] = s1;
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s0,
                            s1 - s0,
                        );
                        let j_low = (j_base + pair_y) & (num_first - 1);
                        let e_in = e_first[j_low];
                        accumulate_dense_entry_coeffs(
                            &mut inner_accum[..num_coeffs_q],
                            &entry_buf[..full_num_coeffs_q],
                            e_in,
                        );
                    }

                    let e_out = e_second[j_high];
                    for k in 0..num_coeffs_q {
                        let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                        outer[k] += e_out.mul_to_product_accum(inner_reduced);
                    }
                    blk = blk_end;
                }
            }
            outer
                .into_iter()
                .map(E::reduce_product_accum)
                .collect::<Vec<_>>()
        };

        let poly = EqFactoredUniPoly::from_q_coeffs(q_coeffs);
        (out, poly)
    }

    fn compute_current_round_eq_poly_from_state(&mut self) -> EqFactoredUniPoly<E> {
        let use_two_round_prefix = self.using_two_round_prefix();
        let use_prefix_x_round = !use_two_round_prefix && self.use_prefix_x_round();
        let use_sparse_x_y_round = !use_two_round_prefix && self.use_sparse_x_y_round();
        let t_round = Instant::now();
        let rounds_completed = self.rounds_completed;
        let poly = if use_two_round_prefix {
            let prefix = self.ensure_two_round_prefix();
            if rounds_completed == 0 {
                prefix.skip_state.reconstruct_round0_eq_poly()
            } else {
                let r0 = prefix
                    .first_challenge
                    .expect("round 1 prefix polynomial requested before ingesting round 0");
                prefix.skip_state.reconstruct_round1_eq_poly(r0)
            }
        } else if self.split_eq.current_scalar().is_zero() {
            EqFactoredUniPoly::from_q_coeffs(vec![E::zero()])
        } else {
            match &self.s_table {
                STable::Compact(s_compact) => {
                    if use_prefix_x_round {
                        self.compute_round_compact_prefix_x(s_compact)
                    } else if use_sparse_x_y_round {
                        self.compute_round_compact_sparse_x_y(s_compact)
                    } else {
                        compute_norm_round_eq_poly_from_s_compact(
                            &self.split_eq,
                            s_compact,
                            &self.range_precomp,
                        )
                    }
                }
                STable::Full(s_full) => {
                    if use_prefix_x_round {
                        self.compute_round_full_prefix_x(s_full)
                    } else if use_sparse_x_y_round {
                        self.compute_round_full_sparse_x_y(s_full)
                    } else {
                        compute_norm_round_eq_poly_from_s(
                            &self.split_eq,
                            &self.range_precomp,
                            |j| (s_full[2 * j], s_full[2 * j + 1]),
                        )
                    }
                }
            }
        };

        if use_two_round_prefix || use_prefix_x_round {
            self.prefix_time_total += t_round.elapsed().as_secs_f64();
        } else {
            self.dense_time_total += t_round.elapsed().as_secs_f64();
        }

        poly
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::compute_round_compact_prefix_x")]
    fn compute_round_compact_prefix_x(&self, s_compact: &[i16]) -> EqFactoredUniPoly<E> {
        debug_assert!(self.rounds_completed < self.col_bits);
        debug_assert_eq!(
            s_compact.len(),
            self.live_x_cols * (1usize << (self.num_vars - self.col_bits))
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        let rp = &self.range_precomp;
        let full_num_coeffs_q = rp.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let q_coeffs = if rp.compact_coeffs_lut(0, 0).is_some() {
            cfg_fold_reduce!(
                0..(1usize << (self.num_vars - self.col_bits)),
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                |mut outer_accum, y| {
                    let row_start = y * self.live_x_cols;
                    let row = &s_compact[row_start..row_start + self.live_x_cols];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let blk_end = (blk + block_size).min(live_pairs);
                        let j_high = (j_base + blk) >> first_bits;
                        let mut inner_pos = [E::MulU64Accum::zero(); MAX_AFFINE_COEFFS];
                        let mut inner_neg = [E::MulU64Accum::zero(); MAX_AFFINE_COEFFS];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
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
                        blk = blk_end;
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
            .collect()
        } else if rp.field_coeffs_lut(0, 0).is_some() {
            cfg_fold_reduce!(
                0..(1usize << (self.num_vars - self.col_bits)),
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                |mut outer_accum, y| {
                    let row_start = y * self.live_x_cols;
                    let row = &s_compact[row_start..row_start + self.live_x_cols];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let blk_end = (blk + block_size).min(live_pairs);
                        let j_high = (j_base + blk) >> first_bits;
                        let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let s0_i = row[left];
                            let s1_i = if left + 1 < self.live_x_cols {
                                row[left + 1]
                            } else {
                                0
                            };
                            let coeffs = rp
                                .field_coeffs_lut(s0_i, s1_i)
                                .expect("missing field coefficient LUT");
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                coeffs,
                                e_in,
                            );
                        }

                        let e_out = e_second[j_high];
                        for k in 0..num_coeffs_q {
                            let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                            outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                        }
                        blk = blk_end;
                    }
                    outer_accum
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
        } else {
            cfg_fold_reduce!(
                0..(1usize << (self.num_vars - self.col_bits)),
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                |mut outer_accum, y| {
                    let row_start = y * self.live_x_cols;
                    let row = &s_compact[row_start..row_start + self.live_x_cols];
                    let j_base = y * current_x_half;
                    let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                    let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let blk_end = (blk + block_size).min(live_pairs);
                        let j_high = (j_base + blk) >> first_bits;
                        let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
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
                                E::from_i64(i64::from(s0_i)),
                                E::from_i64(i64::from(s1_i - s0_i)),
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
                        blk = blk_end;
                    }
                    outer_accum
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

        EqFactoredUniPoly::from_q_coeffs(q_coeffs)
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::compute_round_full_prefix_x")]
    fn compute_round_full_prefix_x(&self, s_full: &[E]) -> EqFactoredUniPoly<E> {
        debug_assert!(self.rounds_completed < self.col_bits);
        let y_len = s_full.len() / self.live_x_cols;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        let range_pc = &self.range_precomp;
        let full_num_coeffs_q = range_pc.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let q_coeffs = cfg_fold_reduce!(
            0..y_len,
            || vec![E::ProductAccum::zero(); num_coeffs_q],
            |mut outer_accum, y| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let row_start = y * self.live_x_cols;
                let row = &s_full[row_start..row_start + self.live_x_cols];
                let j_base = y * current_x_half;
                let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                    let blk_len = blk_end - blk;
                    let full_chunks = blk_len / 4;

                    for chunk in 0..full_chunks {
                        let pair_base = blk + chunk * 4;
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
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_x in blk + full_chunks * 4..blk_end {
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
                        let j_low = (j_base + pair_x) & (num_first - 1);
                        let e_in = e_first[j_low];
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
                    blk = blk_end;
                }

                outer_accum
            },
            |mut ca, cb| {
                for (ai, bi) in ca.iter_mut().zip(cb.iter()) {
                    *ai += *bi;
                }
                ca
            }
        );

        let q_coeffs: Vec<E> = q_coeffs.into_iter().map(E::reduce_product_accum).collect();
        EqFactoredUniPoly::from_q_coeffs(q_coeffs)
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::fold_s_compact_prefix_x")]
    fn fold_s_compact_prefix_x(
        s_compact: &[i16],
        live_x_cols: usize,
        y_len: usize,
        fold_lut: &CompactPairFoldLut<E>,
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
                    let s_1 = if left + 1 < live_x_cols {
                        row[left + 1]
                    } else {
                        0
                    };
                    *dst = fold_lut.fold(row[left], s_1);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row_start = y * live_x_cols;
            let row = &s_compact[row_start..row_start + live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let s_1 = if left + 1 < live_x_cols {
                    row[left + 1]
                } else {
                    0
                };
                *dst = fold_lut.fold(row[left], s_1);
            }
        }

        out
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::fold_s_full_prefix_x")]
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

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::fold_s_full_sparse_x_y")]
    fn fold_s_full_sparse_x_y(s_full: &[E], live_x_cols: usize, y_len: usize, r: E) -> Vec<E> {
        debug_assert_eq!(y_len % 2, 0);
        let next_y_len = y_len / 2;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_y_len)
            .enumerate()
            .for_each(|(x, col_out)| {
                let col = &s_full[x * y_len..(x + 1) * y_len];
                for (pair_y, dst) in col_out.iter_mut().enumerate() {
                    let top = 2 * pair_y;
                    let s_0 = col[top];
                    let s_1 = col[top + 1];
                    *dst = s_0 + r * (s_1 - s_0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (x, col_out) in out.chunks_mut(next_y_len).enumerate() {
            let col = &s_full[x * y_len..(x + 1) * y_len];
            for (pair_y, dst) in col_out.iter_mut().enumerate() {
                let top = 2 * pair_y;
                let s_0 = col[top];
                let s_1 = col[top + 1];
                *dst = s_0 + r * (s_1 - s_0);
            }
        }

        out
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::fold_s_compact_to_full")]
    fn fold_s_compact_to_full(s_compact: &[i16], fold_lut: &CompactPairFoldLut<E>) -> Vec<E> {
        cfg_into_iter!(0..s_compact.len() / 2)
            .map(|j| fold_lut.fold(s_compact[2 * j], s_compact[2 * j + 1]))
            .collect()
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> EqFactoredSumcheckInstanceProver<E>
    for AkitaStage1Prover<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        self.b / 2
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn current_linear_factor_evals(&self) -> (E, E) {
        self.split_eq.linear_factor_evals()
    }

    fn compute_round_eq_factored(&mut self, _round: usize) -> EqFactoredUniPoly<E> {
        if let Some(poly) = self.cached_round_poly.take() {
            poly
        } else {
            self.compute_current_round_eq_poly_from_state()
        }
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let t_fold = Instant::now();
        let _span = tracing::info_span!("AkitaStage1Prover::fold_round").entered();
        if self.using_two_round_prefix() {
            let rounds_completed = self.rounds_completed;
            self.split_eq.bind(r);
            if rounds_completed == 0 {
                self.ensure_two_round_prefix().first_challenge = Some(r);
            } else {
                let r0 = {
                    let prefix = self.ensure_two_round_prefix();
                    prefix
                        .first_challenge
                        .expect("round 1 ingest requires the round 0 challenge")
                };
                let y_len = match &self.s_table {
                    STable::Compact(s_compact) => s_compact.len() / self.live_x_cols,
                    STable::Full(_) => panic!("two-round prefix expected compact table"),
                };
                self.s_table = match std::mem::replace(&mut self.s_table, STable::Full(Vec::new()))
                {
                    STable::Compact(s_compact) => {
                        if self.ring_bits() > 2 {
                            let (s_full, round_poly) =
                                self.fuse_compact_to_round2_and_compute_round(&s_compact, r0, r);
                            self.cached_round_poly = Some(round_poly);
                            STable::Full(s_full)
                        } else {
                            let s_full = Self::fold_s_compact_to_round2(
                                &s_compact,
                                self.live_x_cols,
                                y_len,
                                r0,
                                r,
                            );
                            STable::Full(s_full)
                        }
                    }
                    STable::Full(_) => unreachable!("two-round prefix should hold compact table"),
                };
            }
            self.rounds_completed += 1;
            if self.rounds_completed < self.num_vars {
                if self.cached_round_poly.is_none() {
                    self.cached_round_poly = Some(self.compute_current_round_eq_poly_from_state());
                }
            } else {
                self.cached_round_poly = None;
            }
            drop(_span);
            self.fold_time_total += t_fold.elapsed().as_secs_f64();
            return;
        }

        self.split_eq.bind(r);
        let use_prefix_x_round = self.use_prefix_x_round();
        let use_sparse_x_y_round = self.use_sparse_x_y_round();
        let fuse_next_full_prefix_x =
            use_prefix_x_round && self.next_use_prefix_x_round_after_current();
        let fuse_next_sparse_x_y =
            use_sparse_x_y_round && self.next_use_sparse_x_y_round_after_current();
        let y_len = match &self.s_table {
            STable::Compact(s_compact) => s_compact.len() / self.live_x_cols,
            STable::Full(s_full) => s_full.len() / self.live_x_cols,
        };

        self.s_table = match std::mem::replace(&mut self.s_table, STable::Full(Vec::new())) {
            STable::Compact(s_compact) => {
                let fold_lut = Self::build_compact_s_fold_lut(self.b, r);
                let s_full = if use_prefix_x_round {
                    Self::fold_s_compact_prefix_x(&s_compact, self.live_x_cols, y_len, &fold_lut)
                } else {
                    Self::fold_s_compact_to_full(&s_compact, &fold_lut)
                };
                STable::Full(s_full)
            }
            STable::Full(s_full) => {
                if use_prefix_x_round {
                    if fuse_next_full_prefix_x {
                        let (next_s_full, round_poly) =
                            self.fuse_full_prefix_x_and_compute_round(&s_full, r);
                        self.cached_round_poly = Some(round_poly);
                        STable::Full(next_s_full)
                    } else {
                        let next_s_full =
                            Self::fold_s_full_prefix_x(&s_full, self.live_x_cols, y_len, r);
                        STable::Full(next_s_full)
                    }
                } else if use_sparse_x_y_round {
                    if fuse_next_sparse_x_y {
                        let (next_s_full, round_poly) =
                            self.fuse_full_sparse_x_y_and_compute_round(&s_full, r);
                        self.cached_round_poly = Some(round_poly);
                        STable::Full(next_s_full)
                    } else {
                        let next_s_full =
                            Self::fold_s_full_sparse_x_y(&s_full, self.live_x_cols, y_len, r);
                        STable::Full(next_s_full)
                    }
                } else {
                    let mut s_full = s_full;
                    fold_evals_in_place(&mut s_full, r);
                    STable::Full(s_full)
                }
            }
        };

        if self.in_x_phase() {
            self.live_x_cols = self.live_x_cols.div_ceil(2);
        }
        self.rounds_completed += 1;
        if self.rounds_completed < self.num_vars {
            if self.cached_round_poly.is_none() {
                self.cached_round_poly = Some(self.compute_current_round_eq_poly_from_state());
            }
        } else {
            self.cached_round_poly = None;
        }
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

#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn pad_compact_witness(
    w_prefix: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Vec<i8> {
    let x_len = 1usize << col_bits;
    let y_len = 1usize << ring_bits;
    let mut padded = vec![0i8; x_len * y_len];
    for x in 0..live_x_cols {
        let offset = x * y_len;
        padded[offset..offset + y_len].copy_from_slice(&w_prefix[offset..offset + y_len]);
    }
    padded
}

#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn advance_stage1_claim<
    F: FieldCore + FromPrimitiveInt + akita_field::CanonicalField + HasUnreducedOps,
>(
    prover: &AkitaStage1Prover<F>,
    scaled_claim: F,
    claim_scale: F,
    poly: &EqFactoredUniPoly<F>,
    challenge: F,
) -> (F, F) {
    use akita_sumcheck::advance_eq_factored_claim;
    let (l_at_0, l_at_1) = prover.current_linear_factor_evals();
    advance_eq_factored_claim(scaled_claim, claim_scale, l_at_0, l_at_1, poly, challenge)
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;
    use akita_sumcheck::multilinear_eval;
    use akita_types::reorder_stage1_coords;

    type F = Prime128Offset275;

    fn fold_s_compact_prefix_x_reference(
        s_compact: &[i16],
        live_x_cols: usize,
        y_len: usize,
        r: F,
    ) -> Vec<F> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![F::zero(); y_len * next_live_x_cols];
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row_start = y * live_x_cols;
            let row = &s_compact[row_start..row_start + live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let s_0 = F::from_i64(i64::from(row[left]));
                let s_1 = if left + 1 < live_x_cols {
                    F::from_i64(i64::from(row[left + 1]))
                } else {
                    F::zero()
                };
                *dst = s_0 + r * (s_1 - s_0);
            }
        }
        out
    }

    fn fold_s_compact_to_full_reference(s_compact: &[i16], r: F) -> Vec<F> {
        (0..s_compact.len() / 2)
            .map(|j| {
                let s_0 = F::from_i64(i64::from(s_compact[2 * j]));
                let s_1 = F::from_i64(i64::from(s_compact[2 * j + 1]));
                s_0 + r * (s_1 - s_0)
            })
            .collect()
    }

    #[test]
    fn stage1_compact_fold_lookup_matches_direct_formula() {
        let b = 8usize;
        let r = F::from_u64(41);

        let s_prefix = vec![2, 6, 12, 2, 6, 12, 2, 6, 12, 2];
        let fold_lut = AkitaStage1Prover::<F>::build_compact_s_fold_lut(b, r);
        assert_eq!(
            AkitaStage1Prover::<F>::fold_s_compact_prefix_x(&s_prefix, 5, 2, &fold_lut),
            fold_s_compact_prefix_x_reference(&s_prefix, 5, 2, r)
        );

        let s_dense = vec![2, 6, 12, 2, 6, 12];
        let dense_lut = AkitaStage1Prover::<F>::build_compact_s_fold_lut(b, r);
        assert_eq!(
            AkitaStage1Prover::<F>::fold_s_compact_to_full(&s_dense, &dense_lut),
            fold_s_compact_to_full_reference(&s_dense, r)
        );
    }

    #[test]
    fn stage1_round0_matches_dense_reference() {
        let col_bits = 3usize;
        let ring_bits = 2usize;
        let n = 1usize << (col_bits + ring_bits);
        let tau0: Vec<F> = (0..(col_bits + ring_bits))
            .map(|i| F::from_u64((i as u64) + 2))
            .collect();
        let tau0 = reorder_stage1_coords(&tau0, col_bits, ring_bits);

        for b in [4usize, 8, 16, 32] {
            let half = (b / 2) as i8;
            let w_compact: Vec<i8> = (0..n).map(|i| ((i * 5 + 3) % b) as i8 - half).collect();

            let mut prover = AkitaStage1Prover::new(
                &w_compact,
                &tau0,
                b,
                1usize << col_bits,
                col_bits,
                ring_bits,
            );
            let stage1_poly = prover.compute_round_eq_factored(0);
            let s_compact = build_compact_s_table(&w_compact);
            let reference = compute_norm_round_eq_poly_from_s_compact(
                &prover.split_eq,
                &s_compact,
                &prover.range_precomp,
            );

            assert_eq!(stage1_poly, reference, "stage1 round0 mismatch for b={b}");
        }
    }

    #[test]
    fn stage1_compact_coeff_lut_reaches_b16() {
        for b in [4usize, 8, 16] {
            let precomp = RangeAffineFromSPrecomp::<F>::new(b);
            assert!(
                precomp.compact_coeffs_lut(0, 0).is_some(),
                "expected compact coefficient LUT for b={b}"
            );
        }

        let precomp = RangeAffineFromSPrecomp::<F>::new(32);
        assert!(precomp.compact_coeffs_lut(0, 0).is_none());
    }

    #[test]
    fn stage1_field_coeff_lut_reaches_b32() {
        for b in [4usize, 8, 16] {
            let precomp = RangeAffineFromSPrecomp::<F>::new(b);
            assert!(precomp.field_coeffs_lut(0, 0).is_none());
        }

        let precomp = RangeAffineFromSPrecomp::<F>::new(32);
        assert!(
            precomp.field_coeffs_lut(0, 0).is_some(),
            "expected field coefficient LUT for b=32"
        );
    }

    #[test]
    fn stage1_prefix_aware_rounds_match_explicit_zero_padding() {
        let ring_bits = 2usize;
        for b in [4usize, 8, 16, 32] {
            let half = (b / 2) as i8;
            for live_x_cols in [5usize, 6usize] {
                let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
                let y_len = 1usize << ring_bits;
                let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
                    .map(|i| ((i * 7 + 5) % b) as i8 - half)
                    .collect();
                let w_padded = pad_compact_witness(&w_prefix, live_x_cols, col_bits, ring_bits);
                let tau0: Vec<F> = (0..(col_bits + ring_bits))
                    .map(|i| F::from_u64((i as u64) + 19))
                    .collect();
                let tau0 = reorder_stage1_coords(&tau0, col_bits, ring_bits);
                let mut prefix_prover =
                    AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits);
                let mut padded_prover = AkitaStage1Prover::new(
                    &w_padded,
                    &tau0,
                    b,
                    1usize << col_bits,
                    col_bits,
                    ring_bits,
                );
                let mut challenges = Vec::new();
                let mut prefix_claim = F::zero();
                let mut prefix_scale = F::one();
                let mut padded_claim = F::zero();
                let mut padded_scale = F::one();

                for round in 0..(col_bits + ring_bits) {
                    let prefix_poly = prefix_prover.compute_round_eq_factored(round);
                    let padded_poly = padded_prover.compute_round_eq_factored(round);
                    assert_eq!(
                        prefix_poly, padded_poly,
                        "round {round} polynomial mismatch live_x_cols={live_x_cols} b={b}"
                    );

                    let challenge = F::from_u64((round as u64) + 29);
                    challenges.push(challenge);
                    (prefix_claim, prefix_scale) = advance_stage1_claim(
                        &prefix_prover,
                        prefix_claim,
                        prefix_scale,
                        &prefix_poly,
                        challenge,
                    );
                    (padded_claim, padded_scale) = advance_stage1_claim(
                        &padded_prover,
                        padded_claim,
                        padded_scale,
                        &padded_poly,
                        challenge,
                    );
                    prefix_prover.ingest_challenge(round, challenge);
                    padded_prover.ingest_challenge(round, challenge);
                }

                assert_eq!(prefix_prover.final_s_claim(), padded_prover.final_s_claim());
                assert_eq!(prefix_claim, padded_claim);
                assert_eq!(prefix_scale, padded_scale);
                let s_padded: Vec<F> = build_compact_s_table(&w_padded)
                    .into_iter()
                    .map(|s| F::from_i64(i64::from(s)))
                    .collect();
                assert_eq!(
                    prefix_prover.final_s_claim(),
                    multilinear_eval(&s_padded, &challenges).unwrap(),
                    "final s-claim mismatch live_x_cols={live_x_cols} b={b}"
                );
            }
        }
    }

    #[test]
    fn stage1_fused_round2_transition_matches_two_pass_reference() {
        let col_bits = 3usize;
        let ring_bits = 2usize;
        let live_x_cols = 6usize;
        let y_len = 1usize << ring_bits;
        for b in [4usize, 8] {
            let half = (b / 2) as i8;
            let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
                .map(|i| ((i * 9 + 5) % b) as i8 - half)
                .collect();
            let s_compact = build_compact_s_table(&w_prefix);
            let tau0: Vec<F> = (0..(col_bits + ring_bits))
                .map(|i| F::from_u64((i as u64) + 53))
                .collect();
            let tau0 = reorder_stage1_coords(&tau0, col_bits, ring_bits);

            let mut prover =
                AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits);
            let round0 = prover.compute_round_eq_factored(0);
            let r0 = F::from_u64(61);
            let (claim1, scale1) = advance_stage1_claim(&prover, F::zero(), F::one(), &round0, r0);
            prover.ingest_challenge(0, r0);
            let round1 = prover.compute_round_eq_factored(1);
            let r1 = F::from_u64(67);
            let (_claim2, _scale2) = advance_stage1_claim(&prover, claim1, scale1, &round1, r1);

            let expected_s_full = AkitaStage1Prover::<F>::fold_s_compact_to_round2(
                &s_compact,
                live_x_cols,
                y_len,
                r0,
                r1,
            );
            let mut expected =
                AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits);
            expected.split_eq.bind(r0);
            expected.split_eq.bind(r1);
            expected.rounds_completed = 2;
            let expected_round2 = expected.compute_round_full_prefix_x(&expected_s_full);

            prover.ingest_challenge(1, r1);

            match &prover.s_table {
                STable::Full(s_full) => assert_eq!(s_full, &expected_s_full),
                STable::Compact(_) => {
                    panic!("expected fused stage1 transition to materialize full table")
                }
            }
            assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round2));
        }
    }

    #[test]
    fn stage1_later_full_prefix_fusion_matches_two_pass_reference() {
        let col_bits = 5usize;
        let ring_bits = 2usize;
        let live_x_cols = 12usize;
        let y_len = 1usize << ring_bits;
        for b in [4usize, 8] {
            let half = (b / 2) as i8;
            let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
                .map(|i| ((i * 5 + 11) % b) as i8 - half)
                .collect();
            let tau0: Vec<F> = (0..(col_bits + ring_bits))
                .map(|i| F::from_u64((i as u64) + 101))
                .collect();
            let tau0 = reorder_stage1_coords(&tau0, col_bits, ring_bits);

            let mut prover =
                AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits);
            let round0 = prover.compute_round_eq_factored(0);
            let r0 = F::from_u64(107);
            let (claim1, scale1) = advance_stage1_claim(&prover, F::zero(), F::one(), &round0, r0);
            prover.ingest_challenge(0, r0);

            let round1 = prover.compute_round_eq_factored(1);
            let r1 = F::from_u64(109);
            let (claim2, scale2) = advance_stage1_claim(&prover, claim1, scale1, &round1, r1);
            prover.ingest_challenge(1, r1);

            let round2 = prover.compute_round_eq_factored(2);
            let r2 = F::from_u64(113);
            let (claim3, _scale3) = advance_stage1_claim(&prover, claim2, scale2, &round2, r2);

            let mut expected =
                AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits);
            let expected_round0 = expected.compute_round_eq_factored(0);
            assert_eq!(expected_round0, round0);
            expected.ingest_challenge(0, r0);
            let expected_round1 = expected.compute_round_eq_factored(1);
            assert_eq!(expected_round1, round1);
            expected.ingest_challenge(1, r1);
            let expected_round2 = expected.compute_round_eq_factored(2);
            assert_eq!(expected_round2, round2);

            let current_s_full = match &expected.s_table {
                STable::Full(s_full) => s_full.clone(),
                STable::Compact(_) => panic!("expected later prefix state to be full"),
            };
            let current_y_len = current_s_full.len() / expected.live_x_cols;
            let expected_next_s_full = AkitaStage1Prover::<F>::fold_s_full_prefix_x(
                &current_s_full,
                expected.live_x_cols,
                current_y_len,
                r2,
            );
            expected.split_eq.bind(r2);
            expected.live_x_cols = expected.live_x_cols.div_ceil(2);
            expected.rounds_completed += 1;
            let _ = claim3;
            let expected_round3 = expected.compute_round_full_prefix_x(&expected_next_s_full);

            prover.ingest_challenge(2, r2);

            match &prover.s_table {
                STable::Full(s_full) => assert_eq!(s_full, &expected_next_s_full),
                STable::Compact(_) => panic!("expected fused later prefix stage to stay full"),
            }
            assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round3));
        }
    }

    #[test]
    fn stage1_sparse_x_y_fusion_matches_two_pass_reference() {
        let col_bits = 3usize;
        let ring_bits = 4usize;
        let live_x_cols = 6usize;
        let y_len = 1usize << ring_bits;
        for b in [4usize, 8] {
            let half = (b / 2) as i8;
            let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
                .map(|i| ((i * 7 + 9) % b) as i8 - half)
                .collect();
            let tau0: Vec<F> = (0..(col_bits + ring_bits))
                .map(|i| F::from_u64((i as u64) + 131))
                .collect();
            let tau0 = reorder_stage1_coords(&tau0, col_bits, ring_bits);

            let mut prover =
                AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits);
            let round0 = prover.compute_round_eq_factored(0);
            let r0 = F::from_u64(137);
            let (claim1, scale1) = advance_stage1_claim(&prover, F::zero(), F::one(), &round0, r0);
            prover.ingest_challenge(0, r0);

            let round1 = prover.compute_round_eq_factored(1);
            let r1 = F::from_u64(139);
            let (claim2, scale2) = advance_stage1_claim(&prover, claim1, scale1, &round1, r1);
            prover.ingest_challenge(1, r1);

            let round2 = prover.compute_round_eq_factored(2);
            let r2 = F::from_u64(149);
            let (_claim3, _scale3) = advance_stage1_claim(&prover, claim2, scale2, &round2, r2);

            let mut expected =
                AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits);
            let expected_round0 = expected.compute_round_eq_factored(0);
            assert_eq!(expected_round0, round0);
            expected.ingest_challenge(0, r0);
            let expected_round1 = expected.compute_round_eq_factored(1);
            assert_eq!(expected_round1, round1);
            expected.ingest_challenge(1, r1);
            let expected_round2 = expected.compute_round_eq_factored(2);
            assert_eq!(expected_round2, round2);

            let current_s_full = match &expected.s_table {
                STable::Full(s_full) => s_full.clone(),
                STable::Compact(_) => panic!("expected sparse-x/y state to be full"),
            };
            let current_y_len = current_s_full.len() / expected.live_x_cols;
            let expected_next_s_full = AkitaStage1Prover::<F>::fold_s_full_sparse_x_y(
                &current_s_full,
                expected.live_x_cols,
                current_y_len,
                r2,
            );
            expected.split_eq.bind(r2);
            expected.rounds_completed += 1;
            let expected_round3 = expected.compute_round_full_sparse_x_y(&expected_next_s_full);

            prover.ingest_challenge(2, r2);

            match &prover.s_table {
                STable::Full(s_full) => assert_eq!(s_full, &expected_next_s_full),
                STable::Compact(_) => panic!("expected sparse-x/y fusion to stay full"),
            }
            assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round3));
        }
    }
}
