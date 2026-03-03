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
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::{cfg_fold_reduce, CanonicalField, FieldCore, FromSmallInt};

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
    if b <= 16 {
        NormRoundKernel::AffineCoeffComposition
    } else {
        NormRoundKernel::PointEvalInterpolation
    }
}

#[derive(Clone)]
pub(crate) struct RangeAffinePrecomp<E: FieldCore> {
    /// Flat contiguous storage of `coeff_mix[i][k] = c_{i+k} * binom(i+k, i)`,
    /// where `R(w) = sum_m c_m * w^m` is the range-check polynomial.
    /// Row `i` has length `degree_q - i + 1` and starts at
    /// `row_offsets[i]`.
    coeff_mix_flat: Vec<E>,
    row_offsets: Vec<usize>,
    pub(crate) degree_q: usize,
    /// Precomputed `h_i(w_0)` for all small-integer `w_0 ∈ {-(b-1),...,b-1}`.
    /// Indexed as `small_w_lut[(w_0 + b - 1) * num_rows + i]`.
    /// Used by the round-0 compact accumulation path.
    small_w_lut: Vec<E>,
    b: usize,
}

impl<E: FieldCore + FromSmallInt> RangeAffinePrecomp<E> {
    pub(crate) fn new(b: usize) -> Self {
        assert!(b >= 1, "b must be at least 1");
        let range_coeffs = range_check_coeffs::<E>(b);
        let degree_q = range_coeffs.len() - 1;
        let small_scalars: Vec<E> = (0..=degree_q + 1).map(|x| E::from_u64(x as u64)).collect();
        let inv_small_scalars: Vec<E> = (0..=degree_q + 1)
            .map(|x| {
                if x == 0 {
                    E::zero()
                } else {
                    small_scalars[x]
                        .inv()
                        .expect("field characteristic too small for range-check precomputation")
                }
            })
            .collect();

        let total_elems = (degree_q + 1) * (degree_q + 2) / 2;
        let mut coeff_mix_flat = Vec::with_capacity(total_elems);
        let mut row_offsets = Vec::with_capacity(degree_q + 2);

        for i in 0..=degree_q {
            row_offsets.push(coeff_mix_flat.len());
            let row_len = degree_q - i + 1;
            let mut binom_m_i = E::one(); // binom(i, i)
            for k in 0..row_len {
                let m = i + k;
                coeff_mix_flat.push(range_coeffs[m] * binom_m_i);
                if k + 1 < row_len {
                    let numer = small_scalars[m + 1];
                    let denom_inv = inv_small_scalars[k + 1];
                    binom_m_i = binom_m_i * numer * denom_inv;
                }
            }
        }
        row_offsets.push(coeff_mix_flat.len());

        let num_rows = degree_q + 1;
        let num_w_vals = 2 * b - 1; // w_0 ∈ {-(b-1),...,b-1}
        let mut small_w_lut = vec![E::zero(); num_w_vals * num_rows];
        for (w_idx, w_0_int) in (-(b as i64 - 1)..=(b as i64 - 1)).enumerate() {
            let w_0 = E::from_i64(w_0_int);
            for i in 0..num_rows {
                let row = &coeff_mix_flat[row_offsets[i]..row_offsets[i + 1]];
                let (&last, rest) = row.split_last().unwrap();
                let mut h = last;
                for &c in rest.iter().rev() {
                    h = h * w_0 + c;
                }
                small_w_lut[w_idx * num_rows + i] = h;
            }
        }

        Self {
            coeff_mix_flat,
            row_offsets,
            degree_q,
            small_w_lut,
            b,
        }
    }
}

impl<E: FieldCore> RangeAffinePrecomp<E> {
    #[inline]
    pub(crate) fn row(&self, i: usize) -> &[E] {
        &self.coeff_mix_flat[self.row_offsets[i]..self.row_offsets[i + 1]]
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

/// Coefficients of `R(w) = w * Π_{k=1}^{b-1}(w-k)(w+k)` in increasing degree order.
fn range_check_coeffs<E: FieldCore + FromSmallInt>(b: usize) -> Vec<E> {
    assert!(b >= 1, "b must be at least 1");
    let mut coeffs = vec![E::zero(), E::one()]; // R(w)=w when b=1
    for k in 1..b {
        let k_e = E::from_u64(k as u64);
        let k_sq = k_e * k_e;
        // Multiply by (w^2 - k^2).
        let mut next = vec![E::zero(); coeffs.len() + 2];
        for (idx, c) in coeffs.iter().enumerate() {
            next[idx] -= *c * k_sq;
            next[idx + 2] += *c;
        }
        coeffs = next;
    }
    coeffs
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

#[inline(never)]
pub(crate) fn accumulate_affine_range_coeffs<E: FieldCore>(
    out_coeffs: &mut [E],
    precomp: &RangeAffinePrecomp<E>,
    w_0: E,
    a: E,
    scale: E,
) {
    let mut a_pow = E::one();
    for (i, out) in out_coeffs.iter_mut().enumerate().take(precomp.num_rows()) {
        let row = precomp.row(i);
        debug_assert!(!row.is_empty());
        let (&last, rest) = row.split_last().unwrap();
        let mut h_i_w0 = last;
        for &coeff in rest.iter().rev() {
            h_i_w0 = h_i_w0 * w_0 + coeff;
        }
        *out += scale * a_pow * h_i_w0;
        a_pow = a_pow * a;
    }
}

/// Compact-path variant: uses precomputed `h_i(w_0)` lookup table
/// when `w_0` is a small integer (round 0 with `Vec<i8>` storage).
#[inline(never)]
pub(crate) fn accumulate_affine_range_coeffs_compact<E: FieldCore>(
    out_coeffs: &mut [E],
    precomp: &RangeAffinePrecomp<E>,
    w_0_int: i8,
    a: E,
    scale: E,
) {
    let mut a_pow = E::one();
    for (i, out) in out_coeffs.iter_mut().enumerate().take(precomp.num_rows()) {
        let h_i_w0 = precomp.h_i_lut(w_0_int, i);
        *out += scale * a_pow * h_i_w0;
        a_pow = a_pow * a;
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
pub(crate) fn compute_norm_round_poly<E: FieldCore + FromSmallInt>(
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
                    0..half,
                    || vec![E::zero(); num_coeffs_q],
                    |mut coeffs, j| {
                        let j_low = j & (num_first - 1);
                        let j_high = j >> first_bits;
                        let eq_rem = e_first[j_low] * e_second[j_high];
                        let (w_0, w_1) = w_pair(j);
                        let a = w_1 - w_0;
                        accumulate_affine_range_coeffs(&mut coeffs, rp, w_0, a, eq_rem);
                        coeffs
                    },
                    |mut a, b_vec| {
                        for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                            *ai += *bi;
                        }
                        a
                    }
                )
            };

            trim_trailing_zeros(&mut q_coeffs);
            let q_poly = UniPoly::from_coeffs(q_coeffs);
            split_eq.gruen_mul(&q_poly)
        }
    }
}

/// Compact round-0 variant: uses native i128 arithmetic (point-eval, b<=10)
/// or precomputed LUT (affine-coeff) when w values are small integers.
pub(crate) fn compute_norm_round_poly_compact<E: FieldCore + FromSmallInt + CanonicalField>(
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
                    0..half,
                    || vec![E::zero(); num_coeffs_q],
                    |mut coeffs, j| {
                        let j_low = j & (num_first - 1);
                        let j_high = j >> first_bits;
                        let eq_rem = e_first[j_low] * e_second[j_high];
                        let w_0_int = w_compact[2 * j];
                        let w_1 = E::from_i64(w_compact[2 * j + 1] as i64);
                        let a = w_1 - E::from_i64(w_0_int as i64);
                        accumulate_affine_range_coeffs_compact(&mut coeffs, rp, w_0_int, a, eq_rem);
                        coeffs
                    },
                    |mut a, b_vec| {
                        for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                            *ai += *bi;
                        }
                        a
                    }
                )
            };

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

impl<E: FieldCore + FromSmallInt> NormSumcheckProver<E> {
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

impl<E: FieldCore + FromSmallInt> SumcheckInstanceProver<E> for NormSumcheckProver<E> {
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
                    *eval = *eval + eq_rem * range_check_eval(w_t, b);
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
            let mut affine_coeff = NormSumcheckProver::new_with_kernel(
                &tau,
                w_evals.clone(),
                b,
                NormRoundKernel::AffineCoeffComposition,
            );
            let mut reference = PointEvalReferenceNormSumcheckProver::new(&tau, w_evals, b);

            let mut claim_dispatched = F::zero();
            let mut claim_point = F::zero();
            let mut claim_affine = F::zero();
            let mut claim_reference = F::zero();
            for round in 0..num_vars {
                let g_dispatch = dispatched.compute_round_univariate(round, claim_dispatched);
                let g_point = point_eval.compute_round_univariate(round, claim_point);
                let g_affine = affine_coeff.compute_round_univariate(round, claim_affine);
                let g_ref = reference.compute_round_univariate(round, claim_reference);

                assert_eq!(
                    g_point, g_ref,
                    "point-eval mismatch for case {case_idx} round {round}"
                );
                assert_eq!(
                    g_affine, g_ref,
                    "affine-coeff mismatch for case {case_idx} round {round}"
                );
                match choose_round_kernel(b) {
                    NormRoundKernel::PointEvalInterpolation => {
                        assert_eq!(
                            g_dispatch, g_point,
                            "dispatch mismatch for case {case_idx} round {round}"
                        );
                    }
                    NormRoundKernel::AffineCoeffComposition => {
                        assert_eq!(
                            g_dispatch, g_affine,
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
                claim_affine = g_affine.evaluate(&r);
                claim_reference = g_ref.evaluate(&r);
                dispatched.ingest_challenge(round, r);
                point_eval.ingest_challenge(round, r);
                affine_coeff.ingest_challenge(round, r);
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
            assert_eq!(
                claim_affine, claim_reference,
                "final affine claim mismatch for case {case_idx}"
            );
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
        let (commitment, hint) = Scheme::commit(&poly, &setup).unwrap();

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
        )
        .unwrap();

        let mut w_evals = proof.final_w.clone();
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
