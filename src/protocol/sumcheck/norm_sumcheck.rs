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
use crate::{FieldCore, FromSmallInt};

const SMALL_B_POINT_EVAL_MAX: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NormRoundKernel {
    PointEvalInterpolation,
    AffineCoeffComposition,
}

fn choose_round_kernel(b: usize) -> NormRoundKernel {
    if b <= SMALL_B_POINT_EVAL_MAX {
        NormRoundKernel::PointEvalInterpolation
    } else {
        NormRoundKernel::AffineCoeffComposition
    }
}

#[derive(Clone)]
struct RangeAffinePrecomp<E: FieldCore> {
    /// `coeff_mix[i][k] = c_{i+k} * binom(i+k, i)`, where
    /// `R(w) = sum_m c_m * w^m` is the range-check polynomial.
    coeff_mix: Vec<Vec<E>>,
    degree_q: usize,
}

impl<E: FieldCore + FromSmallInt> RangeAffinePrecomp<E> {
    fn new(b: usize) -> Self {
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
        let mut coeff_mix = Vec::with_capacity(degree_q + 1);

        for i in 0..=degree_q {
            let row_len = degree_q - i + 1;
            let mut row = Vec::with_capacity(row_len);
            let mut binom_m_i = E::one(); // binom(i, i)
            for k in 0..row_len {
                let m = i + k;
                row.push(range_coeffs[m] * binom_m_i);
                if k + 1 < row_len {
                    let numer = small_scalars[m + 1];
                    let denom_inv = inv_small_scalars[k + 1];
                    binom_m_i = binom_m_i * numer * denom_inv;
                }
            }
            coeff_mix.push(row);
        }

        Self {
            coeff_mix,
            degree_q,
        }
    }
}

#[derive(Clone)]
struct PointEvalPrecomp<E: FieldCore> {
    range_offsets: Vec<E>,
}

impl<E: FieldCore + FromSmallInt> PointEvalPrecomp<E> {
    fn new(b: usize) -> Self {
        let range_offsets = (1..b).map(|k| E::from_u64(k as u64)).collect();
        Self { range_offsets }
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
            next[idx] = next[idx] - *c * k_sq;
            next[idx + 2] = next[idx + 2] + *c;
        }
        coeffs = next;
    }
    coeffs
}

fn range_check_eval_precomputed<E: FieldCore>(w: E, range_offsets: &[E]) -> E {
    let mut acc = w;
    for &k in range_offsets {
        acc = acc * (w - k) * (w + k);
    }
    acc
}

fn accumulate_affine_range_coeffs<E: FieldCore>(
    out_coeffs: &mut [E],
    coeff_mix: &[Vec<E>],
    w_0: E,
    a: E,
    scale: E,
) {
    let mut a_pow = E::one();
    for (i, row) in coeff_mix.iter().enumerate() {
        let mut h_i_w0 = E::zero();
        for coeff in row.iter().rev() {
            h_i_w0 = h_i_w0 * w_0 + *coeff;
        }
        out_coeffs[i] = out_coeffs[i] + scale * a_pow * h_i_w0;
        a_pow = a_pow * a;
    }
}

fn trim_trailing_zeros<E: FieldCore>(coeffs: &mut Vec<E>) {
    while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.is_zero()) {
        coeffs.pop();
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

    fn compute_round_univariate_point_eval(&self) -> UniPoly<E> {
        let half = self.w_table.len() / 2;
        let degree_q = 2 * self.b - 1;
        let num_points_q = degree_q + 1;
        let point_precomp = self
            .point_precomp
            .as_ref()
            .expect("point-eval precomputation must exist");
        let range_offsets = &point_precomp.range_offsets;

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();

        #[cfg(feature = "parallel")]
        let q_evals = {
            (0..half)
                .into_par_iter()
                .fold(
                    || vec![E::zero(); num_points_q],
                    |mut evals, j| {
                        let j_low = j & (num_first - 1);
                        let j_high = j >> first_bits;
                        let eq_rem = e_first[j_low] * e_second[j_high];
                        let w_0 = self.w_table[2 * j];
                        let w_1 = self.w_table[2 * j + 1];
                        let delta = w_1 - w_0;
                        let mut w_t = w_0;
                        for eval in evals.iter_mut() {
                            *eval =
                                *eval + eq_rem * range_check_eval_precomputed(w_t, range_offsets);
                            w_t = w_t + delta;
                        }
                        evals
                    },
                )
                .reduce(
                    || vec![E::zero(); num_points_q],
                    |mut a, b_vec| {
                        for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                            *ai = *ai + *bi;
                        }
                        a
                    },
                )
        };
        #[cfg(not(feature = "parallel"))]
        let q_evals = {
            let mut evals = vec![E::zero(); num_points_q];
            for j in 0..half {
                let j_low = j & (num_first - 1);
                let j_high = j >> first_bits;
                let eq_rem = e_first[j_low] * e_second[j_high];
                let w_0 = self.w_table[2 * j];
                let w_1 = self.w_table[2 * j + 1];
                let delta = w_1 - w_0;
                let mut w_t = w_0;
                for eval in evals.iter_mut() {
                    *eval = *eval + eq_rem * range_check_eval_precomputed(w_t, range_offsets);
                    w_t = w_t + delta;
                }
            }
            evals
        };

        let q_poly = UniPoly::from_evals(&q_evals);
        self.split_eq.gruen_mul(&q_poly)
    }

    fn compute_round_univariate_affine_coeff(&self) -> UniPoly<E> {
        let half = self.w_table.len() / 2;
        let range_precomp = self
            .range_precomp
            .as_ref()
            .expect("affine-coeff precomputation must exist");
        let num_coeffs_q = range_precomp.degree_q + 1;

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let coeff_mix = &range_precomp.coeff_mix;

        #[cfg(feature = "parallel")]
        let q_coeffs = {
            (0..half)
                .into_par_iter()
                .fold(
                    || vec![E::zero(); num_coeffs_q],
                    |mut coeffs, j| {
                        let j_low = j & (num_first - 1);
                        let j_high = j >> first_bits;
                        let eq_rem = e_first[j_low] * e_second[j_high];
                        let w_0 = self.w_table[2 * j];
                        let w_1 = self.w_table[2 * j + 1];
                        let a = w_1 - w_0;
                        accumulate_affine_range_coeffs(&mut coeffs, coeff_mix, w_0, a, eq_rem);
                        coeffs
                    },
                )
                .reduce(
                    || vec![E::zero(); num_coeffs_q],
                    |mut a, b_vec| {
                        for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                            *ai = *ai + *bi;
                        }
                        a
                    },
                )
        };
        #[cfg(not(feature = "parallel"))]
        let q_coeffs = {
            let mut coeffs = vec![E::zero(); num_coeffs_q];
            for j in 0..half {
                let j_low = j & (num_first - 1);
                let j_high = j >> first_bits;
                let eq_rem = e_first[j_low] * e_second[j_high];
                let w_0 = self.w_table[2 * j];
                let w_1 = self.w_table[2 * j + 1];
                let a = w_1 - w_0;
                accumulate_affine_range_coeffs(&mut coeffs, coeff_mix, w_0, a, eq_rem);
            }
            coeffs
        };

        let mut q_coeffs = q_coeffs;
        trim_trailing_zeros(&mut q_coeffs);
        let q_poly = UniPoly::from_coeffs(q_coeffs);
        self.split_eq.gruen_mul(&q_poly)
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
        match self.round_kernel {
            NormRoundKernel::PointEvalInterpolation => self.compute_round_univariate_point_eval(),
            NormRoundKernel::AffineCoeffComposition => self.compute_round_univariate_affine_coeff(),
        }
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
    use crate::algebra::ring::CyclotomicRing;
    use crate::algebra::Fp64;
    use crate::primitives::multilinear_evals::DenseMultilinearEvals;
    use crate::protocol::ring_switch::build_w_coeffs;
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

    type F = Fp64<4294967197>;
    const D: usize = 8;
    type Cfg = SmallTestCommitmentConfig;
    type Scheme = HachiCommitmentScheme<{ Cfg::D }, Cfg>;

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
        let coeffs = std::array::from_fn(|_| F::from_u64(value));
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
        let z = vec![
            ring_with_small_coeff(1),
            ring_with_small_coeff(2),
            ring_with_small_coeff(3),
        ];
        let r = vec![ring_with_small_coeff(0), ring_with_small_coeff(1)];
        let mut w_evals = build_w_coeffs::<F, D, SmallTestCommitmentConfig>(&z, &r);

        let target_len = w_evals.len().next_power_of_two();
        w_evals.resize(target_len, F::zero());
        let num_vars = target_len.trailing_zeros() as usize;
        let tau: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let b = 1usize << SmallTestCommitmentConfig::LOG_BASIS;

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
        let poly = DenseMultilinearEvals::new_padded(evals);

        let setup = Scheme::setup_prover(num_vars);
        let (commitment, hint) = Scheme::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let proof = Scheme::prove(
            &setup,
            &poly,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
            &commitment,
        )
        .unwrap();

        let mut w_evals = proof.sumcheck_aux.w.clone();
        let target_len = w_evals.len().next_power_of_two();
        w_evals.resize(target_len, F::zero());
        let num_sumcheck_vars = target_len.trailing_zeros() as usize;
        let tau: Vec<F> = (0..num_sumcheck_vars)
            .map(|i| F::from_u64((i + 3) as u64))
            .collect();
        let b = 1usize << SmallTestCommitmentConfig::LOG_BASIS;

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
        use crate::algebra::fields::ext::Ext2;
        use crate::algebra::fields::lift::LiftBase;

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
}
