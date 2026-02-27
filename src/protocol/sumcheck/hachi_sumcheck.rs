//! Concrete sumcheck instances for the Hachi protocol polynomials F_0 and F_α.
//!
//! - **F_{0,τ₀}(x, y)** = ẽq(τ₀,(x,y)) · w̃(x,y) · (w̃−1)(w̃+1)···(w̃−b+1)(w̃+b−1)
//!   Proves the range-check relation; sum over the boolean hypercube should equal the
//!   caller-supplied claim (zero when w̃ takes values in {−(b−1),…,b−1}).
//!
//! - **F_{α,τ₁}(x, y)** = w̃(x,y) · α̃(y) · m(x)
//!   where m(x) = Σ_i ẽq(τ₁,i) · M̃_α(i,x).
//!   Proves the evaluation relation; sum equals `a = Σ_i ẽq(τ₁,i) · y_i(α)`.

use super::{SumcheckInstanceProver, SumcheckInstanceVerifier, UniPoly};
use crate::algebra::ring::CyclotomicRing;
use crate::protocol::commitment_scheme::eval_ring_at;
use crate::{CanonicalField, FieldCore};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the full eq polynomial evaluation table.
///
/// Returns a vector of size `2^{tau.len()}` where entry `b` (interpreted as a
/// little-endian bit string, i.e. bit `k` of `b` corresponds to `τ[k]`) equals
/// `ẽq(τ, b) = Π_i (τ_i·b_i + (1−τ_i)(1−b_i))`.
///
/// The table is compatible with [`fold_evals`] and [`multilinear_eval`], which
/// bind variables starting from bit 0 (i.e. `τ[0]` first).
pub fn eq_evals<E: FieldCore>(tau: &[E]) -> Vec<E> {
    let size = 1usize << tau.len();
    let mut evals = vec![E::zero(); size];
    evals[0] = E::one();
    let mut len = 1usize;
    for &t in tau.iter().rev() {
        let one_minus_t = E::one() - t;
        for j in (0..len).rev() {
            evals[2 * j + 1] = evals[j] * t;
            evals[2 * j] = evals[j] * one_minus_t;
        }
        len *= 2;
    }
    evals
}

/// Evaluate ẽq(τ, r) at a single point.
pub fn eq_eval<E: FieldCore>(tau: &[E], point: &[E]) -> E {
    debug_assert_eq!(tau.len(), point.len());
    tau.iter().zip(point).fold(E::one(), |acc, (&t, &r)| {
        acc * (t * r + (E::one() - t) * (E::one() - r))
    })
}

/// Evaluate the range-check polynomial `w · Π_{k=1}^{b−1} (w − k)(w + k)`.
///
/// This polynomial vanishes exactly when `w ∈ {−(b−1), …, b−1}`.
/// Total degree in `w` is `2b − 1`.
pub fn range_check_eval<E: FieldCore + CanonicalField>(w: E, b: usize) -> E {
    let mut acc = w;
    for k in 1..b {
        let k_e = E::from_u64(k as u64);
        acc = acc * (w - k_e) * (w + k_e);
    }
    acc
}

/// Evaluate a multilinear polynomial (given by boolean-hypercube evaluations in
/// little-endian bit order) at an arbitrary point via iterated folding.
pub fn multilinear_eval<E: FieldCore>(evals: &[E], point: &[E]) -> E {
    let mut current = evals.to_vec();
    for &r in point {
        let half = current.len() / 2;
        let mut next = Vec::with_capacity(half);
        for i in 0..half {
            next.push(current[2 * i] + r * (current[2 * i + 1] - current[2 * i]));
        }
        current = next;
    }
    current[0]
}

/// Fold an evaluation table by binding its first variable to `r`, halving the
/// table size.
fn fold_evals<E: FieldCore>(evals: &[E], r: E) -> Vec<E> {
    let half = evals.len() / 2;
    let mut out = Vec::with_capacity(half);
    for i in 0..half {
        out.push(evals[2 * i] + r * (evals[2 * i + 1] - evals[2 * i]));
    }
    out
}

// ---------------------------------------------------------------------------
// F_0 sumcheck
// ---------------------------------------------------------------------------

/// Prover for `F_{0,τ₀}(x,y) = ẽq(τ₀,(x,y)) · w̃(x,y) · range_check(w̃(x,y), b)`.
///
/// Stores `eq` and `w` evaluation tables separately so the composite can be
/// evaluated at the `2b + 1` points needed per round (degree `2b`).
pub struct F0Prover<E> {
    eq_table: Vec<E>,
    w_table: Vec<E>,
    num_vars: usize,
    b: usize,
}

impl<E: FieldCore + CanonicalField> F0Prover<E> {
    pub fn new(tau: &[E], w_evals: Vec<E>, b: usize) -> Self {
        let num_vars = tau.len();
        assert_eq!(w_evals.len(), 1 << num_vars);
        Self {
            eq_table: eq_evals(tau),
            w_table: w_evals,
            num_vars,
            b,
        }
    }
}

impl<E: FieldCore + CanonicalField> SumcheckInstanceProver<E> for F0Prover<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2 * self.b
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let half = self.eq_table.len() / 2;
        let degree = 2 * self.b;
        let num_points = degree + 1;
        let mut round_evals = vec![E::zero(); num_points];

        for j in 0..half {
            let eq_0 = self.eq_table[2 * j];
            let eq_1 = self.eq_table[2 * j + 1];
            let w_0 = self.w_table[2 * j];
            let w_1 = self.w_table[2 * j + 1];

            for t in 0..num_points {
                let t_e = E::from_u64(t as u64);
                let eq_t = eq_0 + t_e * (eq_1 - eq_0);
                let w_t = w_0 + t_e * (w_1 - w_0);
                round_evals[t] = round_evals[t] + eq_t * range_check_eval(w_t, self.b);
            }
        }

        UniPoly::from_evals(&round_evals)
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        self.eq_table = fold_evals(&self.eq_table, r);
        self.w_table = fold_evals(&self.w_table, r);
    }
}

/// Verifier for `F_{0,τ₀}`.
pub struct F0Verifier<E> {
    tau: Vec<E>,
    w_evals: Vec<E>,
    num_vars: usize,
    b: usize,
}

impl<E: FieldCore + CanonicalField> F0Verifier<E> {
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

impl<E: FieldCore + CanonicalField> SumcheckInstanceVerifier<E> for F0Verifier<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2 * self.b
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn expected_output_claim(&self, challenges: &[E]) -> E {
        let eq_val = eq_eval(&self.tau, challenges);
        let w_val = multilinear_eval(&self.w_evals, challenges);
        eq_val * range_check_eval(w_val, self.b)
    }
}

// ---------------------------------------------------------------------------
// F_α sumcheck
// ---------------------------------------------------------------------------

/// Prover for `F_{α,τ₁}(x,y) = w̃(x,y) · α̃(y) · m(x)`.
///
/// All three constituent evaluation tables are stored at full domain size
/// (`2^{num_u + num_l}`).  `α̃` is replicated along x dimensions and `m` along
/// y dimensions so that a uniform fold-by-pairs works in every round.
///
/// Round polynomial degree is 2 (product of at most two multilinear factors
/// depending on any single variable).
pub struct FAlphaProver<E> {
    w_table: Vec<E>,
    alpha_table: Vec<E>,
    m_table: Vec<E>,
    num_vars: usize,
}

impl<E: FieldCore + CanonicalField> FAlphaProver<E> {
    /// Construct from the three constituent evaluation tables.
    ///
    /// - `w_evals`: evaluations of `w̃` over `{0,1}^{num_u + num_l}` (full domain).
    /// - `alpha_evals_y`: evaluations of `α̃` over `{0,1}^{num_l}` (compact).
    /// - `m_evals_x`: evaluations of `m` over `{0,1}^{num_u}` (compact).
    ///
    /// The constructor extends the compact tables to the full domain by replication.
    pub fn new(
        w_evals: Vec<E>,
        alpha_evals_y: &[E],
        m_evals_x: &[E],
        num_u: usize,
        num_l: usize,
    ) -> Self {
        let num_vars = num_u + num_l;
        let n = 1usize << num_vars;
        assert_eq!(w_evals.len(), n);
        assert_eq!(alpha_evals_y.len(), 1 << num_l);
        assert_eq!(m_evals_x.len(), 1 << num_u);

        let x_mask = (1usize << num_u) - 1;
        let alpha_table: Vec<E> = (0..n).map(|idx| alpha_evals_y[idx >> num_u]).collect();
        let m_table: Vec<E> = (0..n).map(|idx| m_evals_x[idx & x_mask]).collect();

        Self {
            w_table: w_evals,
            alpha_table,
            m_table,
            num_vars,
        }
    }
}

impl<E: FieldCore + CanonicalField> SumcheckInstanceProver<E> for FAlphaProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let half = self.w_table.len() / 2;
        let num_points = 3; // degree 2 → 3 evaluation points
        let mut round_evals = vec![E::zero(); num_points];

        for j in 0..half {
            let w_0 = self.w_table[2 * j];
            let w_1 = self.w_table[2 * j + 1];
            let a_0 = self.alpha_table[2 * j];
            let a_1 = self.alpha_table[2 * j + 1];
            let m_0 = self.m_table[2 * j];
            let m_1 = self.m_table[2 * j + 1];

            for t in 0..num_points {
                let t_e = E::from_u64(t as u64);
                let w_t = w_0 + t_e * (w_1 - w_0);
                let a_t = a_0 + t_e * (a_1 - a_0);
                let m_t = m_0 + t_e * (m_1 - m_0);
                round_evals[t] = round_evals[t] + w_t * a_t * m_t;
            }
        }

        UniPoly::from_evals(&round_evals)
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        self.w_table = fold_evals(&self.w_table, r);
        self.alpha_table = fold_evals(&self.alpha_table, r);
        self.m_table = fold_evals(&self.m_table, r);
    }
}

/// Verifier for `F_{α,τ₁}`.
pub struct FAlphaVerifier<F: FieldCore, const D: usize> {
    w_evals: Vec<F>,
    alpha_evals_y: Vec<F>,
    m_evals_x: Vec<F>,
    tau: Vec<F>,
    v: Vec<CyclotomicRing<F, D>>,
    u: Vec<CyclotomicRing<F, D>>,
    y_ring: CyclotomicRing<F, D>,
    alpha: F,
    num_u: usize,
    num_l: usize,
}

impl<F: FieldCore + CanonicalField, const D: usize> FAlphaVerifier<F, D> {
    pub fn new(
        w_evals: Vec<F>,
        alpha_evals_y: Vec<F>,
        m_evals_x: Vec<F>,
        tau: Vec<F>,
        v: Vec<CyclotomicRing<F, D>>,
        u: Vec<CyclotomicRing<F, D>>,
        y_ring: CyclotomicRing<F, D>,
        alpha: F,
        num_u: usize,
        num_l: usize,
    ) -> Self {
        assert_eq!(w_evals.len(), 1 << (num_u + num_l));
        assert_eq!(alpha_evals_y.len(), 1 << num_l);
        assert_eq!(m_evals_x.len(), 1 << num_u);
        Self {
            w_evals,
            alpha_evals_y,
            m_evals_x,
            tau,
            v,
            u,
            y_ring,
            alpha,
            num_u,
            num_l,
        }
    }
}

impl<F: FieldCore + CanonicalField, const D: usize> SumcheckInstanceVerifier<F>
    for FAlphaVerifier<F, D>
{
    fn num_rounds(&self) -> usize {
        self.num_u + self.num_l
    }

    fn degree_bound(&self) -> usize {
        2
    }

    fn input_claim(&self) -> F {
        let y_a: Vec<F> = self
            .v
            .iter()
            .chain(self.u.iter())
            .chain(std::iter::once(&self.y_ring))
            .map(|r| eval_ring_at(r, &self.alpha))
            .collect();

        let eq_tau = eq_evals(&self.tau);
        let mut acc = F::zero();
        for (i, eq_i) in eq_tau.iter().enumerate() {
            let y_i = if i < y_a.len() { y_a[i] } else { F::zero() };
            acc = acc + *eq_i * y_i;
        }
        acc
    }

    fn expected_output_claim(&self, challenges: &[F]) -> F {
        let (x_challenges, y_challenges) = challenges.split_at(self.num_u);
        let w_val = multilinear_eval(&self.w_evals, challenges);
        let alpha_val = multilinear_eval(&self.alpha_evals_y, y_challenges);
        let m_val = multilinear_eval(&self.m_evals_x, x_challenges);
        w_val * alpha_val * m_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{CyclotomicRing, Fp64};
    use crate::primitives::multilinear_evals::DenseMultilinearEvals;
    use crate::protocol::commitment_scheme::{build_w_coeffs, rederive_alpha_and_m_a};
    use crate::protocol::transcript::labels;
    use crate::protocol::{
        prove_sumcheck, verify_sumcheck, Blake2bTranscript, CommitmentConfig, CommitmentScheme,
        DefaultCommitmentConfig, HachiCommitmentScheme, Transcript,
    };
    use crate::{CanonicalField, FieldCore};

    type F = Fp64<4294967197>;
    const D: usize = 8;

    fn ring_with_seed(seed: u64) -> CyclotomicRing<F, D> {
        let coeffs = std::array::from_fn(|i| F::from_u64(seed + i as u64));
        CyclotomicRing::from_coefficients(coeffs)
    }

    fn ring_with_small_coeff(value: u64) -> CyclotomicRing<F, D> {
        let coeffs = std::array::from_fn(|_| F::from_u64(value));
        CyclotomicRing::from_coefficients(coeffs)
    }

    #[test]
    fn f0_sumcheck_uses_commitment_w_evals() {
        let z = vec![
            ring_with_small_coeff(1),
            ring_with_small_coeff(2),
            ring_with_small_coeff(3),
        ];
        let r = vec![ring_with_small_coeff(0), ring_with_small_coeff(1)];
        let mut w_evals = build_w_coeffs::<F, D, DefaultCommitmentConfig>(&z, &r);

        let target_len = w_evals.len().next_power_of_two();
        w_evals.resize(target_len, F::zero());
        let num_vars = target_len.trailing_zeros() as usize;
        let tau: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let b = 1usize << DefaultCommitmentConfig::LOG_BASIS;

        let eq_table = eq_evals(&tau);
        let claim: F = (0..w_evals.len())
            .map(|i| eq_table[i] * range_check_eval(w_evals[i], b))
            .fold(F::zero(), |a, v| a + v);

        let mut prover = F0Prover::new(&tau, w_evals.clone(), b);
        let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        pt.append_field(labels::ABSORB_SUMCHECK_CLAIM, &claim);
        let (proof, prover_challenges, final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, claim, &mut pt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        let oracle = eq_eval(&tau, &prover_challenges)
            * range_check_eval(multilinear_eval(&w_evals, &prover_challenges), b);
        assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

        let verifier = F0Verifier::new(tau, w_evals, b);
        let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        vt.append_field(labels::ABSORB_SUMCHECK_CLAIM, &claim);
        let verifier_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut vt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }

    #[test]
    fn f0_sumcheck_uses_prove_w_evals() {
        let alpha = DefaultCommitmentConfig::D.trailing_zeros() as usize;
        let num_vars = DefaultCommitmentConfig::R + DefaultCommitmentConfig::M + alpha;
        let len = 1usize << num_vars;
        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals);

        let setup = HachiCommitmentScheme::setup_prover(num_vars);
        let (commitment, hint) = HachiCommitmentScheme::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let proof = HachiCommitmentScheme::prove(
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
        let b = 1usize << DefaultCommitmentConfig::LOG_BASIS;

        let eq_table = eq_evals(&tau);
        let claim: F = (0..w_evals.len())
            .map(|i| eq_table[i] * range_check_eval(w_evals[i], b))
            .fold(F::zero(), |a, v| a + v);
        println!("Claim is: {:?}", claim);
        assert!(claim.is_zero(), "expected F0 claim to be zero");

        let mut prover = F0Prover::new(&tau, w_evals.clone(), b);
        let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        pt.append_field(labels::ABSORB_SUMCHECK_CLAIM, &claim);
        let (proof_sc, prover_challenges, final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, claim, &mut pt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        let oracle = eq_eval(&tau, &prover_challenges)
            * range_check_eval(multilinear_eval(&w_evals, &prover_challenges), b);
        assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

        let verifier = F0Verifier::new(tau, w_evals, b);
        let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        vt.append_field(labels::ABSORB_SUMCHECK_CLAIM, &claim);
        let verifier_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof_sc, &verifier, &mut vt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }

    #[test]
    fn f_alpha_sumcheck_uses_prove_w_evals() {
        let alpha_bits = DefaultCommitmentConfig::D.trailing_zeros() as usize;
        let num_vars = DefaultCommitmentConfig::R + DefaultCommitmentConfig::M + alpha_bits;
        let len = 1usize << num_vars;
        let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
        let poly = DenseMultilinearEvals::new_padded(evals);

        let setup = HachiCommitmentScheme::setup_prover(num_vars);
        let (commitment, hint) = HachiCommitmentScheme::commit(&poly, &setup).unwrap();

        let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
        let mut prover_transcript = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        let proof = HachiCommitmentScheme::prove(
            &setup,
            &poly,
            &opening_point,
            Some(hint),
            &mut prover_transcript,
            &commitment,
        )
        .unwrap();

        let (alpha, m_a_vec) =
            rederive_alpha_and_m_a(&proof, &setup, &opening_point, &commitment).unwrap();

        let d = DefaultCommitmentConfig::D;
        assert_eq!(proof.sumcheck_aux.w.len() % d, 0);
        let w_u = proof.sumcheck_aux.w.len() / d;
        let rows = DefaultCommitmentConfig::N_D
            + DefaultCommitmentConfig::N_B
            + 1
            + 1
            + DefaultCommitmentConfig::N_A;
        assert!(rows > 0);
        assert_eq!(m_a_vec.len() % rows, 0);
        let cols = m_a_vec.len() / rows;
        assert_eq!(w_u, cols);
        assert_eq!(w_u, cols);

        let num_u = cols.next_power_of_two().trailing_zeros() as usize;
        let num_l = alpha_bits;
        let n = 1usize << (num_u + num_l);

        let mut w_evals = vec![F::zero(); n];
        let y_len = 1usize << num_l;
        let x_len = 1usize << num_u;
        for x in 0..x_len {
            for y in 0..y_len {
                let src = y + (x << num_l);
                if src < proof.sumcheck_aux.w.len() {
                    let dst = x + (y << num_u);
                    w_evals[dst] = proof.sumcheck_aux.w[src];
                }
            }
        }

        let num_i = rows.next_power_of_two().trailing_zeros() as usize;
        let tau1: Vec<F> = (0..num_i).map(|i| F::from_u64((i + 5) as u64)).collect();
        let eq_tau1 = eq_evals(&tau1);

        let mut m_evals_x = vec![F::zero(); x_len];
        for x in 0..x_len {
            let mut acc = F::zero();
            for i in 0..(1usize << num_i) {
                let row_val = if i < rows && x < cols {
                    m_a_vec[i * cols + x]
                } else {
                    F::zero()
                };
                acc = acc + eq_tau1[i] * row_val;
            }
            m_evals_x[x] = acc;
        }

        let mut alpha_evals_y = vec![F::zero(); y_len];
        let mut power = F::one();
        for val in alpha_evals_y.iter_mut() {
            *val = power;
            power = power * alpha;
        }

        let x_mask = x_len - 1;
        let alpha_full: Vec<F> = (0..n).map(|idx| alpha_evals_y[idx >> num_u]).collect();
        let m_full: Vec<F> = (0..n).map(|idx| m_evals_x[idx & x_mask]).collect();
        let claim: F = (0..n)
            .map(|i| w_evals[i] * alpha_full[i] * m_full[i])
            .fold(F::zero(), |a, v| a + v);

        let mut prover =
            FAlphaProver::new(w_evals.clone(), &alpha_evals_y, &m_evals_x, num_u, num_l);
        let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        pt.append_field(labels::ABSORB_SUMCHECK_CLAIM, &claim);
        let (proof_sc, prover_challenges, final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut prover, claim, &mut pt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        let (x_ch, y_ch) = prover_challenges.split_at(num_u);
        let oracle = multilinear_eval(&w_evals, &prover_challenges)
            * multilinear_eval(&alpha_evals_y, y_ch)
            * multilinear_eval(&m_evals_x, x_ch);
        assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

        let verifier = FAlphaVerifier::new(
            w_evals,
            alpha_evals_y,
            m_evals_x,
            tau1,
            proof.v.clone(),
            commitment.u.clone(),
            proof.y_ring,
            alpha,
            num_u,
            num_l,
        );
        let verifier_claim = verifier.input_claim();
        let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
        vt.append_field(labels::ABSORB_SUMCHECK_CLAIM, &verifier_claim);
        let verifier_challenges =
            verify_sumcheck::<F, _, F, _, _>(&proof_sc, &verifier, &mut vt, |tr| {
                tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
            })
            .unwrap();

        assert_eq!(prover_challenges, verifier_challenges);
    }
}
