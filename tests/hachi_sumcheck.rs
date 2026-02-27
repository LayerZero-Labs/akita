#![allow(missing_docs)]

use hachi_pcs::algebra::Fp64;
use hachi_pcs::protocol::sumcheck::hachi_sumcheck::{
    eq_eval, eq_evals, multilinear_eval, range_check_eval, F0Prover, F0Verifier, FAlphaProver,
    FAlphaVerifier,
};
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{prove_sumcheck, verify_sumcheck, Blake2bTranscript, Transcript};
use hachi_pcs::{CanonicalField, FieldCore, FieldSampling};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

type F = Fp64<4294967197>;

// ---------------------------------------------------------------------------
// F_0 sumcheck
// ---------------------------------------------------------------------------

fn run_f0_e2e(num_u: usize, num_l: usize, b: usize) {
    let num_vars = num_u + num_l;
    let n = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(0xF0);

    let w_evals: Vec<F> = (0..n).map(|_| F::sample(&mut rng)).collect();
    let tau0: Vec<F> = (0..num_vars).map(|_| F::sample(&mut rng)).collect();

    // Brute-force claim: Σ eq(τ₀, b) · range_check(w(b), b_param)
    let eq_table = eq_evals(&tau0);
    let claim: F = (0..n)
        .map(|i| eq_table[i] * range_check_eval(w_evals[i], b))
        .fold(F::zero(), |a, v| a + v);

    // ---- Prover ----
    let t0 = Instant::now();
    let mut prover = F0Prover::new(&tau0, w_evals.clone(), b);
    let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    pt.append_field(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let (proof, prover_challenges, final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, claim, &mut pt, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();
    let prove_time = t0.elapsed();

    // Sanity: prover's final claim matches oracle evaluation.
    let oracle = eq_eval(&tau0, &prover_challenges)
        * range_check_eval(multilinear_eval(&w_evals, &prover_challenges), b);
    assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

    // ---- Verifier ----
    let t1 = Instant::now();
    let verifier = F0Verifier::new(tau0, w_evals, b, claim);
    let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    vt.append_field(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let verifier_challenges = verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut vt, |tr| {
        tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
    })
    .unwrap();
    let verify_time = t1.elapsed();

    assert_eq!(prover_challenges, verifier_challenges);

    eprintln!(
        "[F0 e2e] num_u={num_u} num_l={num_l} b={b} n=2^{num_vars}={n}  \
         prove={prove_time:.2?}  verify={verify_time:.2?}  \
         rounds={} degree={}",
        proof.round_polys.len(),
        2 * b,
    );
}

#[test]
fn f0_sumcheck_e2e_small() {
    run_f0_e2e(3, 2, 2);
}

#[test]
fn f0_sumcheck_e2e() {
    run_f0_e2e(4, 3, 2);
}

#[test]
fn f0_sumcheck_e2e_larger_b() {
    run_f0_e2e(3, 3, 3);
}

// ---------------------------------------------------------------------------
// F_α sumcheck
// ---------------------------------------------------------------------------

fn run_f_alpha_e2e(num_u: usize, num_l: usize, num_i: usize) {
    let num_vars = num_u + num_l;
    let n = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(0xFA);

    let w_evals: Vec<F> = (0..n).map(|_| F::sample(&mut rng)).collect();
    let alpha_evals_y: Vec<F> = (0..(1 << num_l)).map(|_| F::sample(&mut rng)).collect();
    let m_alpha_evals: Vec<F> = (0..(1usize << (num_i + num_u)))
        .map(|_| F::sample(&mut rng))
        .collect();
    let tau1: Vec<F> = (0..num_i).map(|_| F::sample(&mut rng)).collect();

    // Compute m(x) = Σ_i ẽq(τ₁, i) · M̃_α(i, x)
    let eq_tau1 = eq_evals(&tau1);
    let num_x = 1usize << num_u;
    let m_evals_x: Vec<F> = (0..num_x)
        .map(|x_idx| {
            (0..(1usize << num_i))
                .map(|i_idx| eq_tau1[i_idx] * m_alpha_evals[i_idx * num_x + x_idx])
                .fold(F::zero(), |a, v| a + v)
        })
        .collect();

    // Extend α̃ and m to full domain for brute-force claim computation.
    let x_mask = (1usize << num_u) - 1;
    let alpha_full: Vec<F> = (0..n).map(|idx| alpha_evals_y[idx >> num_u]).collect();
    let m_full: Vec<F> = (0..n).map(|idx| m_evals_x[idx & x_mask]).collect();

    // Brute-force claim: Σ w(idx) · α(y) · m(x)
    let claim: F = (0..n)
        .map(|i| w_evals[i] * alpha_full[i] * m_full[i])
        .fold(F::zero(), |a, v| a + v);

    // ---- Prover ----
    let t0 = Instant::now();
    let mut prover = FAlphaProver::new(w_evals.clone(), &alpha_evals_y, &m_evals_x, num_u, num_l);
    let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    pt.append_field(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let (proof, prover_challenges, final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, claim, &mut pt, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();
    let prove_time = t0.elapsed();

    // Sanity: prover's final claim matches oracle evaluation.
    let (x_ch, y_ch) = prover_challenges.split_at(num_u);
    let oracle = multilinear_eval(&w_evals, &prover_challenges)
        * multilinear_eval(&alpha_evals_y, y_ch)
        * multilinear_eval(&m_evals_x, x_ch);
    assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

    // ---- Verifier ----
    let t1 = Instant::now();
    let verifier = FAlphaVerifier::new(w_evals, alpha_evals_y, m_evals_x, num_u, num_l, claim);
    let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    vt.append_field(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let verifier_challenges = verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut vt, |tr| {
        tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
    })
    .unwrap();
    let verify_time = t1.elapsed();

    assert_eq!(prover_challenges, verifier_challenges);

    eprintln!(
        "[Fα e2e] num_u={num_u} num_l={num_l} num_i={num_i} n=2^{num_vars}={n}  \
         prove={prove_time:.2?}  verify={verify_time:.2?}  \
         rounds={} degree=2",
        proof.round_polys.len(),
    );
}

#[test]
fn f_alpha_sumcheck_e2e_small() {
    run_f_alpha_e2e(3, 2, 2);
}

#[test]
fn f_alpha_sumcheck_e2e() {
    run_f_alpha_e2e(4, 3, 3);
}

#[test]
fn f_alpha_sumcheck_e2e_asymmetric() {
    run_f_alpha_e2e(5, 2, 4);
}

// ---------------------------------------------------------------------------
// UniPoly::from_evals correctness
// ---------------------------------------------------------------------------

#[test]
fn from_evals_matches_direct_polynomial() {
    use hachi_pcs::protocol::UniPoly;

    // Verify that interpolation at integer points reproduces the polynomial.
    let mut rng = StdRng::seed_from_u64(0xEE);

    for degree in 0..6usize {
        let coeffs: Vec<F> = (0..=degree).map(|_| F::sample(&mut rng)).collect();
        let poly = UniPoly::from_coeffs(coeffs);

        let evals: Vec<F> = (0..=degree)
            .map(|t| poly.evaluate(&F::from_u64(t as u64)))
            .collect();
        let reconstructed = UniPoly::from_evals(&evals);

        for x_u64 in [0u64, 1, 2, 3, 7, 13] {
            let x = F::from_u64(x_u64);
            assert_eq!(
                poly.evaluate(&x),
                reconstructed.evaluate(&x),
                "degree {degree}, x={x_u64}"
            );
        }
    }
}
