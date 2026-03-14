#![allow(missing_docs)]

use hachi_pcs::algebra::ring::CyclotomicRing;
use hachi_pcs::algebra::Fp64;
use hachi_pcs::protocol::sumcheck::eq_poly::EqPolynomial;
use hachi_pcs::protocol::sumcheck::norm_sumcheck::{NormSumcheckProver, NormSumcheckVerifier};
use hachi_pcs::protocol::sumcheck::relation_sumcheck::{
    RelationSumcheckProver, RelationSumcheckVerifier,
};
use hachi_pcs::protocol::sumcheck::{multilinear_eval, range_check_eval};
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{prove_sumcheck, verify_sumcheck, Blake2bTranscript, Transcript};
use hachi_pcs::{FieldCore, FieldSampling, FromSmallInt};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::Mutex;
use std::time::Instant;

type F = Fp64<4294967197>;

static NORM_KERNEL_ENV_LOCK: Mutex<()> = Mutex::new(());

fn with_norm_kernel_override<T>(value: &str, f: impl FnOnce() -> T) -> T {
    let _guard = NORM_KERNEL_ENV_LOCK.lock().unwrap();
    let old = std::env::var("HACHI_NORM_KERNEL").ok();
    std::env::set_var("HACHI_NORM_KERNEL", value);
    let result = f();
    match old {
        Some(old_value) => std::env::set_var("HACHI_NORM_KERNEL", old_value),
        None => std::env::remove_var("HACHI_NORM_KERNEL"),
    }
    result
}

fn run_f0_e2e(num_u: usize, num_l: usize, b: usize) {
    let num_vars = num_u + num_l;
    let n = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(0xF0);

    let half = (b / 2) as i64;
    let w_evals: Vec<F> = (0..n)
        .map(|i| F::from_i64((i as i64 % b as i64) - half))
        .collect();
    let tau0: Vec<F> = (0..num_vars).map(|_| F::sample(&mut rng)).collect();

    let t0 = Instant::now();
    let mut prover = NormSumcheckProver::new(&tau0, w_evals.clone(), b);
    let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    let (proof, prover_challenges, final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut pt, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();
    let prove_time = t0.elapsed();

    // Sanity: prover's final claim matches oracle evaluation.
    let oracle = EqPolynomial::mle(&tau0, &prover_challenges)
        * range_check_eval(multilinear_eval(&w_evals, &prover_challenges).unwrap(), b);
    assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

    let t1 = Instant::now();
    let verifier = NormSumcheckVerifier::new(tau0, w_evals, b);
    let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    let verifier_challenges = verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut vt, |tr| {
        tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
    })
    .unwrap();
    let verify_time = t1.elapsed();

    assert_eq!(prover_challenges, verifier_challenges);

    tracing::info!(
        num_u,
        num_l,
        b,
        n,
        prove_ms = prove_time.as_millis(),
        verify_ms = verify_time.as_millis(),
        rounds = proof.round_polys.len(),
        degree = b + 1,
        "F0 e2e"
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
    run_f0_e2e(3, 3, 4);
}

#[test]
fn f0_sumcheck_e2e_forced_point_eval_kernel() {
    with_norm_kernel_override("point_eval", || run_f0_e2e(3, 3, 8));
}

#[test]
fn f0_sumcheck_e2e_forced_affine_coeff_kernel() {
    with_norm_kernel_override("affine_coeff", || run_f0_e2e(3, 3, 8));
}

fn run_f_alpha_e2e<const D: usize>(num_u: usize, num_i: usize) {
    let num_l = D.trailing_zeros() as usize;
    let num_vars = num_u + num_l;
    let n = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(0xFA);

    let w_evals: Vec<F> = (0..n).map(|_| F::sample(&mut rng)).collect();
    let alpha_evals_y: Vec<F> = (0..D).map(|_| F::sample(&mut rng)).collect();
    let m_alpha_evals: Vec<F> = (0..(1usize << (num_i + num_u)))
        .map(|_| F::sample(&mut rng))
        .collect();
    let tau1: Vec<F> = (0..num_i).map(|_| F::sample(&mut rng)).collect();

    // Compute m(x) = Σ_i ẽq(τ₁, i) · M̃_α(i, x)
    let eq_tau1 = EqPolynomial::evals(&tau1);
    let num_x = 1usize << num_u;
    let m_evals_x: Vec<F> = (0..num_x)
        .map(|x_idx| {
            (0..(1usize << num_i))
                .map(|i_idx| eq_tau1[i_idx] * m_alpha_evals[i_idx * num_x + x_idx])
                .fold(F::zero(), |a, v| a + v)
        })
        .collect();

    // Compute y_a[i] = Σ_x M̃_α(i,x) · w_α(x), where w_α(x) = Σ_y w(x,y) · α̃(y)
    let num_y = D;
    let num_rows = 1usize << num_i;
    let w_alpha: Vec<F> = (0..num_x)
        .map(|x| {
            (0..num_y)
                .map(|y| w_evals[x + y * num_x] * alpha_evals_y[y])
                .fold(F::zero(), |a, v| a + v)
        })
        .collect();
    let y_a: Vec<F> = (0..num_rows)
        .map(|i| {
            (0..num_x)
                .map(|x| m_alpha_evals[i * num_x + x] * w_alpha[x])
                .fold(F::zero(), |a, v| a + v)
        })
        .collect();

    // Embed y_a values as constant ring elements for the verifier.
    let v_rings: Vec<CyclotomicRing<F, D>> = y_a
        .iter()
        .map(|&val| {
            let mut coeffs = [F::zero(); D];
            coeffs[0] = val;
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    let u_rings: Vec<CyclotomicRing<F, D>> = vec![];
    let u_eval_ring = CyclotomicRing::<F, D>::zero();
    let ring_alpha = F::one();

    let t0 = Instant::now();
    let mut prover =
        RelationSumcheckProver::new(w_evals.clone(), &alpha_evals_y, &m_evals_x, num_u, num_l);
    let mut pt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    let (proof, prover_challenges, final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut pt, |tr| {
            tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();
    let prove_time = t0.elapsed();

    // Sanity: prover's final claim matches oracle evaluation.
    let (x_ch, y_ch) = prover_challenges.split_at(num_u);
    let oracle = multilinear_eval(&w_evals, &prover_challenges).unwrap()
        * multilinear_eval(&alpha_evals_y, y_ch).unwrap()
        * multilinear_eval(&m_evals_x, x_ch).unwrap();
    assert_eq!(final_claim, oracle, "prover final claim != oracle eval");

    let t1 = Instant::now();
    let verifier = RelationSumcheckVerifier::<F, D>::new(
        w_evals,
        alpha_evals_y,
        m_evals_x,
        tau1,
        v_rings,
        u_rings,
        u_eval_ring,
        ring_alpha,
        num_u,
        num_l,
    );
    let mut vt = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    let verifier_challenges = verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut vt, |tr| {
        tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
    })
    .unwrap();
    let verify_time = t1.elapsed();

    assert_eq!(prover_challenges, verifier_challenges);

    tracing::info!(
        num_u,
        num_l,
        num_i,
        n,
        prove_ms = prove_time.as_millis(),
        verify_ms = verify_time.as_millis(),
        rounds = proof.round_polys.len(),
        degree = 2,
        "Fα e2e"
    );
}

#[test]
fn f_alpha_sumcheck_e2e_small() {
    run_f_alpha_e2e::<4>(3, 2);
}

#[test]
fn f_alpha_sumcheck_e2e() {
    run_f_alpha_e2e::<8>(4, 3);
}

#[test]
fn f_alpha_sumcheck_e2e_asymmetric() {
    run_f_alpha_e2e::<4>(5, 4);
}

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
