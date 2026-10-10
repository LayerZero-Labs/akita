#![cfg(feature = "labinius")]

mod common;

use akita_algebra::poly::multilinear_eval;
use akita_labinius_prover::root_sumcheck::{prove_product_rounds, ProductSumcheck};
use akita_labinius_verifier::{
    channel::{self, RootChallengeChannel, RootSumcheckProverChannel, RootSumcheckVerifierChannel},
    grinding::RootGrindingPlan,
    root_sumcheck::{
        bind_root_sumcheck_instance, combined_input_claim, combined_shape, combined_terminal,
        product_shape, product_terminal, verify_combined_rounds, verify_product_rounds,
        RootSumcheckInstance,
    },
};
use akita_sumcheck::{prove_sumcheck, InfallibleSumcheck, SumcheckInstanceProver};
use common::combined::CombinedRootSumcheck;
use jolt_field::{One, Prime128Offset275, Ring, Zero};
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use rand::{rngs::StdRng, RngCore, SeedableRng};

type F = Prime128Offset275;
const INVOCATION: u32 = 7;
/// Stored digits are base 16.
const ALPHABET: u64 = 16;

fn random(rng: &mut StdRng) -> F {
    F::from_u64(rng.next_u64())
}

fn dot(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::zero(), |s, (&x, &y)| s + x * y)
}

// Independent Boolean basis expansion: no folding helpers or instance state.
fn mle(table: &[F], point: &[F]) -> F {
    table.iter().enumerate().fold(F::zero(), |sum, (i, &v)| {
        let weight = point.iter().enumerate().fold(F::one(), |w, (j, &r)| {
            w * if i >> j & 1 == 0 { F::one() - r } else { r }
        });
        sum + v * weight
    })
}

fn eq(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::one(), |v, (&x, &y)| {
        v * ((F::one() - x) * (F::one() - y) + x * y)
    })
}

fn alphabet(w: F) -> F {
    (0..ALPHABET).fold(F::one(), |p, a| p * (w - F::from_u64(a)))
}

struct Fixture {
    digits: Vec<u8>,
    w: Vec<F>,
    kw: Vec<F>,
    tau: Vec<F>,
    beta: F,
    s: F,
}

fn fixture(n: usize, rng: &mut StdRng) -> Fixture {
    let digits: Vec<_> = (0..1usize << n)
        .map(|_| (rng.next_u64() % ALPHABET) as u8)
        .collect();
    let w: Vec<_> = digits.iter().map(|&x| F::from_u64(u64::from(x))).collect();
    let kw: Vec<_> = (0..w.len()).map(|_| random(rng)).collect();
    Fixture {
        s: dot(&w, &kw),
        digits,
        w,
        kw,
        tau: (0..n).map(|_| random(rng)).collect(),
        beta: random(rng),
    }
}

// Sum the original definition over remaining Boolean coordinates after a prefix.
fn brute_round(f: &Fixture, prefix: &[F], t: F) -> F {
    let remaining = f.tau.len() - prefix.len() - 1;
    (0..1usize << remaining).fold(F::zero(), |sum, tail| {
        let mut point = prefix.to_vec();
        point.push(t);
        point.extend((0..remaining).map(|i| F::from_u64((tail >> i & 1) as u64)));
        let w = mle(&f.w, &point);
        sum + eq(&f.tau, &point) * alphabet(w) + f.beta * w * mle(&f.kw, &point)
    })
}

/// The dense reference of the combined instance agrees with the defining sums
/// round by round, and with the verifier's terminal. The product kernel's
/// proof replays through the verifier to the defining terminal.
#[test]
fn rounds_match_the_brute_force_definition() {
    let mut rng = StdRng::seed_from_u64(0xfa_754);
    for n in [1, 2, 5, 9] {
        let f = fixture(n, &mut rng);
        let mut instance =
            CombinedRootSumcheck::new(&f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
        let mut claim = instance.input_claim();
        let mut rho = Vec::new();
        for round in 0..n {
            let poly = instance.compute_round_univariate(round, claim);
            assert!(poly.degree() <= instance.degree_bound());
            assert_eq!(poly.evaluate(F::zero()) + poly.evaluate(F::one()), claim);
            if n <= 5 {
                for t in [
                    F::zero(),
                    F::one(),
                    F::from_u64(2),
                    F::from_u64(19),
                    random(&mut rng),
                ] {
                    assert_eq!(poly.evaluate(t), brute_round(&f, &rho, t));
                }
            }
            let r = random(&mut rng);
            claim = poly.evaluate(r);
            instance.ingest_challenge(round, r);
            rho.push(r);
        }
        let w = multilinear_eval(&f.w, &rho).unwrap();
        let kw = multilinear_eval(&f.kw, &rho).unwrap();
        assert_eq!(instance.final_evaluations(), Some((w, kw)));
        assert_eq!(
            claim,
            combined_terminal(&f.tau, &rho, f.beta, w, kw).unwrap()
        );
    }
    for n in [1, 6] {
        let y: Vec<F> = (0..1usize << n).map(|_| random(&mut rng)).collect();
        let ky: Vec<F> = (0..y.len()).map(|_| random(&mut rng)).collect();
        let claim = dot(&y, &ky);
        let mut instance = ProductSumcheck::new(y.clone(), ky.clone(), claim).unwrap();
        let plan =
            RootGrindingPlan::sumcheck_rounds::<F, F>(&[(INVOCATION, product_shape(n).unwrap())])
                .unwrap();
        let mut state = channel::new_prover().unwrap();
        let mut prover = RootSumcheckProverChannel::<F>::new(&mut state);
        RootChallengeChannel::<F>::schedule(&mut prover, plan.clone()).unwrap();
        let (rho, _) =
            prove_product_rounds::<F, F, _>(&mut instance, &mut prover, INVOCATION).unwrap();
        RootChallengeChannel::<F>::finish_schedule(&mut prover).unwrap();
        let proof = channel::finish_prover(state);
        let mut state = channel::new_verifier(&proof).unwrap();
        let mut verifier = RootSumcheckVerifierChannel::<F>::new(&mut state);
        RootChallengeChannel::<F>::schedule(&mut verifier, plan).unwrap();
        let replay = verify_product_rounds::<F, F, _>(&mut verifier, INVOCATION, n, claim).unwrap();
        RootChallengeChannel::<F>::finish_schedule(&mut verifier).unwrap();
        channel::finish_verifier(state).unwrap();
        assert_eq!(replay.challenges, rho);
        let y_eval = multilinear_eval(&y, &rho).unwrap();
        let ky_eval = multilinear_eval(&ky, &rho).unwrap();
        assert_eq!(instance.final_evaluations(), Some((y_eval, ky_eval)));
        assert_eq!(replay.output_claim, product_terminal(y_eval, ky_eval));
    }
}

#[test]
fn unscheduled_adapters_reject_the_first_round_challenge() {
    let mut state = channel::new_prover().unwrap();
    let mut prover = RootSumcheckProverChannel::<F>::new(&mut state);
    assert!(akita_sumcheck::SumcheckProverChannel::<F>::round_challenge(
        &mut prover,
        INVOCATION,
        0,
    )
    .is_err());

    let mut state = channel::new_verifier(&[]).unwrap();
    let mut verifier = RootSumcheckVerifierChannel::<F>::new(&mut state);
    assert!(akita_sumcheck::SumcheckVerifierChannel::<F>::round_challenge(
        &mut verifier,
        INVOCATION,
        0,
    )
    .is_err());
}

// Test-only cheating path. It evaluates the original field table directly and
// adjusts the linear coefficient to satisfy each claimed round sum. The generic
// driver therefore emits a proof, whose external terminal evaluation must fail.
struct CheatingCombined {
    f: Fixture,
    prefix: Vec<F>,
}

impl SumcheckInstanceProver<F> for CheatingCombined {
    fn num_rounds(&self) -> usize {
        self.f.tau.len()
    }
    fn degree_bound(&self) -> usize {
        ALPHABET as usize + 1
    }
    fn input_claim(&self) -> F {
        self.f.beta * self.f.s
    }
    fn compute_round_univariate(&mut self, _round: usize, claim: F) -> UnivariatePoly<F> {
        let values: Vec<_> = (0..=self.degree_bound())
            .map(|t| brute_round(&self.f, &self.prefix, F::from_u64(t as u64)))
            .collect();
        let mut coefficients = UnivariatePoly::from_evals(&values).into_coefficients();
        coefficients[1] += claim - values[0] - values[1];
        UnivariatePoly::new(coefficients)
    }
    fn ingest_challenge(&mut self, _round: usize, r: F) {
        self.prefix.push(r);
    }
}

/// A response table with one entry outside the digit alphabet is rejected by
/// the verifier even when the linear relation holds exactly: the round replay
/// succeeds and the terminal check against the table evaluation fails. The
/// alphabet is the only range the verifier enforces on the response.
#[test]
fn a_digit_outside_the_alphabet_fails_the_terminal_check() {
    let mut rng = StdRng::seed_from_u64(0xbad_515);
    for invalid in [ALPHABET, 255, 1_000_003] {
        let mut f = fixture(2, &mut rng);
        f.w[1] = F::from_u64(invalid);
        f.s = dot(&f.w, &f.kw); // The linear relation is exactly satisfied.
        let z = f.w.iter().enumerate().fold(F::zero(), |sum, (i, &w)| {
            let point: Vec<_> = (0..f.tau.len())
                .map(|j| F::from_u64((i >> j & 1) as u64))
                .collect();
            sum + eq(&f.tau, &point) * alphabet(w)
        });
        assert_ne!(z, F::zero());
        let actual = brute_round(&f, &[], F::zero()) + brute_round(&f, &[], F::one());
        assert_eq!(actual, z + f.beta * f.s);
        assert_ne!(actual, combined_input_claim(f.beta, f.s));
        let shape = combined_shape(2).unwrap();
        let plan = RootGrindingPlan::sumcheck_rounds::<F, F>(&[(INVOCATION, shape)]).unwrap();
        let mut cheat = CheatingCombined {
            f,
            prefix: Vec::new(),
        };
        let mut state = channel::new_prover().unwrap();
        let mut prover = RootSumcheckProverChannel::<F>::new(&mut state);
        RootChallengeChannel::<F>::schedule(&mut prover, plan.clone()).unwrap();
        bind_root_sumcheck_instance(&mut prover, RootSumcheckInstance::Combined, INVOCATION, 2)
            .unwrap();
        prove_sumcheck::<F, F, _, _>(
            &mut InfallibleSumcheck(&mut cheat),
            &mut prover,
            shape,
            INVOCATION,
        )
        .unwrap();
        RootChallengeChannel::<F>::finish_schedule(&mut prover).unwrap();
        let proof = channel::finish_prover(state);
        let f = &cheat.f;
        let mut state = channel::new_verifier(&proof).unwrap();
        let mut verifier = RootSumcheckVerifierChannel::<F>::new(&mut state);
        RootChallengeChannel::<F>::schedule(&mut verifier, plan).unwrap();
        let replay =
            verify_combined_rounds::<F, F, _>(&mut verifier, INVOCATION, 2, f.beta, f.s).unwrap();
        RootChallengeChannel::<F>::finish_schedule(&mut verifier).unwrap();
        channel::finish_verifier(state).unwrap();
        let rho = &replay.challenges;
        assert_ne!(
            replay.output_claim,
            combined_terminal(
                &f.tau,
                rho,
                f.beta,
                multilinear_eval(&f.w, rho).unwrap(),
                multilinear_eval(&f.kw, rho).unwrap(),
            )
            .unwrap()
        );
    }
}
