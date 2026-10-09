#![cfg(feature = "labinius")]

use akita_algebra::{poly::multilinear_eval, Prime64Offset23703, SmoothFftField};
use akita_error::AkitaError;
use akita_labinius_prover::root_sumcheck::{CombinedRootSumcheck, ProductSumcheck};
use akita_labinius_verifier::{
    channel::{self, RootSumcheckProverChannel, RootSumcheckVerifierChannel},
    root_sumcheck::{
        alphabet_polynomial, bind_root_sumcheck_instance, combined_input_claim, combined_shape,
        combined_terminal, product_shape, product_terminal, verify_combined_rounds,
        verify_product_rounds, RootSumcheckInstance,
    },
};
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::{
    prove_sumcheck, verify_sumcheck_rounds, InfallibleSumcheck, SumcheckInstanceProver,
    SumcheckShape,
};
use jolt_field::{CanonicalBytes, Field, One, Prime128OffsetA7F7, Ring, Zero};
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const BASES: [LabiniusDigitBase; 3] = [
    LabiniusDigitBase::Bits1,
    LabiniusDigitBase::Bits2,
    LabiniusDigitBase::Bits4,
];

fn random<F: Field>(rng: &mut StdRng) -> F {
    F::from_u64(rng.next_u64())
}

fn dot<F: Field>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::zero(), |s, (&x, &y)| s + x * y)
}

// Independent Boolean basis expansion: no folding helpers or instance state.
fn mle<F: Field>(table: &[F], point: &[F]) -> F {
    table.iter().enumerate().fold(F::zero(), |sum, (i, &v)| {
        let weight = point.iter().enumerate().fold(F::one(), |w, (j, &r)| {
            w * if i >> j & 1 == 0 { F::one() - r } else { r }
        });
        sum + v * weight
    })
}

fn eq<F: Field>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::one(), |v, (&x, &y)| {
        v * ((F::one() - x) * (F::one() - y) + x * y)
    })
}

fn alphabet<F: Field>(base: LabiniusDigitBase, w: F) -> F {
    (0..1u64 << base.bits()).fold(F::one(), |p, a| p * (w - F::from_u64(a)))
}

struct Fixture<F> {
    digits: Vec<u8>,
    w: Vec<F>,
    kw: Vec<F>,
    tau: Vec<F>,
    beta: F,
    s: F,
}

fn fixture<F: Field>(n: usize, base: LabiniusDigitBase, rng: &mut StdRng) -> Fixture<F> {
    let digits: Vec<_> = (0..1usize << n)
        .map(|_| (rng.next_u32() % (1 << base.bits())) as u8)
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

fn bind_shape<C: channel::ClearChannel>(channel: &mut C, shape: SumcheckShape) {
    let kind = match shape.degree_bound() {
        2 => RootSumcheckInstance::Product,
        3 => RootSumcheckInstance::Combined(LabiniusDigitBase::Bits1),
        5 => RootSumcheckInstance::Combined(LabiniusDigitBase::Bits2),
        17 => RootSumcheckInstance::Combined(LabiniusDigitBase::Bits4),
        _ => panic!("unexpected test shape"),
    };
    bind_root_sumcheck_instance(channel, kind, 7, shape.num_rounds()).unwrap();
}

fn prove<F: SmoothFftField, P: SumcheckInstanceProver<F>>(
    instance: &mut P,
    shape: SumcheckShape,
) -> (Vec<u8>, Vec<F>, F) {
    let mut state = channel::new_prover().unwrap();
    bind_shape(&mut RootSumcheckProverChannel::new(&mut state), shape);
    let (point, claim) = prove_sumcheck::<F, F, _, _>(
        &mut InfallibleSumcheck(instance),
        &mut RootSumcheckProverChannel::new(&mut state),
        shape,
        7,
    )
    .unwrap();
    (channel::finish_prover(state), point, claim)
}

fn replay<F: SmoothFftField>(
    proof: &[u8],
    shape: SumcheckShape,
    claim: F,
) -> Result<(Vec<F>, F), AkitaError> {
    let mut state = channel::new_verifier(proof)?;
    bind_shape(&mut RootSumcheckVerifierChannel::new(&mut state), shape);
    let result = verify_sumcheck_rounds::<F, F, _>(
        &mut RootSumcheckVerifierChannel::new(&mut state),
        7,
        claim,
        shape,
    )?;
    channel::finish_verifier(state)?;
    Ok((result.challenges, result.output_claim))
}

fn accepts<F: SmoothFftField>(proof: &[u8], base: LabiniusDigitBase, f: &Fixture<F>) -> bool {
    let Ok((rho, claim)) = replay(
        proof,
        combined_shape(f.tau.len(), base).unwrap(),
        f.beta * f.s,
    ) else {
        return false;
    };
    claim
        == combined_terminal(
            base,
            &f.tau,
            &rho,
            f.beta,
            multilinear_eval(&f.w, &rho).unwrap(),
            multilinear_eval(&f.kw, &rho).unwrap(),
        )
        .unwrap()
}

fn completeness<F: SmoothFftField>() {
    let mut rng = StdRng::seed_from_u64(0xc0_123);
    for base in BASES {
        for n in [1, 2, 5, 9] {
            let f = fixture::<F>(n, base, &mut rng);
            let mut instance =
                CombinedRootSumcheck::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s)
                    .unwrap();
            assert_eq!(instance.final_evaluations(), None);
            let (proof, rho, claim) = prove(&mut instance, combined_shape(n, base).unwrap());
            let w = multilinear_eval(&f.w, &rho).unwrap();
            let kw = multilinear_eval(&f.kw, &rho).unwrap();
            assert_eq!(instance.final_evaluations(), Some((w, kw)));
            assert_eq!(
                claim,
                combined_terminal(base, &f.tau, &rho, f.beta, w, kw).unwrap()
            );
            assert!(accepts(&proof, base, &f));
            let mut state = channel::new_verifier(&proof).unwrap();
            let result = verify_combined_rounds(
                &mut RootSumcheckVerifierChannel::new(&mut state),
                7,
                n,
                base,
                f.beta,
                f.s,
            )
            .unwrap();
            assert_eq!((result.challenges, result.output_claim), (rho, claim));
            channel::finish_verifier(state).unwrap();
        }
    }
    for n in [1, 6] {
        let y: Vec<F> = (0..1usize << n).map(|_| random(&mut rng)).collect();
        let ky: Vec<F> = (0..y.len()).map(|_| random(&mut rng)).collect();
        let sum = dot(&y, &ky);
        let mut instance = ProductSumcheck::new(y.clone(), ky.clone(), sum).unwrap();
        assert_eq!(instance.final_evaluations(), None);
        let (proof, rho, claim) = prove(&mut instance, product_shape(n).unwrap());
        let y_eval = multilinear_eval(&y, &rho).unwrap();
        let ky_eval = multilinear_eval(&ky, &rho).unwrap();
        assert_eq!(instance.final_evaluations(), Some((y_eval, ky_eval)));
        assert_eq!(claim, product_terminal(y_eval, ky_eval));
        assert_eq!(
            replay(&proof, product_shape(n).unwrap(), sum).unwrap(),
            (rho.clone(), claim)
        );
        let mut state = channel::new_verifier(&proof).unwrap();
        let result =
            verify_product_rounds(&mut RootSumcheckVerifierChannel::new(&mut state), 7, n, sum)
                .unwrap();
        assert_eq!((result.challenges, result.output_claim), (rho, claim));
        channel::finish_verifier(state).unwrap();
    }
}

#[test]
fn real_drivers_completeness() {
    completeness::<Prime64Offset23703>();
    completeness::<Prime128OffsetA7F7>();
}

// Sum the original definition over remaining Boolean coordinates after a prefix.
fn brute_round<F: Field>(base: LabiniusDigitBase, f: &Fixture<F>, prefix: &[F], t: F) -> F {
    let remaining = f.tau.len() - prefix.len() - 1;
    (0..1usize << remaining).fold(F::zero(), |sum, tail| {
        let mut point = prefix.to_vec();
        point.push(t);
        point.extend((0..remaining).map(|i| F::from_u64((tail >> i & 1) as u64)));
        let w = mle(&f.w, &point);
        sum + eq(&f.tau, &point) * alphabet(base, w) + f.beta * w * mle(&f.kw, &point)
    })
}

fn manual<F: SmoothFftField>() {
    let mut rng = StdRng::seed_from_u64(0xfa_754);
    for base in BASES {
        for n in [1, 2, 5, 9] {
            let f = fixture::<F>(n, base, &mut rng);
            let mut instance =
                CombinedRootSumcheck::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s)
                    .unwrap();
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
                        assert_eq!(poly.evaluate(t), brute_round(base, &f, &rho, t));
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
                combined_terminal(base, &f.tau, &rho, f.beta, w, kw).unwrap()
            );
        }
    }
    for n in [1, 6] {
        let y: Vec<F> = (0..1usize << n).map(|_| random(&mut rng)).collect();
        let ky: Vec<F> = (0..y.len()).map(|_| random(&mut rng)).collect();
        let mut claim = dot(&y, &ky);
        let mut instance = ProductSumcheck::new(y.clone(), ky.clone(), claim).unwrap();
        let mut rho = Vec::new();
        for round in 0..n {
            let poly = instance.compute_round_univariate(round, claim);
            assert!(poly.degree() <= 2);
            assert_eq!(poly.evaluate(F::zero()) + poly.evaluate(F::one()), claim);
            let r = random(&mut rng);
            claim = poly.evaluate(r);
            instance.ingest_challenge(round, r);
            rho.push(r);
        }
        let y_eval = multilinear_eval(&y, &rho).unwrap();
        let ky_eval = multilinear_eval(&ky, &rho).unwrap();
        assert_eq!(instance.final_evaluations(), Some((y_eval, ky_eval)));
        assert_eq!(claim, product_terminal(y_eval, ky_eval));
    }
}

#[test]
fn manual_rounds_and_independent_brute_force() {
    manual::<Prime64Offset23703>();
    manual::<Prime128OffsetA7F7>();
}

// Test-only cheating path. It evaluates the original field table directly and
// adjusts the linear coefficient to satisfy each claimed round sum. The generic
// driver therefore emits a proof, whose external terminal evaluation must fail.
struct CheatingCombined<F> {
    base: LabiniusDigitBase,
    f: Fixture<F>,
    prefix: Vec<F>,
}

impl<F: SmoothFftField> SumcheckInstanceProver<F> for CheatingCombined<F> {
    fn num_rounds(&self) -> usize {
        self.f.tau.len()
    }
    fn degree_bound(&self) -> usize {
        (1 << self.base.bits()) + 1
    }
    fn input_claim(&self) -> F {
        self.f.beta * self.f.s
    }
    fn compute_round_univariate(&mut self, _round: usize, claim: F) -> UnivariatePoly<F> {
        let values: Vec<_> = (0..=self.degree_bound())
            .map(|t| brute_round(self.base, &self.f, &self.prefix, F::from_u64(t as u64)))
            .collect();
        let mut coefficients = UnivariatePoly::from_evals(&values).into_coefficients();
        coefficients[1] += claim - values[0] - values[1];
        UnivariatePoly::new(coefficients)
    }
    fn ingest_challenge(&mut self, _round: usize, r: F) {
        self.prefix.push(r);
    }
}

fn alphabet_soundness<F: SmoothFftField>() {
    let mut rng = StdRng::seed_from_u64(0xbad_515);
    for base in BASES {
        for invalid in [1u64 << base.bits(), 255, 1_000_003] {
            let mut f = fixture::<F>(2, base, &mut rng);
            f.w[1] = F::from_u64(invalid);
            f.s = dot(&f.w, &f.kw); // The linear relation is exactly satisfied.
            let z = f.w.iter().enumerate().fold(F::zero(), |sum, (i, &w)| {
                let point: Vec<_> = (0..f.tau.len())
                    .map(|j| F::from_u64((i >> j & 1) as u64))
                    .collect();
                sum + eq(&f.tau, &point) * alphabet(base, w)
            });
            assert_ne!(z, F::zero());
            let actual =
                brute_round(base, &f, &[], F::zero()) + brute_round(base, &f, &[], F::one());
            assert_eq!(actual, z + f.beta * f.s);
            assert_ne!(actual, combined_input_claim(f.beta, f.s));
            let mut cheat = CheatingCombined {
                base,
                f,
                prefix: Vec::new(),
            };
            let (proof, _, _) = prove(&mut cheat, combined_shape(2, base).unwrap());
            assert!(!accepts(&proof, base, &cheat.f));
        }
    }
}

#[test]
fn alphabet_violation_with_exact_linear_relation_fails_terminal() {
    alphabet_soundness::<Prime64Offset23703>();
    alphabet_soundness::<Prime128OffsetA7F7>();
}

fn mutations<F: SmoothFftField>() {
    let mut rng = StdRng::seed_from_u64(0x5ad_444);
    for base in BASES {
        let mut f = fixture::<F>(5, base, &mut rng);
        let mut instance =
            CombinedRootSumcheck::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
        let (proof, rho, claim) = prove(&mut instance, combined_shape(5, base).unwrap());
        let w = multilinear_eval(&f.w, &rho).unwrap();
        let kw = multilinear_eval(&f.kw, &rho).unwrap();
        assert_ne!(w, F::zero());
        assert_ne!(f.beta, F::zero());
        assert_ne!(
            combined_terminal(base, &f.tau, &rho, f.beta, w + F::one(), kw).unwrap(),
            claim
        );
        assert_ne!(
            combined_terminal(base, &f.tau, &rho, f.beta, w, kw + F::one()).unwrap(),
            claim
        );
        assert_ne!(
            combined_terminal(base, &f.tau, &rho, f.beta + F::one(), w, kw).unwrap(),
            claim
        );
        f.tau[0] += F::one();
        assert_ne!(
            combined_terminal(base, &f.tau, &rho, f.beta, w, kw).unwrap(),
            claim
        );
        f.tau[0] -= F::one();
        f.kw[0] += F::one();
        assert!(!accepts(&proof, base, &f));
        f.kw[0] -= F::one();
        f.s += F::one();
        assert!(!accepts(&proof, base, &f));
        let mut wrong =
            CombinedRootSumcheck::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
        let mut state = channel::new_prover().unwrap();
        assert!(matches!(
            prove_sumcheck::<F, F, _, _>(
                &mut InfallibleSumcheck(&mut wrong),
                &mut RootSumcheckProverChannel::new(&mut state),
                combined_shape(5, base).unwrap(),
                7,
            ),
            Err(AkitaError::InvalidInput(_))
        ));
        // Also force a malicious proof with the incorrect relation claim.
        let mut cheat = CheatingCombined {
            base,
            f,
            prefix: Vec::new(),
        };
        let (bad_relation, _, _) = prove(&mut cheat, combined_shape(5, base).unwrap());
        assert!(!accepts(&bad_relation, base, &cheat.f));
        cheat.f.s -= F::one();
        let mut tampered = proof;
        tampered[0] ^= 1;
        assert!(!accepts(&tampered, base, &cheat.f));
    }
}

#[test]
fn relation_and_terminal_mutations_are_rejected() {
    mutations::<Prime64Offset23703>();
    mutations::<Prime128OffsetA7F7>();
}

struct ExcessDegree<F> {
    claim: F,
    degree: usize,
}
impl<F: SmoothFftField> SumcheckInstanceProver<F> for ExcessDegree<F> {
    fn num_rounds(&self) -> usize {
        1
    }
    fn degree_bound(&self) -> usize {
        self.degree
    }
    fn input_claim(&self) -> F {
        self.claim
    }
    fn compute_round_univariate(&mut self, _round: usize, _claim: F) -> UnivariatePoly<F> {
        let mut coefficients = vec![F::zero(); self.degree + 2];
        coefficients[1] = self.claim - F::one();
        coefficients[self.degree + 1] = F::one();
        UnivariatePoly::new(coefficients)
    }
    fn ingest_challenge(&mut self, _round: usize, _r: F) {}
}

#[test]
fn shapes_and_excess_degree_messages() {
    type F = Prime128OffsetA7F7;
    for (base, degree) in BASES.into_iter().zip([3, 5, 17]) {
        let shape = combined_shape(1, base).unwrap();
        assert_eq!(shape.num_rounds(), 1);
        assert_eq!(shape.degree_bound(), degree);
        let mut instance = ExcessDegree {
            claim: F::from_u64(3),
            degree,
        };
        let mut state = channel::new_prover().unwrap();
        assert!(matches!(
            prove_sumcheck::<F, F, _, _>(
                &mut InfallibleSumcheck(&mut instance),
                &mut RootSumcheckProverChannel::new(&mut state),
                shape,
                7,
            ),
            Err(AkitaError::Internal(_))
        ));
    }
    assert_eq!(product_shape(6).unwrap().degree_bound(), 2);
    let mut instance = ExcessDegree {
        claim: F::from_u64(3),
        degree: 2,
    };
    let mut state = channel::new_prover().unwrap();
    assert!(matches!(
        prove_sumcheck::<F, F, _, _>(
            &mut InfallibleSumcheck(&mut instance),
            &mut RootSumcheckProverChannel::new(&mut state),
            product_shape(1).unwrap(),
            7,
        ),
        Err(AkitaError::Internal(_))
    ));
    assert!(matches!(
        combined_shape(usize::BITS as usize, BASES[0]),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        product_shape(usize::MAX),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn truncated_and_extra_round_coefficients_are_rejected() {
    type F = Prime128OffsetA7F7;
    let mut instance =
        ProductSumcheck::new(vec![F::one(); 2], vec![F::one(); 2], F::from_u64(2)).unwrap();
    let (proof, _, _) = prove(&mut instance, product_shape(1).unwrap());
    assert_eq!(proof.len(), 2 * F::NUM_BYTES);
    assert!(matches!(
        replay::<F>(
            &proof[..proof.len() - 1],
            product_shape(1).unwrap(),
            F::from_u64(2)
        ),
        Err(AkitaError::InvalidProof)
    ));
    let mut extra = proof;
    extra.extend(vec![0; F::NUM_BYTES]);
    assert!(matches!(
        replay::<F>(&extra, product_shape(1).unwrap(), F::from_u64(2)),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn constructor_and_terminal_rejections() {
    type F = Prime128OffsetA7F7;
    let invalid = |result: Result<CombinedRootSumcheck<F>, AkitaError>| {
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    };
    for base in BASES {
        invalid(CombinedRootSumcheck::new(
            base,
            &[],
            vec![],
            &[],
            F::one(),
            F::zero(),
        ));
        invalid(CombinedRootSumcheck::new(
            base,
            &[0, 0, 0],
            vec![F::one(); 3],
            &[F::one()],
            F::one(),
            F::zero(),
        ));
        invalid(CombinedRootSumcheck::new(
            base,
            &[0, 1],
            vec![F::one()],
            &[F::one()],
            F::one(),
            F::zero(),
        ));
        invalid(CombinedRootSumcheck::new(
            base,
            &[0, 1],
            vec![F::one(); 2],
            &[],
            F::one(),
            F::zero(),
        ));
        invalid(CombinedRootSumcheck::new(
            base,
            &[0, 1 << base.bits()],
            vec![F::one(); 2],
            &[F::one()],
            F::one(),
            F::zero(),
        ));
        invalid(CombinedRootSumcheck::new(
            base,
            &[0, 255],
            vec![F::one(); 2],
            &[F::one()],
            F::one(),
            F::zero(),
        ));
        assert!(matches!(
            combined_terminal(base, &[F::one()], &[], F::one(), F::one(), F::one()),
            Err(AkitaError::InvalidInput(_))
        ));
    }
    for (y, ky) in [
        (vec![], vec![]),
        (vec![F::one(); 3], vec![F::one(); 3]),
        (vec![F::one(); 2], vec![F::one(); 4]),
    ] {
        assert!(matches!(
            ProductSumcheck::new(y, ky, F::zero()),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}

#[test]
fn alphabet_polynomial_has_exact_digit_roots() {
    fn check<F: SmoothFftField>() {
        for base in BASES {
            for digit in 0..1u64 << base.bits() {
                assert_eq!(alphabet_polynomial(base, F::from_u64(digit)), F::zero());
            }
            for invalid in [1u64 << base.bits(), 255, 1_000_003] {
                let w = F::from_u64(invalid);
                assert_ne!(alphabet_polynomial(base, w), F::zero());
                assert_eq!(alphabet_polynomial(base, w), alphabet(base, w));
            }
        }
    }
    check::<Prime64Offset23703>();
    check::<Prime128OffsetA7F7>();
}

#[test]
fn zero_variable_instances() {
    type F = Prime128OffsetA7F7;
    for base in BASES {
        let mut instance = CombinedRootSumcheck::new(
            base,
            &[1],
            vec![F::from_u64(7)],
            &[],
            F::from_u64(5),
            F::from_u64(7),
        )
        .unwrap();
        let (proof, rho, claim) = prove(&mut instance, combined_shape(0, base).unwrap());
        assert!(rho.is_empty());
        assert_eq!(
            instance.final_evaluations(),
            Some((F::one(), F::from_u64(7)))
        );
        assert_eq!(
            claim,
            combined_terminal(base, &[], &[], F::from_u64(5), F::one(), F::from_u64(7)).unwrap()
        );
        assert_eq!(
            replay::<F>(&proof, combined_shape(0, base).unwrap(), claim).unwrap(),
            (rho, claim)
        );
    }
    let mut instance =
        ProductSumcheck::new(vec![F::from_u64(3)], vec![F::from_u64(7)], F::from_u64(21)).unwrap();
    let (proof, rho, claim) = prove(&mut instance, product_shape(0).unwrap());
    assert_eq!(
        instance.final_evaluations(),
        Some((F::from_u64(3), F::from_u64(7)))
    );
    assert_eq!(claim, product_terminal(F::from_u64(3), F::from_u64(7)));
    assert_eq!(
        replay::<F>(&proof, product_shape(0).unwrap(), claim).unwrap(),
        (rho, claim)
    );
}
