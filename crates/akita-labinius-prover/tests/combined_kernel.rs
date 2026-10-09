#![cfg(feature = "labinius")]

mod combined_support;

use akita_algebra::{poly::multilinear_eval, Prime64Offset23703, SmoothFftField};
use akita_labinius_prover::combined_kernel::{self, CombinedRootKernel};
use akita_labinius_verifier::{
    channel::{self, RootSumcheckProverChannel, RootSumcheckVerifierChannel},
    root_sumcheck::{combined_terminal, verify_combined_rounds},
};
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::SumcheckInstanceProver;
use combined_support::CombinedRootSumcheck;
use jolt_field::{Field, Prime128OffsetA7F7};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const BASES: [LabiniusDigitBase; 3] = [
    LabiniusDigitBase::Bits1,
    LabiniusDigitBase::Bits2,
    LabiniusDigitBase::Bits4,
];
const INVOCATION: u32 = 37;

struct Fixture<F> {
    digits: Vec<u8>,
    digit_factor: Vec<F>,
    compact: Vec<F>,
    kw: Vec<F>,
    tau: Vec<F>,
    beta: F,
    s: F,
}

fn random<F: Field>(rng: &mut StdRng) -> F {
    F::from_u128(u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
}

fn class_rounds(base: LabiniusDigitBase) -> usize {
    match base {
        LabiniusDigitBase::Bits1 => 4,
        LabiniusDigitBase::Bits2 => 3,
        LabiniusDigitBase::Bits4 => 2,
    }
}

fn fixture<F: Field>(nu: usize, base: LabiniusDigitBase, rng: &mut StdRng) -> Fixture<F> {
    fixture_with_factor(nu, nu.min(class_rounds(base)), base, rng)
}

fn fixture_with_factor<F: Field>(
    nu: usize,
    a: usize,
    base: LabiniusDigitBase,
    rng: &mut StdRng,
) -> Fixture<F> {
    let digits: Vec<_> = (0..1usize << nu)
        .map(|_| (rng.next_u32() % (1 << base.bits())) as u8)
        .collect();
    let digit_factor = (0..1usize << a).map(|_| random(rng)).collect();
    let compact = (0..1usize << (nu - a)).map(|_| random(rng)).collect();
    let mut f = Fixture {
        digits,
        digit_factor,
        compact,
        kw: Vec::new(),
        tau: (0..nu).map(|_| random(rng)).collect(),
        beta: random(rng),
        s: F::zero(),
    };
    refresh_weights(&mut f);
    f
}

fn refresh_weights<F: Field>(f: &mut Fixture<F>) {
    f.kw = f
        .compact
        .iter()
        .flat_map(|&coefficient| f.digit_factor.iter().map(move |&digit| digit * coefficient))
        .collect();
    f.s = f.digits.iter().zip(&f.kw).fold(F::zero(), |sum, (&w, &k)| {
        sum + F::from_u64(u64::from(w)) * k
    });
}

fn manual<F: SmoothFftField + Send + Sync>(
    base: LabiniusDigitBase,
    f: &Fixture<F>,
    challenges: &[F],
) {
    let mut reference =
        CombinedRootSumcheck::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
    let mut kernel = CombinedRootKernel::new(
        base,
        &f.digits,
        f.digit_factor.clone(),
        f.compact.clone(),
        &f.tau,
        f.beta,
        f.s,
    )
    .unwrap();
    assert_eq!(reference.num_rounds(), kernel.num_rounds());
    assert_eq!(reference.degree_bound(), kernel.degree_bound());
    assert_eq!(reference.input_claim(), kernel.input_claim());
    assert_eq!(reference.final_evaluations(), kernel.final_evaluations());
    let mut claim = reference.input_claim();
    for (round, &challenge) in challenges.iter().enumerate() {
        let expected = reference.compute_round_univariate(round, claim);
        let actual = kernel.compute_round_univariate(round, claim);
        assert_eq!(expected, actual, "base={base:?}, round={round}");
        // Neither the public round argument nor a previous claim changes the
        // polynomial of the currently bound state, even for false claims.
        assert_eq!(
            reference.compute_round_univariate(usize::MAX, claim + F::one()),
            kernel.compute_round_univariate(usize::MAX, claim + F::one())
        );
        claim = expected.evaluate(challenge);
        reference.ingest_challenge(round, challenge);
        kernel.ingest_challenge(round, challenge);
        assert_eq!(reference.final_evaluations(), kernel.final_evaluations());
    }
    let w: Vec<_> = f
        .digits
        .iter()
        .map(|&digit| F::from_u64(u64::from(digit)))
        .collect();
    assert_eq!(
        kernel.final_evaluations(),
        Some((
            multilinear_eval(&w, challenges).unwrap(),
            multilinear_eval(&f.kw, challenges).unwrap(),
        ))
    );
    reference.ingest_challenge(usize::MAX, F::one());
    kernel.ingest_challenge(usize::MAX, F::one());
    assert_eq!(reference.final_evaluations(), kernel.final_evaluations());
    assert_eq!(
        reference.compute_round_univariate(usize::MAX, claim),
        kernel.compute_round_univariate(usize::MAX, claim)
    );
}

fn transcripts<F: SmoothFftField + Send + Sync>(
    base: LabiniusDigitBase,
    f: &Fixture<F>,
) -> Vec<u8> {
    let mut reference =
        CombinedRootSumcheck::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
    let mut kernel = CombinedRootKernel::new(
        base,
        &f.digits,
        f.digit_factor.clone(),
        f.compact.clone(),
        &f.tau,
        f.beta,
        f.s,
    )
    .unwrap();
    let mut reference_state = channel::new_prover().unwrap();
    let mut kernel_state = channel::new_prover().unwrap();
    let reference_result = combined_support::prove_combined_rounds(
        &mut reference,
        &mut RootSumcheckProverChannel::new(&mut reference_state),
        INVOCATION,
    )
    .expect("honest reference sumcheck succeeds");
    let kernel_result = combined_kernel::prove_combined_rounds(
        &mut kernel,
        &mut RootSumcheckProverChannel::new(&mut kernel_state),
        INVOCATION,
    )
    .expect("honest kernel sumcheck succeeds");
    assert_eq!(reference_result, kernel_result);
    let reference_bytes = channel::finish_prover(reference_state);
    let kernel_bytes = channel::finish_prover(kernel_state);
    assert_eq!(reference_bytes, kernel_bytes);
    assert_eq!(reference.final_evaluations(), kernel.final_evaluations());
    let (point, claim) = kernel_result;
    let mut verifier = channel::new_verifier(&kernel_bytes).unwrap();
    let replay = verify_combined_rounds(
        &mut RootSumcheckVerifierChannel::new(&mut verifier),
        INVOCATION,
        f.tau.len(),
        base,
        f.beta,
        f.s,
    )
    .unwrap();
    assert_eq!(
        (replay.challenges, replay.output_claim),
        (point.clone(), claim)
    );
    channel::finish_verifier(verifier).unwrap();
    let w: Vec<_> = f
        .digits
        .iter()
        .map(|&digit| F::from_u64(u64::from(digit)))
        .collect();
    let evaluations = (
        multilinear_eval(&w, &point).unwrap(),
        multilinear_eval(&f.kw, &point).unwrap(),
    );
    assert_eq!(kernel.final_evaluations(), Some(evaluations));
    assert_eq!(
        claim,
        combined_terminal(base, &f.tau, &point, f.beta, evaluations.0, evaluations.1).unwrap()
    );
    kernel_bytes
}

fn differential<F: SmoothFftField + Send + Sync>() {
    let mut rng = StdRng::seed_from_u64(0xc0_6b_1e);
    for base in BASES {
        // nu=16 also exercises all 65,536 buckets after the small-input cutoff.
        for nu in (0..=13).chain([16]) {
            let f = fixture::<F>(nu, base, &mut rng);
            let challenges: Vec<_> = (0..nu).map(|_| random(&mut rng)).collect();
            manual(base, &f, &challenges);
            transcripts(base, &f);
        }
    }
}

#[test]
fn random_rounds_transcripts_and_terminal_checks_both_fields() {
    differential::<Prime64Offset23703>();
    differential::<Prime128OffsetA7F7>();
}

fn factor_shapes<F: SmoothFftField + Send + Sync>() {
    let mut rng = StdRng::seed_from_u64(0x000f_ac70);
    for base in BASES {
        for a in 0..=8 {
            let mut f = fixture_with_factor::<F>(8, a, base, &mut rng);
            // Nontrivial digit factors and padded coefficient positions expose
            // both the low-variable tensor order and zero compact entries.
            for coefficient in f.compact.iter_mut().skip(1).step_by(3) {
                *coefficient = F::zero();
            }
            refresh_weights(&mut f);
            let challenges: Vec<_> = (0..8).map(|_| random(&mut rng)).collect();
            manual(base, &f, &challenges);
            transcripts(base, &f);
            f.s += F::one();
            manual(base, &f, &challenges);
        }
    }
}

#[test]
fn digit_factor_shapes_and_padded_compact_entries_match_dense_reference() {
    factor_shapes::<Prime64Offset23703>();
    factor_shapes::<Prime128OffsetA7F7>();
}

fn extremal<F: SmoothFftField + Send + Sync>() {
    let mut rng = StdRng::seed_from_u64(0xed_6e);
    for base in BASES {
        let max = ((1 << base.bits()) - 1) as u8;
        for pattern in 0..3 {
            let mut f = fixture::<F>(13, base, &mut rng);
            for (index, digit) in f.digits.iter_mut().enumerate() {
                *digit = match pattern {
                    0 => 0,
                    1 => max,
                    _ => {
                        if index % 2 == 0 {
                            0
                        } else {
                            max
                        }
                    }
                };
            }
            f.compact.fill(F::zero());
            refresh_weights(&mut f);
            f.tau = (0..13).map(|i| F::from_u64((i % 2) as u64)).collect();
            let challenges: Vec<_> = (0..13)
                .map(|i| match i % 3 {
                    0 => F::zero(),
                    1 => F::one(),
                    _ => random(&mut rng),
                })
                .collect();
            manual(base, &f, &challenges);
            transcripts(base, &f);
            f.beta = F::zero();
            manual(base, &f, &challenges);
        }
    }
}

#[test]
fn extremal_digits_zero_weights_and_boolean_equality_points() {
    extremal::<Prime64Offset23703>();
    extremal::<Prime128OffsetA7F7>();
}

fn false_claim<F: SmoothFftField + Send + Sync>() {
    let mut rng = StdRng::seed_from_u64(0xfa_15e);
    for base in BASES {
        let mut f = fixture::<F>(8, base, &mut rng);
        f.beta = F::from_u64(7);
        let honest_proof = transcripts(base, &f);
        f.s += F::one();
        let challenges: Vec<_> = (0..f.tau.len()).map(|_| random(&mut rng)).collect();
        manual(base, &f, &challenges);
        // The shared engine rejects the inconsistent input claim before
        // writing round zero; both instances must report the same error.
        let mut reference =
            CombinedRootSumcheck::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
        let mut kernel = CombinedRootKernel::new(
            base,
            &f.digits,
            f.digit_factor.clone(),
            f.compact.clone(),
            &f.tau,
            f.beta,
            f.s,
        )
        .unwrap();
        let mut reference_state = channel::new_prover().unwrap();
        let mut kernel_state = channel::new_prover().unwrap();
        let reference_result = combined_support::prove_combined_rounds(
            &mut reference,
            &mut RootSumcheckProverChannel::new(&mut reference_state),
            INVOCATION,
        );
        let kernel_result = combined_kernel::prove_combined_rounds(
            &mut kernel,
            &mut RootSumcheckProverChannel::new(&mut kernel_state),
            INVOCATION,
        );
        assert!(reference_result.is_err());
        assert_eq!(reference_result, kernel_result);
        let rejected_prefix = channel::finish_prover(kernel_state);
        assert_eq!(channel::finish_prover(reference_state), rejected_prefix);
        assert!(rejected_prefix.is_empty());
        assert_eq!(reference.final_evaluations(), kernel.final_evaluations());
        // Both provers produced the same proof bytes above. Compressed
        // round replay may reconstruct a polynomial using the false claim;
        // rejection therefore includes the authenticated terminal obligation.
        let mut state = channel::new_verifier(&honest_proof).unwrap();
        if let Ok(replay) = verify_combined_rounds(
            &mut RootSumcheckVerifierChannel::new(&mut state),
            INVOCATION,
            f.tau.len(),
            base,
            f.beta,
            f.s,
        ) {
            channel::finish_verifier(state).unwrap();
            let w: Vec<_> = f
                .digits
                .iter()
                .map(|&digit| F::from_u64(u64::from(digit)))
                .collect();
            let terminal = combined_terminal(
                base,
                &f.tau,
                &replay.challenges,
                f.beta,
                multilinear_eval(&w, &replay.challenges).unwrap(),
                multilinear_eval(&f.kw, &replay.challenges).unwrap(),
            )
            .unwrap();
            assert_ne!(replay.output_claim, terminal);
        }
    }
}

#[test]
fn false_linear_claim_preserves_polynomials_and_is_rejected() {
    false_claim::<Prime64Offset23703>();
    false_claim::<Prime128OffsetA7F7>();
}

fn rejection<F: SmoothFftField + Send + Sync>() {
    for base in BASES {
        let mut cases = vec![
            (vec![], vec![], vec![]),
            (vec![0; 3], vec![F::one(); 3], vec![F::one()]),
            (vec![0; 2], vec![F::one()], vec![F::one()]),
            (vec![0; 2], vec![F::one(); 3], vec![F::one()]),
            (vec![0; 2], vec![F::one(); 2], vec![]),
            (vec![0; 2], vec![F::one(); 2], vec![F::one(); 2]),
            (
                vec![0],
                vec![F::one()],
                vec![F::one(); usize::BITS as usize],
            ),
        ];
        for invalid in [(1 << base.bits()) as u8, 255] {
            cases.push((vec![0, invalid], vec![F::one(); 2], vec![F::one()]));
            // Geometry validation must take precedence over invalid digits.
            cases.push((vec![invalid; 3], vec![F::one(); 3], vec![F::one()]));
        }
        for (digits, kw, tau) in cases {
            let reference =
                CombinedRootSumcheck::new(base, &digits, kw.clone(), &tau, F::one(), F::zero())
                    .err();
            let kernel = CombinedRootKernel::new(
                base,
                &digits,
                vec![F::one()],
                kw,
                &tau,
                F::one(),
                F::zero(),
            )
            .err();
            assert!(reference.is_some());
            assert_eq!(reference, kernel);
        }
    }
}

#[test]
fn constructor_rejections_match_including_validation_order() {
    rejection::<Prime64Offset23703>();
    rejection::<Prime128OffsetA7F7>();
}

fn factor_rejection<F: SmoothFftField + Send + Sync>() {
    for base in BASES {
        let invalid_geometry = CombinedRootSumcheck::new(
            base,
            &[0; 2],
            vec![F::one()],
            &[F::one()],
            F::one(),
            F::zero(),
        )
        .err();
        assert!(invalid_geometry.is_some());
        for (digit_factor, compact) in [
            (vec![], vec![F::one(); 2]),
            (vec![F::one(); 2], vec![]),
            (vec![F::one(); 3], vec![F::one()]),
            (vec![F::one()], vec![F::one(); 3]),
            (vec![F::one()], vec![F::one()]),
            (vec![F::one(); 2], vec![F::one(); 2]),
            // a=2 exceeds nu=1, even with the smallest nonempty compact table.
            (vec![F::one(); 4], vec![F::one()]),
        ] {
            for digits in [vec![0; 2], vec![255; 2]] {
                assert_eq!(
                    CombinedRootKernel::new(
                        base,
                        &digits,
                        digit_factor.clone(),
                        compact.clone(),
                        &[F::one()],
                        F::one(),
                        F::zero(),
                    )
                    .err(),
                    invalid_geometry,
                    "base={base:?}, digit factor={}, compact={}",
                    digit_factor.len(),
                    compact.len(),
                );
            }
        }
    }
}

#[test]
fn malformed_factor_geometry_precedes_digit_validation() {
    factor_rejection::<Prime64Offset23703>();
    factor_rejection::<Prime128OffsetA7F7>();
}

#[cfg(feature = "parallel")]
fn worker_counts<F: SmoothFftField + Send + Sync>() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let three = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut rng = StdRng::seed_from_u64(0x7a_70);
    for base in BASES {
        for a in [
            0,
            class_rounds(base) - 1,
            class_rounds(base),
            class_rounds(base) + 1,
        ] {
            let f = fixture_with_factor::<F>(13, a, base, &mut rng);
            let challenges: Vec<_> = (0..13).map(|_| random(&mut rng)).collect();
            let serial_bytes = one.install(|| {
                manual(base, &f, &challenges);
                transcripts(base, &f)
            });
            for pool in [&three, &four] {
                let parallel_bytes = pool.install(|| {
                    manual(base, &f, &challenges);
                    transcripts(base, &f)
                });
                assert_eq!(serial_bytes, parallel_bytes);
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn rayon_worker_counts_preserve_rounds_and_transcripts() {
    worker_counts::<Prime64Offset23703>();
    worker_counts::<Prime128OffsetA7F7>();
}
