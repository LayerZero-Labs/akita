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
    kw: Vec<F>,
    tau: Vec<F>,
    beta: F,
    s: F,
}

fn random<F: Field>(rng: &mut StdRng) -> F {
    F::from_u128(u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
}

fn fixture<F: Field>(nu: usize, base: LabiniusDigitBase, rng: &mut StdRng) -> Fixture<F> {
    let digits: Vec<_> = (0..1usize << nu)
        .map(|_| (rng.next_u32() % (1 << base.bits())) as u8)
        .collect();
    let kw: Vec<F> = digits.iter().map(|_| random(rng)).collect();
    let s = digits.iter().zip(&kw).fold(F::zero(), |sum, (&w, &k)| {
        sum + F::from_u64(u64::from(w)) * k
    });
    Fixture {
        digits,
        kw,
        tau: (0..nu).map(|_| random(rng)).collect(),
        beta: random(rng),
        s,
    }
}

fn manual<F: SmoothFftField + Send + Sync>(
    base: LabiniusDigitBase,
    f: &Fixture<F>,
    challenges: &[F],
) {
    let mut reference =
        CombinedRootSumcheck::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
    let mut kernel =
        CombinedRootKernel::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
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
    let mut kernel =
        CombinedRootKernel::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
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
        // nu=16 also exercises larger packed classes through the direct path.
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
            f.kw.fill(F::zero());
            f.s = F::zero();
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
        let mut kernel =
            CombinedRootKernel::new(base, &f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
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
            let kernel =
                CombinedRootKernel::new(base, &digits, kw, &tau, F::one(), F::zero()).err();
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

#[cfg(feature = "parallel")]
fn worker_counts<F: SmoothFftField + Send + Sync>(pools: &[rayon::ThreadPool]) {
    let mut rng = StdRng::seed_from_u64(0x7a_70);
    for base in BASES {
        // nu=13 has 64 outer blocks, so all 64 workers are reserved; nu=8
        // also covers bases with no admitted buckets in the larger pools.
        for nu in [8, 13] {
            let f = fixture::<F>(nu, base, &mut rng);
            let challenges: Vec<_> = (0..nu).map(|_| random(&mut rng)).collect();
            let mut serial_bytes = None;
            for pool in pools {
                let bytes = pool.install(|| {
                    manual(base, &f, &challenges);
                    transcripts(base, &f)
                });
                if let Some(expected) = &serial_bytes {
                    assert_eq!(expected, &bytes);
                } else {
                    serial_bytes = Some(bytes);
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn rayon_worker_counts_preserve_rounds_and_transcripts() {
    let pools: Vec<_> = [1, 4, 64]
        .into_iter()
        .map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        })
        .collect();
    worker_counts::<Prime64Offset23703>(&pools);
    worker_counts::<Prime128OffsetA7F7>(&pools);
}
