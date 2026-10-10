#![cfg(feature = "labinius")]

mod common;

use akita_algebra::poly::multilinear_eval;
use akita_labinius_prover::combined_kernel::{self, CombinedRootKernel};
use akita_labinius_verifier::{
    channel::{self, RootChallengeChannel, RootSumcheckProverChannel, RootSumcheckVerifierChannel},
    grinding::RootGrindingPlan,
    root_sumcheck::{combined_shape, combined_terminal, verify_combined_rounds},
};
use akita_sumcheck::SumcheckInstanceProver;
use common::combined::CombinedRootSumcheck;
use jolt_field::{
    CanonicalEncoding, Ext2, ExtField, Field, One, Prime128Offset275, Prime64Offset59, Ring, Zero,
};
use rand::{rngs::StdRng, RngCore, SeedableRng};

type F = Prime128Offset275;
type P64 = Prime64Offset59;
const INVOCATION: u32 = 37;
/// Stored digits are base 16.
const ALPHABET: u32 = 16;
/// Rounds in which the kernel may own a class table.
const CLASS_ROUNDS: usize = 2;
/// Padded and live coefficients of one ring element at the sample geometry.
const PADDED_COEFFICIENTS: usize = 1024;
const LIVE_COEFFICIENTS: usize = 648;

struct Fixture<E> {
    digits: Vec<u8>,
    digit_factor: Vec<E>,
    compact: Vec<E>,
    kw: Vec<E>,
    tau: Vec<E>,
    beta: E,
    s: E,
}

fn fixture<E: Field>(nu: usize, rng: &mut StdRng) -> Fixture<E> {
    fixture_with_factor(nu, nu.min(CLASS_ROUNDS), rng)
}

fn fixture_with_factor<E: Field>(nu: usize, a: usize, rng: &mut StdRng) -> Fixture<E> {
    let digits: Vec<_> = (0..1usize << nu)
        .map(|_| (rng.next_u32() % ALPHABET) as u8)
        .collect();
    let digit_factor = (0..1usize << a).map(|_| E::random(rng)).collect();
    let compact = (0..1usize << (nu - a)).map(|_| E::random(rng)).collect();
    let mut f = Fixture {
        digits,
        digit_factor,
        compact,
        kw: Vec::new(),
        tau: (0..nu).map(|_| E::random(rng)).collect(),
        beta: E::random(rng),
        s: E::zero(),
    };
    refresh_weights(&mut f);
    f
}

fn refresh_weights<E: Field>(f: &mut Fixture<E>) {
    f.kw = f
        .compact
        .iter()
        .flat_map(|&coefficient| f.digit_factor.iter().map(move |&digit| digit * coefficient))
        .collect();
    f.s = f.digits.iter().zip(&f.kw).fold(E::zero(), |sum, (&w, &k)| {
        sum + E::from_u64(u64::from(w)) * k
    });
}

/// Zero the digits outside the live digit slots (bit `slot % 8` of `slots`)
/// and outside the live coefficients, as padding does. Weights stay random
/// there: skipping zero digits must not depend on them.
fn pad<E: Field>(f: &mut Fixture<E>, slots: u8) {
    let factor = f.digit_factor.len();
    for (index, digit) in f.digits.iter_mut().enumerate() {
        let live_slot = (slots >> (index % factor % 8)) & 1 == 1;
        if !live_slot || index / factor % PADDED_COEFFICIENTS >= LIVE_COEFFICIENTS {
            *digit = 0;
        }
    }
    refresh_weights(f);
}

fn manual<E: Field>(f: &Fixture<E>, challenges: &[E]) {
    let mut reference =
        CombinedRootSumcheck::new(&f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
    let mut kernel = CombinedRootKernel::new(
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
        assert_eq!(expected, actual, "round={round}");
        // Neither the public round argument nor a previous claim changes the
        // polynomial of the currently bound state, even for false claims.
        assert_eq!(
            reference.compute_round_univariate(usize::MAX, claim + E::one()),
            kernel.compute_round_univariate(usize::MAX, claim + E::one())
        );
        claim = expected.evaluate(challenge);
        reference.ingest_challenge(round, challenge);
        kernel.ingest_challenge(round, challenge);
        assert_eq!(reference.final_evaluations(), kernel.final_evaluations());
    }
    let w: Vec<_> = f
        .digits
        .iter()
        .map(|&digit| E::from_u64(u64::from(digit)))
        .collect();
    assert_eq!(
        kernel.final_evaluations(),
        Some((
            multilinear_eval(&w, challenges).unwrap(),
            multilinear_eval(&f.kw, challenges).unwrap(),
        ))
    );
    reference.ingest_challenge(usize::MAX, E::one());
    kernel.ingest_challenge(usize::MAX, E::one());
    assert_eq!(reference.final_evaluations(), kernel.final_evaluations());
    assert_eq!(
        reference.compute_round_univariate(usize::MAX, claim),
        kernel.compute_round_univariate(usize::MAX, claim)
    );
}

fn transcripts<B: Field + CanonicalEncoding, E: ExtField<B>>(f: &Fixture<E>) -> Vec<u8> {
    let mut reference =
        CombinedRootSumcheck::new(&f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
    let mut kernel = CombinedRootKernel::new(
        &f.digits,
        f.digit_factor.clone(),
        f.compact.clone(),
        &f.tau,
        f.beta,
        f.s,
    )
    .unwrap();
    let plan = RootGrindingPlan::sumcheck_rounds::<B, E>(&[(
        INVOCATION,
        combined_shape(f.tau.len()).unwrap(),
    )])
    .unwrap();
    let mut reference_state = channel::new_prover().unwrap();
    let mut kernel_state = channel::new_prover().unwrap();
    let mut reference_channel = RootSumcheckProverChannel::<B>::new(&mut reference_state);
    let mut kernel_channel = RootSumcheckProverChannel::<B>::new(&mut kernel_state);
    RootChallengeChannel::<E>::schedule(&mut reference_channel, plan.clone()).unwrap();
    RootChallengeChannel::<E>::schedule(&mut kernel_channel, plan.clone()).unwrap();
    let reference_result = common::combined::prove_combined_rounds::<B, E, _>(
        &mut reference,
        &mut reference_channel,
        INVOCATION,
    )
    .expect("honest reference sumcheck succeeds");
    let kernel_result = combined_kernel::prove_combined_rounds::<B, E, _>(
        &mut kernel,
        &mut kernel_channel,
        INVOCATION,
    )
    .expect("honest kernel sumcheck succeeds");
    RootChallengeChannel::<E>::finish_schedule(&mut reference_channel).unwrap();
    RootChallengeChannel::<E>::finish_schedule(&mut kernel_channel).unwrap();
    assert_eq!(reference_result, kernel_result);
    let reference_bytes = channel::finish_prover(reference_state);
    let kernel_bytes = channel::finish_prover(kernel_state);
    assert_eq!(reference_bytes, kernel_bytes);
    assert_eq!(reference.final_evaluations(), kernel.final_evaluations());
    let (point, claim) = kernel_result;
    let mut verifier = channel::new_verifier(&kernel_bytes).unwrap();
    let mut verifier_channel = RootSumcheckVerifierChannel::<B>::new(&mut verifier);
    RootChallengeChannel::<E>::schedule(&mut verifier_channel, plan).unwrap();
    let replay = verify_combined_rounds::<B, E, _>(
        &mut verifier_channel,
        INVOCATION,
        f.tau.len(),
        f.beta,
        f.s,
    )
    .unwrap();
    RootChallengeChannel::<E>::finish_schedule(&mut verifier_channel).unwrap();
    assert_eq!(
        (replay.challenges, replay.output_claim),
        (point.clone(), claim)
    );
    channel::finish_verifier(verifier).unwrap();
    let w: Vec<_> = f
        .digits
        .iter()
        .map(|&digit| E::from_u64(u64::from(digit)))
        .collect();
    let evaluations = (
        multilinear_eval(&w, &point).unwrap(),
        multilinear_eval(&f.kw, &point).unwrap(),
    );
    assert_eq!(kernel.final_evaluations(), Some(evaluations));
    assert_eq!(
        claim,
        combined_terminal(&f.tau, &point, f.beta, evaluations.0, evaluations.1).unwrap()
    );
    kernel_bytes
}

fn random_rounds<B: Field + CanonicalEncoding, E: ExtField<B>>() {
    let mut rng = StdRng::seed_from_u64(0xc0_6b_1e);
    // nu=16 also exercises larger packed classes through the direct path.
    for nu in (0..=13).chain([16]) {
        let f = fixture::<E>(nu, &mut rng);
        let challenges: Vec<_> = (0..nu).map(|_| E::random(&mut rng)).collect();
        manual(&f, &challenges);
        transcripts::<B, E>(&f);
    }
}

#[test]
fn random_rounds_transcripts_and_terminal_checks() {
    random_rounds::<F, F>();
    random_rounds::<P64, Ext2<P64>>();
}

fn digit_factor_shapes<B: Field + CanonicalEncoding, E: ExtField<B>>() {
    let mut rng = StdRng::seed_from_u64(0x000f_ac70);
    // N = 256 is below every class table. N = 2048 owns one-byte class
    // tables: small factors sum their weights in compact form, large ones
    // add the two weight endpoints of each pair.
    for (nu, a) in (0..=8).map(|a| (8, a)).chain((0..=5).map(|a| (11, a))) {
        let mut f = fixture_with_factor::<E>(nu, a, &mut rng);
        // Nontrivial digit factors and padded coefficient positions expose
        // both the low-variable tensor order and zero compact entries.
        for coefficient in f.compact.iter_mut().skip(1).step_by(3) {
            *coefficient = E::zero();
        }
        refresh_weights(&mut f);
        let challenges: Vec<_> = (0..nu).map(|_| E::random(&mut rng)).collect();
        manual(&f, &challenges);
        transcripts::<B, E>(&f);
        f.s += E::one();
        manual(&f, &challenges);
    }
}

#[test]
fn digit_factor_shapes_and_padded_compact_entries_match_dense_reference() {
    digit_factor_shapes::<F, F>();
    digit_factor_shapes::<P64, Ext2<P64>>();
}

fn structured_tables<B: Field + CanonicalEncoding, E: ExtField<B>>() {
    let mut rng = StdRng::seed_from_u64(0x5747_ab1e);
    // Round 2 reads eight digits as two 16-bit classes: slots 0..4 against
    // slots 4..8. With few live slots the kernel keeps moments per class of
    // the endpoint with the smaller class bound: the right one, the left
    // one, and the left one on a tie. Sixteen slots keep the digit factor
    // alive past the lift. Three live slots on the other side are too many
    // classes for a table of powers at this size, so each pair computes
    // them. N = 2^14 has no two-byte class table.
    for (a, slots) in [
        (3, 0b0001_0011),
        (3, 0b0011_0001),
        (3, 0b0001_0001),
        (4, 0b0001_0011),
        (3, 0b0001_0111),
        (3, 0b0111_0001),
    ] {
        let mut f = fixture_with_factor::<E>(14, a, &mut rng);
        pad(&mut f, slots);
        let challenges: Vec<_> = (0..14).map(|_| E::random(&mut rng)).collect();
        manual(&f, &challenges);
        transcripts::<B, E>(&f);
    }
    // N = 2^18 is the smallest table with a two-byte class table. The
    // response shape (four slots, three live) sums weights in compact form
    // and has a moment round with 4096 classes at both endpoints. The image
    // shape (eight slots, seven live) adds weight endpoints; its moment
    // round has 4096 classes on the right and computes powers of the left.
    for (a, slots) in [(2, 0b0111), (3, 0b0111_1111)] {
        let mut f = fixture_with_factor::<E>(18, a, &mut rng);
        pad(&mut f, slots);
        transcripts::<B, E>(&f);
    }
}

#[test]
fn padded_tables_reach_class_tables_and_the_moment_round() {
    structured_tables::<F, F>();
    structured_tables::<P64, Ext2<P64>>();
}

#[test]
fn extremal_digits_zero_weights_and_boolean_equality_points() {
    let mut rng = StdRng::seed_from_u64(0xed_6e);
    let max = (ALPHABET - 1) as u8;
    for pattern in 0..3 {
        let mut f = fixture::<F>(13, &mut rng);
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
                _ => F::random(&mut rng),
            })
            .collect();
        manual(&f, &challenges);
        transcripts::<F, F>(&f);
        f.beta = F::zero();
        manual(&f, &challenges);
    }
}

#[test]
fn false_linear_claim_preserves_polynomials_and_is_rejected() {
    let mut rng = StdRng::seed_from_u64(0xfa_15e);
    let mut f = fixture::<F>(8, &mut rng);
    f.beta = F::from_u64(7);
    let honest_proof = transcripts::<F, F>(&f);
    f.s += F::one();
    let challenges: Vec<_> = (0..f.tau.len()).map(|_| F::random(&mut rng)).collect();
    manual(&f, &challenges);
    // The shared engine rejects the inconsistent input claim before
    // writing round zero; both instances must report the same error.
    let mut reference =
        CombinedRootSumcheck::new(&f.digits, f.kw.clone(), &f.tau, f.beta, f.s).unwrap();
    let mut kernel = CombinedRootKernel::new(
        &f.digits,
        f.digit_factor.clone(),
        f.compact.clone(),
        &f.tau,
        f.beta,
        f.s,
    )
    .unwrap();
    let plan = RootGrindingPlan::sumcheck_rounds::<F, F>(&[(
        INVOCATION,
        combined_shape(f.tau.len()).unwrap(),
    )])
    .unwrap();
    let mut reference_state = channel::new_prover().unwrap();
    let mut kernel_state = channel::new_prover().unwrap();
    let mut reference_channel = RootSumcheckProverChannel::<F>::new(&mut reference_state);
    let mut kernel_channel = RootSumcheckProverChannel::<F>::new(&mut kernel_state);
    RootChallengeChannel::<F>::schedule(&mut reference_channel, plan.clone()).unwrap();
    RootChallengeChannel::<F>::schedule(&mut kernel_channel, plan.clone()).unwrap();
    let reference_result = common::combined::prove_combined_rounds::<F, F, _>(
        &mut reference,
        &mut reference_channel,
        INVOCATION,
    );
    let kernel_result = combined_kernel::prove_combined_rounds::<F, F, _>(
        &mut kernel,
        &mut kernel_channel,
        INVOCATION,
    );
    assert!(reference_result.is_err());
    assert_eq!(reference_result, kernel_result);
    // Rejection precedes every challenge, leaving both schedules incomplete.
    assert!(RootChallengeChannel::<F>::finish_schedule(&mut reference_channel).is_err());
    assert!(RootChallengeChannel::<F>::finish_schedule(&mut kernel_channel).is_err());
    let rejected_prefix = channel::finish_prover(kernel_state);
    assert_eq!(channel::finish_prover(reference_state), rejected_prefix);
    assert!(rejected_prefix.is_empty());
    assert_eq!(reference.final_evaluations(), kernel.final_evaluations());
    // Both provers produced the same proof bytes above. Compressed
    // round replay may reconstruct a polynomial using the false claim;
    // rejection therefore includes the authenticated terminal obligation.
    let mut state = channel::new_verifier(&honest_proof).unwrap();
    let mut verifier = RootSumcheckVerifierChannel::<F>::new(&mut state);
    RootChallengeChannel::<F>::schedule(&mut verifier, plan).unwrap();
    if let Ok(replay) =
        verify_combined_rounds::<F, F, _>(&mut verifier, INVOCATION, f.tau.len(), f.beta, f.s)
    {
        RootChallengeChannel::<F>::finish_schedule(&mut verifier).unwrap();
        channel::finish_verifier(state).unwrap();
        let w: Vec<_> = f
            .digits
            .iter()
            .map(|&digit| F::from_u64(u64::from(digit)))
            .collect();
        let terminal = combined_terminal(
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
