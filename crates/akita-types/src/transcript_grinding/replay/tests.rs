use super::*;
use crate::transcript::test_transcripts::{prover as new_prover, verifier as new_verifier};
use akita_challenges::{
    Challenges, FoldChallengeDrawDomain, FoldDraw, ForkFoldDraw, SparseChallengeConfig,
};
use akita_params::{ChallengeFieldOrder, GrindingRun};
use jolt_field::Prime128Offset275 as F;
use jolt_transcript::{grinding_predicate_accepts, GRINDING_PREDICATE_LEN};
use std::num::NonZeroU8;

/// The two groups' fold challenges, drawn in group order from `fork`.
fn draw_two_groups<H: Sponge>(
    fork: &mut Fork<H>,
    config: &SparseChallengeConfig,
) -> (Challenges, Challenges) {
    let mut draw = ForkFoldDraw::new(fork);
    let first = draw
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            0,
            2,
            1,
            config,
            None,
        )
        .unwrap();
    let second = draw
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            1,
            1,
            2,
            config,
            None,
        )
        .unwrap();
    (first, second)
}

fn plan() -> GrindingPlan {
    GrindingPlan::new(
        vec![
            GrindingRun::proof_of_work(
                GrindingSite::EvaluationBatch { level: 0 },
                3,
                ChallengeFieldOrder::from_full_capacity(128).unwrap(),
            )
            .unwrap(),
            GrindingRun::fold_response(0),
            GrindingRun::fold_challenge_group(0, 0, 2).unwrap(),
        ],
        ChallengeFieldOrder::from_full_capacity(128).unwrap(),
    )
    .unwrap()
}

#[test]
fn grinding_roundtrip_uses_inline_canonical_nonces_and_eof() {
    let plan = plan();
    let mut transcript = new_prover(b"native-grinding");
    let mut prover = ProverGrinding::new(&mut transcript, &plan);
    let prover_challenge = prover
        .grinded_ext_challenge::<F, F>(GrindingSite::EvaluationBatch { level: 0 })
        .unwrap();
    prover
        .begin_fold_response(GrindingSite::FoldResponse { level: 0 })
        .unwrap();
    let mut prover_fork = prover
        .commit_fold_response(GrindingSite::FoldResponse { level: 0 }, 7)
        .unwrap();
    prover.record_fold_challenges(0, 0, 2).unwrap();
    prover.finish().unwrap();
    let proof = transcript.finish();

    // One PoW nonce and one response nonce; public context records occupy
    // no argument bytes.
    assert!(proof.len() <= plan.nonce_max_bytes());
    let mut transcript = new_verifier(b"native-grinding", &proof);
    let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
    let verifier_challenge = verifier
        .grinded_ext_challenge::<F, F>(GrindingSite::EvaluationBatch { level: 0 })
        .unwrap();
    assert_eq!(verifier_challenge, prover_challenge);
    let mut verifier_fork = verifier
        .read_fold_response(GrindingSite::FoldResponse { level: 0 })
        .unwrap();
    assert_eq!(verifier_fork.squeeze::<32>(), prover_fork.squeeze::<32>());
    verifier.record_fold_challenges(0, 0, 2).unwrap();
    verifier.finish().unwrap();
    transcript.finish().unwrap();
}

#[test]
fn verifier_accepts_a_valid_nonminimal_two_byte_pow_nonce() {
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    let site = GrindingSite::EvaluationBatch { level: 0 };
    let plan = GrindingPlan::new(
        vec![GrindingRun::proof_of_work(site, 2, order).unwrap()],
        order,
    )
    .unwrap();
    let mut state = new_prover(b"native-long-nonce");
    let seed: [u8; FORK_SEED_LEN] = state.challenge_bytes();
    let nonce = (128..=u8::MAX as u32)
        .find(|&candidate| {
            grinding_predicate_accepts(
                &Fork::<jolt_transcript::Blake2b512>::new(&seed, candidate)
                    .squeeze::<GRINDING_PREDICATE_LEN>(),
                NonZeroU8::new(1).unwrap(),
            )
        })
        .expect("the two-byte half of an 8-bit nonce domain must contain a winner");
    state.send_nonce(nonce);
    let proof = state.finish();
    assert_eq!(proof.len(), 2);

    let mut transcript = new_verifier(b"native-long-nonce", &proof);
    let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
    verifier.grinded_ext_challenge::<F, F>(site).unwrap();
    verifier.finish().unwrap();
    transcript.finish().unwrap();
}

#[test]
fn one_grinding_query_can_protect_multiple_draws() {
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    let plan = GrindingPlan::new(
        vec![GrindingRun::proof_of_work(
            GrindingSite::ExtensionOpeningPoint { level: 4 },
            2,
            order,
        )
        .unwrap()],
        order,
    )
    .unwrap();
    let mut transcript = new_prover(b"native-vector-grinding");
    let mut prover = ProverGrinding::new(&mut transcript, &plan);
    let prover_challenges = prover
        .grinded_ext_challenges::<F, F>(GrindingSite::ExtensionOpeningPoint { level: 4 }, 3)
        .unwrap();
    prover.finish().unwrap();
    let proof = transcript.finish();
    assert!(proof.len() <= plan.nonce_max_bytes());

    let mut transcript = new_verifier(b"native-vector-grinding", &proof);
    let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
    let verifier_challenges = verifier
        .grinded_ext_challenges::<F, F>(GrindingSite::ExtensionOpeningPoint { level: 4 }, 3)
        .unwrap();
    assert_eq!(verifier_challenges, prover_challenges);
    verifier.finish().unwrap();
    transcript.finish().unwrap();
}

#[test]
fn grinding_rejects_out_of_range_response_and_incomplete_plan() {
    let plan = GrindingPlan::new(
        vec![GrindingRun::fold_response(0)],
        ChallengeFieldOrder::from_full_capacity(128).unwrap(),
    )
    .unwrap();
    let mut transcript = new_prover(b"native-grinding");
    let mut prover = ProverGrinding::new(&mut transcript, &plan);
    assert!(matches!(
        prover.commit_fold_response(
            GrindingSite::FoldResponse { level: 0 },
            akita_params::FOLD_RESPONSE_ATTEMPTS,
        ),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(prover.finish(), Err(AkitaError::InvalidInput(_))));

    for (proof, expected) in [
        (&[0xff, 0x1f][..], Ok(4095)),
        (&[0x80, 0x20][..], Err(AkitaError::InvalidProof)),
    ] {
        let should_accept = expected.is_ok();
        let mut transcript = new_verifier(b"native-grinding", proof);
        let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
        assert_eq!(
            verifier
                .read_fold_response(GrindingSite::FoldResponse { level: 0 })
                .map(|_| ()),
            expected.map(|_: u32| ())
        );
        let replay = verifier.finish();
        let end = transcript.finish();
        assert_eq!(replay.is_ok(), should_accept);
        assert_eq!(end.is_ok(), should_accept);
    }
}

#[test]
fn replay_poison_survives_post_advance_allocation_failure() {
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    let site = GrindingSite::ExtensionOpeningPoint { level: 9 };
    let plan = GrindingPlan::new(
        vec![GrindingRun::proof_of_work(site, 1, order).unwrap()],
        order,
    )
    .unwrap();

    let mut transcript = new_prover(b"native-poison");
    let mut prover = ProverGrinding::new(&mut transcript, &plan);
    assert_eq!(
        prover.grinded_ext_challenges::<F, F>(site, usize::MAX),
        Err(AkitaError::InvalidProof)
    );
    assert!(matches!(prover.finish(), Err(AkitaError::InvalidInput(_))));

    let mut transcript = new_verifier(b"native-poison", &[]);
    let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
    assert_eq!(
        verifier.grinded_ext_challenges::<F, F>(site, usize::MAX),
        Err(AkitaError::InvalidProof)
    );
    assert!(matches!(verifier.finish(), Err(AkitaError::InvalidProof)));
}

#[test]
fn nonce_width_membership_is_total() {
    assert!(value_fits(0, 0));
    assert!(!value_fits(1, 0));
    assert!(value_fits((1 << 31) - 1, 31));
    assert!(!value_fits(1 << 31, 31));
    assert!(value_fits(u32::MAX, 32));
    assert!(!value_fits(0, 33));
    assert!(!value_fits(u32::MAX, u8::MAX));
}

#[test]
fn fold_candidate_replays_all_groups_as_one_transaction() {
    let plan = GrindingPlan::new(
        vec![
            GrindingRun::fold_response(3),
            GrindingRun::fold_challenge_group(3, 0, 2).unwrap(),
            GrindingRun::fold_challenge_group(3, 1, 2).unwrap(),
        ],
        ChallengeFieldOrder::from_full_capacity(128).unwrap(),
    )
    .unwrap();
    let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    let site = GrindingSite::FoldResponse { level: 3 };
    let nonce = 7;
    let mut transcript = new_prover(b"native-fold-transaction");
    let mut prover = ProverGrinding::new(&mut transcript, &plan);
    prover.begin_fold_response(site).unwrap();
    let candidate = draw_two_groups(
        &mut prover.fold_response_fork(site, nonce).unwrap(),
        &config,
    );
    let committed = draw_two_groups(
        &mut prover.commit_fold_response(site, nonce).unwrap(),
        &config,
    );
    assert_eq!(candidate, committed);
    prover.record_fold_challenges(3, 0, 2).unwrap();
    prover.record_fold_challenges(3, 1, 2).unwrap();
    prover.finish().unwrap();
    let proof = transcript.finish();

    let mut transcript = new_verifier(b"native-fold-transaction", &proof);
    let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
    let verified = draw_two_groups(&mut verifier.read_fold_response(site).unwrap(), &config);
    assert_eq!(verified, committed);
    verifier.record_fold_challenges(3, 0, 2).unwrap();
    verifier.record_fold_challenges(3, 1, 2).unwrap();
    verifier.finish().unwrap();
    transcript.finish().unwrap();
}
