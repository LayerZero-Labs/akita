use super::*;
use akita_challenges::{
    FoldChallengeDrawDomain, FoldDraw, PreviewFoldDraw, ProverFoldDraw, SparseChallengeConfig,
    VerifierFoldDraw,
};
use akita_params::{ChallengeFieldOrder, GrindingRun};
use akita_transcript::{new_prover_channel, new_verifier_channel, preview_grinding_predicate};
use jolt_field::Prime128Offset275 as F;

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
    let state = new_prover_channel(b"native-grinding", b"fixture").unwrap();
    let mut prover = ProverGrinding::new(state, &plan);
    let prover_challenge = prover
        .grinded_ext_challenge::<F, F>(GrindingSite::EvaluationBatch { level: 0 })
        .unwrap();
    prover
        .commit_fold_response(GrindingSite::FoldResponse { level: 0 }, 7)
        .unwrap();
    prover.record_fold_challenges(0, 0, 2).unwrap();
    let proof = prover.finish().unwrap();

    // One PoW nonce and one response nonce; public context records occupy
    // no argument bytes.
    assert!(proof.len() <= plan.nonce_max_bytes());
    let state = new_verifier_channel(b"native-grinding", b"fixture", &proof).unwrap();
    let mut verifier = VerifierGrinding::new(state, &plan);
    let verifier_challenge = verifier
        .grinded_ext_challenge::<F, F>(GrindingSite::EvaluationBatch { level: 0 })
        .unwrap();
    assert_eq!(verifier_challenge, prover_challenge);
    assert_eq!(
        verifier
            .read_fold_response(GrindingSite::FoldResponse { level: 0 })
            .unwrap(),
        7
    );
    verifier.record_fold_challenges(0, 0, 2).unwrap();
    verifier.finish().unwrap();
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
    let mut state = new_prover_channel(b"native-long-nonce", b"fixture").unwrap();
    let nonce = (128..=u8::MAX as u32)
        .find(|&candidate| {
            grinding_predicate_accepts(
                &preview_grinding_predicate(&state, candidate),
                NonZeroU8::new(1).unwrap(),
            )
        })
        .expect("the two-byte half of an 8-bit nonce domain must contain a winner");
    let (nonce_record, predicate_record) = grinding_records(site, 1, 8);
    let _ = commit_grinding_nonce(&mut state, nonce_record, nonce, predicate_record);
    let proof = state.narg_string().to_vec();
    assert_eq!(proof.len(), 2);

    let state = new_verifier_channel(b"native-long-nonce", b"fixture", &proof).unwrap();
    let mut verifier = VerifierGrinding::new(state, &plan);
    verifier.grinded_ext_challenge::<F, F>(site).unwrap();
    verifier.finish().unwrap();
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
    let state = new_prover_channel(b"native-vector-grinding", b"fixture").unwrap();
    let mut prover = ProverGrinding::new(state, &plan);
    let prover_challenges = prover
        .grinded_ext_challenges::<F, F>(GrindingSite::ExtensionOpeningPoint { level: 4 }, 3)
        .unwrap();
    let proof = prover.finish().unwrap();
    assert!(proof.len() <= plan.nonce_max_bytes());

    let state = new_verifier_channel(b"native-vector-grinding", b"fixture", &proof).unwrap();
    let mut verifier = VerifierGrinding::new(state, &plan);
    let verifier_challenges = verifier
        .grinded_ext_challenges::<F, F>(GrindingSite::ExtensionOpeningPoint { level: 4 }, 3)
        .unwrap();
    assert_eq!(verifier_challenges, prover_challenges);
    verifier.finish().unwrap();
}

#[test]
fn grinding_rejects_out_of_range_response_and_incomplete_plan() {
    let plan = GrindingPlan::new(
        vec![GrindingRun::fold_response(0)],
        ChallengeFieldOrder::from_full_capacity(128).unwrap(),
    )
    .unwrap();
    let state = new_prover_channel(b"native-grinding", b"fixture").unwrap();
    let mut preview_prover = ProverGrinding::new(state, &plan);
    assert!(matches!(
        preview_prover.preview_fold_response(
            GrindingSite::FoldResponse { level: 0 },
            akita_params::FOLD_RESPONSE_ATTEMPTS,
        ),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        preview_prover.preview_fold_response(GrindingSite::FoldResponse { level: 1 }, 0),
        Err(AkitaError::InvalidInput(_))
    ));
    preview_prover
        .preview_fold_response(GrindingSite::FoldResponse { level: 0 }, 0)
        .unwrap();
    preview_prover
        .commit_fold_response(GrindingSite::FoldResponse { level: 0 }, 0)
        .unwrap();
    preview_prover.finish().unwrap();

    let state = new_prover_channel(b"native-grinding", b"fixture").unwrap();
    let mut prover = ProverGrinding::new(state, &plan);
    assert!(matches!(
        prover.commit_fold_response(
            GrindingSite::FoldResponse { level: 0 },
            akita_params::FOLD_RESPONSE_ATTEMPTS,
        ),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(prover.finish(), Err(AkitaError::Internal(_))));

    for (proof, expected) in [
        (&[0xff, 0x1f][..], Ok(4095)),
        (&[0x80, 0x20][..], Err(AkitaError::InvalidProof)),
    ] {
        let should_accept = expected.is_ok();
        let state = new_verifier_channel(b"native-grinding", b"fixture", proof).unwrap();
        let mut verifier = VerifierGrinding::new(state, &plan);
        assert_eq!(
            verifier.read_fold_response(GrindingSite::FoldResponse { level: 0 }),
            expected
        );
        if should_accept {
            verifier.finish().unwrap();
        } else {
            assert!(verifier.state_mut().verifier_message::<[u8; 32]>().is_err());
            assert!(matches!(verifier.finish(), Err(AkitaError::InvalidProof)));
        }
    }
}

#[test]
fn grinding_preview_rejects_non_fold_response_query_kind() {
    let plan = plan();
    let state = new_prover_channel(b"native-grinding", b"fixture").unwrap();
    let prover = ProverGrinding::new(state, &plan);
    assert!(matches!(
        prover.preview_fold_response(GrindingSite::EvaluationBatch { level: 0 }, 0),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn grinding_rejects_fold_challenge_coordinate_count_overflow() {
    let plan = plan();
    let state = new_prover_channel(b"native-grinding", b"fixture").unwrap();
    let mut prover = ProverGrinding::new(state, &plan);
    assert!(matches!(
        prover.record_fold_challenges(0, 0, usize::MAX),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(prover.finish(), Err(AkitaError::Internal(_))));
}

#[test]
fn grinding_sumcheck_prover_rejects_nonzero_invocation() {
    let plan = plan();
    let state = new_prover_channel(b"native-grinding", b"fixture").unwrap();
    let mut prover = ProverGrinding::new(state, &plan);
    let mut channel = GrindingSumcheckProver::<F, F>::new(
        &mut prover,
        akita_params::SumcheckProtocol::Stage1,
        0,
        0,
    );
    assert!(matches!(
        channel.round_challenge(1, 0),
        Err(AkitaError::InvalidInput(_))
    ));
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

    let state = new_prover_channel(b"native-poison", b"fixture").unwrap();
    let mut prover = ProverGrinding::new(state, &plan);
    assert_eq!(
        prover.grinded_ext_challenges::<F, F>(site, usize::MAX),
        Err(AkitaError::InvalidProof)
    );
    assert!(matches!(prover.finish(), Err(AkitaError::Internal(_))));

    let state = new_verifier_channel(b"native-poison", b"fixture", &[]).unwrap();
    let mut verifier = VerifierGrinding::new(state, &plan);
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
    let state = new_prover_channel(b"native-fold-transaction", b"fixture").unwrap();
    let mut prover = ProverGrinding::new(state, &plan);
    let (preview_first, preview_second) = {
        let mut preview_state = prover.preview_fold_response(site, nonce).unwrap();
        let first = PreviewFoldDraw::new(&mut preview_state)
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                0,
                2,
                1,
                &config,
                None,
            )
            .unwrap();
        let second = PreviewFoldDraw::new(&mut preview_state)
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                1,
                1,
                2,
                &config,
                None,
            )
            .unwrap();
        (first, second)
    };
    prover.commit_fold_response(site, nonce).unwrap();
    let live_first = ProverFoldDraw::new(prover.state_mut(), 3, 0)
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            0,
            2,
            1,
            &config,
            None,
        )
        .unwrap();
    prover.record_fold_challenges(3, 0, 2).unwrap();
    let live_second = ProverFoldDraw::new(prover.state_mut(), 3, 1)
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            1,
            1,
            2,
            &config,
            None,
        )
        .unwrap();
    prover.record_fold_challenges(3, 1, 2).unwrap();
    assert_eq!(
        (preview_first, preview_second),
        (live_first.clone(), live_second.clone())
    );
    let proof = prover.finish().unwrap();

    let state = new_verifier_channel(b"native-fold-transaction", b"fixture", &proof).unwrap();
    let mut verifier = VerifierGrinding::new(state, &plan);
    assert_eq!(verifier.read_fold_response(site).unwrap(), nonce);
    let verified_first = VerifierFoldDraw::new(verifier.state_mut(), 3, 0)
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            0,
            2,
            1,
            &config,
            None,
        )
        .unwrap();
    verifier.record_fold_challenges(3, 0, 2).unwrap();
    let verified_second = VerifierFoldDraw::new(verifier.state_mut(), 3, 1)
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            1,
            1,
            2,
            &config,
            None,
        )
        .unwrap();
    verifier.record_fold_challenges(3, 1, 2).unwrap();
    assert_eq!((verified_first, verified_second), (live_first, live_second));
    verifier.finish().unwrap();
}
