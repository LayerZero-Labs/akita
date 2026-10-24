use super::*;
use akita_challenges::{
    FoldChallengeDrawDomain, FoldDraw, PreviewFoldDraw, ProverFoldDraw, SparseChallengeConfig,
    VerifierFoldDraw,
};
use akita_params::{ChallengeFieldOrder, GrindingRun};
use akita_transcript::{new_prover_channel, new_verifier_channel, ProverTranscript};
use jolt_field::Prime128Offset275 as F;
use jolt_transcript::Probe;
use spongefish::Encoding;

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
    let mut state = new_prover_channel(b"native-grinding");
    let mut prover = ProverGrinding::new(&mut state, &plan);
    let prover_challenge = prover
        .grinded_ext_challenge::<F>(GrindingSite::EvaluationBatch { level: 0 })
        .unwrap();
    prover
        .commit_fold_response(GrindingSite::FoldResponse { level: 0 }, 7)
        .unwrap();
    prover.record_fold_challenges(0, 0, 2).unwrap();
    prover.finish().unwrap();
    let proof = state.narg_string().to_vec();

    // One PoW nonce and one response nonce; public context records occupy
    // no argument bytes.
    assert!(proof.len() <= plan.nonce_max_bytes());
    let mut state = new_verifier_channel(b"native-grinding", &proof);
    let mut verifier = VerifierGrinding::new(&mut state, &plan);
    let verifier_challenge = verifier
        .grinded_ext_challenge::<F>(GrindingSite::EvaluationBatch { level: 0 })
        .unwrap();
    assert_eq!(verifier_challenge, prover_challenge);
    assert_eq!(
        verifier
            .read_fold_response(GrindingSite::FoldResponse { level: 0 })
            .unwrap(),
        7
    );
    verifier.record_fold_challenges(0, 0, 2).unwrap();
    verifier.finish().unwrap().check_eof(state).unwrap();
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
    let mut state = new_prover_channel(b"native-long-nonce");
    let search = state.message_search();
    let nonce = (128..=u8::MAX as u32)
        .find(|&candidate| {
            let mut check = [0u8; 1];
            search.probe(NonceAtom::new(candidate).encode().as_ref(), &mut check);
            check[0] & 1 == 0
        })
        .expect("the two-byte half of an 8-bit nonce domain must contain a winner");
    state.prover_message(&NonceAtom::new(nonce));
    let _: [u8; 1] = state.verifier_message();
    let proof = state.narg_string().to_vec();
    assert_eq!(proof.len(), 2);

    let mut state = new_verifier_channel(b"native-long-nonce", &proof);
    let mut verifier = VerifierGrinding::new(&mut state, &plan);
    verifier.grinded_ext_challenge::<F>(site).unwrap();
    verifier.finish().unwrap().check_eof(state).unwrap();
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
    let mut state = new_prover_channel(b"native-vector-grinding");
    let mut prover = ProverGrinding::new(&mut state, &plan);
    let prover_challenges = prover
        .grinded_ext_challenges::<F>(GrindingSite::ExtensionOpeningPoint { level: 4 }, 3)
        .unwrap();
    prover.finish().unwrap();
    let proof = state.narg_string().to_vec();
    assert!(proof.len() <= plan.nonce_max_bytes());

    let mut state = new_verifier_channel(b"native-vector-grinding", &proof);
    let mut verifier = VerifierGrinding::new(&mut state, &plan);
    let verifier_challenges = verifier
        .grinded_ext_challenges::<F>(GrindingSite::ExtensionOpeningPoint { level: 4 }, 3)
        .unwrap();
    assert_eq!(verifier_challenges, prover_challenges);
    verifier.finish().unwrap().check_eof(state).unwrap();
}

#[test]
fn query_cursor_errors_poison_both_replay_roles() {
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    let site = GrindingSite::EvaluationBatch { level: 0 };
    let wrong_site = GrindingSite::EvaluationBatch { level: 1 };
    let wrong_kind = GrindingSite::FoldResponse { level: 0 };
    for (runs, requested) in [
        (vec![], site),
        (
            vec![GrindingRun::proof_of_work(site, 1, order).unwrap()],
            wrong_site,
        ),
        (vec![GrindingRun::fold_response(0)], wrong_kind),
    ] {
        let plan = GrindingPlan::new(runs, order).unwrap();
        let mut state = new_prover_channel(b"cursor-query");
        let mut prover = ProverGrinding::new(&mut state, &plan);
        assert!(matches!(
            prover.grind_query(requested),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(prover.finish(), Err(AkitaError::InvalidInput(_))));

        let mut state = new_verifier_channel(b"cursor-query", &[]);
        let mut verifier = VerifierGrinding::new(&mut state, &plan);
        assert!(matches!(
            verifier.grind_query(requested),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            spongefish::Transcript::check(verifier.state_mut(), || true),
            Err(spongefish::VerificationError)
        ));
        assert!(matches!(verifier.finish(), Err(AkitaError::InvalidProof)));
    }
}

#[test]
fn fold_run_cursor_errors_poison_both_replay_roles() {
    let order = ChallengeFieldOrder::from_full_capacity(128).unwrap();
    for runs in [
        vec![],
        vec![GrindingRun::fold_challenge_group(1, 0, 2).unwrap()],
        vec![GrindingRun::fold_challenge_group(0, 1, 2).unwrap()],
        vec![GrindingRun::fold_challenge_group(0, 0, 1).unwrap()],
        vec![GrindingRun::fold_response(0)],
    ] {
        let plan = GrindingPlan::new(runs, order).unwrap();
        let mut state = new_prover_channel(b"cursor-run");
        let mut prover = ProverGrinding::new(&mut state, &plan);
        assert!(matches!(
            prover.record_fold_challenges(0, 0, 2),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(prover.finish(), Err(AkitaError::InvalidInput(_))));

        let mut state = new_verifier_channel(b"cursor-run", &[]);
        let mut verifier = VerifierGrinding::new(&mut state, &plan);
        assert!(matches!(
            verifier.record_fold_challenges(0, 0, 2),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(matches!(
            spongefish::Transcript::check(verifier.state_mut(), || true),
            Err(spongefish::VerificationError)
        ));
        assert!(matches!(verifier.finish(), Err(AkitaError::InvalidProof)));
    }
}

#[test]
fn grinding_sumcheck_rejects_unscheduled_invocation() {
    let plan = plan();
    let mut state = new_verifier_channel(b"sumcheck-invocation", &[]);
    let mut verifier = VerifierGrinding::new(&mut state, &plan);
    let mut channel = GrindingSumcheckVerifier::<F, F>::new(
        &mut verifier,
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
fn grinding_rejects_out_of_range_response_and_incomplete_plan() {
    let plan = GrindingPlan::new(
        vec![GrindingRun::fold_response(0)],
        ChallengeFieldOrder::from_full_capacity(128).unwrap(),
    )
    .unwrap();
    let mut state = new_prover_channel(b"native-grinding");
    let mut preview_prover = ProverGrinding::new(&mut state, &plan);
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

    let mut state = new_prover_channel(b"native-grinding");
    let mut prover = ProverGrinding::new(&mut state, &plan);
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
        let mut state = new_verifier_channel(b"native-grinding", proof);
        let mut verifier = VerifierGrinding::new(&mut state, &plan);
        assert_eq!(
            verifier.read_fold_response(GrindingSite::FoldResponse { level: 0 }),
            expected
        );
        if should_accept {
            verifier.finish().unwrap().check_eof(state).unwrap();
        } else {
            assert!(verifier.state_mut().prover_message::<u8>().is_err());
            assert!(matches!(verifier.finish(), Err(AkitaError::InvalidProof)));
        }
    }
}

#[test]
fn pending_eof_rejects_trailing_bytes() {
    let plan = GrindingPlan::new(
        vec![GrindingRun::fold_response(0)],
        ChallengeFieldOrder::from_full_capacity(128).unwrap(),
    )
    .unwrap();
    let mut state = new_verifier_channel(b"native-grinding", &[0x05, 0x00]);
    let mut verifier = VerifierGrinding::new(&mut state, &plan);
    assert_eq!(
        verifier.read_fold_response(GrindingSite::FoldResponse { level: 0 }),
        Ok(5)
    );
    let pending = verifier.finish().unwrap();
    assert!(matches!(
        pending.check_eof(state),
        Err(AkitaError::InvalidProof)
    ));
}

/// A failed replay rejects the borrowed state itself, so a caller that owns
/// the state cannot accept it afterwards.
#[test]
fn replay_rejection_poisons_the_borrowed_state() {
    let plan = GrindingPlan::new(
        vec![GrindingRun::fold_response(0)],
        ChallengeFieldOrder::from_full_capacity(128).unwrap(),
    )
    .unwrap();
    let mut state = new_verifier_channel(b"native-grinding", &[0x80, 0x20]);
    let mut verifier = VerifierGrinding::new(&mut state, &plan);
    assert!(verifier
        .read_fold_response(GrindingSite::FoldResponse { level: 0 })
        .is_err());
    assert!(state.prover_message::<u8>().is_err());
    assert!(state.check_eof().is_err());
}

#[test]
fn grinding_preview_rejects_non_fold_response_query_kind() {
    let plan = plan();
    let mut state = new_prover_channel(b"native-grinding");
    let prover = ProverGrinding::new(&mut state, &plan);
    assert!(matches!(
        prover.preview_fold_response(GrindingSite::EvaluationBatch { level: 0 }, 0),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn grinding_preview_rejects_exhausted_plan_without_mutating_replay() {
    let plan = GrindingPlan::new(
        vec![],
        ChallengeFieldOrder::from_full_capacity(128).unwrap(),
    )
    .unwrap();
    let mut state = new_prover_channel(b"native-grinding");
    let prover = ProverGrinding::new(&mut state, &plan);
    assert!(matches!(
        prover.preview_fold_response(GrindingSite::FoldResponse { level: 0 }, 0),
        Err(AkitaError::InvalidInput(_))
    ));
    prover.finish().unwrap();
}

#[test]
fn grinding_finish_rejects_unconsumed_plan() {
    let plan = plan();
    let mut state = new_prover_channel(b"native-grinding");
    let prover = ProverGrinding::new(&mut state, &plan);
    assert!(matches!(prover.finish(), Err(AkitaError::InvalidInput(_))));
}

#[test]
fn grinding_rejects_fold_challenge_coordinate_count_overflow() {
    let plan = plan();
    let mut state = new_prover_channel(b"native-grinding");
    let mut prover = ProverGrinding::new(&mut state, &plan);
    assert!(matches!(
        prover.record_fold_challenges(0, 0, usize::MAX),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(prover.finish(), Err(AkitaError::InvalidInput(_))));
}

#[test]
fn grinding_sumcheck_prover_rejects_nonzero_invocation() {
    let plan = plan();
    let mut state = new_prover_channel(b"native-grinding");
    let mut prover = ProverGrinding::new(&mut state, &plan);
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

    let mut state = new_prover_channel(b"native-poison");
    let mut prover = ProverGrinding::new(&mut state, &plan);
    assert!(matches!(
        prover.grinded_ext_challenges::<F>(site, usize::MAX),
        Err(AkitaError::InvalidSetup(_))
    ));
    assert!(matches!(prover.finish(), Err(AkitaError::InvalidInput(_))));

    let mut state = new_verifier_channel(b"native-poison", &[]);
    let mut verifier = VerifierGrinding::new(&mut state, &plan);
    assert_eq!(
        verifier.grinded_ext_challenges::<F>(site, usize::MAX),
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
    let mut state = new_prover_channel(b"native-fold-transaction");
    let mut prover = ProverGrinding::new(&mut state, &plan);
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
    prover.finish().unwrap();
    let proof = state.narg_string().to_vec();

    let mut state = new_verifier_channel(b"native-fold-transaction", &proof);
    let mut verifier = VerifierGrinding::new(&mut state, &plan);
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
    verifier.finish().unwrap().check_eof(state).unwrap();
}
