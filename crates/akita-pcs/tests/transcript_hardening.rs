#![allow(missing_docs)]
#![cfg(all(feature = "logging-transcript", not(feature = "zk")))]

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_transcript::{labels, AkitaTranscript, LoggingTranscript, Transcript, TranscriptEvent};
use akita_verifier::CommitmentVerifier;
use common::*;

type Scheme = AkitaCommitmentScheme<ONEHOT_D, OneHotCfg>;

fn event_label(event: &TranscriptEvent) -> Option<&[u8]> {
    match event {
        TranscriptEvent::Absorb { label, .. }
        | TranscriptEvent::Squeeze { label, .. }
        | TranscriptEvent::Wire { label, .. } => Some(label),
        TranscriptEvent::Preamble { .. } => None,
    }
}

fn first_label_index(events: &[TranscriptEvent], label: &[u8]) -> Option<usize> {
    events
        .iter()
        .position(|event| event_label(event).is_some_and(|candidate| candidate == label))
}

fn first_label_index_after(
    events: &[TranscriptEvent],
    start: usize,
    label: &[u8],
) -> Option<usize> {
    events[start..]
        .iter()
        .position(|event| event_label(event).is_some_and(|candidate| candidate == label))
        .map(|offset| start + offset)
}

fn assert_terminal_event_order(events: &[TranscriptEvent]) {
    let w_hat = first_label_index(events, labels::ABSORB_TERMINAL_W_HAT)
        .expect("terminal transcript must absorb logical w_hat");
    let sparse_seed = first_label_index_after(events, w_hat, labels::CHALLENGE_SPARSE_CHALLENGE)
        .expect("terminal transcript must squeeze sparse seed");
    let remainder =
        first_label_index_after(events, sparse_seed, labels::ABSORB_TERMINAL_W_REMAINDER)
            .expect("terminal transcript must absorb final-witness remainder");
    let alpha = first_label_index_after(events, remainder, labels::CHALLENGE_RING_SWITCH)
        .expect("terminal transcript must squeeze ring-switch alpha");
    let tau1 = first_label_index_after(events, alpha, labels::CHALLENGE_TAU1)
        .expect("terminal transcript must squeeze tau1");

    assert!(w_hat < sparse_seed, "w_hat must precede sparse seed");
    assert!(
        sparse_seed < remainder,
        "sparse seed must precede witness remainder"
    );
    assert!(remainder < alpha, "remainder must precede alpha");
    assert!(alpha < tau1, "alpha must precede tau1");
    assert!(
        events[w_hat..]
            .iter()
            .all(|event| event_label(event) != Some(labels::CHALLENGE_TAU0)),
        "terminal transcript window must not squeeze tau0"
    );
}

#[test]
fn preamble_separation_changes_first_challenge() {
    let mut left = AkitaTranscript::<F>::prover(labels::DOMAIN_AKITA_PROTOCOL, b"descriptor-a");
    let mut right = AkitaTranscript::<F>::prover(labels::DOMAIN_AKITA_PROTOCOL, b"descriptor-b");

    let left_challenge = left.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    let right_challenge = right.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);

    assert_ne!(left_challenge, right_challenge);
}

#[test]
fn event_stream_equality_small() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let num_vars = 10;
        let layout = OneHotCfg::commitment_layout(num_vars).expect("layout");
        let poly = make_onehot_poly(&layout, 0x5151);
        let point = random_point(num_vars, 0x6161);
        let opening = opening_from_poly(&poly, &point, &layout);

        let setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(num_vars, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, ONEHOT_D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");

        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let hints = vec![hint];

        let mut prover_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/onehot"));
        let proof = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            prove_input(
                &point,
                &poly_refs,
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

        let mut verifier_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/onehot"));
        <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("verify");

        prover_transcript.assert_smell_checks();
        verifier_transcript.assert_smell_checks();
        let prover_public = public_transcript_events(prover_transcript.events());
        let verifier_public = public_transcript_events(verifier_transcript.events());
        assert_eq!(prover_public, verifier_public);
        assert_terminal_event_order(&prover_public);
        assert!(matches!(
            prover_transcript.events().first(),
            Some(TranscriptEvent::Preamble { .. })
        ));
    });
}

#[test]
fn pr88_regression_missing_final_w_absorb_fails_smell_check() {
    let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/pr88"));
    transcript.bind_instance_bytes(b"descriptor");

    transcript.record_wire_use(labels::ABSORB_SUMCHECK_W, b"cleartext-final-w");
    transcript.append_bytes(labels::ABSORB_STOP_CONDITION, b"next-w-commitment");
    let _ = transcript.challenge_scalar(labels::CHALLENGE_TAU0);

    let errors = transcript.smell_check_errors();
    assert!(
        errors.iter().any(|error| error.contains("wire `ak/a/w`")),
        "expected wire coverage failure, got {errors:?}"
    );
}

#[test]
fn pr88_regression_mutated_final_w_after_absorb_fails_smell_check() {
    let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/pr88"));
    transcript.bind_instance_bytes(b"descriptor");

    transcript.append_bytes(labels::ABSORB_SUMCHECK_W, b"original-final-w");
    transcript.record_wire_use(labels::ABSORB_SUMCHECK_W, b"mutated-final-w");
    let _ = transcript.challenge_scalar(labels::CHALLENGE_TAU0);

    let errors = transcript.smell_check_errors();
    assert!(
        errors.iter().any(|error| error.contains("wire `ak/a/w`")),
        "expected wire coverage failure, got {errors:?}"
    );
}

#[test]
fn smell_checks_pass_for_matched_wire_absorb() {
    let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/wire"));
    transcript.bind_instance_bytes(b"descriptor");
    transcript.expect_wire_label(labels::ABSORB_SUMCHECK_W);
    transcript.record_wire_use(labels::ABSORB_SUMCHECK_W, b"cleartext-final-w");
    transcript.append_bytes(labels::ABSORB_SUMCHECK_W, b"cleartext-final-w");
    let _ = transcript.challenge_scalar(labels::CHALLENGE_TAU0);

    transcript.assert_smell_checks();
}
