#![allow(missing_docs)]
#![cfg(all(feature = "logging-transcript", not(feature = "zk")))]

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_transcript::{labels, AkitaTranscript, LoggingTranscript, Transcript, TranscriptEvent};
use akita_types::{AkitaBatchedProof, AkitaBatchedRootProof, AkitaProofStep, DirectWitnessProof};
use akita_verifier::CommitmentVerifier;
use common::*;

type Scheme = AkitaCommitmentScheme<ONEHOT_D, OneHotCfg>;

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
        assert_terminal_event_order_if_present(&prover_public)
            .expect("terminal transcript must absorb logical w_hat");
        assert!(matches!(
            prover_transcript.events().first(),
            Some(TranscriptEvent::Preamble { .. })
        ));
    });
}

fn ext_limb_label(label: &[u8], limb: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(label.len() + 12);
    out.extend_from_slice(label);
    out.push(0xff);
    out.extend_from_slice(&(limb as u64).to_le_bytes());
    out.extend_from_slice(b"ext");
    out
}

#[test]
#[should_panic(expected = "terminal transcript window must not squeeze tau0")]
fn terminal_event_order_rejects_extension_tau0_limb() {
    let events = vec![
        TranscriptEvent::Absorb {
            label: labels::ABSORB_TERMINAL_W_HAT.to_vec(),
            bytes_digest: [0; 32],
            bytes_len: 1,
        },
        TranscriptEvent::Squeeze {
            label: labels::CHALLENGE_SPARSE_CHALLENGE.to_vec(),
            len: 0,
        },
        TranscriptEvent::Absorb {
            label: labels::ABSORB_TERMINAL_W_REMAINDER.to_vec(),
            bytes_digest: [0; 32],
            bytes_len: 1,
        },
        TranscriptEvent::Squeeze {
            label: ext_limb_label(labels::CHALLENGE_RING_SWITCH, 0),
            len: 0,
        },
        TranscriptEvent::Squeeze {
            label: ext_limb_label(labels::CHALLENGE_TAU1, 0),
            len: 0,
        },
        TranscriptEvent::Squeeze {
            label: ext_limb_label(labels::CHALLENGE_TAU0, 0),
            len: 0,
        },
    ];

    let _ = assert_terminal_event_order_if_present(&events);
}

fn final_witness_mut(proof: &mut AkitaBatchedProof<F, F>) -> &mut DirectWitnessProof<F> {
    match &mut proof.root {
        AkitaBatchedRootProof::Terminal(terminal) => &mut terminal.final_witness,
        AkitaBatchedRootProof::Fold(_) => proof
            .steps
            .last_mut()
            .and_then(AkitaProofStep::as_terminal_mut)
            .map(|terminal| &mut terminal.final_witness)
            .expect("fold-rooted proof must end in a terminal step"),
        AkitaBatchedRootProof::Direct { .. } => {
            panic!("terminal tamper test requires a folded terminal proof")
        }
    }
}

fn assert_terminal_tamper_rejected(
    mutate: impl FnOnce(&mut DirectWitnessProof<F>) + Send + 'static,
) {
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

        let mut prover_transcript = AkitaTranscript::<F>::new(b"hardening/terminal-tamper");
        let mut proof = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
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

        mutate(final_witness_mut(&mut proof));

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"hardening/terminal-tamper");
        <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect_err("tampered terminal proof must reject");
    });
}

#[test]
fn terminal_final_witness_bit_tamper_rejects() {
    assert_terminal_tamper_rejected(|witness| {
        let DirectWitnessProof::PackedDigits(packed) = witness else {
            panic!("terminal witness should use packed digits");
        };
        packed.data[0] ^= 1;
    });
}

#[test]
fn terminal_final_witness_truncation_rejects() {
    assert_terminal_tamper_rejected(|witness| {
        let DirectWitnessProof::PackedDigits(packed) = witness else {
            panic!("terminal witness should use packed digits");
        };
        packed.num_elems -= 1;
    });
}

#[test]
fn terminal_final_witness_packed_payload_truncation_rejects() {
    assert_terminal_tamper_rejected(|witness| {
        let DirectWitnessProof::PackedDigits(packed) = witness else {
            panic!("terminal witness should use packed digits");
        };
        packed.data.pop();
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
