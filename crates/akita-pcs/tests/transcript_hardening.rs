#![allow(missing_docs)]
#![cfg(feature = "logging-transcript")]

use akita_prover::{ComputeBackendSetup, CpuBackend};

mod common;

use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{
    ext_limb_label, labels, AkitaTranscript, LoggingTranscript, Transcript, TranscriptEvent,
};
use akita_types::{
    AkitaBatchedProof, AkitaBatchedProofShape, NextWitnessBinding, RingVec, TerminalResponse,
    TerminalResponseShape,
};
use common::*;
use jolt_field::{One, Zero};

/// Small singleton onehot instance that exercises a folded path ending in a
/// terminal response.
///
/// The generated schedule may insert or remove recursive folds as planner
/// policy evolves. The test below therefore identifies the final handoff from
/// transcript structure instead of treating the current fold count as a
/// protocol epoch.
const TRANSCRIPT_HARDENING_NUM_VARS: usize = 14;

#[test]
fn preamble_separation_changes_first_challenge() {
    let mut left = AkitaTranscript::<F>::prover(labels::DOMAIN_AKITA_PROTOCOL, b"descriptor-a");
    let mut right = AkitaTranscript::<F>::prover(labels::DOMAIN_AKITA_PROTOCOL, b"descriptor-b");

    let left_challenge = left.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    let right_challenge = right.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);

    assert_ne!(left_challenge, right_challenge);
}

/// Check structural transcript invariants across a complete folded proof.
///
/// This test intentionally does not freeze proof bytes, event counts, or the
/// number of recursive folds. It requires the prover and verifier to replay the
/// same public events, runs the logging smell checks, and verifies the semantic
/// handoff into the terminal level: the last predecessor binds `t`, squeezes
/// its ring-switch challenge, and the terminal level rebinds the same canonical
/// `t` bytes before absorbing its terminal witness data.
#[test]
fn event_stream_equality_small() {
    init_rayon_pool();
    run_on_large_stack(move || {
        let scheme = load_workspace_scheme::<OneHotCfg>().expect("workspace schedule catalog");
        let num_vars = TRANSCRIPT_HARDENING_NUM_VARS;
        let opening_batch =
            akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch");
        let layout = scheme
            .schedules()
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                opening_batch
                    .root_final_group_layout()
                    .expect("singleton group layout"),
            ))
            .expect("layout")
            .schedule()
            .root
            .params
            .final_group();
        let poly = make_onehot_poly::<OneHotCfg>(num_vars, 0x5151);
        let point = random_point(num_vars, 0x6161);
        let opening = opening_from_poly_for_layout(&poly, &point, &layout, BasisMode::Lagrange);

        let setup = scheme.setup_prover(num_vars, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");

        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let hints = vec![hint];

        let mut prover_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/onehot"));
        let proof = scheme
            .batched_prove(
                &setup,
                prove_input::<OneHotCfg, _>(
                    &point,
                    &poly_refs,
                    &commitments[0],
                    hints.into_iter().next().unwrap(),
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");
        let mut verifier_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/onehot"));
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_E_HAT);
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_W_REMAINDER);
        scheme
            .batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input::<OneHotCfg>(&point, &openings, &commitments[0], scheme.schedules()),
                BasisMode::Lagrange,
            )
            .expect("verify");

        prover_transcript.assert_smell_checks();
        verifier_transcript.assert_smell_checks();
        let prover_public = public_transcript_events(prover_transcript.events());
        let verifier_public = public_transcript_events(verifier_transcript.events());
        assert_eq!(prover_public, verifier_public);
        assert_claim_batching_follows_opening_payload(&prover_public);
        let terminal_e = assert_terminal_event_order_if_present(&prover_public)
            .expect("terminal transcript must absorb logical e_hat");
        // Anchor at the terminal window and walk backward. Earlier recursive
        // folds contain the same labels, so forward searches would select the
        // wrong transition whenever the generated schedule gains a level.
        let terminal_t = prover_public[..terminal_e]
            .iter()
            .rposition(|event| {
                event_label(event).is_some_and(|candidate| candidate == labels::ABSORB_COMMITMENT)
            })
            .expect("terminal must rebind carried t as its current state");
        let predecessor_alpha = prover_public[..terminal_t]
            .iter()
            .rposition(|event| {
                event_label(event).is_some_and(|candidate| {
                    candidate == labels::CHALLENGE_RING_SWITCH
                        || akita_transcript::is_ext_limb_label(
                            candidate,
                            labels::CHALLENGE_RING_SWITCH,
                        )
                })
            })
            .expect("predecessor must squeeze ring-switch challenge after binding t");
        let predecessor_t = prover_public[..predecessor_alpha]
            .iter()
            .rposition(|event| {
                event_label(event)
                    .is_some_and(|candidate| candidate == labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING)
            })
            .expect("predecessor must bind terminal t");
        assert!(
            predecessor_t < predecessor_alpha
                && predecessor_alpha < terminal_t
                && terminal_t < terminal_e,
            "terminal t handoff must precede predecessor alpha and terminal e"
        );
        let TranscriptEvent::Absorb {
            bytes_digest: predecessor_digest,
            bytes_len: predecessor_len,
            ..
        } = &prover_public[predecessor_t]
        else {
            unreachable!()
        };
        let TranscriptEvent::Absorb {
            bytes_digest: terminal_digest,
            bytes_len: terminal_len,
            ..
        } = &prover_public[terminal_t]
        else {
            unreachable!()
        };
        assert_eq!(
            (predecessor_digest, predecessor_len),
            (terminal_digest, terminal_len),
            "predecessor and terminal must bind identical canonical t bytes"
        );
        assert!(matches!(
            prover_transcript.events().first(),
            Some(TranscriptEvent::Preamble { .. })
        ));
    });
}

fn absorb_event(label: &[u8]) -> TranscriptEvent {
    TranscriptEvent::Absorb {
        label: label.to_vec(),
        bytes_digest: [0; 32],
        bytes_len: 1,
    }
}

fn squeeze_event(label: impl Into<Vec<u8>>) -> TranscriptEvent {
    TranscriptEvent::Squeeze {
        label: label.into(),
        len: 0,
    }
}

fn assert_terminal_order_panics(events: Vec<TranscriptEvent>, expected: &str) {
    let panic = std::panic::catch_unwind(|| {
        let _ = assert_terminal_event_order_if_present(&events);
    })
    .expect_err("malformed terminal transcript order should panic");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("<non-string panic>");
    assert!(
        message.contains(expected),
        "expected panic containing `{expected}`, got `{message}`"
    );
}

#[test]
fn terminal_event_order_rejects_malformed_windows() {
    for (expected, forbidden) in [
        (
            "terminal must not squeeze alpha",
            labels::CHALLENGE_RING_SWITCH,
        ),
        ("terminal must not squeeze tau1", labels::CHALLENGE_TAU1),
        (
            "terminal must not squeeze stage-2 rounds",
            labels::CHALLENGE_SUMCHECK_ROUND,
        ),
        (
            "terminal must not squeeze stage-2 batching",
            labels::CHALLENGE_SUMCHECK_BATCH,
        ),
        ("terminal must not squeeze tau0", labels::CHALLENGE_TAU0),
    ] {
        let events = vec![
            absorb_event(labels::ABSORB_TERMINAL_E_HAT),
            squeeze_event(labels::CHALLENGE_SPARSE_CHALLENGE),
            absorb_event(labels::ABSORB_TERMINAL_W_REMAINDER),
            squeeze_event(ext_limb_label(forbidden, 0)),
        ];
        assert_terminal_order_panics(events, expected);
    }
}

fn terminal_response_mut(proof: &mut AkitaBatchedProof<F, F>) -> &mut TerminalResponse<F> {
    proof.terminal.terminal_response_mut()
}

#[derive(Clone, Copy)]
enum ProofTamper {
    EHatDigit,
    RemainderDigit,
    WitnessLen,
    PackedPayload,
    ExtraZPayload,
    OversizedRootV,
    WrongOutgoingBinding,
}

impl ProofTamper {
    fn apply(self, proof: &mut AkitaBatchedProof<F, F>) {
        match self {
            Self::EHatDigit => {
                let witness = terminal_response_mut(proof);
                let mut coeffs = witness.e_fields.coeffs().to_vec();
                let first = coeffs
                    .first_mut()
                    .expect("terminal response must carry e field coeffs");
                *first += F::one();
                witness.e_fields = RingVec::from_coeffs(coeffs);
            }
            Self::RemainderDigit => terminal_response_mut(proof).z_payloads[0][0] ^= 1,
            Self::WitnessLen => {
                let witness = terminal_response_mut(proof);
                witness.layout.logical_num_elems =
                    witness.layout.logical_num_elems.saturating_sub(1);
            }
            Self::PackedPayload => {
                terminal_response_mut(proof).z_payloads[0].pop();
            }
            Self::ExtraZPayload => terminal_response_mut(proof).z_payloads.push(vec![0]),
            Self::OversizedRootV => {
                let mut coeffs = proof.root.opening_payload.coeffs().to_vec();
                coeffs.push(F::zero());
                proof.root.opening_payload = RingVec::from_coeffs(coeffs);
            }
            Self::WrongOutgoingBinding => {
                proof.root.stage2.next_witness_binding =
                    match &proof.root.stage2.next_witness_binding {
                        NextWitnessBinding::OuterPayload(_) => {
                            NextWitnessBinding::TerminalInnerState
                        }
                        NextWitnessBinding::TerminalInnerState => {
                            NextWitnessBinding::OuterPayload(RingVec::from_coeffs(vec![F::zero()]))
                        }
                    };
            }
        }
    }
}

fn assert_proof_tamper_rejected_at_num_vars(num_vars: usize, tamper: ProofTamper) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let scheme = load_workspace_scheme::<OneHotCfg>().expect("workspace schedule catalog");
        let opening_batch =
            akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch");
        let layout = scheme
            .schedules()
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                opening_batch
                    .root_final_group_layout()
                    .expect("singleton group layout"),
            ))
            .expect("layout")
            .schedule()
            .root
            .params
            .final_group();
        let poly = make_onehot_poly::<OneHotCfg>(num_vars, 0x5151);
        let point = random_point(num_vars, 0x6161);
        let opening = opening_from_poly_for_layout(&poly, &point, &layout, BasisMode::Lagrange);

        let setup = scheme.setup_prover(num_vars, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");

        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"hardening/terminal-tamper");
        let mut proof = scheme
            .batched_prove(
                &setup,
                prove_input::<OneHotCfg, _>(
                    &point,
                    &poly_refs,
                    &commitments[0],
                    hints.into_iter().next().unwrap(),
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");
        tamper.apply(&mut proof);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"hardening/terminal-tamper");
        scheme
            .batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input::<OneHotCfg>(&point, &openings, &commitments[0], scheme.schedules()),
                BasisMode::Lagrange,
            )
            .expect_err("tampered terminal proof must reject");
    });
}

fn assert_proof_tamper_rejected(tamper: ProofTamper) {
    assert_proof_tamper_rejected_at_num_vars(TRANSCRIPT_HARDENING_NUM_VARS, tamper);
}

#[test]
fn malformed_proof_carriers_reject_before_replay() {
    for tamper in [
        ProofTamper::EHatDigit,
        ProofTamper::RemainderDigit,
        ProofTamper::WitnessLen,
        ProofTamper::PackedPayload,
        ProofTamper::ExtraZPayload,
        ProofTamper::OversizedRootV,
        ProofTamper::WrongOutgoingBinding,
    ] {
        assert_proof_tamper_rejected(tamper);
    }
}

fn terminal_shape_terminal_response_mut(
    shape: &mut AkitaBatchedProofShape,
) -> &mut TerminalResponseShape {
    &mut shape.terminal.terminal_response
}

#[test]
fn terminal_direct_witness_shape_mismatch_rejects_deserialization() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = load_workspace_scheme::<OneHotCfg>().expect("workspace schedule catalog");
        let num_vars = TRANSCRIPT_HARDENING_NUM_VARS;
        let poly = make_onehot_poly::<OneHotCfg>(num_vars, 0x5151);
        let point = random_point(num_vars, 0x6161);

        let setup = scheme.setup_prover(num_vars, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");

        let poly_refs = [&poly];
        let mut prover_transcript = AkitaTranscript::<F>::new(b"hardening/shape-mismatch");
        let proof = scheme
            .batched_prove(
                &setup,
                prove_input::<OneHotCfg, _>(
                    &point,
                    &poly_refs,
                    &commitment,
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("prove");

        let mut bytes = Vec::new();
        proof
            .serialize_compressed(&mut bytes)
            .expect("serialize proof");
        let mut bad_shape = proof.shape();
        let shape = terminal_shape_terminal_response_mut(&mut bad_shape);
        // Segment-typed tails admit exact `z` payloads up to the scheduled
        // upper bound; a *tighter* budget than the encoded payload must reject.
        shape.layout.groups[0].z_payload_bytes = 0;

        AkitaBatchedProof::<F, F>::deserialize_compressed(&bytes[..], &bad_shape)
            .expect_err("terminal direct-witness shape mismatch must reject");
    });
}

#[test]
fn pr88_regression_missing_final_w_absorb_fails_smell_check() {
    let mut transcript = LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/pr88"));
    transcript.bind_instance_bytes(b"descriptor");

    transcript.record_wire_use(
        labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING,
        b"cleartext-final-w",
    );
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

    transcript.append_bytes(
        labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING,
        b"original-final-w",
    );
    transcript.record_wire_use(
        labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING,
        b"mutated-final-w",
    );
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
    transcript.expect_wire_label(labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING);
    transcript.record_wire_use(
        labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING,
        b"cleartext-final-w",
    );
    transcript.append_bytes(
        labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING,
        b"cleartext-final-w",
    );
    let _ = transcript.challenge_scalar(labels::CHALLENGE_TAU0);

    transcript.assert_smell_checks();
}
