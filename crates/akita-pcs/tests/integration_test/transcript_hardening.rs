
use akita_prover::{ComputeBackendSetup, CpuBackend};

#[allow(dead_code)]
#[path = "common/mod.rs"]
mod common;

use akita_field::CanonicalField;
use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{
    ext_limb_label, labels, AkitaTranscript, LoggingTranscript, Transcript, TranscriptEvent,
};
use akita_types::{
    terminal_witness_segment_layout, AkitaBatchedProof, AkitaBatchedProofShape,
    AkitaBatchedRootProof, AkitaLevelProof, CleartextWitnessProof, CleartextWitnessShape,
    TerminalWitnessSegmentLayout,
};
use common::*;

type Scheme = AkitaCommitmentScheme<OneHotCfg>;

/// Singleton onehot `num_vars` large enough that `batched_prove` keeps a root
/// fold and segment-typed terminal direct witness. Smaller values (e.g. 10)
/// fall back to root-direct zero-fold and never emit terminal transcript wire
/// labels.
const TRANSCRIPT_HARDENING_NUM_VARS: usize = 20;

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
    run_on_large_stack(move || {
        let num_vars = TRANSCRIPT_HARDENING_NUM_VARS;
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let poly = make_onehot_poly(&layout, 0x5151);
        let point = random_point(num_vars, 0x6161);
        let opening = opening_from_poly(&poly, &point, &layout);

        let setup = Scheme::setup_prover(num_vars, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = Scheme::setup_verifier(&setup);
        let (commitment, hint) =
            Scheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let hints = vec![hint];

        let mut prover_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/onehot"));
        let proof = Scheme::batched_prove(
            &setup,
            prove_input(
                &point,
                &poly_refs,
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");

        let mut verifier_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/onehot"));
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_E_HAT);
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_W_REMAINDER);
        Scheme::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("verify");

        prover_transcript.assert_smell_checks();
        verifier_transcript.assert_smell_checks();
        let prover_public = public_transcript_events(prover_transcript.events());
        let verifier_public = public_transcript_events(verifier_transcript.events());
        assert_eq!(prover_public, verifier_public);
        assert_terminal_event_order_if_present(&prover_public)
            .expect("terminal transcript must absorb logical e_hat");
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
    for (expected, events) in [
        (
            "terminal transcript window must not squeeze tau0",
            vec![
                absorb_event(labels::ABSORB_TERMINAL_E_HAT),
                squeeze_event(labels::CHALLENGE_SPARSE_CHALLENGE),
                absorb_event(labels::ABSORB_TERMINAL_W_REMAINDER),
                squeeze_event(ext_limb_label(labels::CHALLENGE_RING_SWITCH, 0)),
                squeeze_event(ext_limb_label(labels::CHALLENGE_TAU1, 0)),
                squeeze_event(labels::CHALLENGE_SUMCHECK_ROUND),
                squeeze_event(ext_limb_label(labels::CHALLENGE_TAU0, 0)),
            ],
        ),
        (
            "terminal stage-2 sumcheck must not precede tau1",
            vec![
                absorb_event(labels::ABSORB_TERMINAL_E_HAT),
                squeeze_event(labels::CHALLENGE_SPARSE_CHALLENGE),
                absorb_event(labels::ABSORB_TERMINAL_W_REMAINDER),
                squeeze_event(labels::CHALLENGE_RING_SWITCH),
                squeeze_event(labels::CHALLENGE_SUMCHECK_ROUND),
                squeeze_event(labels::CHALLENGE_TAU1),
                squeeze_event(labels::CHALLENGE_SUMCHECK_ROUND),
            ],
        ),
        (
            "terminal alpha must not precede witness remainder",
            vec![
                absorb_event(labels::ABSORB_TERMINAL_E_HAT),
                squeeze_event(labels::CHALLENGE_SPARSE_CHALLENGE),
                squeeze_event(labels::CHALLENGE_RING_SWITCH),
                absorb_event(labels::ABSORB_TERMINAL_W_REMAINDER),
                squeeze_event(labels::CHALLENGE_RING_SWITCH),
                squeeze_event(labels::CHALLENGE_TAU1),
                squeeze_event(labels::CHALLENGE_SUMCHECK_ROUND),
            ],
        ),
        (
            "terminal tau1 must not precede alpha",
            vec![
                absorb_event(labels::ABSORB_TERMINAL_E_HAT),
                squeeze_event(labels::CHALLENGE_SPARSE_CHALLENGE),
                absorb_event(labels::ABSORB_TERMINAL_W_REMAINDER),
                squeeze_event(labels::CHALLENGE_TAU1),
                squeeze_event(labels::CHALLENGE_RING_SWITCH),
                squeeze_event(labels::CHALLENGE_TAU1),
                squeeze_event(labels::CHALLENGE_SUMCHECK_ROUND),
            ],
        ),
        (
            "terminal tau1 limbs must be contiguous before stage-2 sumcheck",
            vec![
                absorb_event(labels::ABSORB_TERMINAL_E_HAT),
                squeeze_event(labels::CHALLENGE_SPARSE_CHALLENGE),
                absorb_event(labels::ABSORB_TERMINAL_W_REMAINDER),
                squeeze_event(ext_limb_label(labels::CHALLENGE_RING_SWITCH, 0)),
                squeeze_event(ext_limb_label(labels::CHALLENGE_TAU1, 0)),
                squeeze_event(labels::CHALLENGE_SUMCHECK_ROUND),
                squeeze_event(ext_limb_label(labels::CHALLENGE_TAU1, 1)),
            ],
        ),
    ] {
        assert_terminal_order_panics(events, expected);
    }
}

fn final_witness_mut(proof: &mut AkitaBatchedProof<F, F>) -> &mut CleartextWitnessProof<F> {
    match &mut proof.root {
        AkitaBatchedRootProof::Terminal(terminal) => terminal
            .stage2
            .final_witness_mut()
            .expect("terminal root proof must carry terminal stage-2 proof"),
        AkitaBatchedRootProof::Fold(_) => proof
            .steps
            .last_mut()
            .and_then(AkitaLevelProof::as_terminal_mut)
            .map(|terminal| {
                terminal
                    .stage2_mut()
                    .final_witness_mut()
                    .expect("terminal step proof must carry terminal stage-2 proof")
            })
            .expect("fold-rooted proof must end in a terminal step"),
        AkitaBatchedRootProof::ZeroFold { .. } => {
            panic!("terminal tamper test requires a folded terminal proof")
        }
    }
}

#[derive(Clone, Copy)]
enum TerminalTamper {
    EHatDigit,
    RemainderDigit,
    WitnessLen,
    PackedPayload,
}

impl TerminalTamper {
    fn apply(self, witness: &mut CleartextWitnessProof<F>, layout: TerminalWitnessSegmentLayout) {
        match witness {
            CleartextWitnessProof::SegmentTyped(segment) => match self {
                Self::EHatDigit => {
                    let mut coeffs = segment.e_fields.coeffs().to_vec();
                    let first = coeffs
                        .first_mut()
                        .expect("segment-typed terminal must carry e field coeffs");
                    *first += F::one();
                    segment.e_fields = akita_types::RingVec::from_coeffs(coeffs);
                }
                Self::RemainderDigit => {
                    segment.z_payload[0] ^= 1;
                }
                Self::WitnessLen => {
                    segment.layout.logical_num_elems =
                        segment.layout.logical_num_elems.saturating_sub(1);
                }
                Self::PackedPayload => {
                    segment.z_payload.pop();
                }
            },
            CleartextWitnessProof::FieldElements(_) => {
                panic!("terminal tamper test does not cover field-element witnesses");
            }
        }
    }
}

fn assert_terminal_tamper_rejected_at_num_vars(num_vars: usize, tamper: TerminalTamper) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let poly = make_onehot_poly(&layout, 0x5151);
        let point = random_point(num_vars, 0x6161);
        let opening = opening_from_poly(&poly, &point, &layout);

        let setup = Scheme::setup_prover(num_vars, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = Scheme::setup_verifier(&setup);
        let (commitment, hint) =
            Scheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"hardening/terminal-tamper");
        let mut proof = Scheme::batched_prove(
            &setup,
            prove_input(
                &point,
                &poly_refs,
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");
        let terminal_layout = terminal_witness_segment_layout(&layout, 1, 1, F::modulus_bits())
            .expect("terminal layout");

        tamper.apply(final_witness_mut(&mut proof), terminal_layout);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"hardening/terminal-tamper");
        Scheme::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect_err("tampered terminal proof must reject");
    });
}

fn assert_terminal_tamper_rejected(tamper: TerminalTamper) {
    assert_terminal_tamper_rejected_at_num_vars(TRANSCRIPT_HARDENING_NUM_VARS, tamper);
}

#[test]
fn terminal_final_witness_tamper_rejects() {
    for tamper in [
        TerminalTamper::EHatDigit,
        TerminalTamper::RemainderDigit,
        TerminalTamper::WitnessLen,
        TerminalTamper::PackedPayload,
    ] {
        assert_terminal_tamper_rejected(tamper);
    }
}

fn terminal_shape_final_witness_mut(
    shape: &mut AkitaBatchedProofShape,
) -> &mut CleartextWitnessShape {
    match shape {
        AkitaBatchedProofShape::Terminal(terminal) => &mut terminal.final_witness,
        AkitaBatchedProofShape::Fold { step_shapes, .. } => step_shapes
            .last_mut()
            .and_then(|step| match step {
                akita_types::AkitaProofStepShape::Intermediate(_) => None,
                akita_types::AkitaProofStepShape::Terminal(terminal) => {
                    Some(&mut terminal.final_witness)
                }
            })
            .expect("fold-rooted proof must end in a terminal shape"),
        AkitaBatchedProofShape::ZeroFold { .. } => {
            panic!("terminal shape test requires a folded terminal proof")
        }
    }
}

#[test]
fn terminal_direct_witness_shape_mismatch_rejects_deserialization() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let num_vars = TRANSCRIPT_HARDENING_NUM_VARS;
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let poly = make_onehot_poly(&layout, 0x5151);
        let point = random_point(num_vars, 0x6161);

        let setup = Scheme::setup_prover(num_vars, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let (commitment, hint) =
            Scheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("commit");

        let poly_refs = [&poly];
        let mut prover_transcript = AkitaTranscript::<F>::new(b"hardening/shape-mismatch");
        let proof = Scheme::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitment, hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");

        let mut bytes = Vec::new();
        proof
            .serialize_compressed(&mut bytes)
            .expect("serialize proof");
        let mut bad_shape = proof.shape();
        let CleartextWitnessShape::SegmentTyped(shape) =
            terminal_shape_final_witness_mut(&mut bad_shape)
        else {
            panic!("terminal witness should be segment-typed");
        };
        // Segment-typed tails admit exact `z` payloads up to the scheduled
        // upper bound; a *tighter* budget than the encoded payload must reject.
        shape.z_payload_bytes = shape.z_payload_bytes.saturating_sub(1);

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
