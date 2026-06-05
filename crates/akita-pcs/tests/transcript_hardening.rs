#![allow(missing_docs)]
#![cfg(all(feature = "logging-transcript", not(feature = "zk")))]

use akita_prover::{ComputeBackendSetup, CpuBackend};

mod common;

use akita_field::CanonicalField;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{
    ext_limb_label, labels, AkitaTranscript, LoggingTranscript, Transcript, TranscriptEvent,
};
use akita_types::{
    terminal_witness_segment_layout, AkitaBatchedProof, AkitaBatchedProofShape,
    AkitaBatchedRootProof, AkitaProofStep, CleartextWitnessProof, CleartextWitnessShape,
    PackedDigits, TerminalWitnessSegmentLayout,
};
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
    run_on_large_stack(move || {
        let num_vars = 10;
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::ClaimIncidenceSummary::same_point(num_vars, 1)
                .expect("singleton incidence"),
        )
        .expect("layout");
        let poly = make_onehot_poly(&layout, 0x5151);
        let point = random_point(num_vars, 0x6161);
        let opening = opening_from_poly(&poly, &point, &layout);

        let mut setup =
            <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(num_vars, 1, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, hint) = <Scheme as CommitmentProver<F, ONEHOT_D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            std::slice::from_ref(&poly),
        )
        .expect("commit");

        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let hints = vec![hint];

        let mut prover_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/onehot"));
        let proof = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(
                &point,
                &poly_refs,
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");

        let mut verifier_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"hardening/onehot"));
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_E_HAT);
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_W_REMAINDER);
        <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
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
        AkitaBatchedRootProof::Terminal(terminal) => &mut terminal.final_witness,
        AkitaBatchedRootProof::Fold(_) => proof
            .steps
            .last_mut()
            .and_then(AkitaProofStep::as_terminal_mut)
            .map(|terminal| &mut terminal.final_witness)
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
        let packed = packed_digits_mut(witness);
        match self {
            Self::EHatDigit => mutate_packed_digit(packed, layout.e_hat_digit_offset),
            Self::RemainderDigit => {
                let e_hat_end = layout.e_hat_digit_end().expect("terminal range");
                let remainder_idx = if layout.e_hat_digit_offset > 0 {
                    0
                } else {
                    e_hat_end
                };
                assert!(
                    remainder_idx < packed.num_elems,
                    "terminal tamper corpus must include a non-empty remainder"
                );
                mutate_packed_digit(packed, remainder_idx);
            }
            Self::WitnessLen => packed.num_elems -= 1,
            Self::PackedPayload => {
                packed.data.pop();
            }
        }
    }
}

fn assert_terminal_tamper_rejected_at_num_vars(num_vars: usize, tamper: TerminalTamper) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::ClaimIncidenceSummary::same_point(num_vars, 1)
                .expect("singleton incidence"),
        )
        .expect("layout");
        let poly = make_onehot_poly(&layout, 0x5151);
        let point = random_point(num_vars, 0x6161);
        let opening = opening_from_poly(&poly, &point, &layout);

        let mut setup =
            <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(num_vars, 1, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let verifier_setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, hint) = <Scheme as CommitmentProver<F, ONEHOT_D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            std::slice::from_ref(&poly),
        )
        .expect("commit");

        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"hardening/terminal-tamper");
        let mut proof = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(
                &point,
                &poly_refs,
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");
        let terminal_layout = terminal_witness_segment_layout(&layout, 1, 1, F::modulus_bits())
            .expect("terminal layout");

        tamper.apply(final_witness_mut(&mut proof), terminal_layout);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"hardening/terminal-tamper");
        <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
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
    assert_terminal_tamper_rejected_at_num_vars(10, tamper);
}

fn packed_digits_mut(witness: &mut CleartextWitnessProof<F>) -> &mut PackedDigits {
    let CleartextWitnessProof::PackedDigits(packed) = witness else {
        panic!("terminal witness should use packed digits");
    };
    packed
}

fn mutate_packed_digit(packed: &mut PackedDigits, idx: usize) {
    let mut digits = (0..packed.num_elems)
        .map(|digit| packed.digit_at(digit).expect("packed digit"))
        .collect::<Vec<_>>();
    let digit = digits.get_mut(idx).expect("digit index in range");
    *digit = if *digit == -1 { 0 } else { -1 };
    *packed = PackedDigits::from_i8_digits(&digits, packed.bits_per_elem);
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
        let num_vars = 10;
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::ClaimIncidenceSummary::same_point(num_vars, 1)
                .expect("singleton incidence"),
        )
        .expect("layout");
        let poly = make_onehot_poly(&layout, 0x5151);
        let point = random_point(num_vars, 0x6161);

        let mut setup =
            <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(num_vars, 1, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let (commitment, hint) = <Scheme as CommitmentProver<F, ONEHOT_D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            std::slice::from_ref(&poly),
        )
        .expect("commit");

        let poly_refs = [&poly];
        let mut prover_transcript = AkitaTranscript::<F>::new(b"hardening/shape-mismatch");
        let proof = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(&point, &poly_refs, &commitment, hint),
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
        let CleartextWitnessShape::PackedDigits((num_elems, _bits_per_elem)) =
            terminal_shape_final_witness_mut(&mut bad_shape)
        else {
            panic!("terminal witness should use packed digits");
        };
        *num_elems = num_elems
            .checked_add(1)
            .expect("terminal witness shape overflow");

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
