#![allow(missing_docs)]

mod common;

use akita_field::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    sis::MAX_FOLD_GRIND_ATTEMPTS, AkitaBatchedProof, AkitaVerifierSetup, CommittedGroup,
};
use common::*;

type Scheme = AkitaCommitmentScheme<OneHotCfg>;

/// Production-scale fold-linf e2e is exercised at nv=20: still folds with
/// intermediate handles and fold-l∞ grinding, without the nv=28 CI cost.
const FOLD_LINF_E2E_NV: usize = 20;

fn bump_flat_ring_vec(flat: &mut akita_types::RingVec<F>) {
    let mut coeffs = flat.coeffs().to_vec();
    let first = coeffs
        .first_mut()
        .expect("tamper target must contain at least one coefficient");
    *first += F::one();
    *flat = akita_types::RingVec::from_coeffs(coeffs);
}

struct FoldLinfGrindFixture {
    proof: AkitaBatchedProof<F, F>,
    verifier_setup: AkitaVerifierSetup<F>,
    commitment: CommittedGroup<F>,
    point: Vec<F>,
    opening: F,
}

fn prove_fold_linf_grind_onehot_fixture(num_vars: usize, seed: u64) -> FoldLinfGrindFixture {
    let layout = OneHotCfg::select_schedule_for_opening(
        &akita_types::OpeningClaimsLayout::unit_one_hot(num_vars, 1, ONEHOT_K)
            .expect("singleton opening batch"),
    )
    .expect("layout")
    .schedule()
    .root
    .params
    .final_group
    .commitment
    .clone();
    let poly = make_onehot_poly(num_vars, seed);
    let point = random_point(num_vars, seed.wrapping_add(1));
    let opening = opening_from_poly_for_layout(&poly, &point, &layout);

    let setup = Scheme::setup_prover(num_vars, 1).expect("setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepare setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = Scheme::commit::<_, _>(
        &setup,
        std::slice::from_ref(&poly),
        &stack,
        akita_prover::GroupContext::scheduler_without_precommitted_groups(),
    )
    .expect("commit");

    let mut prover_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
    let proof = Scheme::batched_prove::<_, _, _>(
        &setup,
        prove_input::<OneHotCfg, _>(&point, &[&poly], &commitment, hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("prove");

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
    Scheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input::<OneHotCfg>(&point, &[opening], &commitment),
        BasisMode::Lagrange,
    )
    .expect("verify");

    FoldLinfGrindFixture {
        proof,
        verifier_setup,
        commitment,
        point,
        opening,
    }
}

#[test]
fn fold_linf_grind_onehot_e2e_prove_verify() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_fold_linf_grind_onehot_fixture(FOLD_LINF_E2E_NV, 0x51_51_00_01);
        for step in fixture.proof.nonterminal_folds() {
            assert!(
                step.fold_grind_nonce < MAX_FOLD_GRIND_ATTEMPTS,
                "grind nonce must stay within cap"
            );
        }
        assert!(fixture.proof.terminal.fold_grind_nonce < MAX_FOLD_GRIND_ATTEMPTS);
    });
}

#[test]
fn fold_grind_nonce_wire_roundtrip_and_oversized_nonce_rejected() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_fold_linf_grind_onehot_fixture(FOLD_LINF_E2E_NV, 0x51_51_00_02);
        let shape = fixture.proof.shape();
        let mut bytes = Vec::new();
        fixture
            .proof
            .serialize_compressed(&mut bytes)
            .expect("serialize proof");
        let mut roundtrip =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&bytes[..], &shape).expect("decode");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
        Scheme::batched_verify(
            &roundtrip,
            &fixture.verifier_setup,
            &mut verifier_transcript,
            verify_input::<OneHotCfg>(&fixture.point, &[fixture.opening], &fixture.commitment),
            BasisMode::Lagrange,
        )
        .expect("deserialized proof must verify");

        roundtrip.root.fold_grind_nonce = MAX_FOLD_GRIND_ATTEMPTS;
        roundtrip.terminal.fold_grind_nonce = MAX_FOLD_GRIND_ATTEMPTS;

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
        let err = Scheme::batched_verify(
            &roundtrip,
            &fixture.verifier_setup,
            &mut verifier_transcript,
            verify_input::<OneHotCfg>(&fixture.point, &[fixture.opening], &fixture.commitment),
            BasisMode::Lagrange,
        )
        .expect_err("oversized grind nonce must be rejected");
        assert!(
            matches!(err, AkitaError::InvalidProof)
                || matches!(err, AkitaError::InvalidInput(ref message) if message.contains("InvalidProof")),
            "oversized grind nonce returned {err:?}"
        );
    });
}

#[test]
fn fold_recursive_handle_tamper_rejected() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_fold_linf_grind_onehot_fixture(FOLD_LINF_E2E_NV, 0x51_51_00_04);
        let mut malformed = fixture.proof;
        let recursive = malformed
            .recursive_folds
            .first_mut()
            .expect("onehot fixture should include an intermediate fold");
        bump_flat_ring_vec(&mut recursive.opening_payload);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
        let result = Scheme::batched_verify(
            &malformed,
            &fixture.verifier_setup,
            &mut verifier_transcript,
            verify_input::<OneHotCfg>(&fixture.point, &[fixture.opening], &fixture.commitment),
            BasisMode::Lagrange,
        );
        assert_invalid_proof("tampered recursive fold handle", result);
    });
}

#[allow(dead_code)]
fn assert_invalid_proof<T: core::fmt::Debug>(label: &str, result: Result<T, AkitaError>) {
    match result {
        Err(AkitaError::InvalidProof) => {}
        Err(AkitaError::InvalidInput(msg)) if msg.contains("InvalidProof") => {}
        other => panic!("{label}: expected InvalidProof, got {other:?}"),
    }
}

#[cfg(feature = "logging-transcript")]
#[test]
fn logging_transcript_event_stream_equality_with_fold_linf_grind() {
    use akita_transcript::{labels, LoggingTranscript};

    init_rayon_pool();
    run_on_large_stack(|| {
        let num_vars = FOLD_LINF_E2E_NV;
        let layout = OneHotCfg::select_schedule_for_opening(
            &akita_types::OpeningClaimsLayout::unit_one_hot(num_vars, 1, ONEHOT_K)
                .expect("singleton opening batch"),
        )
        .expect("layout")
        .schedule()
        .root
        .params
        .final_group
        .commitment
        .clone();
        let poly = make_onehot_poly(num_vars, 0x61_61);
        let point = random_point(num_vars, 0x71_71);
        let opening = opening_from_poly_for_layout(&poly, &point, &layout);

        let setup = Scheme::setup_prover(num_vars, 1).expect("setup");
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&setup)
            .expect("prepare setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = Scheme::commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");

        let mut prover_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"fold-linf/logging"));
        let proof = Scheme::batched_prove::<_, _, _>(
            &setup,
            prove_input::<OneHotCfg, _>(&point, &[&poly], &commitment, hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

        let mut verifier_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"fold-linf/logging"));
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_E_HAT);
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_W_REMAINDER);
        Scheme::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<OneHotCfg>(&point, &[opening], &commitment),
            BasisMode::Lagrange,
        )
        .expect("verify");

        let prover_public = public_transcript_events(prover_transcript.events());
        let verifier_public = public_transcript_events(verifier_transcript.events());
        assert_eq!(
            prover_public, verifier_public,
            "prover and verifier public transcript events must match across fold grind reroll"
        );
    });
}
