#![allow(missing_docs)]

mod common;

use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{AkitaBatchedProof, AkitaVerifierSetup, CommittedGroup, FOLD_RESPONSE_ATTEMPTS};
use common::*;

type Scheme = AkitaCommitmentScheme<OneHotCfg>;

/// Production-scale fold-linf e2e is exercised at nv=20 for root and terminal
/// grinding without the nv=28 CI cost. Recursive-handle tampering is covered
/// by the two-polynomial nv=20 fixture in `protocol_soundness`.
const FOLD_LINF_E2E_NV: usize = 20;

struct FoldLinfGrindFixture {
    proof: AkitaBatchedProof<F, F>,
    verifier_setup: AkitaVerifierSetup<F>,
    commitment: CommittedGroup<F>,
    point: Vec<F>,
    opening: F,
}

fn prove_fold_linf_grind_onehot_fixture(num_vars: usize, seed: u64) -> FoldLinfGrindFixture {
    let layout = OneHotCfg::resolve_catalog_row_for_opening(
        &akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch"),
    )
    .expect("layout")
    .schedule()
    .root
    .params
    .clone();
    let poly = make_onehot_poly(num_vars, seed);
    let point = random_point(num_vars, seed.wrapping_add(1));
    let opening = opening_from_poly_for_layout(
        &poly,
        &point,
        &layout.final_group_scalar().expect("scalar final group"),
        BasisMode::Lagrange,
    );

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
                step.fold_grind_nonce < FOLD_RESPONSE_ATTEMPTS,
                "grind nonce must stay within cap"
            );
        }
        assert!(fixture.proof.terminal.fold_grind_nonce < FOLD_RESPONSE_ATTEMPTS);
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

        roundtrip.root.fold_grind_nonce = FOLD_RESPONSE_ATTEMPTS;
        roundtrip.terminal.fold_grind_nonce = FOLD_RESPONSE_ATTEMPTS;

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

#[cfg(feature = "logging-transcript")]
#[test]
fn logging_transcript_event_stream_equality_with_fold_linf_grind() {
    use akita_transcript::{labels, LoggingTranscript};

    init_rayon_pool();
    run_on_large_stack(|| {
        let num_vars = FOLD_LINF_E2E_NV;
        let opening_batch =
            akita_types::OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch");
        let layout = OneHotCfg::resolve_catalog_row_for_opening(&opening_batch)
            .expect("layout")
            .schedule()
            .root
            .params
            .final_group();
        let poly = make_onehot_poly(num_vars, 0x61_61);
        let point = random_point(num_vars, 0x71_71);
        let opening = opening_from_poly_for_layout(&poly, &point, &layout, BasisMode::Lagrange);

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
