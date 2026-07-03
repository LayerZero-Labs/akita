#![allow(missing_docs)]

mod common;

use akita_field::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{sis::MAX_FOLD_GRIND_ATTEMPTS, AkitaBatchedProof, AkitaBatchedRootProof};
use akita_verifier::CommitmentVerifier;
use common::*;

type Scheme = AkitaCommitmentScheme<ONEHOT_D, OneHotCfg>;

/// Production-scale fold-linf e2e is exercised at nv=20: still folds with
/// intermediate handles and TailBoundWithGrind, without the nv=28 CI cost.
const FOLD_LINF_E2E_NV: usize = 20;

struct TailBoundGrindFixture {
    proof: AkitaBatchedProof<F, F>,
    verifier_setup: <Scheme as CommitmentProver<F, ONEHOT_D>>::VerifierSetup,
    commitment: <Scheme as CommitmentProver<F, ONEHOT_D>>::Commitment,
    point: Vec<F>,
    opening: F,
}

fn prove_tail_bound_with_grind_onehot_fixture(num_vars: usize, seed: u64) -> TailBoundGrindFixture {
    let layout = OneHotCfg::get_params_for_batched_commitment(
        &akita_types::OpeningBatchShape::new(num_vars, 1).expect("singleton opening batch"),
    )
    .expect("layout");
    assert_eq!(
        layout.fold_witness_linf_cap_policy(),
        akita_types::sis::FoldWitnessLinfCapPolicy::TailBoundWithGrind
    );

    let poly = make_onehot_poly(&layout, seed);
    let point = random_point(num_vars, seed.wrapping_add(1));
    let opening = opening_from_poly(&poly, &point, &layout);

    let setup =
        <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(num_vars, 1).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepare setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
    let (commitment, hint) = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_commit(
        &setup,
        std::slice::from_ref(&poly),
        &stack,
    )
    .expect("commit");

    let mut prover_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
    let proof = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
        &setup,
        prove_input(&point, &[&poly], &commitment, hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("prove");

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
    <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&point, &[opening], &commitment),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("verify");

    TailBoundGrindFixture {
        proof,
        verifier_setup,
        commitment,
        point,
        opening,
    }
}

#[test]
fn tail_bound_with_grind_onehot_e2e_prove_verify() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_tail_bound_with_grind_onehot_fixture(FOLD_LINF_E2E_NV, 0x51_51_00_01);
        assert!(
            matches!(fixture.proof.root, AkitaBatchedRootProof::Fold(_)),
            "expected a folded root proof"
        );
        for step in fixture.proof.fold_levels() {
            assert!(
                step.fold_grind_nonce() < MAX_FOLD_GRIND_ATTEMPTS,
                "grind nonce must stay within cap"
            );
        }
    });
}

#[test]
fn fold_grind_nonce_wire_roundtrip_and_oversized_nonce_rejected() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_tail_bound_with_grind_onehot_fixture(FOLD_LINF_E2E_NV, 0x51_51_00_02);
        let shape = fixture.proof.shape();
        let mut bytes = Vec::new();
        fixture
            .proof
            .serialize_compressed(&mut bytes)
            .expect("serialize proof");
        let mut roundtrip =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&bytes[..], &shape).expect("decode");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
        <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &roundtrip,
            &fixture.verifier_setup,
            &mut verifier_transcript,
            verify_input(&fixture.point, &[fixture.opening], &fixture.commitment),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("deserialized proof must verify");

        if let AkitaBatchedRootProof::Fold(fold) = &mut roundtrip.root {
            fold.fold_grind_nonce = MAX_FOLD_GRIND_ATTEMPTS;
        }
        if let Some(akita_types::AkitaLevelProof::Terminal {
            fold_grind_nonce, ..
        }) = roundtrip.steps.last_mut()
        {
            *fold_grind_nonce = MAX_FOLD_GRIND_ATTEMPTS;
        }

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
        let err = <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &roundtrip,
            &fixture.verifier_setup,
            &mut verifier_transcript,
            verify_input(&fixture.point, &[fixture.opening], &fixture.commitment),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect_err("oversized grind nonce must be rejected");
        assert!(matches!(err, AkitaError::InvalidProof));
    });
}

#[cfg(feature = "logging-transcript")]
#[test]
fn logging_transcript_event_stream_equality_tail_bound_with_grind() {
    use akita_transcript::{labels, LoggingTranscript, Transcript};

    init_rayon_pool();
    run_on_large_stack(|| {
        let num_vars = FOLD_LINF_E2E_NV;
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(num_vars, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let poly = make_onehot_poly(&layout, 0x61_61);
        let point = random_point(num_vars, 0x71_71);
        let opening = opening_from_poly(&poly, &point, &layout);

        let setup =
            <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(num_vars, 1).expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepare setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, hint) = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_commit(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
        )
        .expect("commit");

        let mut prover_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"fold-linf/logging"));
        let proof = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            prove_input(&point, &[&poly], &commitment, hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");

        let mut verifier_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"fold-linf/logging"));
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_E_HAT);
        verifier_transcript.expect_wire_label(labels::ABSORB_TERMINAL_W_REMAINDER);
        <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &[opening], &commitment),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("verify");

        assert_eq!(
            prover_transcript.events(),
            verifier_transcript.events(),
            "prover and verifier transcript events must match across fold grind reroll"
        );
    });
}
