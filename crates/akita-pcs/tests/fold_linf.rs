#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

mod common;

use akita_field::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    sis::MAX_FOLD_GRIND_ATTEMPTS, AkitaBatchedProof, AkitaBatchedRootProof, AkitaLevelProof,
};
use akita_verifier::CommitmentVerifier;
use common::*;

type Scheme = AkitaCommitmentScheme<ONEHOT_D, OneHotCfg>;

fn bump_flat_ring_vec(flat: &mut akita_types::FlatRingVec<F>) {
    let mut coeffs = flat.coeffs().to_vec();
    let first = coeffs
        .first_mut()
        .expect("tamper target must contain at least one coefficient");
    *first += F::one();
    *flat = akita_types::FlatRingVec::from_coeffs(coeffs);
}

fn assert_invalid_proof(case: &str, result: Result<(), AkitaError>) {
    assert!(
        matches!(result, Err(AkitaError::InvalidProof)),
        "{case} must reject with InvalidProof, got {result:?}"
    );
}

fn run_tail_bound_with_grind_onehot_roundtrip(
    num_vars: usize,
    seed: u64,
) -> AkitaBatchedProof<F, F> {
    let layout = OneHotCfg::get_params_for_batched_commitment(
        &akita_types::OpeningBatch::same_point(num_vars, 1).expect("singleton opening batch"),
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
    let verifier_setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
    let (commitment, hint) = <Scheme as CommitmentProver<F, ONEHOT_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .expect("commit");

    let mut prover_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
    let proof = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        prove_input(&point, &[&poly], &commitment, hint),
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

    proof
}

#[test]
fn tail_bound_with_grind_onehot_e2e_prove_verify() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let proof = run_tail_bound_with_grind_onehot_roundtrip(28, 0x51_51_00_01);
        assert!(
            matches!(proof.root, AkitaBatchedRootProof::Fold(_)),
            "expected a folded root proof"
        );
        for step in proof.fold_levels() {
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
        let proof = run_tail_bound_with_grind_onehot_roundtrip(28, 0x51_51_00_02);
        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof
            .serialize_compressed(&mut bytes)
            .expect("serialize proof");
        let mut roundtrip =
            AkitaBatchedProof::<F, F>::deserialize_compressed(&bytes[..], &shape).expect("decode");

        let num_vars = 28;
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatch::same_point(num_vars, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let poly = make_onehot_poly(&layout, 0x51_51_00_02);
        let point = random_point(num_vars, 0x51_51_00_03);
        let opening = opening_from_poly(&poly, &point, &layout);
        let setup =
            <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(num_vars, 1).expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepare setup");
        let verifier_setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, _) = <Scheme as CommitmentProver<F, ONEHOT_D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            std::slice::from_ref(&poly),
        )
        .expect("commit");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
        <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &roundtrip,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &[opening], &commitment),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("deserialized proof must verify");

        if let AkitaBatchedRootProof::Fold(fold) = &mut roundtrip.root {
            fold.fold_grind_nonce = MAX_FOLD_GRIND_ATTEMPTS;
        }
        if let Some(AkitaLevelProof::Terminal {
            fold_grind_nonce, ..
        }) = roundtrip.steps.last_mut()
        {
            *fold_grind_nonce = MAX_FOLD_GRIND_ATTEMPTS;
        }

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
        let err = <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &roundtrip,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &[opening], &commitment),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect_err("oversized grind nonce must be rejected");
        assert!(matches!(err, AkitaError::InvalidProof));
    });
}

#[test]
fn fold_recursive_handle_tamper_rejected() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let proof = run_tail_bound_with_grind_onehot_roundtrip(28, 0x51_51_00_04);
        let mut malformed = proof;
        let recursive = malformed
            .steps
            .iter_mut()
            .find_map(akita_types::AkitaLevelProof::as_intermediate_mut)
            .expect("tail-bound-with-grind onehot nv28 should include an intermediate fold");
        bump_flat_ring_vec(recursive.v_mut());

        let num_vars = 28;
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatch::same_point(num_vars, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let poly = make_onehot_poly(&layout, 0x51_51_00_04);
        let point = random_point(num_vars, 0x51_51_00_05);
        let opening = opening_from_poly(&poly, &point, &layout);
        let setup =
            <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(num_vars, 1).expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepare setup");
        let verifier_setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, _) = <Scheme as CommitmentProver<F, ONEHOT_D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            std::slice::from_ref(&poly),
        )
        .expect("commit");

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"fold-linf/onehot");
        let result = <Scheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &[opening], &commitment),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert_invalid_proof("tampered recursive fold handle", result);
    });
}

#[cfg(feature = "logging-transcript")]
#[test]
fn logging_transcript_event_stream_equality_tail_bound_with_grind() {
    use akita_transcript::{labels, LoggingTranscript, Transcript};

    init_rayon_pool();
    run_on_large_stack(|| {
        let num_vars = 28;
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatch::same_point(num_vars, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let poly = make_onehot_poly(&layout, 0x61_61);
        let point = random_point(num_vars, 0x71_71);
        let opening = opening_from_poly(&poly, &point, &layout);

        let setup =
            <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(num_vars, 1).expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepare setup");
        let verifier_setup = <Scheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, hint) = <Scheme as CommitmentProver<F, ONEHOT_D>>::commit(
            &setup,
            &CpuBackend,
            &prepared,
            std::slice::from_ref(&poly),
        )
        .expect("commit");

        let mut prover_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(b"fold-linf/logging"));
        let proof = <Scheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(&point, &[&poly], &commitment, hint),
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
