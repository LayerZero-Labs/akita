//! End-to-end tests for the setup-side claim-reduction sumcheck flow.
//!
//! These tests mirror `tensor_stage1_e2e.rs` but additionally enable
//! `LevelParams::use_setup_claim_reduction` at the root level. The full proof
//! must therefore include a `SetupClaimReductionPayload` that the verifier
//! consumes to close the stage-2 main sumcheck without the materialized
//! M-table contribution from setup-side rows.
//!
//! The wrapper config that promotes a production preset into a
//! claim-reduction-enabled preset lives in
//! [`akita_config::ClaimReductionCfg`]. Here we exercise the un-tiered
//! (`SHRINK = 1`) shape, which matches the Phase D-full Slice F (`f = 1`,
//! `k = 1`) routing path; the tiered `SHRINK = 8` variant is covered by
//! the dedicated Slice G E2E suite.

#![allow(missing_docs)]

mod common;

use akita_config::{CommitmentConfig, UntieredClaimReductionCfg};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_transcript::Blake2bTranscript;
use akita_verifier::CommitmentVerifier;
use common::*;
use std::sync::Mutex;

static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

type ClaimReductionOneHotCfg = UntieredClaimReductionCfg<OneHotCfg>;
type ClaimReductionDenseCfg = UntieredClaimReductionCfg<DenseCfg>;

#[test]
fn setup_claim_reduction_dense_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 20;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, ClaimReductionDenseCfg>;

        let layout = ClaimReductionDenseCfg::commitment_layout(NV).expect("layout");
        assert!(
            layout.use_setup_claim_reduction,
            "test must exercise setup claim reduction"
        );
        let poly = make_dense_poly(NV, 0x7c1a_1d31);
        let pt = random_point(NV, 0x7c1a_1d32);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/dense_singleton");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("dense claim-reduction prove");

        let fold_root = proof
            .root
            .as_fold()
            .expect("test must exercise tensor root fold");
        let payload = fold_root
            .stage2
            .setup_claim_reduction
            .as_ref()
            .expect("fold root stage-2 proof should carry a setup claim-reduction payload");
        assert!(
            layout
                .setup_polynomial_padded_dims(&[1], 1, 1)
                .expect("setup dims")
                .1
                .trailing_zeros()
                > 0,
            "test must have setup column bits to prove CR rounds omit them",
        );
        let expected_rounds = layout
            .m_row_count(1, 1)
            .next_power_of_two()
            .trailing_zeros() as usize
            + D.trailing_zeros() as usize;
        assert_eq!(
            payload.sumcheck.round_polys.len(),
            expected_rounds,
            "setup claim-reduction rounds must be row_bits + coeff_bits, excluding col_bits"
        );

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/dense_singleton");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("dense claim-reduction verify");
    });
}

#[test]
fn setup_claim_reduction_onehot_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 17;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, ClaimReductionOneHotCfg>;

        let layout = ClaimReductionOneHotCfg::commitment_layout(NV).expect("layout");
        assert!(
            layout.use_setup_claim_reduction,
            "test must exercise setup claim reduction"
        );
        let poly = make_onehot_poly(&layout, 0x7c1a_2222);
        let pt = random_point(NV, 0x7c1a_2223);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/onehot_singleton");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("onehot claim-reduction prove");

        let fold_root = proof
            .root
            .as_fold()
            .expect("test must exercise tensor root fold");
        assert!(
            fold_root.stage2.setup_claim_reduction.is_some(),
            "fold root stage-2 proof should carry a setup claim-reduction payload"
        );

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/onehot_singleton");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("onehot claim-reduction verify");
    });
}

#[test]
fn setup_claim_reduction_dense_recursive_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 20;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, ClaimReductionDenseCfg>;

        let layout = ClaimReductionDenseCfg::commitment_layout(NV).expect("layout");
        let poly = make_dense_poly(NV, 0x7c1a_3331);
        let pt = random_point(NV, 0x7c1a_3332);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/dense_recursive");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("dense recursive claim-reduction prove");

        let fold_root = proof
            .root
            .as_fold()
            .expect("recursive test must exercise tensor root fold");
        assert!(
            fold_root.stage2.setup_claim_reduction.is_some(),
            "fold root stage-2 proof should carry a setup claim-reduction payload"
        );

        let recursive_levels: Vec<_> = proof.fold_levels().collect();
        assert!(
            !recursive_levels.is_empty(),
            "recursive test (NV={NV}) must exercise at least one recursive fold level"
        );
        for level_proof in &recursive_levels {
            assert!(
                level_proof.stage2.setup_claim_reduction.is_some(),
                "every recursive fold level should carry a setup claim-reduction payload"
            );
        }

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/dense_recursive");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("dense recursive claim-reduction verify");
    });
}

#[test]
fn setup_claim_reduction_rejects_tampered_m_setup_eval() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 17;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, ClaimReductionOneHotCfg>;

        let layout = ClaimReductionOneHotCfg::commitment_layout(NV).expect("layout");
        let poly = make_onehot_poly(&layout, 0x7c1a_b001);
        let pt = random_point(NV, 0x7c1a_b002);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/onehot_tamper");
        let mut proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("onehot claim-reduction prove");

        let fold_root = proof
            .root
            .as_fold_mut()
            .expect("tamper test must exercise tensor root fold");
        let payload = fold_root
            .stage2
            .setup_claim_reduction
            .as_mut()
            .expect("tamper test must have a setup claim-reduction payload");
        payload.m_setup_eval += F::from_u64(1);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/onehot_tamper");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "tampered setup claim-reduction m_setup_eval must be rejected"
        );
    });
}

/// Tamper test for the Phase D-full wire field `s_opening_value`.
/// Mirrors the `m_setup_eval` test above: corrupting the prover-claimed
/// `S(r_i, r_x, r_k)` at the fixed main-stage `r_x` must fail verification.
/// The closing-oracle equality and, on non-routed levels, the cleartext
/// derived-polynomial check each suffice to catch the tamper.
#[test]
fn setup_claim_reduction_rejects_tampered_s_opening_value() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 17;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, ClaimReductionOneHotCfg>;

        let layout = ClaimReductionOneHotCfg::commitment_layout(NV).expect("layout");
        let poly = make_onehot_poly(&layout, 0x7c1a_b101);
        let pt = random_point(NV, 0x7c1a_b102);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/onehot_tamper_s");
        let mut proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("onehot claim-reduction prove");

        let fold_root = proof
            .root
            .as_fold_mut()
            .expect("tamper test must exercise tensor root fold");
        let payload = fold_root
            .stage2
            .setup_claim_reduction
            .as_mut()
            .expect("tamper test must have a setup claim-reduction payload");
        payload.s_opening_value += F::from_u64(1);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/onehot_tamper_s");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "tampered setup claim-reduction s_opening_value must be rejected"
        );
    });
}
