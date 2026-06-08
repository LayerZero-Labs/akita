//! End-to-end tests for the **recursive setup-contribution** verifier path.
//!
//! Every other e2e suite exercises [`SetupContributionMode::Direct`], where the
//! verifier scans the expanded setup matrix inline. This suite covers
//! [`SetupContributionMode::Recursive`], where each non-terminal fold level
//! delegates the setup contribution to a setup-product sumcheck (the Stage-3
//! `SetupSumcheckProver` / `SetupSumcheckVerifier` pair) instead.
//!
//! Coverage:
//!
//! * Recursive prove + serialize round-trip + verify succeeds (one-hot, D=64),
//!   across a few arities that actually fold.
//! * The proof carries at least one fold level, so the setup-product sumcheck
//!   path is genuinely exercised (not just a terminal-only proof).
//! * Cross-mode replay is rejected: a Recursive proof must not verify under
//!   Direct mode, and a Direct proof must not verify under Recursive mode. This
//!   pins the setup-product sumcheck as load-bearing rather than cosmetic.

#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_prover::{ComputeBackendSetup, CpuBackend};

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaProofStep, SetupContributionMode,
};
use akita_verifier::CommitmentVerifier;
use common::*;

const TRANSCRIPT_DOMAIN: &[u8] = b"recursive_setup_e2e/onehot";

/// Number of **non-terminal** fold levels in a singleton proof. Only these
/// levels carry the recursive setup-product sumcheck (terminal levels close out
/// the witness directly and never embed a stage-3 proof), so this is the count
/// of levels that exercise the Recursive setup-contribution path.
fn setup_sumcheck_levels<FF: CanonicalField, L: FieldCore>(
    proof: &AkitaBatchedProof<FF, L>,
) -> usize {
    let root_fold = match proof.root {
        AkitaBatchedRootProof::Fold(_) => 1,
        AkitaBatchedRootProof::Terminal(_) | AkitaBatchedRootProof::ZeroFold { .. } => 0,
    };
    let suffix_fold = proof
        .steps
        .iter()
        .filter(|step| matches!(step, AkitaProofStep::Intermediate(_)))
        .count();
    root_fold + suffix_fold
}

struct OnehotProof {
    proof: AkitaBatchedProof<F, F>,
    verifier_setup: akita_types::AkitaVerifierSetup<F>,
    point: Vec<F>,
    opening: F,
    commitment: akita_types::RingCommitment<F, ONEHOT_D>,
}

/// Commit + prove a single one-hot polynomial under the requested setup mode,
/// then round-trip the proof through serialization. Returns everything the
/// verifier needs.
fn prove_onehot(nv: usize, mode: SetupContributionMode) -> OnehotProof {
    let layout = OneHotCfg::get_params_for_batched_commitment(
        &akita_types::ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence"),
    )
    .expect("layout");
    let total_ring = layout.num_blocks * layout.block_len;
    assert_eq!(total_ring * ONEHOT_K, 1usize << nv);

    let poly = make_onehot_poly(&layout, 0xdead_beef_0000 + nv as u64);
    let point = random_point(nv, 0xcafe_0000 + nv as u64);
    let opening = opening_from_poly(&poly, &point, &layout);

    let setup = match mode {
        SetupContributionMode::Direct => <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_prover(nv, 1, 1),
        SetupContributionMode::Recursive => <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_prover_recursion(nv, 1, 1),
    }
    .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
        F,
        ONEHOT_D,
    >>::setup_verifier(&setup);
    let commit_input = std::slice::from_ref(&poly);
    let (commitment, hint) = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
        F,
        ONEHOT_D,
    >>::commit(&setup, &CpuBackend, &prepared, commit_input)
    .expect("commit");

    let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];

    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let proof =
        <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(&point[..], &poly_refs[..], &commitment, hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
            mode,
        )
        .expect("prove");

    let proof_shape = proof.shape();
    let mut serialized = Vec::new();
    proof
        .serialize_compressed(&mut serialized)
        .expect("serialize");
    let proof = AkitaBatchedProof::<F, F>::deserialize_compressed(
        &mut std::io::Cursor::new(serialized),
        &proof_shape,
    )
    .expect("deserialize");

    OnehotProof {
        proof,
        verifier_setup,
        point,
        opening,
        commitment,
    }
}

fn verify_onehot(
    fixture: &OnehotProof,
    mode: SetupContributionMode,
) -> Result<(), akita_field::AkitaError> {
    let openings = [fixture.opening];
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
        &fixture.proof,
        &fixture.verifier_setup,
        &mut verifier_transcript,
        verify_input(&fixture.point[..], &openings[..], &fixture.commitment),
        BasisMode::Lagrange,
        mode,
    )
}

/// Recursive prove + verify round-trips, and the proof actually folds (so the
/// setup-product sumcheck is exercised on at least one level).
fn run_recursive_roundtrip(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let fixture = prove_onehot(nv, SetupContributionMode::Recursive);
        assert!(
            setup_sumcheck_levels(&fixture.proof) > 0,
            "recursive nv={nv} must produce at least one non-terminal fold level \
             so the setup-product sumcheck runs"
        );
        let result = verify_onehot(&fixture, SetupContributionMode::Recursive);
        assert!(
            result.is_ok(),
            "recursive onehot nv={nv} verification failed: {:?}",
            result.err()
        );
    });
}

/// A Recursive proof must not verify under Direct mode, and vice versa. The
/// modes disagree on whether the embedded setup-product sumcheck is present, so
/// each combination is a structural mismatch the verifier must reject.
fn run_cross_mode_rejects(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let recursive = prove_onehot(nv, SetupContributionMode::Recursive);
        assert!(
            setup_sumcheck_levels(&recursive.proof) > 0,
            "cross-mode test needs a folding arity (nv={nv})"
        );
        assert!(
            verify_onehot(&recursive, SetupContributionMode::Direct).is_err(),
            "recursive proof must not verify under Direct mode (nv={nv})"
        );

        let direct = prove_onehot(nv, SetupContributionMode::Direct);
        assert!(
            verify_onehot(&direct, SetupContributionMode::Recursive).is_err(),
            "direct proof must not verify under Recursive mode (nv={nv})"
        );
    });
}

#[test]
fn recursive_onehot_nv20() {
    run_recursive_roundtrip(20);
}

#[test]
fn recursive_onehot_cross_mode_rejects_nv20() {
    run_cross_mode_rejects(20);
}
