//! End-to-end tests for the **recursive setup-contribution** verifier path.
//!
//! Every other e2e suite exercises [`SetupContributionMode::Direct`], where the
//! verifier scans the expanded setup matrix inline. This suite covers
//! [`SetupContributionMode::Recursive`], where each non-terminal fold level
//! delegates the setup contribution to a setup-product sumcheck (the Stage-3
//! `AkitaStage3Prover` / `SetupSumcheckVerifier` pair) instead.
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

use akita_prover::{ComputeBackendSetup, CpuBackend};

mod common;

use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    validate_schedule_ring_dims, AjtaiKeyParams, AkitaBatchedProof, AkitaBatchedRootProof,
    AkitaLevelProof, CommitmentRingDims, OpeningClaimsLayout, Schedule, SetupContributionMode,
    Step,
};
use common::*;

const TRANSCRIPT_DOMAIN: &[u8] = b"recursive_setup_e2e/onehot";

#[derive(Clone, Copy, Debug, Default)]
struct NestedRolesDense;

impl CommitmentConfig for NestedRolesDense {
    type Field = <DenseCfg as CommitmentConfig>::Field;
    type ExtField = <DenseCfg as CommitmentConfig>::ExtField;

    const D: usize = <DenseCfg as CommitmentConfig>::D;

    fn decomposition() -> akita_types::DecompositionParams {
        DenseCfg::decomposition()
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        DenseCfg::ring_challenge_config(d)
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        DenseCfg::sis_modulus_family()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<akita_types::SetupMatrixEnvelope, AkitaError> {
        DenseCfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys)
    }

    fn basis_range() -> (u32, u32) {
        DenseCfg::basis_range()
    }

    fn get_params_for_prove(opening_batch: &OpeningClaimsLayout) -> Result<Schedule, AkitaError> {
        let mut schedule = DenseCfg::get_params_for_prove(opening_batch)?;
        for step in &mut schedule.steps {
            let Step::Fold(fold) = step else {
                continue;
            };
            fold.params.b_key = key_at_dimension(&fold.params.b_key, 32);
            fold.params.d_key = key_at_dimension(&fold.params.d_key, 32);
            fold.params.role_dims = CommitmentRingDims {
                inner: 64,
                outer: 32,
                opening: 32,
            };
        }
        Ok(schedule)
    }
}

fn key_at_dimension(key: &AjtaiKeyParams, dimension: usize) -> AjtaiKeyParams {
    AjtaiKeyParams::new_unchecked(
        key.min_security_bits(),
        key.sis_family(),
        key.row_len(),
        key.col_len(),
        key.coeff_linf_bound(),
        dimension,
    )
}

/// Number of **non-terminal** fold levels in a singleton proof. Only these
/// levels carry the recursive setup-product sumcheck (terminal levels close out
/// the witness directly and never embed a stage-3 proof), so this is the count
/// of levels that exercise the Recursive setup-contribution path.
fn setup_sumcheck_levels<FF: CanonicalField, E: FieldCore>(
    proof: &AkitaBatchedProof<FF, E>,
) -> usize {
    let root_fold = match proof.root {
        AkitaBatchedRootProof::Fold(_) => 1,
        AkitaBatchedRootProof::Terminal(_) | AkitaBatchedRootProof::ZeroFold { .. } => 0,
    };
    let suffix_fold = proof
        .steps
        .iter()
        .filter(|step| matches!(step, AkitaLevelProof::Intermediate { .. }))
        .count();
    root_fold + suffix_fold
}

struct OnehotProof {
    proof: AkitaBatchedProof<F, F>,
    verifier_setup: akita_types::AkitaVerifierSetup<F>,
    point: Vec<F>,
    opening: F,
    commitment: akita_types::Commitment<F>,
}

/// Commit + prove a single one-hot polynomial under the requested setup mode,
/// then round-trip the proof through serialization. Returns everything the
/// verifier needs.
fn prove_onehot(nv: usize, mode: SetupContributionMode) -> OnehotProof {
    prove_onehot_with_setup_mode(nv, mode, mode)
}

fn prove_onehot_with_setup_mode(
    nv: usize,
    proof_mode: SetupContributionMode,
    setup_mode: SetupContributionMode,
) -> OnehotProof {
    let layout = OneHotCfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(nv, 1).expect("singleton opening batch"),
    )
    .expect("layout");
    let total_ring = layout.live_fold_count * layout.fold_position_count;
    // `total_ring` ring elements of degree D cover `2^nv` field elements,
    // independent of the one-hot chunk size K.
    assert_eq!(total_ring * ONEHOT_D, 1usize << nv);

    let poly = make_onehot_poly(&layout, 0xdead_beef_0000 + nv as u64);
    let point = random_point(nv, 0xcafe_0000 + nv as u64);
    let opening = opening_from_poly::<ONEHOT_D, _>(&poly, &point, &layout);

    let setup = match setup_mode {
        SetupContributionMode::Direct => AkitaCommitmentScheme::<OneHotCfg>::setup_prover(nv, 1),
        SetupContributionMode::Recursive => {
            AkitaCommitmentScheme::<OneHotCfg>::setup_prover_recursion(nv, 1)
        }
    }
    .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = AkitaCommitmentScheme::<OneHotCfg>::setup_verifier(&setup);
    let commit_input = std::slice::from_ref(&poly);
    let (commitment, hint) =
        AkitaCommitmentScheme::<OneHotCfg>::commit::<_, _>(&setup, commit_input, &stack)
            .expect("commit");

    let poly_refs: [&OneHotPoly<F, u8>; 1] = [&poly];

    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let proof = AkitaCommitmentScheme::<OneHotCfg>::batched_prove::<_, _, _>(
        &setup,
        prove_input(&point[..], &poly_refs[..], &commitment, hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        proof_mode,
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
    AkitaCommitmentScheme::<OneHotCfg>::batched_verify(
        &fixture.proof,
        &fixture.verifier_setup,
        &mut verifier_transcript,
        verify_input(&fixture.point[..], &openings[..], &fixture.commitment),
        BasisMode::Lagrange,
        mode,
    )
}

fn prove_nested_roles_onehot(nv: usize) -> OnehotProof {
    let opening_batch = OpeningClaimsLayout::new(nv, 1).expect("singleton opening batch");
    let layout = NestedRolesDense::get_params_for_batched_commitment(&opening_batch)
        .expect("nested-role commitment layout");
    let schedule =
        NestedRolesDense::get_params_for_prove(&opening_batch).expect("nested-role schedule");
    for fold in schedule.fold_steps() {
        assert_eq!(
            fold.params.role_dims(),
            CommitmentRingDims {
                inner: 64,
                outer: 32,
                opening: 32,
            }
        );
    }

    let poly = make_dense_poly(nv, 0x6400_3200 + nv as u64);
    let point = random_point(nv, 0x3200_6400 + nv as u64);
    let opening = opening_from_poly::<DENSE_D, _>(&poly, &point, &layout);
    let setup = AkitaCommitmentScheme::<NestedRolesDense>::setup_prover_recursion(nv, 1)
        .expect("nested-role recursive setup");
    validate_schedule_ring_dims(&schedule, setup.expanded.seed())
        .expect("nested-role schedule validation");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let root_dims = schedule
        .fold_steps()
        .next()
        .expect("nested schedule root fold")
        .params
        .role_dims();
    stack
        .ensure_fold_level_role_ntt(setup.expanded.as_ref(), root_dims)
        .expect("warm nested-role NTT slots");
    let verifier_setup = AkitaCommitmentScheme::<NestedRolesDense>::setup_verifier(&setup);
    let (commitment, hint) = AkitaCommitmentScheme::<NestedRolesDense>::commit(
        &setup,
        std::slice::from_ref(&poly),
        &stack,
    )
    .expect("nested-role commit");
    let poly_refs = [&poly];
    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let proof = AkitaCommitmentScheme::<NestedRolesDense>::batched_prove(
        &setup,
        prove_input(&point, &poly_refs, &commitment, hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        SetupContributionMode::Recursive,
    )
    .expect("nested-role recursive prove");
    let shape = proof.shape();
    let mut serialized = Vec::new();
    proof
        .serialize_compressed(&mut serialized)
        .expect("serialize nested-role proof");
    let proof = AkitaBatchedProof::<F, F>::deserialize_compressed(
        &mut std::io::Cursor::new(serialized),
        &shape,
    )
    .expect("deserialize nested-role proof");
    OnehotProof {
        proof,
        verifier_setup,
        point,
        opening,
        commitment,
    }
}

fn verify_nested_roles_onehot(fixture: &OnehotProof) -> Result<(), AkitaError> {
    let openings = [fixture.opening];
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    AkitaCommitmentScheme::<NestedRolesDense>::batched_verify(
        &fixture.proof,
        &fixture.verifier_setup,
        &mut verifier_transcript,
        verify_input(&fixture.point, &openings, &fixture.commitment),
        BasisMode::Lagrange,
        SetupContributionMode::Recursive,
    )
}

/// Recursive prove + verify round-trips, and the proof actually folds (so the
/// setup-product sumcheck is exercised on at least one level).
///
/// Snap-sized schedules at large `nv` (e.g. 20) may not admit setup-prefix
/// repacking into the next fold's `B` geometry; recursive setup still succeeds
/// via the inline-setup fallback exercised in
/// [`run_recursive_missing_prefix_slots_falls_back`]. Prefix slot population
/// for smaller arities is covered by `akita-setup::recursion` unit tests.
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

fn run_recursive_missing_prefix_slots_falls_back(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let fixture = prove_onehot_with_setup_mode(
            nv,
            SetupContributionMode::Recursive,
            SetupContributionMode::Direct,
        );
        assert!(
            fixture.verifier_setup.prefix_slots.is_empty(),
            "direct setup should not populate setup-prefix slots"
        );
        assert!(
            setup_sumcheck_levels(&fixture.proof) > 0,
            "fallback test needs a folding arity (nv={nv})"
        );
        let result = verify_onehot(&fixture, SetupContributionMode::Recursive);
        assert!(
            result.is_ok(),
            "missing-prefix fallback failed: {:?}",
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
fn recursive_onehot_missing_prefix_slots_falls_back_nv20() {
    run_recursive_missing_prefix_slots_falls_back(20);
}

#[test]
fn recursive_onehot_cross_mode_rejects_nv20() {
    run_cross_mode_rejects(20);
}

#[test]
fn recursive_nested_roles_64_32_32_roundtrip() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = prove_nested_roles_onehot(20);
        assert!(
            setup_sumcheck_levels(&fixture.proof) > 0,
            "nested-role proof must exercise recursive Stage 3"
        );
        verify_nested_roles_onehot(&fixture).expect("verify nested-role recursive proof");
    });
}

#[test]
fn recursive_nested_roles_reject_non_nested_geometry() {
    let opening_batch = OpeningClaimsLayout::new(20, 1).expect("opening batch");
    let mut schedule =
        NestedRolesDense::get_params_for_prove(&opening_batch).expect("nested schedule");
    let Some(Step::Fold(first)) = schedule.steps.first_mut() else {
        panic!("recursive schedule must begin with a fold");
    };
    first.params.role_dims.opening = 64;
    let setup =
        AkitaCommitmentScheme::<NestedRolesDense>::setup_prover(20, 1).expect("nested-role setup");
    assert!(matches!(
        validate_schedule_ring_dims(&schedule, setup.expanded.seed()),
        Err(AkitaError::InvalidSetup(_))
    ));
}
