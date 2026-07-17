//! End-to-end coverage for the mixed distributed (multi-chunk) + recursive
//! setup-offload profile.
//!
//! This test uses `RecursiveCommitmentConfig<fp128::D64OneHotMultiChunkW4R2>`
//! (the `fp128_d64_onehot_recursive_multi_chunk_w4r2` family): two precommitted
//! singleton groups at `nv=16` and a two-polynomial final group at `nv=32`. That
//! schedule combines the `W4R2` chunked witness layout (4 chunks on the two
//! leading fold levels) with recursive setup offloading (Stage-3 setup-product
//! sum-check and a carried setup-prefix opening), so a successful proof exercises
//! the mix: chunked folds that also run the offloaded setup-contribution path.

#![allow(missing_docs)]

use akita_config::{CommitmentConfig, ConservativeCommitmentConfig, RecursiveCommitmentConfig};
use akita_prover::{ComputeBackendSetup, CpuBackend};

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaScheduleLookupKey, BasisMode, OpeningClaims,
    OpeningClaimsLayout, PointVariableSelection, PolynomialGroupClaims, PolynomialGroupLayout,
    PrecommittedGroupParams, Schedule, SetupContributionMode, Step,
};
use common::*;

const TRANSCRIPT_DOMAIN: &[u8] = b"distributed_setup_offload_e2e/w4r2";
const PRE_NV: usize = 16;
const FINAL_NV: usize = 32;
const PRE_GROUPS: usize = 2;
const PRE_GROUP_SIZE: usize = 1;
const FINAL_GROUP_SIZE: usize = 2;
const TOTAL_GROUP_SIZE: usize = PRE_GROUPS * PRE_GROUP_SIZE + FINAL_GROUP_SIZE;

/// Base preset carrying the `W4R2` chunked witness layout.
type MultiChunkBase = fp128::D64OneHotMultiChunkW4R2;
/// The mix config: recursive setup offloading over the chunked base.
type MixCfg = RecursiveCommitmentConfig<MultiChunkBase>;
/// Conservative adapter used to freeze the precommitted singleton groups. The
/// base preset delegates every conservative-relevant parameter to `D64OneHot`,
/// so the frozen params match the profiling key baked into the generated table.
type ConservativeBaseCfg = ConservativeCommitmentConfig<MultiChunkBase>;
type MixScheme = AkitaCommitmentScheme<MixCfg>;
type ConservativeBaseScheme = AkitaCommitmentScheme<ConservativeBaseCfg>;

fn multi_group_root_params(schedule: &Schedule) -> &LevelParams {
    match schedule.steps.first().expect("mix profile root step") {
        Step::Direct(direct) => direct.params.as_ref().expect("multi-group root params"),
        Step::Fold(fold) => &fold.params,
    }
}

fn schedule_uses_setup_prefix(schedule: &Schedule) -> bool {
    schedule.steps.iter().any(|step| {
        matches!(
            step,
            Step::Fold(fold) if fold.params.setup_prefix.is_some()
        )
    })
}

fn schedule_has_recursive_fold(schedule: &Schedule) -> bool {
    schedule.steps.iter().any(|step| {
        matches!(
            step,
            Step::Fold(fold)
                if fold.params.setup_contribution_mode == SetupContributionMode::Recursive
        )
    })
}

fn schedule_has_chunked_fold(schedule: &Schedule) -> bool {
    schedule.steps.iter().any(|step| {
        matches!(
            step,
            Step::Fold(fold) if fold.params.witness_chunk.num_chunks > 1
        )
    })
}

fn schedule_has_chunked_recursive_fold(schedule: &Schedule) -> bool {
    schedule.steps.iter().any(|step| {
        matches!(
            step,
            Step::Fold(fold)
                if fold.params.witness_chunk.num_chunks > 1
                    && fold.params.setup_contribution_mode == SetupContributionMode::Recursive
        )
    })
}

fn proof_has_recursive_setup_sumcheck(proof: &AkitaBatchedProof<F, F>) -> bool {
    let root_has_stage3 = match &proof.root {
        AkitaBatchedRootProof::Fold(fold) => fold.stage3_sumcheck_proof.is_some(),
        AkitaBatchedRootProof::Terminal(_) | AkitaBatchedRootProof::ZeroFold { .. } => false,
    };
    let suffix_has_stage3 = proof
        .steps
        .iter()
        .any(|step| step.stage3_sumcheck_proof().is_some());
    root_has_stage3 || suffix_has_stage3
}

fn mix_profile_key() -> (AkitaScheduleLookupKey, Vec<PolynomialGroupLayout>) {
    let pre_key = PolynomialGroupLayout::new(PRE_NV, PRE_GROUP_SIZE);
    let pre_layout = OpeningClaimsLayout::new(PRE_NV, PRE_GROUP_SIZE)
        .expect("precommit batch")
        .root_final_group_layout()
        .expect("precommit group layout");
    assert_eq!(pre_layout, pre_key);

    let pre_params = ConservativeBaseCfg::get_params_for_batched_commitment(
        &OpeningClaimsLayout::new(PRE_NV, PRE_GROUP_SIZE).expect("precommit batch"),
    )
    .expect("conservative precommit params");
    let pre_frozen = PrecommittedGroupParams::from_params(pre_key, &pre_params);
    let precommitteds = vec![pre_frozen, pre_frozen];
    let pre_keys = vec![pre_key; PRE_GROUPS];
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(FINAL_NV, FINAL_GROUP_SIZE),
        precommitteds,
    };
    (key, pre_keys)
}

#[test]
fn mix_multi_chunk_recursive_profile_proves_and_verifies() {
    init_rayon_pool();
    run_on_large_stack(|| {
        // Sanity: the config really is the mix (chunked base + recursion adapter).
        assert!(MixCfg::chunked_witness_cfg().uses_multi_chunk());
        assert!(MixCfg::recursive_setup_planning());
        assert_eq!(MixCfg::D, akita_types::SETUP_OFFLOAD_D_SETUP);

        let (schedule_key, pre_keys) = mix_profile_key();
        let schedule =
            MixCfg::runtime_schedule(schedule_key).expect("mix profile schedule resolves");
        assert!(
            schedule_uses_setup_prefix(&schedule),
            "mix profile must carry setup-prefix metadata"
        );
        assert!(
            schedule_has_recursive_fold(&schedule),
            "mix profile must contain a recursive setup-contribution fold"
        );
        assert!(
            schedule_has_chunked_fold(&schedule),
            "mix profile must contain a chunked (multi-chunk) fold"
        );
        assert!(
            schedule_has_chunked_recursive_fold(&schedule),
            "mix profile must contain a fold that is BOTH chunked and recursive"
        );
        let root_params = multi_group_root_params(&schedule);

        let setup =
            MixScheme::setup_prover(FINAL_NV, TOTAL_GROUP_SIZE).expect("mix recursive setup");
        assert!(
            !setup.prefix_slots.is_empty(),
            "mix setup must precompute setup-prefix slots for the profile"
        );
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");

        let pre_layout = ConservativeBaseCfg::get_params_for_batched_commitment(
            &OpeningClaimsLayout::new(PRE_NV, PRE_GROUP_SIZE).expect("precommit batch"),
        )
        .expect("conservative precommit params");
        let mut pre_polys_by_group = Vec::new();
        let mut pre_commitments = Vec::new();
        let mut pre_hints = Vec::new();
        for group_idx in 0..PRE_GROUPS {
            let poly = make_onehot_poly(&pre_layout, 0x0bee_fcaf_2026_4000 + group_idx as u64);
            let (commitment, hint) =
                ConservativeBaseScheme::batched_commit(&setup, std::slice::from_ref(&poly), &stack)
                    .expect("precommit group");
            pre_polys_by_group.push(vec![poly]);
            pre_commitments.push(commitment);
            pre_hints.push(hint);
        }

        let final_polys: Vec<OneHotPoly<F, u8>> = (0..FINAL_GROUP_SIZE)
            .map(|poly_idx| make_onehot_poly(root_params, 0x0bee_fcaf_2026_5000 + poly_idx as u64))
            .collect();
        let (final_commitment, final_hint) =
            MixScheme::commit_final_group(&setup, &final_polys, &stack, pre_keys)
                .expect("final mix-profile commitment");

        let point = random_point(FINAL_NV, 0xcafe_2026_4001);
        let pre_openings: Vec<Vec<F>> = pre_polys_by_group
            .iter()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| {
                        opening_from_poly::<ONEHOT_D, _>(poly, &point[..PRE_NV], &pre_layout)
                    })
                    .collect()
            })
            .collect();
        let final_openings: Vec<F> = final_polys
            .iter()
            .map(|poly| opening_from_poly::<ONEHOT_D, _>(poly, &point, root_params))
            .collect();

        let pre_refs_by_group: Vec<Vec<&OneHotPoly<F, u8>>> = pre_polys_by_group
            .iter()
            .map(|polys| polys.iter().collect())
            .collect();
        let final_refs: Vec<&OneHotPoly<F, u8>> = final_polys.iter().collect();

        let mut prover_groups = Vec::new();
        for (group_idx, openings) in pre_openings.iter().enumerate() {
            prover_groups.push(
                PolynomialGroupClaims::new(
                    PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                    openings.clone(),
                    pre_commitments[group_idx].clone(),
                )
                .expect("pre prover group"),
            );
        }
        prover_groups.push(
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
                final_openings.clone(),
                final_commitment.clone(),
            )
            .expect("final prover group"),
        );

        let mut prover_polys: Vec<&[&OneHotPoly<F, u8>]> = Vec::new();
        for refs in &pre_refs_by_group {
            prover_polys.push(&refs[..]);
        }
        prover_polys.push(&final_refs[..]);
        let mut prover_hints = pre_hints;
        prover_hints.push(final_hint);

        let prover_claims = ProverOpeningData::new(
            OpeningClaims::from_groups(point.clone(), prover_groups).expect("prover claims"),
            prover_hints,
            prover_polys,
        )
        .expect("mix-profile prover data");

        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        let proof = MixScheme::batched_prove(
            &setup,
            prover_claims,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            SetupContributionMode::Recursive,
        )
        .expect("mix-profile recursive proof");
        assert!(
            proof_has_recursive_setup_sumcheck(&proof),
            "mix proof must carry stage-3 setup sumcheck evidence"
        );

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof
            .serialize_compressed(&mut bytes)
            .expect("serialize mix-profile proof");
        let proof = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize mix-profile proof");

        let verifier_setup = setup.verifier_setup().expect("verifier setup");
        // Build the verifier claims from a (possibly tampered) set of final
        // openings so the positive and negative checks share the proof.
        let build_verify_claims = |final_openings: Vec<F>| {
            let mut verifier_groups = Vec::new();
            for (group_idx, openings) in pre_openings.iter().enumerate() {
                verifier_groups.push(
                    PolynomialGroupClaims::new(
                        PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                        openings.clone(),
                        &pre_commitments[group_idx],
                    )
                    .expect("pre verifier group"),
                );
            }
            verifier_groups.push(
                PolynomialGroupClaims::new(
                    PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
                    final_openings,
                    &final_commitment,
                )
                .expect("final verifier group"),
            );
            OpeningClaims::from_groups(point.clone(), verifier_groups).expect("verifier claims")
        };

        let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        MixScheme::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            build_verify_claims(final_openings.clone()),
            BasisMode::Lagrange,
            SetupContributionMode::Recursive,
        )
        .expect("mix-profile recursive verify");

        // Negative: a corrupted final opening must be rejected without panic
        // (reuses the same proof; verify is cheap relative to prove).
        let mut tampered_openings = final_openings;
        tampered_openings[0] += F::from_canonical_u128_reduced(1);
        let mut tampered_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        let tampered = MixScheme::batched_verify(
            &proof,
            &verifier_setup,
            &mut tampered_transcript,
            build_verify_claims(tampered_openings),
            BasisMode::Lagrange,
            SetupContributionMode::Recursive,
        );
        assert!(
            tampered.is_err(),
            "mix verify must reject a tampered final opening"
        );
    });
}
