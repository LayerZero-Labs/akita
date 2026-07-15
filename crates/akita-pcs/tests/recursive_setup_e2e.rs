//! End-to-end coverage for the generated recursive setup-offload profile.
//!
//! This test intentionally uses the profile emitted in
//! `fp128_d64_onehot_recursive`: two precommitted singleton groups at `nv=16`
//! and a two-polynomial final group at `nv=32`. That generated schedule carries
//! setup-prefix metadata, so a successful recursive proof exercises the
//! offloaded setup-contribution path rather than the inline direct setup scan.

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

const TRANSCRIPT_DOMAIN: &[u8] = b"recursive_setup_e2e/generated_onehot";
const PRE_NV: usize = 16;
const FINAL_NV: usize = 32;
const PRE_GROUPS: usize = 2;
const PRE_GROUP_SIZE: usize = 1;
const FINAL_GROUP_SIZE: usize = 2;
const TOTAL_GROUP_SIZE: usize = PRE_GROUPS * PRE_GROUP_SIZE + FINAL_GROUP_SIZE;

type RecursiveOneHotCfg = RecursiveCommitmentConfig<OneHotCfg>;
type ConservativeOneHotCfg = ConservativeCommitmentConfig<OneHotCfg>;
type RecursiveOneHotScheme = AkitaCommitmentScheme<RecursiveOneHotCfg>;
type ConservativeOneHotScheme = AkitaCommitmentScheme<ConservativeOneHotCfg>;

fn multi_group_root_params(schedule: &Schedule) -> &LevelParams {
    match schedule.steps.first().expect("generated profile root step") {
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

fn generated_recursive_profile_key() -> (AkitaScheduleLookupKey, Vec<PolynomialGroupLayout>) {
    let pre_key = PolynomialGroupLayout::new(PRE_NV, PRE_GROUP_SIZE);
    let pre_layout = OpeningClaimsLayout::new(PRE_NV, PRE_GROUP_SIZE)
        .expect("precommit batch")
        .root_final_group_layout()
        .expect("precommit group layout");
    assert_eq!(pre_layout, pre_key);

    let pre_params = ConservativeOneHotCfg::get_params_for_batched_commitment(
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
fn generated_recursive_onehot_profile_proves_with_setup_offload() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let (schedule_key, pre_keys) = generated_recursive_profile_key();
        let schedule = RecursiveOneHotCfg::runtime_schedule(schedule_key)
            .expect("generated recursive profile schedule");
        assert!(
            schedule_uses_setup_prefix(&schedule),
            "generated recursive profile must carry setup-prefix metadata"
        );
        let root_params = multi_group_root_params(&schedule);

        let setup = RecursiveOneHotScheme::setup_prover(FINAL_NV, TOTAL_GROUP_SIZE)
            .expect("recursive setup");
        assert!(
            !setup.prefix_slots.is_empty(),
            "recursive setup must precompute setup-prefix slots for the generated profile"
        );
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");

        let pre_layout = ConservativeOneHotCfg::get_params_for_batched_commitment(
            &OpeningClaimsLayout::new(PRE_NV, PRE_GROUP_SIZE).expect("precommit batch"),
        )
        .expect("conservative precommit params");
        let mut pre_polys_by_group = Vec::new();
        let mut pre_commitments = Vec::new();
        let mut pre_hints = Vec::new();
        for group_idx in 0..PRE_GROUPS {
            let poly = make_onehot_poly(&pre_layout, 0x0bee_fcaf_2026_0000 + group_idx as u64);
            let (commitment, hint) = ConservativeOneHotScheme::batched_commit(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
            )
            .expect("precommit group");
            pre_polys_by_group.push(vec![poly]);
            pre_commitments.push(commitment);
            pre_hints.push(hint);
        }

        let final_polys: Vec<OneHotPoly<F, u8>> = (0..FINAL_GROUP_SIZE)
            .map(|poly_idx| make_onehot_poly(root_params, 0x0bee_fcaf_2026_1000 + poly_idx as u64))
            .collect();
        let (final_commitment, final_hint) =
            RecursiveOneHotScheme::commit_final_group(&setup, &final_polys, &stack, pre_keys)
                .expect("final generated-profile commitment");

        let point = random_point(FINAL_NV, 0xcafe_2026_0001);
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
        .expect("generated-profile prover data");

        let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        let proof = RecursiveOneHotScheme::batched_prove(
            &setup,
            prover_claims,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            SetupContributionMode::Recursive,
        )
        .expect("generated-profile recursive proof");
        assert!(
            proof_has_recursive_setup_sumcheck(&proof),
            "recursive proof must carry stage-3 setup sumcheck evidence"
        );

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof
            .serialize_compressed(&mut bytes)
            .expect("serialize generated-profile proof");
        let proof = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize generated-profile proof");

        let verifier_setup = setup.verifier_setup().expect("verifier setup");
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
        let verify_claims =
            OpeningClaims::from_groups(point, verifier_groups).expect("verifier claims");
        let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
        RecursiveOneHotScheme::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_claims,
            BasisMode::Lagrange,
            SetupContributionMode::Recursive,
        )
        .expect("generated-profile recursive verify");
    });
}
