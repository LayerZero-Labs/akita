use super::*;
use akita_config::{
    policy_of, proof_optimized::fp128, CommitmentConfig, RecursiveCommitmentConfig,
};
use akita_params::zk::{MaskedFoldSizing, ZkOpeningSchedule, ZK_CUT_LEVEL};

/// W2-d: at k = 2 the one-shot digit ranges of levels 0 and 1 re-price A at
/// the ranks the clear schedule already has.
#[test]
fn one_shot_sizing_keeps_the_clear_a_ranks_at_the_masked_levels() {
    type Cfg = RecursiveCommitmentConfig<fp128::OneHot>;
    const NUM_VARS: usize = 36;

    let chunking = akita_params::ChunkedWitnessCfg::default_non_chunked();
    let policy = policy_of::<Cfg>(chunking);
    let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>(chunking)
        .expect("recursive one-hot catalog");
    let row = catalog
        .resolve_key(&akita_params::ScheduleLookupKey::single(
            PolynomialGroupLayout::new(NUM_VARS, 1),
        ))
        .expect("headline-shaped row");
    let source = Cfg::committed_source_contract().unwrap();
    let zk = ZkOpeningSchedule::new(ZK_CUT_LEVEL).unwrap();
    for level in 0..=ZK_CUT_LEVEL as usize {
        let sizing = MaskedFoldSizing::new(row.schedule(), level, source).unwrap();
        let params = sizing.params();
        assert!(
            matches!(
                params.inner().matrix.security_route(),
                akita_params::InnerCommitSecurityRoute::Linf(_)
            ),
            "masked level {level} must use the Linf route"
        );
        let derive = |masking: LevelMasking| {
            let query = sizing.query(masking.failure_target()).unwrap();
            derive_inner_commitment_candidate(InnerCommitmentCandidateRequest {
                policy: &policy,
                fold_policy: sizing.policy(),
                ring_challenge_cfg: query.challenge_config,
                challenge_dimension: query.challenge_dimension,
                dimensions: params.role_dims(),
                num_claims: query.num_claims,
                num_live_ring_elements_per_claim: query.num_live_ring_elements_per_claim,
                num_live_blocks: query.num_live_blocks,
                num_positions_per_block: query.num_positions_per_block,
                num_chunks: query.num_chunks,
                witness_norms: query.witness_norms,
                log_basis_open: query.log_basis_response,
                width_s: params.inner().matrix.input_width(),
                modeled_linf_cap: None,
                masking,
            })
            .unwrap()
            .expect("admitted candidate")
        };
        // The scheduled digits are the policy's own at both levels: neither
        // carries a modeled cap below the universal bound.
        assert_eq!(
            derive(LevelMasking::Clear).num_digits_fold,
            params.num_digits_fold()
        );
        let one_shot = derive(LevelMasking::Masked(zk.fold_failure_target(1).unwrap()));
        assert_eq!(
            one_shot.inner_commit_matrix.output_rank(),
            params.inner().matrix.output_rank(),
            "level {level}: one-shot digits {} vs clear {}",
            one_shot.num_digits_fold,
            params.num_digits_fold(),
        );
    }
}
