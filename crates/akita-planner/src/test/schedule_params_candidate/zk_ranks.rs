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

/// W5e: every checked-in zero-knowledge row commits the residual (the level
/// after the cut) with A_a and [B_a | B′_a] at least at their Σ-binding
/// ranks, from the Σ-binding table, at both families, and lists both
/// matrices among its SIS occurrences at the Σ radius, which
/// `catalog_security` estimates.
#[test]
fn checked_in_zk_rows_commit_the_residual_at_its_sigma_binding_ranks() {
    use akita_params::sis::sigma_binding::{sigma_binding_min_rank, sigma_binding_radius};
    use akita_params::ScheduleSisBound;

    fn check<Cfg: CommitmentConfig>() -> usize {
        let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        )
        .unwrap();
        for row in catalog.rows() {
            let schedule = row.schedule();
            let residual = schedule.recursive_folds[ZK_CUT_LEVEL as usize]
                .params
                .own_group();
            let inner = &residual.profile.inner.matrix;
            let outer = &residual.profile.outer.matrix;
            for (rank, width, d, modulus) in [
                (
                    inner.output_rank(),
                    inner.input_width(),
                    inner.ring_dimension(),
                    inner.sis_modulus_profile(),
                ),
                (
                    outer.output_rank(),
                    outer.input_width(),
                    outer.ring_dimension(),
                    outer.sis_modulus_profile(),
                ),
            ] {
                let binding = sigma_binding_min_rank(modulus, d, width).unwrap();
                assert!(rank >= binding, "rank {rank} < Σ-binding rank {binding}");
            }
            let at_sigma_radius = schedule
                .sis_occurrences()
                .unwrap()
                .into_iter()
                .filter(|occurrence| {
                    occurrence.bound == ScheduleSisBound::Linf(sigma_binding_radius())
                })
                .map(|occurrence| (occurrence.output_rank, occurrence.input_width))
                .collect::<Vec<_>>();
            assert_eq!(
                at_sigma_radius,
                [
                    (inner.output_rank(), inner.input_width()),
                    (outer.output_rank(), outer.input_width()),
                ]
            );
        }
        let rows = catalog.rows().len();
        rows
    }
    assert_eq!(check::<akita_config::ZkCommitmentConfig<fp128::Dense>>(), 1);
    assert_eq!(
        check::<akita_config::ZkCommitmentConfig<RecursiveCommitmentConfig<fp128::OneHot>>>(),
        2
    );
}

/// A masked level's witness and the residual commitment's opening are
/// folded, never sent: the search offers no terminal at the cut or the level
/// after it, so it plans past the residual even where a terminal at the
/// cut level is cheaper (the checked-in 14-variable statement).
#[test]
fn zk_search_never_ends_at_a_masked_or_residual_level() {
    type Cfg = akita_config::ZkCommitmentConfig<fp128::Dense>;
    let policy = policy_of::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let planned = crate::find_zk_schedule(
        &akita_params::ScheduleLookupKey::single(PolynomialGroupLayout::singleton(14)),
        Cfg::committed_source_contract().unwrap(),
        &[],
        &policy,
        Cfg::ring_challenge_config,
        ZkOpeningSchedule::new(ZK_CUT_LEVEL).unwrap(),
    )
    .unwrap();
    assert!(planned.schedule.recursive_folds.len() > ZK_CUT_LEVEL as usize);
}
