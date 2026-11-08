use super::*;
use akita_config::{
    policy_of, proof_optimized::fp128, CommitmentConfig, RecursiveCommitmentConfig,
};
use akita_params::zk::{MaskedFoldSizing, ZkOpeningSchedule, ZK_CUT_LEVEL};

/// W2-d: at every masked level of the clear 36:1 row, re-pricing the row's
/// fixed geometry with the one-shot digit ranges keeps the clear A rank,
/// except at level 2. There the row's digits come from the planner's modeled
/// per-attempt ℓ∞ cap, which a masked level never takes, and the one-shot
/// digits need one more A rank.
#[test]
fn one_shot_sizing_a_ranks_at_the_masked_levels() {
    type Cfg = RecursiveCommitmentConfig<fp128::OneHot>;
    const NUM_VARS: usize = 36;
    const LEVELS: usize = ZK_CUT_LEVEL as usize + 1;
    const CAPPED: [bool; LEVELS] = [false, false, true];
    const EXTRA_A_RANK: [usize; LEVELS] = [0, 0, 1];

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
        assert_eq!(
            params.num_digits_fold() < derive(LevelMasking::Clear).num_digits_fold,
            CAPPED[level],
            "level {level}: whether the row caps its digits changed",
        );
        let one_shot = derive(LevelMasking::Masked(zk.fold_failure_target(1).unwrap()));
        assert_eq!(
            one_shot.inner_commit_matrix.output_rank(),
            params.inner().matrix.output_rank() + EXTRA_A_RANK[level],
            "level {level}: one-shot digits {} vs clear {}",
            one_shot.num_digits_fold,
            params.num_digits_fold(),
        );
    }
}

/// A masked level never takes a per-attempt cap: at every masked level of
/// the 36:1 row, a modeled ℓ∞ cap one digit below the clear derivation's
/// lowers the clear digits and leaves the one-shot digits unchanged.
#[test]
fn one_shot_sizing_ignores_the_per_attempt_cap() {
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
        let derive = |masking: LevelMasking, modeled_linf_cap: Option<u128>| {
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
                modeled_linf_cap,
                masking,
            })
            .unwrap()
            .expect("admitted candidate")
        };
        let uncapped = derive(LevelMasking::Clear, None);
        // The largest value `uncapped.num_digits_fold − 1` balanced digits
        // of base `b` hold: (b/2 − 1)(b^n − 1)/(b − 1).
        let log_basis = sizing
            .query(LevelMasking::Clear.failure_target())
            .unwrap()
            .log_basis_response;
        let base = 1u128 << log_basis;
        let digits = u32::try_from(uncapped.num_digits_fold - 1).unwrap();
        let cap = (base / 2 - 1) * ((base.pow(digits) - 1) / (base - 1));
        let capped = derive(LevelMasking::Clear, Some(cap));
        assert_eq!(
            capped.num_digits_fold,
            uncapped.num_digits_fold - 1,
            "level {level}: the cap lowers the clear digits by one"
        );
        let target = LevelMasking::Masked(zk.fold_failure_target(1).unwrap());
        assert_eq!(
            derive(target, Some(cap)).num_digits_fold,
            derive(target, None).num_digits_fold,
            "level {level}: a one-shot sizing ignores the per-attempt cap"
        );
    }
}

/// W5e: every checked-in zero-knowledge row commits the residual (the level
/// after the cut) with A_a and [B_a | B′_a] at least at their Σ-binding
/// ranks, from the Σ-binding table, at both families, and lists both
/// matrices among its SIS occurrences at the Σ radius, which
/// `catalog_security` estimates. O6: the residual is over
/// `R_{RESIDUAL_RING_DIMENSION}`, above a role ring of the cut level on
/// every row (the ceiling exception, `specs/whiteout-akita.md` §5.5.3).
#[test]
fn checked_in_zk_rows_commit_the_residual_at_its_sigma_binding_ranks() {
    use akita_params::sis::sigma_binding::{sigma_binding_min_rank, sigma_binding_radius};
    use akita_params::zk::RESIDUAL_RING_DIMENSION;
    use akita_params::ScheduleSisBound;

    // O6 fixes the residual ring at R_128; every check below reads the
    // constant, so pin its value here.
    assert_eq!(RESIDUAL_RING_DIMENSION, 128);

    /// The rows, and those whose cut level has a role ring below the
    /// residual's.
    fn check<Cfg: CommitmentConfig>() -> (usize, usize) {
        let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        )
        .unwrap();
        let mut above_the_cut_ring = 0;
        for row in catalog.rows() {
            let schedule = row.schedule();
            let after_cut = &schedule.recursive_folds[ZK_CUT_LEVEL as usize].params;
            assert_eq!(
                after_cut.role_dims(),
                CommitmentRingDims::uniform(RESIDUAL_RING_DIMENSION)
            );
            let cut = schedule.recursive_folds[ZK_CUT_LEVEL as usize - 1]
                .params
                .role_dims();
            if cut.d_a().min(cut.d_b()).min(cut.d_d()) < RESIDUAL_RING_DIMENSION {
                above_the_cut_ring += 1;
            }
            let residual = after_cut.own_group();
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
        (rows, above_the_cut_ring)
    }
    assert_eq!(
        check::<akita_config::ZkCommitmentConfig<fp128::Dense>>(),
        (1, 1)
    );
    assert_eq!(
        check::<akita_config::ZkCommitmentConfig<RecursiveCommitmentConfig<fp128::OneHot>>>(),
        (2, 2)
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
