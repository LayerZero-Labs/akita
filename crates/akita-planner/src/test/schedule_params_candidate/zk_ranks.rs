#[cfg(feature = "catalog-gen")]
use super::*;

/// The inner-candidate request that sizes `params` at `level` of a schedule
/// for `Cfg`, with the given failure target and no modeled cap.
#[cfg(feature = "catalog-gen")]
fn level_request<'a, Cfg: akita_config::CommitmentConfig>(
    policy: &'a PlannerPolicy,
    fold_policy: &'a dyn HonestFoldPolicy,
    challenge: &'a SparseChallengeConfig,
    params: &CommittedGroupParams,
    num_vars: usize,
    level: usize,
    failure_target: FoldFailureTarget,
) -> InnerCommitmentCandidateRequest<'a> {
    let blocks = params.blocks();
    let inner = params.inner().digits;
    let witness_norms = if level == 0 {
        Cfg::committed_source_contract()
            .unwrap()
            .source_norms(
                inner.log_basis,
                inner.num_digits,
                params.d_a(),
                1 << num_vars,
            )
            .unwrap()
            .fold_witness
    } else {
        FoldWitnessNorms::bounded(inner.log_basis, params.d_a())
    };
    InnerCommitmentCandidateRequest {
        policy,
        fold_policy,
        ring_challenge_cfg: challenge,
        challenge_dimension: match params.opening_method() {
            akita_params::OpeningMethod::EvaluationTrace => params.d_a(),
            akita_params::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => challenge_subring_dimension,
        },
        dimensions: params.role_dims(),
        num_claims: 1,
        num_live_ring_elements_per_claim: blocks.live_ring_elements_per_claim,
        num_live_blocks: blocks.live_blocks,
        num_positions_per_block: blocks.positions_per_block,
        num_chunks: params.witness_chunk.num_chunks,
        witness_norms,
        log_basis_open: params.open().digits.log_basis,
        width_s: params.inner().matrix.input_width(),
        modeled_linf_cap: None,
        failure_target,
    }
}

/// W2-d: at k = 2 the one-shot digit ranges of levels 0 and 1 re-price A at
/// the ranks the clear schedule already has.
#[cfg(feature = "catalog-gen")]
#[test]
fn one_shot_sizing_keeps_the_clear_a_ranks_at_the_masked_levels() {
    use akita_config::{policy_of, proof_optimized::fp128, RecursiveCommitmentConfig};
    use akita_params::zk::{ZkOpeningSchedule, ZK_CUT_LEVEL};
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
    let zk = ZkOpeningSchedule::for_tests(row.schedule().clone(), ZK_CUT_LEVEL).unwrap();
    let target = zk.fold_failure_target();
    let one_hot = akita_config::honest_fold_policy_of::<Cfg>();
    let balanced = BalancedSignedDigitFoldPolicy::universal(policy.decomposition.field_bits());
    for level in 0..=ZK_CUT_LEVEL as usize {
        let params = if level == 0 {
            &zk.schedule().root.params
        } else {
            &zk.schedule().recursive_folds[level - 1].params
        };
        assert!(
            matches!(
                params.inner().matrix.security_route(),
                akita_params::InnerCommitSecurityRoute::Linf(_)
            ),
            "masked level {level} must use the Linf route"
        );
        let fold_policy: &dyn HonestFoldPolicy = if level == 0 { &one_hot } else { &balanced };
        let challenge = params.fold_challenge_config();
        let derive = |failure_target| {
            derive_inner_commitment_candidate(level_request::<Cfg>(
                &policy,
                fold_policy,
                &challenge,
                params,
                NUM_VARS,
                level,
                failure_target,
            ))
            .unwrap()
            .expect("admitted candidate")
        };
        let clear = derive(FoldFailureTarget::PerAttempt);
        // The reconstruction reproduces the root exactly; a recursive level
        // may sit below the universal bound through its modeled cap.
        if level == 0 {
            assert_eq!(clear.num_digits_fold, params.num_digits_fold());
        } else {
            assert!(clear.num_digits_fold >= params.num_digits_fold());
        }
        let one_shot = derive(target);
        assert!(one_shot.num_digits_fold >= clear.num_digits_fold);
        assert_eq!(
            one_shot.inner_commit_matrix.output_rank(),
            params.inner().matrix.output_rank(),
            "level {level}: one-shot digits {} vs clear {}",
            one_shot.num_digits_fold,
            params.num_digits_fold(),
        );
    }
}
