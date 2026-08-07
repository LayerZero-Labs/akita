use super::recursive::{
    recursive_candidate_order_key, recursive_split_lower_bound, seed_recursive_split_candidates,
    RecursiveSplitLowerBoundInput,
};
use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_types::{PolynomialGroupLayout, SisModulusProfileId};

fn grouped_level_params() -> CommittedGroupParams {
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let mut params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        3,
        2,
        2,
        2,
        fold_challenge_config,
    )
    .with_decomp(2, 2, 2, 2, 2)
    .expect("grouped params");
    let precommitted = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        3,
        2,
        2,
        2,
        fold_challenge_config,
    )
    .with_decomp(2, 2, 2, 2, 2)
    .expect("precommitted params");
    params.precommitted_groups = vec![PrecommittedLevelParams {
        layout: CommittedGroupProfile::from_params(PolynomialGroupLayout::new(6, 1), &precommitted),
        log_basis_open: precommitted.log_basis_open,
        fold_challenge_config: precommitted.fold_challenge_config,
        num_digits_open: precommitted.num_digits_open,
        num_digits_fold: precommitted.num_digits_fold,
    }];
    params
}

#[test]
fn planned_next_witness_len_rejects_multi_group_root_level_params() {
    let grouped = grouped_level_params();
    let err = planned_next_witness_len(128, &grouped, 1, 1)
        .expect_err("multi-group root suffix sizing must use output_witness_len");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn seed_recursive_split_candidates_falls_back_to_exhaustive_for_small_domains() {
    assert_eq!(
        seed_recursive_split_candidates(64, 5, 1, 22, 1),
        vec![4, 3, 2, 1]
    );
}

#[test]
fn seed_recursive_split_candidates_includes_endpoints_and_unique_window() {
    let candidates = seed_recursive_split_candidates(8192, 13, 1, 22, 1);
    assert!(candidates.contains(&1));
    assert!(candidates.contains(&12));
    assert!(
        candidates.windows(2).all(|pair| pair[0] > pair[1]),
        "candidates must be unique and descending: {candidates:?}"
    );
}

#[test]
fn recursive_split_lower_bound_prices_score_floor() {
    assert_eq!(
        recursive_split_lower_bound(RecursiveSplitLowerBoundInput {
            num_ring_elems: 100,
            ring_dimension: 64,
            reduced_vars: 7,
            r: 3,
            delta_commit: 1,
            delta_open: 4,
            num_chunks: 2,
            requested_fold_shape: TensorChallengeShape::Flat,
        }),
        Some(5646)
    );
}

#[test]
fn recursive_candidate_order_preserves_exhaustive_tie_break() {
    let score = (100, 90, 5, 0);
    assert!(
        recursive_candidate_order_key(score, 9) < recursive_candidate_order_key(score, 8),
        "the old descending exhaustive scan retained the larger split on a tie"
    );
    assert!(
        recursive_candidate_order_key((99, 98, 1, 0), 1) < recursive_candidate_order_key(score, 9),
        "the exact layout score must remain the primary objective"
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn recursive_frontier_retains_linf_and_smaller_l2_rank() {
    use akita_config::{policy_of, proof_optimized::fp128::D64OneHot, CommitmentConfig};
    use akita_types::InnerCommitSecurityRoute;

    let policy = policy_of::<D64OneHot>();
    let challenge = D64OneHot::ring_challenge_config(64).expect("D64 challenge");
    let candidates = derive_candidate_level_params_frontier(
        &policy,
        akita_types::CommitmentPayloadMode::Compressed,
        &challenge,
        CommitmentRingDims::uniform(64),
        948_672,
        4,
        3,
        None,
        TensorChallengeShape::Flat,
    )
    .expect("late-fold rank frontier");
    let linf_rank = candidates
        .iter()
        .find_map(|(params, _)| {
            matches!(
                params.inner_commit_matrix.security_route(),
                InnerCommitSecurityRoute::Linf(_)
            )
            .then(|| params.inner_commit_matrix.output_rank())
        })
        .expect("L-infinity fallback");
    let l2_rank = candidates
        .iter()
        .find_map(|(params, _)| {
            matches!(
                params.inner_commit_matrix.security_route(),
                InnerCommitSecurityRoute::L2 { .. }
            )
            .then(|| params.inner_commit_matrix.output_rank())
        })
        .expect("measured L2 candidate");
    assert!(l2_rank < linf_rank);
    assert_eq!(
        candidates
            .iter()
            .map(|(params, _)| params.inner_commit_matrix.output_rank())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        candidates.len(),
        "frontier keeps at most one local layout per secure A rank"
    );
}
