use super::recursive::{recursive_candidate_order_key, recursive_split_search_domain};
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
fn scalar_next_witness_len_rejects_multi_group_root_level_params() {
    let grouped = grouped_level_params();
    let err = planned_next_witness_len(128, &grouped, 1, 1)
        .expect_err("multi-group root suffix sizing must use output_witness_len");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
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

#[test]
fn recursive_split_policy_controls_the_shared_search_domain() {
    assert_eq!(
        recursive_split_search_domain(
            crate::RecursiveSplitSearchPolicy::BoundedBalancedExtremesV1,
            1 << 12,
            12,
            4,
            4,
            1,
        ),
        (1..12).rev().collect::<Vec<_>>()
    );
    assert_eq!(
        recursive_split_search_domain(
            crate::RecursiveSplitSearchPolicy::BoundedBalancedExtremesV1,
            1 << 16,
            16,
            4,
            4,
            1,
        ),
        vec![15, 10, 9, 8, 7, 6, 1]
    );
    assert_eq!(
        recursive_split_search_domain(
            crate::RecursiveSplitSearchPolicy::Exhaustive,
            1 << 16,
            16,
            4,
            4,
            1,
        ),
        (1..16).rev().collect::<Vec<_>>()
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn response_model_deduplicates_equal_linf_and_adds_one_l2_split() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};
    use akita_types::InnerCommitSecurityRoute;

    let policy = policy_of::<OneHot>();
    let challenge = OneHot::ring_challenge_config(64).expect("D64 challenge");
    let candidates = derive_candidate_level_params(
        None,
        &policy,
        akita_types::CommitmentPayloadMode::Compressed,
        &challenge,
        CommitmentRingDims::uniform(64),
        948_672,
        crate::InnerBasisSource::BalancedDigits { log_basis: 4 },
        4,
        4,
        3,
        None,
        Some(crate::response_model::SourceMomentEstimate::new(1_000_000).unwrap()),
    )
    .expect("modeled late-fold candidates");
    let linf = candidates
        .iter()
        .filter(|(params, _)| {
            matches!(
                params.inner_commit_matrix.security_route(),
                InnerCommitSecurityRoute::Linf(_)
            )
        })
        .count();
    let l2 = candidates
        .iter()
        .filter(|(params, _)| {
            matches!(
                params.inner_commit_matrix.security_route(),
                InnerCommitSecurityRoute::L2 { .. }
            )
        })
        .count();
    assert_eq!(linf, 1);
    assert_eq!(l2, 1);
    assert_eq!(candidates.len(), 2);
}

#[cfg(feature = "catalog-gen")]
#[test]
fn setup_prefix_frontier_excludes_unsupported_compression_sources() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    let challenge = Recursive::ring_challenge_config(64).expect("challenge config");
    let mut cache = SetupPrefixSearchCache::default();
    for log_prefix in 12..=20 {
        let groups = derive_setup_prefix_groups(
            &mut cache,
            &policy,
            &challenge,
            3,
            1usize << log_prefix,
            1,
            64,
            64,
        )
        .expect("setup-prefix frontier");
        for params in groups {
            akita_types::setup_prefix_slot_field_elements(&akita_types::setup_prefix_slot_id(
                1usize << log_prefix,
                params,
            ))
            .expect("frontier candidate must support its compression source");
        }
    }
}
