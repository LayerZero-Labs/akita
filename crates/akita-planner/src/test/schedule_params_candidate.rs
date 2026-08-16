use super::recursive::{
    recursive_candidate_order_key, recursive_split_lower_bound, recursive_split_search_domain,
    RecursiveSplitLowerBoundInput,
};
use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_types::{PolynomialGroupLayout, SisModulusProfileId};

fn synthetic_profile(
    group: PolynomialGroupLayout,
    params: &CommittedGroupParams,
) -> CommittedGroupProfile {
    CommittedGroupProfile {
        version: CommittedGroupProfile::VERSION,
        source_encoding: akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
        group,
        num_live_ring_elements_per_claim: params.num_live_ring_elements_per_claim,
        num_positions_per_block: params.num_positions_per_block,
        num_live_blocks: params.num_live_blocks,
        outer_slice_count: params.outer_slice_count,
        log_basis_inner: params.log_basis_inner,
        num_digits_inner: params.num_digits_inner,
        inner_commit_matrix: params.inner_commit_matrix,
        log_basis_outer: params.log_basis_outer,
        num_digits_outer: params.num_digits_outer,
        outer_commit_matrix: params.outer_commit_matrix,
    }
}

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
        layout: synthetic_profile(PolynomialGroupLayout::new(6, 1), &precommitted),
        opening: akita_types::GroupOpeningPlan::evaluation_trace(
            precommitted.fold_challenge_config,
            precommitted.log_basis_open,
            precommitted.num_digits_open,
            precommitted.num_digits_fold,
        ),
    }];
    params
}

#[test]
fn scalar_next_witness_len_rejects_multi_group_root_level_params() {
    let grouped = grouped_level_params();
    let err = planned_next_witness_len(128, 1, &grouped, 1, 1)
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
fn recursive_split_bound_prices_packing_e_at_its_physical_width() {
    let input = RecursiveSplitLowerBoundInput {
        num_ring_elems: 1 << 12,
        ring_dimension: 256,
        opening_width: 128,
        reduced_vars: 12,
        r: 6,
        delta_commit: 3,
        delta_open: 4,
        num_chunks: 8,
    };
    let blocks = (1usize << 12).div_ceil(1 << 6);
    let expected_body = blocks * 4 * 128 + blocks * 4 * 256 + (1 << 6) * 3 * 8 * 256;
    assert_eq!(
        recursive_split_lower_bound(input),
        Some(expected_body + 2 * blocks)
    );
    assert!(
        recursive_split_lower_bound(RecursiveSplitLowerBoundInput {
            opening_width: 256,
            ..input
        }) > recursive_split_lower_bound(input)
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
fn response_model_deduplicates_linf_and_keeps_one_l2_split() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};
    use akita_types::InnerCommitSecurityRoute;

    let policy = policy_of::<OneHot>();
    let challenge = OneHot::ring_challenge_config(64).expect("D64 challenge");
    let candidates = derive_candidate_level_params(
        None,
        &policy,
        akita_types::CommitmentPayloadMode::Compressed,
        PlannerOpeningCandidate::evaluation_trace(challenge),
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
    assert!(l2 > 0);
    let l2_block_index_bits = candidates
        .iter()
        .filter_map(|(params, _)| {
            matches!(
                params.inner_commit_matrix.security_route(),
                InnerCommitSecurityRoute::L2 { .. }
            )
            .then_some(params.block_index_bits())
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(l2_block_index_bits.len(), 1);
}

#[cfg(feature = "catalog-gen")]
#[test]
fn recursive_packing_candidate_uses_exact_geometry_and_linf_route() {
    use akita_config::{policy_of, proof_optimized::fp64::Dense};
    use akita_types::{InnerCommitSecurityRoute, OpeningMethod};

    let policy = policy_of::<Dense>();
    let dimensions = CommitmentRingDims {
        inner: 256,
        outer: 128,
        opening: 64,
    };
    let opening =
        PlannerOpeningCandidate::coefficient_packing(1, policy.claim_ext_degree, dimensions, 64)
            .expect("packing geometry");
    let candidates = derive_candidate_level_params(
        None,
        &policy,
        akita_types::CommitmentPayloadMode::Compressed,
        opening,
        dimensions,
        948_672,
        crate::InnerBasisSource::BalancedDigits { log_basis: 3 },
        3,
        3,
        1,
        None,
        Some(crate::response_model::SourceMomentEstimate::new(1_000_000).unwrap()),
    )
    .expect("packing candidates");
    assert!(!candidates.is_empty());
    for (params, next_witness_len) in &candidates {
        assert_eq!(
            params.opening_method,
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64
            }
        );
        assert_eq!(
            params.source_encoding,
            akita_types::CommittedSourceEncoding::CanonicalCoefficientTable
        );
        assert!(matches!(
            params.inner_commit_matrix.security_route(),
            InnerCommitSecurityRoute::Linf(_)
        ));
        assert_eq!(
            params.open_commit_matrix.input_width(),
            akita_types::opening_d_segment_width(
                params.opening_method,
                policy.claim_ext_degree,
                dimensions.d_a(),
                dimensions.d_d(),
                params.num_digits_open,
                params.num_live_blocks,
                1,
            )
            .unwrap()
        );
        assert_eq!(
            *next_witness_len,
            planned_next_witness_len(
                policy.decomposition.field_bits(),
                policy.claim_ext_degree,
                params,
                1,
                policy.chunks_at_level(1),
            )
            .unwrap()
            .unwrap()
        );
    }
    let mut prefix_cache = SetupPrefixSearchCache::default();
    let with_prefix = derive_candidate_level_params(
        Some(&mut prefix_cache),
        &policy,
        akita_types::CommitmentPayloadMode::Compressed,
        opening,
        dimensions,
        948_672,
        crate::InnerBasisSource::BalancedDigits { log_basis: 3 },
        3,
        3,
        1,
        Some(1 << 14),
        Some(crate::response_model::SourceMomentEstimate::new(1_000_000).unwrap()),
    )
    .expect("packing candidates with setup prefix");
    assert!(!with_prefix.is_empty());
    for (params, next_witness_len) in with_prefix {
        let prefix = params.setup_prefix.as_ref().expect("attached setup prefix");
        assert_eq!(
            prefix.commitment_params.opening.opening_method,
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64
            }
        );
        assert_eq!(
            prefix.commitment_params.layout.source_encoding,
            akita_types::CommittedSourceEncoding::CanonicalCoefficientTable
        );
        let d_d = params.role_dims().d_d();
        let witness_width = akita_types::opening_d_segment_width(
            params.opening_method,
            policy.claim_ext_degree,
            params.d_a(),
            d_d,
            params.num_digits_open,
            params.num_live_blocks,
            1,
        )
        .unwrap();
        let prefix_width = prefix
            .commitment_params
            .d_segment_width(policy.claim_ext_degree, d_d)
            .unwrap();
        assert_eq!(
            params.open_commit_matrix.input_width(),
            witness_width + prefix_width
        );
        assert_eq!(
            next_witness_len,
            planned_next_witness_len(
                policy.decomposition.field_bits(),
                policy.claim_ext_degree,
                &params,
                1,
                policy.chunks_at_level(1),
            )
            .unwrap()
            .unwrap()
        );
    }
}

#[cfg(feature = "catalog-gen")]
#[test]
fn packing_split_bounds_preserve_the_exhaustive_candidate_frontier() {
    use akita_config::{
        policy_of,
        proof_optimized::{fp128, fp32, fp64},
    };

    let cases = [
        (
            policy_of::<fp128::Dense>(),
            CommitmentRingDims {
                inner: 128,
                outer: 128,
                opening: 64,
            },
            4,
        ),
        (
            policy_of::<fp64::Dense>(),
            CommitmentRingDims {
                inner: 256,
                outer: 128,
                opening: 64,
            },
            3,
        ),
        (
            policy_of::<fp32::Dense>(),
            CommitmentRingDims {
                inner: 1024,
                outer: 128,
                opening: 64,
            },
            3,
        ),
    ];
    for (policy, dimensions, log_basis) in cases {
        let opening = PlannerOpeningCandidate::coefficient_packing(
            1,
            policy.claim_ext_degree,
            dimensions,
            64,
        )
        .expect("production packing geometry");
        let derive = |without_bounds| {
            let arguments = (
                None,
                &policy,
                akita_types::CommitmentPayloadMode::Compressed,
                opening,
                dimensions,
                948_672,
                crate::InnerBasisSource::BalancedDigits { log_basis },
                log_basis,
                log_basis,
                1,
                None,
                Some(crate::response_model::SourceMomentEstimate::new(1_000_000).unwrap()),
            );
            if without_bounds {
                derive_candidate_level_params_split_frontier_without_bounds(
                    arguments.0,
                    arguments.1,
                    arguments.2,
                    arguments.3,
                    arguments.4,
                    arguments.5,
                    arguments.6,
                    arguments.7,
                    arguments.8,
                    arguments.9,
                    arguments.10,
                    arguments.11,
                )
            } else {
                derive_candidate_level_params_split_frontier(
                    arguments.0,
                    arguments.1,
                    arguments.2,
                    arguments.3,
                    arguments.4,
                    arguments.5,
                    arguments.6,
                    arguments.7,
                    arguments.8,
                    arguments.9,
                    arguments.10,
                    arguments.11,
                )
            }
        };
        let canonical = |candidates: Vec<(CommittedGroupParams, usize)>| {
            candidates
                .into_iter()
                .map(|(params, next)| (params.canonical_descriptor_bytes(), next))
                .collect::<std::collections::BTreeSet<_>>()
        };
        let exhaustive = canonical(derive(true).expect("bounds-disabled frontier"));
        assert!(!exhaustive.is_empty());
        assert_eq!(
            canonical(derive(false).expect("bounded frontier")),
            exhaustive,
            "split bounds must not change the exact frontier for {dimensions:?}",
        );
    }
}

#[cfg(feature = "catalog-gen")]
#[test]
fn root_packing_candidates_use_adversarial_linf_and_exact_d_width() {
    use akita_config::{
        honest_fold_policy_of, policy_of, proof_optimized::fp64::Dense, CommitmentConfig,
    };
    use akita_types::{AkitaScheduleLookupKey, InnerCommitSecurityRoute, OpeningMethod};

    let policy = policy_of::<Dense>();
    let dimensions = CommitmentRingDims {
        inner: 256,
        outer: 128,
        opening: 64,
    };
    let opening =
        PlannerOpeningCandidate::coefficient_packing(0, policy.claim_ext_degree, dimensions, 64)
            .unwrap();
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(16, 2));
    let candidates = crate::planner::root_level_candidates_for_basis(
        &key,
        honest_fold_policy_of::<Dense>(),
        &[],
        &policy,
        dimensions,
        opening,
        &[],
        1 << 16,
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        false,
    )
    .expect("root packing candidates");
    assert!(!candidates.is_empty());
    let (first_params, first_next_witness_len) = &candidates[0];
    let (packing_direct_bytes, _) =
        akita_schedules::planner_support::nonterminal_level_payload_bytes(
            &policy,
            0,
            key.final_group,
            first_params,
            None,
            1 << 16,
            *first_next_witness_len,
        )
        .expect("packing level payload");
    assert_eq!(
        packing_direct_bytes,
        akita_types::level_proof_bytes(
            policy.decomposition.field_bits(),
            policy.challenge_field_bits().unwrap(),
            first_params,
            None,
            *first_next_witness_len,
            Some(akita_types::NextWitnessBindingPolicy::TerminalInnerState),
        )
        .expect("packing direct payload without EOR"),
    );
    assert!(
        akita_types::extension_opening_reduction_level_bytes(
            policy.challenge_field_bits().unwrap(),
            policy.claim_ext_degree,
            0,
            key.final_group,
            1 << 16,
            first_params.d_a(),
        )
        .expect("legacy EOR price")
            > 0,
        "packing must skip a nonzero legacy EOR payload",
    );
    for (params, next_witness_len) in &candidates {
        assert_eq!(
            params.opening_method,
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64
            }
        );
        assert!(matches!(
            params.inner_commit_matrix.security_route(),
            InnerCommitSecurityRoute::Linf(_)
        ));
        assert_eq!(
            params.open_commit_matrix.input_width(),
            akita_types::opening_d_segment_width(
                params.opening_method,
                policy.claim_ext_degree,
                dimensions.d_a(),
                dimensions.d_d(),
                params.num_digits_open,
                params.num_live_blocks,
                key.final_group.num_polynomials(),
            )
            .unwrap()
        );
        let opening_batch = key.opening_layout().unwrap();
        assert_eq!(
            *next_witness_len,
            params
                .output_witness_len_for_field_bits(
                    policy.decomposition.field_bits(),
                    policy.claim_ext_degree,
                    &opening_batch,
                )
                .unwrap()
        );
    }
    let frozen_group = synthetic_profile(key.final_group, &candidates[0].0);
    let grouped_key = AkitaScheduleLookupKey {
        final_group: key.final_group,
        precommitteds: vec![frozen_group],
    };
    let precommit_opening =
        PlannerOpeningCandidate::coefficient_packing(0, policy.claim_ext_degree, dimensions, 128)
            .unwrap();
    let grouped = crate::planner::root_level_candidates_for_basis(
        &grouped_key,
        honest_fold_policy_of::<Dense>(),
        &[honest_fold_policy_of::<Dense>()],
        &policy,
        dimensions,
        opening,
        &[precommit_opening],
        1 << 16,
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        false,
    )
    .expect("group-local packing candidates");
    assert!(!grouped.is_empty());
    for (params, _) in grouped {
        assert_eq!(params.precommitted_groups.len(), 1);
        assert_eq!(
            params.precommitted_groups[0].opening.opening_method,
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 128
            }
        );
        let d_d = params.role_dims().d_d();
        let final_width = akita_types::opening_d_segment_width(
            params.opening_method,
            policy.claim_ext_degree,
            params.d_a(),
            d_d,
            params.num_digits_open,
            params.num_live_blocks,
            grouped_key.final_group.num_polynomials(),
        )
        .unwrap();
        let precommit_width = params.precommitted_groups[0]
            .d_segment_width(policy.claim_ext_degree, d_d)
            .unwrap();
        assert_eq!(
            params.open_commit_matrix.input_width(),
            final_width + precommit_width
        );
    }
    let trace_precommit = PlannerOpeningCandidate::evaluation_trace(
        SparseChallengeConfig::production_for_ring_dim(dimensions.d_a()).unwrap(),
    );
    assert!(crate::planner::root_level_candidates_for_basis(
        &grouped_key,
        honest_fold_policy_of::<Dense>(),
        &[honest_fold_policy_of::<Dense>()],
        &policy,
        dimensions,
        opening,
        &[trace_precommit],
        1 << 16,
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        false,
    )
    .unwrap()
    .is_empty());

    let product_key = AkitaScheduleLookupKey {
        final_group: grouped_key.final_group,
        precommitteds: vec![frozen_group, frozen_group],
    };
    let opening_products = crate::schedule_params::suffix_dp::packing_precommit_opening_products(
        &policy,
        dimensions,
        &product_key,
    )
    .expect("root precommit opening products");
    assert_eq!(opening_products.len(), 4);
    assert!(opening_products
        .iter()
        .all(|assignment| assignment.len() == 2));
    assert!(opening_products.iter().flatten().all(|opening| matches!(
        opening.method(),
        OpeningMethod::SubringCoefficientPacking { .. }
    )));
    let mut tensor_profile = product_key.precommitteds[0];
    tensor_profile.source_encoding =
        akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
            extension_degree: policy.claim_ext_degree,
        };
    let tensor_key = AkitaScheduleLookupKey {
        final_group: product_key.final_group,
        precommitteds: vec![tensor_profile],
    };
    assert!(
        crate::schedule_params::suffix_dp::packing_precommit_opening_products(
            &policy,
            dimensions,
            &tensor_key,
        )
        .unwrap()
        .is_empty()
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn tensor_frozen_precommit_uses_uniform_evaluation_trace_fallback() {
    use akita_config::{
        honest_fold_policy_of, policy_of, proof_optimized::fp64::Dense, CommitmentConfig,
    };
    use akita_types::{AkitaScheduleLookupKey, OpeningMethod};

    let mut policy = policy_of::<Dense>();
    let dimensions = CommitmentRingDims::uniform(256);
    policy.uniform_ring_dimension = 256;
    policy.ring_dimension_schedule_mode = crate::RingDimensionScheduleMode::UniformDimension {
        ring_dimension: 256,
    };
    policy.selection_policy = crate::SelectionPolicyId::for_policy(
        policy.recursive_setup_planning,
        policy.ring_dimension_schedule_mode,
    );
    let pre_group = PolynomialGroupLayout::new(14, 1);
    let pre_key = AkitaScheduleLookupKey::single(pre_group);
    let pre_candidates = crate::planner::root_level_candidates_for_basis(
        &pre_key,
        honest_fold_policy_of::<Dense>(),
        &[],
        &policy,
        dimensions,
        PlannerOpeningCandidate::evaluation_trace(Dense::ring_challenge_config(256).unwrap()),
        &[],
        1 << 14,
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        false,
    )
    .expect("standalone precommit candidates");
    let mut precommitted = CommittedGroupProfile::try_from_params(
        pre_group,
        &pre_candidates.first().expect("precommit candidate").0,
    )
    .expect("standalone precommit profile");
    precommitted.source_encoding = akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
        extension_degree: 2,
    };
    precommitted
        .validate_frozen_precommit(policy.decomposition.field_bits())
        .expect("tensor source is valid for the frozen A ring");
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(16, 1),
        precommitteds: vec![precommitted],
    };
    let planned = crate::planner::find_schedule(
        &key,
        honest_fold_policy_of::<Dense>(),
        &[honest_fold_policy_of::<Dense>()],
        &policy,
        Dense::ring_challenge_config,
    )
    .expect("grouped ET fallback schedule");
    let root = &planned.schedule.root.params;
    assert_eq!(
        root.final_group.commitment.opening_method,
        OpeningMethod::EvaluationTrace,
    );
    assert!(root.precommitted_groups.iter().all(|group| {
        group.commitment.opening.opening_method == OpeningMethod::EvaluationTrace
    }));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn setup_prefix_cache_separates_equal_width_opening_methods() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, RecursiveCommitmentConfig};

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    let dimensions = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let challenge = SparseChallengeConfig::production_for_ring_dim(128).unwrap();
    let trace = PlannerOpeningCandidate::evaluation_trace(challenge);
    let exact_packing =
        PlannerOpeningCandidate::coefficient_packing(1, policy.claim_ext_degree, dimensions, 128)
            .unwrap();
    let reduced_packing =
        PlannerOpeningCandidate::coefficient_packing(1, policy.claim_ext_degree, dimensions, 64)
            .unwrap();
    let mut cache = SetupPrefixSearchCache::default();
    let request = |opening| SetupPrefixSearchRequest {
        policy: &policy,
        opening,
        log_basis_open: 3,
        n_prefix: 1 << 14,
        num_chunks: 1,
        inner_ring_dimension: dimensions.d_a(),
        outer_ring_dimension: dimensions.d_b(),
    };
    let trace_groups = derive_setup_prefix_groups(&mut cache, request(trace)).unwrap();
    let exact_groups = derive_setup_prefix_groups(&mut cache, request(exact_packing)).unwrap();
    let reduced_groups = derive_setup_prefix_groups(&mut cache, request(reduced_packing)).unwrap();
    assert!(!trace_groups.is_empty() && !exact_groups.is_empty() && !reduced_groups.is_empty());
    assert!(trace_groups.iter().all(|group| {
        group.opening.opening_method == akita_types::OpeningMethod::EvaluationTrace
    }));
    assert!(exact_groups.iter().all(|group| {
        group.opening.opening_method
            == akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 128,
            }
    }));
    assert!(reduced_groups.iter().all(|group| {
        group.opening.opening_method
            == akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64,
            }
    }));
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
            SetupPrefixSearchRequest {
                policy: &policy,
                opening: PlannerOpeningCandidate::evaluation_trace(challenge),
                log_basis_open: 3,
                n_prefix: 1usize << log_prefix,
                num_chunks: 1,
                inner_ring_dimension: 64,
                outer_ring_dimension: 64,
            },
        )
        .expect("setup-prefix frontier");
        for params in groups {
            akita_types::setup_prefix_slot_field_elements(
                &akita_types::scheduled_setup_prefix(1usize << log_prefix, params).slot_id(),
            )
            .expect("frontier candidate must support its compression source");
        }
    }
}

#[cfg(feature = "catalog-gen")]
#[test]
fn shared_ab_derivation_centralizes_rank_and_compression_rejection() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot};

    struct FixedFoldPolicy;

    impl HonestFoldPolicy for FixedFoldPolicy {
        fn num_digits_fold(&self, _query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError> {
            Ok(2)
        }
    }

    let policy = policy_of::<OneHot>();
    let challenge = SparseChallengeConfig::pm1_only(3);
    let candidate = |dimensions, outer_slice_count, width_s| {
        derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
            policy: &policy,
            fold_policy: &FixedFoldPolicy,
            ring_challenge_cfg: &challenge,
            dimensions,
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            num_claims: 1,
            num_live_ring_elements_per_claim: 64,
            num_live_blocks: 8,
            num_positions_per_block: 8,
            num_chunks: 1,
            outer_slice_count,
            witness_norms: FoldWitnessNorms::bounded(3, dimensions.d_a()),
            log_basis_open: 3,
            width_s,
            num_digits_outer: 2,
            modeled_linf_cap: None,
        })
        .unwrap()
    };

    for outer_slice_count in akita_types::CommitmentSliceCount::ALL {
        assert!(
            candidate(CommitmentRingDims::uniform(64), outer_slice_count, 8).is_some(),
            "shared A/B request should admit S={}",
            outer_slice_count.get(),
        );
    }

    assert!(candidate(
        CommitmentRingDims::uniform(128),
        akita_types::CommitmentSliceCount::FOUR,
        8,
    )
    .is_some());
    assert!(candidate(
        CommitmentRingDims::uniform(128),
        akita_types::CommitmentSliceCount::EIGHT,
        8,
    )
    .is_none());

    assert!(
        derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
            policy: &policy,
            fold_policy: &FixedFoldPolicy,
            ring_challenge_cfg: &challenge,
            dimensions: CommitmentRingDims::uniform(64),
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            num_claims: 1,
            num_live_ring_elements_per_claim: 64,
            num_live_blocks: 8,
            num_positions_per_block: 8,
            num_chunks: 1,
            outer_slice_count: akita_types::CommitmentSliceCount::ONE,
            witness_norms: FoldWitnessNorms::bounded(3, 64),
            log_basis_open: 3,
            width_s: usize::MAX,
            num_digits_outer: 2,
            modeled_linf_cap: None,
        })
        .is_err()
    );

    struct OversizedFoldPolicy;

    impl HonestFoldPolicy for OversizedFoldPolicy {
        fn num_digits_fold(&self, _query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError> {
            Ok(1 << 20)
        }
    }

    assert!(
        derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
            policy: &policy,
            fold_policy: &OversizedFoldPolicy,
            ring_challenge_cfg: &challenge,
            dimensions: CommitmentRingDims::uniform(64),
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            num_claims: 1,
            num_live_ring_elements_per_claim: 64,
            num_live_blocks: 8,
            num_positions_per_block: 8,
            num_chunks: 1,
            outer_slice_count: akita_types::CommitmentSliceCount::ONE,
            witness_norms: FoldWitnessNorms::bounded(3, 64),
            log_basis_open: 3,
            width_s: 8,
            num_digits_outer: 2,
            modeled_linf_cap: None,
        })
        .unwrap()
        .is_none()
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn raw_candidate_is_not_subject_to_the_compression_source_cap() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot};

    struct FixedFoldPolicy;

    impl HonestFoldPolicy for FixedFoldPolicy {
        fn num_digits_fold(&self, _query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError> {
            Ok(2)
        }
    }

    let policy = policy_of::<OneHot>();
    let challenge = SparseChallengeConfig::pm1_only(3);
    let dimensions = CommitmentRingDims::uniform(256);
    let num_claims = 1;
    let width_s = 8;
    let mut raw_candidate = derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
        policy: &policy,
        fold_policy: &FixedFoldPolicy,
        ring_challenge_cfg: &challenge,
        dimensions,
        payload_mode: akita_types::CommitmentPayloadMode::Raw,
        num_claims,
        num_live_ring_elements_per_claim: 64,
        num_live_blocks: 8,
        num_positions_per_block: 8,
        num_chunks: 1,
        outer_slice_count: akita_types::CommitmentSliceCount::ONE,
        witness_norms: FoldWitnessNorms::bounded(3, dimensions.d_a()),
        log_basis_open: 3,
        width_s,
        num_digits_outer: 2,
        modeled_linf_cap: None,
    })
    .unwrap()
    .expect("raw candidate has certified minimum A/B ranks");
    let outer = raw_candidate.outer_commit_matrix;
    let field_bytes = outer.sis_modulus_profile().field_bits().div_ceil(8) as usize;
    let over_cap_rank =
        akita_types::MAX_COMPRESSION_INPUT_BYTES.div_ceil(dimensions.d_b() * field_bytes) + 1;
    raw_candidate.outer_commit_matrix = OuterCommitMatrixParams::try_new(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        over_cap_rank.max(outer.output_rank()),
        outer.input_width(),
        outer.coeff_linf_bound(),
        outer.ring_dimension(),
    )
    .expect("larger-than-minimum rank remains SIS certified");

    let mut params = CommittedGroupParams::params_only(
        policy.sis_modulus_profile,
        dimensions.d_a(),
        3,
        raw_candidate.inner_commit_matrix.output_rank(),
        raw_candidate.outer_commit_matrix.output_rank(),
        1,
        challenge,
    )
    .with_decomp(width_s, width_s * 8, 1, 2, 2)
    .unwrap();
    params.payload_mode = akita_types::CommitmentPayloadMode::Raw;
    params.inner_commit_matrix = raw_candidate.inner_commit_matrix;
    params.outer_commit_matrix = raw_candidate.outer_commit_matrix;
    params.num_digits_fold = raw_candidate.num_digits_fold;
    assert!(params.compression_sources_supported().unwrap());
    params
        .validate_commitment_request(2, num_claims)
        .expect("raw S1 geometry does not execute compression");

    let mut compressed = params;
    compressed.payload_mode = akita_types::CommitmentPayloadMode::Compressed;
    assert!(!compressed.compression_sources_supported().unwrap());
    assert!(compressed
        .validate_commitment_request(2, num_claims)
        .is_err());
}
