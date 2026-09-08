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
) -> GroupCommitPhaseParams {
    GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        group,
        blocks: params.blocks(),

        outer_slice_count: params.outer_slice_count(),
        inner: akita_types::RoleParams::new(
            akita_types::GadgetDigits::new(
                params.inner().digits.log_basis,
                params.inner().digits.num_digits,
            ),
            params.inner().matrix,
        ),
        outer: akita_types::RoleParams::new(
            akita_types::GadgetDigits::new(
                params.outer().digits.log_basis,
                params.outer().digits.num_digits,
            ),
            params.outer().matrix,
        ),
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
    params
        .set_precommitted_groups(vec![GroupOpenPhaseParams {
            setup_natural_len: None,
            profile: synthetic_profile(PolynomialGroupLayout::new(6, 1), &precommitted),
            opening: akita_types::GroupOpeningPlan::evaluation_trace(
                precommitted.fold_challenge_config(),
                precommitted.open().digits.log_basis,
                precommitted.open().digits.num_digits,
                precommitted.num_digits_fold(),
            ),
        }])
        .expect("valid precommitted group topology");
    params
}

#[cfg(feature = "catalog-gen")]
fn audited_grouped_level_params() -> CommittedGroupParams {
    use akita_types::{
        InnerCommitMatrixParams, OpenCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole,
    };

    let params = grouped_level_params();
    let key = |role| {
        akita_types::sis::sis_table_key_for_linf_bound(
            akita_types::sis::DEFAULT_SIS_SECURITY_POLICY,
            akita_types::SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            role,
            64,
            1,
        )
        .expect("synthetic matrix has an audited SIS bucket")
    };
    let groups = params
        .groups()
        .iter()
        .cloned()
        .map(|mut group| {
            group.opening.fold_challenge_config =
                SparseChallengeConfig::production_for_ring_dim(64)
                    .expect("production synthetic challenge");
            group.profile.inner.matrix = InnerCommitMatrixParams::try_new_with_min_rank(
                key(SisMatrixRole::Inner),
                group.profile.inner.matrix.input_width(),
            )
            .expect("audited synthetic A matrix");
            let outer_width = akita_types::CommitmentSliceGeometry::try_new(
                group.profile.outer_slice_count,
                group.profile.blocks.live_blocks,
                1,
                group.profile.inner.matrix.output_rank(),
                group.profile.outer.digits.num_digits,
                group.profile.inner.matrix.ring_dimension(),
                group.profile.outer.matrix.ring_dimension(),
            )
            .expect("synthetic B geometry")
            .physical_input_width();
            group.profile.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(
                key(SisMatrixRole::Outer),
                outer_width,
            )
            .expect("audited synthetic B matrix");
            group
        })
        .collect();
    let open_matrix = OpenCommitMatrixParams::try_new_with_min_rank(
        key(SisMatrixRole::Open),
        params.open().matrix.input_width(),
    )
    .expect("audited synthetic D matrix");
    CommittedGroupParams::try_new(
        groups,
        open_matrix,
        params.payload_mode,
        params.ring_relation_mode,
        params.source_encoding,
        params.witness_chunk,
    )
    .expect("audited grouped params")
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
    let candidates = derive_fold_candidates(
        RecursiveCandidateRequest {
            policy: &policy,
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            opening: PlannerOpeningCandidate::evaluation_trace(challenge),
            dimensions: CommitmentRingDims::uniform(64),
            current_witness_len: 948_672,
            source: crate::InnerBasisSource::BalancedDigits { log_basis: 4 },
            log_basis_inner: 4,
            log_basis_open: 4,
            fold_level: 3,
            source_moment: Some(
                crate::response_model::SourceMomentEstimate::new(1_000_000).unwrap(),
            ),
            relation_traversal_order: RelationTraversalOrder::Canonical,
            guide: None,
        },
        RecursiveFoldWork::direct(RelationSearchDomain::QuotientOnly),
        FoldCandidatePolicy::Best,
    )
    .expect("modeled late-fold candidates");
    let linf = candidates
        .iter()
        .filter(|(candidate, _)| {
            matches!(
                candidate.inner().matrix.security_route(),
                InnerCommitSecurityRoute::Linf(_)
            )
        })
        .count();
    let l2 = candidates
        .iter()
        .filter(|(candidate, _)| {
            matches!(
                candidate.inner().matrix.security_route(),
                InnerCommitSecurityRoute::L2 { .. }
            )
        })
        .count();
    assert_eq!(linf, 1);
    assert!(l2 > 0);
    let l2_block_index_bits = candidates
        .iter()
        .filter_map(|(candidate, _)| {
            matches!(
                candidate.inner().matrix.security_route(),
                InnerCommitSecurityRoute::L2 { .. }
            )
            .then_some(candidate.block_index_bits())
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
            .expect("valid packing request")
            .expect("packing geometry");
    let request = RecursiveCandidateRequest {
        policy: &policy,
        payload_mode: akita_types::CommitmentPayloadMode::Compressed,
        opening,
        dimensions,
        current_witness_len: 948_672,
        source: crate::InnerBasisSource::BalancedDigits { log_basis: 3 },
        log_basis_inner: 3,
        log_basis_open: 3,
        fold_level: 1,
        source_moment: Some(crate::response_model::SourceMomentEstimate::new(1_000_000).unwrap()),
        relation_traversal_order: RelationTraversalOrder::Canonical,
        guide: None,
    };
    let candidates = derive_fold_candidates(
        request,
        RecursiveFoldWork::direct(RelationSearchDomain::QuotientOnly),
        FoldCandidatePolicy::Best,
    )
    .expect("packing candidates");
    assert!(!candidates.is_empty());
    for (candidate, next_witness_len) in &candidates {
        let params = &candidate;
        assert_eq!(
            params.opening_method(),
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64
            }
        );
        assert_eq!(
            params.source_encoding,
            akita_types::CommittedSourceEncoding::CanonicalCoefficientTable
        );
        assert!(matches!(
            params.inner().matrix.security_route(),
            InnerCommitSecurityRoute::Linf(_)
        ));
        assert_eq!(
            params.open().matrix.input_width(),
            akita_types::opening_d_segment_width(
                params.opening_method(),
                policy.claim_ext_degree,
                dimensions.d_a(),
                dimensions.d_d(),
                params.open().digits.num_digits,
                params.blocks().live_blocks,
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
    let with_prefix = derive_fold_candidates(
        request,
        RecursiveFoldWork::setup_prefixed(&mut prefix_cache, 1 << 14),
        FoldCandidatePolicy::Best,
    )
    .expect("packing candidates with setup prefix");
    assert!(!with_prefix.is_empty());
    for (candidate, next_witness_len) in with_prefix {
        let params = candidate;
        let prefix = params.setup_prefix().expect("attached setup prefix");
        assert_eq!(
            prefix.opening.opening_method,
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64
            }
        );
        assert_eq!(
            akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
            akita_types::CommittedSourceEncoding::CanonicalCoefficientTable
        );
        let d_d = params.role_dims().d_d();
        let witness_width = akita_types::opening_d_segment_width(
            params.opening_method(),
            policy.claim_ext_degree,
            params.d_a(),
            d_d,
            params.open().digits.num_digits,
            params.blocks().live_blocks,
            1,
        )
        .unwrap();
        let prefix_width = prefix
            .d_segment_width(policy.claim_ext_degree, d_d)
            .unwrap();
        assert_eq!(
            params.open().matrix.input_width(),
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
        .expect("valid packing request")
        .expect("production packing geometry");
        let derive = |without_bounds| {
            let request = RecursiveCandidateRequest {
                policy: &policy,
                payload_mode: akita_types::CommitmentPayloadMode::Compressed,
                opening,
                dimensions,
                current_witness_len: 948_672,
                source: crate::InnerBasisSource::BalancedDigits { log_basis },
                log_basis_inner: log_basis,
                log_basis_open: log_basis,
                fold_level: 1,
                source_moment: Some(
                    crate::response_model::SourceMomentEstimate::new(1_000_000).unwrap(),
                ),
                relation_traversal_order: RelationTraversalOrder::Canonical,
                guide: None,
            };
            let split_bounds = if without_bounds {
                SplitBoundPolicy::DisabledForOracle
            } else {
                SplitBoundPolicy::Enabled
            };
            derive_fold_candidates(
                request,
                RecursiveFoldWork::direct(RelationSearchDomain::QuotientOnly),
                FoldCandidatePolicy::Frontier(split_bounds),
            )
        };
        let canonical = |candidates: Vec<(CommittedGroupParams, usize)>| {
            candidates
                .into_iter()
                .map(|(candidate, next)| (candidate.canonical_descriptor_bytes(), next))
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
            .unwrap()
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
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        None,
    )
    .expect("root packing candidates");
    assert!(!candidates.is_empty());
    let (first_params, first_next_witness_len) = &candidates[0];
    let opening_layout = key.opening_layout().expect("root opening layout");
    let terminal = akita_types::TerminalFoldParams::from_expanded_group(first_params.clone());
    let packing_payload = akita_schedules::planner_support::nonterminal_level_payload_bytes(
        &policy,
        first_params,
        &opening_layout,
        akita_types::FoldSuccessor::Terminal(&terminal),
        *first_next_witness_len,
    )
    .expect("packing level payload");
    let packing_direct_bytes = packing_payload.direct;
    assert_eq!(
        packing_direct_bytes,
        akita_types::level_proof_bytes(
            policy.decomposition.field_bits(),
            policy.challenge_field_bits().unwrap(),
            first_params,
            first_params
                .relation_address_geometry(
                    &opening_layout,
                    policy.claim_ext_degree,
                    terminal.d_a(),
                    *first_next_witness_len,
                )
                .unwrap(),
            None,
        )
        .expect("packing direct payload without EOR"),
    );
    assert!(
        akita_types::extension_opening_reduction_level_bytes(
            policy.challenge_field_bits().unwrap(),
            policy.claim_ext_degree,
            PolynomialGroupLayout::new(16, key.final_group.num_polynomials()),
        )
        .expect("legacy EOR price")
            > 0,
        "packing must skip a nonzero legacy EOR payload",
    );
    for (params, next_witness_len) in &candidates {
        assert_eq!(
            params.opening_method(),
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64
            }
        );
        assert!(matches!(
            params.inner().matrix.security_route(),
            InnerCommitSecurityRoute::Linf(_)
        ));
        assert_eq!(
            params.open().matrix.input_width(),
            akita_types::opening_d_segment_width(
                params.opening_method(),
                policy.claim_ext_degree,
                dimensions.d_a(),
                dimensions.d_d(),
                params.open().digits.num_digits,
                params.blocks().live_blocks,
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
            .unwrap()
            .unwrap();
    let grouped = crate::planner::root_level_candidates_for_basis(
        &grouped_key,
        honest_fold_policy_of::<Dense>(),
        &[honest_fold_policy_of::<Dense>()],
        &policy,
        dimensions,
        opening,
        &[precommit_opening],
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        None,
    )
    .expect("group-local packing candidates");
    assert!(!grouped.is_empty());
    for (params, _) in grouped {
        assert_eq!(params.precommitted_groups().len(), 1);
        assert_eq!(
            params.precommitted_groups()[0].opening.opening_method,
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 128
            }
        );
        let d_d = params.role_dims().d_d();
        let final_width = akita_types::opening_d_segment_width(
            params.opening_method(),
            policy.claim_ext_degree,
            params.d_a(),
            d_d,
            params.open().digits.num_digits,
            params.blocks().live_blocks,
            grouped_key.final_group.num_polynomials(),
        )
        .unwrap();
        let precommit_width = params.precommitted_groups()[0]
            .d_segment_width(policy.claim_ext_degree, d_d)
            .unwrap();
        assert_eq!(
            params.open().matrix.input_width(),
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
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        None,
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
        &[
            honest_fold_policy_of::<Dense>(),
            honest_fold_policy_of::<Dense>(),
        ],
        None,
    )
    .expect("root precommit opening products");
    assert_eq!(opening_products.len(), 3);
    assert!(opening_products
        .iter()
        .all(|assignment| assignment.len() == 2));
    assert!(opening_products.iter().flatten().all(|opening| matches!(
        opening.method(),
        OpeningMethod::SubringCoefficientPacking { .. }
    )));

    let precommit_domain = PlannerOpeningCandidate::coefficient_packing_domain(
        0,
        policy.claim_ext_degree,
        CommitmentRingDims {
            inner: frozen_group.inner.matrix.ring_dimension(),
            outer: frozen_group.outer.matrix.ring_dimension(),
            opening: dimensions.d_d(),
        },
    )
    .expect("precommit opening domain");
    let exhaustive_products = precommit_domain
        .iter()
        .flat_map(|&left| precommit_domain.iter().map(move |&right| vec![left, right]))
        .collect::<Vec<_>>();
    let materialized_root_domain = |products: &[Vec<PlannerOpeningCandidate>]| {
        products
            .iter()
            .flat_map(|product| {
                crate::planner::root_level_candidates_for_basis(
                    &product_key,
                    honest_fold_policy_of::<Dense>(),
                    &[
                        honest_fold_policy_of::<Dense>(),
                        honest_fold_policy_of::<Dense>(),
                    ],
                    &policy,
                    dimensions,
                    opening,
                    product,
                    Dense::inner_basis_range().0,
                    Dense::opening_basis_range().0,
                    None,
                )
                .expect("materialized root candidate domain")
            })
            .map(|(params, output_witness_len)| {
                (params.canonical_descriptor_bytes(), output_witness_len)
            })
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert_eq!(
        materialized_root_domain(&opening_products),
        materialized_root_domain(&exhaustive_products),
        "one canonical representative per multiset must preserve the exhaustive root domain"
    );

    let repeated_key = AkitaScheduleLookupKey {
        final_group: grouped_key.final_group,
        precommitteds: vec![frozen_group; 16],
    };
    let repeated_products = crate::schedule_params::suffix_dp::packing_precommit_opening_products(
        &policy,
        dimensions,
        &repeated_key,
        &vec![honest_fold_policy_of::<Dense>(); 16],
        None,
    )
    .expect("symmetric root precommit opening products");
    assert_eq!(repeated_products.len(), 17);
    assert!(repeated_products
        .iter()
        .all(|assignment| assignment.len() == 16));
    let capped_error = crate::schedule_params::suffix_dp::packing_precommit_opening_products(
        &policy,
        dimensions,
        &repeated_key,
        &vec![honest_fold_policy_of::<Dense>(); 16],
        Some(16),
    )
    .expect_err("the cap must reject before materializing 17 assignments");
    assert!(matches!(capped_error, AkitaError::UnsupportedSchedule(_)));

    let incompatible_products =
        crate::schedule_params::suffix_dp::packing_precommit_opening_products(
            &policy,
            CommitmentRingDims {
                inner: 512,
                outer: 128,
                opening: 512,
            },
            &product_key,
            &[
                honest_fold_policy_of::<Dense>(),
                honest_fold_policy_of::<Dense>(),
            ],
            None,
        )
        .expect("incompatible shared opening dimension is an empty candidate domain");
    assert!(incompatible_products.is_empty());
}

#[cfg(feature = "catalog-gen")]
#[test]
fn guided_root_slice_survives_grouped_local_pruning() {
    use akita_config::{
        honest_fold_policy_of, policy_of, proof_optimized::fp64::Dense, CommitmentConfig,
    };
    use akita_types::AkitaScheduleLookupKey;

    let mut policy = policy_of::<Dense>();
    policy.selection_policy =
        crate::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3;
    let dimensions = CommitmentRingDims {
        inner: 256,
        outer: 128,
        opening: 64,
    };
    let opening =
        PlannerOpeningCandidate::coefficient_packing(0, policy.claim_ext_degree, dimensions, 64)
            .expect("valid final packing opening")
            .expect("final packing geometry");
    let scalar_key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(24, 2));
    let scalar = crate::planner::root_level_candidates_for_basis(
        &scalar_key,
        honest_fold_policy_of::<Dense>(),
        &[],
        &policy,
        dimensions,
        opening,
        &[],
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        None,
    )
    .expect("scalar root candidates");
    let frozen = synthetic_profile(
        scalar_key.final_group,
        &scalar.first().expect("scalar root candidate").0,
    );
    let grouped_key = AkitaScheduleLookupKey {
        final_group: scalar_key.final_group,
        precommitteds: vec![frozen],
    };
    let precommit_opening =
        PlannerOpeningCandidate::coefficient_packing(0, policy.claim_ext_degree, dimensions, 128)
            .expect("valid precommit packing opening")
            .expect("precommit packing geometry");
    let derive = |guide| {
        crate::planner::root_level_candidates_for_basis(
            &grouped_key,
            honest_fold_policy_of::<Dense>(),
            &[honest_fold_policy_of::<Dense>()],
            &policy,
            dimensions,
            opening,
            &[precommit_opening],
            Dense::inner_basis_range().0,
            Dense::opening_basis_range().0,
            guide,
        )
    };
    let unguided = derive(None).expect("unguided grouped root candidates");
    let unguided_layouts = unguided
        .iter()
        .map(|(candidate, _)| {
            (
                candidate.blocks().position_index_bits(),
                candidate.outer_slice_count(),
            )
        })
        .collect::<std::collections::HashSet<_>>();
    let reduced_vars =
        grouped_key.final_group.num_vars() - dimensions.d_a().trailing_zeros() as usize;

    for position_index_bits in 1..reduced_vars {
        for outer_slice_count in akita_types::CommitmentSliceCount::ALL {
            let guide = CandidateLayoutGuide {
                position_index_bits,
                outer_slice_count,
                inner_route: CandidateInnerRoute::Linf,
                setup_prefix: None,
            };
            let guided = derive(Some(guide)).expect("guided grouped root candidate");
            if !guided.is_empty()
                && unguided_layouts
                    .iter()
                    .any(|(position, _)| *position == position_index_bits)
                && !unguided_layouts.contains(&(position_index_bits, outer_slice_count))
            {
                assert!(guided.iter().all(|(candidate, _)| {
                    candidate.blocks().position_index_bits() == position_index_bits
                        && candidate.outer_slice_count() == outer_slice_count
                }));
                return;
            }
        }
    }
    panic!("fixture must contain a feasible root slice outside the grouped local frontier");
}

#[cfg(feature = "catalog-gen")]
#[test]
fn tensor_params_cannot_be_frozen_as_a_precommit_profile() {
    use akita_config::{
        honest_fold_policy_of, policy_of, proof_optimized::fp64::Dense, CommitmentConfig,
    };
    use akita_types::AkitaScheduleLookupKey;

    let mut policy = policy_of::<Dense>();
    let dimensions = CommitmentRingDims::uniform(256);
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
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        None,
    )
    .expect("standalone precommit candidates");
    let mut tensor_params = pre_candidates
        .first()
        .expect("precommit candidate")
        .0
        .clone();
    tensor_params.source_encoding =
        akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
            extension_degree: 2,
        };
    let error = GroupCommitPhaseParams::try_from_params(pre_group, &tensor_params)
        .expect_err("tensor params cannot be frozen into canonical commitment identity");
    assert!(matches!(error, AkitaError::InvalidSetup(_)));
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
            .unwrap()
            .unwrap();
    let reduced_packing =
        PlannerOpeningCandidate::coefficient_packing(1, policy.claim_ext_degree, dimensions, 64)
            .unwrap()
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
        guide: None,
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

    let natural_len = (1 << 14) - 513;
    let n_prefix = akita_types::padded_setup_prefix_len(natural_len);
    let full_prefix_groups = derive_setup_prefix_groups(
        &mut cache,
        SetupPrefixSearchRequest {
            n_prefix,
            ..request(trace)
        },
    )
    .unwrap();
    assert!(!full_prefix_groups.is_empty());
    for group in full_prefix_groups {
        assert_eq!(
            group.profile.blocks.live_ring_elements_per_claim
                * group.profile.inner.matrix.ring_dimension(),
            n_prefix
        );
        assert_eq!(
            group.profile.blocks.live_blocks * group.profile.blocks.positions_per_block,
            group.profile.blocks.live_ring_elements_per_claim
        );
        akita_types::scheduled_setup_prefix(natural_len, group)
            .validate()
            .expect("full setup prefix covers its complete power-of-two domain");
    }

    let error = derive_setup_prefix_groups(
        &mut cache,
        SetupPrefixSearchRequest {
            n_prefix: natural_len,
            ..request(trace)
        },
    )
    .expect_err("a natural length is not a setup-prefix commitment domain");
    assert!(error.to_string().contains("nonzero power of two"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn runtime_eor_pricing_uses_larger_incoming_prefix_arity() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot};

    let mut policy = policy_of::<OneHot>();
    policy.claim_ext_degree = 2;
    let mut params = audited_grouped_level_params();
    let mut cache = SetupPrefixSearchCache::default();
    let prefix_params = derive_setup_prefix_groups(
        &mut cache,
        SetupPrefixSearchRequest {
            policy: &policy,
            opening: PlannerOpeningCandidate::evaluation_trace(params.fold_challenge_config()),
            log_basis_open: params.open().digits.log_basis,
            n_prefix: 1 << 6,
            num_chunks: 1,
            inner_ring_dimension: params.d_a(),
            outer_ring_dimension: params.outer().matrix.ring_dimension(),
            guide: None,
        },
    )
    .expect("setup-prefix candidates")
    .into_iter()
    .next()
    .expect("synthetic prefix params");
    params
        .set_setup_prefix(Some(akita_types::scheduled_setup_prefix(
            1 << 6,
            prefix_params,
        )))
        .expect("valid setup-prefix topology");
    let output_witness_len = 1 << 6;
    let final_group = PolynomialGroupLayout::singleton(4);
    let opening_layout = params
        .opening_layout_for_final_group(final_group)
        .expect("prefix-consuming opening layout");
    assert_eq!(opening_layout.max_num_vars(), 6);
    let opening_shape = opening_layout
        .aggregate_polynomial_group_layout()
        .expect("aggregate EOR shape");
    let expected_eor = akita_types::extension_opening_reduction_level_bytes(
        policy.challenge_field_bits().unwrap(),
        policy.claim_ext_degree,
        opening_shape,
    )
    .expect("aggregate EOR bytes");
    let base = akita_types::level_proof_bytes(
        policy.decomposition.field_bits(),
        policy.challenge_field_bits().unwrap(),
        &params,
        params
            .relation_address_geometry(
                &opening_layout,
                policy.claim_ext_degree,
                params.d_a(),
                output_witness_len,
            )
            .unwrap(),
        None,
    )
    .expect("base level payload");
    let terminal = akita_types::TerminalFoldParams::from_expanded_group(params.clone());
    let runtime = akita_schedules::planner_support::nonterminal_level_payload_bytes(
        &policy,
        &params,
        &opening_layout,
        akita_types::FoldSuccessor::Terminal(&terminal),
        output_witness_len,
    )
    .expect("runtime level payload");
    assert_eq!(runtime.stage3, 0);
    assert_eq!(runtime.direct - base, expected_eor);
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
                guide: None,
            },
        )
        .expect("setup-prefix frontier");
        for params in groups {
            akita_types::setup_prefix_slot_field_elements(
                &akita_types::scheduled_setup_prefix(1usize << log_prefix, params)
                    .slot_id()
                    .expect("setup prefix group"),
            )
            .expect("frontier candidate must support its compression source");
        }
    }
}

#[path = "schedule_params_candidate/compression_source.rs"]
mod compression_source;

#[path = "schedule_params_candidate/shared_ab.rs"]
mod shared_ab;
