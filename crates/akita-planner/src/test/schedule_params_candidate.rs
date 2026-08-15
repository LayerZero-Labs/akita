use super::recursive::{recursive_candidate_order_key, recursive_split_search_domain};
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
