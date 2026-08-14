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
                ring_challenge_cfg: &challenge,
                log_basis_open: 3,
                n_prefix: 1usize << log_prefix,
                num_chunks: 1,
                inner_ring_dimension: 64,
                outer_ring_dimension: 64,
            },
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

#[cfg(feature = "catalog-gen")]
#[test]
fn shared_ab_derivation_forwards_the_exact_slice_count() {
    use std::cell::Cell;

    use akita_config::{policy_of, proof_optimized::fp128::OneHot};

    struct RecordingPolicy<'a>(&'a Cell<Option<(usize, usize)>>);

    impl HonestFoldPolicy for RecordingPolicy<'_> {
        fn num_digits_fold(&self, query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError> {
            self.0
                .set(Some((query.outer_slice_count.get(), query.num_fold_coeffs)));
            Err(AkitaError::InvalidSetup("stop after query capture".into()))
        }
    }

    let policy = policy_of::<OneHot>();
    let challenge = SparseChallengeConfig::pm1_only(3);
    for outer_slice_count in akita_types::CommitmentSliceCount::ALL {
        let seen = Cell::new(None);
        let fold_policy = RecordingPolicy(&seen);
        assert!(
            derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
                policy: &policy,
                fold_policy: &fold_policy,
                ring_challenge_cfg: &challenge,
                dimensions: CommitmentRingDims::uniform(64),
                payload_mode: akita_types::CommitmentPayloadMode::Compressed,
                num_claims: 1,
                num_live_blocks: 8,
                num_chunks: 1,
                outer_slice_count,
                witness_norms: FoldWitnessNorms::bounded(3, 64),
                log_basis_open: 3,
                width_s: 8,
                num_digits_outer: 2,
            })
            .unwrap()
            .is_none()
        );
        assert_eq!(seen.get(), Some((outer_slice_count.get(), 512)));
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
            num_live_blocks: 8,
            num_chunks: 1,
            outer_slice_count,
            witness_norms: FoldWitnessNorms::bounded(3, dimensions.d_a()),
            log_basis_open: 3,
            width_s,
            num_digits_outer: 2,
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
            num_live_blocks: 8,
            num_chunks: 1,
            outer_slice_count: akita_types::CommitmentSliceCount::ONE,
            witness_norms: FoldWitnessNorms::bounded(3, 64),
            log_basis_open: 3,
            width_s: usize::MAX,
            num_digits_outer: 2,
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
            num_live_blocks: 8,
            num_chunks: 1,
            outer_slice_count: akita_types::CommitmentSliceCount::ONE,
            witness_norms: FoldWitnessNorms::bounded(3, 64),
            log_basis_open: 3,
            width_s: 8,
            num_digits_outer: 2,
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
        num_live_blocks: 8,
        num_chunks: 1,
        outer_slice_count: akita_types::CommitmentSliceCount::ONE,
        witness_norms: FoldWitnessNorms::bounded(3, dimensions.d_a()),
        log_basis_open: 3,
        width_s,
        num_digits_outer: 2,
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
