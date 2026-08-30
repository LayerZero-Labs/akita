#[cfg(feature = "catalog-gen")]
use super::*;

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
    let candidate = |dimensions: CommitmentRingDims, outer_slice_count, width_s| {
        let challenge = SparseChallengeConfig::production_for_ring_dim(dimensions.d_a())
            .expect("production challenge for candidate A dimension");
        derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
            policy: &policy,
            fold_policy: &FixedFoldPolicy,
            ring_challenge_cfg: &challenge,
            challenge_dimension: dimensions.d_a(),
            dimensions,
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            num_claims: 1,
            num_live_ring_elements_per_claim: 64,
            num_positions_per_block: 8,
            num_live_blocks: 8,
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

    let d64_challenge =
        SparseChallengeConfig::production_for_ring_dim(64).expect("production D64 challenge");
    assert!(
        derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
            policy: &policy,
            fold_policy: &FixedFoldPolicy,
            ring_challenge_cfg: &d64_challenge,
            challenge_dimension: 64,
            dimensions: CommitmentRingDims::uniform(64),
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            num_claims: 1,
            num_live_ring_elements_per_claim: 64,
            num_positions_per_block: 8,
            num_live_blocks: 8,
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
            ring_challenge_cfg: &d64_challenge,
            challenge_dimension: 64,
            dimensions: CommitmentRingDims::uniform(64),
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            num_claims: 1,
            num_live_ring_elements_per_claim: 64,
            num_positions_per_block: 8,
            num_live_blocks: 8,
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
