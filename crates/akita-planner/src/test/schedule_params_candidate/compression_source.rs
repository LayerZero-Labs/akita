use super::*;

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
    let dimensions = CommitmentRingDims::uniform(256);
    let challenge = SparseChallengeConfig::production_for_ring_dim(dimensions.d_a())
        .expect("production challenge for candidate A dimension");
    let num_claims = 1;
    let width_s = 8;
    let mut raw_candidate = derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
        policy: &policy,
        fold_policy: &FixedFoldPolicy,
        ring_challenge_cfg: &challenge,
        challenge_dimension: dimensions.d_a(),
        dimensions,
        payload_mode: akita_types::CommitmentPayloadMode::Raw,
        num_claims,

        num_live_ring_elements_per_claim: 64,
        num_positions_per_block: 8,
        num_live_blocks: 8,

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
    params.own_group_mut().profile.inner.matrix = raw_candidate.inner_commit_matrix;
    params.own_group_mut().profile.outer.matrix = raw_candidate.outer_commit_matrix;
    params.own_group_mut().profile.group = PolynomialGroupLayout::singleton(14);
    params.own_group_mut().opening.num_digits_fold = raw_candidate.num_digits_fold;
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
