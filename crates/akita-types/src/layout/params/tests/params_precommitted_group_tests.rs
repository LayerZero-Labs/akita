use super::*;
use crate::WitnessLayout;

#[test]
fn multi_group_m_row_count_matches_canonical_layout() {
    let (lp, _) = sample_multi_group_root_params();
    let n_a_final = lp.inner_commit_matrix.output_rank();
    let n_b_final = lp.outer_commit_matrix.output_rank();
    let n_a_pre = lp.precommitted_groups[0].inner_commit_matrix.output_rank();
    let n_b_pre = lp.precommitted_groups[0].outer_commit_matrix.output_rank();
    let n_d = lp.open_commit_matrix.output_rank();

    assert_eq!(
        lp.relation_matrix_row_count(2).unwrap(),
        1 + n_a_final + n_b_final + 1 + n_a_pre + n_b_pre + n_d
    );
}

#[test]
fn multi_group_row_offsets_match_a_before_b_layout() {
    let (lp, batch) = sample_multi_group_root_params();
    let n_a_final = lp.inner_commit_matrix.output_rank();
    let n_b_final = lp.outer_commit_matrix.output_rank();
    let n_a_pre = lp.precommitted_groups[0].inner_commit_matrix.output_rank();
    let n_b_pre = lp.precommitted_groups[0].outer_commit_matrix.output_rank();
    let final_group = batch.root_final_group_index().expect("final group");

    assert_eq!(
        lp.a_row_range(&batch, final_group).unwrap(),
        1..1 + n_a_final
    );
    assert_eq!(
        lp.commitment_row_range(&batch, final_group).unwrap(),
        1 + n_a_final..1 + n_a_final + n_b_final
    );
    assert_eq!(
        lp.a_row_range(&batch, 0).unwrap(),
        2 + n_a_final + n_b_final..2 + n_a_final + n_b_final + n_a_pre
    );
    assert_eq!(
        lp.commitment_row_range(&batch, 0).unwrap(),
        2 + n_a_final + n_b_final + n_a_pre..2 + n_a_final + n_b_final + n_a_pre + n_b_pre
    );
    assert_eq!(lp.consistency_row_index(&batch, final_group).unwrap(), 0);
    assert_eq!(
        lp.consistency_row_index(&batch, 0).unwrap(),
        1 + n_a_final + n_b_final
    );
}

#[test]
fn multi_group_root_accepts_multi_chunk_witness_layout() {
    let (mut lp, batch) = sample_multi_group_root_params();
    lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
        num_chunks: 2,
        num_activated_levels: 1,
    };
    lp.evaluation_trace_row_index(&batch)
        .expect("canonical product layout supports grouped chunks");
}

#[test]
fn group_role_dims_use_group_a_b_and_level_shared_d() {
    let (mut lp, batch) = sample_multi_group_root_params();
    let precommitted = &mut lp.precommitted_groups[0];
    let outer = &precommitted.outer_commit_matrix;
    precommitted.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * 2,
        outer.coeff_linf_bound(),
        32,
    );
    precommitted.layout.outer_ring_dimension = 32;
    let dims = lp
        .group_role_dims(&batch, 0)
        .expect("precommitted group role dimensions");
    assert_eq!(
        dims,
        CommitmentRingDims {
            inner: 64,
            outer: 32,
            opening: 64,
        }
    );
    let final_group = batch.root_final_group_index().expect("final group");
    assert_eq!(
        lp.group_role_dims(&batch, final_group)
            .expect("final group role dimensions"),
        lp.role_dims()
    );
}

#[test]
fn precommitted_params_reject_frozen_matrix_dimension_mismatch() {
    let (mut lp, _) = sample_multi_group_root_params();
    let precommitted = &mut lp.precommitted_groups[0];
    precommitted.layout.outer_ring_dimension /= 2;
    let err = precommitted
        .validate()
        .expect_err("frozen B dimension must match the serialized B matrix");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn relation_witness_carrier_is_independent_of_final_group_order() {
    use akita_field::Prime128OffsetA7F7;

    let (mut lp, batch) = sample_multi_group_root_params();
    let precommitted = &mut lp.precommitted_groups[0];
    let inner = &precommitted.inner_commit_matrix;
    precommitted.inner_commit_matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner.coeff_linf_bound(),
        128,
    );
    precommitted.layout.inner_ring_dimension = 128;
    let outer = &precommitted.outer_commit_matrix;
    precommitted.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * 2,
        outer.coeff_linf_bound(),
        outer.ring_dimension(),
    );
    precommitted.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(128).expect("D128 challenge");

    assert_eq!(lp.d_a(), 64, "the final group remains native at A=64");
    assert_eq!(
        lp.group_role_dims(&batch, 0)
            .expect("precommitted group dimensions")
            .d_a(),
        128
    );
    let witness_layout = WitnessLayout::new(
        &lp,
        &batch,
        lp.witness_chunk.num_chunks,
        crate::r_decomp_levels::<Prime128OffsetA7F7>(lp.log_basis_open),
    )
    .expect("witness layout");
    assert_eq!(
        lp.output_witness_len::<Prime128OffsetA7F7>(&batch)
            .expect("output witness length"),
        witness_layout.live_coeff_len()
    );
    assert!(witness_layout
        .units_for_group(0)
        .expect("precommitted units")
        .iter()
        .all(|unit| unit.z_range().len().is_multiple_of(128)));
}
