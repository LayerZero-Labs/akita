use super::*;

#[test]
fn multi_group_m_row_count_matches_canonical_layout() {
    let (lp, _) = sample_multi_group_root_params();
    let n_a_final = lp.a_key.row_len();
    let n_b_final = lp.b_key.row_len();
    let n_a_pre = lp.precommitted_groups[0].a_key.row_len();
    let n_b_pre = lp.precommitted_groups[0].b_key.row_len();
    let n_d = lp.d_key.row_len();

    assert_eq!(
        lp.relation_matrix_row_count_for(2, RelationMatrixRowLayout::WithDBlock)
            .unwrap(),
        1 + n_a_final + n_b_final + n_a_pre + n_b_pre + n_d
    );
    assert_eq!(
        lp.relation_matrix_row_count_for(2, RelationMatrixRowLayout::WithoutDBlock)
            .unwrap(),
        1 + n_a_final + n_b_final + n_a_pre + n_b_pre
    );
}

#[test]
fn multi_group_row_offsets_match_a_before_b_layout() {
    let (lp, batch) = sample_multi_group_root_params();
    let n_a_final = lp.a_key.row_len();
    let n_b_final = lp.b_key.row_len();
    let n_a_pre = lp.precommitted_groups[0].a_key.row_len();
    let n_b_pre = lp.precommitted_groups[0].b_key.row_len();
    let layout = RelationMatrixRowLayout::WithDBlock;
    let final_group = batch.root_final_group_index().expect("final group");

    assert_eq!(
        lp.a_row_range(&batch, final_group, layout).unwrap(),
        1..1 + n_a_final
    );
    assert_eq!(
        lp.commitment_row_range(&batch, final_group, layout)
            .unwrap(),
        1 + n_a_final..1 + n_a_final + n_b_final
    );
    assert_eq!(
        lp.a_row_range(&batch, 0, layout).unwrap(),
        1 + n_a_final + n_b_final..1 + n_a_final + n_b_final + n_a_pre
    );
    assert_eq!(
        lp.commitment_row_range(&batch, 0, layout).unwrap(),
        1 + n_a_final + n_b_final + n_a_pre..1 + n_a_final + n_b_final + n_a_pre + n_b_pre
    );
}

#[test]
fn multi_group_root_accepts_multi_chunk_witness_layout() {
    let (mut lp, batch) = sample_multi_group_root_params();
    lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
        num_chunks: 2,
        num_activated_levels: 1,
    };
    lp.evaluation_trace_row_index_for_layout(RelationMatrixRowLayout::WithDBlock, &batch)
        .expect("canonical product layout supports grouped shards");
}
