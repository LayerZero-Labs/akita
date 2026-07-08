use super::*;
use crate::schedule::PrecommittedGroupParams;

fn sample_grouped_root_params() -> (LevelParams, OpeningClaimsLayout) {
    let lp = sample_params_only()
        .with_layout(&sample_layout_lp(), 128)
        .unwrap();
    let precommit_lp = sample_params_only()
        .with_layout(&sample_layout_lp(), 128)
        .unwrap();
    let precommit = PrecommittedLevelParams {
        layout: PrecommittedGroupParams::from_params(
            PolynomialGroupLayout::new(4, 1),
            &precommit_lp,
        ),
        a_key: precommit_lp.a_key.clone(),
        b_key: AjtaiKeyParams::new_unchecked(
            precommit_lp.b_key.min_security_bits(),
            precommit_lp.b_key.sis_family(),
            5,
            precommit_lp.b_key.col_len(),
            precommit_lp.b_key.coeff_linf_bound(),
            precommit_lp.ring_dimension,
        ),
        num_blocks: precommit_lp.num_blocks,
        block_len: precommit_lp.block_len,
        num_digits_commit: precommit_lp.num_digits_commit,
        num_digits_open: precommit_lp.num_digits_open,
        num_digits_fold_one: precommit_lp.num_digits_fold_one,
    };
    let mut grouped = lp;
    grouped.precommitted_groups = vec![precommit];
    let batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 1]).expect("layout");
    (grouped, batch)
}

#[test]
fn grouped_m_row_count_matches_canonical_layout() {
    let (lp, _) = sample_grouped_root_params();
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
fn grouped_row_offsets_match_a_before_b_layout() {
    let (lp, batch) = sample_grouped_root_params();
    let n_a_final = lp.a_key.row_len();
    let n_b_final = lp.b_key.row_len();
    let n_a_pre = lp.precommitted_groups[0].a_key.row_len();
    let n_b_pre = lp.precommitted_groups[0].b_key.row_len();
    let layout = RelationMatrixRowLayout::WithDBlock;
    let final_group = batch.root_final_group_index().expect("final group");

    assert_eq!(
        lp.root_a_row_range(&batch, final_group, layout).unwrap(),
        1..1 + n_a_final
    );
    assert_eq!(
        lp.root_commitment_row_range(&batch, final_group, layout)
            .unwrap(),
        1 + n_a_final..1 + n_a_final + n_b_final
    );
    assert_eq!(
        lp.root_a_row_range(&batch, 0, layout).unwrap(),
        1 + n_a_final + n_b_final..1 + n_a_final + n_b_final + n_a_pre
    );
    assert_eq!(
        lp.root_commitment_row_range(&batch, 0, layout).unwrap(),
        1 + n_a_final + n_b_final + n_a_pre..1 + n_a_final + n_b_final + n_a_pre + n_b_pre
    );
}

#[test]
fn grouped_root_rejects_multi_chunk_witness_layout() {
    let (mut lp, _) = sample_grouped_root_params();
    lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
        num_chunks: 2,
        num_activated_levels: 1,
    };
    let err = lp
        .reject_grouped_multi_chunk("test")
        .expect_err("grouped multi-chunk must reject");
    assert!(
        format!("{err:?}").contains(crate::GROUPED_ROOT_MULTI_CHUNK_UNSUPPORTED),
        "unexpected error: {err:?}"
    );
}
