use super::*;
use crate::schedule::PrecommittedGroupParams;

#[test]
fn evaluation_trace_row_is_last_after_quotient_rows() {
    let lp = sample_params_only()
        .with_layout(&sample_layout_lp(), 128)
        .unwrap();
    let batch = OpeningClaimsLayout::new(4, 1).expect("batch");
    let quotient = lp
        .relation_matrix_row_count_for(1, RelationMatrixRowLayout::WithDBlock)
        .unwrap();

    assert_eq!(
        lp.evaluation_trace_row_for_layout(RelationMatrixRowLayout::WithDBlock, &batch)
            .expect("row"),
        quotient
    );
    assert_eq!(
        lp.relation_row_index_num_vars_for_layout(RelationMatrixRowLayout::WithDBlock, &batch)
            .unwrap(),
        (quotient + 1).next_power_of_two().trailing_zeros() as usize
    );
}

#[test]
fn multi_group_evaluation_trace_row_matches_legacy_quotient_count() {
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
    let batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 1]).expect("batch");
    let quotient = grouped
        .relation_matrix_row_count_for(2, RelationMatrixRowLayout::WithDBlock)
        .unwrap();

    assert_eq!(
        grouped
            .evaluation_trace_row_for_layout(RelationMatrixRowLayout::WithDBlock, &batch)
            .expect("row"),
        quotient
    );
}
