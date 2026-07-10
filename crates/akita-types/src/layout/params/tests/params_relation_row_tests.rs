use super::*;
use crate::proof::relation::{relation_rhs_layout_for, relation_rhs_row_count};

#[test]
fn eight_quotient_rows_adds_one_tau1_var_for_evaluation_trace() {
    let mut lp = laid_out_sample_lp();
    lp.a_key = AjtaiKeyParams::new_unchecked(
        lp.a_key.min_security_bits(),
        lp.a_key.sis_family(),
        2,
        lp.a_key.col_len(),
        lp.a_key.coeff_linf_bound(),
        lp.ring_dimension,
    );
    lp.b_key = AjtaiKeyParams::new_unchecked(
        lp.b_key.min_security_bits(),
        lp.b_key.sis_family(),
        3,
        lp.b_key.col_len(),
        lp.b_key.coeff_linf_bound(),
        lp.ring_dimension,
    );
    lp.d_key = AjtaiKeyParams::new_unchecked(
        lp.d_key.min_security_bits(),
        lp.d_key.sis_family(),
        2,
        lp.d_key.col_len(),
        lp.d_key.coeff_linf_bound(),
        lp.ring_dimension,
    );
    let batch = OpeningClaimsLayout::new(4, 1).expect("batch");
    let quotient = lp
        .relation_matrix_row_count_for(1, RelationMatrixRowLayout::WithDBlock)
        .unwrap();
    assert_eq!(quotient, 8);

    let quotient_only_vars = quotient.next_power_of_two().trailing_zeros() as usize;
    assert_eq!(quotient_only_vars, 3);
    assert_eq!(
        lp.evaluation_trace_row_index_for_layout(RelationMatrixRowLayout::WithDBlock, &batch)
            .expect("row"),
        quotient
    );
    assert_eq!(
        lp.relation_row_index_num_vars_for_layout(RelationMatrixRowLayout::WithDBlock, &batch)
            .unwrap(),
        4
    );
}

#[test]
fn evaluation_trace_row_is_last_after_quotient_rows() {
    let lp = laid_out_sample_lp();
    let batch = OpeningClaimsLayout::new(4, 1).expect("batch");
    let quotient = lp
        .relation_matrix_row_count_for(1, RelationMatrixRowLayout::WithDBlock)
        .unwrap();

    assert_eq!(
        lp.evaluation_trace_row_index_for_layout(RelationMatrixRowLayout::WithDBlock, &batch)
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
fn multi_group_evaluation_trace_row_matches_quotient_count() {
    let (grouped, batch) = sample_multi_group_root_params();
    let quotient = grouped
        .relation_matrix_row_count_for(2, RelationMatrixRowLayout::WithDBlock)
        .unwrap();

    assert_eq!(
        grouped
            .evaluation_trace_row_index_for_layout(RelationMatrixRowLayout::WithDBlock, &batch)
            .expect("row"),
        quotient
    );
    assert_eq!(
        grouped
            .relation_row_index_num_vars_for_layout(RelationMatrixRowLayout::WithDBlock, &batch)
            .unwrap(),
        (quotient + 1).next_power_of_two().trailing_zeros() as usize
    );
}

#[test]
fn relation_rhs_row_count_matches_level_params() {
    let lp = laid_out_sample_lp();
    let batch = OpeningClaimsLayout::new(4, 1).expect("batch");
    for layout in [
        RelationMatrixRowLayout::WithDBlock,
        RelationMatrixRowLayout::WithoutDBlock,
    ] {
        let rhs_layout = relation_rhs_layout_for(&lp, &batch, layout).expect("rhs layout");
        assert_eq!(
            relation_rhs_row_count(&rhs_layout),
            lp.relation_matrix_row_count_for(batch.num_groups(), layout)
                .expect("row count"),
            "scalar layout {:?}",
            layout
        );
    }

    let (grouped_lp, grouped_batch) = sample_multi_group_root_params();
    for layout in [
        RelationMatrixRowLayout::WithDBlock,
        RelationMatrixRowLayout::WithoutDBlock,
    ] {
        let rhs_layout =
            relation_rhs_layout_for(&grouped_lp, &grouped_batch, layout).expect("rhs layout");
        assert_eq!(
            relation_rhs_row_count(&rhs_layout),
            grouped_lp
                .relation_matrix_row_count_for(grouped_batch.num_groups(), layout)
                .expect("row count"),
            "grouped layout {:?}",
            layout
        );
    }
}
