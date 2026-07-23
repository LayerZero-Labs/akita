use super::*;
use crate::proof::relation::{relation_rhs_layout_for, relation_rhs_row_count};

#[test]
fn eight_quotient_rows_adds_one_tau1_var_for_evaluation_trace() {
    let mut lp = laid_out_sample_lp();
    lp.inner_commit_matrix = InnerCommitMatrixParams::new_unchecked(
        lp.inner_commit_matrix.security_policy(),
        lp.inner_commit_matrix.sis_table_key().table_digest,
        lp.inner_commit_matrix.sis_modulus_profile(),
        2,
        lp.inner_commit_matrix.input_width(),
        lp.inner_commit_matrix.coeff_linf_bound(),
        lp.d_a(),
    );
    lp.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        lp.outer_commit_matrix.security_policy(),
        lp.outer_commit_matrix.sis_table_key().table_digest,
        lp.outer_commit_matrix.sis_modulus_profile(),
        3,
        lp.outer_commit_matrix.input_width(),
        lp.outer_commit_matrix.coeff_linf_bound(),
        lp.d_a(),
    );
    lp.open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
        lp.open_commit_matrix.security_policy(),
        lp.open_commit_matrix.sis_table_key().table_digest,
        lp.open_commit_matrix.sis_modulus_profile(),
        2,
        lp.open_commit_matrix.input_width(),
        lp.open_commit_matrix.coeff_linf_bound(),
        lp.d_a(),
    );
    let batch = OpeningClaimsLayout::new(4, 1).expect("batch");
    let quotient = lp.relation_matrix_row_count(1).unwrap();
    assert_eq!(quotient, 8);

    let quotient_only_vars = quotient.next_power_of_two().trailing_zeros() as usize;
    assert_eq!(quotient_only_vars, 3);
    assert_eq!(
        lp.evaluation_trace_row_index(&batch).expect("row"),
        quotient
    );
    assert_eq!(lp.relation_row_index_num_vars(&batch).unwrap(), 4);
}

#[test]
fn evaluation_trace_row_is_last_after_quotient_rows() {
    let lp = laid_out_sample_lp();
    let batch = OpeningClaimsLayout::new(4, 1).expect("batch");
    let quotient = lp.relation_matrix_row_count(1).unwrap();

    assert_eq!(
        lp.evaluation_trace_row_index(&batch).expect("row"),
        quotient
    );
    assert_eq!(
        lp.relation_row_index_num_vars(&batch).unwrap(),
        (quotient + 1).next_power_of_two().trailing_zeros() as usize
    );
}

#[test]
fn multi_group_evaluation_trace_row_matches_quotient_count() {
    let (grouped, batch) = sample_multi_group_root_params();
    let quotient = grouped.relation_matrix_row_count(2).unwrap();

    assert_eq!(
        grouped.evaluation_trace_row_index(&batch).expect("row"),
        quotient
    );
    assert_eq!(
        grouped.relation_row_index_num_vars(&batch).unwrap(),
        (quotient + 1).next_power_of_two().trailing_zeros() as usize
    );
}

#[test]
fn relation_rhs_row_count_matches_level_params() {
    let lp = laid_out_sample_lp();
    let batch = OpeningClaimsLayout::new(4, 1).expect("batch");
    let rhs_layout = relation_rhs_layout_for(&lp, &batch).expect("rhs layout");
    assert_eq!(
        relation_rhs_row_count(&rhs_layout),
        lp.relation_matrix_row_count(batch.num_groups())
            .expect("row count"),
    );

    let (grouped_lp, grouped_batch) = sample_multi_group_root_params();
    let rhs_layout = relation_rhs_layout_for(&grouped_lp, &grouped_batch).expect("rhs layout");
    assert_eq!(
        relation_rhs_row_count(&rhs_layout),
        grouped_lp
            .relation_matrix_row_count(grouped_batch.num_groups())
            .expect("row count"),
    );
}
