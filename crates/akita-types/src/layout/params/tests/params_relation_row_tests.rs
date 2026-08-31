use super::*;
use crate::proof::relation::{
    assemble_compressed_relation_rhs, relation_rhs_coeff_len, relation_rhs_row_count,
    RelationRowFamily, RelationWitnessGeometry,
};
use crate::WitnessLayout;
use jolt_field::{One, Prime128OffsetA7F7, Zero};

#[test]
fn compression_quotient_rows_are_included_before_evaluation_trace() {
    let mut lp = laid_out_sample_lp();
    lp.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        lp.inner().matrix.security_policy(),
        lp.inner()
            .matrix
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        lp.inner().matrix.sis_modulus_profile(),
        2,
        lp.inner().matrix.input_width(),
        lp.inner()
            .matrix
            .coeff_linf_bound()
            .expect("L infinity test matrix"),
        lp.d_a(),
    );
    lp.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        lp.outer().matrix.security_policy(),
        lp.outer().matrix.sis_table_key().table_digest,
        lp.outer().matrix.sis_modulus_profile(),
        3,
        lp.outer().matrix.input_width(),
        lp.outer().matrix.coeff_linf_bound(),
        lp.d_a(),
    );
    lp.open_matrix = OpenCommitMatrixParams::new_unchecked(
        lp.open().matrix.security_policy(),
        lp.open().matrix.sis_table_key().table_digest,
        lp.open().matrix.sis_modulus_profile(),
        2,
        lp.open().matrix.input_width(),
        lp.open().matrix.coeff_linf_bound(),
        lp.d_a(),
    );
    let batch = OpeningClaimsLayout::new(4, 1).expect("batch");
    let quotient = lp.relation_matrix_row_count(1).unwrap();
    assert_eq!(quotient, 12);

    let quotient_only_vars = quotient.next_power_of_two().trailing_zeros() as usize;
    assert_eq!(quotient_only_vars, 4);
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
    let joint_geometry =
        RelationWitnessGeometry::for_evaluation_trace_execution(&lp, &batch).expect("geometry");
    let rhs_layout = joint_geometry.rhs_layout();
    assert_eq!(
        relation_rhs_row_count(rhs_layout),
        lp.relation_matrix_row_count(batch.num_groups())
            .expect("row count"),
    );
    let compression_rows = rhs_layout
        .row_families()
        .expect("row families")
        .into_iter()
        .filter(|row| {
            matches!(
                row,
                RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(compression_rows.len(), 4);
    assert!(matches!(
        compression_rows.as_slice(),
        [
            RelationRowFamily::CompressionF { map_index: 0, .. },
            RelationRowFamily::CompressionH { map_index: 0, .. },
            RelationRowFamily::CompressionF { map_index: 1, .. },
            RelationRowFamily::CompressionH { map_index: 1, .. }
        ]
    ));
    let witness_layout = WitnessLayout::new(
        &lp,
        &batch,
        &joint_geometry,
        1,
        crate::RelationQuotientPlan::quotient_lift(2).unwrap(),
    )
    .expect("witness layout");
    let relation_geometry = lp
        .relation_address_geometry(&batch, 1, lp.d_a(), witness_layout.live_coeff_len())
        .expect("A/B/D geometry");
    let compression_geometry = lp
        .compression_relation_address_geometry(&batch, 1, lp.d_a(), witness_layout.live_coeff_len())
        .expect("F/H geometry");
    assert_eq!(
        relation_geometry.relation_coefficient_block_len(),
        lp.role_dims().common_relation_coeff_count()
    );
    assert!(
        compression_geometry.coefficient_block_len()
            < relation_geometry.relation_coefficient_block_len()
    );

    let group_terminal = vec![
        Prime128OffsetA7F7::one();
        rhs_layout
            .group_compression_plan(0)
            .expect("F plan")
            .1
            .terminal_coefficients()
    ];
    let opening_terminal = vec![
        Prime128OffsetA7F7::one();
        rhs_layout
            .opening_compression_plan()
            .expect("H plan")
            .terminal_coefficients()
    ];
    let rhs = assemble_compressed_relation_rhs(
        rhs_layout,
        &[group_terminal.as_slice()],
        &opening_terminal,
    )
    .expect("compressed rhs");
    assert_eq!(
        rhs.coeff_len(),
        relation_rhs_coeff_len(rhs_layout).expect("rhs coefficient length")
    );
    assert_eq!(
        rhs.coeffs()
            .iter()
            .filter(|coefficient| !coefficient.is_zero())
            .count(),
        group_terminal.len() + opening_terminal.len()
    );

    let (grouped_lp, grouped_batch) = sample_multi_group_root_params();
    let grouped_geometry =
        RelationWitnessGeometry::for_evaluation_trace_execution(&grouped_lp, &grouped_batch)
            .expect("geometry");
    let rhs_layout = grouped_geometry.rhs_layout();
    assert_eq!(
        relation_rhs_row_count(rhs_layout),
        grouped_lp
            .relation_matrix_row_count(grouped_batch.num_groups())
            .expect("row count"),
    );
    assert_eq!(
        rhs_layout
            .row_families()
            .expect("grouped row families")
            .into_iter()
            .filter(|row| matches!(row, RelationRowFamily::CompressionF { .. }))
            .count(),
        2 * grouped_batch.num_groups()
    );
}
