//! Schedule-level coverage for mixed commitment-matrix ring dimensions.

#![allow(missing_docs)]

use akita_config::{policy_of, proof_optimized::fp128, CommitmentConfig};
use akita_pcs::test_support::{
    matrix_ring_transition_schedule, mixed_matrix_root_schedule, MixedMatrixRootConfig,
    ThreeBandMatrixRingConfig,
};
use akita_types::sis::{decomposed_t_ring_count, decomposed_w_ring_count};
use akita_types::{
    setup_matrix_envelope_for_schedule, CommitmentRingDims, CommittedGroupParams, FoldSchedule,
    SisTableDigest,
};

const NUM_VARS: usize = 36;

fn assert_exact_matrix_widths(params: &CommittedGroupParams, num_polynomials: usize) {
    let dims = params.role_dims();
    let native_outer = decomposed_t_ring_count(
        params.inner_commit_matrix.output_rank(),
        params.num_digits_outer,
        params.num_live_blocks,
        num_polynomials,
    )
    .expect("native B width");
    let native_open = decomposed_w_ring_count(
        params.num_digits_open,
        params.num_live_blocks,
        num_polynomials,
    )
    .expect("native D width");
    assert_eq!(
        params.outer_commit_matrix.input_width(),
        native_outer * (dims.d_a() / dims.d_b())
    );
    assert_eq!(
        params.open_commit_matrix.input_width(),
        native_open * (dims.d_a() / dims.d_d())
    );
}

fn assert_suffix_matches_plan<Cfg: CommitmentConfig>(
    schedule: &FoldSchedule,
    first_actual_fold: usize,
    start_level: usize,
    start_witness_len: usize,
    start_log_basis: u32,
) {
    let suffix = akita_planner::plan_optimal_suffix(
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
        NUM_VARS,
        start_level,
        start_witness_len,
        start_log_basis,
    )
    .expect("independent suffix plan");
    let actual = &schedule.recursive_folds[first_actual_fold..];
    assert_eq!(actual.len(), suffix.folds.len());
    for (step, planned) in actual.iter().zip(&suffix.folds) {
        assert_eq!(step.params.witness, planned.params);
        assert_eq!(step.input_witness_len, planned.input_witness_len);
        assert_eq!(step.output_witness_len, planned.output_witness_len);
    }
    assert_eq!(schedule.terminal.params.witness, suffix.terminal.params);
    assert_eq!(
        schedule.terminal.input_witness_len,
        suffix.terminal.input_witness_len
    );
    assert_eq!(
        schedule.terminal.params.response_shape,
        suffix.terminal.response_shape
    );
}

fn three_band_matrix_schedule<Root>(expected_root_d: usize) -> FoldSchedule
where
    Root: akita_config::CommitmentConfig<Field = fp128::Field, ExtField = fp128::Field>,
{
    let schedule = matrix_ring_transition_schedule::<Root, fp128::D128OneHot, fp128::D64OneHot>(
        NUM_VARS,
        1,
        CommitmentRingDims {
            inner: expected_root_d,
            outer: 128,
            opening: 128,
        },
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
    )
    .expect("tableless three-band matrix-ring schedule");

    assert_eq!(
        schedule.root.params.final_group.commitment.role_dims(),
        CommitmentRingDims {
            inner: expected_root_d,
            outer: 128,
            opening: 128,
        }
    );
    assert_exact_matrix_widths(&schedule.root.params.final_group.commitment, 1);
    assert_eq!(
        schedule.recursive_folds[0].params.witness.role_dims(),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        }
    );
    assert_exact_matrix_widths(&schedule.recursive_folds[0].params.witness, 1);
    assert_eq!(
        schedule.recursive_folds[1].params.witness.role_dims(),
        CommitmentRingDims::uniform(64)
    );
    let l1 = &schedule.recursive_folds[0];
    assert_suffix_matches_plan::<fp128::D64OneHot>(
        &schedule,
        1,
        2,
        l1.output_witness_len,
        l1.params.witness.log_basis_open,
    );
    schedule
}

#[test]
fn tableless_d256_root_uses_offline_planner() {
    let _schedule = three_band_matrix_schedule::<fp128::D256OneHot>(256);
}

#[test]
fn d512_root_uses_additive_a_role_sis_table() {
    let schedule = three_band_matrix_schedule::<fp128::D512OneHot>(512);

    assert_eq!(
        schedule
            .root
            .params
            .final_group
            .commitment
            .inner_commit_matrix
            .sis_table_key()
            .table_digest,
        SisTableDigest::Q128_INNER_D512
    );
    let root = &schedule.root.params.final_group.commitment;
    assert_eq!(
        root.flat_field_len().expect("root flat length"),
        1usize << NUM_VARS
    );
    let required = setup_matrix_envelope_for_schedule(&schedule).expect("schedule envelope");
    let configured = <ThreeBandMatrixRingConfig<
        fp128::D512OneHot,
        fp128::D128OneHot,
        fp128::D64OneHot,
        128,
        64,
    > as CommitmentConfig>::max_setup_matrix_size(NUM_VARS, 1)
    .expect("configured envelope");
    assert_eq!(configured, required);
}

#[test]
fn mixed_matrix_root_replans_its_complete_suffix() {
    let schedule = mixed_matrix_root_schedule::<fp128::D128OneHot>(NUM_VARS, 1, 32, 64)
        .expect("mixed-matrix root schedule");
    let root = &schedule.root.params.final_group.commitment;
    assert_eq!(
        root.role_dims(),
        CommitmentRingDims {
            inner: 128,
            outer: 32,
            opening: 64,
        }
    );
    assert_exact_matrix_widths(root, 1);
    assert_suffix_matches_plan::<fp128::D128OneHot>(
        &schedule,
        0,
        1,
        schedule.root.output_witness_len,
        root.log_basis_open,
    );

    let required = setup_matrix_envelope_for_schedule(&schedule).expect("schedule envelope");
    let configured =
        <MixedMatrixRootConfig<fp128::D128OneHot, 32, 64> as CommitmentConfig>::
            max_setup_matrix_size(NUM_VARS, 1)
            .expect("configured envelope");
    assert_eq!(configured, required);
}
