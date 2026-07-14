use super::plan::SetupContributionGroupPlan;
use super::weights::setup_z_col_weights;
use super::*;
use crate::{
    gadget_row_scalars, AkitaExpandedSetup, AkitaSetupSeed, CommitmentRingDims, FlatMatrix,
    LevelParams, MachineChunkId, OpeningBatchWitnessGroup, OpeningBatchWitnessLayout,
    OpeningBlockLayout, RelationMatrixRowLayout, SemanticGroupId, SetupContributionStatic,
    SetupIndexWeightEvaluator, WitnessOwnershipUnit,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_field::Prime128OffsetA7F7;

type F = Prime128OffsetA7F7;
const TEST_D: usize = 64;

type SingleGroupPlanParts = (
    SetupContributionPlan<F>,
    SetupContributionStatic<F>,
    Vec<SetupContributionGroupInputs>,
);

type StructuredWeightFixture = (
    SetupContributionPlanInputs<F>,
    Vec<SetupContributionGroupInputs>,
    SetupContributionStatic<F>,
    SetupContributionPlan<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
);

fn test_scalar(value: u128) -> F {
    F::from_canonical_u128(value)
}

fn finalize_test_plan(
    d_rows: usize,
    d_physical_cols: usize,
    groups: Vec<SetupContributionGroupPlan<F>>,
    role_dims: CommitmentRingDims,
) -> SetupContributionPlan<F> {
    let a_footprint = groups
        .iter()
        .map(|group| group.n_a * group.z_cols)
        .max()
        .unwrap();
    let b_footprint = groups
        .iter()
        .map(|group| group.n_b * group.t_cols)
        .max()
        .unwrap();
    let d_footprint = d_rows * d_physical_cols;
    let projection_geometry = SetupProjectionGeometry::from_role_footprints(
        role_dims,
        a_footprint,
        b_footprint,
        d_footprint,
    )
    .unwrap();
    let mut plan = SetupContributionPlan {
        groups,
        d_rows,
        d_physical_cols,
        projection_geometry,
    };
    for group in &mut plan.groups {
        group
            .refresh_segments(plan.d_rows, plan.d_physical_cols)
            .expect("valid cached setup scan segments");
    }
    plan
}

fn prepare_single_group_plan(
    inputs: &SetupContributionPlanInputs<F>,
    full_vec_randomness: &[F],
    eq_low: Option<&[F]>,
    z_block_low_eq: Option<&[F]>,
    fold_gadget: &[F],
    layout: &OpeningBatchWitnessLayout,
) -> Result<SetupContributionPlan<F>, AkitaError> {
    prepare_single_group_plan_parts(
        inputs,
        full_vec_randomness,
        eq_low,
        z_block_low_eq,
        fold_gadget,
        layout,
    )
    .map(|(plan, _, _)| plan)
}

fn prepare_single_group_plan_parts(
    inputs: &SetupContributionPlanInputs<F>,
    full_vec_randomness: &[F],
    eq_low: Option<&[F]>,
    z_block_low_eq: Option<&[F]>,
    fold_gadget: &[F],
    layout: &OpeningBatchWitnessLayout,
) -> Result<SingleGroupPlanParts, AkitaError> {
    let opening_layout = OpeningBlockLayout::new(1, layout.total_len())?;
    let single_group =
        SetupContributionGroupInputs::single_group_layout(inputs, layout, opening_layout, 0)?;
    let groups = vec![single_group.group];
    let static_plan = SetupContributionPlan::prepare_static(
        inputs,
        &groups,
        single_group.d_row_start,
        single_group.d_rows,
        single_group.d_physical_cols,
    )?;
    let plan = SetupContributionPlan::finish_plan::<F>(
        &static_plan,
        full_vec_randomness,
        eq_low,
        z_block_low_eq,
        Some(fold_gadget),
        &groups,
        CommitmentRingDims::uniform(TEST_D),
    )?;
    Ok((plan, static_plan, groups))
}

fn structured_weight_fixture(
    live_fold_count: usize,
    ownership_widths: &[usize],
    role_dims: CommitmentRingDims,
) -> StructuredWeightFixture {
    let num_claims = 2;
    let depth_open = 2;
    let depth_commit = 2;
    let depth_fold = 2;
    let fold_position_count = 8;
    let n_a = 2;
    let n_b = 2;
    let n_d = 2;
    let log_basis = 4;
    assert_eq!(ownership_widths.iter().sum::<usize>(), live_fold_count);
    let group_descriptor = OpeningBatchWitnessGroup {
        id: SemanticGroupId(0),
        num_claims,
        live_fold_count,
        fold_position_count,
        depth_open,
        depth_commit,
        depth_fold,
        n_a,
        e_setup_col_offset: 0,
    };
    let z_len = fold_position_count * depth_commit * depth_fold;
    let mut cursor = 0usize;
    let mut global_block_base = 0usize;
    let ownership_units = ownership_widths
        .iter()
        .copied()
        .enumerate()
        .map(|(chunk, blocks)| {
            let z_range = cursor..cursor + z_len;
            let e_len = num_claims * depth_open * blocks;
            let e_range = z_range.end..z_range.end + e_len;
            let t_len = n_a * num_claims * depth_open * blocks;
            let t_range = e_range.end..e_range.end + t_len;
            cursor = t_range.end;
            let unit = WitnessOwnershipUnit {
                group: SemanticGroupId(0),
                machine_chunk: MachineChunkId(chunk),
                global_block_base,
                blocks,
                z_range,
                e_range,
                t_range,
            };
            global_block_base += blocks;
            unit
        })
        .collect::<Vec<_>>();
    let layout = OpeningBatchWitnessLayout {
        groups: vec![group_descriptor],
        machine_chunks: (0..ownership_widths.len()).map(MachineChunkId).collect(),
        transcript_group_order: vec![SemanticGroupId(0)],
        relation_group_order: vec![SemanticGroupId(0)],
        ownership_units,
        r_range: cursor..cursor + n_d * depth_fold,
        relation_rows: n_d,
        quotient_depth: depth_fold,
    };
    let rows = 1 + n_a + n_b + n_d;
    let tau1 = (0..3)
        .map(|idx| test_scalar(31 + idx as u128))
        .collect::<Vec<_>>();
    let inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
        rows,
        n_a,
        n_b,
        n_d,
        num_groups: 1,
        num_polys_per_group: vec![num_claims],
        num_t_vectors: num_claims,
        num_claims,
        live_fold_count,
        fold_position_count,
        depth_open,
        depth_commit,
        depth_fold,
        inner_width: fold_position_count * depth_commit,
        eq_tau1: EqPolynomial::evals(&tau1).unwrap().into(),
    };
    let full_vec_randomness = (0..18)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let opening_layout = OpeningBlockLayout::new(8, layout.total_len().div_ceil(8)).unwrap();
    let groups = vec![SetupContributionGroupInputs {
        group_id: SemanticGroupId(0),
        e_col_offset: 0,
        num_claims,
        live_fold_count,
        fold_position_count,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        n_a,
        n_b,
        t_cols_per_vector: n_a * depth_open * live_fold_count,
        a_row_start: 1,
        b_row_start: 1 + n_a,
        layout: std::sync::Arc::new(layout),
        opening_layout,
    }];
    let static_plan = SetupContributionPlan::prepare_static(
        &inputs,
        &groups,
        rows - n_d,
        n_d,
        num_claims * live_fold_count * depth_open,
    )
    .unwrap();
    let plan = SetupContributionPlan::finish_plan::<F>(
        &static_plan,
        &full_vec_randomness,
        None,
        None,
        Some(&fold_gadget),
        &groups,
        role_dims,
    )
    .unwrap();
    (
        inputs,
        groups,
        static_plan,
        plan,
        tau1,
        full_vec_randomness,
        fold_gadget,
    )
}

fn expected_z_setup_weights(
    layout: &OpeningBatchWitnessLayout,
    opening_layout: OpeningBlockLayout,
    group_id: SemanticGroupId,
    fold_gadget: &[F],
    full_vec_randomness: &[F],
) -> Vec<F> {
    let group = layout.group(group_id).expect("test group must exist");
    let z_cols = group.fold_position_count * group.depth_commit;
    (0..z_cols)
        .map(|column| {
            let position = column / group.depth_commit;
            let commit_digit = column % group.depth_commit;
            let mut weight = F::zero();
            for unit in layout
                .ownership_units
                .iter()
                .filter(|unit| unit.group == group_id)
            {
                for (fold_digit, &fold) in fold_gadget.iter().enumerate().take(group.depth_fold) {
                    let physical = unit.z_range.start
                        + fold_digit
                        + group.depth_fold * (commit_digit + group.depth_commit * position);
                    let physical_block = physical / opening_layout.fold_position_count();
                    let physical_position = physical % opening_layout.fold_position_count();
                    let virtual_address =
                        physical_block * opening_layout.position_stride() + physical_position;
                    weight -= eq_eval_at_index(full_vec_randomness, virtual_address) * fold;
                }
            }
            weight
        })
        .collect()
}

fn rho_for_required(required: usize) -> Vec<F> {
    let bits = required.next_power_of_two().trailing_zeros() as usize;
    (0..bits)
        .map(|idx| test_scalar(901 + idx as u128))
        .collect()
}

fn projection_scales(alpha: F, base_d: usize, role_d: usize) -> Vec<F> {
    scalar_powers(alpha, role_d)
        .chunks(base_d)
        .map(|chunk| chunk[0])
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn projected_setup_weight_reference(
    plan: &SetupContributionPlan<F>,
    rho: &[F],
    required: usize,
    a_ratio: usize,
    b_ratio: usize,
    d_ratio: usize,
    a_scales: &[F],
    b_scales: &[F],
    d_scales: &[F],
) -> F {
    let mut acc = F::zero();
    for base_idx in 0..required {
        let mut weight = F::zero();
        for group in &plan.groups {
            let d_idx = base_idx / d_ratio;
            if d_idx < plan.d_rows * plan.d_physical_cols {
                let d_col = d_idx % plan.d_physical_cols;
                let d_row = d_idx / plan.d_physical_cols;
                if d_col >= group.e_col_offset
                    && d_col < group.e_col_offset + group.e_eq_slice.len()
                {
                    weight += d_scales[base_idx % d_ratio]
                        * group.d_weights[d_row]
                        * group.e_eq_slice[d_col - group.e_col_offset];
                }
            }

            let b_idx = base_idx / b_ratio;
            if b_idx < group.n_b * group.t_cols {
                let b_col = b_idx % group.t_cols;
                let b_row = b_idx / group.t_cols;
                weight +=
                    b_scales[base_idx % b_ratio] * group.b_weights[b_row] * group.t_eq_slice[b_col];
            }

            let a_idx = base_idx / a_ratio;
            if a_idx < group.n_a * group.z_cols {
                let a_col = a_idx % group.z_cols;
                let a_row = a_idx / group.z_cols;
                weight += a_scales[base_idx % a_ratio]
                    * group.a_row_weights[a_row]
                    * group.z_eq_slice[a_col];
            }
        }
        acc += eq_eval_at_index(rho, base_idx) * weight;
    }
    acc
}

#[test]
fn setup_index_weight_evaluator_matches_packed_mle_single_chunk() {
    let (inputs, groups, _static_plan, plan, tau1, full_vec_randomness, fold_gadget) =
        structured_weight_fixture(8, &[8], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let evaluator = SetupIndexWeightEvaluator::new::<F>(
        &inputs,
        &plan,
        &groups,
        &tau1,
        &full_vec_randomness,
        &fold_gadget,
        alpha,
    )
    .unwrap();
    assert_eq!(evaluator.required(), plan.required());

    let rho = rho_for_required(evaluator.required());
    let got = evaluator.evaluate(&rho).unwrap();
    let expected = plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn setup_index_weight_evaluator_matches_packed_mle_multi_chunk() {
    let (inputs, groups, _static_plan, plan, tau1, full_vec_randomness, fold_gadget) =
        structured_weight_fixture(8, &[2, 2, 2, 2], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let evaluator = SetupIndexWeightEvaluator::new::<F>(
        &inputs,
        &plan,
        &groups,
        &tau1,
        &full_vec_randomness,
        &fold_gadget,
        alpha,
    )
    .unwrap();
    let rho = rho_for_required(evaluator.required());
    assert_eq!(
        evaluator.evaluate(&rho).unwrap(),
        plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap()
    );
}

#[test]
fn setup_index_weight_evaluator_supports_non_power_of_two_ownership_widths() {
    let (inputs, groups, _static_plan, plan, tau1, full_vec_randomness, fold_gadget) =
        structured_weight_fixture(8, &[3, 5], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let evaluator = SetupIndexWeightEvaluator::new::<F>(
        &inputs,
        &plan,
        &groups,
        &tau1,
        &full_vec_randomness,
        &fold_gadget,
        alpha,
    )
    .unwrap();
    let rho = rho_for_required(evaluator.required());
    assert_eq!(
        evaluator.evaluate(&rho).unwrap(),
        plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap()
    );
}

#[test]
fn setup_index_weight_evaluator_applies_mixed_role_projection_lanes() {
    let alpha = test_scalar(3);
    let role_dims = crate::CommitmentRingDims {
        inner: 64,
        outer: 32,
        opening: 32,
    };
    let setup_ring_dim = 32;
    for ownership_widths in [&[8][..], &[2, 2, 2, 2][..], &[3, 5][..]] {
        let (inputs, groups, _static_plan, plan, tau1, full_vec_randomness, fold_gadget) =
            structured_weight_fixture(8, ownership_widths, role_dims);
        let evaluator = SetupIndexWeightEvaluator::new::<F>(
            &inputs,
            &plan,
            &groups,
            &tau1,
            &full_vec_randomness,
            &fold_gadget,
            alpha,
        )
        .unwrap();
        let rho = rho_for_required(evaluator.required());
        let got = evaluator.evaluate(&rho).unwrap();
        let expected = projected_setup_weight_reference(
            &plan,
            &rho,
            evaluator.required(),
            role_dims.d_a() / setup_ring_dim,
            role_dims.d_b() / setup_ring_dim,
            role_dims.d_d() / setup_ring_dim,
            &projection_scales(alpha, setup_ring_dim, role_dims.d_a()),
            &projection_scales(alpha, setup_ring_dim, role_dims.d_b()),
            &projection_scales(alpha, setup_ring_dim, role_dims.d_d()),
        );
        assert_eq!(got, expected, "ownership widths {ownership_widths:?}");
    }
}

#[test]
fn dense_z_eq_slice_uses_relative_high_carry() {
    let fold_position_count = 12;
    let depth_commit = 3;
    let depth_fold = 2;
    let num_points = 1;
    let full_vec_randomness = (0..9)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
    let inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: RelationMatrixRowLayout::WithoutDBlock,
        rows: 2,
        n_a: 1,
        n_b: 0,
        n_d: 0,
        num_groups: num_points,
        num_polys_per_group: vec![0],
        num_t_vectors: 0,
        num_claims: 1,
        live_fold_count: 4,
        fold_position_count,
        depth_open: 16,
        depth_commit,
        depth_fold,
        inner_width: fold_position_count * depth_commit,
        eq_tau1: vec![test_scalar(11), test_scalar(12)].into(),
    };

    let layout = OpeningBatchWitnessLayout::new(
        vec![OpeningBatchWitnessGroup {
            id: SemanticGroupId(0),
            num_claims: inputs.num_claims,
            live_fold_count: inputs.live_fold_count,
            fold_position_count: inputs.fold_position_count,
            depth_open: inputs.depth_open,
            depth_commit: inputs.depth_commit,
            depth_fold: inputs.depth_fold,
            n_a: inputs.n_a,
            e_setup_col_offset: 0,
        }],
        vec![SemanticGroupId(0)],
        vec![SemanticGroupId(0)],
        1,
        1,
        inputs.depth_fold,
    )
    .expect("layout");
    let plan = prepare_single_group_plan(
        &inputs,
        &full_vec_randomness,
        None,
        None,
        &fold_gadget,
        &layout,
    )
    .unwrap();

    let expected = expected_z_setup_weights(
        &layout,
        OpeningBlockLayout::new(1, layout.total_len()).unwrap(),
        SemanticGroupId(0),
        &fold_gadget,
        &full_vec_randomness,
    );

    assert_eq!(plan.groups[0].z_eq_slice, expected);
}

#[test]
fn setup_a_z_weights_do_not_include_commit_gadget() {
    let fold_position_count = 8;
    let depth_commit = 3;
    let depth_fold = 2;
    let num_points = 1;
    let log_basis = 4;
    let full_vec_randomness = (0..8)
        .map(|idx| test_scalar(701 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let commit_gadget = gadget_row_scalars::<F>(depth_commit, log_basis);
    let inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: RelationMatrixRowLayout::WithoutDBlock,
        rows: 2,
        n_a: 1,
        n_b: 0,
        n_d: 0,
        num_groups: num_points,
        num_polys_per_group: vec![0],
        num_t_vectors: 0,
        num_claims: 1,
        live_fold_count: 4,
        fold_position_count,
        depth_open: 16,
        depth_commit,
        depth_fold,
        inner_width: fold_position_count * depth_commit,
        eq_tau1: vec![test_scalar(11), test_scalar(12)].into(),
    };
    let layout = OpeningBatchWitnessLayout::new(
        vec![OpeningBatchWitnessGroup {
            id: SemanticGroupId(0),
            num_claims: inputs.num_claims,
            live_fold_count: inputs.live_fold_count,
            fold_position_count: inputs.fold_position_count,
            depth_open: inputs.depth_open,
            depth_commit: inputs.depth_commit,
            depth_fold: inputs.depth_fold,
            n_a: inputs.n_a,
            e_setup_col_offset: 0,
        }],
        vec![SemanticGroupId(0)],
        vec![SemanticGroupId(0)],
        1,
        1,
        inputs.depth_fold,
    )
    .expect("layout");

    let plan = prepare_single_group_plan(
        &inputs,
        &full_vec_randomness,
        None,
        None,
        &fold_gadget,
        &layout,
    )
    .unwrap();

    let expected = expected_z_setup_weights(
        &layout,
        OpeningBlockLayout::new(1, layout.total_len()).unwrap(),
        SemanticGroupId(0),
        &fold_gadget,
        &full_vec_randomness,
    );
    let wrong_with_commit_gadget = expected
        .iter()
        .enumerate()
        .map(|(k, &weight)| weight * commit_gadget[k % depth_commit])
        .collect::<Vec<_>>();

    assert_eq!(plan.groups[0].z_eq_slice, expected);
    assert_ne!(
        plan.groups[0].z_eq_slice, wrong_with_commit_gadget,
        "A setup weights are for A * G_fold * z_hat, not A * G_commit * G_fold * z_hat"
    );
}

#[test]
fn z_setup_weight_oracle_covers_multi_block_virtual_gaps() {
    let group_id = SemanticGroupId(0);
    let fold_position_count = 3;
    let depth_commit = 2;
    let depth_fold = 2;
    let layout = OpeningBatchWitnessLayout::new(
        vec![OpeningBatchWitnessGroup {
            id: group_id,
            num_claims: 1,
            live_fold_count: 2,
            fold_position_count,
            depth_open: 2,
            depth_commit,
            depth_fold,
            n_a: 1,
            e_setup_col_offset: 0,
        }],
        vec![group_id],
        vec![group_id],
        2,
        1,
        1,
    )
    .unwrap();
    let live_fold_count = [8usize, 4, 2]
        .into_iter()
        .find(|&blocks| {
            let physical_block_len = layout.total_len().div_ceil(blocks);
            physical_block_len != 0 && !physical_block_len.is_power_of_two()
        })
        .expect("test layout admits a gapped partition");
    let physical_block_len = layout.total_len().div_ceil(live_fold_count);
    let opening_layout = OpeningBlockLayout::new(live_fold_count, physical_block_len).unwrap();
    let point = (0..opening_layout.opening_len().trailing_zeros() as usize)
        .map(|index| test_scalar(1201 + index as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
    let mut got = vec![F::zero(); fold_position_count * depth_commit];
    setup_z_col_weights(
        &layout,
        opening_layout,
        group_id,
        fold_position_count,
        depth_commit,
        depth_fold,
        &point,
        &fold_gadget,
        &mut got,
    )
    .unwrap();
    let expected =
        expected_z_setup_weights(&layout, opening_layout, group_id, &fold_gadget, &point);
    assert_eq!(got, expected);
    assert!(layout.ownership_units.iter().any(|unit| {
        (0..fold_position_count).any(|position| {
            (0..depth_commit).any(|commit_digit| {
                (0..depth_fold).any(|fold_digit| {
                    let physical = unit.z_range.start
                        + fold_digit
                        + depth_fold * (commit_digit + depth_commit * position);
                    let virtual_address = (physical / physical_block_len)
                        * opening_layout.position_stride()
                        + physical % physical_block_len;
                    physical != virtual_address
                })
            })
        })
    }));
}

#[test]
fn single_group_plan_supports_multi_chunk_weights() {
    let live_fold_count = 4;
    let blocks_per_chunk = 2;
    let num_claims = 3;
    let depth_open = 2;
    let depth_commit = 2;
    let depth_fold = 2;
    let fold_position_count = 4;
    let n_a = 2;
    let n_b = 2;
    let n_d = 1;
    let log_basis = 4;
    let rows = 1 + n_a + n_b + n_d;
    let layout = OpeningBatchWitnessLayout::new(
        vec![OpeningBatchWitnessGroup {
            id: SemanticGroupId(0),
            num_claims,
            live_fold_count,
            fold_position_count,
            depth_open,
            depth_commit,
            depth_fold,
            n_a,
            e_setup_col_offset: 0,
        }],
        vec![SemanticGroupId(0)],
        vec![SemanticGroupId(0)],
        live_fold_count / blocks_per_chunk,
        n_d,
        depth_fold,
    )
    .expect("layout");
    let opening_layout = OpeningBlockLayout::new(1, layout.total_len()).unwrap();
    let groups = [SetupContributionGroupInputs {
        group_id: SemanticGroupId(0),
        e_col_offset: 0,
        num_claims,
        live_fold_count,
        fold_position_count,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        n_a,
        n_b,
        t_cols_per_vector: n_a * depth_open * live_fold_count,
        a_row_start: 1,
        b_row_start: 1 + n_a,
        layout: std::sync::Arc::new(layout),
        opening_layout,
    }];
    let inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
        rows,
        n_a,
        n_b,
        n_d,
        num_groups: 1,
        num_polys_per_group: vec![num_claims],
        num_t_vectors: num_claims,
        num_claims,
        live_fold_count,
        fold_position_count,
        depth_open,
        depth_commit,
        depth_fold,
        inner_width: fold_position_count * depth_commit,
        eq_tau1: (0..rows.next_power_of_two())
            .map(|idx| test_scalar(11 + idx as u128))
            .collect(),
    };
    let full_vec_randomness = (0..10)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let static_plan = SetupContributionPlan::prepare_static(
        &inputs,
        &groups,
        rows - n_d,
        n_d,
        num_claims * live_fold_count * depth_open,
    )
    .unwrap();
    let plan = SetupContributionPlan::finish_plan::<F>(
        &static_plan,
        &full_vec_randomness,
        None,
        None,
        Some(&fold_gadget),
        &groups,
        CommitmentRingDims::uniform(TEST_D),
    )
    .unwrap();

    let setup_len = plan.required();
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: TEST_D,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            TEST_D,
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
        .unwrap();
    assert_eq!(got, expected);

    let setup_index_weight = plan
        .materialize_setup_index_weights(test_scalar(3))
        .unwrap();
    let setup_view = setup
        .shared_matrix()
        .ring_view::<TEST_D>(1, setup_index_weight.len())
        .unwrap();
    let tie: F = setup_index_weight
        .iter()
        .zip(setup_view.as_slice())
        .map(|(w, ring)| eval_ring_at_pows(ring, &alpha_pows) * *w)
        .sum();
    assert_eq!(tie, got);
}

#[test]
fn packed_direct_matches_row_fallback_with_d_offset() {
    let plan = finalize_test_plan(
        2,
        5,
        vec![SetupContributionGroupPlan {
            e_col_offset: 2,
            t_cols: 4,
            z_cols: 3,
            n_a: 2,
            n_b: 2,
            required: 0,
            segments: Vec::new().into(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
            b_weights: vec![test_scalar(37), test_scalar(41)].into(),
            d_weights: vec![test_scalar(43), test_scalar(47)].into(),
        }],
        CommitmentRingDims::uniform(TEST_D),
    );
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: TEST_D,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            TEST_D,
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
        .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn multi_group_packed_direct_matches_row_fallback() {
    let plan = finalize_test_plan(
        2,
        5,
        vec![
            SetupContributionGroupPlan {
                e_col_offset: 2,
                t_cols: 4,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                required: 0,
                segments: Vec::new().into(),
                e_eq_slice: vec![test_scalar(2), test_scalar(3)],
                t_eq_slice: vec![
                    test_scalar(5),
                    test_scalar(7),
                    test_scalar(11),
                    test_scalar(13),
                ],
                z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
                a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
                b_weights: vec![test_scalar(37), test_scalar(41)].into(),
                d_weights: vec![test_scalar(43), test_scalar(47)].into(),
            },
            SetupContributionGroupPlan {
                e_col_offset: 0,
                t_cols: 4,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                required: 0,
                segments: Vec::new().into(),
                e_eq_slice: vec![test_scalar(53), test_scalar(59)],
                t_eq_slice: vec![
                    test_scalar(61),
                    test_scalar(67),
                    test_scalar(71),
                    test_scalar(73),
                ],
                z_eq_slice: vec![test_scalar(79), test_scalar(83), test_scalar(89)],
                a_row_weights: vec![test_scalar(97), test_scalar(101)].into(),
                b_weights: vec![test_scalar(103), test_scalar(107)].into(),
                d_weights: vec![test_scalar(109), test_scalar(113)].into(),
            },
        ],
        CommitmentRingDims::uniform(TEST_D),
    );
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: TEST_D,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            TEST_D,
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
        .unwrap();
    assert_eq!(got, expected);

    let setup_index_weight = plan
        .materialize_setup_index_weights(test_scalar(3))
        .unwrap();
    let setup_view = setup
        .shared_matrix()
        .ring_view::<TEST_D>(1, setup_index_weight.len())
        .unwrap();
    let tie: F = setup_index_weight
        .iter()
        .zip(setup_view.as_slice())
        .map(|(w, ring)| eval_ring_at_pows(ring, &alpha_pows) * *w)
        .sum();
    assert_eq!(tie, got);
}

#[test]
fn packed_direct_matches_row_fallback_with_nested_role_dims() {
    const D: usize = 64;
    const D_B: usize = 32;
    const D_D: usize = 32;
    let plan = finalize_test_plan(
        2,
        5,
        vec![SetupContributionGroupPlan {
            e_col_offset: 2,
            t_cols: 4,
            z_cols: 3,
            n_a: 2,
            n_b: 2,
            required: 0,
            segments: Vec::new().into(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
            b_weights: vec![test_scalar(37), test_scalar(41)].into(),
            d_weights: vec![test_scalar(43), test_scalar(47)].into(),
        }],
        CommitmentRingDims {
            inner: D,
            outer: D_B,
            opening: D_D,
        },
    );
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: D,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            D,
        ),
    );
    let alpha = test_scalar(3);
    let alpha_pows_a = scalar_powers(alpha, D);
    let alpha_pows_b = scalar_powers(alpha, D_B);
    let alpha_pows_d = scalar_powers(alpha, D_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d, D)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d)
        .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn packed_direct_rejects_non_decomposable_role_alpha_pows() {
    const D_A: usize = 64;
    const D_B: usize = 32;
    const D_D: usize = 32;
    let plan = finalize_test_plan(
        2,
        5,
        vec![SetupContributionGroupPlan {
            e_col_offset: 2,
            t_cols: 4,
            z_cols: 3,
            n_a: 2,
            n_b: 2,
            required: 0,
            segments: Vec::new().into(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
            b_weights: vec![test_scalar(37), test_scalar(41)].into(),
            d_weights: vec![test_scalar(43), test_scalar(47)].into(),
        }],
        CommitmentRingDims {
            inner: D_A,
            outer: D_B,
            opening: D_D,
        },
    );
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: D_A,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * D_A)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            D_A,
        ),
    );
    let alpha = test_scalar(3);
    let alpha_pows_a = scalar_powers(alpha, D_A);
    let mut alpha_pows_b = scalar_powers(alpha, D_B);
    let alpha_pows_d = scalar_powers(alpha, D_D);
    alpha_pows_b[1] += test_scalar(1);

    assert!(matches!(
        plan.evaluate_direct::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn packed_direct_accepts_d_footprint_at_nested_d_d() {
    // D-role columns are counted at d_d; comparing `required` against
    // total_ring_elements_at_dyn(d_a) falsely rejects valid setups when
    // d_d < d_a and the D footprint dominates.
    const D_A: usize = 64;
    const D_B: usize = 64;
    const D_D: usize = 32;
    let plan = finalize_test_plan(
        2,
        11,
        vec![SetupContributionGroupPlan {
            e_col_offset: 0,
            t_cols: 4,
            z_cols: 3,
            n_a: 2,
            n_b: 2,
            required: 0,
            segments: Vec::new().into(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
            b_weights: vec![test_scalar(37), test_scalar(41)].into(),
            d_weights: vec![test_scalar(43), test_scalar(47)].into(),
        }],
        CommitmentRingDims {
            inner: D_A,
            outer: D_B,
            opening: D_D,
        },
    );
    let setup_ring_elements = 20usize;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: D_A,
            max_setup_len: setup_ring_elements,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_ring_elements * D_A)
                .map(|idx| test_scalar(311 + idx as u128))
                .collect(),
            D_A,
        ),
    );
    let alpha = test_scalar(3);
    let alpha_pows_a = scalar_powers(alpha, D_A);
    let alpha_pows_b = scalar_powers(alpha, D_B);
    let alpha_pows_d = scalar_powers(alpha, D_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d, D_A)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d)
        .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn multi_group_packed_direct_matches_row_fallback_with_mismatched_t_cols() {
    let plan = finalize_test_plan(
        2,
        5,
        vec![
            SetupContributionGroupPlan {
                e_col_offset: 2,
                t_cols: 4,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                required: 0,
                segments: Vec::new().into(),
                e_eq_slice: vec![test_scalar(2), test_scalar(3)],
                t_eq_slice: vec![
                    test_scalar(5),
                    test_scalar(7),
                    test_scalar(11),
                    test_scalar(13),
                ],
                z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
                a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
                b_weights: vec![test_scalar(37), test_scalar(41)].into(),
                d_weights: vec![test_scalar(43), test_scalar(47)].into(),
            },
            SetupContributionGroupPlan {
                e_col_offset: 0,
                t_cols: 6,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                required: 0,
                segments: Vec::new().into(),
                e_eq_slice: vec![test_scalar(53), test_scalar(59)],
                t_eq_slice: vec![
                    test_scalar(61),
                    test_scalar(67),
                    test_scalar(71),
                    test_scalar(73),
                    test_scalar(79),
                    test_scalar(83),
                ],
                z_eq_slice: vec![test_scalar(89), test_scalar(97), test_scalar(101)],
                a_row_weights: vec![test_scalar(103), test_scalar(107)].into(),
                b_weights: vec![test_scalar(109), test_scalar(113)].into(),
                d_weights: vec![test_scalar(127), test_scalar(131)].into(),
            },
        ],
        CommitmentRingDims::uniform(TEST_D),
    );
    let setup_len = 12;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: TEST_D,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            TEST_D,
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
        .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn from_level_params_rejects_non_pow2_num_blocks() {
    let mut lp = LevelParams::log_basis_stub(3);
    lp.ring_dimension = 64;
    lp.role_dims = crate::CommitmentRingDims::uniform(64);
    lp.live_fold_count = 3;
    lp.fold_position_count = 8;
    lp.num_digits_commit = 2;
    lp.num_digits_open = 3;
    assert!(SetupContributionPlanInputs::<F>::from_level_params(
        &lp,
        &[2],
        RelationMatrixRowLayout::WithoutDBlock,
        2,
    )
    .is_err());
}
