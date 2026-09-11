use super::plan::{
    DirectScanState, DirectScanWeights, PhysicalBSetupPlan, ReducedDirectScanWeights,
    ReducedRoleCoefficientState, SetupContributionGroupPlan,
};
use super::test_oracle_weights::{setup_z_col_weights, RoleLaneSpec, RoleLaneWeighting};
use super::*;
use crate::{
    dyadic_block_ranges, gadget_row_scalars, AkitaExpandedSetup, AkitaSetupDescriptor,
    CommitmentRingDims, CommittedGroupParams, FlatMatrix, OpeningClaimsLayout, OpeningMethod,
    RingRole, WitnessLayout, WitnessQuotientRowLayout, WitnessUnitLayout,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::scalar_powers;
use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use jolt_field::{CanonicalEncoding, One, Prime128OffsetA7F7, Zero};

mod address_spans;
mod direct_evaluation_regressions;
mod fused_scan;
mod prepare;
mod span_evaluators;

type F = Prime128OffsetA7F7;
const TEST_D: usize = 64;
type StructuredWeightFixture = (
    TestSetupInputs,
    Vec<SetupContributionGroupInputs>,
    WitnessLayout,
    SetupContributionPlan<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
);
struct TestSetupInputs {
    level_params: CommittedGroupParams,
    opening_batch: OpeningClaimsLayout,
    eq_tau1: std::sync::Arc<[F]>,
}
impl TestSetupInputs {
    fn n_a(&self) -> usize {
        self.level_params.inner().matrix.output_rank()
    }
    fn num_positions_per_block(&self) -> usize {
        self.level_params.blocks().positions_per_block
    }
    fn depth_open(&self) -> usize {
        self.level_params.open().digits.num_digits
    }
    fn depth_commit(&self) -> usize {
        self.level_params.inner().digits.num_digits
    }
    fn depth_fold(&self) -> Result<usize, AkitaError> {
        Ok(self.level_params.num_digits_fold())
    }
}
fn test_scalar(value: u128) -> F {
    F::from_u128_checked(value).expect("test scalar must be canonical")
}

fn retarget_test_role_dims(params: &mut CommittedGroupParams, role_dims: CommitmentRingDims) {
    params.own_group_mut().opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(role_dims.d_a())
            .expect("test ring has a production challenge");
    let inner_bound = crate::sis::rounded_up_role_a_inf_norm(
        params.inner().matrix.security_policy(),
        params
            .inner()
            .matrix
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        params.inner().matrix.sis_modulus_profile(),
        role_dims.d_a(),
        params.open().digits.log_basis,
        &params.fold_challenge_config(),
        params.num_digits_fold(),
        params.witness_chunk.num_chunks,
    )
    .expect("retargeted exact A bound");
    let inner = &params.inner().matrix;
    params.own_group_mut().profile.inner.matrix = crate::InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner_bound,
        role_dims.d_a(),
    );
    let outer = &params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = crate::OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        outer.coeff_linf_bound(),
        role_dims.d_b(),
    );
    let open = &params.open().matrix;
    params.open_matrix = crate::OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        open.output_rank(),
        open.input_width(),
        open.coeff_linf_bound(),
        role_dims.d_d(),
    );
}

fn retarget_precommitted_test_role_dims(
    params: &mut CommittedGroupParams,
    group_id: usize,
    inner_ring_dimension: usize,
    outer_ring_dimension: usize,
) {
    let group = params.preceding_group_mut_for_test(group_id).unwrap();
    group.opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(inner_ring_dimension)
            .expect("test precommitted ring has a production challenge");
    let inner_bound = crate::sis::rounded_up_role_a_inf_norm(
        group.profile.inner.matrix.security_policy(),
        group
            .profile
            .inner
            .matrix
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        group.profile.inner.matrix.sis_modulus_profile(),
        inner_ring_dimension,
        group.opening.log_basis_open,
        &group.opening.fold_challenge_config,
        group.opening.num_digits_fold,
        1,
    )
    .expect("retargeted precommitted exact A bound");
    let mut layout = group.profile;
    let inner = &layout.inner.matrix;
    let inner_output_rank = inner.output_rank();
    layout.inner.matrix = crate::InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner_bound,
        inner_ring_dimension,
    );
    let outer = &layout.outer.matrix;
    let projected_width = inner_output_rank
        .checked_mul(layout.outer.digits.num_digits)
        .and_then(|width| width.checked_mul(group.profile.blocks.live_blocks))
        .and_then(|width| width.checked_mul(group.profile.group.num_polynomials()))
        .and_then(|width| width.checked_mul(inner_ring_dimension / outer_ring_dimension))
        .expect("test precommitted B width");
    layout.outer.matrix = crate::OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        projected_width,
        outer.coeff_linf_bound(),
        outer_ring_dimension,
    );
    group.profile = layout;
}

#[allow(clippy::too_many_arguments)]
fn test_inputs(
    n_a: usize,
    n_b: usize,
    n_d: usize,
    num_claims: usize,
    num_live_blocks: usize,
    num_positions_per_block: usize,
    depth_open: usize,
    depth_commit: usize,
    depth_fold: usize,
    log_basis: u32,
    eq_tau1: Vec<F>,
) -> TestSetupInputs {
    test_inputs_for_group_sizes(
        n_a,
        n_b,
        n_d,
        &[num_claims],
        num_live_blocks,
        num_positions_per_block,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        eq_tau1,
    )
}
#[allow(clippy::too_many_arguments)]
fn test_inputs_for_group_sizes(
    n_a: usize,
    n_b: usize,
    n_d: usize,
    group_sizes: &[usize],
    num_live_blocks: usize,
    num_positions_per_block: usize,
    depth_open: usize,
    depth_commit: usize,
    depth_fold: usize,
    log_basis: u32,
    mut eq_tau1: Vec<F>,
) -> TestSetupInputs {
    let num_claims: usize = group_sizes.iter().copied().sum();
    let mut lp = CommittedGroupParams::params_only(
        crate::sis::SisModulusProfileId::Q128OffsetA7F7,
        TEST_D,
        log_basis,
        n_a,
        n_b,
        n_d,
        SparseChallengeConfig::production_for_ring_dim(TEST_D)
            .expect("test ring has a production challenge"),
    )
    .with_decomp(
        num_positions_per_block,
        num_live_blocks * num_positions_per_block,
        depth_commit,
        depth_open,
        depth_open,
    )
    .expect("test level params");
    lp.own_group_mut().opening.num_digits_fold = depth_fold;
    let expected_b_width = num_claims
        .checked_mul(n_a)
        .and_then(|width| width.checked_mul(depth_open))
        .and_then(|width| width.checked_mul(num_live_blocks))
        .expect("test B width");
    if lp.outer().matrix.input_width() < expected_b_width {
        lp.own_group_mut().profile.outer.matrix = crate::OuterCommitMatrixParams::new_unchecked(
            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            crate::sis::SisModulusProfileId::Q128OffsetA7F7,
            n_b,
            expected_b_width,
            3,
            TEST_D,
        );
    }
    if lp.inner().matrix.coeff_linf_bound() == Some(0) {
        let a_bound = crate::sis::rounded_up_role_a_inf_norm(
            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            crate::sis::SisModulusProfileId::Q128OffsetA7F7,
            TEST_D,
            lp.open().digits.log_basis,
            &lp.fold_challenge_config(),
            lp.num_digits_fold(),
            lp.witness_chunk.num_chunks,
        )
        .expect("exact test A bound");
        lp.own_group_mut().profile.inner.matrix = crate::InnerCommitMatrixParams::new_unchecked(
            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            crate::sis::SisModulusProfileId::Q128OffsetA7F7,
            n_a,
            lp.inner().matrix.input_width(),
            a_bound,
            TEST_D,
        );
    }
    if lp.outer().matrix.coeff_linf_bound() == 0 {
        lp.own_group_mut().profile.outer.matrix = crate::OuterCommitMatrixParams::new_unchecked(
            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            crate::sis::SisModulusProfileId::Q128OffsetA7F7,
            n_b,
            lp.outer().matrix.input_width(),
            3,
            TEST_D,
        );
    }
    if group_sizes.len() > 1 {
        lp.set_precommitted_groups(
            group_sizes[..group_sizes.len() - 1]
                .iter()
                .map(|&_group_size| {
                    let mut layout = crate::GroupCommitPhaseParams::from_params_unchecked_for_test(
                        crate::PolynomialGroupLayout::new(0, 1),
                        &lp,
                    );
                    let expected_group_b_width = lp
                        .inner()
                        .matrix
                        .output_rank()
                        .checked_mul(lp.outer().digits.num_digits)
                        .and_then(|width| width.checked_mul(layout.blocks.live_blocks))
                        .and_then(|width| width.checked_mul(layout.group.num_polynomials()))
                        .expect("test precommitted B width");
                    let outer_commit_matrix = crate::OuterCommitMatrixParams::new_unchecked(
                        lp.outer().matrix.security_policy(),
                        lp.outer().matrix.sis_table_key().table_digest,
                        lp.outer().matrix.sis_modulus_profile(),
                        lp.outer().matrix.output_rank(),
                        expected_group_b_width,
                        lp.outer().matrix.coeff_linf_bound(),
                        lp.d_a(),
                    );
                    layout.outer.matrix = outer_commit_matrix;
                    crate::GroupOpenPhaseParams {
                        setup_natural_len: None,
                        profile: layout,
                        opening: crate::GroupOpeningPlan::evaluation_trace(
                            lp.fold_challenge_config(),
                            lp.open().digits.log_basis,
                            lp.open().digits.num_digits,
                            depth_fold,
                        ),
                    }
                })
                .collect(),
        )
        .unwrap();
    }
    let opening_batch =
        OpeningClaimsLayout::from_group_sizes(0, group_sizes).expect("test opening batch");
    let relation_rows = lp
        .relation_matrix_row_count(opening_batch.num_groups())
        .expect("test relation rows");
    eq_tau1.resize(relation_rows, F::zero());
    TestSetupInputs {
        level_params: lp,
        opening_batch,
        eq_tau1: eq_tau1.into(),
    }
}
#[allow(clippy::too_many_arguments)]
fn test_witness_layout(
    num_claims: usize,
    num_live_blocks: usize,
    num_positions_per_block: usize,
    depth_open: usize,
    depth_commit: usize,
    depth_fold: usize,
    n_a: usize,
    num_chunks: usize,
    relation_rows: usize,
    quotient_depth: usize,
) -> WitnessLayout {
    let mut cursor = 0usize;
    let chunk_ranges =
        dyadic_block_ranges(num_live_blocks, num_chunks).expect("test chunk partition");
    let mut units = Vec::with_capacity(num_chunks);
    for (chunk_index, global_block_range) in chunk_ranges.into_iter().enumerate() {
        let chunk_num_live_blocks = global_block_range.len();
        let z_len = num_positions_per_block * depth_commit * depth_fold * TEST_D;
        let z_range = cursor..cursor + z_len;
        let e_range =
            z_range.end..z_range.end + num_claims * chunk_num_live_blocks * depth_open * TEST_D;
        let t_range = e_range.end
            ..e_range.end + num_claims * chunk_num_live_blocks * n_a * depth_commit * TEST_D;
        cursor = t_range.end;
        units.push(WitnessUnitLayout::new_for_test(
            0,
            chunk_index,
            global_block_range.start,
            chunk_num_live_blocks,
            z_range,
            e_range,
            crate::RelationRowGeometry::native(TEST_D).unwrap(),
            t_range,
        ));
    }
    let r_rows = (0..relation_rows)
        .map(|_| {
            let range = cursor..cursor + quotient_depth * TEST_D;
            cursor = range.end;
            WitnessQuotientRowLayout::new_for_test(
                crate::RelationRowGeometry::native(TEST_D).unwrap(),
                range,
            )
        })
        .collect();
    WitnessLayout::new_for_test(units, r_rows, quotient_depth)
}
fn prepare_test_plan(
    inputs: &TestSetupInputs,
    witness_layout: &WitnessLayout,
    opening_source_len: usize,
    groups: &[SetupContributionGroupInputs],
    full_vec_randomness: &[F],
    fold_gadget: Option<&[F]>,
    role_dims: CommitmentRingDims,
) -> Result<SetupContributionPlan<F>, AkitaError> {
    let relation_address_geometry =
        crate::RelationAddressGeometry::new(role_dims, role_dims.d_a(), opening_source_len)?;
    let mut plan = SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        inputs.eq_tau1.clone(),
        witness_layout,
        groups,
        PreparedRelationAddress::new(full_vec_randomness)?,
        fold_gadget,
        relation_address_geometry,
    )?;
    plan.materialize_direct_scan(PreparedCoefficientFunctional::lifted_power(test_scalar(3)))?;
    Ok(plan)
}
fn finalize_test_plan(
    d_rows: usize,
    d_physical_cols: usize,
    groups: Vec<(SetupContributionGroupPlan<F>, DirectScanWeights<F>)>,
    role_dims: CommitmentRingDims,
) -> SetupContributionPlan<F> {
    let (groups, direct_groups): (Vec<_>, Vec<_>) = groups.into_iter().unzip();
    let a_footprint = groups
        .iter()
        .map(|group| group.n_a * group.z_cols)
        .max()
        .unwrap();
    let b_footprint = groups
        .iter()
        .map(|group| group.physical_b.physical_footprint().unwrap())
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
        d_weights: (0..d_rows)
            .map(|idx| test_scalar(43 + 4 * idx as u128))
            .collect::<Vec<_>>()
            .into(),
        setup_index_tensors: Vec::new(),
        relation_address: PreparedRelationAddress::new(&[]).unwrap(),
        setup_relation_address: PreparedRelationAddress::new(&[]).unwrap(),
        relation_base_bridge_point: Vec::new().into(),
        relation_address_geometry: crate::RelationAddressGeometry::new(
            role_dims,
            role_dims.d_a(),
            role_dims.common_relation_coeff_count(),
        )
        .unwrap(),
        projection_geometry,
        direct_scan_state: DirectScanState::Unprepared,
    };
    for (group, weights) in plan.groups.iter_mut().zip(&direct_groups) {
        group.role_dims = role_dims;
        group
            .set_projection_ratios(
                plan.projection_geometry.base_ring_dim(),
                plan.relation_address_geometry
                    .relation_coefficient_block_len(),
            )
            .expect("valid test group projection");
        group
            .refresh_segments(weights, &plan.d_weights, plan.d_rows, plan.d_physical_cols)
            .expect("valid cached setup scan segments");
    }
    plan.direct_scan_state = DirectScanState::Lifted {
        alpha: test_scalar(3),
        groups: direct_groups,
    };
    plan
}

#[allow(clippy::too_many_arguments)]
fn test_group_plan(
    d_col_range: std::ops::Range<usize>,
    t_cols: usize,
    z_cols: usize,
    n_a: usize,
    n_b: usize,
    e_eq_slice: Vec<F>,
    t_eq_slice: Vec<F>,
    z_eq_slice: Vec<F>,
    a_row_weights: Vec<F>,
    b_weights: Vec<F>,
) -> (SetupContributionGroupPlan<F>, DirectScanWeights<F>) {
    let physical_b = PhysicalBSetupPlan::new(
        crate::CommitmentSliceGeometry::try_new(
            crate::CommitmentSliceCount::ONE,
            t_cols,
            1,
            1,
            1,
            64,
            64,
        )
        .unwrap(),
        n_b,
        b_weights.into(),
    )
    .unwrap();
    let group = SetupContributionGroupPlan {
        group_id: 0,
        opening_method: OpeningMethod::EvaluationTrace,
        role_dims: CommitmentRingDims::uniform(64),
        a_ratio: 1,
        b_ratio: 1,
        d_ratio: 1,
        a_relation_ratio: 1,
        b_relation_ratio: 1,
        d_relation_ratio: 1,
        opening_subcolumns: 1,
        consistency_weight: F::one(),
        num_claims: 0,
        num_live_blocks: 0,
        num_positions_per_block: z_cols,
        depth_witness: 1,
        depth_commit: 1,
        depth_open: 1,
        log_basis_inner: 1,
        log_basis_outer: 1,
        log_basis_open: 1,
        d_col_range,
        z_cols,
        n_a,
        physical_b,
        required: 0,
        segments: Vec::new().into(),
        a_row_weights: a_row_weights.into(),
        fold_gadget: vec![F::one()].into(),
        active_unit_ranges: Vec::new().into(),
        num_physical_units: 0,
        d_tensors: Vec::new(),
        a_tensors: Vec::new(),
    };
    (
        group,
        DirectScanWeights {
            e: e_eq_slice,
            t: t_eq_slice,
            z: z_eq_slice,
        },
    )
}

#[test]
fn structured_evaluation_rejects_alpha_mismatch_after_direct_materialization() {
    let plan = finalize_test_plan(
        1,
        1,
        vec![test_group_plan(
            0..1,
            1,
            1,
            1,
            1,
            vec![test_scalar(2)],
            vec![test_scalar(5)],
            vec![test_scalar(7)],
            vec![test_scalar(11)],
            vec![test_scalar(13)],
        )],
        CommitmentRingDims::uniform(TEST_D),
    );
    assert!(matches!(
        plan.evaluate_structured_group::<F>(0, &[], &[], test_scalar(17)),
        Err(AkitaError::InvalidSetup(_))
    ));
}

fn prepare_single_group_plan(
    inputs: &TestSetupInputs,
    full_vec_randomness: &[F],
    fold_gadget: &[F],
    layout: &WitnessLayout,
) -> Result<SetupContributionPlan<F>, AkitaError> {
    let group = test_single_group_descriptor(inputs)?;
    prepare_test_plan(
        inputs,
        layout,
        layout.live_coeff_len(),
        &[group],
        full_vec_randomness,
        Some(fold_gadget),
        CommitmentRingDims::uniform(TEST_D),
    )
}
fn test_single_group_descriptor(
    inputs: &TestSetupInputs,
) -> Result<SetupContributionGroupInputs, AkitaError> {
    let order = inputs.opening_batch.root_group_order()?;
    let [group_index] = order.as_slice() else {
        return Err(AkitaError::InvalidSetup(
            "single-group test fixture requires exactly one commitment group".into(),
        ));
    };
    let group_lp = inputs
        .level_params
        .group_params(&inputs.opening_batch, *group_index)?;
    let group_layout = inputs.opening_batch.group_layout(*group_index)?;
    let num_claims = group_layout.num_polynomials();
    let a_range = inputs
        .level_params
        .a_row_range(&inputs.opening_batch, *group_index)?;
    let b_range = inputs
        .level_params
        .commitment_row_range(&inputs.opening_batch, *group_index)?;
    Ok(SetupContributionGroupInputs {
        group_id: *group_index,
        num_claims,
        depth_fold: group_lp.num_digits_fold(),
        a_row_start: a_range.start,
        b_row_start: b_range.start,
    })
}
fn structured_weight_fixture(
    num_live_blocks: usize,
    ownership_widths: &[usize],
    role_dims: CommitmentRingDims,
) -> StructuredWeightFixture {
    structured_weight_fixture_with_outgoing(
        num_live_blocks,
        ownership_widths,
        role_dims,
        role_dims.d_a(),
    )
}

fn structured_weight_fixture_with_outgoing(
    num_live_blocks: usize,
    ownership_widths: &[usize],
    role_dims: CommitmentRingDims,
    outgoing_ring_dim: usize,
) -> StructuredWeightFixture {
    structured_weight_fixture_with_slices(
        num_live_blocks,
        ownership_widths,
        role_dims,
        outgoing_ring_dim,
        crate::CommitmentSliceCount::ONE,
    )
}

fn structured_weight_fixture_with_slices(
    num_live_blocks: usize,
    ownership_widths: &[usize],
    role_dims: CommitmentRingDims,
    outgoing_ring_dim: usize,
    outer_slice_count: crate::CommitmentSliceCount,
) -> StructuredWeightFixture {
    let num_claims = 2;
    let depth_open = 2;
    let depth_commit = 2;
    let depth_fold = 2;
    let num_positions_per_block = 8;
    let n_a = 2;
    let n_b = if outer_slice_count.is_sliced() { 1 } else { 2 };
    let n_d = 2;
    let log_basis = 4;
    assert_eq!(ownership_widths.iter().sum::<usize>(), num_live_blocks);
    let z_len = num_positions_per_block * depth_commit * depth_fold * role_dims.d_a();
    let mut cursor = 0usize;
    let mut global_block_base = 0usize;
    let ownership_units = ownership_widths
        .iter()
        .copied()
        .enumerate()
        .map(|(chunk, blocks)| {
            let z_range = cursor..cursor + z_len;
            let e_len = num_claims * depth_open * blocks * role_dims.d_a();
            let e_range = z_range.end..z_range.end + e_len;
            let t_len = n_a * num_claims * depth_commit * blocks * role_dims.d_a();
            let t_range = e_range.end..e_range.end + t_len;
            cursor = t_range.end;
            let unit = WitnessUnitLayout::new_for_test(
                0,
                chunk,
                global_block_base,
                blocks,
                z_range,
                e_range,
                crate::RelationRowGeometry::native(role_dims.d_a()).unwrap(),
                t_range,
            );
            global_block_base += blocks;
            unit
        })
        .collect::<Vec<_>>();
    let r_rows = (0..n_d)
        .map(|_| {
            let range = cursor..cursor + depth_fold * role_dims.d_d();
            cursor = range.end;
            WitnessQuotientRowLayout::new_for_test(
                crate::RelationRowGeometry::native(role_dims.d_d()).unwrap(),
                range,
            )
        })
        .collect();
    let layout = WitnessLayout::new_for_test(ownership_units, r_rows, depth_fold);
    let tau1 = (0..4)
        .map(|idx| test_scalar(31 + idx as u128))
        .collect::<Vec<_>>();
    let mut inputs = test_inputs(
        n_a,
        n_b,
        n_d,
        num_claims,
        num_live_blocks,
        num_positions_per_block,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        EqPolynomial::evals(&tau1).unwrap(),
    );
    retarget_test_role_dims(&mut inputs.level_params, role_dims);
    inputs
        .level_params
        .own_group_mut()
        .profile
        .outer_slice_count = outer_slice_count;
    let slice_geometry = crate::CommitmentSliceGeometry::try_new(
        outer_slice_count,
        num_live_blocks,
        num_claims,
        n_a,
        depth_commit,
        role_dims.d_a(),
        role_dims.d_b(),
    )
    .unwrap();
    let outer = &inputs.level_params.outer().matrix;
    inputs.level_params.own_group_mut().profile.outer.matrix =
        crate::OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank(),
            slice_geometry.physical_input_width(),
            outer.coeff_linf_bound(),
            role_dims.d_b(),
        );
    let relation_rows = inputs
        .level_params
        .relation_matrix_row_count(inputs.opening_batch.num_groups())
        .unwrap();
    let mut eq_tau1 = EqPolynomial::evals(&tau1).unwrap();
    eq_tau1.resize(relation_rows, F::zero());
    inputs.eq_tau1 = eq_tau1.into();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let opening_source_len = layout.live_coeff_len();
    let relation_address_geometry =
        crate::RelationAddressGeometry::new(role_dims, outgoing_ring_dim, opening_source_len)
            .unwrap();
    let address_bits = relation_address_geometry.relation_lane_variable_count();
    let full_vec_randomness = (0..address_bits)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let groups = vec![SetupContributionGroupInputs {
        group_id: 0,
        num_claims,
        depth_fold,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    }];
    let mut plan = SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        inputs.eq_tau1.clone(),
        &layout,
        &groups,
        PreparedRelationAddress::new(&full_vec_randomness).unwrap(),
        Some(&fold_gadget),
        relation_address_geometry,
    )
    .unwrap();
    plan.materialize_direct_scan(PreparedCoefficientFunctional::lifted_power(test_scalar(3)))
        .unwrap();
    (
        inputs,
        groups,
        layout,
        plan,
        tau1,
        full_vec_randomness,
        fold_gadget,
    )
}
fn expected_z_setup_weights(
    layout: &WitnessLayout,
    opening_source_len: usize,
    group_id: usize,
    num_positions_per_block: usize,
    depth_commit: usize,
    fold_gadget: &[F],
    full_vec_randomness: &[F],
) -> Vec<F> {
    let depth_fold = fold_gadget.len();
    let z_cols = num_positions_per_block * depth_commit;
    (0..z_cols)
        .map(|column| {
            let position = column / depth_commit;
            let commit_digit = column % depth_commit;
            let mut weight = F::zero();
            for unit in layout.units_for_group(group_id).unwrap() {
                for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                    let physical = unit.z_range().start
                        + TEST_D
                            * (fold_digit + depth_fold * (commit_digit + depth_commit * position));
                    let opening_address =
                        crate::checked_opening_source_index(opening_source_len, physical / TEST_D)
                            .unwrap();
                    weight -= eq_eval_at_index(full_vec_randomness, opening_address) * fold;
                }
            }
            weight
        })
        .collect()
}
struct HeterogeneousSetupFixture {
    inputs: TestSetupInputs,
    groups: Vec<SetupContributionGroupInputs>,
    witness_layout: WitnessLayout,
    relation_address_geometry: crate::RelationAddressGeometry,
    relation_point: Vec<F>,
    fold_gadget: Vec<F>,
}

fn heterogeneous_setup_fixture() -> HeterogeneousSetupFixture {
    let quotient_depth = 2;
    let group_shapes = [
        // Relation order deliberately differs from numeric group order.
        (1usize, 1usize, 1usize, 1usize, 1usize),
        (0usize, 1usize, 1usize, 1usize, 1usize),
    ];
    let tau1 = vec![
        test_scalar(31),
        test_scalar(32),
        test_scalar(33),
        test_scalar(34),
    ];
    let mut inputs = test_inputs_for_group_sizes(
        1,
        1,
        1,
        &[1, 1],
        1,
        2,
        1,
        1,
        quotient_depth,
        4,
        EqPolynomial::evals(&tau1).unwrap(),
    );
    retarget_test_role_dims(
        &mut inputs.level_params,
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
    );
    retarget_precommitted_test_role_dims(&mut inputs.level_params, 0, 64, 64);
    let joint_geometry = crate::RelationWitnessGeometry::for_evaluation_trace_execution(
        &inputs.level_params,
        &inputs.opening_batch,
    )
    .unwrap();
    let witness_layout = WitnessLayout::new(
        &inputs.level_params,
        &inputs.opening_batch,
        &joint_geometry,
        1,
        crate::RelationQuotientPlan::quotient_lift(quotient_depth).unwrap(),
    )
    .unwrap();
    let opening_source_len = witness_layout.live_coeff_len();
    let groups: Vec<_> = group_shapes
        .iter()
        .map(
            |&(group_id, num_claims, _num_live_blocks, _depth_open, _depth_commit)| {
                let a_range = inputs
                    .level_params
                    .a_row_range(&inputs.opening_batch, group_id)
                    .unwrap();
                let b_range = inputs
                    .level_params
                    .commitment_row_range(&inputs.opening_batch, group_id)
                    .unwrap();
                SetupContributionGroupInputs {
                    group_id,
                    num_claims,
                    depth_fold: quotient_depth,
                    a_row_start: a_range.start,
                    b_row_start: b_range.start,
                }
            },
        )
        .collect();
    validate_setup_inputs(
        &inputs.level_params,
        &inputs.opening_batch,
        &witness_layout,
        &groups,
    )
    .unwrap();
    let relation_address_geometry = inputs
        .level_params
        .relation_address_geometry(&inputs.opening_batch, 1, 128, opening_source_len)
        .unwrap();
    let randomness_bits = relation_address_geometry.relation_lane_variable_count();
    let full_vec_randomness = (0..randomness_bits)
        .map(|index| test_scalar(101 + index as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(quotient_depth, 4);
    HeterogeneousSetupFixture {
        inputs,
        groups,
        witness_layout,
        relation_address_geometry,
        relation_point: full_vec_randomness,
        fold_gadget,
    }
}

#[test]
fn heterogeneous_relation_ordered_setup_layout_matches_structured_oracles() {
    let HeterogeneousSetupFixture {
        inputs,
        groups,
        witness_layout,
        relation_address_geometry,
        relation_point: full_vec_randomness,
        fold_gadget,
    } = heterogeneous_setup_fixture();
    let mut plan = SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        inputs.eq_tau1.clone(),
        &witness_layout,
        &groups,
        PreparedRelationAddress::new(&full_vec_randomness).unwrap(),
        Some(&fold_gadget),
        relation_address_geometry,
    )
    .unwrap();
    plan.materialize_direct_scan(PreparedCoefficientFunctional::lifted_power(test_scalar(3)))
        .unwrap();
    assert_eq!(
        plan.groups
            .iter()
            .find(|group| group.group_id == 1)
            .unwrap()
            .d_col_range,
        0..2
    );
    assert_eq!(
        plan.groups
            .iter()
            .find(|group| group.group_id == 0)
            .unwrap()
            .d_col_range,
        2..3
    );
    let alpha = test_scalar(3);
    let setup_idx_bits = plan.required().next_power_of_two().trailing_zeros() as usize;
    let rho_setup_idx = (0..setup_idx_bits)
        .map(|index| test_scalar(1301 + index as u128))
        .collect::<Vec<_>>();
    let dense_mle = plan
        .materialize_setup_index_weights(alpha)
        .unwrap()
        .into_iter()
        .enumerate()
        .fold(F::zero(), |acc, (index, weight)| {
            acc + eq_eval_at_index(&rho_setup_idx, index) * weight
        });
    assert_eq!(
        plan.evaluate_setup_index_weight_mle(&rho_setup_idx, alpha)
            .unwrap(),
        dense_mle,
        "multi-group setup-index MLE must match the full plan"
    );
    for (group_index, group) in plan.groups.iter().enumerate() {
        let block_challenges = (0..group.num_claims * group.num_live_blocks)
            .map(|index| test_scalar(1501 + 17 * group.group_id as u128 + index as u128))
            .collect::<Vec<_>>();
        let opening_a_evals = (0..group.num_positions_per_block)
            .map(|index| test_scalar(1601 + 19 * group.group_id as u128 + index as u128))
            .collect::<Vec<_>>();
        let reference = span_evaluators::structured_slice_reference(
            group,
            plan.direct_scan_state.weights(group_index).unwrap(),
            &block_challenges,
            &opening_a_evals,
            alpha,
        );
        assert_eq!(
            plan.evaluate_structured_group::<F>(
                group.group_id,
                &block_challenges,
                &opening_a_evals,
                alpha,
            )
            .unwrap(),
            reference,
            "full structured evaluation must match group {} dense oracle",
            group.group_id
        );
    }
}

#[test]
fn setup_a_z_weights_do_not_include_commit_gadget() {
    let num_positions_per_block = 8;
    let depth_commit = 3;
    let depth_fold = 2;
    let log_basis = 4;
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let commit_gadget = gadget_row_scalars::<F>(depth_commit, log_basis);
    let inputs = test_inputs(
        1,
        1,
        1,
        1,
        4,
        num_positions_per_block,
        16,
        depth_commit,
        depth_fold,
        log_basis,
        vec![test_scalar(11), test_scalar(12)],
    );
    let joint_geometry = crate::RelationWitnessGeometry::for_evaluation_trace_execution(
        &inputs.level_params,
        &inputs.opening_batch,
    )
    .unwrap();
    let layout = WitnessLayout::new(
        &inputs.level_params,
        &inputs.opening_batch,
        &joint_geometry,
        1,
        crate::RelationQuotientPlan::quotient_lift(inputs.depth_fold().unwrap()).unwrap(),
    )
    .unwrap();
    let relation_geometry = inputs
        .level_params
        .relation_address_geometry(&inputs.opening_batch, 1, TEST_D, layout.live_coeff_len())
        .unwrap();
    let full_vec_randomness = (0..relation_geometry.relation_lane_variable_count())
        .map(|idx| test_scalar(701 + idx as u128))
        .collect::<Vec<_>>();
    let plan =
        prepare_single_group_plan(&inputs, &full_vec_randomness, &fold_gadget, &layout).unwrap();
    let expected = expected_z_setup_weights(
        &layout,
        layout.live_coeff_len(),
        0,
        num_positions_per_block,
        depth_commit,
        &fold_gadget,
        &full_vec_randomness,
    );
    let wrong_with_commit_gadget = expected
        .iter()
        .enumerate()
        .map(|(k, &weight)| weight * commit_gadget[k % depth_commit])
        .collect::<Vec<_>>();
    let z_eq_slice = plan.group_column_eq_slices(0).unwrap().2;
    assert_eq!(z_eq_slice, expected);
    assert_ne!(
        z_eq_slice, wrong_with_commit_gadget,
        "A setup weights are for A * G_fold * z_hat, not A * G_commit * G_fold * z_hat"
    );
}
#[test]
fn z_setup_weight_oracle_uses_physical_addresses() {
    let group_id = 0;
    let num_positions_per_block = 4;
    let depth_commit = 2;
    let depth_fold = 2;
    let layout = test_witness_layout(
        1,
        2,
        num_positions_per_block,
        2,
        depth_commit,
        depth_fold,
        1,
        2,
        1,
        1,
    );
    let opening_source_len = layout.live_coeff_len();
    let point = (0..crate::opening_domain_len(opening_source_len)
        .unwrap()
        .trailing_zeros() as usize)
        .map(|index| test_scalar(1201 + index as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
    let mut got = vec![F::zero(); num_positions_per_block * depth_commit];
    let eq_window = akita_algebra::offset_eq::OffsetEqWindow::new(&point).unwrap();
    let uniform_lane_alpha = [F::one()];
    let uniform_spec = RoleLaneSpec {
        a_ratio: 1,
        role_subcolumns: 1,
        role_lanes: 1,
        weighting: RoleLaneWeighting::Lifted(&uniform_lane_alpha),
    };
    setup_z_col_weights(
        &layout,
        opening_source_len,
        group_id,
        num_positions_per_block,
        depth_commit,
        depth_fold,
        &eq_window,
        &fold_gadget,
        &uniform_spec,
        &mut got,
    )
    .unwrap();
    let expected = expected_z_setup_weights(
        &layout,
        opening_source_len,
        group_id,
        num_positions_per_block,
        depth_commit,
        &fold_gadget,
        &point,
    );
    assert_eq!(got, expected);
    assert_eq!(
        crate::checked_opening_source_index(opening_source_len, opening_source_len - 1).unwrap(),
        opening_source_len - 1
    );
}
#[test]
fn single_group_plan_supports_multi_chunk_weights() {
    let num_live_blocks = 5;
    let num_chunks = 2;
    let num_claims = 3;
    let depth_open = 2;
    let depth_commit = 2;
    let depth_fold = 2;
    let num_positions_per_block = 4;
    let n_a = 2;
    let n_b = 2;
    let n_d = 1;
    let log_basis = 4;
    let rows = 1 + n_a + n_b + n_d;
    let layout = test_witness_layout(
        num_claims,
        num_live_blocks,
        num_positions_per_block,
        depth_open,
        depth_commit,
        depth_fold,
        n_a,
        num_chunks,
        n_d,
        depth_fold,
    );
    assert_eq!(layout.units()[0].global_block_range(), 0..2);
    assert_eq!(layout.units()[1].global_block_range(), 2..5);
    let opening_source_len = layout.live_coeff_len();
    let group = SetupContributionGroupInputs {
        group_id: 0,
        num_claims,
        depth_fold,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    };
    let inputs = test_inputs(
        n_a,
        n_b,
        n_d,
        num_claims,
        num_live_blocks,
        num_positions_per_block,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        (0..rows.next_power_of_two())
            .map(|idx| test_scalar(11 + idx as u128))
            .collect(),
    );
    let groups = vec![group];
    let address_bits = crate::RelationAddressGeometry::new(
        CommitmentRingDims::uniform(TEST_D),
        TEST_D,
        opening_source_len,
    )
    .unwrap()
    .relation_lane_variable_count();
    let full_vec_randomness = (0..address_bits)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let plan = prepare_test_plan(
        &inputs,
        &layout,
        opening_source_len,
        &groups,
        &full_vec_randomness,
        Some(&fold_gadget),
        CommitmentRingDims::uniform(TEST_D),
    )
    .unwrap();
    let setup_len = plan.required();
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_len * TEST_D,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan.evaluate_direct::<F>(&setup).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn packed_direct_matches_row_fallback_with_d_offset() {
    let plan = finalize_test_plan(
        2,
        5,
        vec![test_group_plan(
            2..4,
            4,
            3,
            2,
            2,
            vec![test_scalar(2), test_scalar(3)],
            vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            vec![test_scalar(29), test_scalar(31)],
            vec![test_scalar(37), test_scalar(41)],
        )],
        CommitmentRingDims::uniform(TEST_D),
    );
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_len * TEST_D,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan.evaluate_direct::<F>(&setup).unwrap();
    assert_eq!(got, expected);
}
#[test]
fn multi_group_packed_direct_matches_row_fallback() {
    let plan = finalize_test_plan(
        2,
        5,
        vec![
            test_group_plan(
                2..4,
                4,
                3,
                2,
                2,
                vec![test_scalar(2), test_scalar(3)],
                vec![
                    test_scalar(5),
                    test_scalar(7),
                    test_scalar(11),
                    test_scalar(13),
                ],
                vec![test_scalar(17), test_scalar(19), test_scalar(23)],
                vec![test_scalar(29), test_scalar(31)],
                vec![test_scalar(37), test_scalar(41)],
            ),
            test_group_plan(
                0..2,
                4,
                3,
                2,
                2,
                vec![test_scalar(53), test_scalar(59)],
                vec![
                    test_scalar(61),
                    test_scalar(67),
                    test_scalar(71),
                    test_scalar(73),
                ],
                vec![test_scalar(79), test_scalar(83), test_scalar(89)],
                vec![test_scalar(97), test_scalar(101)],
                vec![test_scalar(103), test_scalar(107)],
            ),
        ],
        CommitmentRingDims::uniform(TEST_D),
    );
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_len * TEST_D,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan.evaluate_direct::<F>(&setup).unwrap();
    assert_eq!(got, expected);
}
#[test]
fn packed_direct_matches_row_fallback_with_nested_role_dims() {
    const D: usize = 128;
    const D_B: usize = 64;
    const D_D: usize = 64;
    let plan = finalize_test_plan(
        2,
        5,
        vec![test_group_plan(
            2..4,
            4,
            3,
            2,
            2,
            vec![test_scalar(2), test_scalar(3)],
            vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            vec![test_scalar(29), test_scalar(31)],
            vec![test_scalar(37), test_scalar(41)],
        )],
        CommitmentRingDims {
            inner: D,
            outer: D_B,
            opening: D_D,
        },
    );
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_len * D,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
        ),
    );
    let alpha = test_scalar(3);
    let alpha_pows_a = scalar_powers(alpha, D);
    let alpha_pows_b = scalar_powers(alpha, D_B);
    let alpha_pows_d = scalar_powers(alpha, D_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d, D)
        .unwrap();
    let got = plan.evaluate_direct::<F>(&setup).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn packed_direct_accepts_d_footprint_at_nested_d_d() {
    // D-role columns are counted at d_d; comparing `required` against
    // total_ring_elements_at_dyn(d_a) falsely rejects valid setups when
    // d_d < d_a and the D footprint dominates.
    const D_A: usize = 128;
    const D_B: usize = 128;
    const D_D: usize = 64;
    let plan = finalize_test_plan(
        2,
        11,
        vec![test_group_plan(
            0..2,
            4,
            3,
            2,
            2,
            vec![test_scalar(2), test_scalar(3)],
            vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            vec![test_scalar(29), test_scalar(31)],
            vec![test_scalar(37), test_scalar(41)],
        )],
        CommitmentRingDims {
            inner: D_A,
            outer: D_B,
            opening: D_D,
        },
    );
    let setup_ring_elements = 20usize;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_ring_elements * D_A,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_ring_elements * D_A)
                .map(|idx| test_scalar(311 + idx as u128))
                .collect(),
        ),
    );
    let alpha = test_scalar(3);
    let alpha_pows_a = scalar_powers(alpha, D_A);
    let alpha_pows_b = scalar_powers(alpha, D_B);
    let alpha_pows_d = scalar_powers(alpha, D_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d, D_A)
        .unwrap();
    let got = plan.evaluate_direct::<F>(&setup).unwrap();
    assert_eq!(got, expected);
}
