use super::plan::SetupContributionGroupPlan;
use super::weights::setup_z_col_weights;
use super::*;
use crate::{
    gadget_row_scalars, AkitaExpandedSetup, AkitaSetupSeed, CommitmentRingDims, FlatMatrix,
    LevelParams, OpeningClaimsLayout, RelationMatrixRowLayout, SetupIndexWeightEvaluator,
    WitnessLayout, WitnessUnitLayout,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_challenges::SparseChallengeConfig;
use akita_field::Prime128OffsetA7F7;

mod prepare;

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
    level_params: LevelParams,
    opening_batch: OpeningClaimsLayout,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    eq_tau1: std::sync::Arc<[F]>,
}
impl TestSetupInputs {
    fn n_a(&self) -> usize {
        self.level_params.a_key.row_len()
    }
    fn num_claims(&self) -> usize {
        self.opening_batch.num_total_polynomials()
    }
    fn num_live_blocks(&self) -> usize {
        self.level_params.num_live_blocks
    }
    fn num_positions_per_block(&self) -> usize {
        self.level_params.num_positions_per_block
    }
    fn depth_open(&self) -> usize {
        self.level_params.num_digits_open
    }
    fn depth_commit(&self) -> usize {
        self.level_params.num_digits_inner
    }
    fn depth_fold(&self) -> Result<usize, AkitaError> {
        self.level_params.num_digits_fold(
            self.opening_batch.num_total_polynomials(),
            self.level_params.field_bits_for_cache(),
        )
    }
}
fn test_scalar(value: u128) -> F {
    F::from_canonical_u128(value)
}
#[allow(clippy::too_many_arguments)]
fn test_inputs(
    relation_matrix_row_layout: RelationMatrixRowLayout,
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
        relation_matrix_row_layout,
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
    relation_matrix_row_layout: RelationMatrixRowLayout,
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
    eq_tau1: Vec<F>,
) -> TestSetupInputs {
    let num_claims: usize = group_sizes.iter().copied().sum();
    let mut lp = LevelParams::params_only(
        crate::sis::SisModulusProfileId::Q128OffsetA7F7,
        TEST_D,
        log_basis,
        n_a,
        n_b,
        n_d,
        SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(
        num_positions_per_block,
        num_live_blocks * num_positions_per_block,
        depth_commit,
        depth_open,
        depth_open,
    )
    .expect("test level params");
    let expected_b_width = num_claims
        .checked_mul(n_a)
        .and_then(|width| width.checked_mul(depth_open))
        .and_then(|width| width.checked_mul(num_live_blocks))
        .expect("test B width");
    if lp.b_key.col_len() < expected_b_width {
        lp.b_key = crate::AjtaiKeyParams::new_unchecked(
            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            crate::sis::SisModulusProfileId::Q128OffsetA7F7,
            crate::sis::SisMatrixRole::B,
            n_b,
            expected_b_width,
            1,
            TEST_D,
        );
    }
    if lp.a_key.coeff_linf_bound() == 0 {
        lp.a_key = crate::AjtaiKeyParams::new_unchecked(
            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            crate::sis::SisModulusProfileId::Q128OffsetA7F7,
            crate::sis::SisMatrixRole::A,
            n_a,
            lp.a_key.col_len(),
            1,
            TEST_D,
        );
    }
    lp.num_digits_fold_one = depth_fold;
    lp.cached_num_digits_block_claims = num_claims;
    lp.cached_num_digits_fold_value = depth_fold;
    if group_sizes.len() > 1 {
        lp.precommitted_groups = group_sizes[..group_sizes.len() - 1]
            .iter()
            .map(|&_group_size| {
                let layout = crate::PrecommittedGroupParams::from_params(
                    crate::PolynomialGroupLayout::new(0, 1),
                    &lp,
                );
                let expected_group_b_width = lp
                    .a_key
                    .row_len()
                    .checked_mul(lp.num_digits_outer)
                    .and_then(|width| width.checked_mul(layout.num_live_blocks))
                    .and_then(|width| width.checked_mul(layout.group.num_polynomials()))
                    .expect("test precommitted B width");
                let b_key = crate::AjtaiKeyParams::new_unchecked(
                    lp.b_key.security_policy(),
                    lp.b_key.sis_table_key().table_digest,
                    lp.b_key.sis_modulus_profile(),
                    lp.b_key.sis_table_key().role,
                    lp.b_key.row_len(),
                    expected_group_b_width,
                    lp.b_key.coeff_linf_bound(),
                    lp.ring_dimension,
                );
                crate::PrecommittedLevelParams {
                    layout,
                    a_key: lp.a_key.clone(),
                    b_key,
                    log_basis_open: lp.log_basis_open,
                    num_digits_inner: lp.num_digits_inner,
                    num_digits_outer: lp.num_digits_outer,
                    num_digits_open: lp.num_digits_open,
                    num_digits_fold_one: depth_fold,
                }
            })
            .collect();
    }
    let opening_batch =
        OpeningClaimsLayout::from_group_sizes(0, group_sizes).expect("test opening batch");
    TestSetupInputs {
        level_params: lp,
        opening_batch,
        relation_matrix_row_layout,
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
    let mut global_block_start = 0usize;
    let base = num_live_blocks / num_chunks;
    let extra = num_live_blocks % num_chunks;
    let mut units = Vec::with_capacity(num_chunks);
    for chunk_index in 0..num_chunks {
        let chunk_num_live_blocks = base + usize::from(chunk_index < extra);
        let z_len = num_positions_per_block * depth_commit * depth_fold;
        let z_range = cursor..cursor + z_len;
        let e_range = z_range.end..z_range.end + num_claims * chunk_num_live_blocks * depth_open;
        let t_range =
            e_range.end..e_range.end + num_claims * chunk_num_live_blocks * n_a * depth_open;
        cursor = t_range.end;
        units.push(WitnessUnitLayout::new_for_test(
            0,
            chunk_index,
            global_block_start,
            chunk_num_live_blocks,
            z_range,
            e_range,
            t_range,
        ));
        global_block_start += chunk_num_live_blocks;
    }
    WitnessLayout::new_for_test(units, cursor..cursor + relation_rows * quotient_depth)
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
    SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        inputs.relation_matrix_row_layout,
        inputs.eq_tau1.clone(),
        witness_layout,
        opening_source_len,
        groups,
        full_vec_randomness,
        fold_gadget,
        role_dims,
    )
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
        d_weights: (0..d_rows)
            .map(|idx| test_scalar(43 + 4 * idx as u128))
            .collect::<Vec<_>>()
            .into(),
        projection_geometry,
    };
    for group in &mut plan.groups {
        group
            .refresh_segments(&plan.d_weights, plan.d_rows, plan.d_physical_cols)
            .expect("valid cached setup scan segments");
    }
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
) -> SetupContributionGroupPlan<F> {
    SetupContributionGroupPlan {
        d_col_range,
        t_cols,
        z_cols,
        n_a,
        n_b,
        required: 0,
        segments: Vec::new().into(),
        a_row_weights: a_row_weights.into(),
        b_weights: b_weights.into(),
        e_eq_slice,
        t_eq_slice,
        z_eq_slice,
    }
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
        layout.total_len(),
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
    let a_range = inputs.level_params.a_row_range(
        &inputs.opening_batch,
        *group_index,
        inputs.relation_matrix_row_layout,
    )?;
    let b_range = inputs.level_params.commitment_row_range(
        &inputs.opening_batch,
        *group_index,
        inputs.relation_matrix_row_layout,
    )?;
    Ok(SetupContributionGroupInputs {
        group_id: *group_index,
        num_claims,
        depth_fold: inputs.level_params.num_digits_fold_for_params(
            group_lp,
            num_claims,
            inputs.level_params.field_bits_for_cache(),
        )?,
        a_row_start: a_range.start,
        b_row_start: b_range.start,
    })
}
fn structured_weight_fixture(
    num_live_blocks: usize,
    ownership_widths: &[usize],
    role_dims: CommitmentRingDims,
) -> StructuredWeightFixture {
    let num_claims = 2;
    let depth_open = 2;
    let depth_commit = 2;
    let depth_fold = 2;
    let num_positions_per_block = 8;
    let n_a = 2;
    let n_b = 2;
    let n_d = 2;
    let log_basis = 4;
    assert_eq!(ownership_widths.iter().sum::<usize>(), num_live_blocks);
    let z_len = num_positions_per_block * depth_commit * depth_fold;
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
            let unit = WitnessUnitLayout::new_for_test(
                0,
                chunk,
                global_block_base,
                blocks,
                z_range,
                e_range,
                t_range,
            );
            global_block_base += blocks;
            unit
        })
        .collect::<Vec<_>>();
    let layout = WitnessLayout::new_for_test(ownership_units, cursor..cursor + n_d * depth_fold);
    let tau1 = (0..3)
        .map(|idx| test_scalar(31 + idx as u128))
        .collect::<Vec<_>>();
    let inputs = test_inputs(
        RelationMatrixRowLayout::WithDBlock,
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
    let full_vec_randomness = (0..18)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let opening_source_len = layout.total_len();
    let groups = vec![SetupContributionGroupInputs {
        group_id: 0,
        num_claims,
        depth_fold,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    }];
    let plan = prepare_test_plan(
        &inputs,
        &layout,
        opening_source_len,
        &groups,
        &full_vec_randomness,
        Some(&fold_gadget),
        role_dims,
    )
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
                        + fold_digit
                        + depth_fold * (commit_digit + depth_commit * position);
                    let opening_address =
                        crate::checked_opening_source_index(opening_source_len, physical).unwrap();
                    weight -= eq_eval_at_index(full_vec_randomness, opening_address) * fold;
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
#[test]
fn relation_ordered_setup_layout_matches_structured_direct_and_dense_oracles() {
    let rows = 6;
    let quotient_depth = 2;
    let group_shapes = [
        // Relation order deliberately differs from numeric group order.
        (1usize, 1usize, 1usize, 1usize, 1usize),
        (0usize, 1usize, 1usize, 1usize, 1usize),
    ];
    let tau1 = vec![test_scalar(31), test_scalar(32), test_scalar(33)];
    let inputs = test_inputs_for_group_sizes(
        RelationMatrixRowLayout::WithDBlock,
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
    let mut cursor = 0usize;
    let units = group_shapes
        .iter()
        .map(
            |&(group_id, num_claims, num_live_blocks, depth_open, depth_commit)| {
                let z_len = 2 * depth_commit * quotient_depth;
                let z_range = cursor..cursor + z_len;
                let e_range = z_range.end..z_range.end + num_claims * num_live_blocks * depth_open;
                let t_range = e_range.end..e_range.end + num_claims * num_live_blocks * depth_open;
                cursor = t_range.end;
                WitnessUnitLayout::new_for_test(
                    group_id,
                    0,
                    0,
                    num_live_blocks,
                    z_range,
                    e_range,
                    t_range,
                )
            },
        )
        .collect();
    let witness_layout = WitnessLayout::new_for_test(units, cursor..cursor + rows * quotient_depth);
    let opening_source_len = witness_layout.total_len();
    let groups: Vec<_> = group_shapes
        .iter()
        .map(
            |&(group_id, num_claims, _num_live_blocks, _depth_open, _depth_commit)| {
                let a_range = inputs
                    .level_params
                    .a_row_range(
                        &inputs.opening_batch,
                        group_id,
                        inputs.relation_matrix_row_layout,
                    )
                    .unwrap();
                let b_range = inputs
                    .level_params
                    .commitment_row_range(
                        &inputs.opening_batch,
                        group_id,
                        inputs.relation_matrix_row_layout,
                    )
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
        inputs.relation_matrix_row_layout,
        &witness_layout,
        &groups,
    )
    .unwrap();
    assert_eq!(
        get_d_col_range(&inputs.level_params, &inputs.opening_batch, &groups, 1).unwrap(),
        0..1
    );
    assert_eq!(
        get_d_col_range(&inputs.level_params, &inputs.opening_batch, &groups, 0).unwrap(),
        1..2
    );
    let randomness_bits = crate::opening_domain_len(opening_source_len)
        .unwrap()
        .trailing_zeros() as usize;
    let full_vec_randomness = (0..randomness_bits)
        .map(|index| test_scalar(101 + index as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(quotient_depth, 4);
    let plan = prepare_test_plan(
        &inputs,
        &witness_layout,
        opening_source_len,
        &groups,
        &full_vec_randomness,
        Some(&fold_gadget),
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
                .map(|index| test_scalar(211 + index as u128))
                .collect(),
            TEST_D,
        ),
    );
    let alpha = test_scalar(3);
    let alpha_pows = scalar_powers(alpha, TEST_D);
    assert_eq!(
        plan.evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
            .unwrap(),
        plan.evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D,)
            .unwrap(),
    );
    let evaluator = SetupIndexWeightEvaluator::new::<F>(
        &plan,
        &inputs.level_params,
        &inputs.opening_batch,
        inputs.relation_matrix_row_layout,
        &witness_layout,
        opening_source_len,
        &groups,
        &tau1,
        &full_vec_randomness,
        &fold_gadget,
        alpha,
    )
    .unwrap();
    let rho = rho_for_required(plan.required());
    assert_eq!(
        evaluator.evaluate(&rho).unwrap(),
        plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap(),
    );
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
                if group.d_col_range.contains(&d_col) {
                    weight += d_scales[base_idx % d_ratio]
                        * plan.d_weights[d_row]
                        * group.e_eq_slice[d_col - group.d_col_range.start];
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
    let (inputs, groups, witness_layout, plan, tau1, full_vec_randomness, fold_gadget) =
        structured_weight_fixture(8, &[8], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let evaluator = SetupIndexWeightEvaluator::new::<F>(
        &plan,
        &inputs.level_params,
        &inputs.opening_batch,
        inputs.relation_matrix_row_layout,
        &witness_layout,
        witness_layout.total_len(),
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
    let (inputs, groups, witness_layout, plan, tau1, full_vec_randomness, fold_gadget) =
        structured_weight_fixture(8, &[2, 2, 2, 2], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let evaluator = SetupIndexWeightEvaluator::new::<F>(
        &plan,
        &inputs.level_params,
        &inputs.opening_batch,
        inputs.relation_matrix_row_layout,
        &witness_layout,
        witness_layout.total_len(),
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
    let (inputs, groups, witness_layout, plan, tau1, full_vec_randomness, fold_gadget) =
        structured_weight_fixture(8, &[3, 5], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let evaluator = SetupIndexWeightEvaluator::new::<F>(
        &plan,
        &inputs.level_params,
        &inputs.opening_batch,
        inputs.relation_matrix_row_layout,
        &witness_layout,
        witness_layout.total_len(),
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
        let (inputs, groups, witness_layout, plan, tau1, full_vec_randomness, fold_gadget) =
            structured_weight_fixture(8, ownership_widths, role_dims);
        let evaluator = SetupIndexWeightEvaluator::new::<F>(
            &plan,
            &inputs.level_params,
            &inputs.opening_batch,
            inputs.relation_matrix_row_layout,
            &witness_layout,
            witness_layout.total_len(),
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
    let num_positions_per_block = 16;
    let depth_commit = 3;
    let depth_fold = 2;
    let full_vec_randomness = (0..9)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
    let inputs = test_inputs(
        RelationMatrixRowLayout::WithoutDBlock,
        1,
        0,
        0,
        1,
        4,
        num_positions_per_block,
        16,
        depth_commit,
        depth_fold,
        4,
        vec![test_scalar(11), test_scalar(12)],
    );
    let layout = test_witness_layout(
        inputs.num_claims(),
        inputs.num_live_blocks(),
        inputs.num_positions_per_block(),
        inputs.depth_open(),
        inputs.depth_commit(),
        inputs.depth_fold().unwrap(),
        inputs.n_a(),
        1,
        1,
        inputs.depth_fold().unwrap(),
    );
    let plan =
        prepare_single_group_plan(&inputs, &full_vec_randomness, &fold_gadget, &layout).unwrap();
    let expected = expected_z_setup_weights(
        &layout,
        layout.total_len(),
        0,
        num_positions_per_block,
        depth_commit,
        &fold_gadget,
        &full_vec_randomness,
    );
    assert_eq!(plan.groups[0].z_eq_slice, expected);
}
#[test]
fn setup_a_z_weights_do_not_include_commit_gadget() {
    let num_positions_per_block = 8;
    let depth_commit = 3;
    let depth_fold = 2;
    let log_basis = 4;
    let full_vec_randomness = (0..8)
        .map(|idx| test_scalar(701 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let commit_gadget = gadget_row_scalars::<F>(depth_commit, log_basis);
    let inputs = test_inputs(
        RelationMatrixRowLayout::WithoutDBlock,
        1,
        0,
        0,
        1,
        4,
        num_positions_per_block,
        16,
        depth_commit,
        depth_fold,
        log_basis,
        vec![test_scalar(11), test_scalar(12)],
    );
    let layout = test_witness_layout(
        inputs.num_claims(),
        inputs.num_live_blocks(),
        inputs.num_positions_per_block(),
        inputs.depth_open(),
        inputs.depth_commit(),
        inputs.depth_fold().unwrap(),
        inputs.n_a(),
        1,
        1,
        inputs.depth_fold().unwrap(),
    );
    let plan =
        prepare_single_group_plan(&inputs, &full_vec_randomness, &fold_gadget, &layout).unwrap();
    let expected = expected_z_setup_weights(
        &layout,
        layout.total_len(),
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
    assert_eq!(plan.groups[0].z_eq_slice, expected);
    assert_ne!(
        plan.groups[0].z_eq_slice, wrong_with_commit_gadget,
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
    let opening_source_len = layout.total_len();
    let point = (0..crate::opening_domain_len(opening_source_len)
        .unwrap()
        .trailing_zeros() as usize)
        .map(|index| test_scalar(1201 + index as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
    let mut got = vec![F::zero(); num_positions_per_block * depth_commit];
    let eq_window = akita_algebra::offset_eq::OffsetEqWindow::new(&point).unwrap();
    setup_z_col_weights(
        &layout,
        opening_source_len,
        group_id,
        num_positions_per_block,
        depth_commit,
        depth_fold,
        &eq_window,
        &fold_gadget,
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
    let num_live_blocks = 4;
    let blocks_per_chunk = 2;
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
        num_live_blocks / blocks_per_chunk,
        n_d,
        depth_fold,
    );
    let opening_source_len = layout.total_len();
    let group = SetupContributionGroupInputs {
        group_id: 0,
        num_claims,
        depth_fold,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    };
    let inputs = test_inputs(
        RelationMatrixRowLayout::WithDBlock,
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
    let full_vec_randomness = (0..10)
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
                6,
                3,
                2,
                2,
                vec![test_scalar(53), test_scalar(59)],
                vec![
                    test_scalar(61),
                    test_scalar(67),
                    test_scalar(71),
                    test_scalar(73),
                    test_scalar(79),
                    test_scalar(83),
                ],
                vec![test_scalar(89), test_scalar(97), test_scalar(101)],
                vec![test_scalar(103), test_scalar(107)],
                vec![test_scalar(109), test_scalar(113)],
            ),
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
