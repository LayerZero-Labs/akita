use super::weights::setup_z_col_weights;
use super::*;
use crate::{
    gadget_row_scalars, AkitaExpandedSetup, AkitaSetupSeed, CommitmentRingDims,
    CommittedGroupParams, FlatMatrix, OpeningClaimsLayout, WitnessLayout, WitnessUnitLayout,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, Prime128OffsetA7F7};

mod prepare;

type F = Prime128OffsetA7F7;
const TEST_D: usize = 64;
type StructuredWeightFixture = (
    TestSetupInputs,
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
        self.level_params.inner_commit_matrix.output_rank()
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
    eq_tau1: Vec<F>,
) -> TestSetupInputs {
    let num_claims: usize = group_sizes.iter().copied().sum();
    let mut lp = CommittedGroupParams::params_only(
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
    if lp.outer_commit_matrix.input_width() < expected_b_width {
        lp.outer_commit_matrix = crate::OuterCommitMatrixParams::new_unchecked(
            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            crate::sis::SisModulusProfileId::Q128OffsetA7F7,
            n_b,
            expected_b_width,
            1,
            TEST_D,
        );
    }
    if lp.inner_commit_matrix.coeff_linf_bound() == 0 {
        lp.inner_commit_matrix = crate::InnerCommitMatrixParams::new_unchecked(
            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            crate::sis::SisModulusProfileId::Q128OffsetA7F7,
            n_a,
            lp.inner_commit_matrix.input_width(),
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
                let layout = crate::PrecommittedGroupDescriptor::from_params(
                    crate::PolynomialGroupLayout::new(0, 1),
                    &lp,
                );
                let expected_group_b_width = lp
                    .inner_commit_matrix
                    .output_rank()
                    .checked_mul(lp.num_digits_outer)
                    .and_then(|width| width.checked_mul(layout.num_live_blocks))
                    .and_then(|width| width.checked_mul(layout.group.num_polynomials()))
                    .expect("test precommitted B width");
                let outer_commit_matrix = crate::OuterCommitMatrixParams::new_unchecked(
                    lp.outer_commit_matrix.security_policy(),
                    lp.outer_commit_matrix.sis_table_key().table_digest,
                    lp.outer_commit_matrix.sis_modulus_profile(),
                    lp.outer_commit_matrix.output_rank(),
                    expected_group_b_width,
                    lp.outer_commit_matrix.coeff_linf_bound(),
                    lp.d_a(),
                );
                crate::PrecommittedLevelParams {
                    layout,
                    inner_commit_matrix: lp.inner_commit_matrix.clone(),
                    outer_commit_matrix,
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
    full_vec_randomness: &[F],
    role_dims: CommitmentRingDims,
) -> Result<SetupContributionPlan<F>, AkitaError> {
    SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        inputs.eq_tau1.clone(),
        witness_layout,
        full_vec_randomness,
        role_dims,
        role_dims.d_a(),
    )
}
fn prepare_single_group_plan(
    inputs: &TestSetupInputs,
    full_vec_randomness: &[F],
    layout: &WitnessLayout,
) -> Result<SetupContributionPlan<F>, AkitaError> {
    prepare_test_plan(
        inputs,
        layout,
        full_vec_randomness,
        CommitmentRingDims::uniform(TEST_D),
    )
}
fn structured_weight_fixture(
    num_live_blocks: usize,
    ownership_widths: &[usize],
    role_dims: CommitmentRingDims,
) -> StructuredWeightFixture {
    let num_claims = 2;
    let depth_open = 2;
    let depth_commit = 2;
    let depth_fold = 3;
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
    let full_vec_randomness = (0..relation_address_bits(&layout, role_dims, role_dims.d_a()))
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let plan = prepare_test_plan(&inputs, &layout, &full_vec_randomness, role_dims).unwrap();
    (inputs, layout, plan, tau1, full_vec_randomness, fold_gadget)
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

#[test]
fn structured_span_contraction_matches_dense_oracle_for_uniform_and_mixed_roles() {
    for role_dims in [
        CommitmentRingDims::uniform(TEST_D),
        CommitmentRingDims {
            inner: TEST_D,
            outer: TEST_D / 2,
            opening: TEST_D / 2,
        },
    ] {
        let (inputs, layout, plan, _, x_challenges, fold_gadget) =
            structured_weight_fixture(8, &[3, 5], role_dims);
        let group_id = 0;
        let group_params = inputs
            .level_params
            .group_params(&inputs.opening_batch, group_id)
            .unwrap();
        let num_claims = inputs
            .opening_batch
            .group_layout(group_id)
            .unwrap()
            .num_polynomials();
        let num_live_blocks = group_params.num_live_blocks();
        let depth_open = group_params.num_digits_open();
        let depth_commit = group_params.num_digits_outer();
        let depth_witness = group_params.num_digits_inner();
        let depth_fold = inputs
            .level_params
            .num_digits_fold_for_params(
                group_params,
                num_claims,
                inputs.level_params.field_bits_for_cache(),
            )
            .unwrap();
        let num_positions = group_params.num_positions_per_block();
        let block_challenges = (0..num_claims * num_live_blocks)
            .map(|index| test_scalar(401 + index as u128))
            .collect::<Vec<_>>();
        let opening_a_evals = (0..num_positions)
            .map(|index| test_scalar(601 + index as u128))
            .collect::<Vec<_>>();
        let alpha = test_scalar(3);
        let common_coeff_count = role_dims.common_relation_witness_coeff_count(role_dims.d_a());
        let inner_lane_powers = scalar_powers(alpha, role_dims.d_a())
            .into_iter()
            .step_by(common_coeff_count)
            .collect::<Vec<_>>();
        let inner_lane_count = inner_lane_powers.len();
        let eq_window = akita_algebra::offset_eq::OffsetEqWindow::new(&x_challenges).unwrap();
        let lane_weight = |witness_column: usize| {
            inner_lane_powers
                .iter()
                .copied()
                .enumerate()
                .fold(F::zero(), |sum, (lane, power)| {
                    sum + eq_window.eval(witness_column * inner_lane_count + lane) * power
                })
        };
        let consistency = inputs.eq_tau1[0];
        let opening_gadget = gadget_row_scalars::<F>(depth_open, group_params.log_basis_open());
        let commitment_gadget =
            gadget_row_scalars::<F>(depth_commit, group_params.log_basis_outer());
        let witness_gadget = gadget_row_scalars::<F>(depth_witness, group_params.log_basis_inner());
        let a_rows = inputs
            .level_params
            .a_row_range(&inputs.opening_batch, group_id)
            .unwrap();
        let mut expected = F::zero();
        for unit in layout.units_for_group(group_id).unwrap() {
            for claim in 0..num_claims {
                for local_block in 0..unit.num_live_blocks() {
                    let block = unit.global_block_start() + local_block;
                    let challenge = block_challenges[claim * num_live_blocks + block];
                    for (digit, &gadget) in opening_gadget.iter().enumerate() {
                        let witness = unit
                            .e_index(num_claims, depth_open, claim, block, digit)
                            .unwrap();
                        expected += challenge * consistency * lane_weight(witness) * gadget;
                    }
                    for a_row in 0..group_params.a_rows_len() {
                        let row_weight = inputs.eq_tau1[a_rows.start + a_row];
                        for (digit, &gadget) in commitment_gadget.iter().enumerate() {
                            let witness = unit
                                .t_index(
                                    num_claims,
                                    group_params.a_rows_len(),
                                    depth_commit,
                                    claim,
                                    block,
                                    a_row,
                                    digit,
                                )
                                .unwrap();
                            expected += challenge * row_weight * lane_weight(witness) * gadget;
                        }
                    }
                }
            }
            for (position, &opening) in opening_a_evals.iter().enumerate() {
                for (witness_digit, &witness_gadget) in witness_gadget.iter().enumerate() {
                    for (fold_digit, &fold) in fold_gadget.iter().take(depth_fold).enumerate() {
                        let witness = unit
                            .z_index(
                                num_positions,
                                depth_witness,
                                depth_fold,
                                position,
                                witness_digit,
                                fold_digit,
                            )
                            .unwrap();
                        expected -=
                            lane_weight(witness) * consistency * opening * witness_gadget * fold;
                    }
                }
            }
        }
        let got = plan
            .evaluate_structured_group::<F>(group_id, &block_challenges, &opening_a_evals, alpha)
            .unwrap();
        assert_eq!(got, expected, "role dimensions {role_dims:?}");
    }
}

#[test]
fn mixed_role_direct_scan_matches_materialized_span_weights() {
    const BASE_D: usize = TEST_D / 2;
    let role_dims = CommitmentRingDims {
        inner: TEST_D,
        outer: BASE_D,
        opening: BASE_D,
    };
    let (_, _, plan, _, _, _) = structured_weight_fixture(8, &[3, 5], role_dims);
    let setup_ring_elements = plan.required().div_ceil(TEST_D / BASE_D);
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: TEST_D,
            max_setup_len: setup_ring_elements,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_ring_elements * TEST_D)
                .map(|index| test_scalar(801 + index as u128))
                .collect(),
            TEST_D,
        ),
    );
    let alpha = test_scalar(3);
    let weights = plan.materialize_setup_index_weights(alpha).unwrap();
    let setup_view = setup
        .shared_matrix()
        .ring_view::<BASE_D>(1, plan.required())
        .unwrap();
    let base_powers = scalar_powers(alpha, BASE_D);
    let expected = setup_view
        .as_slice()
        .iter()
        .zip(weights)
        .fold(F::zero(), |sum, (ring, weight)| {
            sum + eval_ring_at_pows(ring, &base_powers) * weight
        });
    assert_eq!(plan.evaluate_direct::<F>(&setup, alpha).unwrap(), expected);
}

fn rho_for_required(required: usize) -> Vec<F> {
    let bits = required.next_power_of_two().trailing_zeros() as usize;
    (0..bits)
        .map(|idx| test_scalar(901 + idx as u128))
        .collect()
}

fn relation_address_bits(
    layout: &WitnessLayout,
    role_dims: CommitmentRingDims,
    outgoing_ring_dim: usize,
) -> usize {
    let coeff_count = role_dims.common_relation_witness_coeff_count(outgoing_ring_dim);
    (crate::opening_domain_len(layout.total_len()).unwrap() * outgoing_ring_dim / coeff_count)
        .trailing_zeros() as usize
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
    let groups = group_shapes
        .iter()
        .map(|&(group_id, ..)| group_id)
        .collect::<Vec<_>>();
    validate_setup_inputs(&inputs.level_params, &inputs.opening_batch, &witness_layout).unwrap();
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
    let plan = prepare_test_plan(
        &inputs,
        &witness_layout,
        &full_vec_randomness,
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
        plan.evaluate_direct::<F>(&setup, alpha).unwrap(),
        plan.evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D,)
            .unwrap(),
    );
    let rho = rho_for_required(plan.required());
    let expected = plan
        .materialize_setup_index_weights(alpha)
        .unwrap()
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (idx, &weight)| {
            acc + eq_eval_at_index(&rho, idx) * weight
        });
    assert_eq!(
        expected,
        plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap(),
    );
}
#[test]
fn setup_index_weight_point_contraction_matches_materialization_single_chunk() {
    let (_, _, plan, _, _, _) =
        structured_weight_fixture(8, &[8], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let rho = rho_for_required(plan.required());
    let got = plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap();
    let dense = plan.materialize_setup_index_weights(alpha).unwrap();
    let expected = dense
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (idx, &weight)| {
            acc + eq_eval_at_index(&rho, idx) * weight
        });
    assert_eq!(got, expected);
}
#[test]
fn setup_index_weight_point_contraction_matches_materialization_multi_chunk() {
    let (_, _, plan, _, _, _) =
        structured_weight_fixture(8, &[2, 2, 2, 2], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let rho = rho_for_required(plan.required());
    let expected = plan
        .materialize_setup_index_weights(alpha)
        .unwrap()
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (idx, &weight)| {
            acc + eq_eval_at_index(&rho, idx) * weight
        });
    assert_eq!(
        expected,
        plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap()
    );
}
#[test]
fn setup_index_weight_point_contraction_supports_non_power_of_two_ownership_widths() {
    let (_, _, plan, _, _, _) =
        structured_weight_fixture(8, &[3, 5], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let rho = rho_for_required(plan.required());
    let expected = plan
        .materialize_setup_index_weights(alpha)
        .unwrap()
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (idx, &weight)| {
            acc + eq_eval_at_index(&rho, idx) * weight
        });
    assert_eq!(
        expected,
        plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap()
    );
}
#[test]
fn setup_index_weight_point_contraction_applies_mixed_role_projection_lanes() {
    let alpha = test_scalar(3);
    let role_dims = crate::CommitmentRingDims {
        inner: 64,
        outer: 32,
        opening: 32,
    };
    for ownership_widths in [&[8][..], &[2, 2, 2, 2][..], &[3, 5][..]] {
        let (_, _, plan, _, _, _) = structured_weight_fixture(8, ownership_widths, role_dims);
        let rho = rho_for_required(plan.required());
        let got = plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap();
        let expected = plan
            .materialize_setup_index_weights(alpha)
            .unwrap()
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (idx, &weight)| {
                acc + eq_eval_at_index(&rho, idx) * weight
            });
        assert_eq!(got, expected, "ownership widths {ownership_widths:?}");
    }
}
#[test]
fn dense_z_eq_slice_uses_relative_high_carry() {
    let num_positions_per_block = 16;
    let depth_commit = 3;
    let depth_fold = 2;
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
    let inputs = test_inputs(
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
    let role_dims = CommitmentRingDims::uniform(TEST_D);
    let full_vec_randomness = (0..relation_address_bits(&layout, role_dims, TEST_D))
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let plan = prepare_single_group_plan(&inputs, &full_vec_randomness, &layout).unwrap();
    let expected = expected_z_setup_weights(
        &layout,
        layout.total_len(),
        0,
        num_positions_per_block,
        depth_commit,
        &fold_gadget,
        &full_vec_randomness,
    );
    assert_eq!(
        plan.group_column_eq_slices_for_test(0, test_scalar(3))
            .unwrap()
            .2,
        expected.as_slice()
    );
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
    let role_dims = CommitmentRingDims::uniform(TEST_D);
    let full_vec_randomness = (0..relation_address_bits(&layout, role_dims, TEST_D))
        .map(|idx| test_scalar(701 + idx as u128))
        .collect::<Vec<_>>();
    let plan = prepare_single_group_plan(&inputs, &full_vec_randomness, &layout).unwrap();
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
    let z_eq_slice = plan
        .group_column_eq_slices_for_test(0, test_scalar(3))
        .unwrap()
        .2;
    assert_eq!(z_eq_slice, expected.as_slice());
    assert_ne!(
        z_eq_slice,
        wrong_with_commit_gadget.as_slice(),
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
    let group_params = inputs
        .level_params
        .group_params(&inputs.opening_batch, 0)
        .unwrap();
    let depth_fold = inputs
        .level_params
        .num_digits_fold_for_params(
            group_params,
            num_claims,
            inputs.level_params.field_bits_for_cache(),
        )
        .unwrap();
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
    let role_dims = CommitmentRingDims::uniform(TEST_D);
    let full_vec_randomness = (0..relation_address_bits(&layout, role_dims, TEST_D))
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let plan = prepare_test_plan(
        &inputs,
        &layout,
        &full_vec_randomness,
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
    let got = plan.evaluate_direct::<F>(&setup, test_scalar(3)).unwrap();
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
