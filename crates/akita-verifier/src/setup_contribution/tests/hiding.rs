use super::*;

const B_RANDOMNESS: usize = 3;
const D_RANDOMNESS: usize = 2;

/// Builds a relation point from its bit count and the witness layout.
type RelationPointFn<'a> = &'a dyn Fn(usize, &WitnessLayout) -> Vec<F>;

/// A multi-claim unsliced group with B and D hiding randomness, its witness
/// layout, and its setup plan at `relation_point` (or a generic point).
fn multi_claim_hiding_plan(
    relation_point: Option<RelationPointFn<'_>>,
) -> (SetupContributionPlan<F>, WitnessLayout) {
    let num_claims = 2;
    let num_live_blocks = 2;
    let depth_open = 2;
    let depth_commit = 2;
    let depth_fold = 2;
    let num_positions_per_block = 8;
    let n_a = 2;
    let n_b = 2;
    let n_d = 2;
    let log_basis = 4;
    let z_len = num_positions_per_block * depth_commit * depth_fold * TEST_D;
    let e_len = num_claims * depth_open * num_live_blocks * TEST_D;
    let t_len = n_a * num_claims * depth_commit * num_live_blocks * TEST_D;
    let unit = WitnessUnitLayout::new_for_test(
        0,
        0,
        0,
        num_live_blocks,
        0..z_len,
        z_len..z_len + e_len,
        akita_params::RelationRowGeometry::native(TEST_D).unwrap(),
        z_len + e_len..z_len + e_len + t_len,
    );
    let mut cursor = z_len + e_len + t_len;
    let r_rows = (0..n_d)
        .map(|_| {
            let range = cursor..cursor + depth_fold * TEST_D;
            cursor = range.end;
            WitnessQuotientRowLayout::new_for_test(
                akita_params::RelationRowGeometry::native(TEST_D).unwrap(),
                range,
            )
        })
        .collect();
    let outer_span = cursor..cursor + B_RANDOMNESS * TEST_D;
    let opening_span = outer_span.end..outer_span.end + D_RANDOMNESS * TEST_D;
    let layout = WitnessLayout::new_for_test(vec![unit], r_rows, depth_fold)
        .with_randomness_for_test(vec![(0, outer_span)], opening_span);

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
    let lp = &mut inputs.level_params;
    let geometry = akita_params::CommitmentSliceGeometry::try_new(
        akita_params::CommitmentSliceCount::ONE,
        num_live_blocks,
        num_claims,
        n_a,
        depth_commit,
        TEST_D,
        TEST_D,
    )
    .unwrap()
    .with_randomness_columns(B_RANDOMNESS)
    .unwrap();
    let outer = lp.outer().matrix;
    lp.own_group_mut().profile.randomness_columns = B_RANDOMNESS;
    lp.own_group_mut().profile.outer.matrix = akita_params::OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        geometry.physical_input_width(),
        outer.coeff_linf_bound(),
        TEST_D,
    );
    let open = lp.open().matrix;
    lp.open_randomness_columns = D_RANDOMNESS;
    lp.open_matrix = akita_params::OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        open.output_rank(),
        num_claims * num_live_blocks * depth_open + D_RANDOMNESS,
        open.coeff_linf_bound(),
        TEST_D,
    );
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let relation_address_geometry = akita_params::RelationAddressGeometry::new(
        CommitmentRingDims::uniform(TEST_D),
        TEST_D,
        layout.live_coeff_len(),
    )
    .unwrap();
    let bits = relation_address_geometry.relation_lane_variable_count();
    let point = relation_point.map_or_else(
        || {
            (0..bits)
                .map(|idx| test_scalar(101 + idx as u128))
                .collect::<Vec<_>>()
        },
        |point| point(bits, &layout),
    );
    let groups = vec![SetupContributionGroupInputs {
        group_id: 0,
        num_claims,
        depth_fold,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    }];
    let plan = SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        inputs.eq_tau1.clone(),
        &layout,
        &groups,
        PreparedRelationAddress::new(&point).unwrap(),
        Some(&fold_gadget),
        relation_address_geometry,
    )
    .unwrap();
    (plan, layout)
}

/// A multi-claim unsliced group with B and D hiding randomness: the Stage-3
/// setup-index MLE, built from compacted message tensors plus the randomness
/// tensors, matches the dense setup-index weights.
#[test]
fn multi_claim_hiding_randomness_setup_index_mle_matches_dense_weights() {
    let (plan, _) = multi_claim_hiding_plan(None);
    let group = &plan.groups()[0];
    assert_eq!(group.physical_b().randomness_tensors().len(), 1);
    assert_eq!(
        group.d_column_len(),
        group.d_col_range().len() + D_RANDOMNESS
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
        SetupIndexWeightMle::new(&plan)
            .unwrap()
            .evaluate(&rho_setup_idx, alpha)
            .unwrap(),
        dense_mle
    );
}

/// Shifting one witness digit by `delta` moves the relation residual by
/// `W(p) * delta`, where `W(p)` is that address's relation weight. At the
/// Boolean relation point of a randomness ring's lane, the setup contribution
/// is exactly that lane's setup weight, so a nonzero contribution means one
/// shifted randomness digit (B, and separately D) changes the residual.
#[test]
fn shifting_one_randomness_digit_changes_the_relation_residual() {
    let lane_point = |span_start: fn(&WitnessLayout) -> usize| {
        move |bits: usize, layout: &WitnessLayout| -> Vec<F> {
            let lane = span_start(layout) / TEST_D;
            let point = (0..bits)
                .map(|bit| {
                    if (lane >> bit) & 1 == 1 {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(eq_eval_at_index(&point, lane), F::one());
            point
        }
    };
    let b_point = lane_point(|layout| layout.randomness().outer(0).start);
    let d_point = lane_point(|layout| layout.randomness().opening().start);
    for (section, point) in [("B", &b_point as RelationPointFn<'_>), ("D", &d_point)] {
        let (plan, _) = multi_claim_hiding_plan(Some(point));
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
        let scan = lifted_test_scan(plan);
        let weight = scan.evaluate_direct::<F>(&setup).unwrap();
        let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
        assert_eq!(
            weight,
            scan.evaluate_direct_by_rows::<F>(
                &setup,
                &alpha_pows,
                &alpha_pows,
                &alpha_pows,
                TEST_D
            )
            .unwrap()
        );
        assert_ne!(
            weight,
            F::zero(),
            "{section} randomness carries no relation weight"
        );
    }
}

/// D randomness follows the opening columns of every group, preceding groups
/// (setup prefixes, precommitted groups) included, and fixes the D width.
#[test]
fn d_randomness_follows_every_group_and_fixes_the_d_width() {
    let plan_with_width = |extra: usize| {
        let HeterogeneousSetupFixture {
            mut inputs,
            groups,
            relation_point,
            fold_gadget,
            ..
        } = heterogeneous_setup_fixture();
        // The fixture's two groups take D columns 0..2 and 2..3.
        let lp = &mut inputs.level_params;
        let open = lp.open().matrix;
        lp.open_randomness_columns = D_RANDOMNESS;
        lp.open_matrix = akita_params::OpenCommitMatrixParams::new_unchecked(
            open.security_policy(),
            open.sis_table_key().table_digest,
            open.sis_modulus_profile(),
            open.output_rank(),
            3 + D_RANDOMNESS + extra,
            open.coeff_linf_bound(),
            open.ring_dimension(),
        );
        let joint_geometry = akita_params::RelationWitnessGeometry::for_evaluation_trace_execution(
            &inputs.level_params,
            &inputs.opening_batch,
        )
        .unwrap();
        let witness_layout = WitnessLayout::new(
            &inputs.level_params,
            &inputs.opening_batch,
            &joint_geometry,
            1,
            akita_params::RelationQuotientPlan::quotient_lift(2).unwrap(),
        )
        .unwrap();
        let relation_address_geometry = inputs
            .level_params
            .relation_address_geometry(
                &inputs.opening_batch,
                1,
                128,
                witness_layout.live_coeff_len(),
            )
            .unwrap();
        let point = (0..relation_address_geometry.relation_lane_variable_count())
            .map(|index| relation_point.get(index).copied().unwrap_or(test_scalar(7)))
            .collect::<Vec<_>>();
        SetupContributionPlan::prepare::<F>(
            &inputs.level_params,
            &inputs.opening_batch,
            1,
            inputs.eq_tau1.clone(),
            &witness_layout,
            &groups,
            PreparedRelationAddress::new(&point).unwrap(),
            Some(&fold_gadget),
            relation_address_geometry,
        )
    };
    let plan = plan_with_width(0).unwrap();
    assert_eq!(plan.d_physical_cols(), 3 + D_RANDOMNESS);
    let owner = plan
        .groups()
        .iter()
        .find(|group| group.d_col_range() == (2..3))
        .unwrap();
    assert_eq!(owner.d_column_len(), 1 + D_RANDOMNESS);
    assert!(plan
        .groups()
        .iter()
        .filter(|group| group.d_col_range() != (2..3))
        .all(|group| group.d_column_len() == group.d_col_range().len()));
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
        SetupIndexWeightMle::new(&plan)
            .unwrap()
            .evaluate(&rho_setup_idx, alpha)
            .unwrap(),
        dense_mle
    );
    assert!(plan_with_width(1).is_err());
}
