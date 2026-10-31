use super::fused_scan::reduced_direct_literal_oracle;
use super::*;
use jolt_field::Ring;

fn check_shared_reduced_scan<E>(blocks: usize, scalar: impl Fn(u64) -> E, zero_rows: bool)
where
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let role_dims = CommitmentRingDims {
        inner: 64,
        outer: 64,
        opening: 64,
    };
    let (inputs, groups, layout, base_plan, _, base_relation_point, fold_gadget) =
        structured_weight_fixture_with_outgoing(blocks, &[blocks], role_dims, 64);
    let geometry = base_plan.relation_address_geometry();
    let relation_point = base_relation_point
        .iter()
        .enumerate()
        .map(|(index, _)| scalar(601 + index as u64))
        .collect::<Vec<_>>();
    let coefficient_point = (0..geometry.relation_coefficient_variable_count())
        .map(|index| scalar(701 + index as u64))
        .collect::<Vec<_>>();
    let row_weights = inputs
        .eq_tau1
        .iter()
        .copied()
        .map(|weight| {
            if zero_rows {
                E::zero()
            } else {
                E::lift_base(weight)
            }
        })
        .collect::<Vec<_>>();
    let alpha = scalar(809);
    let plan = SetupContributionPlan::<E>::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        row_weights.clone().into(),
        &layout,
        &groups,
        PreparedRelationAddress::new(&relation_point).unwrap(),
        Some(&fold_gadget),
        geometry,
    )
    .unwrap();
    let scan = DirectScan::new(
        plan,
        PreparedCoefficientFunctional::reduced_evaluation(alpha, &coefficient_point, geometry)
            .unwrap(),
    )
    .unwrap();
    let plan = scan.plan();
    // Check that the fixture exercises the single functional, ratio-one case.
    let DirectScanMode::Reduced { groups: direct, .. } = &scan.mode else {
        panic!("expected reduced fixture");
    };
    assert_eq!(plan.projection_geometry().base_ring_dim(), 64);
    let first = &direct[0].roles[0].functional;
    assert_eq!(first.len(), 64);
    for group in direct {
        for role in &group.roles {
            assert_eq!(role.functional.as_ref(), first.as_ref());
        }
    }
    if blocks == 131 {
        // Multiple 2048-ring jobs and a partial last job, without padding away
        // the unequal ownership or physical-prefix boundaries.
        assert!(plan.required() > super::super::segments::SETUP_SCAN_JOB_RINGS);
        assert_ne!(
            plan.required() % super::super::segments::SETUP_SCAN_JOB_RINGS,
            0
        );
    }
    let setup_coefficients = plan.required() * 64;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_coefficients,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_coefficients)
                .map(|index| test_scalar(901 + index as u128))
                .collect(),
        ),
    );
    let expected = reduced_direct_literal_oracle(
        plan,
        &setup,
        &inputs,
        &groups,
        &layout,
        &fold_gadget,
        &row_weights,
        &coefficient_point,
        alpha,
    );
    assert_eq!(scan.evaluate_direct::<F>(&setup).unwrap(), expected);
    if zero_rows {
        assert_eq!(expected, E::zero());
    }

    // A truncated public setup is still rejected at the existing boundary.
    let truncated = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_coefficients - 64,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(vec![F::zero(); setup_coefficients - 64]),
    );
    assert!(scan.evaluate_direct::<F>(&truncated).is_err());
}

#[test]
fn shared_reduced_scan_matches_literal_oracle_across_jobs() {
    check_shared_reduced_scan(131, F::from_u64, false);
}

#[test]
fn shared_reduced_scan_handles_zero_weights_and_boolean_points() {
    check_shared_reduced_scan(5, |value| F::from_u64(value % 2), true);
    check_shared_reduced_scan(5, |value| F::from_u64(value % 2), false);
}

#[test]
fn shared_reduced_scan_matches_literal_oracle_over_extension_field() {
    type X = jolt_field::Ext2<F>;
    check_shared_reduced_scan(
        5,
        |value| X::from_base_slice(&[F::from_u64(value), F::from_u64(value + 10_000)]),
        false,
    );
}
