use super::*;

#[test]
fn public_rhs_accepts_zero_packing_consistency_and_rejects_nonzero_payload() {
    let fixture = fixture::<F, E>(
        SisModulusProfileId::Q64Offset59,
        256,
        128,
        64,
        6,
        4,
        11,
        2,
        1,
    );
    let layout = fixture
        .relation_plan
        .relation_witness_geometry()
        .rhs_layout();
    assert_eq!(
        relation_claim_from_compressed_rhs_extension::<F, E>(
            layout,
            &fixture.tau1,
            E::from_u64(23),
            fixture.relation.rhs(),
        )
        .unwrap(),
        E::zero()
    );
    let mut malformed = fixture.relation.rhs().coeffs().to_vec();
    malformed[0] = F::one();
    assert!(relation_claim_from_compressed_rhs_extension::<F, E>(
        layout,
        &fixture.tau1,
        E::from_u64(23),
        &RingVec::from_coeffs(malformed),
    )
    .is_err());

    let alpha = E::from_u64(29);
    let families = layout.row_families().unwrap();
    let mut native_rhs = vec![F::zero(); relation_rhs_coeff_len(layout).unwrap()];
    let mut expected = E::zero();
    let mut offset = 0usize;
    for (row_index, family) in families.into_iter().enumerate() {
        let geometry = family.geometry();
        if matches!(
            family,
            RelationRowFamily::Outer { .. } | RelationRowFamily::Opening { .. }
        ) {
            let coefficient = 3.min(geometry.polynomial_modulus_dimension() - 1);
            let value = F::from_u64((row_index + 2) as u64);
            native_rhs[offset + coefficient] = value;
            expected += relation_row_weight(row_index, &fixture.tau1).unwrap()
                * scalar_powers(alpha, geometry.polynomial_modulus_dimension())[coefficient]
                * E::lift_base(value);
        }
        offset += geometry.physical_coefficient_width();
    }
    assert_eq!(
        relation_claim_from_compressed_rhs_extension::<F, E>(
            layout,
            &fixture.tau1,
            alpha,
            &RingVec::from_coeffs(native_rhs),
        )
        .unwrap(),
        expected
    );
}

#[test]
fn degree_one_packing_consistency_rhs_must_be_zero() {
    type Base = Prime128OffsetA7F7;
    let fixture = fixture::<Base, Base>(
        SisModulusProfileId::Q128OffsetA7F7,
        128,
        64,
        64,
        6,
        4,
        10,
        1,
        1,
    );
    let layout = fixture
        .relation_plan
        .relation_witness_geometry()
        .rhs_layout();
    let mut malformed = fixture.relation.rhs().coeffs().to_vec();
    malformed[0] = Base::one();

    assert!(relation_claim_from_compressed_rhs_extension::<Base, Base>(
        layout,
        &fixture.tau1,
        Base::from_u64(23),
        &RingVec::from_coeffs(malformed),
    )
    .is_err());
}

#[test]
fn alpha_and_role_dimensions_are_bound_by_the_shared_plan() {
    let fixture = fixture::<F, E>(
        SisModulusProfileId::Q64Offset59,
        256,
        128,
        64,
        6,
        4,
        11,
        2,
        1,
    );
    let at_zero = prepare(&fixture, E::zero());
    let at_one = prepare(&fixture, E::one());
    assert_eq!(
        at_zero.relation_events().alpha_powers(),
        scalar_powers(E::zero(), 64)
    );
    assert_eq!(
        at_one.relation_events().alpha_powers(),
        scalar_powers(E::one(), 64)
    );
    assert_ne!(
        materialize_events(at_zero.relation_events()),
        materialize_events(at_one.relation_events())
    );

    let wrong_relation = RingRelationInstance::new(
        fixture.relation.group_openings().to_vec(),
        fixture.relation.extension_degree(),
        fixture.opening_batch.clone(),
        fixture.relation.gamma().to_vec(),
        fixture.relation.row_coefficient_rings().clone(),
        fixture.relation.rhs().clone(),
        RingVec::from_coeffs(Vec::new()),
        CommitmentRingDims {
            inner: fixture.params.role_dims().d_a(),
            outer: fixture.params.role_dims().d_b(),
            opening: 64,
        },
    )
    .unwrap();
    assert!(
        prepare_coefficient_packing_group_semantics(CoefficientPackingGroupSemanticInputs {
            level_params: &fixture.params,
            opening_batch: &fixture.opening_batch,
            relation_plan: &fixture.relation_plan,
            relation: &wrong_relation,
            group_index: 0,
            prepared_point: &fixture.prepared_point,
            alpha: E::from_u64(3),
            tau1: &fixture.tau1,
            claim_coefficients: &fixture.claim_coefficients,
        })
        .is_err()
    );

    let mut unsupported = fixture.params.clone();
    let inner = unsupported.inner.matrix;
    unsupported.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().unwrap().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner.coeff_linf_bound().unwrap_or(0),
        1usize << 20,
    );
    assert!(
        prepare_coefficient_packing_group_semantics(CoefficientPackingGroupSemanticInputs {
            level_params: &unsupported,
            opening_batch: &fixture.opening_batch,
            relation_plan: &fixture.relation_plan,
            relation: &fixture.relation,
            group_index: 0,
            prepared_point: &fixture.prepared_point,
            alpha: E::from_u64(3),
            tau1: &fixture.tau1,
            claim_coefficients: &fixture.claim_coefficients,
        })
        .is_err()
    );
}
