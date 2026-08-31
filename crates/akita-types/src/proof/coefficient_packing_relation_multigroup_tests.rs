use super::*;

#[test]
fn multi_group_semantics_follow_authenticated_root_order_and_claim_ranges() {
    let base = fixture::<F, E>(
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
    let mut params = base.params.clone();
    let mut frozen = params.with_decomp(4, 8, 2, 2, 2).unwrap();
    frozen.set_precommitted_groups(Vec::new()).unwrap();
    frozen.own_group_mut().profile.inner.digits.log_basis = 9;
    let a_bound = *crate::sis::inner_coeff_linf_bounds(
        frozen.inner().matrix.sis_modulus_profile(),
        u32::try_from(frozen.d_a()).expect("test ring dimension"),
    )
    .first()
    .expect("exact frozen A bounds");
    let inner = frozen.inner().matrix;
    frozen.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().unwrap().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        a_bound,
        inner.ring_dimension(),
    );
    let outer = frozen.outer().matrix;
    frozen.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        3,
        outer.ring_dimension(),
    );
    let pre_layout = PolynomialGroupLayout::new(11, 1);
    params
        .set_precommitted_groups(vec![GroupOpenPhaseParams {
            setup_natural_len: None,
            profile: GroupCommitPhaseParams::from_params_unchecked_for_test(pre_layout, &frozen),
            opening: GroupOpeningPlan {
                opening_method: OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension: 64,
                },
                fold_challenge_config: params.fold_challenge_config(),
                log_basis_open: params.open().digits.log_basis,
                num_digits_open: params.open().digits.num_digits,
                num_digits_fold: params.num_digits_fold(),
            },
        }])
        .unwrap();
    let final_layout = PolynomialGroupLayout::new(11, 2);
    let opening_batch = OpeningClaimsLayout::from_root_groups(&[pre_layout], final_layout).unwrap();
    let relation_geometry = RelationWitnessGeometry::for_level(&params, &opening_batch, 2).unwrap();
    let witness_layout = WitnessLayout::new(
        &params,
        &opening_batch,
        &relation_geometry,
        params.witness_chunk.num_chunks,
        crate::RelationQuotientPlan::quotient_lift(r_decomp_levels::<F>(
            params.open().digits.log_basis,
        ))
        .unwrap(),
    )
    .unwrap();
    let relation_address = RelationAddressGeometry::for_relation(
        &relation_geometry,
        params.role_dims().d_d(),
        witness_layout.live_coeff_len(),
    )
    .unwrap();
    let relation_plan = RelationRangeImagePlan::new(
        relation_geometry.clone(),
        relation_address,
        DigitRangePlan::new(4).unwrap(),
        witness_layout,
        &opening_batch,
    )
    .unwrap();
    assert_eq!(
        relation_plan
            .groups()
            .iter()
            .map(|group| group.group_index())
            .collect::<Vec<_>>(),
        vec![1, 0]
    );

    let config = params.fold_challenge_config();
    let make_challenges = |claims: usize| {
        Challenges::from_sparse(
            (0..claims * params.blocks().live_blocks)
                .map(|challenge| SparseChallenge {
                    positions: (0..config.weight())
                        .map(|term| ((term + challenge) % 64) as u32)
                        .collect(),
                    coeffs: (0..config.count_pm1)
                        .map(|_| 1)
                        .chain((0..config.count_pm2).map(|_| 2))
                        .collect(),
                })
                .collect(),
            params.blocks().live_blocks,
            claims,
        )
        .unwrap()
    };
    let geometry = SubringCoefficientPackingGeometry::try_new(2, 256, 64).unwrap();
    let openings = vec![
        RingRelationGroupOpening::coefficient_packing(
            CoefficientPackingChallenges::new(geometry, make_challenges(1)).unwrap(),
        ),
        RingRelationGroupOpening::coefficient_packing(
            CoefficientPackingChallenges::new(geometry, make_challenges(2)).unwrap(),
        ),
    ];
    let total_claims = opening_batch.num_total_polynomials();
    let gamma = (0..total_claims)
        .map(|claim| F::from_u64((claim + 2) as u64))
        .collect::<Vec<_>>();
    let mut row_coefficients = vec![F::zero(); total_claims * 256];
    for (claim, &coefficient) in gamma.iter().enumerate() {
        row_coefficients[claim * 256] = coefficient;
    }
    let relation = RingRelationInstance::new(
        openings,
        2,
        opening_batch.clone(),
        gamma,
        RingVec::from_coeffs_with_ring_dim(row_coefficients, 256).unwrap(),
        RingVec::from_coeffs(vec![
            F::zero();
            relation_rhs_coeff_len(relation_geometry.rhs_layout())
                .unwrap()
        ]),
        RingVec::from_coeffs(Vec::new()),
        params.role_dims(),
    )
    .unwrap();
    let claim_coefficients = (0..total_claims)
        .map(|claim| E::from_u64((claim + 11) as u64))
        .collect::<Vec<_>>();
    let tau1 = vec![E::from_u64(7); relation_plan.relation_row_index_num_vars().unwrap()];
    let mut points = Vec::new();
    for group in relation_plan.groups() {
        let group_index = group.group_index();
        let group_layout = opening_batch.group_layout(group_index).unwrap();
        let group_params = params
            .group_params_geometry(&opening_batch, group_index)
            .unwrap();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            group_params.num_live_ring_elements_per_claim(),
            group_params.num_positions_per_block(),
            group_layout.num_vars(),
            &vec![E::from_u64(5 + group_index as u64); group_layout.num_vars()],
        )
        .unwrap();
        points.push((group_index, point));
    }
    let point_refs = points
        .iter()
        .map(|(group, point)| (*group, point))
        .collect::<Vec<_>>();
    let compact_batch = prepare_coefficient_packing_verifier_batch_semantics(
        CoefficientPackingBatchSemanticInputs {
            level_params: &params,
            opening_batch: &opening_batch,
            relation_plan: &relation_plan,
            relation: &relation,
            prepared_points: &point_refs,
            alpha: E::from_u64(37),
            tau1: &tau1,
            claim_coefficients: &claim_coefficients,
        },
    )
    .unwrap();
    for group in relation_plan.groups() {
        let group_index = group.group_index();
        let group_params = params
            .group_params_geometry(&opening_batch, group_index)
            .unwrap();
        let point = points
            .iter()
            .find_map(|(candidate, point)| (*candidate == group_index).then_some(point))
            .unwrap();
        let semantics =
            prepare_coefficient_packing_group_semantics(CoefficientPackingGroupSemanticInputs {
                level_params: &params,
                opening_batch: &opening_batch,
                relation_plan: &relation_plan,
                relation: &relation,
                group_index,
                prepared_point: point,
                alpha: E::from_u64(37),
                tau1: &tau1,
                claim_coefficients: &claim_coefficients,
            })
            .unwrap();
        let compact = compact_batch
            .groups()
            .iter()
            .find(|candidate| candidate.group_index() == group_index)
            .unwrap();
        assert_eq!(semantics.group_index(), group_index);
        assert_eq!(
            semantics.stage2_terms().group_claim_range(),
            group.claim_range()
        );
        let point = (0..semantics
            .relation_events()
            .physical_field_len()
            .next_power_of_two()
            .trailing_zeros())
            .map(|bit| E::from_u64(41 + u64::from(bit)))
            .collect::<Vec<_>>();
        assert_eq!(
            compact
                .compact_factors()
                .evaluate_relation_at_point(&point)
                .unwrap(),
            semantics
                .relation_events()
                .evaluate_at_point(&point)
                .unwrap()
        );
        assert_eq!(
            compact
                .compact_factors()
                .evaluate_stage2_at_point(&point)
                .unwrap(),
            semantics.stage2_terms().evaluate_at_point(&point).unwrap()
        );
        if group_index == 0 {
            assert_eq!(group_params.log_basis_inner(), 9);
        }
    }
}
