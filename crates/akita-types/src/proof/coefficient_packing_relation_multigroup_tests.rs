use super::*;

#[test]
fn multi_group_semantics_follow_authenticated_root_order_and_claim_ranges() {
    let super::test_fixtures::CoefficientPackingMultigroupFixture {
        params,
        opening_batch,
        relation_plan,
        relation,
        points,
        claim_coefficients,
        tau1,
    } = super::test_fixtures::coefficient_packing_multigroup_fixture();
    assert_eq!(
        relation_plan
            .groups()
            .iter()
            .map(|group| group.group_index())
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
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
        assert_eq!(semantics.group_index(), group_index);
        assert_eq!(
            semantics.stage2_terms().group_claim_range(),
            group.claim_range()
        );
        let padded_len = semantics
            .relation_events()
            .physical_field_len()
            .next_power_of_two();
        let point = (0..padded_len.trailing_zeros())
            .map(|bit| E::from_u64(41 + u64::from(bit)))
            .collect::<Vec<_>>();
        let mut dense_events = materialize_events(semantics.relation_events());
        dense_events.resize(padded_len, E::zero());
        assert_eq!(
            semantics
                .relation_events()
                .evaluate_at_point(&point)
                .unwrap(),
            multilinear_eval(&dense_events, &point).unwrap()
        );
        if group_index == 0 {
            assert_eq!(group_params.log_basis_inner(), 9);
        }
    }
}
