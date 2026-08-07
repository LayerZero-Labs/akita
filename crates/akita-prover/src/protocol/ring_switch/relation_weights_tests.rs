use super::*;
use akita_types::CommitmentRingDims;
use jolt_field::One;
use jolt_field::Prime128OffsetA7F7;

type TestField = Prime128OffsetA7F7;

fn mixed_dimension_events() -> RelationWeightEvents<TestField> {
    let role_dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 32,
    };
    let mut events = RelationWeightEvents {
        events: Vec::new(),
        alpha_powers: scalar_powers(TestField::from_u64(7), role_dims.d_a()),
        relation_coefficient_block_len: 32,
        physical_field_len: 256,
        setup_is_deferred: false,
    };
    events
        .push(
            0,
            128,
            0,
            TestField::from_u64(2),
            RelationWeightContribution::Constraint,
        )
        .unwrap();
    events
        .push(
            32,
            32,
            0,
            TestField::from_u64(3),
            RelationWeightContribution::SetupMatrix,
        )
        .unwrap();
    events
        .push(
            64,
            64,
            64,
            TestField::from_u64(5),
            RelationWeightContribution::Constraint,
        )
        .unwrap();
    events
        .push(
            128,
            64,
            0,
            TestField::from_u64(11),
            RelationWeightContribution::SetupMatrix,
        )
        .unwrap();
    events
        .push(
            96,
            64,
            0,
            TestField::from_u64(13),
            RelationWeightContribution::Constraint,
        )
        .unwrap();
    events
}

#[test]
fn mixed_dimension_factorization_reconstructs_dense_weights() {
    let events = mixed_dimension_events();
    let dense = events.materialize_dense().unwrap();
    let factorization = events.factor_common_alpha().unwrap();
    assert_eq!(factorization.common_alpha_factor().len(), 32);
    assert_eq!(
        factorization.relation_lane_weights().len(),
        dense.len() / 32
    );
    for (lane, &lane_weight) in factorization.relation_lane_weights().iter().enumerate() {
        for (coefficient, &alpha_power) in factorization.common_alpha_factor().iter().enumerate() {
            assert_eq!(
                dense[lane * factorization.common_alpha_factor().len() + coefficient],
                lane_weight * alpha_power,
            );
        }
    }
}

#[test]
fn factorization_materialization_preserves_relation_weights() {
    let events = mixed_dimension_events();
    let point = (0..8)
        .map(|index| TestField::from_u64(101 + index))
        .collect::<Vec<_>>();
    let expected_dense = events.materialize_dense().unwrap();
    let expected_factorization = events.factor_common_alpha().unwrap();
    let expected_evaluation = events.evaluate_at_point(&point, None).unwrap();
    assert_eq!(
        expected_factorization.materialize_dense().unwrap(),
        expected_dense
    );
    assert_eq!(
        events.evaluate_at_point(&point, None).unwrap(),
        expected_evaluation
    );
}

#[test]
fn factorization_rejects_an_unaligned_alpha_reset() {
    let mut events = mixed_dimension_events();
    events.events.clear();
    assert!(matches!(
        events.push(
            0,
            32,
            16,
            TestField::one(),
            RelationWeightContribution::Constraint,
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}
