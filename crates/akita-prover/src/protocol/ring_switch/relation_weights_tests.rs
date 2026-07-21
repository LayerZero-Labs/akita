use super::*;
use akita_field::Prime128OffsetA7F7;

type TestField = Prime128OffsetA7F7;

fn mixed_dimension_events() -> RelationWeightEvents<TestField> {
    let role_dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 32,
    };
    let mut events = RelationWeightEvents {
        events: Vec::new(),
        inner_alpha_powers: scalar_powers(TestField::from_u64(7), role_dims.d_a()),
        role_dims,
        opening_source_len: 3,
        opening_ring_dim: 128,
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
fn factorization_rejects_an_unaligned_alpha_reset() {
    let mut events = mixed_dimension_events();
    events.events.clear();
    events
        .push(
            0,
            32,
            16,
            TestField::one(),
            RelationWeightContribution::Constraint,
        )
        .unwrap();
    assert!(matches!(
        events.factor_common_alpha(),
        Err(AkitaError::InvalidSetup(_))
    ));
}
