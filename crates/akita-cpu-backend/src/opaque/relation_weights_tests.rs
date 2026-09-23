use super::*;
use akita_types::CommitmentRingDims;
use jolt_field::{One, Prime128OffsetA7F7, Zero};

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

#[test]
fn setup_columns_batch_logical_weights_over_one_physical_family() {
    let row_0 = [
        TestField::from_u64(1),
        TestField::from_u64(2),
        TestField::from_u64(3),
        TestField::from_u64(4),
    ];
    let row_1 = [
        TestField::from_u64(5),
        TestField::from_u64(6),
        TestField::from_u64(7),
        TestField::from_u64(8),
    ];
    let family = SetupRows {
        rows: vec![&row_0, &row_1],
        ring_d: 2,
    };
    let alpha = TestField::from_u64(11);
    let alpha_powers = [TestField::one(), alpha];
    let row_weights = vec![
        (0, vec![TestField::from_u64(2), TestField::from_u64(3)]),
        (1, vec![TestField::from_u64(5), TestField::zero()]),
    ];

    let evaluated = contract_setup_columns(&family, 0..2, &row_weights, 2, 1, |coefficients| {
        Ok(vec![eval_flat_ring_at_pows_fast(
            coefficients,
            &alpha_powers,
        )])
    })
    .unwrap();
    for column in 0..2 {
        let row_0_eval = row_0[2 * column] + alpha * row_0[2 * column + 1];
        let row_1_eval = row_1[2 * column] + alpha * row_1[2 * column + 1];
        assert_eq!(
            evaluated.get_scalar(0, column).unwrap(),
            TestField::from_u64(2) * row_0_eval + TestField::from_u64(5) * row_1_eval,
        );
        assert_eq!(
            evaluated.get_scalar(1, column).unwrap(),
            TestField::from_u64(3) * row_0_eval,
        );
    }
}
