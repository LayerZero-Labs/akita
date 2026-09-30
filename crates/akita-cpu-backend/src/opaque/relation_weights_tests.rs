use super::*;
use jolt_field::{One, Prime128OffsetA7F7, Zero};

type TestField = Prime128OffsetA7F7;

const BLOCK_LEN: usize = 32;
const FIELD_LEN: usize = 256;

/// `(physical_start, coefficient_count, alpha_exponent_start, scalar,
/// contribution)` over roles of dimensions 128, 64, and 32.
const MIXED_DIMENSION_CONTRIBUTIONS: [(usize, usize, usize, u64, RelationWeightContribution); 5] = [
    (0, 128, 0, 2, RelationWeightContribution::Constraint),
    (32, 32, 0, 3, RelationWeightContribution::SetupMatrix),
    (64, 64, 64, 5, RelationWeightContribution::Constraint),
    (128, 64, 0, 11, RelationWeightContribution::SetupMatrix),
    (96, 64, 0, 13, RelationWeightContribution::Constraint),
];

fn alpha_powers() -> Vec<TestField> {
    scalar_powers(TestField::from_u64(7), 128)
}

fn empty_weights(setup_is_deferred: bool) -> RelationLaneWeights<TestField> {
    RelationLaneWeights::new(alpha_powers(), BLOCK_LEN, FIELD_LEN, setup_is_deferred).unwrap()
}

#[test]
fn mixed_dimension_factorization_reconstructs_dense_weights() {
    let alpha_powers = alpha_powers();
    let mut weights = empty_weights(false);
    let mut expected = vec![TestField::zero(); FIELD_LEN];
    for (start, count, alpha_start, scalar, contribution) in MIXED_DIMENSION_CONTRIBUTIONS {
        let scalar = TestField::from_u64(scalar);
        weights
            .push(start, count, alpha_start, scalar, contribution)
            .unwrap();
        for offset in 0..count {
            expected[start + offset] += scalar * alpha_powers[alpha_start + offset];
        }
    }
    let factorization = weights.into_factorization().unwrap();
    assert_eq!(factorization.common_alpha_factor().len(), BLOCK_LEN);
    assert_eq!(
        factorization.relation_lane_weights().len(),
        FIELD_LEN / BLOCK_LEN
    );
    assert_eq!(factorization.materialize_dense().unwrap(), expected);
}

#[test]
fn deferred_setup_rejects_setup_terms_and_factorization() {
    let mut weights = empty_weights(true);
    assert!(matches!(
        weights.push(
            0,
            32,
            0,
            TestField::one(),
            RelationWeightContribution::SetupMatrix,
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
    weights
        .push(
            0,
            32,
            0,
            TestField::one(),
            RelationWeightContribution::Constraint,
        )
        .unwrap();
    assert!(matches!(
        weights.into_factorization(),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn factorization_rejects_an_unaligned_alpha_reset() {
    assert!(matches!(
        empty_weights(false).push(
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
