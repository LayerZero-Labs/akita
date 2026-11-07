use super::*;
use jolt_field::{One, Prime128OffsetA7F7 as F, Ring};

fn check_distinct_and_repeated_weights(weights: Vec<u64>) {
    let count = weights.len();
    let sum = weights.iter().sum::<u64>();
    let families = weights
        .into_iter()
        .map(|weight| {
            EqPairTensorFamily::new(
                0,
                0,
                F::one(),
                vec![
                    EqPairTensorAxis::bit_product(0, 3, vec![[F::from_u64(weight), F::one()]])
                        .unwrap(),
                ],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut work = 0;
    let result = eval_unfactored_tensor_families::<F, false, false>(
        &[],
        &[F::from_u64(2), F::from_u64(3)],
        &families,
        &mut work,
    )
    .unwrap();
    // The only right addresses are 0 and 3, with equality weights 2 and 6.
    assert_eq!(result, F::from_u64(2 * sum + 6 * count as u64));
    assert!(work < 256 * count, "grouping work must remain linear");
    let mut exhausted = crate::offset_eq::MAX_COMPACT_STRIDE_TERMS;
    assert!(eval_unfactored_tensor_families::<F, false, false>(
        &[],
        &[F::from_u64(2), F::from_u64(3)],
        &families,
        &mut exhausted,
    )
    .is_err());
}

#[test]
fn many_distinct_weight_groups_use_the_singleton_fallback() {
    check_distinct_and_repeated_weights((1..=50_000).collect());
}

#[test]
fn repeated_weights_still_batch_after_the_group_limit() {
    check_distinct_and_repeated_weights((0..50_000).map(|i| i % 37 + 1).collect());
}
