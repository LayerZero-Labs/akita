use super::*;
use jolt_field::Zero;

#[test]
fn committed_quotient_rows_have_zero_targets_and_authenticate_all_segments() {
    type F = Prime128Offset275;
    let input = fixture();
    let fixed = build(&input).unwrap();
    let committed = fixed
        .committed_quotient(&input.descriptor, &input.matrix, &input.challenges, 256)
        .unwrap();
    let plan = committed.relation.repacking();
    assert_eq!(committed.bounds.digit_count(), 8);
    assert_eq!(committed.bounds.carry_bound(), 9_586_980);
    assert!(committed
        .relation
        .groups()
        .iter()
        .flat_map(|g| &g.rows)
        .all(|row| row.target.iter().all(F::is_zero)));
    let table = committed
        .digits
        .iter()
        .map(|&x| F::from_i64(i64::from(x)))
        .collect::<Vec<_>>();
    plan.validate_digit_table(&table).unwrap();
    for (element, row) in fixed.rows.iter().enumerate() {
        for (coefficient, &residual) in row.remainder.iter().enumerate() {
            assert_eq!(
                plan.recompose_coefficient(&table, 2, element, coefficient)
                    .unwrap(),
                F::from_i128(residual / i128::from(Q24)),
            );
        }
    }
    let padding = (0..plan.domain_len())
        .filter(|&index| plan.is_padding_address(index).unwrap())
        .collect::<Vec<_>>();
    let weights = committed
        .relation
        .ring_switch(
            F::from_u64(17),
            &vec![F::one(); committed.relation.row_count()],
            &vec![F::one(); padding.len()],
        )
        .unwrap();
    weights.check_table(&table).unwrap();
    // These mutate the authenticated table, not just an honest construction hint.
    for address in [
        plan.coefficient_address(0, 0, 0, 0).unwrap(), // response
        plan.coefficient_address(1, 0, 0, 0).unwrap(), // image
        plan.coefficient_address(2, 0, 0, 0).unwrap(), // quotient
        plan.coefficient_address(2, 0, 7, 647).unwrap(), // highest quotient digit
        padding[0],
    ] {
        let mut mutated = table.clone();
        mutated[address] += F::one();
        assert!(weights.check_table(&mutated).is_err());
    }
}

#[test]
fn quotient_construction_rejects_indivisible_and_out_of_range_residuals() {
    let input = fixture();
    let mut fixed = build(&input).unwrap();
    let original = fixed.rows[0].remainder[0];
    fixed.rows[0].remainder[0] += 1;
    assert!(fixed
        .committed_quotient(&input.descriptor, &input.matrix, &input.challenges, 256)
        .is_err());
    for carry in [-9_586_981i128, 7_190_236] {
        fixed.rows[0].remainder[0] = i128::from(Q24) * carry;
        assert!(fixed
            .committed_quotient(&input.descriptor, &input.matrix, &input.challenges, 256)
            .is_err());
    }
    fixed.rows[0].remainder[0] = original;
    assert!(fixed
        .committed_quotient(&input.descriptor, &input.matrix, &input.challenges, 648)
        .is_err());
}

#[test]
fn quotient_revalidates_retained_public_construction_fields() {
    let input = fixture();
    let original = build(&input).unwrap();
    let rejects = |fixed: BinaryRootRelation| {
        assert!(fixed
            .committed_quotient(&input.descriptor, &input.matrix, &input.challenges, 256,)
            .is_err());
    };
    let mut fixed = original.clone();
    fixed.rows[0].remainder[0] += i128::from(Q24);
    rejects(fixed);
    let mut fixed = original.clone();
    // An unbounded challenge invalidates Gamma, even if a forged residual
    // happens to produce a satisfied equation modulo the proof field.
    fixed.embedded_challenges[0][0] = i128::MAX;
    rejects(fixed);
    let mut fixed = original.clone();
    fixed.response[0][0] = i128::from(BINARY_RESPONSE_INTERVAL.1) + 1;
    rejects(fixed);
    let mut fixed = original.clone();
    fixed.images.pop();
    rejects(fixed);
    let mut fixed = original;
    fixed.response.pop();
    rejects(fixed);
}
