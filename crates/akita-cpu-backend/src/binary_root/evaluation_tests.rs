use super::*;
use akita_algebra::binary::BinaryField162;
use akita_types::binary_source::{
    binary_scalar_geometry, binary_scalar_lift, BinaryEvaluationSegments, BinaryTerminalClaim,
};
use jolt_algebra::Zero;

#[test]
fn terminal_carries_authenticate_response_partials_and_value() {
    type F = Prime128Offset275;
    let input = fixture();
    let response = build(&input).unwrap();
    let row = vec![
        BinaryField162::from_words([13, 0, 0]).unwrap().to_bytes(),
        BinaryField162::from_words([19, 0, 0]).unwrap().to_bytes(),
    ];
    let col = vec![BinaryField162::from_words([23, 0, 0]).unwrap().to_bytes()];
    let provisional = BinaryTerminalClaim::new(
        &input.descriptor,
        [1; 32],
        row.clone(),
        col.clone(),
        [0; 21],
    )
    .unwrap();
    let partials = binary_row_partials(&input.descriptor, &provisional, &input.sources).unwrap();
    let (_, columns) = provisional.equality_weights(&input.descriptor).unwrap();
    let mut value = BinaryField162::ZERO;
    for (column, weight) in columns.into_iter().enumerate() {
        let mut bytes = [0u8; 21];
        for (coefficient, &digit) in partials[column * 162..(column + 1) * 162]
            .iter()
            .enumerate()
        {
            bytes[coefficient / 8] |= ((digit & 1) as u8) << (coefficient % 8);
        }
        value += weight * BinaryField162::from_bytes(&bytes).unwrap();
    }
    let claim = BinaryTerminalClaim::new(
        &input.descriptor,
        [1; 32],
        row.clone(),
        col.clone(),
        value.to_bytes(),
    )
    .unwrap();
    let modulus = u128::MAX - 274;
    let carry = BinaryEvaluationWitness::build(
        &input.descriptor,
        &claim,
        &partials,
        &response.response,
        &input.challenges,
        modulus,
    )
    .unwrap();
    // Independent previous construction: reduce each dense integer product
    // before summing. This pins low-digit signs and the final carry encoding
    // against the shared unreduced convolution and single final reduction.
    let (row_weights, _) = claim.equality_weights(&input.descriptor).unwrap();
    let scalar_challenges =
        akita_types::binary_source::binary_scalar_challenges(&input.descriptor, &input.challenges)
            .unwrap();
    let mut parity = [0i128; 162];
    for (row, weight) in row_weights.into_iter().enumerate() {
        let low = response.response[row / 4].map(|coefficient| (coefficient + 2340) % 8 - 4);
        let components =
            akita_algebra::trinomial::module_map::unpack_signed_components(&low).unwrap();
        let product = akita_algebra::trinomial::signed::signed_trinomial_product::<
            162,
            akita_algebra::PlusTrinomial,
        >(&binary_scalar_lift(weight), &components[row % 4])
        .unwrap();
        for (sum, value) in parity.iter_mut().zip(product) {
            *sum += value;
        }
    }
    for (column, challenge) in scalar_challenges.iter().take(2).enumerate() {
        let product = akita_algebra::trinomial::signed::signed_trinomial_product::<
            162,
            akita_algebra::PlusTrinomial,
        >(&partials[column * 162..(column + 1) * 162], challenge)
        .unwrap();
        for (sum, value) in parity.iter_mut().zip(product) {
            *sum -= value;
        }
    }
    for (coefficient, integer) in parity.into_iter().enumerate() {
        assert_eq!(integer % 2, 0);
        let offset = (integer / 2 + carry.bounds.parity_carry_bound as i128) as u128;
        for (digit, row) in carry.parity_digits.chunks_exact(162).take(9).enumerate() {
            assert_eq!(row[coefficient], ((offset >> (digit * 3)) & 7) as i8 - 4);
        }
    }
    let (base, base_digits) =
        BinaryRootRelation::successor_table(&response.response, &response.images, 256).unwrap();
    let scalar = binary_scalar_geometry().unwrap();
    let mut segments = base.segments().to_vec();
    for (elements, digits) in [(4, 1), (2, 9), (2, 6)] {
        segments.push(RootRepackingSegment {
            geometry: scalar,
            elements,
            digits,
        });
    }
    let plan = RootCoefficientRepacking::new(segments, 256).unwrap();
    let mut table = vec![F::zero(); plan.domain_len()];
    for (target, digit) in table.iter_mut().zip(base_digits) {
        *target = F::from_i64(i64::from(digit));
    }
    for (segment, natural) in [
        (2, &partials),
        (3, &carry.parity_digits),
        (4, &carry.value_digits),
    ] {
        let shape = plan.segments()[segment];
        for (index, &digit) in natural.iter().enumerate() {
            let lane = index / 162;
            let address = plan
                .coefficient_address(
                    segment,
                    lane / shape.digits,
                    lane % shape.digits,
                    index % 162,
                )
                .unwrap();
            table[address] = F::from_i64(i64::from(digit));
        }
    }
    let addresses = BinaryEvaluationSegments {
        response: 0,
        partials: 2,
        parity_carry: 3,
        value_carry: 4,
    };
    let weights = claim
        .relation_weights(
            &input.descriptor,
            &plan,
            addresses,
            &input.challenges,
            F::from_u64(17),
            [F::from_u64(29), F::from_u64(31)],
            modulus,
        )
        .unwrap();
    weights.check_table(&table).unwrap();
    plan.validate_digit_table(&table).unwrap();
    let dense = weights.materialize().unwrap();
    for segment in [0, 2, 3, 4] {
        let range = plan.segment_range(segment).unwrap();
        let address = range
            .clone()
            .find(|&address| !dense[address].is_zero())
            .unwrap();
        let mut altered = table.clone();
        altered[address] += F::one();
        assert!(weights.check_table(&altered).is_err());
    }
    for which in 0..3 {
        let mut changed_value = value.to_bytes();
        let mut changed_row = row.clone();
        let mut changed_col = col.clone();
        match which {
            0 => changed_value[0] ^= 1,
            1 => changed_row[0][0] ^= 1,
            _ => changed_col[0][0] ^= 1,
        };
        let altered = BinaryTerminalClaim::new(
            &input.descriptor,
            [1; 32],
            changed_row,
            changed_col,
            changed_value,
        )
        .unwrap();
        let altered_weights = altered
            .relation_weights(
                &input.descriptor,
                &plan,
                addresses,
                &input.challenges,
                F::from_u64(17),
                [F::from_u64(29), F::from_u64(31)],
                modulus,
            )
            .unwrap();
        assert!(altered_weights.check_table(&table).is_err());
    }
}
