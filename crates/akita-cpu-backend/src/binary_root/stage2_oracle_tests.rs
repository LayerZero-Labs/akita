//! Stage 2 on block profiles against Stage 2 on the materialized root table.

use super::*;
use akita_types::binary_source::{root_weight_fixtures, RootFixtureField as E};
use jolt_algebra::{One, Zero};

/// `eq(point, x)` for every address `x`, with bit `i` of `x` against `point[i]`.
fn equality_table(point: &[E]) -> Vec<E> {
    let mut table = vec![E::one()];
    for &r in point {
        let low = table.iter().map(|&value| value * (E::one() - r));
        let high = table.iter().map(|&value| value * r);
        table = low.chain(high).collect::<Vec<_>>();
    }
    table
}

#[test]
fn profiled_root_stage2_matches_the_dense_table_round_by_round() {
    for fixture in root_weight_fixtures() {
        let dense = fixture.weights.materialize().unwrap();
        let domain_len = dense.len();
        let num_vars = domain_len.trailing_zeros() as usize;
        let block_bits = num_vars.min(ROOT_PROFILE_BLOCK_BITS);
        // Negative-binary addresses hold -1 or 0, so their `w(w+1)` terms vanish.
        let mut witness = (0..domain_len)
            .map(|index| {
                if index < fixture.logical_len {
                    ((index * 2_654_435_761) >> 7 & 7) as i8 - 4
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        for &(address, _) in &fixture.negative_binary_weights {
            witness[address] = -((address & 1) as i8);
        }
        let point = (0..num_vars as u64)
            .map(|bit| E::from_u64(1_000_033 + 17 * bit))
            .collect::<Vec<_>>();
        let range_image = equality_table(&point).into_iter().zip(&witness).fold(
            E::zero(),
            |sum, (equality, &digit)| {
                let digit = E::from_i64(i64::from(digit));
                sum + equality * digit * (digit + E::one())
            },
        );
        let claim = dense
            .iter()
            .zip(&witness)
            .fold(E::zero(), |sum, (&weight, &digit)| {
                sum + weight * E::from_i64(i64::from(digit))
            });
        let batching = E::from_u64(1_000_211);
        let mut expected = CpuStage2SessionHandle::from_dense_relation(
            witness.clone(),
            dense,
            claim,
            &point,
            range_image,
            batching,
        )
        .unwrap()
        .with_negative_binary_weights(fixture.negative_binary_weights.clone())
        .unwrap();
        let live_len = fixture.logical_len.next_multiple_of(1 << block_bits);
        let mut actual = CpuStage2SessionHandle::from_structured_relation(
            PackedSignedDigits::from_i8_digits_auto(witness[..live_len].to_vec()),
            Box::new(fixture.weights.profiles(block_bits).unwrap()),
            block_bits,
            claim,
            &point,
            range_image,
            batching,
        )
        .unwrap()
        .with_negative_binary_weights(fixture.negative_binary_weights)
        .unwrap();
        assert_eq!(actual.input_claim(), expected.input_claim());
        assert_eq!(actual.num_rounds(), num_vars);
        let mut previous = expected.input_claim();
        for round in 0..num_vars {
            let polynomial = expected.round_polynomial(round, previous).unwrap();
            assert_eq!(
                actual.round_polynomial(round, previous).unwrap(),
                polynomial,
                "{} round {round}",
                fixture.name
            );
            let challenge = E::from_u64(2_000_003 + 101 * round as u64);
            previous = polynomial.evaluate(challenge);
            expected.bind_challenge(round, challenge).unwrap();
            actual.bind_challenge(round, challenge).unwrap();
        }
        let expected = expected.finish().unwrap();
        let actual = actual.finish().unwrap();
        assert_eq!(actual.witness_evaluation(), expected.witness_evaluation());
        assert_eq!(actual.final_claim(), expected.final_claim());
    }
}
