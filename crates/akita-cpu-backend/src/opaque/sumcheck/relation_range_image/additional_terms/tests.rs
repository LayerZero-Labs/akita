use super::*;
use akita_algebra::offset_eq::eq_eval_at_index;
use jolt_field::One;
use jolt_field::Prime128OffsetA7F7 as F;
use std::collections::BTreeMap;

#[test]
fn folded_extension_round_matches_direct_evaluation_with_mixed_support() {
    use jolt_field::{Ext2, ExtField, Prime64Offset59 as B};
    type E = Ext2<B>;
    let value = |index: usize| {
        E::from_base_slice(&[
            -B::from_u64(13 * index as u64 + 7),
            B::from_u64(19 * index as u64 + 3),
        ])
    };
    let domain_len = 1 << 14;
    let weights = (0..domain_len)
        .filter(|index| index % 5 != 0)
        .map(|index| SparseWeight {
            index,
            linear: value(index),
            binary: if index % 17 < 3 {
                value(index + 1)
            } else {
                E::zero()
            },
        })
        .collect();
    let terms = AdditionalRelationTerms {
        weights,
        binary_batching: value(31),
        domain_len,
    };
    let witness = (0..domain_len).map(value).collect::<Vec<_>>();
    let expected_at = |point: E| {
        parent_pairs(&terms.weights)
            .map(|(parent, linear, binary)| {
                let w =
                    witness[2 * parent] + point * (witness[2 * parent + 1] - witness[2 * parent]);
                let linear = linear[0] + point * (linear[1] - linear[0]);
                let binary = binary[0] + point * (binary[1] - binary[0]);
                w * linear + terms.binary_batching * binary * w * (w + E::one())
            })
            .sum::<E>()
    };
    let claim = expected_at(E::zero()) + expected_at(E::one());
    let polynomial = terms.round_message_folded(&witness).into_polynomial(claim);
    for point in [E::zero(), E::one(), value(2), value(5), value(11)] {
        assert_eq!(polynomial.evaluate(point), expected_at(point));
    }
}

fn equality_point(domain_len: usize) -> Vec<F> {
    (0..domain_len.trailing_zeros() as usize)
        .map(|index| F::from_u64(2 + index as u64))
        .collect()
}

fn packed(witness: &[i8]) -> PackedSignedDigits {
    PackedSignedDigits::from_i8_digits_auto(witness.to_vec())
}

fn reference_round_evaluation(terms: &AdditionalRelationTerms<F>, witness: &[i8], point: F) -> F {
    let mut evaluation = F::zero();
    let mut cursor = 0usize;
    while cursor < terms.weights.len() {
        let parent = terms.weights[cursor].index >> 1;
        let mut linear = [F::zero(); 2];
        let mut binary = [F::zero(); 2];
        while cursor < terms.weights.len() && terms.weights[cursor].index >> 1 == parent {
            let weight = terms.weights[cursor];
            let side = weight.index & 1;
            linear[side] = weight.linear;
            binary[side] = weight.binary;
            cursor += 1;
        }
        let witness_at = |index| {
            witness
                .get(index)
                .map_or_else(F::zero, |&value| F::from_i64(i64::from(value)))
        };
        let left = witness_at(2 * parent);
        let witness_at_point = left + point * (witness_at(2 * parent + 1) - left);
        let linear_at_point = linear[0] + point * (linear[1] - linear[0]);
        let binary_at_point = binary[0] + point * (binary[1] - binary[0]);
        evaluation += witness_at_point * linear_at_point
            + terms.binary_batching
                * binary_at_point
                * witness_at_point
                * (witness_at_point + F::one());
    }
    evaluation
}

/// A support spanning several tasks, with parents across task range ends
/// and parents whose bound weights cancel, matches a serial reference.
#[test]
fn multi_task_rounds_and_binds_match_serial_reference() {
    let domain_len = 1 << 16;
    let witness = (0..domain_len)
        .map(|index| ((index * 5 + 3) % 8) as i8 - 4)
        .collect::<Vec<_>>();
    // With the first challenge `2`, a parent with even weight `2c` and odd
    // weight `c` and no binary weight binds to zero.
    let cancelling = 50_000..50_400;
    let linear = (1..domain_len)
        .filter(|index| index % 3 != 1)
        .map(|index| {
            let value = if cancelling.contains(&index) {
                ((index >> 1) % 7 + 1) as u64 * if index & 1 == 0 { 2 } else { 1 }
            } else {
                index as u64 % 11 + 1
            };
            (index, F::from_u64(value))
        })
        .collect();
    let packed_witness = packed(&witness);
    let mut terms = AdditionalRelationTerms::new(
        &packed_witness,
        domain_len,
        linear,
        &[5..9_001, 20_000..40_000],
        &equality_point(domain_len),
        F::from_u64(13),
    )
    .unwrap();
    assert!(parent_ranges(&terms.weights)
        .iter()
        .any(|range| range.len() == TASK_WEIGHTS + 1));
    let mut folded = witness
        .iter()
        .map(|&digit| F::from_i64(i64::from(digit)))
        .collect::<Vec<_>>();
    let mut claim = terms.input_claim(&packed_witness);
    for round in 0..6u64 {
        let polynomial = if round == 0 {
            terms
                .round_message_compact(packed_witness.view(), &[])
                .into_polynomial(claim)
        } else {
            terms.round_message_folded(&folded).into_polynomial(claim)
        };
        assert_eq!(
            polynomial.evaluate(F::zero()) + polynomial.evaluate(F::one()),
            claim
        );
        let challenge = F::from_u64(if round == 0 { 2 } else { 29 + round });
        claim = polynomial.evaluate(challenge);

        let mut expected = BTreeMap::<usize, (F, F)>::new();
        for weight in &terms.weights {
            let scale = if weight.index & 1 == 0 {
                F::one() - challenge
            } else {
                challenge
            };
            let entry = expected
                .entry(weight.index >> 1)
                .or_insert((F::zero(), F::zero()));
            entry.0 += scale * weight.linear;
            entry.1 += scale * weight.binary;
        }
        let parents = expected.len();
        expected.retain(|_, (linear, binary)| !linear.is_zero() || !binary.is_zero());
        if round == 0 {
            assert!(expected.len() < parents);
        }
        terms.bind(challenge);
        assert_eq!(
            terms
                .weights
                .iter()
                .map(|weight| (weight.index, (weight.linear, weight.binary)))
                .collect::<Vec<_>>(),
            expected.into_iter().collect::<Vec<_>>()
        );
        folded = folded
            .chunks(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect();
    }
    let direct = terms
        .weights
        .iter()
        .map(|weight| {
            let witness = folded[weight.index];
            witness * weight.linear
                + terms.binary_batching * weight.binary * witness * (witness + F::one())
        })
        .sum::<F>();
    assert_eq!(direct, claim);
}

#[test]
fn round_polynomial_matches_boolean_sum_and_fold() {
    let witness = [-1, 0, 2, -2];
    let linear = vec![
        (0, F::from_u64(3)),
        (1, F::from_u64(5)),
        (2, F::from_u64(7)),
        (3, F::from_u64(11)),
    ];
    let rho = F::from_u64(13);
    let equality_point = equality_point(4);
    let claim = witness.iter().zip([3, 5, 7, 11]).enumerate().fold(
        F::zero(),
        |sum, (index, (&witness, linear))| {
            let witness = F::from_i64(i64::from(witness));
            let binary = if index < 2 {
                eq_eval_at_index(&equality_point, index)
            } else {
                F::zero()
            };
            sum + witness * F::from_u64(linear) + rho * binary * witness * (witness + F::one())
        },
    );
    let binary_interval = 0..2;
    let packed_witness = packed(&witness);
    let mut prover = AdditionalRelationTerms::new(
        &packed_witness,
        4,
        linear,
        std::slice::from_ref(&binary_interval),
        &equality_point,
        rho,
    )
    .unwrap();
    assert_eq!(prover.input_claim(&packed_witness), claim);
    let polynomial = prover
        .round_message_compact(packed_witness.view(), &[])
        .into_polynomial(claim);
    assert_eq!(
        polynomial.evaluate(F::zero()) + polynomial.evaluate(F::one()),
        claim
    );
    let challenge = F::from_u64(17);
    let next_claim = polynomial.evaluate(challenge);
    prover.bind(challenge);
    let next = prover
        .round_message_compact(packed_witness.view(), &[challenge])
        .into_polynomial(next_claim);
    assert_eq!(
        next.evaluate(F::zero()) + next.evaluate(F::one()),
        next_claim
    );
}

#[test]
fn nonbinary_digit_inside_support_contributes_a_nonzero_constraint() {
    let rho = F::from_u64(13);
    let binary_interval = 0..1;
    let support = std::slice::from_ref(&binary_interval);
    let equality_point = equality_point(2);
    let invalid_witness = packed(&[2, 0]);
    let invalid = AdditionalRelationTerms::new(
        &invalid_witness,
        2,
        Vec::new(),
        support,
        &equality_point,
        rho,
    )
    .unwrap();
    assert_eq!(
        invalid.input_claim(&invalid_witness),
        rho * eq_eval_at_index(&equality_point, 0) * F::from_u64(6)
    );

    let valid_witness = packed(&[-1, 0]);
    let valid =
        AdditionalRelationTerms::new(&valid_witness, 2, Vec::new(), support, &equality_point, rho)
            .unwrap();
    assert_eq!(valid.input_claim(&valid_witness), F::zero());
}

#[test]
fn coefficient_kernel_matches_direct_cubic_evaluation() {
    let witness = [-1, 0, 2, -2, 1, 3, -4, 0];
    let linear = vec![
        (0, F::from_u64(3)),
        (1, F::from_u64(5)),
        (3, F::from_u64(7)),
        (6, F::from_u64(11)),
    ];
    let packed_witness = packed(&witness);
    let terms = AdditionalRelationTerms::new(
        &packed_witness,
        8,
        linear,
        &[1..4, 6..8],
        &equality_point(8),
        F::from_u64(13),
    )
    .unwrap();
    let claim = reference_round_evaluation(&terms, &witness, F::zero())
        + reference_round_evaluation(&terms, &witness, F::one());
    let polynomial = terms
        .round_message_compact(packed_witness.view(), &[])
        .into_polynomial(claim);
    for point in 0..=5 {
        let point = F::from_u64(point);
        assert_eq!(
            polynomial.evaluate(point),
            reference_round_evaluation(&terms, &witness, point)
        );
    }
}

#[test]
fn construction_linearly_merges_duplicates_and_binary_support() {
    let equality_point = equality_point(8);
    let packed_witness = packed(&[0; 8]);
    let terms = AdditionalRelationTerms::new(
        &packed_witness,
        8,
        vec![
            (0, F::from_u64(2)),
            (0, F::from_u64(3)),
            (3, F::from_u64(7)),
            (7, F::from_u64(11)),
        ],
        &[1..4, 6..8],
        &equality_point,
        F::from_u64(13),
    )
    .unwrap();
    assert_eq!(
        terms
            .weights
            .iter()
            .map(|weight| (weight.index, weight.linear, weight.binary))
            .collect::<Vec<_>>(),
        vec![
            (0, F::from_u64(5), F::zero()),
            (1, F::zero(), eq_eval_at_index(&equality_point, 1)),
            (2, F::zero(), eq_eval_at_index(&equality_point, 2)),
            (3, F::from_u64(7), eq_eval_at_index(&equality_point, 3)),
            (6, F::zero(), eq_eval_at_index(&equality_point, 6)),
            (7, F::from_u64(11), eq_eval_at_index(&equality_point, 7)),
        ]
    );
}

#[test]
fn construction_rejects_unsorted_linear_weights() {
    let packed_witness = packed(&[0; 4]);
    assert!(AdditionalRelationTerms::new(
        &packed_witness,
        4,
        vec![(2, F::one()), (1, F::one())],
        &[],
        &equality_point(4),
        F::one(),
    )
    .is_err());
}
