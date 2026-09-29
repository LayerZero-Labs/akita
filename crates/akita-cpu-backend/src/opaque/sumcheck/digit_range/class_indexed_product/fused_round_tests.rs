use super::*;
use crate::opaque::sumcheck::digit_range::compact_digit_source::CompactDigitSource;
use crate::sources::packed_digits::PackedSignedDigits;
use akita_algebra::eq_poly::EqPolynomial;
use akita_sumcheck::EqFactoredSumcheckInstanceProver;
use akita_types::{DigitRangePlan, FlatBooleanDomain};
use jolt_field::{Ext2, Field, Fold, FpExt4, One, Prime128Offset275, Prime32Offset99};
use jolt_field::{Prime64Offset59, Ring, Unreduced, Zero};

fn eval_leaf<E: Field>(coefficients: &[E], value: E) -> E {
    coefficients
        .iter()
        .rev()
        .copied()
        .fold(E::zero(), |accumulator, coefficient| {
            accumulator * value + coefficient
        })
}

fn row_for_digit<E: Field>(digit: i8, leaf_polynomials: &[Vec<E>]) -> [E; 8] {
    let class = if digit >= 0 {
        i64::from(digit)
    } else {
        -i64::from(digit) - 1
    };
    let range_image = E::from_i64(class * (class + 1));
    std::array::from_fn(|lane| eval_leaf(&leaf_polynomials[lane], range_image))
}

fn direct_equality_weight<E: Field>(point: &[E], index: usize) -> E {
    point
        .iter()
        .enumerate()
        .fold(E::one(), |weight, (bit, &coordinate)| {
            weight
                * if (index >> bit) & 1 == 0 {
                    E::one() - coordinate
                } else {
                    coordinate
                }
        })
}

fn direct_product_coefficients<E: Field, const LANES: usize>(
    left: [E; LANES],
    right: [E; LANES],
    arity: usize,
    parent_weights: &[E],
) -> [E; 5] {
    let mut coefficients = [E::zero(); 5];
    for (parent, &weight) in parent_weights.iter().enumerate() {
        let first_lane = parent * arity;
        let mut product = [E::one(), E::zero(), E::zero(), E::zero(), E::zero()];
        for (degree, lane) in (first_lane..first_lane + arity).enumerate() {
            let slope = right[lane] - left[lane];
            let mut next = [E::zero(); 5];
            for coefficient in 0..=degree {
                next[coefficient] += product[coefficient] * left[lane];
                next[coefficient + 1] += product[coefficient] * slope;
            }
            product = next;
        }
        for coefficient in 0..=arity {
            coefficients[coefficient] += weight * product[coefficient];
        }
    }
    coefficients
}

fn direct_round_coefficients<E: Field, const LANES: usize>(
    table: &[[E; LANES]],
    equality_suffix: &[E],
    arity: usize,
    parent_weights: &[E],
) -> [E; 5] {
    let mut coefficients = [E::zero(); 5];
    for (pair_index, pair) in table.chunks_exact(2).enumerate() {
        let pair_coefficients =
            direct_product_coefficients(pair[0], pair[1], arity, parent_weights);
        let equality_weight = direct_equality_weight(equality_suffix, pair_index);
        for coefficient in 0..=arity {
            coefficients[coefficient] += equality_weight * pair_coefficients[coefficient];
        }
    }
    coefficients
}

fn assert_product_prover_matches_dense_formula<E, const LANES: usize>(
    num_vars: usize,
    live_lengths: &[usize],
    challenge: impl Fn(usize) -> E,
) where
    E: Field + Ring + Fold + Unreduced,
{
    let domain_len = 1usize << num_vars;
    let plan = DigitRangePlan::new(64).unwrap();
    let parent_weights = [E::from_u64(5), E::from_u64(13)];
    let leaf_polynomials = (0..8)
        .map(|lane| {
            vec![
                E::from_u64(2 + lane as u64),
                E::from_u64(7 + 3 * lane as u64),
                E::from_u64(11 + 5 * lane as u64),
            ]
        })
        .collect::<Vec<_>>();
    let challenges = (0..num_vars).map(&challenge).collect::<Vec<_>>();

    for &live_len in live_lengths {
        let digits = (0..live_len)
            .map(|index| ((index * 29 + 7) % 64) as i8 - 32)
            .collect::<Vec<_>>();
        let source = CompactDigitSource::new(
            PackedSignedDigits::from_i8_digits_auto(digits.clone()),
            FlatBooleanDomain::new(live_len, num_vars).unwrap(),
            plan,
        )
        .unwrap();
        let mut prover = ClassIndexedProductSubcheckProver::<E, LANES>::new(
            source,
            plan,
            &leaf_polynomials,
            1,
            parent_weights.to_vec(),
            &challenges,
            E::zero(),
        )
        .unwrap();

        let mut reference_table = (0..domain_len)
            .map(|index| row_for_digit(digits.get(index).copied().unwrap_or(0), &leaf_polynomials))
            .collect::<Vec<_>>();
        for round in 0..num_vars {
            let message = prover.compute_round_eq_factored(round);
            let expected = direct_round_coefficients(
                &reference_table,
                &challenges[round + 1..],
                4,
                &parent_weights,
            );
            assert_eq!(LANES, 8);
            assert_eq!(message.coefficients(), &expected[1..=4]);

            let round_challenge = challenges[round];
            prover.ingest_challenge(round, round_challenge);
            reference_table = reference_table
                .chunks_exact(2)
                .map(|pair| {
                    std::array::from_fn(|lane| {
                        pair[0][lane] + round_challenge * (pair[1][lane] - pair[0][lane])
                    })
                })
                .collect();
        }
        assert_eq!(prover.final_child_claims(), reference_table[0]);
    }
}

fn direct_folded_round_coefficients<E: Field + Ring, const LANES: usize>(
    table: &super::super::exact_prefix::ExactPrefixTable<[E; LANES]>,
    fold_challenge: E,
    equality_point: &[E],
) -> ([E; 5], [E; LANES]) {
    let folded_default = table.default_value();
    let folded: Vec<[E; LANES]> = (0..table.domain_len() / 2)
        .map(|index| {
            let left = table.value_or_default(2 * index);
            let right = table.value_or_default(2 * index + 1);
            std::array::from_fn(|lane| left[lane] + fold_challenge * (right[lane] - left[lane]))
        })
        .collect::<Vec<_>>();
    let mut expected = [E::zero(); 5];
    for pair_index in 0..table.domain_len() / 4 {
        let pair_coefficients = direct_product_coefficients(
            folded[2 * pair_index],
            folded[2 * pair_index + 1],
            2,
            &[E::one()],
        );
        let equality_weight = direct_equality_weight(equality_point, pair_index);
        for coefficient in 0..=2 {
            expected[coefficient] += equality_weight * pair_coefficients[coefficient];
        }
    }
    (expected, folded_default)
}

fn check_fused_then_fallback_boundary<E: Field + Ring + Fold + Unreduced>(
    value: impl Fn(usize) -> E,
) {
    type Row<E> = [E; 2];
    let initial_domain = 1usize << 15;
    let explicit = (0..8_193)
        .map(|row| std::array::from_fn(|lane| value(3 * row + lane + 1)))
        .collect::<Vec<_>>();
    let default = [value(101), value(107)];
    let mut table = super::super::exact_prefix::ExactPrefixTable::<Row<E>>::new(
        initial_domain,
        explicit,
        default,
    )
    .unwrap();
    let mut scratch = Vec::new();
    let parent_weights = [E::one()];

    for (domain_len, equality_bits, challenge_index, expected_explicit_len) in [
        (initial_domain, 13, 211, 8_193),
        (initial_domain / 2, 12, 223, 4_097),
    ] {
        assert_eq!(table.domain_len(), domain_len);
        let explicit_len = table.explicit_len();
        assert_eq!(explicit_len, expected_explicit_len);
        let equality_point = (0..equality_bits)
            .map(|index| value(307 + index))
            .collect::<Vec<_>>();
        let split = equality_point.len().min(8);
        let first = EqPolynomial::evals(&equality_point[..split]).unwrap();
        let second = EqPolynomial::evals(&equality_point[split..]).unwrap();
        let fold_challenge = value(challenge_index);
        let (expected, expected_default) =
            direct_folded_round_coefficients(&table, fold_challenge, &equality_point);
        let actual = super::super::round_accumulation::fold_and_accumulate_equality_weighted_round(
            &mut table,
            &mut scratch,
            &first,
            &second,
            |left, right| {
                std::array::from_fn(|lane| left[lane] + fold_challenge * (right[lane] - left[lane]))
            },
            |left, right| direct_product_coefficients(left, right, 2, &parent_weights),
        );

        assert_eq!(&actual[..=2], &expected[..=2]);
        assert_eq!(table.domain_len(), domain_len / 2);
        assert_eq!(table.explicit_len(), explicit_len.div_ceil(2));
        assert_eq!(table.default_value(), expected_default);
    }
}

#[test]
fn materialized_tree_product_rounds_match_base_field_formula() {
    assert_product_prover_matches_dense_formula::<Prime128Offset275, 8>(
        6,
        &[1, 17, 53, 64],
        |index| Prime128Offset275::from_u64(19 + 7 * index as u64),
    );
}

#[test]
fn fused_then_size_fallback_matches_dense_base_field_formula() {
    check_fused_then_fallback_boundary::<Prime128Offset275>(|index| {
        Prime128Offset275::from_u64(19 + 7 * index as u64)
    });
}

#[test]
fn fused_then_size_fallback_matches_dense_ext2_formula() {
    check_fused_then_fallback_boundary::<Ext2<Prime64Offset59>>(|index| {
        Ext2::new(
            Prime64Offset59::from_u64(17 + 3 * index as u64),
            Prime64Offset59::from_u64(29 + 5 * index as u64),
        )
    });
}

#[test]
fn fused_then_size_fallback_matches_dense_ext4_formula() {
    check_fused_then_fallback_boundary::<FpExt4<Prime32Offset99>>(|index| {
        FpExt4::new(std::array::from_fn(|coordinate| {
            Prime32Offset99::from_u64(13 + (coordinate as u64 + 1) * (index as u64 + 3) * 3)
        }))
    });
}

#[test]
fn materialized_tree_product_rounds_match_ext2_formula() {
    assert_product_prover_matches_dense_formula::<Ext2<Prime64Offset59>, 8>(
        6,
        &[1, 17, 53, 64],
        |index| {
            Ext2::new(
                Prime64Offset59::from_u64(17 + 3 * index as u64),
                Prime64Offset59::from_u64(29 + 5 * index as u64),
            )
        },
    );
}

#[test]
fn materialized_tree_product_rounds_match_ext4_formula() {
    assert_product_prover_matches_dense_formula::<FpExt4<Prime32Offset99>, 8>(
        6,
        &[1, 17, 53, 64],
        |index| {
            FpExt4::new(std::array::from_fn(|coordinate| {
                Prime32Offset99::from_u64(13 + (coordinate as u64 + 1) * (index as u64 + 3) * 3)
            }))
        },
    );
}

#[test]
fn materialized_tree_product_handles_zero_and_one_fold_challenges() {
    type E = Ext2<Prime64Offset59>;
    let nonbase = |index: usize| {
        E::new(
            Prime64Offset59::from_u64(17 + 3 * index as u64),
            Prime64Offset59::from_u64(29 + 5 * index as u64),
        )
    };
    assert_product_prover_matches_dense_formula::<E, 8>(6, &[17], |index| match index {
        2 => E::zero(),
        3 => E::one(),
        _ => nonbase(index),
    });
    assert_product_prover_matches_dense_formula::<E, 8>(6, &[17], |index| match index {
        2 => E::one(),
        3 => E::zero(),
        _ => nonbase(index),
    });
}

#[test]
fn materialized_tree_product_handles_multiple_output_chunks() {
    type E = Ext2<Prime64Offset59>;
    assert_product_prover_matches_dense_formula::<E, 8>(16, &[32_769], |index| {
        Ext2::new(
            Prime64Offset59::from_u64(17 + 3 * index as u64),
            Prime64Offset59::from_u64(29 + 5 * index as u64),
        )
    });
}

#[test]
fn fused_round_handles_empty_explicit_prefix_and_default_suffix() {
    type E = Ext2<Prime64Offset59>;
    type Row = [E; 2];
    let default = [
        E::new(Prime64Offset59::from_u64(7), Prime64Offset59::from_u64(3)),
        E::new(Prime64Offset59::from_u64(11), Prime64Offset59::from_u64(5)),
    ];
    let point = [
        E::new(Prime64Offset59::from_u64(13), Prime64Offset59::from_u64(2)),
        E::new(Prime64Offset59::from_u64(17), Prime64Offset59::from_u64(4)),
    ];
    let first = EqPolynomial::evals(&point[..1]).unwrap();
    let second = EqPolynomial::evals(&point[1..]).unwrap();
    let fold_challenge = E::new(Prime64Offset59::from_u64(23), Prime64Offset59::from_u64(9));
    let mut table =
        super::super::exact_prefix::ExactPrefixTable::new(16, Vec::<Row>::new(), default).unwrap();
    let mut scratch = Vec::new();
    let folded_default = std::array::from_fn(|lane| {
        default[lane] + fold_challenge * (default[lane] - default[lane])
    });
    let expected_pair = direct_product_coefficients(folded_default, folded_default, 2, &[E::one()]);
    let total_suffix_weight: E = (0..4)
        .map(|pair| first[pair % first.len()] * second[pair / first.len()])
        .sum();
    let expected = expected_pair.map(|coefficient| total_suffix_weight * coefficient);

    let actual = super::super::round_accumulation::fold_and_accumulate_equality_weighted_round(
        &mut table,
        &mut scratch,
        &first,
        &second,
        |left, right| {
            std::array::from_fn(|lane| left[lane] + fold_challenge * (right[lane] - left[lane]))
        },
        |left, right| direct_product_coefficients(left, right, 2, &[E::one()]),
    );

    assert_eq!(actual, expected);
    assert_eq!(table.domain_len(), 8);
    assert_eq!(table.explicit_len(), 0);
    assert_eq!(table.default_value(), folded_default);
}
