use super::{LowBasisRangeCheckProver, PackedSignedDigits};
use akita_sumcheck::EqFactoredSumcheckInstanceProver;
use akita_types::DigitRangePlan;
use jolt_field::{
    Ext2, Field, Fold, FpExt4, Prime128Offset275, Prime32Offset99, Prime64Offset59, Ring, Unreduced,
};

fn unique_roots(basis: usize) -> Vec<i64> {
    let half = (basis / 2) as i64;
    let mut roots: Vec<_> = (-half..half).map(|digit| digit * (digit + 1)).collect();
    roots.sort_unstable();
    roots.dedup();
    roots
}

fn equality_weight<E: Field + Ring>(index: usize, point: &[E]) -> E {
    point
        .iter()
        .enumerate()
        .fold(E::one(), |weight, (bit, &coordinate)| {
            if (index >> bit) & 1 == 0 {
                weight * (E::one() - coordinate)
            } else {
                weight * coordinate
            }
        })
}

fn direct_range_round<E: Field + Ring>(table: &[E], point: &[E], x: E, roots: &[i64]) -> E {
    table
        .chunks_exact(2)
        .enumerate()
        .fold(E::zero(), |sum, (pair_index, pair)| {
            let folded = pair[0] + x * (pair[1] - pair[0]);
            let range_value = roots.iter().fold(E::one(), |product, &root| {
                product * (folded - E::from_i64(root))
            });
            sum + equality_weight(pair_index, point) * range_value
        })
}

fn direct_multilinear_evaluation<E: Field + Ring>(table: &[E], point: &[E]) -> E {
    table
        .iter()
        .enumerate()
        .fold(E::zero(), |sum, (index, &value)| {
            sum + equality_weight(index, point) * value
        })
}

fn assert_stage1_matches_direct_formula<E>(value: fn(usize) -> E)
where
    E: Field + Ring + Unreduced + Fold,
{
    let shapes = [(2usize, 1usize, 3usize), (3, 2, 5), (4, 1, 9), (2, 3, 3)];
    for basis in [4usize, 8] {
        let half = (basis / 2) as i8;
        let roots = unique_roots(basis);

        for (col_bits, ring_bits, live_x_cols) in shapes {
            let num_vars = col_bits + ring_bits;
            let live_digits = live_x_cols << ring_bits;
            let witness: Vec<i8> = (0..live_digits)
                .map(|index| ((index * 7 + 3) % basis) as i8 - half)
                .collect();
            let tau: Vec<E> = (0..num_vars).map(|index| value(101 + index)).collect();
            let mut initial_table: Vec<E> = witness
                .iter()
                .map(|&digit| {
                    let digit = E::from_i64(i64::from(digit));
                    digit * (digit + E::one())
                })
                .collect();
            initial_table.resize(1usize << num_vars, E::zero());

            for challenge_pattern in 0..3 {
                let challenges: Vec<E> = (0..num_vars)
                    .map(|round| match challenge_pattern {
                        0 => E::zero(),
                        1 => E::one(),
                        _ if round == 0 => E::zero(),
                        _ if round == 1 => E::one(),
                        _ => value(10_007 + round),
                    })
                    .collect();
                let mut reference_table = initial_table.clone();
                let mut prover = LowBasisRangeCheckProver::new(
                    PackedSignedDigits::from_i8_digits_auto(witness.clone()),
                    &tau,
                    DigitRangePlan::new(basis).unwrap(),
                    live_x_cols,
                    col_bits,
                    ring_bits,
                )
                .unwrap();
                let mut claim = E::zero();

                for round in 0..num_vars {
                    let message = prover.compute_round_eq_factored(round);
                    let tau_round = prover.current_tau();
                    assert_eq!(tau_round, tau[round]);
                    assert_eq!(message.degree(), roots.len());

                    let constant = claim - tau_round * message.nonconstant_term_sum_at_one();
                    assert_eq!(
                        constant,
                        direct_range_round(
                            &reference_table,
                            &tau[round + 1..],
                            E::zero(),
                            &roots,
                        ),
                        "constant term: basis={basis}, shape={col_bits}/{ring_bits}/{live_x_cols}, round={round}"
                    );

                    for point in 0..=roots.len() {
                        let x = E::from_u64(point as u64);
                        assert_eq!(
                            constant + message.evaluate_nonconstant_terms(x),
                            direct_range_round(
                                &reference_table,
                                &tau[round + 1..],
                                x,
                                &roots,
                            ),
                            "round polynomial: basis={basis}, shape={col_bits}/{ring_bits}/{live_x_cols}, round={round}, x={point}, pattern={challenge_pattern}"
                        );
                    }

                    let challenge = challenges[round];
                    claim = constant + message.evaluate_nonconstant_terms(challenge);
                    prover.ingest_challenge(round, challenge);
                    reference_table = reference_table
                        .chunks_exact(2)
                        .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                        .collect();
                }

                let direct_terminal = direct_multilinear_evaluation(&initial_table, &challenges);
                assert_eq!(reference_table.len(), 1);
                assert_eq!(reference_table[0], direct_terminal);
                assert_eq!(prover.final_range_image_eval(), direct_terminal);
                let expected_claim = roots.iter().fold(E::one(), |product, &root| {
                    product * (direct_terminal - E::from_i64(root))
                });
                assert_eq!(claim, expected_claim);
            }
        }
    }
}

fn prime128_value(index: usize) -> Prime128Offset275 {
    -Prime128Offset275::from_u64((index as u64).wrapping_mul(37) + 11)
}

fn ext2_value(index: usize) -> Ext2<Prime64Offset59> {
    type F = Prime64Offset59;
    Ext2::new(
        -F::from_u64((index as u64).wrapping_mul(19) + 7),
        -F::from_u64((index as u64).wrapping_mul(29) + 13),
    )
}

fn fp_ext4_value(index: usize) -> FpExt4<Prime32Offset99> {
    type F = Prime32Offset99;
    FpExt4::new(std::array::from_fn(|coordinate| {
        -F::from_u64(
            (index as u64)
                .wrapping_mul(23 + coordinate as u64 * 6)
                .wrapping_add(5 + coordinate as u64),
        )
    }))
}

#[test]
fn stage1_rounds_and_terminal_opening_match_direct_base_field_formula() {
    assert_stage1_matches_direct_formula(prime128_value);
}

#[test]
fn stage1_rounds_and_terminal_opening_match_direct_ext2_formula() {
    assert_stage1_matches_direct_formula(ext2_value);
}

#[test]
fn stage1_rounds_and_terminal_opening_match_direct_ext4_formula() {
    assert_stage1_matches_direct_formula(fp_ext4_value);
}
