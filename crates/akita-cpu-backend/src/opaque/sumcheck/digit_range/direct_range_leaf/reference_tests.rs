use super::{LowBasisRangeCheckProver, PackedSignedDigits};
use akita_sumcheck::EqFactoredSumcheckInstanceProver;
use akita_types::DigitRangePlan;
use jolt_field::{One, Prime128Offset275, Ring, Zero};

type F = Prime128Offset275;

fn root_product(value: F, roots: &[i64]) -> F {
    roots.iter().fold(F::one(), |product, &root| {
        product * (value - F::from_i64(root))
    })
}

fn assert_independent_rounds(
    basis: usize,
    bit_width: u8,
    col_bits: usize,
    ring_bits: usize,
    live_x_cols: usize,
) {
    let half = (basis / 2) as i64;
    // Balanced digits -b/2..b/2 have paired images k(k+1); each distinct
    // image contributes exactly one factor, with monic normalization.
    let mut roots: Vec<i64> = (-half..half).map(|k| k * (k + 1)).collect();
    roots.sort_unstable();
    roots.dedup();
    assert_eq!(roots.len(), basis / 2);
    let plan = DigitRangePlan::new(basis).unwrap();
    let verifier_coeffs = plan.leaf_coeffs::<F>();
    assert_eq!(verifier_coeffs.len(), 1);
    for point in [-7, 0, 1, 2, 6, 12, 19, 101] {
        let point = F::from_i64(point);
        assert_eq!(
            root_product(point, &roots),
            plan.evaluate_leaf_polynomial(&verifier_coeffs[0], point)
        );
    }

    let num_vars = col_bits + ring_bits;
    let witness: Vec<i8> = (0..live_x_cols << ring_bits)
        .map(|i| {
            let mixed = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58;
            (mixed as usize % basis) as i8 - half as i8
        })
        .collect();
    // The flat table is column-major with ring bits low, bound low bit first.
    let tau: Vec<F> = (0..num_vars)
        .map(|i| F::from_u64(3 * i as u64 + 43))
        .collect();
    let mut table: Vec<F> = witness
        .iter()
        .map(|&digit| {
            let digit = F::from_i64(i64::from(digit));
            digit * (digit + F::one())
        })
        .collect();
    table.resize(1 << num_vars, F::zero());
    let mut prover = LowBasisRangeCheckProver::new(
        PackedSignedDigits::from_i8_digits(witness, bit_width).unwrap(),
        &tau,
        plan,
        live_x_cols,
        col_bits,
        ring_bits,
    )
    .unwrap();
    let shape = format!(
        "basis={basis} width={bit_width} col_bits={col_bits} ring_bits={ring_bits} live={live_x_cols}"
    );
    let mut normalized_claim = F::zero();
    for round in 0..num_vars {
        let message = prover.compute_round_eq_factored(round);
        assert_eq!(prover.current_tau(), tau[round], "{shape} round={round}");
        assert_eq!(prover.degree_bound(), roots.len());
        assert_eq!(message.degree(), roots.len(), "{shape} round={round}");
        // claim = (1-tau)q(0) + tau*q(1), hence q(0) = claim - tau*sum(q_i).
        let constant = normalized_claim - tau[round] * message.nonconstant_term_sum_at_one();
        let weights: Vec<F> = (0..table.len() / 2)
            .map(|index| {
                tau[round + 1..]
                    .iter()
                    .enumerate()
                    .fold(F::one(), |weight, (bit, &coordinate)| {
                        weight
                            * if (index >> bit) & 1 == 0 {
                                F::one() - coordinate
                            } else {
                                coordinate
                            }
                    })
            })
            .collect();
        // A degree-b/2 q is determined by b/2+1 distinct evaluations.
        // No production range arithmetic, split-eq tables, or fold helpers.
        for point in 0..=roots.len() {
            let x = F::from_u64(point as u64);
            let expected: F = table
                .chunks_exact(2)
                .zip(&weights)
                .map(|(pair, &weight)| {
                    let folded = (F::one() - x) * pair[0] + x * pair[1];
                    weight * root_product(folded, &roots)
                })
                .sum();
            assert_eq!(
                constant + message.evaluate_nonconstant_terms(x),
                expected,
                "{shape} round={round} point={point}"
            );
        }
        let r = F::from_u64(5 * round as u64 + 71);
        normalized_claim = constant + message.evaluate_nonconstant_terms(r);
        table = table
            .chunks_exact(2)
            .map(|pair| (F::one() - r) * pair[0] + r * pair[1])
            .collect();
        prover.ingest_challenge(round, r);
    }
    assert_eq!(table.len(), 1);
    assert_eq!(prover.final_range_image_eval(), table[0], "{shape}");
    assert_eq!(
        normalized_claim,
        root_product(table[0], &roots),
        "{shape} final normalized claim"
    );
}

#[test]
fn stage1_rounds_match_independent_root_product_reference() {
    const SHAPES: [(usize, usize, usize); 19] = [
        (3, 2, 5),
        (3, 2, 6),
        (3, 2, 8),
        (1, 2, 2),
        (2, 2, 3),
        (3, 3, 5),
        (0, 4, 1),
        (3, 4, 6),
        (2, 6, 3),
        (5, 2, 12),
        (4, 6, 16),
        (2, 3, 1),
        (7, 7, 100),
        (5, 0, 20),
        (4, 1, 9),
        (1, 1, 2),
        (0, 2, 1),
        (2, 1, 3),
        (0, 3, 1),
    ];
    for basis in [4usize, 8] {
        let log_basis = basis.trailing_zeros() as u8;
        for bit_width in [log_basis, log_basis + 1, 8] {
            for (col_bits, ring_bits, live_x_cols) in SHAPES {
                assert_independent_rounds(basis, bit_width, col_bits, ring_bits, live_x_cols);
            }
        }
    }
}
