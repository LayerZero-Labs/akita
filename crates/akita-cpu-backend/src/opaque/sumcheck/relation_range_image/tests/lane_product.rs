use super::trace_prefix::two_source_linear_terms;
use super::*;

/// Every round of the sumcheck is consistent with its claim, and the last
/// claim is the direct evaluation of the Stage-2 summand at the challenges.
/// The relation lane weights are nonzero past the live lanes, so the lane
/// rounds must carry that tail.
#[test]
fn stage2_rounds_reach_the_direct_final_evaluation() {
    let lane_bits = 5usize;
    let live_lane_count = 19usize;
    for (b, coefficient_bits) in [(8usize, 6usize), (4, 3), (8, 2), (4, 1), (8, 0)] {
        let coeff_count = 1usize << coefficient_bits;
        let num_vars = lane_bits + coefficient_bits;
        let half = (b / 2) as i8;
        let compact_witness = (0..live_lane_count * coeff_count)
            .map(|index| ((5 * index + 3) % b) as i8 - half)
            .collect::<Vec<_>>();
        let stage1_point = (0..num_vars)
            .map(|index| F::from_u64(401 + 13 * index as u64))
            .collect::<Vec<_>>();
        let common_alpha_factor = (0..coeff_count)
            .map(|index| F::from_u64(503 + 17 * index as u64))
            .collect::<Vec<_>>();
        let relation_lane_weights = (0..1usize << lane_bits)
            .map(|index| F::from_u64(601 + 19 * index as u64))
            .collect::<Vec<_>>();
        let (structured, linear_weights) = two_source_linear_terms(live_lane_count, coeff_count);
        let batching_coeff = F::from_u64(701);
        let mut prover = new_stage2_test_prover_with_linear_terms(
            batching_coeff,
            compact_witness.clone(),
            common_alpha_factor.clone(),
            relation_lane_weights.clone(),
            linear_weights.clone(),
            structured,
            Stage2Params {
                stage1_point: &stage1_point,
                b,
                live_lane_count,
                lane_bits,
                coefficient_bits,
            },
        );

        let mut claim = prover.input_claim();
        let mut challenges = Vec::with_capacity(num_vars);
        for round in 0..num_vars {
            let poly = prover.compute_round_univariate(round, claim);
            assert_eq!(
                poly.evaluate(F::zero()) + poly.evaluate(F::one()),
                claim,
                "b {b}, coefficient bits {coefficient_bits}, round {round}"
            );
            let challenge = F::from_u64(809 + 23 * round as u64);
            claim = poly.evaluate(challenge);
            prover.ingest_challenge(round, challenge);
            challenges.push(challenge);
        }

        let eq = EqPolynomial::evals(&challenges).unwrap();
        let padded = pad_compact_witness(
            &compact_witness,
            live_lane_count,
            lane_bits,
            coefficient_bits,
        );
        let witness = eq
            .iter()
            .zip(&padded)
            .map(|(&weight, &digit)| weight * F::from_i64(i64::from(digit)))
            .sum::<F>();
        let relation_weight = eq
            .iter()
            .enumerate()
            .map(|(index, &weight)| {
                weight
                    * common_alpha_factor[index % coeff_count]
                    * relation_lane_weights[index / coeff_count]
            })
            .sum::<F>();
        let linear_weight = eq
            .iter()
            .zip(&linear_weights)
            .map(|(&weight, &linear)| weight * linear)
            .sum::<F>();
        let range_eq = stage1_point
            .iter()
            .zip(&challenges)
            .map(|(&tau, &r)| tau * r + (F::one() - tau) * (F::one() - r))
            .product::<F>();
        let expected = batching_coeff * range_eq * witness * (witness + F::one())
            + witness * (relation_weight + linear_weight);
        assert_eq!(
            claim, expected,
            "b {b}, coefficient bits {coefficient_bits}"
        );
        assert_eq!(prover.expected_final_claim().unwrap(), claim);
    }
}
