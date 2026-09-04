use super::*;

#[test]
fn reduced_dense_oracle_matches_factored_stage2_across_all_rounds() {
    let lane_bits = 3;
    let coefficient_bits = 2;
    let live_lane_count = 5;
    let coeff_count = 1usize << coefficient_bits;
    let lane_capacity = 1usize << lane_bits;
    let stage1_point = (0..lane_bits + coefficient_bits)
        .map(|index| F::from_u64(17 + index as u64))
        .collect::<Vec<_>>();
    let witness = (0..live_lane_count * coeff_count)
        .map(|index| ((index * 5 + 3) % 8) as i8 - 4)
        .collect::<Vec<_>>();
    let common = (0..coeff_count)
        .map(|index| F::from_u64(29 + index as u64))
        .collect::<Vec<_>>();
    let lanes = (0..lane_capacity)
        .map(|index| F::from_u64(41 + 3 * index as u64))
        .collect::<Vec<_>>();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b: 8,
        live_lane_count,
        lane_bits,
        coefficient_bits,
    };
    let mut factored = new_stage2_test_prover(
        F::from_u64(53),
        witness.clone(),
        common.clone(),
        lanes.clone(),
        params,
    );
    let direct = direct_relation_range_image_evaluation(
        F::from_u64(53),
        &witness,
        &common,
        &lanes,
        &vec![F::zero(); witness.len()],
        &params,
    );
    let mut dense_weights = vec![F::zero(); lane_capacity * coeff_count];
    for lane in 0..lane_capacity {
        for coefficient in 0..coeff_count {
            dense_weights[lane * coeff_count + coefficient] = common[coefficient] * lanes[lane];
        }
    }
    let mut dense = RelationRangeImageProver::new(
        F::from_u64(53),
        packed(&witness),
        &stage1_point,
        direct.range_image,
        8,
        RelationWeightOracle::ReducedDense(
            DenseRelationWeights::new(dense_weights, witness.len()).unwrap(),
        ),
        live_lane_count,
        lane_bits,
        coefficient_bits,
        direct.relation,
        PreparedProverLinearTerms::zero(live_lane_count, coeff_count),
        F::zero(),
        None,
    )
    .unwrap();

    let mut claim = factored.input_claim();
    assert_eq!(claim, dense.input_claim());
    for round in 0..lane_bits + coefficient_bits {
        let factored_poly = factored.compute_round_univariate(round, claim);
        let dense_poly = dense.compute_round_univariate(round, claim);
        assert_eq!(dense_poly, factored_poly, "round {round}");
        let challenge = F::from_u64(71 + round as u64);
        claim = factored_poly.evaluate(&challenge);
        factored.ingest_challenge(round, challenge);
        dense.ingest_challenge(round, challenge);
    }
    assert_eq!(dense.final_w_eval(), factored.final_w_eval());
    assert_eq!(dense.expected_final_claim().unwrap(), claim);
    assert_eq!(factored.expected_final_claim().unwrap(), claim);
}

#[test]
fn reduced_dense_oracle_rejects_a_live_domain_that_disagrees_with_the_witness() {
    let lane_bits = 2;
    let coefficient_bits = 2;
    let live_lane_count = 3;
    let coeff_count = 1usize << coefficient_bits;
    let domain_len = (1usize << lane_bits) * coeff_count;
    let witness_len = live_lane_count * coeff_count;
    let stage1_point = vec![F::from_u64(3); lane_bits + coefficient_bits];
    let result = RelationRangeImageProver::new(
        F::one(),
        packed(&vec![0; witness_len]),
        &stage1_point,
        F::zero(),
        4,
        RelationWeightOracle::ReducedDense(
            DenseRelationWeights::new(vec![F::zero(); domain_len], witness_len - 1).unwrap(),
        ),
        live_lane_count,
        lane_bits,
        coefficient_bits,
        F::zero(),
        PreparedProverLinearTerms::zero(live_lane_count, coeff_count),
        F::zero(),
        None,
    );
    assert!(matches!(
        result,
        Err(AkitaError::InvalidSize {
            expected,
            actual
        }) if expected == witness_len && actual == witness_len - 1
    ));
}
