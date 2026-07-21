use super::*;

#[test]
fn stage2_trace_two_round_prefix_matches_direct_path() {
    let lane_bits = 5usize;
    let coefficient_bits = 4usize;
    let live_lane_count = 19usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let coeff_count = 1usize << coefficient_bits;
    let w_prefix: Vec<i8> = (0..(live_lane_count * coeff_count))
        .map(|i| ((i * 17 + 5) % b) as i8 - half)
        .collect();
    let trace_compact: Vec<F> = (0..(live_lane_count * coeff_count))
        .map(|i| F::from_u64((19 * i as u64) + 23))
        .collect();
    let stage1_point: Vec<F> = (0..(lane_bits + coefficient_bits))
        .map(|i| F::from_u64((3 * i as u64) + 31))
        .collect();
    let common_alpha_factor: Vec<F> = (0..coeff_count)
        .map(|i| F::from_u64((5 * i as u64) + 37))
        .collect();
    let relation_lane_weights: Vec<F> = (0..(1usize << lane_bits))
        .map(|i| F::from_u64((7 * i as u64) + 41))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_lane_count,
        lane_bits,
        coefficient_bits,
    };

    let mut prover = new_stage2_test_prover_with_trace(
        F::from_u64(43),
        w_prefix.clone(),
        common_alpha_factor.clone(),
        relation_lane_weights.clone(),
        trace_compact.clone(),
        params,
    );
    assert!(prover.can_use_two_round_prefix());
    let mut direct = new_stage2_test_prover_with_trace(
        F::from_u64(43),
        w_prefix,
        common_alpha_factor,
        relation_lane_weights,
        trace_compact.clone(),
        params,
    );
    direct.prefix_r_stage1 = None;
    assert!(!direct.can_use_two_round_prefix());

    let mut prover_claim = prover.input_claim();
    let mut direct_claim = direct.input_claim();
    assert_eq!(prover_claim, direct_claim);
    for round in 0..(lane_bits + coefficient_bits) {
        let prover_poly = prover.compute_round_univariate(round, prover_claim);
        let direct_poly = direct.compute_round_univariate(round, direct_claim);
        assert_eq!(
            prover_poly, direct_poly,
            "trace two-round prefix mismatch at round {round}"
        );

        let challenge = F::from_u64((11 * round as u64) + 47);
        prover_claim = prover_poly.evaluate(&challenge);
        direct_claim = direct_poly.evaluate(&challenge);
        prover.ingest_challenge(round, challenge);
        direct.ingest_challenge(round, challenge);
    }

    assert_eq!(prover_claim, direct_claim);
    assert_eq!(prover.final_w_eval(), direct.final_w_eval());
}

#[test]
fn stage2_trace_two_round_prefix_matches_padded_reference() {
    let lane_bits = 5usize;
    let coefficient_bits = 4usize;
    let live_lane_count = 19usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let coeff_count = 1usize << coefficient_bits;
    let w_prefix: Vec<i8> = (0..(live_lane_count * coeff_count))
        .map(|i| ((i * 23 + 7) % b) as i8 - half)
        .collect();
    let trace_compact: Vec<F> = (0..(live_lane_count * coeff_count))
        .map(|i| F::from_u64((29 * i as u64) + 53))
        .collect();
    let w_padded = pad_compact_witness(&w_prefix, live_lane_count, lane_bits, coefficient_bits);
    let trace_padded =
        pad_trace_compact(&trace_compact, live_lane_count, lane_bits, coefficient_bits);
    let stage1_point: Vec<F> = (0..(lane_bits + coefficient_bits))
        .map(|i| F::from_u64((13 * i as u64) + 59))
        .collect();
    let common_alpha_factor: Vec<F> = (0..coeff_count)
        .map(|i| F::from_u64((17 * i as u64) + 61))
        .collect();
    let relation_lane_weights: Vec<F> = (0..(1usize << lane_bits))
        .map(|i| F::from_u64((19 * i as u64) + 67))
        .collect();

    let mut prefix_prover = new_stage2_test_prover_with_trace(
        F::from_u64(71),
        w_prefix,
        common_alpha_factor.clone(),
        relation_lane_weights.clone(),
        trace_compact,
        Stage2Params {
            stage1_point: &stage1_point,
            b,
            live_lane_count,
            lane_bits,
            coefficient_bits,
        },
    );
    let mut padded_prover = new_stage2_test_prover_with_trace(
        F::from_u64(71),
        w_padded,
        common_alpha_factor,
        relation_lane_weights,
        trace_padded,
        Stage2Params {
            stage1_point: &stage1_point,
            b,
            live_lane_count: 1usize << lane_bits,
            lane_bits,
            coefficient_bits,
        },
    );

    let mut prefix_claim = prefix_prover.input_claim();
    let mut padded_claim = padded_prover.input_claim();
    assert_eq!(prefix_claim, padded_claim);
    for round in 0..(lane_bits + coefficient_bits) {
        let prefix_poly = prefix_prover.compute_round_univariate(round, prefix_claim);
        let padded_poly = padded_prover.compute_round_univariate(round, padded_claim);
        assert_eq!(
            prefix_poly, padded_poly,
            "trace prefix/padded mismatch at round {round}"
        );

        let challenge = F::from_u64((23 * round as u64) + 73);
        prefix_claim = prefix_poly.evaluate(&challenge);
        padded_claim = padded_poly.evaluate(&challenge);
        prefix_prover.ingest_challenge(round, challenge);
        padded_prover.ingest_challenge(round, challenge);
    }

    assert_eq!(prefix_claim, padded_claim);
    assert_eq!(prefix_prover.final_w_eval(), padded_prover.final_w_eval());
}

#[test]
fn stage2_trace_round2_cached_poly_matches_reference() {
    let lane_bits = 4usize;
    let coefficient_bits = 4usize;
    let live_lane_count = 11usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let coeff_count = 1usize << coefficient_bits;
    let w_prefix: Vec<i8> = (0..(live_lane_count * coeff_count))
        .map(|i| ((i * 31 + 11) % b) as i8 - half)
        .collect();
    let trace_compact: Vec<F> = (0..(live_lane_count * coeff_count))
        .map(|i| F::from_u64((37 * i as u64) + 79))
        .collect();
    let stage1_point: Vec<F> = (0..(lane_bits + coefficient_bits))
        .map(|i| F::from_u64((29 * i as u64) + 83))
        .collect();
    let common_alpha_factor: Vec<F> = (0..coeff_count)
        .map(|i| F::from_u64((31 * i as u64) + 89))
        .collect();
    let relation_lane_weights: Vec<F> = (0..(1usize << lane_bits))
        .map(|i| F::from_u64((37 * i as u64) + 97))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_lane_count,
        lane_bits,
        coefficient_bits,
    };

    let mut prover = new_stage2_test_prover_with_trace(
        F::from_u64(101),
        w_prefix.clone(),
        common_alpha_factor.clone(),
        relation_lane_weights.clone(),
        trace_compact.clone(),
        params,
    );
    let round0 = prover.compute_round_univariate(0, prover.input_claim());
    let r0 = F::from_u64(103);
    prover.ingest_challenge(0, r0);
    let round1 = prover.compute_round_univariate(1, round0.evaluate(&r0));
    let r1 = F::from_u64(107);

    let expected_w_full = AkitaStage2Prover::<F>::fold_compact_to_round2(
        &w_prefix,
        live_lane_count,
        coeff_count,
        r0,
        r1,
    );
    let expected_alpha_round2 =
        AkitaStage2Prover::<F>::fold_alpha_to_round2(&common_alpha_factor, r0, r1);
    let mut expected_trace = PreparedProverEvaluationTrace::from_dense(
        trace_compact.clone(),
        live_lane_count,
        coeff_count,
    );
    expected_trace.fold_two_coefficients(r0, r1);
    let expected_relation_lane_weights = prover.relation_lane_weights.clone();

    let mut expected = new_stage2_test_prover_with_trace(
        F::from_u64(101),
        w_prefix,
        common_alpha_factor,
        relation_lane_weights,
        trace_compact.clone(),
        params,
    );
    let expected_round0 = expected.compute_round_univariate(0, expected.input_claim());
    assert_eq!(expected_round0, round0);
    expected.ingest_challenge(0, r0);
    let expected_round1 = expected.compute_round_univariate(1, expected_round0.evaluate(&r0));
    assert_eq!(expected_round1, round1);
    expected.prev_norm_claim = expected
        .prev_norm_poly
        .as_ref()
        .expect("round1 norm poly should be cached")
        .evaluate(&r1);
    expected.split_eq.bind(r1);
    expected.w_table = WTable::Full(expected_w_full.clone());
    expected.common_alpha_factor = expected_alpha_round2.clone();
    expected.evaluation_trace = expected_trace;
    expected.rounds_completed = 2;
    expected.relation_lane_weights = expected_relation_lane_weights.clone();
    let expected_round2 = expected.compute_current_round_poly_from_state();

    prover.ingest_challenge(1, r1);

    match &prover.w_table {
        WTable::Full(w_full) => assert_eq!(w_full, &expected_w_full),
        WTable::Compact(_) => {
            panic!("expected fused trace stage2 transition to materialize full table")
        }
    }
    assert_eq!(prover.common_alpha_factor, expected_alpha_round2);
    let expected_trace_round2 = trace_compact
        .chunks_exact(4)
        .map(|quad| {
            AkitaStage2Prover::<F>::direct_fold_e_quad_to_round2(
                quad[0], quad[1], quad[2], quad[3], r0, r1,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        prover.evaluation_trace.materialize_dense(),
        expected_trace_round2,
        "two-round handoff must preserve the folded trace"
    );
    assert_eq!(prover.relation_lane_weights, expected_relation_lane_weights);
    assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round2));
}
