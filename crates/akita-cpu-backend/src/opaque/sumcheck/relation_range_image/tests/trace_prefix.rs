use super::*;
use std::sync::Arc;

pub(super) fn two_source_linear_terms(
    live_lane_count: usize,
    coeff_count: usize,
) -> (PreparedProverLinearTerms<F>, Vec<F>) {
    let source0 = (0..coeff_count)
        .map(|index| F::from_u64(101 + 3 * index as u64))
        .collect::<Vec<_>>();
    let source1 = (0..coeff_count)
        .map(|index| F::from_u64(211 + 5 * index as u64))
        .collect::<Vec<_>>();
    let mut segments = Vec::with_capacity(2 * live_lane_count);
    let mut terms = Vec::with_capacity(2 * live_lane_count);
    let mut dense = vec![F::zero(); live_lane_count * coeff_count];
    for lane in 0..live_lane_count {
        let target = lane * coeff_count;
        for (source_index, source) in [&source0, &source1].into_iter().enumerate() {
            let factor = F::from_u64(307 + 7 * lane as u64 + 11 * source_index as u64);
            let segment_start = segments.len();
            segments.push(StructuredLinearSegment {
                physical_coefficient_start: target,
                source_coefficient_start: 0,
                coefficient_count: coeff_count,
            });
            terms.push(StructuredLinearTerm {
                factor,
                source_index,
                segment_range: segment_start..segments.len(),
            });
            for coefficient in 0..coeff_count {
                dense[target + coefficient] += factor * source[coefficient];
            }
        }
    }
    let weights = StructuredLinearWeights {
        sources: vec![Arc::from(source0), Arc::from(source1)],
        segments,
        terms,
        physical_field_len: dense.len(),
    };
    let prepared = PreparedProverLinearTerms::from_structured_weights(&weights, coeff_count)
        .expect("two shared structured sources should prepare");
    assert_eq!(prepared.source_count(), 2);
    assert_eq!(prepared.materialize_dense(), dense);
    (prepared, dense)
}

#[test]
fn stage2_two_shared_sources_match_direct_path_through_all_transitions() {
    let lane_bits = 5usize;
    let coefficient_bits = 4usize;
    let live_lane_count = 19usize;
    let coeff_count = 1usize << coefficient_bits;
    let b = 8usize;
    let half = (b / 2) as i8;
    let compact_witness = (0..live_lane_count * coeff_count)
        .map(|index| ((13 * index + 3) % b) as i8 - half)
        .collect::<Vec<_>>();
    let stage1_point = (0..lane_bits + coefficient_bits)
        .map(|index| F::from_u64(401 + 13 * index as u64))
        .collect::<Vec<_>>();
    let common_alpha_factor = (0..coeff_count)
        .map(|index| F::from_u64(503 + 17 * index as u64))
        .collect::<Vec<_>>();
    let relation_lane_weights = (0..1usize << lane_bits)
        .map(|index| F::from_u64(601 + 19 * index as u64))
        .collect::<Vec<_>>();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_lane_count,
        lane_bits,
        coefficient_bits,
    };
    let (structured, dense) = two_source_linear_terms(live_lane_count, coeff_count);
    let mut optimized = new_stage2_test_prover_with_linear_terms(
        F::from_u64(701),
        compact_witness.clone(),
        common_alpha_factor.clone(),
        relation_lane_weights.clone(),
        dense.clone(),
        structured,
        params,
    );
    assert!(optimized.compact_quotient_prefix().is_some());
    let (structured, _) = two_source_linear_terms(live_lane_count, coeff_count);
    let mut direct = new_stage2_test_prover_with_linear_terms(
        F::from_u64(701),
        compact_witness,
        common_alpha_factor,
        relation_lane_weights,
        dense,
        structured,
        params,
    );
    direct.disable_compact_quotient_prefix();

    let mut optimized_claim = optimized.input_claim();
    let mut direct_claim = direct.input_claim();
    assert_eq!(optimized_claim, direct_claim);
    for round in 0..lane_bits + coefficient_bits {
        let optimized_poly = optimized.compute_round_univariate(round, optimized_claim);
        let direct_poly = direct.compute_round_univariate(round, direct_claim);
        assert_eq!(optimized_poly, direct_poly, "mismatch at round {round}");
        let challenge = F::from_u64(809 + 23 * round as u64);
        optimized_claim = optimized_poly.evaluate(challenge);
        direct_claim = direct_poly.evaluate(challenge);
        optimized.ingest_challenge(round, challenge);
        direct.ingest_challenge(round, challenge);
    }
    assert_eq!(optimized_claim, direct_claim);
    assert_eq!(optimized.final_w_eval(), direct.final_w_eval());
    assert_eq!(optimized.expected_final_claim().unwrap(), optimized_claim);
    assert_eq!(direct.expected_final_claim().unwrap(), direct_claim);
}

#[test]
fn stage2_trace_deferred_compact_prefix_matches_direct_path() {
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
    assert!(prover.compact_quotient_prefix().is_some());
    let mut direct = new_stage2_test_prover_with_trace(
        F::from_u64(43),
        w_prefix,
        common_alpha_factor,
        relation_lane_weights,
        trace_compact.clone(),
        params,
    );
    direct.disable_compact_quotient_prefix();
    assert!(direct.compact_quotient_prefix().is_none());

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
        prover_claim = prover_poly.evaluate(challenge);
        direct_claim = direct_poly.evaluate(challenge);
        prover.ingest_challenge(round, challenge);
        direct.ingest_challenge(round, challenge);
    }

    assert_eq!(prover_claim, direct_claim);
    assert_eq!(prover.final_w_eval(), direct.final_w_eval());
}

#[test]
fn stage2_trace_deferred_compact_prefix_matches_padded_reference() {
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
        prefix_claim = prefix_poly.evaluate(challenge);
        padded_claim = padded_poly.evaluate(challenge);
        prefix_prover.ingest_challenge(round, challenge);
        padded_prover.ingest_challenge(round, challenge);
    }

    assert_eq!(prefix_claim, padded_claim);
    assert_eq!(prefix_prover.final_w_eval(), padded_prover.final_w_eval());
}
