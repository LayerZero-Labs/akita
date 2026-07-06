use super::*;

#[test]
fn witness_fold_kind_compact_remaps_coefficient_axis() {
    assert_eq!(
        AkitaStage2Prover::<F>::witness_fold_kind(
            FoldRoundKind::EmbeddedCoefficientAxis,
            true,
        ),
        FoldRoundKind::FlatPair,
    );
    assert_eq!(
        AkitaStage2Prover::<F>::witness_fold_kind(
            FoldRoundKind::EmbeddedCoefficientAxis,
            false,
        ),
        FoldRoundKind::EmbeddedCoefficientAxis,
    );
    assert_eq!(
        AkitaStage2Prover::<F>::witness_fold_kind(
            FoldRoundKind::EmbeddedSegmentAxis,
            true,
        ),
        FoldRoundKind::EmbeddedSegmentAxis,
    );
}

#[test]
fn fold_round_kind_matches_lifecycle_matrix() {
    let coeff_bits = 2usize;
    let live_segments = 5usize;
    let segment_bits = live_segments.next_power_of_two().trailing_zeros() as usize;
    let coeff_len = 1usize << coeff_bits;
    let stage1_point: Vec<F> = (0..(segment_bits + coeff_bits))
        .map(|i| F::from_u64(i as u64 + 1))
        .collect();
    let alpha_evals_coeff: Vec<F> = (0..coeff_len)
        .map(|i| F::from_u64(i as u64 + 3))
        .collect();
    let m_evals_segment: Vec<F> = (0..(1usize << segment_bits))
        .map(|i| F::from_u64(i as u64 + 5))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b: 8,
        live_segments,
        segment_bits,
        coeff_bits,
    };
    let w_compact: Vec<i8> = (0..(live_segments * coeff_len))
        .map(|i| ((i % 8) as i8) - 4)
        .collect();

    let prover = new_stage2_test_prover(
        F::from_u64(7),
        w_compact,
        alpha_evals_coeff,
        m_evals_segment,
        params,
    );
    assert_eq!(
        prover.fold_round_kind(!prover.in_coefficient_round()),
        FoldRoundKind::EmbeddedCoefficientAxis,
    );

    let mut after_coeff = prover;
    after_coeff.rounds_completed = coeff_bits;
    assert_eq!(
        after_coeff.fold_round_kind(!after_coeff.in_coefficient_round()),
        FoldRoundKind::EmbeddedSegmentAxis,
    );

    let mut saturated = after_coeff;
    saturated.live_segments = 1usize << segment_bits;
    assert_eq!(
        saturated.fold_round_kind(!saturated.in_coefficient_round()),
        FoldRoundKind::FlatPair,
    );
}

#[test]
fn fold_witness_polynomial_compact_coefficient_round_matches_flat_pair() {
    let r = F::from_u64(41);
    let w_compact: Vec<i8> = vec![1, -1, 2, 0, 3, 1, -2, 2, 0, 1];
    let fold_lut = AkitaStage2Prover::<F>::build_compact_w_fold_lut(&w_compact, r);
    let witness_kind = AkitaStage2Prover::<F>::witness_fold_kind(
        FoldRoundKind::EmbeddedCoefficientAxis,
        true,
    );
    assert_eq!(witness_kind, FoldRoundKind::FlatPair);
    assert_eq!(
        AkitaStage2Prover::<F>::fold_witness_polynomial(
            WitnessFoldInput::Compact {
                digits: &w_compact,
                fold_lut: &fold_lut,
            },
            witness_kind,
            5,
            2,
        ),
        fold_compact_to_full_reference(&w_compact, r),
    );
}

#[test]
fn fold_witness_flat_compact_odd_length_zero_pads_tail() {
    let r = F::from_u64(71);
    let w_compact = vec![1i8, 2, 3];
    let fold_lut = AkitaStage2Prover::<F>::build_compact_w_fold_lut(&w_compact, r);
    let folded = AkitaStage2Prover::<F>::fold_witness_flat_compact(&w_compact, &fold_lut);
    assert_eq!(folded.len(), 2);
    let w_0 = F::from_i64(1);
    let w_1 = F::from_i64(2);
    let w_2 = F::from_i64(3);
    let w_3 = F::zero();
    assert_eq!(folded[0], w_0 + r * (w_1 - w_0));
    assert_eq!(folded[1], w_2 + r * (w_3 - w_2));
}

#[test]
fn fold_relation_weight_flat_pair_saturated_segment_round() {
    let r = F::from_u64(59);
    let coeff_len = 2usize;
    let live_segments = 8usize;
    let evals: Vec<F> = (0..(live_segments * coeff_len))
        .map(|i| F::from_u64(i as u64 + 1))
        .collect();
    let mut prover = new_stage2_test_prover(
        F::from_u64(11),
        vec![0i8; live_segments * coeff_len],
        vec![F::one(); coeff_len],
        vec![F::one(); live_segments],
        Stage2Params {
            stage1_point: &[F::from_u64(1), F::from_u64(2), F::from_u64(3), F::from_u64(4)],
            b: 4,
            live_segments,
            segment_bits: 3,
            coeff_bits: 1,
        },
    );
    prover.relation_weight = RelationWeightPolynomial::from_live_evals(evals.clone(), evals.len())
        .unwrap();
    prover.relation_coeff_len = coeff_len;
    prover.live_segments = live_segments;
    prover.rounds_completed = 2;

    let expected: Vec<F> = (0..evals.len().div_ceil(2))
        .map(|i| {
            let left = 2 * i;
            let a = evals[left];
            let b = evals.get(left + 1).copied().unwrap_or(F::zero());
            a + r * (b - a)
        })
        .collect();
    prover.fold_relation_weight(r, FoldRoundKind::FlatPair);
    assert_eq!(prover.relation_weight.evals(), &expected);
    assert_eq!(prover.relation_weight_coeff_len(), coeff_len);
}

#[test]
fn fold_witness_full_owned_flat_pair_reuses_allocation() {
    let r = F::from_u64(67);
    let evals: Vec<F> = (0..16).map(|i| F::from_u64(i)).collect();
    let ptr = evals.as_ptr();
    let cap = evals.capacity();
    let folded = AkitaStage2Prover::<F>::fold_witness_full_owned(
        evals,
        FoldRoundKind::FlatPair,
        4,
        4,
        r,
        true,
    );
    assert_eq!(folded.len(), 8);
    assert_eq!(folded.as_ptr(), ptr);
    assert_eq!(folded.capacity(), cap);
}

fn fold_compact_to_full_reference(w_compact: &[i8], r: F) -> Vec<F> {
    (0..w_compact.len().div_ceil(2))
        .map(|j| {
            let w_0 = F::from_i64(w_compact[2 * j] as i64);
            let w_1 = w_compact
                .get(2 * j + 1)
                .copied()
                .map(|w| F::from_i64(w as i64))
                .unwrap_or(F::zero());
            w_0 + r * (w_1 - w_0)
        })
        .collect()
}
