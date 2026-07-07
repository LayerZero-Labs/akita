use super::*;

fn fold_compact_flat_reference(w_compact: &[i8], r: F) -> Vec<F> {
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

#[test]
fn fold_witness_compact_to_field_matches_reference() {
    let r = F::from_u64(53);
    let w_live = vec![1i8, 2, 3, 1, 2, 3, 1, 2, 3, 1];
    let fold_lut = AkitaStage2Prover::<F>::build_compact_w_fold_lut(&w_live, r);
    assert_eq!(
        AkitaStage2Prover::<F>::fold_witness_compact_to_field(&w_live, &fold_lut),
        fold_compact_flat_reference(&w_live, r)
    );

    let w_dense = vec![1i8, 2, 3, 1, 2, 3];
    let dense_lut = AkitaStage2Prover::<F>::build_compact_w_fold_lut(&w_dense, r);
    assert_eq!(
        AkitaStage2Prover::<F>::fold_witness_compact_to_field(&w_dense, &dense_lut),
        fold_compact_flat_reference(&w_dense, r)
    );
}

#[test]
fn fold_witness_compact_odd_length_zero_pads_tail() {
    let r = F::from_u64(71);
    let w_compact = vec![1i8, 2, 3];
    let fold_lut = AkitaStage2Prover::<F>::build_compact_w_fold_lut(&w_compact, r);
    let folded = AkitaStage2Prover::<F>::fold_witness_compact_to_field(&w_compact, &fold_lut);
    assert_eq!(folded.len(), 2);
    assert_eq!(folded, fold_compact_flat_reference(&w_compact, r));
}

#[test]
fn fold_relation_weight_flat_zero_pads_live_tail() {
    let r = F::from_u64(59);
    let evals: Vec<F> = (0..10).map(|i| F::from_u64(i + 1)).collect();
    let stage1_point: Vec<F> = (0..4).map(|i| F::from_u64(i + 1)).collect();
    let mut prover = AkitaStage2Prover::new(
        F::from_u64(11),
        vec![0i8; 10],
        &stage1_point,
        F::zero(),
        4,
        evals.clone(),
        F::zero(),
        Stage2Geometry::new(10, 4).unwrap(),
    )
    .unwrap();

    let expected: Vec<F> = (0..evals.len().div_ceil(2))
        .map(|i| {
            let left = 2 * i;
            let a = evals[left];
            let b = evals.get(left + 1).copied().unwrap_or(F::zero());
            a + r * (b - a)
        })
        .collect();
    prover.fold_relation_weight_flat(r);
    assert_eq!(prover.relation_weight.evals(), &expected);
}

#[test]
fn fold_through_two_challenges_matches_sequential_flat_fold() {
    let r0 = F::from_u64(17);
    let r1 = F::from_u64(23);
    let w_compact: Vec<i8> = (0..16).map(|i| ((i % 8) as i8) - 4).collect();
    let relation: Vec<F> = (0..16).map(|i| F::from_u64(i + 3)).collect();

    assert_eq!(
        AkitaStage2Prover::<F>::fold_witness_through_two_challenges(&w_compact, r0, r1),
        AkitaStage2Prover::<F>::fold_witness_field_flat(
            {
                let lut = AkitaStage2Prover::<F>::build_compact_w_fold_lut(&w_compact, r0);
                AkitaStage2Prover::<F>::fold_witness_compact_to_field(&w_compact, &lut)
            },
            r1,
        ),
    );
    assert_eq!(
        AkitaStage2Prover::<F>::fold_relation_weight_through_two_challenges(&relation, r0, r1),
        AkitaStage2Prover::<F>::fold_witness_field_flat(
            AkitaStage2Prover::<F>::fold_witness_field_flat(relation, r0),
            r1,
        ),
    );
}

#[test]
fn fold_witness_field_flat_folds_singleton_against_zero() {
    let r = F::from_u64(41);
    let evals = vec![F::from_u64(7)];
    assert_eq!(
        AkitaStage2Prover::<F>::fold_witness_field_flat(evals.clone(), r),
        AkitaStage2Prover::<F>::fold_relation_field_flat(&evals, r),
    );
}
