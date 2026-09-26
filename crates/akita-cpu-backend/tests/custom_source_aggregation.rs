#![allow(missing_docs)]

use akita_cpu_backend::custom_source::{aggregate_decompose_fold_witnesses, DecomposeFoldWitness};

const D: usize = 4;

#[test]
fn downstream_batch_kernel_reuses_checked_aggregation() {
    let witnesses = [
        DecomposeFoldWitness::from_centered_rows(vec![[1i32; D]]),
        DecomposeFoldWitness::from_centered_rows(vec![[2i32; D]]),
    ];

    let got = aggregate_decompose_fold_witnesses::<D>(witnesses.iter().cloned().map(Ok)).unwrap();
    assert_eq!(got.centered_coeffs_flat(), &[3; D]);

    let mismatched = [
        DecomposeFoldWitness::from_centered_rows(vec![[1i32; D]]),
        DecomposeFoldWitness::from_centered_rows(vec![[1i32; D]; 2]),
    ];
    assert!(aggregate_decompose_fold_witnesses::<D>(mismatched.into_iter().map(Ok)).is_err());
}
