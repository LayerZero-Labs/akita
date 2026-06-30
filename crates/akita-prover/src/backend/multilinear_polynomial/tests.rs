use super::{MultilinearPolynomial, MultilinearPolynomialBatchView, MultilinearPolynomialView};
use crate::backend::{DenseBatchView, OneHotBatchView};
use crate::compute::{
    BatchDecomposeFoldOutcome, CpuBackend, DecomposeFoldBatchPlan, OpeningBatchKernel,
    RootOpeningSource, RootPolyShape, RootTensorSource, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};
use crate::{DensePoly, OneHotPoly};
use akita_field::{CanonicalField, ExtField, FpExt4, Prime24Offset3};

fn sample_dense<const D: usize>() -> DensePoly<Prime24Offset3, D> {
    let num_vars = 5;
    let evals = (0..(1usize << num_vars))
        .map(|idx| Prime24Offset3::from_canonical_u128_reduced(17 * idx as u128 + 9))
        .collect::<Vec<_>>();
    DensePoly::from_field_evals(num_vars, &evals).unwrap()
}

fn sample_onehot<const D: usize>() -> OneHotPoly<Prime24Offset3, D> {
    OneHotPoly::<Prime24Offset3, D>::new(
        8,
        vec![
            Some(0usize),
            Some(7),
            None,
            Some(3),
            Some(5),
            Some(1),
            None,
            Some(6),
        ],
    )
    .unwrap()
}

fn sample_point<E: ExtField<Prime24Offset3>>(num_vars: usize) -> Vec<E> {
    (0..num_vars)
        .map(|idx| {
            E::from_base_slice(&[
                Prime24Offset3::from_canonical_u128_reduced(5 * idx as u128 + 2),
                Prime24Offset3::from_canonical_u128_reduced(5 * idx as u128 + 3),
                Prime24Offset3::from_canonical_u128_reduced(5 * idx as u128 + 5),
                Prime24Offset3::from_canonical_u128_reduced(5 * idx as u128 + 7),
            ])
        })
        .collect()
}

#[test]
fn multilinear_polynomial_forwards_onehot_chunk_size_from_inner() {
    const D: usize = 16;
    let onehot = OneHotPoly::<Prime24Offset3, D>::new(256, vec![Some(1), None]).unwrap();
    let dense = sample_dense::<D>();
    assert_eq!(
        RootPolyShape::onehot_chunk_size(&MultilinearPolynomial::onehot(onehot)),
        Some(256)
    );
    assert_eq!(
        RootPolyShape::onehot_chunk_size(
            &MultilinearPolynomial::<Prime24Offset3, D, usize>::dense(dense)
        ),
        None
    );
}

#[test]
fn multilinear_kernel_homogeneous_dense_tensor_batch_matches_inner() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let dense0 = sample_dense::<D>();
    let dense1 = sample_dense::<D>();
    let num_vars = RootPolyShape::num_vars(&dense0);
    let wrapped = [
        MultilinearPolynomial::dense(dense0),
        MultilinearPolynomial::dense(dense1),
    ];
    let wrapped_refs = [&wrapped[0], &wrapped[1]];
    let point = sample_point::<E>(num_vars);
    let backend = CpuBackend;

    let inner_refs: Vec<&DensePoly<F, D>> = wrapped
        .iter()
        .map(|poly| match poly {
            MultilinearPolynomial::Dense(dense) => dense,
            MultilinearPolynomial::OneHot(_) => unreachable!(),
        })
        .collect();
    let dense_view = DensePoly::<F, D>::tensor_batch(&inner_refs).unwrap();
    let expected =
        TensorProjectionBatchKernel::<DenseBatchView<'_, F, D>, F, E, D>::column_partials_batch(
            &backend, None, dense_view, &point,
        )
        .unwrap();
    let batch_view = MultilinearPolynomial::<F, D>::tensor_batch(&wrapped_refs).unwrap();
    let got = TensorProjectionBatchKernel::<
        MultilinearPolynomialBatchView<'_, F, D>,
        F,
        E,
        D,
    >::column_partials_batch(&backend, None, batch_view, &point)
    .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn multilinear_kernel_homogeneous_onehot_tensor_batch_matches_inner() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let onehot0 = sample_onehot::<D>();
    let onehot1 = sample_onehot::<D>();
    let num_vars = RootPolyShape::num_vars(&onehot0);
    let wrapped = [
        MultilinearPolynomial::onehot(onehot0),
        MultilinearPolynomial::onehot(onehot1),
    ];
    let wrapped_refs = [&wrapped[0], &wrapped[1]];
    let point = sample_point::<E>(num_vars);
    let backend = CpuBackend;

    let inner_refs: Vec<&OneHotPoly<F, D>> = wrapped
        .iter()
        .map(|poly| match poly {
            MultilinearPolynomial::OneHot(onehot) => onehot,
            MultilinearPolynomial::Dense(_) => unreachable!(),
        })
        .collect();
    let onehot_view = OneHotPoly::<F, D>::tensor_batch(&inner_refs).unwrap();
    let expected =
        TensorProjectionBatchKernel::<OneHotBatchView<'_, F, D>, F, E, D>::column_partials_batch(
            &backend,
            None,
            onehot_view,
            &point,
        )
        .unwrap();
    let batch_view = MultilinearPolynomial::<F, D>::tensor_batch(&wrapped_refs).unwrap();
    let got = TensorProjectionBatchKernel::<
        MultilinearPolynomialBatchView<'_, F, D>,
        F,
        E,
        D,
    >::column_partials_batch(&backend, None, batch_view, &point)
    .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn multilinear_kernel_mixed_batch_column_partials_falls_back_per_poly() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let onehot = sample_onehot::<D>();
    let num_vars = RootPolyShape::num_vars(&onehot);
    let evals = (0..(1usize << num_vars))
        .map(|idx| Prime24Offset3::from_canonical_u128_reduced(17 * idx as u128 + 9))
        .collect::<Vec<_>>();
    let dense = DensePoly::from_field_evals(num_vars, &evals).unwrap();
    let wrapped = [
        MultilinearPolynomial::dense(dense),
        MultilinearPolynomial::onehot(onehot),
    ];
    let wrapped_refs = [&wrapped[0], &wrapped[1]];
    let point = sample_point::<E>(num_vars);
    let backend = CpuBackend;

    let expected = wrapped_refs
        .iter()
        .map(|poly| {
            let view = poly.tensor_view().unwrap();
            TensorProjectionKernel::<MultilinearPolynomialView<'_, F, D>, F, E, D>::column_partials(
                &backend, None, view, &point,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let batch_view = MultilinearPolynomial::<F, D>::tensor_batch(&wrapped_refs).unwrap();
    let got = TensorProjectionBatchKernel::<
        MultilinearPolynomialBatchView<'_, F, D>,
        F,
        E,
        D,
    >::column_partials_batch(&backend, None, batch_view, &point)
    .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn multilinear_kernel_mixed_batch_sparse_linear_combination_returns_none() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let onehot = sample_onehot::<D>();
    let num_vars = RootPolyShape::num_vars(&onehot);
    let evals = (0..(1usize << num_vars))
        .map(|idx| Prime24Offset3::from_canonical_u128_reduced(17 * idx as u128 + 9))
        .collect::<Vec<_>>();
    let dense = DensePoly::from_field_evals(num_vars, &evals).unwrap();
    let wrapped = [
        MultilinearPolynomial::dense(dense),
        MultilinearPolynomial::onehot(onehot),
    ];
    let wrapped_refs = [&wrapped[0], &wrapped[1]];
    let coeffs = vec![E::one(), E::one()];
    let backend = CpuBackend;

    let batch_view = MultilinearPolynomial::<F, D>::tensor_batch(&wrapped_refs).unwrap();
    let got = TensorProjectionBatchKernel::<
        MultilinearPolynomialBatchView<'_, F, D>,
        F,
        E,
        D,
    >::sparse_linear_combination(&backend, None, batch_view, &coeffs)
    .unwrap();
    assert!(got.is_none());
}

#[test]
fn multilinear_mixed_sparse_batch_fold_returns_fallback_per_poly() {
    type F = Prime24Offset3;
    const D: usize = 16;

    let onehot = sample_onehot::<D>();
    let num_vars = RootPolyShape::num_vars(&onehot);
    let evals = (0..(1usize << num_vars))
        .map(|idx| Prime24Offset3::from_canonical_u128_reduced(17 * idx as u128 + 9))
        .collect::<Vec<_>>();
    let dense = DensePoly::from_field_evals(num_vars, &evals).unwrap();
    let wrapped = [
        MultilinearPolynomial::dense(dense),
        MultilinearPolynomial::onehot(onehot),
    ];
    let wrapped_refs = [&wrapped[0], &wrapped[1]];
    let batch_view = MultilinearPolynomial::<F, D>::opening_batch(&wrapped_refs).unwrap();
    let outcome =
        OpeningBatchKernel::<MultilinearPolynomialBatchView<'_, F, D>, F, D>::decompose_fold_batch(
            &CpuBackend,
            None,
            batch_view,
            DecomposeFoldBatchPlan::Sparse {
                challenges: &[],
                block_len: 1,
                num_digits: 1,
                num_digits_fold: 1,
                log_basis: 1,
            },
        )
        .expect("batch fold outcome");
    assert!(matches!(
        outcome,
        BatchDecomposeFoldOutcome::FallbackPerPoly
    ));
}
