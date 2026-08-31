use super::{MultilinearPolynomial, MultilinearPolynomialBatchView, MultilinearPolynomialView};
use crate::backend::OneHotView;
use crate::compute::{
    BatchDecomposeFoldOutcome, CommitInnerPlan, ComputeBackendSetup, CpuBackend,
    DecomposeFoldBatchPlan, OpeningBatchKernel, RootCommitKernel, RootCommitSource,
    RootOpeningSource, RootPolyShape,
};
use crate::{AkitaProverSetup, DensePoly, OneHotPoly};
use akita_types::{AkitaSetupSeed, SetupMatrixCapacity};
use jolt_field::{CanonicalEncoding, Prime24Offset3};

fn sample_dense<const D: usize>() -> DensePoly<Prime24Offset3> {
    let num_vars = 5;
    let evals = (0..(1usize << num_vars))
        .map(|idx| Prime24Offset3::from_u128_reduced(17 * idx as u128 + 9))
        .collect::<Vec<_>>();
    DensePoly::from_field_evals(num_vars, &evals).unwrap()
}

fn sample_onehot<const D: usize>() -> OneHotPoly<Prime24Offset3> {
    OneHotPoly::<Prime24Offset3>::new(
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

#[test]
fn multilinear_polynomial_forwards_onehot_chunk_size_from_inner() {
    const D: usize = 16;
    let onehot = OneHotPoly::<Prime24Offset3>::new(256, vec![Some(1), None]).unwrap();
    let dense = sample_dense::<D>();
    assert_eq!(
        RootPolyShape::<Prime24Offset3, D>::onehot_chunk_size(&MultilinearPolynomial::<
            Prime24Offset3,
            usize,
        >::onehot(onehot)),
        Some(256)
    );
    assert_eq!(
        RootPolyShape::<Prime24Offset3, D>::onehot_chunk_size(&MultilinearPolynomial::<
            Prime24Offset3,
            usize,
        >::dense(dense)),
        None
    );
}

#[test]
fn multilinear_onehot_group_commit_matches_inner_kernel() {
    type F = Prime24Offset3;
    const D: usize = 16;

    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        8,
        1,
        SetupMatrixCapacity {
            num_field_elements: 4096,
        },
        AkitaSetupSeed::DEFAULT,
    )
    .unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let plan = CommitInnerPlan {
        n_a: 2,
        num_positions_per_block: 2,
        num_digits_inner: 1,
        log_basis_inner: 2,
    };

    let inner = [sample_onehot::<D>(), sample_onehot::<D>()];
    let inner_views = inner
        .iter()
        .map(RootCommitSource::<F, D>::commit_view)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let expected = RootCommitKernel::<OneHotView<'_, F, D>, F, D>::commit_inner_group(
        &CpuBackend::DEFAULT,
        &prepared,
        inner_views,
        plan,
    )
    .unwrap();

    let wrapped = inner.map(MultilinearPolynomial::onehot);
    let wrapped_views = wrapped
        .iter()
        .map(RootCommitSource::<F, D>::commit_view)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let got = RootCommitKernel::<MultilinearPolynomialView<'_, F, D>, F, D>::commit_inner_group(
        &CpuBackend::DEFAULT,
        &prepared,
        wrapped_views,
        plan,
    )
    .unwrap();
    assert_eq!(got.len(), expected.len());
    for (got, expected) in got.iter().zip(&expected) {
        assert_eq!(got.inner_rows.ring_dim(), expected.inner_rows.ring_dim());
        assert_eq!(got.inner_rows.coeffs(), expected.inner_rows.coeffs());
    }
}

#[test]
fn multilinear_mixed_sparse_batch_fold_returns_fallback_per_poly() {
    type F = Prime24Offset3;
    const D: usize = 16;

    let onehot = sample_onehot::<D>();
    let num_vars = RootPolyShape::<F, D>::num_vars(&onehot);
    let evals = (0..(1usize << num_vars))
        .map(|idx| Prime24Offset3::from_u128_reduced(17 * idx as u128 + 9))
        .collect::<Vec<_>>();
    let dense = DensePoly::from_field_evals(num_vars, &evals).unwrap();
    let wrapped = [
        MultilinearPolynomial::dense(dense),
        MultilinearPolynomial::onehot(onehot),
    ];
    let wrapped_refs = [&wrapped[0], &wrapped[1]];
    let batch_view =
        <MultilinearPolynomial<F> as RootOpeningSource<F, D>>::opening_batch(&wrapped_refs)
            .unwrap();
    let outcome =
        OpeningBatchKernel::<MultilinearPolynomialBatchView<'_, F, D>, F, D>::decompose_fold_batch(
            &CpuBackend::DEFAULT,
            None,
            batch_view,
            DecomposeFoldBatchPlan::Sparse {
                challenges: &[],
                num_positions_per_block: 1,
                num_digits: 1,
                log_basis: 1,
            },
        )
        .expect("batch fold outcome");
    assert!(matches!(
        outcome,
        BatchDecomposeFoldOutcome::FallbackPerPoly
    ));
}
