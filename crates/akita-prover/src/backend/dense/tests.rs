use super::poly::DensePoly;
use crate::compute::DirectRootWitnessSource;
use akita_algebra::CyclotomicRing;
use akita_field::Prime128OffsetA7F7 as F;
use akita_field::{ExtField, FpExt4};
use akita_types::{
    tensor_column_partials_from_base_evals, tensor_packed_witness_evals, CleartextWitnessProof,
};

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

#[test]
fn ring_fold_matches_dense_multiplication_reference() {
    const D: usize = 8;
    let coeffs = (0..4).map(|idx| ring::<D>(10 * idx)).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_ring_coeffs(coeffs.clone());
    let scalars = vec![ring::<D>(100), ring::<D>(200)];
    let got = poly.fold_blocks_ring(&scalars, 2);
    let expected = coeffs
        .chunks(2)
        .map(|block| {
            block
                .iter()
                .zip(scalars.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (coeff, scalar)| {
                    acc + (*coeff * *scalar)
                })
        })
        .collect::<Vec<_>>();

    assert_eq!(got, expected);
}

#[test]
fn dense_direct_witness_is_field_elements() {
    const D: usize = 8;
    let num_vars = 4;
    let evals = (0..(1usize << num_vars))
        .map(|idx| F::from_u64(idx as u64 + 1))
        .collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(num_vars, D, &evals).unwrap();
    let witness =
        <DensePoly<F> as DirectRootWitnessSource<F, D>>::direct_root_witness(&poly).unwrap();
    assert!(matches!(witness, CleartextWitnessProof::FieldElements(_)));
}

#[test]
fn dense_tensor_opening_methods_match_flat_reference() {
    const D: usize = 8;
    type E = FpExt4<F>;

    let num_vars = 5;
    let evals = (0..(1usize << num_vars))
        .map(|idx| F::from_u64(17 * idx as u64 + 9))
        .collect::<Vec<_>>();
    let point = (0..num_vars)
        .map(|idx| {
            E::from_base_slice(&[
                F::from_u64(idx as u64 + 2),
                F::from_u64(3 * idx as u64 + 4),
                F::from_u64(5 * idx as u64 + 6),
                F::from_u64(7 * idx as u64 + 8),
            ])
        })
        .collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(num_vars, D, &evals).unwrap();

    let expected_partials =
        tensor_column_partials_from_base_evals::<F, E>(num_vars, &evals, &point).unwrap();
    let got_partials = poly
        .tensor_extension_column_partials::<E, D>(&point)
        .unwrap();
    assert_eq!(got_partials, expected_partials);

    let expected_packed = tensor_packed_witness_evals::<F, E>(num_vars, &evals).unwrap();
    let got_packed = poly.tensor_packed_extension_evals::<E, D>().unwrap();
    assert_eq!(got_packed, expected_packed);
}
