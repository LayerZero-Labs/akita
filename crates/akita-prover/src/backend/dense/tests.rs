use super::poly::DensePoly;
use akita_algebra::CyclotomicRing;
use akita_field::Prime128OffsetA7F7 as F;
use akita_field::{Ext2, ExtField, FpExt4};
use akita_types::{
    embed_ring_subfield_vector, tensor_column_partials_from_base_evals, tensor_packed_witness_evals,
};

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

#[test]
fn ring_fold_matches_dense_multiplication_reference() {
    const D: usize = 8;
    let coeffs = (0..2).map(|idx| ring::<D>(10 * idx)).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_ring_coeffs(coeffs.clone());
    let scalars = vec![
        ring::<D>(100),
        ring::<D>(200),
        ring::<D>(300),
        ring::<D>(400),
    ];
    let got = poly.fold_blocks_ring(&scalars, 4);
    let expected = coeffs
        .chunks(4)
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

    let packed_len = D / <E as ExtField<F>>::EXT_DEGREE;
    let expected_projection = expected_packed
        .chunks(packed_len)
        .map(|chunk| {
            embed_ring_subfield_vector::<F, E, D>(
                chunk,
                akita_field::AkitaError::InvalidInput("test projection shape".to_string()),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let got_projection = poly.tensor_packed_extension_poly::<E, D>().unwrap();
    assert_eq!(
        got_projection.ring_coeffs::<D>().unwrap(),
        expected_projection
    );

    type E2 = Ext2<F>;
    let expected_packed_e2 = tensor_packed_witness_evals::<F, E2>(num_vars, &evals).unwrap();
    let expected_projection_e2 = expected_packed_e2
        .chunks(D / <E2 as ExtField<F>>::EXT_DEGREE)
        .map(|chunk| {
            embed_ring_subfield_vector::<F, E2, D>(
                chunk,
                akita_field::AkitaError::InvalidInput("test projection shape".to_string()),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let got_projection_e2 = poly.tensor_packed_extension_poly::<E2, D>().unwrap();
    assert_eq!(
        got_projection_e2.ring_coeffs::<D>().unwrap(),
        expected_projection_e2
    );
}

#[test]
fn dense_constructor_reuses_owned_evaluation_buffer() {
    const D: usize = 64;
    let evals = (0..256).map(F::from_u64).collect::<Vec<_>>();
    let allocation = evals.as_ptr();
    let poly = DensePoly::<F>::from_field_evals(8, D, evals).unwrap();
    assert_eq!(poly.field_coeffs().as_ptr(), allocation);
}
