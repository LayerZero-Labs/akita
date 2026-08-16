use super::poly::DensePoly;
use akita_algebra::CyclotomicRing;
use akita_field::Prime128OffsetA7F7 as F;
use akita_field::{Ext2, ExtField, FpExt4, FpExt8};
use akita_types::{
    embed_ring_subfield_vector, tensor_column_partials_from_base_evals,
    tensor_packed_witness_evals, FpExtEncoding,
};

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

fn assert_dense_tensor_projection_matches_reference<E, const D: usize>(num_vars: usize)
where
    E: FpExtEncoding<F>,
{
    let evals = (0..(1usize << num_vars))
        .map(|idx| F::from_u64(17 * idx as u64 + 9))
        .collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(num_vars, D, &evals).unwrap();
    let packed = tensor_packed_witness_evals::<F, E>(num_vars, &evals).unwrap();
    let packed_len = D / E::EXT_DEGREE;
    let expected = packed
        .chunks(packed_len)
        .map(|chunk| {
            let mut padded = chunk.to_vec();
            padded.resize(packed_len, E::zero());
            embed_ring_subfield_vector::<F, E, D>(
                &padded,
                akita_field::AkitaError::InvalidInput("test projection shape".to_string()),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let got = poly.tensor_packed_extension_poly::<E, D>().unwrap();

    assert_eq!(got.ring_coeffs::<D>().unwrap(), expected);
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
}

#[test]
fn dense_tensor_projection_matches_reference_for_every_supported_degree() {
    const D: usize = 16;

    assert_dense_tensor_projection_matches_reference::<F, D>(4);
    assert_dense_tensor_projection_matches_reference::<Ext2<F>, D>(4);
    assert_dense_tensor_projection_matches_reference::<FpExt4<F>, D>(4);
    assert_dense_tensor_projection_matches_reference::<FpExt8<F>, D>(4);
}

#[test]
fn dense_tensor_projection_preserves_transformed_padded_ring_coefficients() {
    assert_dense_tensor_projection_matches_reference::<FpExt8<F>, 16>(3);
}

#[test]
fn dense_constructor_reuses_owned_evaluation_buffer() {
    const D: usize = 64;
    let evals = (0..256).map(F::from_u64).collect::<Vec<_>>();
    let allocation = evals.as_ptr();
    let poly = DensePoly::<F>::from_field_evals(8, D, evals).unwrap();
    assert_eq!(poly.field_coeffs().as_ptr(), allocation);
}
