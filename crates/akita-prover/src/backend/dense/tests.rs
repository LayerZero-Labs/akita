use super::poly::DensePoly;
use crate::compute::DirectRootWitnessSource;
use crate::DecomposeFoldWitness;
use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallenges};
use akita_field::Prime128OffsetA7F7 as F;
use akita_field::{ExtField, FieldCore, FpExt4};
use akita_types::{
    tensor_column_partials_from_base_evals, tensor_packed_witness_evals, CleartextWitnessProof,
};

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

fn tensor_oracle_challenges<const D: usize>() -> TensorChallenges {
    TensorChallenges {
        left: vec![
            SparseChallenge {
                positions: vec![0],
                coeffs: vec![1],
            },
            SparseChallenge {
                positions: vec![(D - 1) as u32],
                coeffs: vec![1],
            },
            SparseChallenge {
                positions: vec![2],
                coeffs: vec![-1],
            },
            SparseChallenge {
                positions: vec![5],
                coeffs: vec![1],
            },
        ],
        right: vec![
            SparseChallenge {
                positions: vec![1],
                coeffs: vec![1],
            },
            SparseChallenge {
                positions: vec![3],
                coeffs: vec![-1],
            },
            SparseChallenge {
                positions: vec![0],
                coeffs: vec![1],
            },
            SparseChallenge {
                positions: vec![4],
                coeffs: vec![1],
            },
        ],
        left_len: 2,
        right_len: 2,
        num_claims: 2,
    }
}

fn aggregate_witnesses<Fp: FieldCore, const D: usize>(
    witnesses: &[DecomposeFoldWitness<Fp, D>],
) -> DecomposeFoldWitness<Fp, D> {
    let mut acc = witnesses[0].clone();
    for witness in &witnesses[1..] {
        for (dst, src) in acc
            .z_folded_rings
            .iter_mut()
            .zip(witness.z_folded_rings.iter())
        {
            *dst += *src;
        }
        for (dst, src) in acc
            .centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs.iter())
        {
            for k in 0..D {
                dst[k] += src[k];
            }
        }
    }
    acc.centered_inf_norm = acc
        .centered_coeffs
        .iter()
        .flat_map(|coeffs| coeffs.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);
    acc
}

#[test]
fn ring_fold_matches_dense_multiplication_reference() {
    const D: usize = 8;
    let coeffs = (0..4).map(|idx| ring::<D>(10 * idx)).collect::<Vec<_>>();
    let poly = DensePoly::<F, D>::from_ring_coeffs(coeffs.clone());
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
fn dense_tensor_decompose_fold_matches_expanded_reference() {
    const D: usize = 16;
    let block_len = 2;
    let num_digits = 2;
    let log_basis = 3;
    let tensor = tensor_oracle_challenges::<D>();
    let polys = [
        DensePoly::<F, D>::from_ring_coeffs((0..8).map(|idx| ring::<D>(10 * idx)).collect()),
        DensePoly::<F, D>::from_ring_coeffs((0..8).map(|idx| ring::<D>(100 + 7 * idx)).collect()),
    ];
    let expanded = tensor
        .expand_integer::<D>()
        .unwrap()
        .into_iter()
        .map(|challenge| challenge.try_to_sparse_i8().unwrap())
        .collect::<Vec<_>>();

    let expected = aggregate_witnesses(
        &polys
            .iter()
            .zip(expanded.chunks(4))
            .map(|(poly, challenges)| {
                poly.decompose_fold(challenges, block_len, num_digits, log_basis)
            })
            .collect::<Vec<_>>(),
    );
    let poly_refs = polys.iter().collect::<Vec<_>>();
    let got = DensePoly::<F, D>::decompose_fold_tensor_batched(
        &poly_refs, &tensor, block_len, num_digits, log_basis,
    )
    .unwrap()
    .unwrap();

    assert_eq!(got, expected);
}

#[test]
fn dense_direct_witness_is_field_elements() {
    const D: usize = 8;
    let num_vars = 4;
    let evals = (0..(1usize << num_vars))
        .map(|idx| F::from_u64(idx as u64 + 1))
        .collect::<Vec<_>>();
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
    let witness =
        <DensePoly<F, D> as DirectRootWitnessSource<F, D>>::direct_root_witness(&poly).unwrap();
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
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();

    let expected_partials =
        tensor_column_partials_from_base_evals::<F, E>(num_vars, &evals, &point).unwrap();
    let got_partials = poly.tensor_extension_column_partials::<E>(&point).unwrap();
    assert_eq!(got_partials, expected_partials);

    let expected_packed = tensor_packed_witness_evals::<F, E>(num_vars, &evals).unwrap();
    let got_packed = poly.tensor_packed_extension_evals::<E>().unwrap();
    assert_eq!(got_packed, expected_packed);
}
