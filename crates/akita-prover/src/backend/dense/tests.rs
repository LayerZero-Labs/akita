use super::*;
use akita_challenges::{SparseChallenge, TensorChallengeSet, TensorChallenges};
use akita_field::fields::{TowerBasisFp4, TwoNr, UnitNr};
use akita_field::Prime128OffsetA7F7 as F;
use akita_sumcheck::{tensor_column_partials_from_base_evals, tensor_packed_witness_evals};

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

fn dense_poly<const D: usize>(num_rings: usize, seed: u64) -> DensePoly<F, D> {
    let coeffs = (0..num_rings)
        .map(|ring_idx| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|coeff_idx| {
                F::from_u64(((seed + 7 * ring_idx as u64 + 11 * coeff_idx as u64) % 23) + 1)
            }))
        })
        .collect::<Vec<_>>();
    DensePoly::<F, D>::from_ring_coeffs(coeffs)
}

fn dense_tensor_challenges() -> TensorChallengeSet {
    TensorChallengeSet::new(
        vec![
            SparseChallenge {
                positions: vec![0, 3],
                coeffs: vec![1, -1],
            },
            SparseChallenge {
                positions: vec![2],
                coeffs: vec![1],
            },
            SparseChallenge {
                positions: vec![1, 4],
                coeffs: vec![-1, 1],
            },
            SparseChallenge {
                positions: vec![3],
                coeffs: vec![2],
            },
        ],
        vec![
            SparseChallenge {
                positions: vec![0],
                coeffs: vec![1],
            },
            SparseChallenge {
                positions: vec![5, 6],
                coeffs: vec![1, -1],
            },
            SparseChallenge {
                positions: vec![6],
                coeffs: vec![-1],
            },
            SparseChallenge {
                positions: vec![2, 7],
                coeffs: vec![1, 1],
            },
        ],
        2,
        2,
        2,
    )
    .unwrap()
}

fn aggregate_witnesses<const D: usize>(
    witnesses: &[DecomposeFoldWitness<F, D>],
) -> DecomposeFoldWitness<F, D> {
    let (first, rest) = witnesses
        .split_first()
        .expect("aggregate_witnesses requires at least one witness");
    let mut z_pre = first.z_pre.clone();
    let mut centered_coeffs = first.centered_coeffs.clone();
    for witness in rest {
        for (dst, src) in z_pre.iter_mut().zip(witness.z_pre.iter()) {
            *dst += *src;
        }
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs.iter())
        {
            for k in 0..D {
                dst[k] += src[k];
            }
        }
    }
    let centered_inf_norm = centered_coeffs
        .iter()
        .flat_map(|coeffs| coeffs.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);
    DecomposeFoldWitness {
        z_pre,
        centered_coeffs,
        centered_inf_norm,
    }
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
fn dense_tensor_opening_methods_match_flat_reference() {
    const D: usize = 8;
    type E = TowerBasisFp4<F, TwoNr, UnitNr>;

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

#[test]
fn tensor_direct_single_digit_matches_expanded_integer_reference() {
    const D: usize = 8;
    let block_len = 4;
    let num_digits = 1;
    let log_basis = 6;
    let polys = [dense_poly::<D>(16, 3), dense_poly::<D>(16, 19)];
    let tensor = dense_tensor_challenges();
    let challenges = TensorChallenges::Tensor(tensor.clone());
    let expanded = challenges.expand_integer::<D>().unwrap();
    let expected = aggregate_witnesses(
        &polys
            .iter()
            .zip(expanded.chunks(4))
            .map(|(poly, poly_challenges)| {
                poly.decompose_fold(poly_challenges, block_len, num_digits, log_basis)
            })
            .collect::<Vec<_>>(),
    );

    let poly_refs = polys.iter().collect::<Vec<_>>();
    let got = <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
        &poly_refs,
        &TensorChallenges::Tensor(tensor),
        block_len,
        num_digits,
        log_basis,
    )
    .unwrap()
    .expect("dense tensor path should apply");

    assert_eq!(got, expected);
}

#[test]
fn tensor_direct_multi_digit_partial_blocks_match_expanded_integer_reference() {
    const D: usize = 8;
    let block_len = 3;
    let num_digits = 2;
    let log_basis = 4;
    let polys = [dense_poly::<D>(10, 5), dense_poly::<D>(11, 23)];
    let tensor = dense_tensor_challenges();
    let challenges = TensorChallenges::Tensor(tensor.clone());
    let expanded = challenges.expand_integer::<D>().unwrap();
    let expected = aggregate_witnesses(
        &polys
            .iter()
            .zip(expanded.chunks(4))
            .map(|(poly, poly_challenges)| {
                poly.decompose_fold(poly_challenges, block_len, num_digits, log_basis)
            })
            .collect::<Vec<_>>(),
    );

    let poly_refs = polys.iter().collect::<Vec<_>>();
    let got = <DensePoly<F, D> as AkitaPolyOps<F, D>>::decompose_fold_tensor_batched(
        &poly_refs,
        &TensorChallenges::Tensor(tensor),
        block_len,
        num_digits,
        log_basis,
    )
    .unwrap()
    .expect("dense tensor path should apply");

    assert_eq!(got, expected);
}
