#![allow(clippy::indexing_slicing, clippy::unwrap_used)]

use core::fmt::Debug;

use jolt_field::{Field, Prime128Offset275, Prime31Offset19, Zero};
use rand::{rngs::StdRng, RngCore, SeedableRng};

use super::{exact_convolution, SetupAdjointWeight, TrinomialAdjoint};
use crate::ring::trinomial::{
    signed::{signed_trinomial_product, signed_trinomial_remainder},
    MinusTrinomial, PlusTrinomial, TrinomialModulus, TrinomialRing,
};

const DEGREE: usize = 648;

fn random<F: Field>(rng: &mut StdRng) -> F {
    F::from_u128(u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64()))
}

fn signed<F: Field>(value: i128) -> F {
    let magnitude = F::from_u128(value.unsigned_abs());
    if value < 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn dot<F: Field>(lhs: &[F], rhs: &[F]) -> F {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .fold(F::zero(), |sum, (&a, &b)| sum + a * b)
}

fn schoolbook<F: Field>(lhs: &[F], rhs: &[F]) -> Vec<F> {
    let mut output = vec![F::zero(); lhs.len() + rhs.len() - 1];
    for (i, &a) in lhs.iter().enumerate() {
        for (j, &b) in rhs.iter().enumerate() {
            output[i + j] += a * b;
        }
    }
    output
}

/// The existing signed reducer supplies each column of the dense tensor K.
fn dense_g<F: Field, const D: usize, M: TrinomialModulus>(rho: &[F]) -> Vec<F> {
    let mut monomial = vec![0i128; 2 * D - 1];
    (0..monomial.len())
        .map(|exponent| {
            if exponent > 0 {
                monomial[exponent - 1] = 0;
            }
            monomial[exponent] = 1;
            let remainder = signed_trinomial_remainder::<D, M>(&monomial).unwrap();
            rho.iter()
                .zip(remainder)
                .fold(F::zero(), |sum, (&coefficient, basis)| {
                    sum + coefficient * signed::<F>(basis)
                })
        })
        .collect()
}

fn dense_h<F: Field, const D: usize, M: TrinomialModulus>(rho: &[F], v: &[F]) -> Vec<F> {
    let g = dense_g::<F, D, M>(rho);
    (0..D)
        .map(|s| {
            v.iter()
                .enumerate()
                .fold(F::zero(), |sum, (k, &value)| sum + value * g[s + k])
        })
        .collect()
}

fn full_degree_identity<F: Field + Debug>(seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let adjoint = TrinomialAdjoint::<DEGREE, MinusTrinomial>::new().unwrap();
    for _ in 0..3 {
        let rho: Vec<F> = (0..DEGREE).map(|_| random(&mut rng)).collect();
        let a: Vec<i128> = (0..DEGREE)
            .map(|_| i128::from(rng.next_u32() % 17) - 8)
            .collect();
        let v: Vec<i128> = (0..DEGREE)
            .map(|_| i128::from(rng.next_u32() % 17) - 8)
            .collect();
        let product = signed_trinomial_product::<DEGREE, MinusTrinomial>(&a, &v).unwrap();
        let a: Vec<F> = a.into_iter().map(signed).collect();
        let v: Vec<F> = v.into_iter().map(signed).collect();
        let product: Vec<F> = product.into_iter().map(signed).collect();
        let h = adjoint.apply(&rho, &v).unwrap();
        assert_eq!(dot(&rho, &product), dot(&a, &h));
    }
    // Full-field inputs also exercise wraparound in both field implementations.
    let rho: Vec<F> = (0..DEGREE).map(|_| random(&mut rng)).collect();
    let a: Vec<F> = (0..DEGREE).map(|_| random(&mut rng)).collect();
    let v: Vec<F> = (0..DEGREE).map(|_| random(&mut rng)).collect();
    let h = adjoint.apply(&rho, &v).unwrap();
    assert_eq!(h, dense_h::<F, DEGREE, MinusTrinomial>(&rho, &v));
    let a_ring = TrinomialRing::<F, DEGREE, MinusTrinomial>::try_from_vec(a.clone()).unwrap();
    let v_ring = TrinomialRing::<F, DEGREE, MinusTrinomial>::try_from_vec(v).unwrap();
    let product = a_ring.schoolbook_mul(&v_ring).unwrap();
    assert_eq!(dot(&rho, product.coefficients()), dot(&a, &h));
}

#[test]
fn full_degree_adjoint_matches_signed_products_and_dense_field_definition() {
    full_degree_identity::<Prime128Offset275>(0x648_275);
    full_degree_identity::<Prime31Offset19>(0x648_019);
}

fn monomials<F: Field + Debug, const D: usize, M: TrinomialModulus>(seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let rho: Vec<F> = (0..D).map(|_| random(&mut rng)).collect();
    let adjoint = TrinomialAdjoint::<D, M>::new().unwrap();
    let g = adjoint.extend_row(&rho).unwrap();
    assert_eq!(g.len(), 2 * D - 1);
    assert_eq!(g, dense_g::<F, D, M>(&rho));
    // Exhaust all tensor slices at toy sizes, including both reduction signs.
    if D <= 8 {
        for t in 0..D {
            let mut rho = vec![F::zero(); D];
            rho[t] = F::one();
            assert_eq!(adjoint.extend_row(&rho).unwrap(), dense_g::<F, D, M>(&rho));
            for k in 0..D {
                let mut v = vec![F::zero(); D];
                v[k] = F::one();
                assert_eq!(
                    adjoint.apply(&rho, &v).unwrap(),
                    dense_h::<F, D, M>(&rho, &v)
                );
            }
        }
    }
}

#[test]
fn row_extension_and_toy_dense_tensor_exhaust_monomials() {
    monomials::<Prime128Offset275, DEGREE, MinusTrinomial>(0x1295);
    monomials::<Prime31Offset19, DEGREE, MinusTrinomial>(0x1296);
    monomials::<Prime128Offset275, 2, MinusTrinomial>(2);
    monomials::<Prime31Offset19, 4, MinusTrinomial>(4);
    monomials::<Prime128Offset275, 6, PlusTrinomial>(6);
    monomials::<Prime31Offset19, 8, PlusTrinomial>(8);
}

fn convolution_checks<F: Field + Debug>(seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    for length in [1, 2, 3, 7, 17, 33, 81, DEGREE] {
        let lhs: Vec<F> = (0..length).map(|_| random(&mut rng)).collect();
        let rhs: Vec<F> = (0..length).map(|_| random(&mut rng)).collect();
        assert_eq!(
            exact_convolution(&lhs, &rhs).unwrap(),
            schoolbook(&lhs, &rhs)
        );
    }
}

#[test]
fn exact_convolution_matches_schoolbook_for_unit_even_and_odd_lengths() {
    convolution_checks::<Prime128Offset275>(0x275);
    convolution_checks::<Prime31Offset19>(0x019);
}

fn block_identity<F: Field + Debug, M: TrinomialModulus>(seed: u64) {
    const D: usize = 6;
    const ROWS: usize = 2;
    const COLUMNS: usize = 3;
    let mut rng = StdRng::seed_from_u64(seed);
    let matrix: Vec<Vec<Vec<F>>> = (0..ROWS)
        .map(|_| {
            (0..COLUMNS)
                .map(|_| (0..D).map(|_| random(&mut rng)).collect())
                .collect()
        })
        .collect();
    let u: Vec<Vec<F>> = (0..ROWS)
        .map(|_| (0..D).map(|_| random(&mut rng)).collect())
        .collect();
    let v: Vec<Vec<F>> = (0..COLUMNS)
        .map(|_| (0..D).map(|_| random(&mut rng)).collect())
        .collect();
    let adjoint = TrinomialAdjoint::<D, M>::new().unwrap();
    let mut forward = vec![vec![F::zero(); D]; ROWS];
    let mut transpose = vec![vec![F::zero(); D]; COLUMNS];
    for (i, row) in matrix.iter().enumerate() {
        for (j, coefficients) in row.iter().enumerate() {
            let a = TrinomialRing::<F, D, M>::try_from_vec(coefficients.clone()).unwrap();
            let b = TrinomialRing::<F, D, M>::try_from_vec(v[j].clone()).unwrap();
            let product = a.schoolbook_mul(&b).unwrap();
            let contribution = adjoint.apply(&u[i], coefficients).unwrap();
            for k in 0..D {
                forward[i][k] += product.coefficients()[k];
                transpose[j][k] += contribution[k];
            }
        }
    }
    let lhs = u
        .iter()
        .zip(&forward)
        .fold(F::zero(), |sum, (a, b)| sum + dot(a, b));
    let rhs = transpose
        .iter()
        .zip(&v)
        .fold(F::zero(), |sum, (a, b)| sum + dot(a, b));
    assert_eq!(lhs, rhs);
}

#[test]
fn block_matrix_has_genuine_coefficient_adjoint() {
    block_identity::<Prime128Offset275, MinusTrinomial>(0x26275);
    block_identity::<Prime31Offset19, MinusTrinomial>(0x26019);
    block_identity::<Prime128Offset275, PlusTrinomial>(0x26276);
    block_identity::<Prime31Offset19, PlusTrinomial>(0x26020);
}

fn equality<F: Field>(point: &[F], index: usize) -> F {
    point
        .iter()
        .enumerate()
        .fold(F::one(), |weight, (bit, &r)| {
            weight
                * if (index >> bit) & 1 == 0 {
                    F::one() - r
                } else {
                    r
                }
        })
}

fn rank_one_weights<F: Field + Debug>(seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let adjoint = TrinomialAdjoint::<DEGREE, MinusTrinomial>::new().unwrap();
    let rho: Vec<F> = (0..DEGREE).map(|_| random(&mut rng)).collect();
    let r_j: Vec<F> = (0..3).map(|_| random(&mut rng)).collect();
    let r_k: Vec<F> = (0..10).map(|_| random(&mut rng)).collect();
    let weights = SetupAdjointWeight::new(&adjoint, &rho, &r_j, &r_k, 5).unwrap();
    let v: Vec<F> = (0..DEGREE).map(|k| equality(&r_k, k)).collect();
    let h = dense_h::<F, DEGREE, MinusTrinomial>(&rho, &v);
    assert_eq!(weights.coefficients(), h);
    for j in 0..5 {
        for (s, &coefficient) in h.iter().enumerate() {
            assert_eq!(
                weights.weight(j, s).unwrap(),
                equality(&r_j, j) * coefficient
            );
        }
    }
    assert!(weights.weight(5, 0).is_err());
    assert!(weights.weight(0, DEGREE).is_err());
    // A Boolean point outside the unpadded coefficient domain is zero-defined.
    let outside = vec![F::one(); 10];
    let zero = SetupAdjointWeight::new(&adjoint, &rho, &r_j, &outside, 5).unwrap();
    assert!(zero.coefficients().iter().all(|&value| value == F::zero()));
    // With no row-index bits, the sole column has equality weight one.
    let one = SetupAdjointWeight::new(&adjoint, &rho, &[], &r_k, 1).unwrap();
    assert_eq!(one.weight(0, DEGREE - 1).unwrap(), h[DEGREE - 1]);
}

#[test]
fn setup_weights_factor_over_columns_and_coefficients_with_zero_padding() {
    rank_one_weights::<Prime128Offset275>(0x53275);
    rank_one_weights::<Prime31Offset19>(0x53019);
}

#[test]
fn malformed_shapes_and_unrepresentable_degrees_return_errors() {
    type F = Prime31Offset19;
    assert!(TrinomialAdjoint::<0, MinusTrinomial>::new().is_err());
    assert!(TrinomialAdjoint::<3, MinusTrinomial>::new().is_err());
    assert!(TrinomialAdjoint::<{ usize::MAX - 1 }, MinusTrinomial>::new().is_err());
    let adjoint = TrinomialAdjoint::<4, MinusTrinomial>::new().unwrap();
    let row = [F::zero(); 4];
    assert!(adjoint.extend_row(&row[..3]).is_err());
    assert!(adjoint.apply(&row[..3], &row).is_err());
    assert!(adjoint.apply(&row, &row[..3]).is_err());
    assert!(exact_convolution::<F>(&[], &[]).is_err());
    assert!(exact_convolution(&row, &row[..3]).is_err());
    assert!(SetupAdjointWeight::new(&adjoint, &row, &[], &row[..2], 0).is_err());
    assert!(SetupAdjointWeight::new(&adjoint, &row, &[], &row[..2], 2).is_err());
    assert!(SetupAdjointWeight::new(&adjoint, &row, &row[..1], &row[..1], 2).is_err());
    assert!(SetupAdjointWeight::new(&adjoint, &row, &row[..2], &row[..2], 2).is_err());
    assert!(SetupAdjointWeight::new(&adjoint, &row[..3], &[], &row[..2], 1).is_err());
}
