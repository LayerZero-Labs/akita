#![allow(missing_docs)]

use akita_challenges::SparseChallenge;
use jolt_field::{CanonicalEncoding, Field, Fp64, One};

type F = Fp64<4294967197>;

const D: usize = 32;

/// Local helper: convert to dense ring coefficients for layout/validation tests.
fn sparse_challenge_to_dense<F: Field + CanonicalEncoding, const D: usize>(
    c: &SparseChallenge,
) -> Result<[F; D], &'static str> {
    if c.positions.len() != c.coeffs.len() {
        return Err("positions and coeffs must have same length");
    }
    let mut out = [F::zero(); D];
    let mut seen = vec![false; D];
    for (&pos, &coeff) in c.positions.iter().zip(c.coeffs.iter()) {
        if coeff == 0 {
            return Err("coeffs must not contain 0");
        }
        let idx = pos as usize;
        if idx >= D {
            return Err("position out of range");
        }
        if seen[idx] {
            return Err("positions must be unique");
        }
        seen[idx] = true;
        out[idx] += F::from_i64(coeff as i64);
    }
    Ok(out)
}

fn dense_hamming_weight<F: Field, const D: usize>(coeffs: &[F; D]) -> usize {
    coeffs
        .iter()
        .filter(|coefficient| !coefficient.is_zero())
        .count()
}

#[test]
fn sparse_challenge_to_dense_lays_out_coefficients() {
    let s = SparseChallenge {
        positions: vec![0, 7, 12].into(),
        coeffs: vec![1, -1, 1].into(),
    };
    let dense = sparse_challenge_to_dense::<F, D>(&s).unwrap();
    assert_eq!(dense_hamming_weight(&dense), 3);
    assert_eq!(dense[0], F::one());
    assert_eq!(dense[7], -F::one());
    assert_eq!(dense[12], F::one());
}

#[test]
fn sparse_challenge_to_dense_rejects_invalid_inputs() {
    let mismatched = SparseChallenge {
        positions: vec![0, 1].into(),
        coeffs: vec![1].into(),
    };
    assert!(sparse_challenge_to_dense::<F, D>(&mismatched).is_err());

    let zero_coeff = SparseChallenge {
        positions: vec![0, 1].into(),
        coeffs: vec![1, 0].into(),
    };
    assert!(sparse_challenge_to_dense::<F, D>(&zero_coeff).is_err());

    let out_of_range = SparseChallenge {
        positions: vec![0, D as u32].into(),
        coeffs: vec![1, 1].into(),
    };
    assert!(sparse_challenge_to_dense::<F, D>(&out_of_range).is_err());

    let duplicate = SparseChallenge {
        positions: vec![3, 3].into(),
        coeffs: vec![1, 1].into(),
    };
    assert!(sparse_challenge_to_dense::<F, D>(&duplicate).is_err());
}

#[test]
fn challenge_layout_rejects_mismatched_vector_length() {
    let challenge = SparseChallenge {
        positions: Vec::new().into(),
        coeffs: Vec::new().into(),
    };
    assert!(akita_challenges::Challenges::from_sparse(vec![challenge], 2, 1).is_err());
}
