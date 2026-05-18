#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_algebra::ring::CyclotomicRing;
use akita_challenges::{sample_sparse_challenges, SparseChallenge, SparseChallengeConfig};
use akita_field::{CanonicalField, FieldCore, Fp64};
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{Blake2bTranscript, Transcript};

type F = Fp64<4294967197>;

const D: usize = 32;

/// Local helper: count non-zero positions in a sparse challenge. The crate no
/// longer ships a dedicated `hamming_weight` accessor since it was only ever
/// used by these tests.
fn hamming_weight(c: &SparseChallenge) -> usize {
    debug_assert_eq!(c.positions.len(), c.coeffs.len());
    c.positions.len()
}

/// Local helper: integer L1 norm of a sparse challenge.
fn l1_norm(c: &SparseChallenge) -> u64 {
    c.coeffs
        .iter()
        .map(|&v| (v as i32).unsigned_abs() as u64)
        .sum()
}

/// Local helper: convert to a dense ring element for layout/validation tests.
fn sparse_challenge_to_dense<F: FieldCore + CanonicalField, const D: usize>(
    c: &SparseChallenge,
) -> Result<CyclotomicRing<F, D>, &'static str> {
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
    Ok(CyclotomicRing::from_coefficients(out))
}

#[test]
fn sparse_challenge_to_dense_lays_out_coefficients() {
    let s = SparseChallenge {
        positions: vec![0, 7, 12],
        coeffs: vec![1, -1, 1],
    };
    let dense = sparse_challenge_to_dense::<F, D>(&s).unwrap();
    assert_eq!(dense.hamming_weight(), 3);
    assert_eq!(dense.coefficients()[0], F::one());
    assert_eq!(dense.coefficients()[7], -F::one());
    assert_eq!(dense.coefficients()[12], F::one());
}

#[test]
fn sparse_challenge_to_dense_rejects_invalid_inputs() {
    let mismatched = SparseChallenge {
        positions: vec![0, 1],
        coeffs: vec![1],
    };
    assert!(sparse_challenge_to_dense::<F, D>(&mismatched).is_err());

    let zero_coeff = SparseChallenge {
        positions: vec![0, 1],
        coeffs: vec![1, 0],
    };
    assert!(sparse_challenge_to_dense::<F, D>(&zero_coeff).is_err());

    let out_of_range = SparseChallenge {
        positions: vec![0, D as u32],
        coeffs: vec![1, 1],
    };
    assert!(sparse_challenge_to_dense::<F, D>(&out_of_range).is_err());

    let duplicate = SparseChallenge {
        positions: vec![3, 3],
        coeffs: vec![1, 1],
    };
    assert!(sparse_challenge_to_dense::<F, D>(&duplicate).is_err());
}

#[test]
fn uniform_sampling_is_deterministic_and_exact_weight() {
    let cfg = SparseChallengeConfig::Uniform {
        weight: 8,
        nonzero_coeffs: vec![-1, 1],
    };

    let mut t1 = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    let mut t2 = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t1.append_field(b"seed", &F::from_u64(123));
    t2.append_field(b"seed", &F::from_u64(123));

    let c1 = sample_sparse_challenges::<F, _, D>(&mut t1, b"c", 1, &cfg)
        .unwrap()
        .pop()
        .unwrap();
    let c2 = sample_sparse_challenges::<F, _, D>(&mut t2, b"c", 1, &cfg)
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(c1, c2);
    assert_eq!(hamming_weight(&c1), 8);
    assert_eq!(l1_norm(&c1), cfg.l1_norm() as u64);
}

#[test]
fn bounded_l1_validate_d32_preset() {
    let cfg = SparseChallengeConfig::BoundedL1Norm;
    cfg.validate::<D>().unwrap();
    assert_eq!(cfg.l1_norm(), 121);
    assert_eq!(cfg.infinity_norm(), 8);

    // The bounded-L1 variant is a fixed D=32 production preset.
    assert!(SparseChallengeConfig::BoundedL1Norm
        .validate::<3>()
        .is_err());
}

#[test]
fn bounded_l1_domain_separator_is_canonical() {
    // tag=2, then the fixed M and B preset values as u64 little-endian.
    let cfg = SparseChallengeConfig::BoundedL1Norm;
    let bytes = cfg.domain_separator_bytes();
    let mut expected = vec![2u8];
    expected.extend_from_slice(&8u64.to_le_bytes());
    expected.extend_from_slice(&121u64.to_le_bytes());
    assert_eq!(bytes, expected);

    // Distinct from the other surviving variants for the same numeric content.
    let uniform = SparseChallengeConfig::Uniform {
        weight: 8,
        nonzero_coeffs: vec![1],
    }
    .domain_separator_bytes();
    let shell = SparseChallengeConfig::ExactShell {
        count_mag1: 8,
        count_mag2: 0,
    }
    .domain_separator_bytes();
    assert_ne!(bytes, uniform);
    assert_ne!(bytes, shell);
}

#[test]
fn bounded_l1_sampling_is_deterministic_and_within_bounds() {
    let cfg = SparseChallengeConfig::BoundedL1Norm;

    let mut t1 = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    let mut t2 = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t1.append_field(b"seed", &F::from_u64(42));
    t2.append_field(b"seed", &F::from_u64(42));

    let c1 = sample_sparse_challenges::<F, _, D>(&mut t1, b"l1", 1, &cfg)
        .unwrap()
        .pop()
        .unwrap();
    let c2 = sample_sparse_challenges::<F, _, D>(&mut t2, b"l1", 1, &cfg)
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(c1, c2, "sampling must be deterministic");

    assert!(hamming_weight(&c1) <= D);
    assert!(l1_norm(&c1) <= cfg.l1_norm() as u64);
    for &coef in &c1.coeffs {
        assert!(coef != 0, "stored coefficients must be nonzero");
        assert!(u32::from(coef.unsigned_abs()) <= cfg.infinity_norm());
    }
}

#[test]
fn bounded_l1_reference_vector_d32_m8_b121() {
    // Locks the canonical byte order, coefficient order, and rejection-loop
    // behaviour for the (D=32, M=8, B=121) preset. Updating these expected
    // values is a transcript-distribution change.
    let cfg = SparseChallengeConfig::BoundedL1Norm;
    let mut t = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t.append_field(b"seed", &F::from_u64(0xC0FFEE));
    let c = sample_sparse_challenges::<F, _, D>(&mut t, b"ref", 1, &cfg)
        .unwrap()
        .pop()
        .unwrap();

    // Canonical fixture under the magnitude-first bucket layout
    // `0, -1, +1, -2, +2, ...`. Updating these expected values is a
    // transcript-distribution change.
    let expected_positions: Vec<u32> = vec![
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30,
    ];
    let expected_coeffs: Vec<i8> = vec![
        -6, 1, -4, -7, 7, -3, -5, -2, -2, 4, -7, -8, -1, -1, -1, -5, -4, -6, 7, -7, -8, -3, -2, 8,
        4, 2, 1, 1, 4,
    ];
    assert_eq!(c.positions, expected_positions);
    assert_eq!(c.coeffs, expected_coeffs);
    assert!(l1_norm(&c) <= 121);
}

#[test]
fn bounded_l1_rejects_non_d32_ring() {
    // The bounded-L1 sampler is the fixed D=32 preset. Any other `D` must be
    // rejected before sampling instead of silently using the D=32 DP table.
    const D_SMALL: usize = 3;
    let cfg = SparseChallengeConfig::BoundedL1Norm;
    assert!(cfg.validate::<D_SMALL>().is_err());

    let mut t = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t.append_field(b"seed", &F::from_u64(0xDADADA));
    let err = sample_sparse_challenges::<F, _, D_SMALL>(&mut t, b"non-d32", 1, &cfg)
        .expect_err("non-D=32 BoundedL1Norm must be rejected");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("only D = 32"),
        "expected rejection to mention D = 32 requirement, got: {msg}"
    );
}

#[test]
fn bounded_l1_d32_samples_are_in_norm_bound() {
    // The D=32, M=8, B=121 production preset has WAYS[32][121] ~= 2^128.133,
    // so the truncated-2^128 sampler is well-defined. Every produced sample
    // must satisfy the structural invariants and the L_inf / L1 bounds. We
    // sample a healthy batch to exercise more than one descent path.
    let cfg = SparseChallengeConfig::BoundedL1Norm;
    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(0xBEEF));
    let challenges =
        sample_sparse_challenges::<F, _, D>(&mut transcript, b"norm-check", 4096, &cfg).unwrap();
    for c in &challenges {
        assert_eq!(c.positions.len(), c.coeffs.len());
        assert!(l1_norm(c) <= 121, "l1 norm {} > 121", l1_norm(c));
        for &v in &c.coeffs {
            assert!(
                v != 0 && v.unsigned_abs() <= 8,
                "out-of-bound coefficient {v}"
            );
        }
        assert!(hamming_weight(c) <= D);
    }
}

#[test]
fn exact_shell_sampling_has_exact_magnitude_counts() {
    let cfg = SparseChallengeConfig::ExactShell {
        count_mag1: 4,
        count_mag2: 2,
    };
    cfg.validate::<D>().unwrap();

    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(789));
    let challenge = sample_sparse_challenges::<F, _, D>(&mut transcript, b"shell", 1, &cfg)
        .unwrap()
        .pop()
        .unwrap();

    assert_eq!(hamming_weight(&challenge), 6);
    assert_eq!(l1_norm(&challenge), cfg.l1_norm() as u64);
    assert_eq!(
        challenge.coeffs.iter().filter(|&&c| c.abs() == 1).count(),
        4
    );
    assert_eq!(
        challenge.coeffs.iter().filter(|&&c| c.abs() == 2).count(),
        2
    );
}
