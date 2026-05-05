#![allow(missing_docs)]

use akita_challenges::{sample_sparse_challenges, SparseChallenge, SparseChallengeConfig};
use akita_field::Fp64;
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

#[test]
fn sparse_challenge_to_dense_lays_out_coefficients() {
    let s = SparseChallenge {
        positions: vec![0, 7, 12],
        coeffs: vec![1, -1, 1],
    };
    let dense = s.to_dense::<F, D>().unwrap();
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
    assert!(mismatched.to_dense::<F, D>().is_err());

    let zero_coeff = SparseChallenge {
        positions: vec![0, 1],
        coeffs: vec![1, 0],
    };
    assert!(zero_coeff.to_dense::<F, D>().is_err());

    let out_of_range = SparseChallenge {
        positions: vec![0, D as u32],
        coeffs: vec![1, 1],
    };
    assert!(out_of_range.to_dense::<F, D>().is_err());

    let duplicate = SparseChallenge {
        positions: vec![3, 3],
        coeffs: vec![1, 1],
    };
    assert!(duplicate.to_dense::<F, D>().is_err());
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
    assert_eq!(l1_norm(&c1), cfg.l1_mass() as u64);
}

#[test]
fn bounded_l1_validate_d32_m8_b121() {
    let cfg = SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 8,
        l1_bound: 121,
    };
    cfg.validate::<D>().unwrap();
    assert_eq!(cfg.l1_mass(), 121);
    assert_eq!(cfg.max_abs_coeff(), 8);

    // Validation rejects zero parameters and B > D * M.
    assert!(SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 0,
        l1_bound: 1,
    }
    .validate::<D>()
    .is_err());
    assert!(SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 1,
        l1_bound: 0,
    }
    .validate::<D>()
    .is_err());
    // D * M = 32 * 8 = 256, so 257 must fail.
    assert!(SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 8,
        l1_bound: 257,
    }
    .validate::<D>()
    .is_err());
}

#[test]
fn bounded_l1_domain_separator_is_canonical() {
    // tag=2, then M and B as u64 little-endian.
    let cfg = SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 8,
        l1_bound: 121,
    };
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
    let cfg = SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 8,
        l1_bound: 121,
    };

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
    assert!(l1_norm(&c1) <= cfg.l1_mass() as u64);
    for &coef in &c1.coeffs {
        assert!(coef != 0, "stored coefficients must be nonzero");
        assert!(coef.unsigned_abs() <= cfg.max_abs_coeff() as u16);
    }
}

#[test]
fn bounded_l1_reference_vector_d32_m8_b121() {
    // Locks the canonical byte order, coefficient order, and rejection-loop
    // behaviour for the (D=32, M=8, B=121) preset. Updating these expected
    // values is a transcript-distribution change.
    let cfg = SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 8,
        l1_bound: 121,
    };
    let mut t = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t.append_field(b"seed", &F::from_u64(0xC0FFEE));
    let c = sample_sparse_challenges::<F, _, D>(&mut t, b"ref", 1, &cfg)
        .unwrap()
        .pop()
        .unwrap();

    // Canonical fixture under the `akita/sparse-challenge-prg` PRG domain
    // (renamed from `hachi/sparse-challenge-prg` during the akita-challenges
    // crate refactor; see `specs/bounded-l1-sparse-challenge.md`).
    let expected_positions: Vec<u32> = vec![
        0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26,
        28, 29, 30, 31,
    ];
    let expected_coeffs: Vec<i16> = vec![
        1, 5, 4, -4, 3, 2, 6, -7, -4, -5, -1, -2, -4, -1, -2, 5, 2, -6, 2, 7, 7, -5, -2, -2, 4, 7,
        8, -5, 1,
    ];
    assert_eq!(c.positions, expected_positions);
    assert_eq!(c.coeffs, expected_coeffs);
    assert!(l1_norm(&c) <= 121);
}

#[test]
fn bounded_l1_undersized_support_is_rejected() {
    // The truncated-2^128 sampler requires WAYS[D][B] >= 2^128. Tiny
    // (D=3, M=2, B=3) has ball size 25, well below 2^128, so building the
    // sampler scratch must fail loudly instead of silently producing
    // out-of-ball samples or panicking deeper in the descent loop.
    const D_SMALL: usize = 3;
    let cfg = SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 2,
        l1_bound: 3,
    };
    cfg.validate::<D_SMALL>().unwrap();

    let mut t = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t.append_field(b"seed", &F::from_u64(0xDADADA));
    let err = sample_sparse_challenges::<F, _, D_SMALL>(&mut t, b"undersized", 1, &cfg)
        .expect_err("undersized BoundedL1Ball must be rejected");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("< 2^128"),
        "expected rejection message to mention < 2^128, got: {msg}"
    );
}

#[test]
fn bounded_l1_d32_samples_are_in_ball() {
    // The D=32, M=8, B=121 production preset has WAYS[32][121] ~= 2^128.133,
    // so the truncated-2^128 sampler is well-defined. Every produced sample
    // must satisfy the structural invariants and the L_inf / L1 bounds. We
    // sample a healthy batch to exercise more than one descent path.
    let cfg = SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 8,
        l1_bound: 121,
    };
    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(0xBEEF));
    let challenges =
        sample_sparse_challenges::<F, _, D>(&mut transcript, b"ball-check", 4096, &cfg).unwrap();
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
    assert_eq!(l1_norm(&challenge), cfg.l1_mass() as u64);
    assert_eq!(
        challenge.coeffs.iter().filter(|&&c| c.abs() == 1).count(),
        4
    );
    assert_eq!(
        challenge.coeffs.iter().filter(|&&c| c.abs() == 2).count(),
        2
    );
}
