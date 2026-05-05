#![allow(missing_docs)]

use akita_algebra::ring::{CyclotomicRing, SparseChallenge, SparseChallengeConfig};
use akita_challenges::sparse::sparse_challenge_from_transcript;
use akita_field::fields::LiftBase;
use akita_field::Fp64;
use akita_field::{FieldCore, FromSmallInt};
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{Blake2bTranscript, Transcript};

type F = Fp64<4294967197>;

const D: usize = 32;

fn dense_eval<E: FieldCore + LiftBase<F>>(alpha: E, x: &CyclotomicRing<F, D>) -> E {
    let mut acc = E::zero();
    let mut pow = E::one();
    for c in x.coefficients().iter().copied() {
        acc += E::lift_base(c) * pow;
        pow = pow * alpha;
    }
    acc
}

#[test]
fn sparse_challenge_validate_and_to_dense() {
    let cfg = SparseChallengeConfig::Uniform {
        weight: 3,
        nonzero_coeffs: vec![-1, 1],
    };
    cfg.validate::<D>().unwrap();

    let s = SparseChallenge {
        positions: vec![0, 7, 12],
        coeffs: vec![1, -1, 1],
    };
    s.validate::<D>().unwrap();
    assert_eq!(s.hamming_weight(), 3);
    assert_eq!(s.l1_norm(), 3);

    let dense = s.to_dense::<F, D>().unwrap();
    assert_eq!(dense.hamming_weight(), 3);
    assert_eq!(dense.coefficients()[0], F::one());
    assert_eq!(dense.coefficients()[7], -F::one());
    assert_eq!(dense.coefficients()[12], F::one());
}

#[test]
fn sparse_eval_at_alpha_matches_dense_eval() {
    let alpha = F::from_u64(5);
    let alpha_pows = {
        let mut out = Vec::with_capacity(D);
        let mut acc = F::one();
        for _ in 0..D {
            out.push(acc);
            acc *= alpha;
        }
        out
    };

    let s = SparseChallenge {
        positions: vec![1, 3, 9],
        coeffs: vec![2, -1, 1],
    };
    let dense = s.to_dense::<F, D>().unwrap();

    let sparse_eval = s.eval_at_alpha::<F, F, D>(&alpha_pows).unwrap();
    let dense_eval = dense_eval::<F>(alpha, &dense);
    assert_eq!(sparse_eval, dense_eval);
}

#[test]
fn sparse_challenge_sampling_is_deterministic_and_exact_weight() {
    let cfg = SparseChallengeConfig::Uniform {
        weight: 8,
        nonzero_coeffs: vec![-1, 1],
    };

    let mut t1 = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    let mut t2 = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);

    // Make transcript state non-empty to avoid degenerate behavior.
    t1.append_field(b"seed", &F::from_u64(123));
    t2.append_field(b"seed", &F::from_u64(123));

    let c1 = sparse_challenge_from_transcript::<F, _, D>(&mut t1, b"c", 0, &cfg).unwrap();
    let c2 = sparse_challenge_from_transcript::<F, _, D>(&mut t2, b"c", 0, &cfg).unwrap();
    assert_eq!(c1, c2);
    c1.validate::<D>().unwrap();
    assert_eq!(c1.hamming_weight(), cfg.hamming_weight());
    assert_eq!(c1.l1_norm(), cfg.l1_mass() as u64);

    // Different instance_idx should change the sample.
    let mut t3 = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t3.append_field(b"seed", &F::from_u64(123));
    let c3 = sparse_challenge_from_transcript::<F, _, D>(&mut t3, b"c", 1, &cfg).unwrap();
    assert_ne!(c1, c3);
}

#[test]
fn split_ring_sampling_respects_half_budgets() {
    let cfg = SparseChallengeConfig::SplitRing {
        half_weight: 3,
        max_mag2_per_half: 1,
    };
    cfg.validate::<D>().unwrap();

    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(456));
    let challenge =
        sparse_challenge_from_transcript::<F, _, D>(&mut transcript, b"split", 0, &cfg).unwrap();

    challenge.validate::<D>().unwrap();
    assert_eq!(challenge.hamming_weight(), cfg.hamming_weight());
    assert!(challenge.l1_norm() <= cfg.l1_mass() as u64);

    let mut even_count = 0usize;
    let mut odd_count = 0usize;
    let mut even_mag2 = 0usize;
    let mut odd_mag2 = 0usize;
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        if (pos as usize) % 2 == 0 {
            even_count += 1;
            if coeff.abs() == 2 {
                even_mag2 += 1;
            }
        } else {
            odd_count += 1;
            if coeff.abs() == 2 {
                odd_mag2 += 1;
            }
        }
    }
    assert_eq!(even_count, 3);
    assert_eq!(odd_count, 3);
    assert!(even_mag2 <= 1);
    assert!(odd_mag2 <= 1);
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
    assert_eq!(cfg.max_hamming_weight::<D>(), 32);
    assert_eq!(cfg.max_hamming_weight::<128>(), 121);

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
    // tag=3, then M and B as u64 little-endian.
    let cfg = SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 8,
        l1_bound: 121,
    };
    let bytes = cfg.domain_separator_bytes();
    let mut expected = vec![3u8];
    expected.extend_from_slice(&8u64.to_le_bytes());
    expected.extend_from_slice(&121u64.to_le_bytes());
    assert_eq!(bytes, expected);

    // Distinct from the legacy variants for the same numeric content.
    let uniform = SparseChallengeConfig::Uniform {
        weight: 8,
        nonzero_coeffs: vec![1],
    }
    .domain_separator_bytes();
    let split = SparseChallengeConfig::SplitRing {
        half_weight: 8,
        max_mag2_per_half: 0,
    }
    .domain_separator_bytes();
    let shell = SparseChallengeConfig::ExactShell {
        count_mag1: 8,
        count_mag2: 0,
    }
    .domain_separator_bytes();
    assert_ne!(bytes, uniform);
    assert_ne!(bytes, split);
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

    let c1 = sparse_challenge_from_transcript::<F, _, D>(&mut t1, b"l1", 0, &cfg).unwrap();
    let c2 = sparse_challenge_from_transcript::<F, _, D>(&mut t2, b"l1", 0, &cfg).unwrap();
    assert_eq!(c1, c2, "sampling must be deterministic");

    // Shape invariants under the configured ball.
    c1.validate::<D>().unwrap();
    assert!(c1.hamming_weight() <= cfg.max_hamming_weight::<D>());
    assert!(c1.l1_norm() <= cfg.l1_mass() as u64);
    for &coef in &c1.coeffs {
        assert!(coef != 0, "stored coefficients must be nonzero");
        assert!(coef.unsigned_abs() <= cfg.max_abs_coeff() as u16);
    }

    // Different instance_idx must change the sample (overwhelming probability).
    let mut t3 = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t3.append_field(b"seed", &F::from_u64(42));
    let c3 = sparse_challenge_from_transcript::<F, _, D>(&mut t3, b"l1", 1, &cfg).unwrap();
    assert_ne!(c1, c3);
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
    let c = sparse_challenge_from_transcript::<F, _, D>(&mut t, b"ref", 0, &cfg).unwrap();

    let expected_positions: Vec<u32> = vec![
        0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 27, 28, 30,
        31,
    ];
    let expected_coeffs: Vec<i16> = vec![
        3, 1, -2, 8, -8, 4, 3, 4, 6, -3, -8, 2, 4, 7, -3, 3, 8, 2, -1, -3, 7, 7, 1, 2, 6, 8,
    ];
    assert_eq!(c.positions, expected_positions);
    assert_eq!(c.coeffs, expected_coeffs);
    c.validate::<D>().unwrap();
    assert!(c.l1_norm() <= 121);
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
    let err = sparse_challenge_from_transcript::<F, _, D_SMALL>(&mut t, b"undersized", 0, &cfg)
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
    use akita_challenges::sparse::sample_sparse_challenges;

    let cfg = SparseChallengeConfig::BoundedL1Ball {
        max_abs_coeff: 8,
        l1_bound: 121,
    };
    let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(0xBEEF));
    let challenges =
        sample_sparse_challenges::<F, _, D>(&mut transcript, b"ball-check", 4096, &cfg).unwrap();
    for c in &challenges {
        c.validate::<D>().unwrap();
        assert!(c.l1_norm() <= 121, "l1 norm {} > 121", c.l1_norm());
        for &v in &c.coeffs {
            assert!(
                v != 0 && v.unsigned_abs() <= 8,
                "out-of-bound coefficient {v}"
            );
        }
        assert!(c.hamming_weight() <= 32);
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
    let challenge =
        sparse_challenge_from_transcript::<F, _, D>(&mut transcript, b"shell", 0, &cfg).unwrap();

    challenge.validate::<D>().unwrap();
    assert_eq!(challenge.hamming_weight(), cfg.hamming_weight());
    assert_eq!(challenge.l1_norm(), cfg.l1_mass() as u64);
    assert_eq!(
        challenge.coeffs.iter().filter(|&&c| c.abs() == 1).count(),
        4
    );
    assert_eq!(
        challenge.coeffs.iter().filter(|&&c| c.abs() == 2).count(),
        2
    );
}
