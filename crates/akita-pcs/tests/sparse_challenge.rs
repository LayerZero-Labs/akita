#![allow(missing_docs)]

use akita_algebra::fields::LiftBase;
use akita_algebra::ring::{CyclotomicRing, SparseChallenge, SparseChallengeConfig};
use akita_algebra::Fp64;
use akita_challenges::sparse::sparse_challenge_from_transcript;
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
