#![allow(missing_docs)]

use akita_challenges::{
    fold_challenge_sample_label, sample_sparse_challenges, FoldDraw, LiveFoldDraw, SparseChallenge,
    SparseChallengeConfig,
};
use akita_field::{CanonicalField, FieldCore, Fp64};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, DOMAIN_AKITA_PROTOCOL};
use akita_transcript::{AkitaTranscript, Transcript};

type F = Fp64<4294967197>;

const D: usize = 32;

#[derive(Default)]
struct RecordingFoldDraw {
    absorb_labels: Vec<Vec<u8>>,
}

impl FoldDraw for RecordingFoldDraw {
    fn absorb_and_squeeze(&mut self, label: &[u8], _payload: &[u8]) -> Vec<u8> {
        self.absorb_labels.push(label.to_vec());
        vec![0; 32]
    }
}

/// Local helper: count non-zero positions in a sparse challenge.
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

/// Local helper: convert to dense ring coefficients for layout/validation tests.
fn sparse_challenge_to_dense<F: FieldCore + CanonicalField, const D: usize>(
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

fn dense_hamming_weight<F: FieldCore, const D: usize>(coeffs: &[F; D]) -> usize {
    coeffs
        .iter()
        .filter(|coefficient| !coefficient.is_zero())
        .count()
}

#[test]
fn sparse_challenge_to_dense_lays_out_coefficients() {
    let s = SparseChallenge {
        positions: vec![0, 7, 12],
        coeffs: vec![1, -1, 1],
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
fn pm1_only_sampling_is_deterministic_and_exact_weight() {
    let cfg = SparseChallengeConfig::pm1_only(8);

    let mut t1 = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    let mut t2 = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t1.append_field(b"seed", &F::from_u64(123));
    t2.append_field(b"seed", &F::from_u64(123));

    let c1 = sample_sparse_challenges::<F, _>(&mut t1, b"c", D, 1, &cfg, 0)
        .unwrap()
        .pop()
        .unwrap();
    let c2 = sample_sparse_challenges::<F, _>(&mut t2, b"c", D, 1, &cfg, 0)
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(c1, c2);
    assert_eq!(hamming_weight(&c1), 8);
    assert_eq!(l1_norm(&c1), cfg.l1_norm() as u64);
}

#[test]
fn grind_nonce_changes_sparse_challenge_stream() {
    const D: usize = 32;
    let cfg = SparseChallengeConfig::pm1_only(3);
    let mut t0 = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    let mut t1 = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t0.append_field(b"seed", &F::from_u64(42));
    t1.append_field(b"seed", &F::from_u64(42));

    let c0 = sample_sparse_challenges::<F, _>(&mut t0, b"fold", D, 1, &cfg, 0)
        .unwrap()
        .pop()
        .unwrap();
    let c1 = sample_sparse_challenges::<F, _>(&mut t1, b"fold", D, 1, &cfg, 1)
        .unwrap()
        .pop()
        .unwrap();
    assert_ne!(c0, c1);
}

#[test]
fn signed_sparse_sampling_has_exact_magnitude_counts() {
    let cfg = SparseChallengeConfig {
        count_pm1: 4,
        count_pm2: 2,
    };
    cfg.validate::<D>().unwrap();

    let mut transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(789));
    let challenge = sample_sparse_challenges::<F, _>(&mut transcript, b"shell", D, 1, &cfg, 0)
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

#[test]
fn signed_sparse_sampling_handles_weight_above_sign_stack_chunk() {
    const DR: usize = 128;
    let cfg = SparseChallengeConfig {
        count_pm1: 65,
        count_pm2: 0,
    };
    cfg.validate::<DR>().unwrap();

    let sample = || {
        let mut transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
        transcript.append_field(b"seed", &F::from_u64(0x516E));
        sample_sparse_challenges::<F, _>(&mut transcript, b"large-shell", DR, 3, &cfg, 0).unwrap()
    };

    let first = sample();
    let second = sample();
    assert_eq!(first, second);
    for c in &first {
        assert_eq!(hamming_weight(c), 65);
        assert_eq!(l1_norm(c), 65);
        assert!(c.coeffs.iter().all(|&v| v == 1 || v == -1));
    }
}

#[test]
fn fold_sampling_draws_one_challenge_per_claim_and_block() {
    const DR: usize = 8;
    let cfg = SparseChallengeConfig::pm1_only(2);
    let mut transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(7));

    let challenges = LiveFoldDraw::<F, _>::new(&mut transcript)
        .draw_folding_challenges(DR, 3, 5, 2, &cfg, 0)
        .unwrap();

    assert_eq!(challenges.len(), 10);
    assert_eq!(challenges.num_claims(), 2);
    assert_eq!(challenges.num_live_blocks_per_claim(), 5);
    assert!(challenges
        .as_slice()
        .iter()
        .all(|challenge| challenge.positions.len() == 2));
}

#[test]
fn challenge_layout_rejects_mismatched_vector_length() {
    let challenge = SparseChallenge {
        positions: Vec::new(),
        coeffs: Vec::new(),
    };
    assert!(akita_challenges::Challenges::from_sparse(vec![challenge], 2, 1).is_err());
}

#[test]
fn fold_sampling_binds_group_geometry() {
    let first = fold_challenge_sample_label(0, 5, 2).unwrap();
    let second = fold_challenge_sample_label(1, 5, 2).unwrap();
    let resized = fold_challenge_sample_label(0, 6, 2).unwrap();

    assert_ne!(first, second);
    assert_ne!(first, resized);

    let cfg = SparseChallengeConfig::pm1_only(2);
    let mut draw = RecordingFoldDraw::default();
    draw.draw_folding_challenges(8, 0, 5, 1, &cfg, 0).unwrap();
    assert_eq!(draw.absorb_labels, vec![ABSORB_SPARSE_CHALLENGE]);
}
