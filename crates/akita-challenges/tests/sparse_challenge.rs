#![allow(missing_docs)]

use akita_challenges::{
    preview_folding_challenges, sample_folding_challenges, sample_sparse_challenges,
    tensor_left_digest, ChallengeLabels, ChallengeShape, Challenges, SparseChallenge,
    SparseChallengeConfig, TensorChallenges,
};
use akita_field::{CanonicalField, FieldCore, Fp64};
use akita_transcript::labels::{
    ABSORB_TENSOR_FOLD_LEFT, CHALLENGE_TENSOR_FOLD_LEFT, CHALLENGE_TENSOR_FOLD_RIGHT,
    CHALLENGE_WITNESS_FOLD, DOMAIN_AKITA_PROTOCOL,
};
use akita_transcript::{AkitaTranscript, Transcript};

/// Stage-1 fold label bundle reused by every tensor-vs-flat sampling test.
fn fold_challenge_labels() -> ChallengeLabels<'static> {
    ChallengeLabels {
        flat: CHALLENGE_WITNESS_FOLD,
        tensor_left: CHALLENGE_TENSOR_FOLD_LEFT,
        tensor_left_digest: ABSORB_TENSOR_FOLD_LEFT,
        tensor_right: CHALLENGE_TENSOR_FOLD_RIGHT,
    }
}

type F = Fp64<4294967197>;

const D: usize = 32;

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

/// Local helper: scalar power table `[1, alpha, alpha^2, ..., alpha^{D-1}]`.
fn scalar_powers<F: FieldCore, const D: usize>(alpha: F) -> Vec<F> {
    (0..D)
        .scan(F::one(), |power, _| {
            let out = *power;
            *power *= alpha;
            Some(out)
        })
        .collect()
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
    coeffs.iter().filter(|coeff| !coeff.is_zero()).count()
}

fn dense_negacyclic_mul<F: FieldCore, const D: usize>(lhs: &[F; D], rhs: &[F; D]) -> [F; D] {
    let mut out = [F::zero(); D];
    for (i, &left) in lhs.iter().enumerate() {
        if left.is_zero() {
            continue;
        }
        for (j, &right) in rhs.iter().enumerate() {
            if right.is_zero() {
                continue;
            }
            let degree = i + j;
            if degree < D {
                out[degree] += left * right;
            } else {
                out[degree - D] -= left * right;
            }
        }
    }
    out
}

fn eval_dense_at_pows<F: FieldCore, const D: usize>(coeffs: &[F; D], alpha_pows: &[F]) -> F {
    coeffs
        .iter()
        .zip(alpha_pows.iter())
        .fold(F::zero(), |acc, (&coeff, &power)| acc + coeff * power)
}

fn tensor_product_eval<F: FieldCore + CanonicalField, const D: usize>(
    left: &SparseChallenge,
    right: &SparseChallenge,
    alpha_pows: &[F],
) -> F {
    let product = dense_negacyclic_mul(
        &sparse_challenge_to_dense::<F, D>(left).unwrap(),
        &sparse_challenge_to_dense::<F, D>(right).unwrap(),
    );
    eval_dense_at_pows(&product, alpha_pows)
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
fn dense_negacyclic_product_reference_handles_wrap_and_cancellation() {
    const TD: usize = 8;
    let left = SparseChallenge {
        positions: vec![0, 1],
        coeffs: vec![1, 1],
    };
    let right = SparseChallenge {
        positions: vec![0, TD as u32 - 1],
        coeffs: vec![1, 1],
    };

    let dense_product = dense_negacyclic_mul(
        &sparse_challenge_to_dense::<F, TD>(&left).unwrap(),
        &sparse_challenge_to_dense::<F, TD>(&right).unwrap(),
    );
    let mut expected = [F::zero(); TD];
    expected[1] = F::one();
    expected[TD - 1] = F::one();

    assert_eq!(dense_product, expected);
}

#[test]
fn tensor_sampling_uses_two_vectors() {
    const TD: usize = 8;
    let cfg = SparseChallengeConfig::pm1_only(2);
    let mut transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(7));

    let challenges = sample_folding_challenges::<F, _>(
        &mut transcript,
        TD,
        8,
        2,
        &cfg,
        &ChallengeShape::Tensor,
        fold_challenge_labels(),
        0,
    )
    .unwrap();

    let Challenges::Tensor {
        factored: tensor, ..
    } = challenges
    else {
        panic!("expected tensor challenges");
    };
    assert_eq!(tensor.left_len, 2);
    assert_eq!(tensor.right_len, 4);
    assert_eq!(tensor.left.len(), 4);
    assert_eq!(tensor.right.len(), 8);
    assert_eq!(tensor.total_blocks().unwrap(), 16);
}

#[test]
fn tensor_effective_l2_sq_max_is_deterministic_product_envelope() {
    let d64 = SparseChallengeConfig {
        count_pm1: akita_challenges::D64_PRODUCTION_PM1_COUNT,
        count_pm2: akita_challenges::D64_PRODUCTION_PM2_COUNT,
    };
    assert_eq!(d64.l1_norm(), 51);
    assert_eq!(d64.challenge_l2_sq_max(), 71);
    assert_eq!(ChallengeShape::Flat.effective_l2_sq_max(&d64), 71);
    assert_eq!(
        ChallengeShape::Tensor.effective_l2_sq_max(&d64),
        51u128 * 51 * 71
    );

    let d128 = SparseChallengeConfig::pm1_only(31);
    assert_eq!(d128.l1_norm(), 31);
    assert_eq!(d128.challenge_l2_sq_max(), 31);
    assert_eq!(ChallengeShape::Flat.effective_l2_sq_max(&d128), 31);
    assert_eq!(
        ChallengeShape::Tensor.effective_l2_sq_max(&d128),
        31u128 * 31 * 31
    );

    let d256 = SparseChallengeConfig::pm1_only(23);
    assert_eq!(d256.l1_norm(), 23);
    assert_eq!(d256.challenge_l2_sq_max(), 23);
    assert_eq!(ChallengeShape::Flat.effective_l2_sq_max(&d256), 23);
    assert_eq!(
        ChallengeShape::Tensor.effective_l2_sq_max(&d256),
        23u128 * 23 * 23
    );
}

#[test]
fn tensor_sampling_absorbs_left_digest_before_right() {
    const TD: usize = 8;
    let cfg = SparseChallengeConfig::pm1_only(2);

    let mut sampled_transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    sampled_transcript.append_field(b"seed", &F::from_u64(0x5151));
    let sampled = sample_folding_challenges::<F, _>(
        &mut sampled_transcript,
        TD,
        8,
        2,
        &cfg,
        &ChallengeShape::Tensor,
        fold_challenge_labels(),
        0,
    )
    .unwrap();
    let Challenges::Tensor {
        factored: sampled, ..
    } = sampled
    else {
        panic!("expected tensor challenges");
    };

    let mut manual_transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    manual_transcript.append_field(b"seed", &F::from_u64(0x5151));
    let left = sample_sparse_challenges::<F, _>(
        &mut manual_transcript,
        CHALLENGE_TENSOR_FOLD_LEFT,
        TD,
        sampled.left.len(),
        &cfg,
        0,
    )
    .unwrap();
    let left_digest = tensor_left_digest(&left, sampled.left_len, sampled.num_claims, TD).unwrap();
    manual_transcript.append_bytes(ABSORB_TENSOR_FOLD_LEFT, &left_digest);
    let right = sample_sparse_challenges::<F, _>(
        &mut manual_transcript,
        CHALLENGE_TENSOR_FOLD_RIGHT,
        TD,
        sampled.right.len(),
        &cfg,
        0,
    )
    .unwrap();

    // The right factor must be sampled after absorbing the left digest.
    let mut nodigest_transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    nodigest_transcript.append_field(b"seed", &F::from_u64(0x5151));
    let _nodigest_left = sample_sparse_challenges::<F, _>(
        &mut nodigest_transcript,
        CHALLENGE_TENSOR_FOLD_LEFT,
        TD,
        sampled.left.len(),
        &cfg,
        0,
    )
    .unwrap();
    let nodigest_right = sample_sparse_challenges::<F, _>(
        &mut nodigest_transcript,
        CHALLENGE_TENSOR_FOLD_RIGHT,
        TD,
        sampled.right.len(),
        &cfg,
        0,
    )
    .unwrap();

    assert_eq!(sampled.left, left);
    assert_eq!(sampled.right, right);
    assert_ne!(
        sampled.right, nodigest_right,
        "right challenges must be bound to the tensor-left output digest"
    );
}

#[test]
fn tensor_preview_matches_live_sample_without_advancing_transcript() {
    const TD: usize = 8;
    let cfg = SparseChallengeConfig::pm1_only(2);
    let mut transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(0x7171));

    let previewed = preview_folding_challenges(
        &transcript,
        TD,
        8,
        2,
        &cfg,
        &ChallengeShape::Tensor,
        fold_challenge_labels(),
        7,
    )
    .unwrap();
    let live = sample_folding_challenges::<F, _>(
        &mut transcript,
        TD,
        8,
        2,
        &cfg,
        &ChallengeShape::Tensor,
        fold_challenge_labels(),
        7,
    )
    .unwrap();

    assert_eq!(previewed, live);
}

#[test]
fn tensor_left_digest_rejects_duplicate_positions() {
    const TD: usize = 8;
    let left = vec![SparseChallenge {
        positions: vec![0, 0],
        coeffs: vec![1, -1],
    }];

    let err = tensor_left_digest(&left, 1, 1, TD).unwrap_err();

    assert!(matches!(err, akita_field::AkitaError::InvalidInput(msg) if msg.contains("unique")));
}

#[test]
fn tensor_lazy_evals_match_ring_product_reference() {
    const TD: usize = 8;
    let cfg = SparseChallengeConfig::pm1_only(2);
    let mut transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(99));
    let challenges = sample_folding_challenges::<F, _>(
        &mut transcript,
        TD,
        8,
        1,
        &cfg,
        &ChallengeShape::Tensor,
        fold_challenge_labels(),
        0,
    )
    .unwrap();

    let alpha_pows = scalar_powers::<F, TD>(F::from_u64(5));
    let lazy = challenges.evals_at_pows::<F, F>(&alpha_pows).unwrap();
    let Challenges::Tensor { factored } = &challenges else {
        panic!("expected tensor challenges");
    };
    let expected = (0..factored.total_blocks().unwrap())
        .map(|block_idx| {
            let (_, _, left, right) = factored.factors_for_logical_block(block_idx).unwrap();
            tensor_product_eval::<F, TD>(left, right, &alpha_pows)
        })
        .collect::<Vec<_>>();

    assert_eq!(lazy, expected);
}

#[test]
fn tensor_factored_aggregate_matches_ring_product_reference() {
    const TD: usize = 8;
    let tensor = TensorChallenges {
        left: vec![
            SparseChallenge {
                positions: vec![0, 6],
                coeffs: vec![2, -1],
            },
            SparseChallenge {
                positions: vec![1, 3],
                coeffs: vec![1, 3],
            },
            SparseChallenge {
                positions: vec![2, 7],
                coeffs: vec![-2, 1],
            },
            SparseChallenge {
                positions: vec![0, 5],
                coeffs: vec![1, -3],
            },
        ],
        right: vec![
            SparseChallenge {
                positions: vec![0],
                coeffs: vec![1],
            },
            SparseChallenge {
                positions: vec![2],
                coeffs: vec![-1],
            },
            SparseChallenge {
                positions: vec![4],
                coeffs: vec![2],
            },
            SparseChallenge {
                positions: vec![6],
                coeffs: vec![1],
            },
            SparseChallenge {
                positions: vec![1, 5],
                coeffs: vec![2, 1],
            },
            SparseChallenge {
                positions: vec![3, 7],
                coeffs: vec![-1, 2],
            },
            SparseChallenge {
                positions: vec![0, 4],
                coeffs: vec![1, -2],
            },
            SparseChallenge {
                positions: vec![2, 6],
                coeffs: vec![3, 1],
            },
        ],
        left_len: 2,
        right_len: 4,
        num_claims: 2,
    };
    let claim_idx = 1;
    let u_weights = vec![F::from_i64(3), -F::from_i64(2)];
    let v_weights = vec![F::from_i64(5), F::zero(), -F::from_i64(7), F::from_i64(11)];
    let alpha = F::from_u64(13);
    let alpha_pows = scalar_powers::<F, TD>(alpha);

    let got = tensor
        .eval_factored_aggregate_at_pows::<F, F, TD>(claim_idx, &u_weights, &v_weights, &alpha_pows)
        .unwrap();

    let mut expected = F::zero();
    for (p, &u) in u_weights.iter().enumerate() {
        for (q, &v) in v_weights.iter().enumerate() {
            let block_idx =
                claim_idx * tensor.left_len * tensor.right_len + p * tensor.right_len + q;
            let (_, _, left, right) = tensor.factors_for_logical_block(block_idx).unwrap();
            expected += u * v * tensor_product_eval::<F, TD>(left, right, &alpha_pows);
        }
    }

    assert_eq!(got, expected);
}

#[test]
fn tensor_evals_at_pows_match_ring_product_reference() {
    const TD: usize = 8;
    let tensor = TensorChallenges {
        left: vec![
            SparseChallenge {
                positions: vec![0, 3],
                coeffs: vec![1, -2],
            },
            SparseChallenge {
                positions: vec![2, 7],
                coeffs: vec![2, 1],
            },
        ],
        right: vec![
            SparseChallenge {
                positions: vec![1, 6],
                coeffs: vec![-1, 2],
            },
            SparseChallenge {
                positions: vec![0, 5],
                coeffs: vec![3, -1],
            },
        ],
        left_len: 2,
        right_len: 2,
        num_claims: 1,
    };
    let alpha_pows = scalar_powers::<F, TD>(F::from_u64(13));

    let got = tensor.evals_at_pows::<F, F>(&alpha_pows).unwrap();
    let expected = (0..tensor.total_blocks().unwrap())
        .map(|block_idx| {
            let (_, _, left, right) = tensor.factors_for_logical_block(block_idx).unwrap();
            tensor_product_eval::<F, TD>(left, right, &alpha_pows)
        })
        .collect::<Vec<_>>();

    assert_eq!(got, expected);
}

#[test]
fn tensor_product_only_formula_is_not_exact_for_generic_alpha() {
    // The naive product formula ignores the negacyclic wrap term. At
    // `alpha = 5, D = 2` the wrap term `α^D + 1` is non-zero, so the exact
    // aggregate must differ from the bare product of evaluations.
    const TD: usize = 2;
    let tensor = TensorChallenges {
        left: vec![SparseChallenge {
            positions: vec![1],
            coeffs: vec![1],
        }],
        right: vec![SparseChallenge {
            positions: vec![1],
            coeffs: vec![1],
        }],
        left_len: 1,
        right_len: 1,
        num_claims: 1,
    };
    let alpha = F::from_u64(5);
    let alpha_pows = scalar_powers::<F, TD>(alpha);
    let weights = [F::one()];

    let exact = tensor
        .eval_factored_aggregate_at_pows::<F, F, TD>(0, &weights, &weights, &alpha_pows)
        .unwrap();
    let product_only = tensor.left[0].eval_at_pows::<F, F>(&alpha_pows).unwrap()
        * tensor.right[0].eval_at_pows::<F, F>(&alpha_pows).unwrap();

    assert_eq!(exact, -F::one());
    assert_ne!(exact, product_only);
}

#[test]
fn tensor_exact_aggregate_collapses_to_product_at_negacyclic_root() {
    // When `alpha^D + 1 == 0` the negacyclic wrap term vanishes, so the
    // exact aggregate degenerates to the bare product of evaluations.
    const TD: usize = 2;
    let tensor = TensorChallenges {
        left: vec![SparseChallenge {
            positions: vec![1],
            coeffs: vec![1],
        }],
        right: vec![SparseChallenge {
            positions: vec![1],
            coeffs: vec![1],
        }],
        left_len: 1,
        right_len: 1,
        num_claims: 1,
    };
    let alpha = F::from_u64(983_270_775);
    let alpha_pows = scalar_powers::<F, TD>(alpha);
    let alpha_pow_d_plus_one = alpha_pows[TD - 1] * alpha + F::one();
    let weights = [F::one()];

    assert_eq!(alpha_pow_d_plus_one, F::zero());
    let exact = tensor
        .eval_factored_aggregate_at_pows::<F, F, TD>(0, &weights, &weights, &alpha_pows)
        .unwrap();
    let product_only = tensor.left[0].eval_at_pows::<F, F>(&alpha_pows).unwrap()
        * tensor.right[0].eval_at_pows::<F, F>(&alpha_pows).unwrap();

    assert_eq!(exact, product_only);
}
