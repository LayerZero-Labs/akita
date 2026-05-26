#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_challenges::{
    sample_folding_challenges, sample_sparse_challenges, tensor_left_digest, ChallengeLabels,
    ChallengeShape, Challenges, IntegerChallenge, SparseChallenge, SparseChallengeConfig,
    TensorChallenges,
};
use akita_field::{CanonicalField, FieldCore, Fp64};
use akita_transcript::labels::{
    ABSORB_TENSOR_FOLD_LEFT, CHALLENGE_STAGE1_FOLD, CHALLENGE_TENSOR_FOLD_LEFT,
    CHALLENGE_TENSOR_FOLD_RIGHT, DOMAIN_AKITA_PROTOCOL,
};
use akita_transcript::{AkitaTranscript, Transcript};

/// Stage-1 fold label bundle reused by every tensor-vs-flat sampling test.
fn fold_challenge_labels() -> ChallengeLabels<'static> {
    ChallengeLabels {
        flat: CHALLENGE_STAGE1_FOLD,
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

/// Local helper: convert an integer challenge to dense ring coefficients.
fn integer_challenge_to_dense<F: FieldCore + CanonicalField, const D: usize>(
    c: &IntegerChallenge,
) -> [F; D] {
    let mut out = [F::zero(); D];
    for (&pos, &coeff) in c.positions.iter().zip(c.coeffs.iter()) {
        out[pos as usize] += F::from_i64(i64::from(coeff));
    }
    out
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
fn uniform_sampling_is_deterministic_and_exact_weight() {
    let cfg = SparseChallengeConfig::Uniform {
        weight: 8,
        nonzero_coeffs: vec![-1, 1],
    };

    let mut t1 = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    let mut t2 = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
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

    let mut t1 = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    let mut t2 = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
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
    let mut t = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
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

    let mut t = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
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
    let mut transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
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

    let mut transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
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

#[test]
fn tensor_product_matches_dense_ring_product() {
    const TD: usize = 8;
    let left = SparseChallenge {
        positions: vec![0, 6],
        coeffs: vec![2, -1],
    };
    let right = SparseChallenge {
        positions: vec![3, 5],
        coeffs: vec![1, 4],
    };

    let product = IntegerChallenge::tensor_product::<TD>(&left, &right).unwrap();
    let dense_product = dense_negacyclic_mul(
        &sparse_challenge_to_dense::<F, TD>(&left).unwrap(),
        &sparse_challenge_to_dense::<F, TD>(&right).unwrap(),
    );

    assert_eq!(integer_challenge_to_dense::<F, TD>(&product), dense_product);
}

#[test]
fn tensor_sampling_uses_two_vectors() {
    const TD: usize = 8;
    let cfg = SparseChallengeConfig::Uniform {
        weight: 2,
        nonzero_coeffs: vec![-1, 1],
    };
    let mut transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(7));

    let challenges = sample_folding_challenges::<F, _, TD>(
        &mut transcript,
        8,
        2,
        &cfg,
        &ChallengeShape::Tensor,
        fold_challenge_labels(),
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
    assert_eq!(tensor.expand_integer::<TD>().unwrap().len(), 16);
}

#[test]
fn tensor_sampling_absorbs_left_digest_before_right() {
    const TD: usize = 8;
    let cfg = SparseChallengeConfig::Uniform {
        weight: 2,
        nonzero_coeffs: vec![-1, 1],
    };

    let mut sampled_transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    sampled_transcript.append_field(b"seed", &F::from_u64(0x5151));
    let sampled = sample_folding_challenges::<F, _, TD>(
        &mut sampled_transcript,
        8,
        2,
        &cfg,
        &ChallengeShape::Tensor,
        fold_challenge_labels(),
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
    let left = sample_sparse_challenges::<F, _, TD>(
        &mut manual_transcript,
        CHALLENGE_TENSOR_FOLD_LEFT,
        sampled.left.len(),
        &cfg,
    )
    .unwrap();
    let left_digest =
        tensor_left_digest::<TD>(&left, sampled.left_len, sampled.num_claims).unwrap();
    manual_transcript.append_bytes(ABSORB_TENSOR_FOLD_LEFT, &left_digest);
    let right = sample_sparse_challenges::<F, _, TD>(
        &mut manual_transcript,
        CHALLENGE_TENSOR_FOLD_RIGHT,
        sampled.right.len(),
        &cfg,
    )
    .unwrap();

    // The right factor must be sampled after absorbing the left digest.
    let mut nodigest_transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    nodigest_transcript.append_field(b"seed", &F::from_u64(0x5151));
    let _nodigest_left = sample_sparse_challenges::<F, _, TD>(
        &mut nodigest_transcript,
        CHALLENGE_TENSOR_FOLD_LEFT,
        sampled.left.len(),
        &cfg,
    )
    .unwrap();
    let nodigest_right = sample_sparse_challenges::<F, _, TD>(
        &mut nodigest_transcript,
        CHALLENGE_TENSOR_FOLD_RIGHT,
        sampled.right.len(),
        &cfg,
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
fn tensor_left_digest_rejects_duplicate_positions() {
    const TD: usize = 8;
    let left = vec![SparseChallenge {
        positions: vec![0, 0],
        coeffs: vec![1, -1],
    }];

    let err = tensor_left_digest::<TD>(&left, 1, 1).unwrap_err();

    assert!(matches!(err, akita_field::AkitaError::InvalidInput(msg) if msg.contains("unique")));
}

#[test]
fn tensor_lazy_evals_match_expanded_products() {
    const TD: usize = 8;
    let cfg = SparseChallengeConfig::Uniform {
        weight: 2,
        nonzero_coeffs: vec![-1, 1],
    };
    let mut transcript = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    transcript.append_field(b"seed", &F::from_u64(99));
    let challenges = sample_folding_challenges::<F, _, TD>(
        &mut transcript,
        8,
        1,
        &cfg,
        &ChallengeShape::Tensor,
        fold_challenge_labels(),
    )
    .unwrap();

    let alpha_pows = scalar_powers::<F, TD>(F::from_u64(5));
    let lazy = challenges.evals_at_pows::<F, F, TD>(&alpha_pows).unwrap();
    let Challenges::Tensor { factored } = &challenges else {
        panic!("expected tensor challenges");
    };
    let expanded = factored
        .expand_integer::<TD>()
        .unwrap()
        .iter()
        .map(|challenge| challenge.eval_at_pows::<F, F, TD>(&alpha_pows))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(lazy, expanded);
}

#[test]
fn tensor_factored_aggregate_matches_expanded_products() {
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

    let expanded = tensor.expand_integer::<TD>().unwrap();
    let start = claim_idx * tensor.left_len * tensor.right_len;
    let mut expected = F::zero();
    for (p, &u) in u_weights.iter().enumerate() {
        for (q, &v) in v_weights.iter().enumerate() {
            let idx = start + p * tensor.right_len + q;
            expected += u * v * expanded[idx].eval_at_pows::<F, F, TD>(&alpha_pows).unwrap();
        }
    }

    assert_eq!(got, expected);
}

#[test]
fn tensor_evals_at_pows_match_expanded_integer_reference() {
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

    let got = tensor.evals_at_pows::<F, F, TD>(&alpha_pows).unwrap();
    let expected = tensor
        .expand_integer::<TD>()
        .unwrap()
        .iter()
        .map(|challenge| challenge.eval_at_pows::<F, F, TD>(&alpha_pows).unwrap())
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
    let product_only = tensor.left[0]
        .eval_at_pows::<F, F, TD>(&alpha_pows)
        .unwrap()
        * tensor.right[0]
            .eval_at_pows::<F, F, TD>(&alpha_pows)
            .unwrap();

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
    let product_only = tensor.left[0]
        .eval_at_pows::<F, F, TD>(&alpha_pows)
        .unwrap()
        * tensor.right[0]
            .eval_at_pows::<F, F, TD>(&alpha_pows)
            .unwrap();

    assert_eq!(exact, product_only);
}
