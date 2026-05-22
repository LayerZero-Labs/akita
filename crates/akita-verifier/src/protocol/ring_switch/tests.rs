use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_field::Fp32;
use akita_types::SisModulusFamily;

type F = Fp32<251>;
const D: usize = 32;

fn stage1_config() -> SparseChallengeConfig {
    SparseChallengeConfig::Uniform {
        weight: 1,
        nonzero_coeffs: vec![1],
    }
}

fn empty_flat_challenges() -> TensorChallenges {
    TensorChallenges::Flat(Vec::new())
}

#[test]
fn ring_switch_prepare_rejects_invalid_log_basis() {
    let lp = LevelParams::params_only(SisModulusFamily::Q32, D, 0, 1, 1, 1, stage1_config());
    let challenges = empty_flat_challenges();
    let err = match prepare_ring_switch_row_eval::<F, F, D>(
        &challenges,
        F::one(),
        &lp,
        &[],
        &[],
        &[],
        &[],
        &[],
        1,
        0,
        &[],
        &[],
    ) {
        Ok(_) => panic!("invalid log_basis should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn ring_switch_prepare_rejects_zero_num_blocks() {
    let lp = LevelParams::params_only(SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config());
    let challenges = empty_flat_challenges();
    let err = match prepare_ring_switch_row_eval::<F, F, D>(
        &challenges,
        F::one(),
        &lp,
        &[],
        &[],
        &[],
        &[],
        &[],
        1,
        0,
        &[],
        &[],
    ) {
        Ok(_) => panic!("zero num_blocks should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn multiplier_block_summary_rejects_malformed_shapes() {
    let eq_low = vec![F::one(); 2];

    let err = summarize_pow2_multiplier_block_carries(&eq_low, 0, 3, |_| Ok(F::one())).unwrap_err();
    assert!(matches!(err, AkitaError::InvalidInput(_)));

    let err = summarize_pow2_multiplier_block_carries(&eq_low, 2, 2, |_| Ok(F::one())).unwrap_err();
    assert!(matches!(err, AkitaError::InvalidInput(_)));

    let err =
        summarize_pow2_multiplier_block_carries(&eq_low[..1], 0, 2, |_| Ok(F::one())).unwrap_err();
    assert!(matches!(err, AkitaError::InvalidSize { .. }));
}

/// Build a small `TensorChallenges::Tensor` value and check that the
/// factored-aggregate carry summary (`PreparedChallengeEvals::Tensor`)
/// agrees with the materialised flat summary (`PreparedChallengeEvals::Flat`)
/// at every legal `(offset_low, x_low)` combination. Guards the
/// equivalence invariant the perf refactor depends on so future kernel
/// changes that drift the two paths apart fail loudly instead of
/// silently producing wrong c_alpha summaries on the tensor path.
#[test]
fn factored_carry_summary_matches_flat_for_tensor_challenges() {
    use akita_algebra::eq_poly::EqPolynomial;
    use akita_algebra::ring::scalar_powers;
    use akita_challenges::{SparseChallenge, TensorChallengeSet};

    type FF = Fp32<251>;
    const DD: usize = 32;

    let sparse = |positions: &[u32], coeffs: &[i8]| -> SparseChallenge {
        SparseChallenge {
            positions: positions.to_vec(),
            coeffs: coeffs.to_vec(),
        }
    };

    let num_claims = 2usize;
    let left_len = 4usize;
    let right_len = 4usize;
    let num_blocks = left_len * right_len; // = 16, power of two

    let left = vec![
        sparse(&[0, 6], &[1, -1]),
        sparse(&[1, 7], &[1, 1]),
        sparse(&[3, 12], &[-1, 1]),
        sparse(&[2, 9], &[1, -1]),
        sparse(&[5, 10], &[1, 1]),
        sparse(&[4, 8], &[-1, -1]),
        sparse(&[11, 13], &[1, 1]),
        sparse(&[15, 30], &[1, -1]),
    ];
    let right = vec![
        sparse(&[0], &[1]),
        sparse(&[2], &[-1]),
        sparse(&[4], &[1]),
        sparse(&[6], &[-1]),
        sparse(&[8], &[1]),
        sparse(&[10], &[-1]),
        sparse(&[12], &[1]),
        sparse(&[14], &[-1]),
    ];
    let set = TensorChallengeSet::new(left, right, left_len, right_len, num_claims).unwrap();
    let tensor_challenges = TensorChallenges::Tensor(set.clone());

    let alpha = FF::from_u64(11);
    let alpha_pows = scalar_powers(alpha, DD);
    let alpha_pow_d_plus_one = alpha_pows[DD - 1] * alpha_pows[1] + FF::one();

    let flat_evals = tensor_challenges
        .evals_at_pows::<FF, FF, DD>(&alpha_pows)
        .expect("flat tensor materialisation");
    assert_eq!(flat_evals.len(), num_claims * num_blocks);

    let flat = PreparedChallengeEvals::Flat(flat_evals.clone());
    let factored = PreparedChallengeEvals::Tensor {
        challenges: set,
        alpha_pows: alpha_pows.clone(),
        alpha_pow_d_plus_one,
    };

    // `right_bits + left_bits = log₂(num_blocks)`. With `num_blocks = 16`
    // we need a 4-element low-bit challenge vector.
    let block_bits = num_blocks.trailing_zeros() as usize;
    assert_eq!(block_bits, 4);
    let x_low_cases = [
        vec![FF::from_u64(2), FF::from_u64(3), FF::zero(), FF::one()],
        vec![
            FF::from_u64(7),
            -FF::from_u64(4),
            FF::from_u64(5),
            FF::from_u64(9),
        ],
        vec![FF::zero(), FF::one(), -FF::from_u64(2), FF::from_u64(3)],
    ];

    for x_low in x_low_cases {
        let eq_low = EqPolynomial::evals(&x_low).expect("eq_low evals");
        for offset_low in 0..num_blocks {
            let got_factored = factored
                .summarize_all_block_carries::<DD>(
                    num_claims, &x_low, &eq_low, offset_low, num_blocks,
                )
                .expect("factored summary");
            let got_flat = flat
                .summarize_all_block_carries::<DD>(
                    num_claims, &x_low, &eq_low, offset_low, num_blocks,
                )
                .expect("flat summary");
            assert_eq!(
                got_factored, got_flat,
                "factored summary mismatch for x_low={x_low:?}, offset_low={offset_low}"
            );
        }
    }
}
