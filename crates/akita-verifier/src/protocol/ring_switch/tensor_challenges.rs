use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::summarize_pow2_block_carries;
use akita_challenges::TensorChallenges as TensorChallengeSet;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};

/// Challenge evaluations used by deferred ring-switch row replay.
#[derive(Clone)]
pub(crate) enum PreparedChallengeEvals<F: FieldCore> {
    Flat(Vec<F>),
    Tensor {
        challenges: TensorChallengeSet,
        alpha_pows: Vec<F>,
    },
}

impl<F: FieldCore> PreparedChallengeEvals<F> {
    pub(crate) fn as_flat(&self) -> Option<&[F]> {
        match self {
            Self::Flat(c_alphas) => Some(c_alphas),
            Self::Tensor { .. } => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn summarize_chunk_block_carries<Base, const D: usize>(
        &self,
        num_claims: usize,
        x_low_challenges: &[F],
        eq_low: &[F],             // length == blocks_per_chunk
        offset_low: usize,        // chunk.offset_e & (blocks_per_chunk - 1)
        global_block_base: usize, // chunk.global_block_base
        blocks_per_chunk: usize,
        num_blocks: usize,
    ) -> Result<Vec<[F; 2]>, AkitaError>
    where
        Base: FieldCore + FromPrimitiveInt,
        F: MulBase<Base>,
    {
        match self {
            Self::Flat(c_alphas) => (0..num_claims)
                .map(|claim_idx| {
                    let claim_start = claim_idx.checked_mul(num_blocks).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "flat chunk summary claim offset overflow".to_string(),
                        )
                    })?;
                    let start = claim_start.checked_add(global_block_base).ok_or_else(|| {
                        AkitaError::InvalidSetup("flat chunk summary start overflow".to_string())
                    })?;
                    let end = start.checked_add(blocks_per_chunk).ok_or_else(|| {
                        AkitaError::InvalidSetup("flat chunk summary end overflow".to_string())
                    })?;
                    let values = c_alphas.get(start..end).ok_or(AkitaError::InvalidSize {
                        expected: end,
                        actual: c_alphas.len(),
                    })?;
                    summarize_pow2_block_carries(eq_low, offset_low, values)
                })
                .collect(),
            // Full range only → legacy tensor summary. Call the helper DIRECTLY to
            // avoid recursion with the wrapper below.
            Self::Tensor {
                challenges,
                alpha_pows,
            } if global_block_base == 0 && blocks_per_chunk == num_blocks => {
                summarize_tensor_all_block_carries::<Base, F, D>(
                    challenges,
                    num_claims,
                    x_low_challenges,
                    offset_low,
                    num_blocks,
                    alpha_pows,
                )
            }
            Self::Tensor { .. } => Err(AkitaError::InvalidInput(
                "chunked tensor challenge summaries are not implemented".to_string(),
            )),
        }
    }

    pub(crate) fn summarize_all_block_carries<Base, const D: usize>(
        &self,
        num_claims: usize,
        x_low_challenges: &[F],
        eq_low: &[F],
        offset_low: usize,
        num_blocks: usize,
    ) -> Result<Vec<[F; 2]>, AkitaError>
    where
        Base: FieldCore + FromPrimitiveInt,
        F: MulBase<Base>,
    {
        match self {
            Self::Flat(c_alphas) => (0..num_claims)
                .map(|claim_idx| {
                    let start = claim_idx.checked_mul(num_blocks).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "flat challenge summary offset overflow".to_string(),
                        )
                    })?;
                    let end = start.checked_add(num_blocks).ok_or_else(|| {
                        AkitaError::InvalidSetup("flat challenge summary end overflow".to_string())
                    })?;
                    let values = c_alphas.get(start..end).ok_or(AkitaError::InvalidSize {
                        expected: end,
                        actual: c_alphas.len(),
                    })?;
                    summarize_pow2_block_carries(eq_low, offset_low, values)
                })
                .collect(),
            Self::Tensor {
                challenges,
                alpha_pows,
            } => summarize_tensor_all_block_carries::<Base, F, D>(
                challenges,
                num_claims,
                x_low_challenges,
                offset_low,
                num_blocks,
                alpha_pows,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn summarize_tensor_all_block_carries<Base, F, const D: usize>(
    challenges: &TensorChallengeSet,
    num_claims: usize,
    x_low_challenges: &[F],
    offset_low: usize,
    num_blocks: usize,
    alpha_pows: &[F],
) -> Result<Vec<[F; 2]>, AkitaError>
where
    Base: FieldCore + FromPrimitiveInt,
    F: FieldCore + MulBase<Base>,
{
    if num_claims > challenges.num_claims {
        return Err(AkitaError::InvalidSize {
            expected: challenges.num_claims,
            actual: num_claims,
        });
    }
    if challenges.left_len.checked_mul(challenges.right_len) != Some(num_blocks) {
        return Err(AkitaError::InvalidSize {
            expected: num_blocks,
            actual: challenges.left_len.saturating_mul(challenges.right_len),
        });
    }
    if !challenges.left_len.is_power_of_two() || !challenges.right_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "tensor challenge dimensions must be powers of two".to_string(),
        ));
    }
    if offset_low >= num_blocks {
        return Err(AkitaError::InvalidInput(format!(
            "low offset {offset_low} out of range for {num_blocks} blocks"
        )));
    }

    let right_bits = challenges.right_len.trailing_zeros() as usize;
    let left_bits = challenges.left_len.trailing_zeros() as usize;
    if x_low_challenges.len() != right_bits + left_bits {
        return Err(AkitaError::InvalidSize {
            expected: right_bits + left_bits,
            actual: x_low_challenges.len(),
        });
    }

    let eq_right = EqPolynomial::evals(&x_low_challenges[..right_bits])?;
    let eq_left = EqPolynomial::evals(&x_low_challenges[right_bits..])?;
    let right_mask = challenges.right_len - 1;
    let left_mask = challenges.left_len - 1;
    let offset_right = offset_low & right_mask;
    let offset_left = offset_low >> right_bits;

    let mut out = vec![[F::zero(), F::zero()]; num_claims];
    let mut v_weights = vec![F::zero(); challenges.right_len];
    let mut u_weights = vec![F::zero(); challenges.left_len];
    for carry_q in 0..=1 {
        v_weights.fill(F::zero());
        let mut has_v_weight = false;
        for (q, v_weight) in v_weights.iter_mut().enumerate() {
            let shifted = offset_right + q;
            if (shifted >> right_bits) == carry_q {
                *v_weight = eq_right[shifted & right_mask];
                has_v_weight |= !v_weight.is_zero();
            }
        }
        if !has_v_weight {
            continue;
        }

        for final_carry in 0..=1 {
            u_weights.fill(F::zero());
            let mut has_u_weight = false;
            for (p, u_weight) in u_weights.iter_mut().enumerate() {
                let shifted = offset_left + p + carry_q;
                if (shifted >> left_bits) == final_carry {
                    *u_weight = eq_left[shifted & left_mask];
                    has_u_weight |= !u_weight.is_zero();
                }
            }
            if !has_u_weight {
                continue;
            }
            for (claim_idx, out_terms) in out.iter_mut().enumerate() {
                out_terms[final_carry] += challenges
                    .eval_factored_aggregate_at_pows::<Base, F, D>(
                        claim_idx, &u_weights, &v_weights, alpha_pows,
                    )?;
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ring::scalar_powers;
    use akita_challenges::{Challenges, SparseChallenge};
    use akita_field::Fp32;

    #[test]
    fn factored_carry_summary_matches_flat_for_tensor_challenges() {
        type F = Fp32<251>;
        const D: usize = 32;

        let sparse = |positions: &[u32], coeffs: &[i8]| -> SparseChallenge {
            SparseChallenge {
                positions: positions.to_vec(),
                coeffs: coeffs.to_vec(),
            }
        };

        let num_claims = 2usize;
        let left_len = 4usize;
        let right_len = 4usize;
        let num_blocks = left_len * right_len;
        let set = TensorChallengeSet {
            left: vec![
                sparse(&[0, 6], &[1, -1]),
                sparse(&[1, 7], &[1, 1]),
                sparse(&[3, 12], &[-1, 1]),
                sparse(&[2, 9], &[1, -1]),
                sparse(&[5, 10], &[1, 1]),
                sparse(&[4, 8], &[-1, -1]),
                sparse(&[11, 13], &[1, 1]),
                sparse(&[15, 30], &[1, -1]),
            ],
            right: vec![
                sparse(&[0], &[1]),
                sparse(&[2], &[-1]),
                sparse(&[4], &[1]),
                sparse(&[6], &[-1]),
                sparse(&[8], &[1]),
                sparse(&[10], &[-1]),
                sparse(&[12], &[1]),
                sparse(&[14], &[-1]),
            ],
            left_len,
            right_len,
            num_claims,
        };
        let tensor_challenges = Challenges::from_tensor::<D>(set.clone()).unwrap();

        let alpha = F::from_u64(11);
        let alpha_pows = scalar_powers(alpha, D);
        let flat_evals = tensor_challenges
            .evals_at_pows::<F, F, D>(&alpha_pows)
            .unwrap();
        assert_eq!(flat_evals.len(), num_claims * num_blocks);

        let flat = PreparedChallengeEvals::Flat(flat_evals);
        let factored = PreparedChallengeEvals::Tensor {
            challenges: set,
            alpha_pows,
        };

        let x_low_cases = [
            vec![F::from_u64(2), F::from_u64(3), F::zero(), F::one()],
            vec![
                F::from_u64(7),
                -F::from_u64(4),
                F::from_u64(5),
                F::from_u64(9),
            ],
            vec![F::zero(), F::one(), -F::from_u64(2), F::from_u64(3)],
        ];

        for x_low in x_low_cases {
            let eq_low = EqPolynomial::evals(&x_low).unwrap();
            for offset_low in 0..num_blocks {
                let got_factored = factored
                    .summarize_all_block_carries::<F, D>(
                        num_claims, &x_low, &eq_low, offset_low, num_blocks,
                    )
                    .unwrap();
                let got_flat = flat
                    .summarize_all_block_carries::<F, D>(
                        num_claims, &x_low, &eq_low, offset_low, num_blocks,
                    )
                    .unwrap();
                assert_eq!(
                    got_factored, got_flat,
                    "factored summary mismatch for x_low={x_low:?}, offset_low={offset_low}"
                );
            }
        }
    }
}
