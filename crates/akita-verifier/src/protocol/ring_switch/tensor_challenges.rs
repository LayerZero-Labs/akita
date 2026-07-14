#[cfg(test)]
use akita_algebra::eq_poly::EqPolynomial;
#[cfg(test)]
use akita_algebra::offset_eq::summarize_pow2_block_carries;
use akita_challenges::TensorChallenges as TensorChallengeSet;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};

/// Challenge evaluations used by relation-matrix challenge replay.
#[derive(Clone)]
pub(crate) enum PreparedChallengeEvals<F: FieldCore> {
    Flat(Vec<F>),
    Tensor {
        challenges: TensorChallengeSet,
        #[cfg_attr(not(test), allow(dead_code))]
        alpha_pows: Vec<F>,
        exact_evals: Vec<F>,
    },
}

impl<F: FieldCore> PreparedChallengeEvals<F> {
    #[cfg(test)]
    pub(crate) fn as_flat(&self) -> Option<&[F]> {
        match self {
            Self::Flat(c_alphas) => Some(c_alphas),
            Self::Tensor { .. } => None,
        }
    }

    pub(crate) fn eval_at<Base>(
        &self,
        claim: usize,
        block: usize,
        live_fold_count: usize,
    ) -> Result<F, AkitaError>
    where
        Base: FieldCore + FromPrimitiveInt,
        F: MulBase<Base>,
    {
        if block >= live_fold_count {
            return Err(AkitaError::InvalidInput(
                "challenge block index out of range".into(),
            ));
        }
        match self {
            Self::Flat(c_alphas) => c_alphas
                .get(
                    claim
                        .checked_mul(live_fold_count)
                        .and_then(|base| base.checked_add(block))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("challenge index overflow".into())
                        })?,
                )
                .copied()
                .ok_or(AkitaError::InvalidProof),
            Self::Tensor {
                challenges,
                exact_evals,
                ..
            } => {
                if claim >= challenges.num_claims
                    || challenges.live_folds_per_claim != live_fold_count
                    || challenges.fold_low_len == 0
                {
                    return Err(AkitaError::InvalidSetup(
                        "tensor challenge shape does not match witness blocks".into(),
                    ));
                }
                exact_evals
                    .get(
                        claim
                            .checked_mul(live_fold_count)
                            .and_then(|base| base.checked_add(block))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("tensor challenge index overflow".into())
                            })?,
                    )
                    .copied()
                    .ok_or(AkitaError::InvalidProof)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn summarize_all_block_carries<Base, const D: usize>(
        &self,
        num_claims: usize,
        x_low_challenges: &[F],
        eq_low: &[F],
        offset_low: usize,
        live_fold_count: usize,
    ) -> Result<Vec<[F; 2]>, AkitaError>
    where
        Base: FieldCore + FromPrimitiveInt,
        F: MulBase<Base>,
    {
        match self {
            Self::Flat(c_alphas) => (0..num_claims)
                .map(|claim_idx| {
                    let start = claim_idx.checked_mul(live_fold_count).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "flat challenge summary offset overflow".to_string(),
                        )
                    })?;
                    let end = start.checked_add(live_fold_count).ok_or_else(|| {
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
                ..
            } => summarize_tensor_all_block_carries::<Base, F, D>(
                challenges,
                num_claims,
                x_low_challenges,
                offset_low,
                live_fold_count,
                alpha_pows,
            ),
        }
    }

    /// Per-chunk generalization of [`Self::summarize_all_block_carries`].
    ///
    /// Summarizes the per-claim two-bucket `c_alpha` block carries for one
    /// chunk's block window `[global_block_base, global_block_base +
    /// blocks_per_chunk)`, peeling `blocks_per_chunk` instead of the full
    /// `live_fold_count`. `global_block_base = 0` and `blocks_per_chunk = live_fold_count`
    /// reproduce [`Self::summarize_all_block_carries`] exactly (the single-chunk
    /// case). The flat-challenge path reads the global-block-indexed slice
    /// `c_alpha[claim·live_fold_count + global_block_base ..][..blocks_per_chunk]`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError`] for an out-of-range window, an `eq_low`/window
    /// length mismatch, or a chunked (`blocks_per_chunk < live_fold_count`) tensor
    /// challenge set (the factored chunk window is a follow-up).
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    pub(crate) fn summarize_chunk_block_carries<Base, const D: usize>(
        &self,
        num_claims: usize,
        x_low_challenges: &[F],
        eq_low: &[F],
        offset_low: usize,
        global_block_base: usize,
        blocks_per_chunk: usize,
        live_fold_count: usize,
    ) -> Result<Vec<[F; 2]>, AkitaError>
    where
        Base: FieldCore + FromPrimitiveInt,
        F: MulBase<Base>,
    {
        match self {
            Self::Flat(c_alphas) => (0..num_claims)
                .map(|claim_idx| {
                    let claim_start = claim_idx.checked_mul(live_fold_count).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "chunk challenge summary claim offset overflow".to_string(),
                        )
                    })?;
                    let start = claim_start.checked_add(global_block_base).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "chunk challenge summary window offset overflow".to_string(),
                        )
                    })?;
                    let end = start.checked_add(blocks_per_chunk).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "chunk challenge summary window end overflow".to_string(),
                        )
                    })?;
                    let values = c_alphas.get(start..end).ok_or(AkitaError::InvalidSize {
                        expected: end,
                        actual: c_alphas.len(),
                    })?;
                    summarize_pow2_block_carries(eq_low, offset_low, values)
                })
                .collect(),
            Self::Tensor { .. }
                if blocks_per_chunk == live_fold_count && global_block_base == 0 =>
            {
                self.summarize_all_block_carries::<Base, D>(
                    num_claims,
                    x_low_challenges,
                    eq_low,
                    offset_low,
                    live_fold_count,
                )
            }
            Self::Tensor { .. } => Err(AkitaError::InvalidInput(
                "chunked tensor challenge summaries are not implemented".to_string(),
            )),
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn summarize_tensor_all_block_carries<Base, F, const D: usize>(
    challenges: &TensorChallengeSet,
    num_claims: usize,
    x_low_challenges: &[F],
    offset_low: usize,
    live_fold_count: usize,
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
    if challenges.live_folds_per_claim != live_fold_count {
        return Err(AkitaError::InvalidSize {
            expected: live_fold_count,
            actual: challenges.live_folds_per_claim,
        });
    }
    if !challenges.fold_high_len().is_power_of_two() || !challenges.fold_low_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "tensor challenge dimensions must be powers of two".to_string(),
        ));
    }
    if offset_low >= live_fold_count {
        return Err(AkitaError::InvalidInput(format!(
            "low offset {offset_low} out of range for {live_fold_count} blocks"
        )));
    }

    let right_bits = challenges.fold_low_len.trailing_zeros() as usize;
    let left_bits = challenges.fold_high_len().trailing_zeros() as usize;
    if x_low_challenges.len() != right_bits + left_bits {
        return Err(AkitaError::InvalidSize {
            expected: right_bits + left_bits,
            actual: x_low_challenges.len(),
        });
    }

    let eq_right = EqPolynomial::evals(&x_low_challenges[..right_bits])?;
    let eq_left = EqPolynomial::evals(&x_low_challenges[right_bits..])?;
    let right_mask = challenges.fold_low_len - 1;
    let left_mask = challenges.fold_high_len() - 1;
    let offset_right = offset_low & right_mask;
    let offset_left = offset_low >> right_bits;

    let mut out = vec![[F::zero(), F::zero()]; num_claims];
    let mut v_weights = vec![F::zero(); challenges.fold_low_len];
    let mut u_weights = vec![F::zero(); challenges.fold_high_len()];
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
        let fold_high_len = 4usize;
        let fold_low_len = 4usize;
        let live_fold_count = fold_high_len * fold_low_len;
        let set = TensorChallengeSet {
            fold_high: vec![
                sparse(&[0, 6], &[1, -1]),
                sparse(&[1, 7], &[1, 1]),
                sparse(&[3, 12], &[-1, 1]),
                sparse(&[2, 9], &[1, -1]),
                sparse(&[5, 10], &[1, 1]),
                sparse(&[4, 8], &[-1, -1]),
                sparse(&[11, 13], &[1, 1]),
                sparse(&[15, 30], &[1, -1]),
            ],
            fold_low: vec![
                sparse(&[0], &[1]),
                sparse(&[2], &[-1]),
                sparse(&[4], &[1]),
                sparse(&[6], &[-1]),
                sparse(&[8], &[1]),
                sparse(&[10], &[-1]),
                sparse(&[12], &[1]),
                sparse(&[14], &[-1]),
            ],
            live_folds_per_claim: live_fold_count,
            fold_low_len,
            num_claims,
        };
        let tensor_challenges = Challenges::from_tensor::<D>(set.clone()).unwrap();

        let alpha = F::from_u64(11);
        let alpha_pows = scalar_powers(alpha, D);
        let flat_evals = tensor_challenges
            .evals_at_pows::<F, F>(&alpha_pows)
            .unwrap();
        assert_eq!(flat_evals.len(), num_claims * live_fold_count);

        let flat = PreparedChallengeEvals::Flat(flat_evals);
        let factored = PreparedChallengeEvals::Tensor {
            challenges: set,
            alpha_pows,
            exact_evals: flat.as_flat().unwrap().to_vec(),
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
            for offset_low in 0..live_fold_count {
                let got_factored = factored
                    .summarize_all_block_carries::<F, D>(
                        num_claims,
                        &x_low,
                        &eq_low,
                        offset_low,
                        live_fold_count,
                    )
                    .unwrap();
                let got_flat = flat
                    .summarize_all_block_carries::<F, D>(
                        num_claims,
                        &x_low,
                        &eq_low,
                        offset_low,
                        live_fold_count,
                    )
                    .unwrap();
                assert_eq!(
                    got_factored, got_flat,
                    "factored summary mismatch for x_low={x_low:?}, offset_low={offset_low}"
                );
            }
        }
    }

    #[test]
    fn chunk_summary_indexes_global_block_windows() {
        type F = Fp32<251>;
        const D: usize = 32;

        let num_claims = 2usize;
        let live_fold_count = 8usize;
        let flat = PreparedChallengeEvals::<F>::Flat(
            (0..num_claims * live_fold_count)
                .map(|idx| F::from_u64(17 + idx as u64))
                .collect(),
        );

        for w in [1usize, 2, 4, 8] {
            let bpc = live_fold_count / w;
            let bits = bpc.trailing_zeros() as usize;
            let x_low: Vec<F> = (0..bits).map(|i| F::from_u64(3 + i as u64)).collect();
            let eq_low = EqPolynomial::evals(&x_low).unwrap();
            for offset_low in 0..bpc {
                for chunk in 0..w {
                    let global_block_base = chunk * bpc;
                    let got = flat
                        .summarize_chunk_block_carries::<F, D>(
                            num_claims,
                            &x_low,
                            &eq_low,
                            offset_low,
                            global_block_base,
                            bpc,
                            live_fold_count,
                        )
                        .unwrap();
                    // Expected: peel each claim's window slice directly.
                    let c_alphas = flat.as_flat().unwrap();
                    for (claim_idx, got_terms) in got.iter().enumerate() {
                        let start = claim_idx * live_fold_count + global_block_base;
                        let expected = summarize_pow2_block_carries(
                            &eq_low,
                            offset_low,
                            &c_alphas[start..start + bpc],
                        )
                        .unwrap();
                        assert_eq!(
                            *got_terms, expected,
                            "W={w} chunk={chunk} claim={claim_idx} offset_low={offset_low}"
                        );
                    }
                }
            }
        }

        // W=1 (whole-block window) equals the established summary.
        let bits = live_fold_count.trailing_zeros() as usize;
        let x_low: Vec<F> = (0..bits).map(|i| F::from_u64(5 + i as u64)).collect();
        let eq_low = EqPolynomial::evals(&x_low).unwrap();
        for offset_low in 0..live_fold_count {
            let chunked = flat
                .summarize_chunk_block_carries::<F, D>(
                    num_claims,
                    &x_low,
                    &eq_low,
                    offset_low,
                    0,
                    live_fold_count,
                    live_fold_count,
                )
                .unwrap();
            let whole = flat
                .summarize_all_block_carries::<F, D>(
                    num_claims,
                    &x_low,
                    &eq_low,
                    offset_low,
                    live_fold_count,
                )
                .unwrap();
            assert_eq!(
                chunked, whole,
                "W=1 chunk summary must equal whole-block summary"
            );
        }
    }
}
