use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::summarize_pow2_block_carries;
use akita_challenges::TensorChallengeSet;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};

/// Carrier for the per-block challenge evaluations consumed by
/// [`RingSwitchDeferredRowEval`]'s deferred row-MLE replay.
///
/// `Flat` stores the materialised dense per-block evaluations exactly as
/// `TensorChallenges::Flat::evals_at_pows` would produce them: one entry per
/// `(claim, block)`, packed claim-major. `Tensor` keeps the factored
/// `(left, right)` challenge set together with the cached
/// `alpha_pows` / `alpha_pow_d_plus_one` so consumers can summarise carry
/// terms via the factored aggregate without ever materialising a length-
/// `num_claims * num_blocks` vector.
pub(crate) enum PreparedChallengeEvals<F: FieldCore> {
    /// Dense per-block evaluations packed as `[claim_0_block_0, ..., claim_0_block_{B-1}, claim_1_block_0, ...]`.
    Flat(Vec<F>),
    /// Factored tensor challenges plus the cached `alpha` powers needed for
    /// `eval_factored_aggregate_at_pows`. `alpha_pow_d_plus_one = α^D + 1`,
    /// pre-derived from `alpha_pows` because the consumer reuses it on every
    /// `(carry_q, final_carry)` combination.
    Tensor {
        challenges: TensorChallengeSet,
        alpha_pows: Vec<F>,
        alpha_pow_d_plus_one: F,
    },
}

impl<F: FieldCore> PreparedChallengeEvals<F> {
    /// Return the materialised flat per-block evaluations, or `None` if the
    /// challenges are stored in factored tensor form. Test-only helper used
    /// by the `slice_mle` regression tests that compare against an expanded
    /// reference; production code consumes through
    /// [`Self::summarize_all_block_carries`] instead.
    #[cfg(test)]
    pub(crate) fn as_flat(&self) -> Option<&[F]> {
        match self {
            Self::Flat(c_alphas) => Some(c_alphas),
            Self::Tensor { .. } => None,
        }
    }

    /// Summarise `(carry=0, carry=1)` block-carry pairs for every claim.
    ///
    /// Returns `Vec<[F; 2]>` of length `num_claims`. The shape matches what
    /// `(0..num_claims).map(|claim| summarize_pow2_block_carries(eq_low,
    /// offset_low, &flat_c_alphas[claim * num_blocks .. (claim + 1) *
    /// num_blocks]))` would produce, but the `Tensor` variant computes each
    /// `[F; 2]` in `O(√num_blocks · D)` work via
    /// `TensorChallengeSet::eval_factored_aggregate_at_pows` instead of
    /// touching every block index.
    ///
    /// `x_low_challenges` are the raw low-bit challenge scalars that produce
    /// `eq_low = EqPolynomial::evals(x_low_challenges)`. The `Flat` arm only
    /// consults `eq_low`; the `Tensor` arm splits `x_low_challenges` into the
    /// left/right halves matching the tensor factor sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_claims` is out of range for the underlying
    /// challenges, if `offset_low >= num_blocks`, or if the factored
    /// aggregate rejects the supplied weights/powers.
    pub(crate) fn summarize_all_block_carries<const D: usize>(
        &self,
        num_claims: usize,
        x_low_challenges: &[F],
        eq_low: &[F],
        offset_low: usize,
        num_blocks: usize,
    ) -> Result<Vec<[F; 2]>, AkitaError>
    where
        F: FromPrimitiveInt + MulBase<F>,
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
                alpha_pow_d_plus_one,
            } => summarize_tensor_all_block_carries::<F, D>(
                challenges,
                num_claims,
                x_low_challenges,
                offset_low,
                num_blocks,
                alpha_pows,
                *alpha_pow_d_plus_one,
            ),
        }
    }
}

/// Factored-aggregate replacement for the `(0..num_claims)`-loop of
/// `summarize_pow2_block_carries` calls.
///
/// Each block index `u ∈ [0, num_blocks)` decomposes as
/// `u = p * right_len + q`. The weights `eq_low[(offset_low + u) mod
/// num_blocks]` factor across `(p, q)` via the two half-eq tables `eq_left`,
/// `eq_right` derived from the high and low halves of `x_low_challenges`.
/// For every `(carry_q ∈ {0, 1}, final_carry ∈ {0, 1})` combination we build
/// the matching `(u_weights, v_weights)` pair, call
/// `eval_factored_aggregate_at_pows` once per claim, and accumulate into the
/// per-claim `[carry_0, carry_1]` output.
///
/// Total work is `O((left_len + right_len) · D · num_claims)` versus the flat
/// path's `O(num_blocks · num_claims)` — the win grows with `num_blocks`.
#[allow(clippy::too_many_arguments)]
fn summarize_tensor_all_block_carries<F, const D: usize>(
    challenges: &TensorChallengeSet,
    num_claims: usize,
    x_low_challenges: &[F],
    offset_low: usize,
    num_blocks: usize,
    alpha_pows: &[F],
    alpha_pow_d_plus_one: F,
) -> Result<Vec<[F; 2]>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt + MulBase<F>,
{
    if num_claims > challenges.num_claims() {
        return Err(AkitaError::InvalidSize {
            expected: challenges.num_claims(),
            actual: num_claims,
        });
    }
    let (left_bits, right_bits) = challenges.validate_power_of_two_dimensions(num_blocks)?;
    if offset_low >= num_blocks {
        return Err(AkitaError::InvalidInput(format!(
            "low offset {offset_low} out of range for {num_blocks} blocks"
        )));
    }

    if x_low_challenges.len() != right_bits + left_bits {
        return Err(AkitaError::InvalidSize {
            expected: right_bits + left_bits,
            actual: x_low_challenges.len(),
        });
    }

    let eq_right = EqPolynomial::evals(&x_low_challenges[..right_bits])?;
    let eq_left = EqPolynomial::evals(&x_low_challenges[right_bits..])?;
    let right_mask = challenges.right_len() - 1;
    let left_mask = challenges.left_len() - 1;
    let offset_right = offset_low & right_mask;
    let offset_left = offset_low >> right_bits;

    let mut out = vec![[F::zero(), F::zero()]; num_claims];
    let mut v_weights = vec![F::zero(); challenges.right_len()];
    let mut u_weights = vec![F::zero(); challenges.left_len()];
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
                out_terms[final_carry] += challenges.eval_factored_aggregate_at_pows::<F, F, D>(
                    claim_idx,
                    &u_weights,
                    &v_weights,
                    alpha_pows,
                    alpha_pow_d_plus_one,
                )?;
            }
        }
    }

    Ok(out)
}
