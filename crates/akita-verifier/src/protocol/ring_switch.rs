//! Verifier-side ring-switch replay.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{
    eval_offset_eq_peeled_carry_terms, eval_offset_eq_tensor, summarize_pow2_block_carries,
};
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_challenges::{Stage1Challenges, TensorStage1Challenges};
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_transcript::labels::{
    ABSORB_SUMCHECK_W, CHALLENGE_RING_SWITCH, CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_challenge_scalars, Transcript};
use akita_types::{
    checked_num_claims_from_group_sizes, gadget_row_scalars, r_decomp_levels,
    validate_opening_points_for_claims, AkitaExpandedSetup, FlatRingVec, GroupLayout, LevelParams,
    MRowLayout, RingMatrixView, RingOpeningPoint,
};

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub struct RingSwitchVerifyOutput<F: FieldCore> {
    /// Prepared data for deferred M-table MLE evaluation.
    pub prepared_m_eval: PreparedMEval<F>,
    /// Evaluation table of alpha powers over the ring-coordinate dimension.
    pub alpha_evals_y: Vec<F>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for the stage-1 sumcheck.
    pub tau0: Vec<F>,
    /// Challenge tau1 for the stage-2 M-row combination.
    pub tau1: Vec<F>,
    /// Basis size `b = 2^log_basis`.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: F,
}

/// Precomputed challenge-derived data for deferred M-table MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
pub struct PreparedMEval<F: FieldCore> {
    alpha: F,
    alpha_pows: Vec<F>,
    challenge_evals: PreparedChallengeEvals<F>,
    eq_tau1: Vec<F>,
    total_blocks: usize,
    num_blocks: usize,
    num_claims: usize,
    depth_open: usize,
    depth_commit: usize,
    depth_fold: usize,
    block_len: usize,
    inner_width: usize,
    log_basis: u32,
    n_a: usize,
    n_d: usize,
    n_b: usize,
    num_commitment_groups: usize,
    group_layouts: Vec<GroupLayout>,
    row_layout: MRowLayout,
    rows: usize,
    z_first: bool,
    claim_to_group: Vec<(usize, usize)>,
    num_points: usize,
    num_eval_rows: usize,
    gamma: Vec<F>,
    claim_to_point: Vec<usize>,
    /// Phase D-full Slice G tier shape for the routed `S` group at this
    /// level (book §5.4). `un_tiered()` (`f = 1`, `k = 1`) keeps the
    /// Slice F single-chunk relation with 5 row families. `f ≥ 2`
    /// activates the book's 10-check-group relation: per-chunk `D` and
    /// `B` sub-relations (groups 1–2, block-diagonal with shared
    /// `D_chunk` / `B_chunk`) plus 5 meta-tier row families (groups
    /// 6–10) for the tier-3 binding `(c_meta, v_meta, u_meta)`. The
    /// per-chunk and meta-tier setup material is verifier-derivable
    /// from the shared matrix at the chunk's row offset + the meta
    /// tier's offset (book line 698 "proof independent of k").
    tier_setup_params: akita_types::TieredSetupParams,
}

/// Additive decomposition of the prepared verifier M-table evaluation.
///
/// The algebraic term contains rows derived from public openings, folding
/// challenges, gadget scalars, and quotient rows. The setup term contains the
/// parts that read the shared D/B/A setup matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedMEvalSplit<F: FieldCore> {
    /// Verifier-computable contribution independent of setup matrix entries.
    pub algebraic: F,
    /// Contribution from shared setup matrix rows.
    pub setup: F,
}

impl<F: FieldCore> PreparedMEvalSplit<F> {
    /// Return the original M-table evaluation represented by this split.
    #[inline]
    pub fn combined(self) -> F {
        self.algebraic + self.setup
    }
}

/// Tier-aware B-row count contributed by a single commitment group.
///
/// Untiered groups contribute `n_B = spec.b_key.row_len()` rows (the
/// standard B-binding family). Tier-marked groups contribute
/// `tier.num_chunks * n_B_chunk` rows under shared `B_chunk` (book §5.4
/// per-chunk B-checks). This helper mirrors
/// [`LevelParams::total_b_row_count`] but operates directly on a
/// [`GroupLayout`] so the M-eval path can iterate `group_layouts` once.
#[inline]
fn group_b_row_count(layout: &GroupLayout) -> usize {
    // After Drift 3 γ-aggregation each tier-marked chunks group carries
    // claim_count = 1 (one aggregated chunks claim under shared B_chunk).
    // The per-group B row count is therefore `claim_count * n_B_g`.
    layout.claim_count * layout.spec.b_key.row_len()
}

#[inline]
fn has_tiered_group(layouts: &[GroupLayout]) -> bool {
    layouts
        .iter()
        .any(|layout| layout.spec.tier.is_some_and(|t| t.is_tiered()))
}

#[inline]
fn group_d_row_count(layout: &GroupLayout, n_d: usize, tiered_relation: bool) -> usize {
    if !tiered_relation {
        return 0;
    }
    // After Drift 3 γ-aggregation each tier-marked chunks group has
    // claim_count = 1, so the per-group D row count is `claim_count * n_d`.
    layout.claim_count * n_d
}

enum PreparedChallengeEvals<F: FieldCore> {
    Flat(Vec<F>),
    Tensor {
        challenges: TensorStage1Challenges,
        alpha_pow_d_plus_one: F,
    },
}

impl<F: FieldCore + CanonicalField> PreparedChallengeEvals<F> {
    fn prepare<const D: usize>(
        challenges: &Stage1Challenges,
        alpha: F,
        alpha_pows: &[F],
    ) -> Result<Self, AkitaError> {
        match challenges {
            Stage1Challenges::Flat(_) => {
                Ok(Self::Flat(challenges.evals_at_pows::<F, D>(alpha_pows)?))
            }
            Stage1Challenges::Tensor(tensor) => {
                if alpha_pows.len() != D {
                    return Err(AkitaError::InvalidSize {
                        expected: D,
                        actual: alpha_pows.len(),
                    });
                }
                if D == 0 {
                    return Err(AkitaError::InvalidInput(
                        "ring dimension must be non-zero".to_string(),
                    ));
                }
                Ok(Self::Tensor {
                    challenges: tensor.clone(),
                    alpha_pow_d_plus_one: alpha_pows[D - 1] * alpha + F::one(),
                })
            }
        }
    }

    fn expanded_evals<const D: usize>(&self, alpha_pows: &[F]) -> Result<Vec<F>, AkitaError> {
        match self {
            Self::Flat(c_alphas) => Ok(c_alphas.clone()),
            Self::Tensor { challenges, .. } => challenges.evals_at_pows::<F, D>(alpha_pows),
        }
    }

    #[inline]
    fn is_tensor(&self) -> bool {
        matches!(self, Self::Tensor { .. })
    }

    #[inline]
    fn tensor_alpha_pow_d_plus_one(&self) -> Option<F> {
        match self {
            Self::Flat(_) => None,
            Self::Tensor {
                alpha_pow_d_plus_one,
                ..
            } => Some(*alpha_pow_d_plus_one),
        }
    }

    fn summarize_all_block_carries<const D: usize>(
        &self,
        num_claims: usize,
        x_low_challenges: &[F],
        eq_low: &[F],
        offset_low: usize,
        num_blocks: usize,
        alpha_pows: &[F],
    ) -> Result<Vec<[F; 2]>, AkitaError> {
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
                    Ok(summarize_pow2_block_carries(eq_low, offset_low, values))
                })
                .collect(),
            Self::Tensor {
                challenges,
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

#[allow(clippy::too_many_arguments)]
fn summarize_tensor_all_block_carries<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &TensorStage1Challenges,
    num_claims: usize,
    x_low_challenges: &[F],
    offset_low: usize,
    num_blocks: usize,
    alpha_pows: &[F],
    alpha_pow_d_plus_one: F,
) -> Result<Vec<[F; 2]>, AkitaError> {
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

    let eq_right = EqPolynomial::evals(&x_low_challenges[..right_bits]);
    let eq_left = EqPolynomial::evals(&x_low_challenges[right_bits..]);
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
                out_terms[final_carry] += challenges.eval_factored_aggregate_at_pows::<F, D>(
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

/// Replay the verifier half of ring switching.
///
/// This handles multiple opening points, arbitrary claim-to-point mapping, and
/// arbitrary commitment grouping. The recursive/single-point path is the
/// `opening_points = [pt]`, `claim_to_point = [0]`,
/// `claim_group_sizes = [1]`, `num_eval_rows = 1` specialization.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or M-eval
/// preparation fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub fn ring_switch_verifier<F, T, const D: usize>(
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    challenges: &Stage1Challenges,
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
    lp: &LevelParams,
    claim_group_sizes: &[usize],
    gamma: &[F],
    num_eval_rows: usize,
) -> Result<RingSwitchVerifyOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
{
    transcript.append_serde(ABSORB_SUMCHECK_W, w_commitment);

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();
    if !lp.groups_are_homogeneous() {
        validate_grouped_opening_points(opening_points, claim_to_point, lp, claim_group_sizes)?;
    } else {
        validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    }

    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    let ring_bits = D.trailing_zeros() as usize;
    let m_rows = lp.m_row_count(num_commitment_groups, num_eval_rows);
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows.next_power_of_two().trailing_zeros() as usize;

    let tau0 = sample_challenge_scalars::<F, T>(transcript, CHALLENGE_TAU0, num_sc_vars);
    let tau1 = sample_challenge_scalars::<F, T>(transcript, CHALLENGE_TAU1, num_i);
    let alpha_evals_y = scalar_powers(alpha, D);
    let prepared_m_eval = prepare_m_eval::<F, D>(
        challenges,
        alpha,
        lp,
        &tau1,
        claim_group_sizes,
        gamma,
        num_eval_rows,
        opening_points.len(),
        claim_to_point,
    )?;

    Ok(RingSwitchVerifyOutput {
        prepared_m_eval,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize << lp.log_basis,
        alpha,
    })
}

/// Prepare deferred verifier M-table evaluation data.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "prepare_m_eval")]
pub fn prepare_m_eval<F: FieldCore + CanonicalField, const D: usize>(
    challenges: &Stage1Challenges,
    alpha: F,
    lp: &LevelParams,
    tau1: &[F],
    claim_group_sizes: &[usize],
    gamma: &[F],
    num_eval_rows: usize,
    opening_points_len: usize,
    claim_to_point: &[usize],
) -> Result<PreparedMEval<F>, AkitaError> {
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    let num_commitment_groups = claim_group_sizes.len();

    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: gamma.len(),
        });
    }

    let group_layouts = lp.group_layouts(claim_group_sizes, num_eval_rows)?;
    let total_blocks = group_layouts
        .last()
        .map(|layout| layout.block_start + layout.claim_count * layout.spec.num_blocks)
        .unwrap_or(0);
    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold;
    let log_basis = lp.log_basis;
    let num_blocks = lp.num_blocks;
    // For heterogeneous groups (e.g. tiered W + k chunks + meta with
    // distinct per-group `num_blocks`), the prover rounds the stage-1
    // challenge count up to the next power of two. Accept >= here so
    // the surplus challenge slots (zero-weighted at row evaluation
    // time) don't fail the consistency check.
    if challenges.logical_len() < total_blocks {
        return Err(AkitaError::InvalidSetup(format!(
            "prepare_m_eval challenge count mismatch: expected at least {total_blocks}, actual {}, claim_group_sizes={claim_group_sizes:?}, lp.num_blocks={}",
            challenges.logical_len(),
            lp.num_blocks
        )));
    }
    let block_len = lp.block_len;
    let inner_width = block_len * depth_commit;
    let num_points = opening_points_len.max(1);
    let row_layout = lp.m_row_layout(num_commitment_groups, num_eval_rows);
    let rows = row_layout.rows;

    let eq_tau1 = EqPolynomial::evals(tau1);
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let challenge_evals = PreparedChallengeEvals::prepare::<D>(challenges, alpha, &alpha_pows)?;

    let z_first = lp.m_vars >= lp.r_vars;

    let claim_to_group: Vec<(usize, usize)> = claim_group_sizes
        .iter()
        .enumerate()
        .flat_map(|(group_idx, &group_size)| {
            (0..group_size).map(move |within_group| (group_idx, within_group))
        })
        .collect();

    // Phase D-full Slice G: pick the tier shape from any `GroupSpec.tier`
    // override on the multi-group LP. The S-group's tier dictates the
    // relation shape: `None`/`f = 1` keeps the existing 5-row-family
    // relation; `f ≥ 2` activates the book §5.4 10-check-group relation.
    // When multiple groups carry overrides they must agree on the tier;
    // otherwise prepare_m_eval rejects loudly so the tiered code path
    // never silently mixes shapes.
    let tier_setup_params = if let Some(groups) = &lp.groups {
        let mut chosen: Option<akita_types::TieredSetupParams> = None;
        for spec in groups {
            if let Some(tier) = spec.tier {
                match chosen {
                    None => chosen = Some(tier),
                    Some(prev) if prev == tier => {}
                    Some(prev) => {
                        return Err(AkitaError::InvalidSetup(format!(
                            "prepare_m_eval: conflicting tier specs across groups: \
                             {prev:?} vs {tier:?}"
                        )));
                    }
                }
            }
        }
        chosen.unwrap_or_else(akita_types::TieredSetupParams::un_tiered)
    } else {
        akita_types::TieredSetupParams::un_tiered()
    };

    Ok(PreparedMEval {
        alpha,
        alpha_pows,
        challenge_evals,
        eq_tau1,
        total_blocks,
        num_blocks,
        num_claims,
        depth_open,
        depth_commit,
        depth_fold,
        block_len,
        inner_width,
        log_basis,
        n_a: lp.a_key.row_len(),
        n_d: lp.d_key.row_len(),
        n_b: lp.b_key.row_len(),
        num_commitment_groups,
        group_layouts,
        row_layout,
        rows,
        z_first,
        claim_to_group,
        num_points,
        num_eval_rows,
        gamma: gamma.to_vec(),
        claim_to_point: claim_to_point.to_vec(),
        tier_setup_params,
    })
}

fn validate_grouped_opening_points<F: FieldCore>(
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    lp: &LevelParams,
    claim_group_sizes: &[usize],
) -> Result<(), AkitaError> {
    if opening_points.is_empty() {
        return Err(AkitaError::InvalidInput(
            "multipoint ring switch requires at least one opening point".to_string(),
        ));
    }
    let layouts = lp.group_layouts(claim_group_sizes, opening_points.len())?;
    let num_claims = checked_num_claims_from_group_sizes(claim_group_sizes)?;
    if claim_to_point.len() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: claim_to_point.len(),
        });
    }
    for layout in &layouts {
        let group_claim_to_point =
            &claim_to_point[layout.claim_start..(layout.claim_start + layout.claim_count)];
        for &point_idx in group_claim_to_point {
            let Some(opening_point) = opening_points.get(point_idx) else {
                return Err(AkitaError::InvalidInput(
                    "multipoint ring switch claim-to-point index out of range".to_string(),
                ));
            };
            if opening_point.a.len() < layout.spec.block_len
                || opening_point.b.len() < layout.spec.num_blocks
            {
                return Err(AkitaError::InvalidInput(
                    "multipoint ring switch grouped opening-point layout mismatch".to_string(),
                ));
            }
        }
    }
    Ok(())
}

impl<F: FieldCore + CanonicalField> PreparedMEval<F> {
    /// Return true when M-eval preparation kept tensor challenge data compact.
    #[inline]
    pub fn challenge_evals_are_tensor(&self) -> bool {
        self.challenge_evals.is_tensor()
    }

    /// Return the tensor correction scalar `alpha^D + 1`, when tensor challenge
    /// data is stored compactly.
    #[inline]
    pub fn tensor_alpha_pow_d_plus_one(&self) -> Option<F> {
        self.challenge_evals.tensor_alpha_pow_d_plus_one()
    }

    /// Expand prepared challenge evaluations into the flat reference order.
    ///
    /// Test-only diagnostic bridge.
    ///
    /// # Errors
    ///
    /// Returns an error if compact tensor challenge expansion or evaluation
    /// fails.
    #[cfg(test)]
    pub fn debug_expanded_challenge_evals<const D: usize>(&self) -> Result<Vec<F>, AkitaError> {
        self.challenge_evals
            .expanded_evals::<D>(self.alpha_pows_for_eval::<D>(self.alpha)?)
    }

    fn alpha_pows_for_eval<const D: usize>(&self, alpha: F) -> Result<&[F], AkitaError> {
        if alpha != self.alpha {
            return Err(AkitaError::InvalidInput(
                "PreparedMEval evaluated with a different ring-switch alpha".to_string(),
            ));
        }
        if self.alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.alpha_pows.len(),
            });
        }
        Ok(&self.alpha_pows)
    }

    /// Evaluate the prepared verifier M-table at the supplied point.
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha` differs from the ring-switch challenge used
    /// during preparation, if the setup matrix cannot be viewed at `D`, or if
    /// an internal offset-eq evaluation receives inconsistent dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the prepared state was built for a layout inconsistent with
    /// the provided setup, opening points, or challenge vector. Callers should
    /// build values through [`prepare_m_eval`] or [`ring_switch_verifier`].
    #[inline]
    pub fn eval_at_point<const D: usize>(
        &self,
        x_challenges: &[F],
        setup: &AkitaExpandedSetup<F>,
        opening_points: &[RingOpeningPoint<F>],
        alpha: F,
    ) -> Result<F, AkitaError> {
        Ok(self
            .eval_split_at_point::<D>(x_challenges, setup, opening_points, alpha)?
            .combined())
    }

    /// Evaluate and decompose the prepared verifier M-table at the supplied
    /// point into algebraic and setup-matrix contributions.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::eval_at_point`].
    #[inline]
    pub fn eval_split_at_point<const D: usize>(
        &self,
        x_challenges: &[F],
        setup: &AkitaExpandedSetup<F>,
        opening_points: &[RingOpeningPoint<F>],
        alpha: F,
    ) -> Result<PreparedMEvalSplit<F>, AkitaError> {
        if !self.uses_homogeneous_outer_layout() {
            return self.eval_split_at_point_grouped::<D>(
                x_challenges,
                setup,
                opening_points,
                alpha,
            );
        }
        let alpha_pows = self.alpha_pows_for_eval::<D>(alpha)?;
        let g1_open = gadget_row_scalars::<F>(self.depth_open, self.log_basis);
        let g1_commit = gadget_row_scalars::<F>(self.depth_commit, self.log_basis);
        let fold_gadget = gadget_row_scalars::<F>(self.depth_fold, self.log_basis);
        let levels = r_decomp_levels::<F>(self.log_basis);
        let r_gadget = gadget_row_scalars::<F>(levels, self.log_basis);

        let stride = setup.seed.max_stride;
        let d_view = setup.shared_matrix.ring_view::<D>(self.n_d, stride);
        let b_view = setup.shared_matrix.ring_view::<D>(self.n_b, stride);
        let a_view = setup.shared_matrix.ring_view::<D>(self.n_a, stride);

        let row_layout = &self.row_layout;
        let consistency_weight = self.eq_tau1[row_layout.original_fold];
        let d_start = row_layout.original_d.start;
        let commitment_row_count = self.n_b * self.num_commitment_groups;
        let b_start = d_start + self.n_d;
        let a_start = b_start + commitment_row_count;
        let public_weights = &self.eq_tau1[row_layout.original_eval.clone()];
        let a_weights = &self.eq_tau1[a_start..(a_start + self.n_a)];

        let total_blocks = self.total_blocks;
        let num_blocks = self.num_blocks;
        let num_claims = self.num_claims;
        let depth_open = self.depth_open;
        let depth_commit = self.depth_commit;
        let depth_fold = self.depth_fold;
        let block_len = self.block_len;
        let inner_width = self.inner_width;
        let n_d = self.n_d;
        let n_b = self.n_b;
        let n_a = self.n_a;
        let rows = self.rows;
        let num_points = self.num_points;
        let eq_tau1 = &self.eq_tau1;
        let d_weights = &eq_tau1[d_start..(d_start + n_d)];
        let claim_to_group = &self.claim_to_group;
        let claim_to_point = &self.claim_to_point;
        let gamma = &self.gamma;

        let w_len = depth_open * total_blocks;
        let t_len = depth_open * n_a * total_blocks;
        let z_total_blocks = num_points * block_len;
        let z_len = depth_fold * depth_commit * z_total_blocks;
        let r_tail_len = rows * levels;

        let is_multi_point = num_points > 1;

        let offset_z = if self.z_first { 0 } else { w_len + t_len };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first { z_len + w_len } else { w_len };
        let block_bits = num_blocks.trailing_zeros() as usize;
        let block_low_eq = EqPolynomial::evals(&x_challenges[..block_bits]);
        let block_offset_low = offset_w & (num_blocks - 1);
        debug_assert_eq!(block_offset_low, offset_t & (num_blocks - 1));

        let opening_point_block_summaries: Vec<[F; 2]> = opening_points
            .iter()
            .map(|opening_point| {
                summarize_pow2_block_carries(&block_low_eq, block_offset_low, &opening_point.b)
            })
            .collect();
        let challenge_block_summaries = self.challenge_evals.summarize_all_block_carries::<D>(
            num_claims,
            &x_challenges[..block_bits],
            &block_low_eq,
            block_offset_low,
            num_blocks,
            alpha_pows,
        )?;

        let mut w_carry_terms = vec![[F::zero(), F::zero()]; num_claims * depth_open];
        for (dig, &g_open) in g1_open.iter().enumerate() {
            let q_base = dig * num_claims;
            for claim_idx in 0..num_claims {
                let q = q_base + claim_idx;
                let point_idx = if is_multi_point {
                    claim_to_point[claim_idx]
                } else {
                    0
                };
                let [public_low0, public_low1] = opening_point_block_summaries[point_idx];
                let public_scale = public_weights[point_idx] * gamma[claim_idx] * g_open;
                w_carry_terms[q][0] += public_scale * public_low0;
                w_carry_terms[q][1] += public_scale * public_low1;

                let [challenge_low0, challenge_low1] = challenge_block_summaries[claim_idx];
                let challenge_scale = consistency_weight * g_open;
                w_carry_terms[q][0] += challenge_scale * challenge_low0;
                w_carry_terms[q][1] += challenge_scale * challenge_low1;
            }
        }
        let w_sep = {
            let _span = tracing::info_span!("m_eval_w_sep").entered();
            eval_offset_eq_peeled_carry_terms(x_challenges, offset_w, block_bits, &w_carry_terms)
        };
        let w_d = {
            let _span = tracing::info_span!("m_eval_w_d").entered();
            eval_d_matrix_w_residual_direct(
                x_challenges,
                offset_w,
                num_blocks,
                num_claims,
                depth_open,
                d_weights,
                d_view,
                alpha_pows,
            )
        };

        let mut t_carry_terms = vec![[F::zero(), F::zero()]; num_claims * depth_open * n_a];
        for (a_idx, &a_weight) in a_weights.iter().enumerate() {
            for (digit_idx, &g_open) in g1_open.iter().enumerate() {
                let q_base = num_claims * (digit_idx + depth_open * a_idx);
                let scale = a_weight * g_open;
                for (claim_idx, &[challenge_low0, challenge_low1]) in
                    challenge_block_summaries.iter().enumerate()
                {
                    let q = q_base + claim_idx;
                    t_carry_terms[q][0] += scale * challenge_low0;
                    t_carry_terms[q][1] += scale * challenge_low1;
                }
            }
        }
        let t_sep = {
            let _span = tracing::info_span!("m_eval_t_sep").entered();
            eval_offset_eq_peeled_carry_terms(x_challenges, offset_t, block_bits, &t_carry_terms)
        };

        let t_b = {
            let _span = tracing::info_span!("m_eval_t_b").entered();
            eval_b_matrix_t_residual_direct(
                x_challenges,
                offset_t,
                num_blocks,
                num_claims,
                depth_open,
                n_a,
                n_b,
                eq_tau1,
                b_start,
                claim_to_group,
                b_view,
                alpha_pows,
            )
        };

        let z_base_len = num_points * inner_width;
        let (z_base_alg, z_base_setup): (Vec<F>, Vec<F>) = {
            let _span = tracing::info_span!("m_eval_z_base").entered();
            let pairs: Vec<(F, F)> = cfg_into_iter!(0..z_base_len)
                .map(|k| {
                    let point_idx = if is_multi_point { k / inner_width } else { 0 };
                    let local_k = if is_multi_point { k % inner_width } else { k };
                    let block_idx = local_k / depth_commit;
                    let digit_idx = local_k % depth_commit;
                    let opening_point = &opening_points[point_idx];
                    let alg =
                        consistency_weight * opening_point.a[block_idx] * g1_commit[digit_idx];
                    let mut setup = F::zero();
                    for (a_idx, eq_i) in a_weights.iter().enumerate() {
                        if !eq_i.is_zero() {
                            setup +=
                                *eq_i * eval_ring_at_pows(&a_view.row(a_idx)[local_k], alpha_pows);
                        }
                    }
                    (alg, setup)
                })
                .collect();
            pairs.into_iter().unzip()
        };

        let (z_dense_alg, z_dense_setup) = {
            let _span = tracing::info_span!("m_eval_z_dense").entered();
            let z_segment_alg: Vec<F> = cfg_into_iter!(0..z_len)
                .map(|x| {
                    let compound_dig = x / z_total_blocks;
                    let global_blk = x % z_total_blocks;
                    let dc = compound_dig / depth_fold;
                    let df = compound_dig % depth_fold;
                    let point_idx = global_blk / block_len;
                    let blk = global_blk % block_len;
                    let phys_k = point_idx * inner_width + blk * depth_commit + dc;
                    -(z_base_alg[phys_k] * fold_gadget[df])
                })
                .collect();
            let z_segment_setup: Vec<F> = cfg_into_iter!(0..z_len)
                .map(|x| {
                    let compound_dig = x / z_total_blocks;
                    let global_blk = x % z_total_blocks;
                    let dc = compound_dig / depth_fold;
                    let df = compound_dig % depth_fold;
                    let point_idx = global_blk / block_len;
                    let blk = global_blk % block_len;
                    let phys_k = point_idx * inner_width + blk * depth_commit + dc;
                    -(z_base_setup[phys_k] * fold_gadget[df])
                })
                .collect();
            (
                eval_offset_eq_tensor(
                    x_challenges,
                    offset_z,
                    F::one(),
                    &[z_segment_alg.as_slice()],
                ),
                eval_offset_eq_tensor(
                    x_challenges,
                    offset_z,
                    F::one(),
                    &[z_segment_setup.as_slice()],
                ),
            )
        };

        let alpha_pow_d = alpha_pows[D - 1] * alpha;
        let denom = alpha_pow_d + F::one();

        let r_tail_dims_pow2 = levels.is_power_of_two();
        let offset_r = w_len + t_len + z_len;

        let r_sep = if r_tail_dims_pow2 {
            eval_offset_eq_tensor(
                x_challenges,
                offset_r,
                -denom,
                &[&r_gadget, &eq_tau1[..rows]],
            )
        } else {
            F::zero()
        };
        let r_dense = if r_tail_dims_pow2 {
            F::zero()
        } else {
            let _span = tracing::info_span!("m_eval_r_dense").entered();
            let r_tail: Vec<F> = cfg_into_iter!(0..r_tail_len)
                .map(|idx| {
                    let row_idx = idx / levels;
                    let level_idx = idx % levels;
                    -(eq_tau1[row_idx] * denom * r_gadget[level_idx])
                })
                .collect();
            eval_offset_eq_tensor(x_challenges, offset_r, F::one(), &[r_tail.as_slice()])
        };

        Ok(PreparedMEvalSplit {
            algebraic: z_dense_alg + w_sep + t_sep + r_sep + r_dense,
            setup: z_dense_setup + w_d + t_b,
        })
    }

    fn uses_homogeneous_outer_layout(&self) -> bool {
        // Tier-marked groups (book §5.4) require the heterogeneous
        // grouped path even when their shape happens to match the outer
        // LP: the grouped path's tier branch reads `D_chunk` / `B_chunk`
        // shared columns under the block-diagonal collapse, while the
        // homogeneous fast path walks per-claim columns linearly and
        // would overflow the chunk's matrix view at `claim_count > 1`.
        self.group_layouts.iter().all(|layout| {
            layout.spec.tier.is_none()
                && layout.spec.num_blocks == self.num_blocks
                && layout.spec.block_len == self.block_len
                && layout.spec.num_digits_open == self.depth_open
                && layout.spec.num_digits_commit == self.depth_commit
                && layout.spec.num_digits_fold == self.depth_fold
                && layout.spec.b_key.row_len() == self.n_b
        })
    }

    fn segment_lengths_grouped(&self) -> (usize, usize, usize) {
        let w_len = self
            .group_layouts
            .last()
            .map(|layout| {
                layout.w_hat_start
                    + layout.claim_count * layout.spec.num_blocks * layout.spec.num_digits_open
            })
            .unwrap_or(0);
        let t_len = self
            .group_layouts
            .last()
            .map(|layout| {
                layout.t_hat_start
                    + layout.claim_count
                        * layout.spec.num_blocks
                        * self.n_a
                        * layout.spec.num_digits_open
            })
            .unwrap_or(0);
        let z_len = self
            .group_layouts
            .last()
            .map(|layout| {
                layout.z_hat_start
                    + self.num_eval_rows
                        * layout.spec.block_len
                        * layout.spec.num_digits_commit
                        * layout.spec.num_digits_fold
            })
            .unwrap_or(0);
        (w_len, t_len, z_len)
    }

    fn eval_split_at_point_grouped<const D: usize>(
        &self,
        x_challenges: &[F],
        setup: &AkitaExpandedSetup<F>,
        opening_points: &[RingOpeningPoint<F>],
        alpha: F,
    ) -> Result<PreparedMEvalSplit<F>, AkitaError> {
        let alpha_pows = self.alpha_pows_for_eval::<D>(alpha)?;
        let challenge_evals = self.challenge_evals.expanded_evals::<D>(alpha_pows)?;
        // Heterogeneous-group path: the prover rounds the stage-1
        // challenge count up to the next power of two so the tensor
        // split divides evenly. Surplus challenge evals beyond
        // `self.total_blocks` are zero-weighted at row evaluation time
        // (the per-group `block_start..block_end` ranges only index
        // into the natural totals).
        if challenge_evals.len() < self.total_blocks {
            return Err(AkitaError::InvalidSize {
                expected: self.total_blocks,
                actual: challenge_evals.len(),
            });
        }

        let stride = setup.seed.max_stride;
        let d_view = setup.shared_matrix.ring_view::<D>(self.n_d, stride);
        let a_view = setup.shared_matrix.ring_view::<D>(self.n_a, stride);

        let row_layout = &self.row_layout;
        let consistency_weight = self.eq_tau1[row_layout.original_fold];
        let d_start = row_layout.original_d.start;
        let tiered_d_relation = has_tiered_group(&self.group_layouts);
        let total_d_row_count = if tiered_d_relation {
            self.group_layouts
                .iter()
                .map(|layout| group_d_row_count(layout, self.n_d, true))
                .sum::<usize>()
        } else {
            self.n_d
        };
        let b_start = row_layout.original_b.start;
        // Tier-aware total B-row count: per book §5.4 line 752, tier-marked
        // groups contribute `tier.num_chunks * n_B_chunk` rows under shared
        // `B_chunk` (block-diagonal). Group iteration order matches the
        // prover's `commitment_cyclic_rows` assembly in `compute_r_split_eq`,
        // which concatenates per-claim B-output rows in the natural group
        // order (W, chunks, meta, …). The verifier's `eq_tau1` indexing
        // tracks a running B-offset per group to mirror that order.
        let d_weights = &self.eq_tau1[d_start..(d_start + total_d_row_count)];

        let (w_len, t_len, z_len) = self.segment_lengths_grouped();
        let offset_z = if self.z_first { 0 } else { w_len + t_len };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first { z_len + w_len } else { w_len };

        let mut algebraic = F::zero();
        let mut setup_part = F::zero();
        let mut d_running_offset = 0usize;
        let mut b_running_offset = 0usize;

        for layout in &self.group_layouts {
            let spec = &layout.spec;
            let is_tiered = spec.tier.is_some_and(|t| t.is_tiered());
            let group_role = if tiered_d_relation && self.group_layouts.len() >= 3 {
                if layout.group_idx == 0 {
                    0usize
                } else if layout.group_idx + 1 == self.group_layouts.len() {
                    2usize
                } else {
                    1usize
                }
            } else if tiered_d_relation && layout.group_idx + 1 == self.group_layouts.len() {
                2usize
            } else {
                1usize
            };
            let is_meta_group = group_role == 2;
            let fold_weight = if group_role == 0 {
                row_layout
                    .w_fold
                    .map(|row| self.eq_tau1[row])
                    .unwrap_or(consistency_weight)
            } else if is_meta_group {
                row_layout
                    .meta_fold
                    .map(|row| self.eq_tau1[row])
                    .unwrap_or(consistency_weight)
            } else {
                consistency_weight
            };
            let eval_row = if group_role == 0 {
                row_layout.w_eval.start
            } else if is_meta_group {
                row_layout.meta_eval.start
            } else {
                row_layout.original_eval.start
                    + (layout.group_idx - usize::from(self.group_layouts.len() >= 3))
            };
            let public_weight = self.eq_tau1.get(eval_row).copied().unwrap_or(F::zero());
            let g_open = gadget_row_scalars::<F>(spec.num_digits_open, self.log_basis);
            let g_commit = gadget_row_scalars::<F>(spec.num_digits_commit, self.log_basis);
            let fold_gadget = gadget_row_scalars::<F>(spec.num_digits_fold, self.log_basis);
            let group_blocks = layout.claim_count * spec.num_blocks;
            let b_view = setup
                .shared_matrix
                .ring_view::<D>(spec.b_key.row_len(), stride);
            let n_b_chunk = spec.b_key.row_len();
            let b_weights_count = group_b_row_count(layout);
            let b_base = if group_role == 0 {
                row_layout.w_b.start
            } else if is_meta_group {
                row_layout.meta_b.start
            } else {
                b_start + b_running_offset
            };
            let b_weights = &self.eq_tau1[b_base..(b_base + b_weights_count)];
            let d_weights_count = group_d_row_count(layout, self.n_d, tiered_d_relation);
            let group_d_weights = if tiered_d_relation {
                let base = if group_role == 0 {
                    row_layout.w_d.start
                } else if is_meta_group {
                    row_layout.meta_d.start
                } else {
                    row_layout.original_d.start + d_running_offset
                };
                &self.eq_tau1[base..base + d_weights_count]
            } else {
                d_weights
            };
            let group_a_weights = if group_role == 0 {
                &self.eq_tau1[row_layout.w_a.clone()]
            } else if is_meta_group {
                &self.eq_tau1[row_layout.meta_a.clone()]
            } else {
                &self.eq_tau1[row_layout.original_a.clone()]
            };

            for (dig, &open_gadget) in g_open.iter().enumerate() {
                for local_blk in 0..group_blocks {
                    let claim_within = local_blk / spec.num_blocks;
                    let block_idx = local_blk % spec.num_blocks;
                    let claim_idx = layout.claim_start + claim_within;
                    let point_idx = self.claim_to_point[claim_idx];
                    let x_local = dig * group_blocks + local_blk;
                    let eq_x =
                        eq_weight_at_index(x_challenges, offset_w + layout.w_hat_start + x_local);
                    if !eq_x.is_zero() {
                        let opening_point = &opening_points[point_idx];
                        algebraic += eq_x
                            * (public_weight * self.gamma[claim_idx] * opening_point.b[block_idx]
                                + fold_weight * challenge_evals[layout.block_start + local_blk])
                            * open_gadget;
                        // Tier-marked groups (book §5.4) share `D_chunk` across
                        // chunks via block-diagonal structure: every chunk reads
                        // the same first `num_blocks * num_digits_open` columns
                        // of the shared D matrix. For all groups, D's
                        // per-group inner-width cols are used (matching the
                        // commit's `[0, inner_width_g)` cols and the
                        // per-group A-cols pattern of the Z-part fix).
                        // Use `block_idx` (mod-num_blocks within the group)
                        // so tier-marked chunks share `D_chunk` and
                        // un-tiered single-claim groups still index
                        // `[0, inner_d_g)`.
                        let d_col = block_idx * spec.num_digits_open + dig;
                        let d_row_base = if tiered_d_relation && is_tiered {
                            claim_within * self.n_d
                        } else {
                            0
                        };
                        for (row, &row_weight) in group_d_weights[d_row_base..d_row_base + self.n_d]
                            .iter()
                            .enumerate()
                        {
                            if !row_weight.is_zero() {
                                setup_part += eq_x
                                    * row_weight
                                    * eval_ring_at_pows(&d_view.row(row)[d_col], alpha_pows);
                            }
                        }
                    }
                }
            }

            for (a_idx, &a_weight) in group_a_weights.iter().enumerate() {
                for (digit_idx, &open_gadget) in g_open.iter().enumerate() {
                    let compound = a_idx * spec.num_digits_open + digit_idx;
                    for local_blk in 0..group_blocks {
                        let claim_within = local_blk / spec.num_blocks;
                        let block_idx = local_blk % spec.num_blocks;
                        let x_local = compound * group_blocks + local_blk;
                        let eq_x = eq_weight_at_index(
                            x_challenges,
                            offset_t + layout.t_hat_start + x_local,
                        );
                        if eq_x.is_zero() {
                            continue;
                        }
                        algebraic += eq_x
                            * a_weight
                            * challenge_evals[layout.block_start + local_blk]
                            * open_gadget;
                        // Tier-marked groups share `B_chunk`: read at the
                        // within-chunk column. Untiered groups with
                        // `claim_count > 1` (currently unused — the
                        // chunks-as-1-group merge only triggers for
                        // tier-marked claims) would walk per-claim sub-blocks
                        // of B's columns.
                        let local_col = if is_tiered {
                            block_idx * self.n_a * spec.num_digits_open + compound
                        } else {
                            claim_within * spec.num_blocks * self.n_a * spec.num_digits_open
                                + block_idx * self.n_a * spec.num_digits_open
                                + compound
                        };
                        // For tier-marked groups, the chunk's row weights live
                        // in the slice `chunk_b_weights[claim_within * n_B_chunk
                        // .. (claim_within + 1) * n_B_chunk]` of the `k *
                        // n_B_chunk` chunk_b range. Other chunks contribute
                        // zero to this row because of the block-diagonal
                        // structure.
                        let b_row_base = if is_tiered {
                            claim_within * n_b_chunk
                        } else {
                            0
                        };
                        for (row_offset, &row_weight) in b_weights
                            [b_row_base..b_row_base + n_b_chunk]
                            .iter()
                            .enumerate()
                        {
                            if !row_weight.is_zero() {
                                setup_part += eq_x
                                    * row_weight
                                    * eval_ring_at_pows(
                                        &b_view.row(row_offset)[local_col],
                                        alpha_pows,
                                    );
                            }
                        }
                    }
                }
            }

            let z_total_blocks = self.num_eval_rows * spec.block_len;
            let inner_width = spec.block_len * spec.num_digits_commit;
            for (dc, &commit_gadget) in g_commit.iter().enumerate() {
                for (df, &fold_g) in fold_gadget.iter().enumerate() {
                    let compound = dc * spec.num_digits_fold + df;
                    for global_blk in 0..z_total_blocks {
                        let point_idx = global_blk / spec.block_len;
                        if !(layout.claim_start..layout.claim_start + layout.claim_count)
                            .any(|claim_idx| self.claim_to_point[claim_idx] == point_idx)
                        {
                            continue;
                        }
                        let blk = global_blk % spec.block_len;
                        // Per book §3.4 eq:batched-root-A, A is applied
                        // per opening point with its first `inner_width_p`
                        // cols. The per-group slot at `point_idx` reads
                        // A's cols `[blk * num_digits_commit + dc]`
                        // (within-group), not `layout.z_base_start +
                        // point_idx * inner_width + ...`. The earlier
                        // formula assumed A's cols were partitioned across
                        // groups, which breaks the M relation in ring for
                        // multi-group openings (book §3.4 line 715).
                        let phys_k = blk * spec.num_digits_commit + dc;
                        let _ = inner_width;
                        let x_local = compound * z_total_blocks + global_blk;
                        let eq_x = eq_weight_at_index(
                            x_challenges,
                            offset_z + layout.z_hat_start + x_local,
                        );
                        if eq_x.is_zero() {
                            continue;
                        }
                        let opening_point = &opening_points[point_idx];
                        if blk >= opening_point.a.len() {
                            return Err(AkitaError::InvalidSetup(format!(
                                "grouped z opening a-vector too short: blk={blk}, a_len={}, group={}, block_len={}, m_vars={}, r_vars={}",
                                opening_point.a.len(),
                                layout.group_idx,
                                spec.block_len,
                                spec.m_vars,
                                spec.r_vars
                            )));
                        }
                        algebraic -=
                            eq_x * fold_weight * opening_point.a[blk] * commit_gadget * fold_g;
                        for (row, &row_weight) in group_a_weights.iter().enumerate() {
                            if !row_weight.is_zero() {
                                setup_part -= eq_x
                                    * fold_g
                                    * row_weight
                                    * eval_ring_at_pows(&a_view.row(row)[phys_k], alpha_pows);
                            }
                        }
                    }
                }
            }

            if group_role == 1 {
                d_running_offset += d_weights_count;
                b_running_offset += b_weights_count;
            }
        }

        let levels = r_decomp_levels::<F>(self.log_basis);
        let r_gadget = gadget_row_scalars::<F>(levels, self.log_basis);
        let alpha_pow_d = alpha_pows[D - 1] * alpha;
        let denom = alpha_pow_d + F::one();
        let offset_r = w_len + t_len + z_len;
        for row_idx in 0..self.rows {
            for (level_idx, &r_g) in r_gadget.iter().enumerate() {
                let x_idx = row_idx * levels + level_idx;
                let eq_x = eq_weight_at_index(x_challenges, offset_r + x_idx);
                if !eq_x.is_zero() {
                    algebraic -= eq_x * self.eq_tau1[row_idx] * denom * r_g;
                }
            }
        }

        Ok(PreparedMEvalSplit {
            algebraic,
            setup: setup_part,
        })
    }

    /// Evaluate **only** the algebraic part of the prepared M-table at the
    /// supplied point.
    ///
    /// Skips the setup-matrix iterations (`w_d`, `t_b`, `z_dense_setup`) that
    /// dominate the cost of [`Self::eval_split_at_point`], so callers that
    /// reduce the setup contribution to a separate sumcheck (claim reduction)
    /// don't pay for the setup-iteration work twice.
    ///
    /// # Errors
    ///
    /// Returns any error surfaced by [`Self::eval_at_point`].
    #[allow(clippy::too_many_lines)]
    pub fn eval_algebraic_at_point<const D: usize>(
        &self,
        x_challenges: &[F],
        opening_points: &[RingOpeningPoint<F>],
        alpha: F,
    ) -> Result<F, AkitaError> {
        if !self.uses_homogeneous_outer_layout() {
            return self.eval_algebraic_at_point_grouped::<D>(x_challenges, opening_points, alpha);
        }
        let alpha_pows = self.alpha_pows_for_eval::<D>(alpha)?;
        let g1_open = gadget_row_scalars::<F>(self.depth_open, self.log_basis);
        let g1_commit = gadget_row_scalars::<F>(self.depth_commit, self.log_basis);
        let fold_gadget = gadget_row_scalars::<F>(self.depth_fold, self.log_basis);
        let levels = r_decomp_levels::<F>(self.log_basis);
        let r_gadget = gadget_row_scalars::<F>(levels, self.log_basis);

        let consistency_weight = self.eq_tau1[0];
        let public_weights = &self.eq_tau1[1..(1 + self.num_eval_rows)];

        let total_blocks = self.total_blocks;
        let num_blocks = self.num_blocks;
        let num_claims = self.num_claims;
        let depth_open = self.depth_open;
        let depth_commit = self.depth_commit;
        let depth_fold = self.depth_fold;
        let block_len = self.block_len;
        let inner_width = self.inner_width;
        let rows = self.rows;
        let num_points = self.num_points;
        let eq_tau1 = &self.eq_tau1;
        let claim_to_point = &self.claim_to_point;
        let gamma = &self.gamma;

        let w_len = depth_open * total_blocks;
        let t_len = depth_open * self.n_a * total_blocks;
        let z_total_blocks = num_points * block_len;
        let z_len = depth_fold * depth_commit * z_total_blocks;
        let r_tail_len = rows * levels;

        let is_multi_point = num_points > 1;

        let offset_z = if self.z_first { 0 } else { w_len + t_len };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first { z_len + w_len } else { w_len };
        let block_bits = num_blocks.trailing_zeros() as usize;
        let block_low_eq = EqPolynomial::evals(&x_challenges[..block_bits]);
        let block_offset_low = offset_w & (num_blocks - 1);
        debug_assert_eq!(block_offset_low, offset_t & (num_blocks - 1));

        let opening_point_block_summaries: Vec<[F; 2]> = opening_points
            .iter()
            .map(|opening_point| {
                summarize_pow2_block_carries(&block_low_eq, block_offset_low, &opening_point.b)
            })
            .collect();
        let challenge_block_summaries = self.challenge_evals.summarize_all_block_carries::<D>(
            num_claims,
            &x_challenges[..block_bits],
            &block_low_eq,
            block_offset_low,
            num_blocks,
            alpha_pows,
        )?;

        let mut w_carry_terms = vec![[F::zero(), F::zero()]; num_claims * depth_open];
        for (dig, &g_open) in g1_open.iter().enumerate() {
            let q_base = dig * num_claims;
            for claim_idx in 0..num_claims {
                let q = q_base + claim_idx;
                let point_idx = if is_multi_point {
                    claim_to_point[claim_idx]
                } else {
                    0
                };
                let [public_low0, public_low1] = opening_point_block_summaries[point_idx];
                let public_scale = public_weights[point_idx] * gamma[claim_idx] * g_open;
                w_carry_terms[q][0] += public_scale * public_low0;
                w_carry_terms[q][1] += public_scale * public_low1;

                let [challenge_low0, challenge_low1] = challenge_block_summaries[claim_idx];
                let challenge_scale = consistency_weight * g_open;
                w_carry_terms[q][0] += challenge_scale * challenge_low0;
                w_carry_terms[q][1] += challenge_scale * challenge_low1;
            }
        }
        let w_sep = {
            let _span = tracing::info_span!("m_eval_w_sep").entered();
            eval_offset_eq_peeled_carry_terms(x_challenges, offset_w, block_bits, &w_carry_terms)
        };

        let a_start = 1 + self.num_eval_rows + self.n_d + self.n_b * self.num_commitment_groups;
        let a_weights = &self.eq_tau1[a_start..(a_start + self.n_a)];

        let mut t_carry_terms = vec![[F::zero(), F::zero()]; num_claims * depth_open * self.n_a];
        for (a_idx, &a_weight) in a_weights.iter().enumerate() {
            for (digit_idx, &g_open) in g1_open.iter().enumerate() {
                let q_base = num_claims * (digit_idx + depth_open * a_idx);
                let scale = a_weight * g_open;
                for (claim_idx, &[challenge_low0, challenge_low1]) in
                    challenge_block_summaries.iter().enumerate()
                {
                    let q = q_base + claim_idx;
                    t_carry_terms[q][0] += scale * challenge_low0;
                    t_carry_terms[q][1] += scale * challenge_low1;
                }
            }
        }
        let t_sep = {
            let _span = tracing::info_span!("m_eval_t_sep").entered();
            eval_offset_eq_peeled_carry_terms(x_challenges, offset_t, block_bits, &t_carry_terms)
        };

        let z_base_len = num_points * inner_width;
        let z_base_alg: Vec<F> = {
            let _span = tracing::info_span!("m_eval_z_base_alg").entered();
            cfg_into_iter!(0..z_base_len)
                .map(|k| {
                    let point_idx = if is_multi_point { k / inner_width } else { 0 };
                    let local_k = if is_multi_point { k % inner_width } else { k };
                    let block_idx = local_k / depth_commit;
                    let digit_idx = local_k % depth_commit;
                    let opening_point = &opening_points[point_idx];
                    consistency_weight * opening_point.a[block_idx] * g1_commit[digit_idx]
                })
                .collect()
        };

        let z_dense_alg = {
            let _span = tracing::info_span!("m_eval_z_dense_alg").entered();
            let z_segment_alg: Vec<F> = cfg_into_iter!(0..z_len)
                .map(|x| {
                    let compound_dig = x / z_total_blocks;
                    let global_blk = x % z_total_blocks;
                    let dc = compound_dig / depth_fold;
                    let df = compound_dig % depth_fold;
                    let point_idx = global_blk / block_len;
                    let blk = global_blk % block_len;
                    let phys_k = point_idx * inner_width + blk * depth_commit + dc;
                    -(z_base_alg[phys_k] * fold_gadget[df])
                })
                .collect();
            eval_offset_eq_tensor(
                x_challenges,
                offset_z,
                F::one(),
                &[z_segment_alg.as_slice()],
            )
        };

        let alpha_pow_d = alpha_pows[D - 1] * alpha;
        let denom = alpha_pow_d + F::one();

        let r_tail_dims_pow2 = levels.is_power_of_two();
        let offset_r = w_len + t_len + z_len;

        let r_sep = if r_tail_dims_pow2 {
            eval_offset_eq_tensor(
                x_challenges,
                offset_r,
                -denom,
                &[&r_gadget, &eq_tau1[..rows]],
            )
        } else {
            F::zero()
        };
        let r_dense = if r_tail_dims_pow2 {
            F::zero()
        } else {
            let _span = tracing::info_span!("m_eval_r_dense_alg").entered();
            let r_tail: Vec<F> = cfg_into_iter!(0..r_tail_len)
                .map(|idx| {
                    let row_idx = idx / levels;
                    let level_idx = idx % levels;
                    -(eq_tau1[row_idx] * denom * r_gadget[level_idx])
                })
                .collect();
            eval_offset_eq_tensor(x_challenges, offset_r, F::one(), &[r_tail.as_slice()])
        };

        Ok(z_dense_alg + w_sep + t_sep + r_sep + r_dense)
    }

    fn eval_algebraic_at_point_grouped<const D: usize>(
        &self,
        x_challenges: &[F],
        opening_points: &[RingOpeningPoint<F>],
        alpha: F,
    ) -> Result<F, AkitaError> {
        let alpha_pows = self.alpha_pows_for_eval::<D>(alpha)?;
        let challenge_evals = self.challenge_evals.expanded_evals::<D>(alpha_pows)?;
        let row_layout = &self.row_layout;
        let consistency_weight = self.eq_tau1[row_layout.original_fold];
        let tiered_d_relation = has_tiered_group(&self.group_layouts);
        let (w_len, t_len, z_len) = self.segment_lengths_grouped();
        let offset_z = if self.z_first { 0 } else { w_len + t_len };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first { z_len + w_len } else { w_len };

        let mut algebraic = F::zero();
        for layout in &self.group_layouts {
            let spec = &layout.spec;
            let group_role = if tiered_d_relation && self.group_layouts.len() >= 3 {
                if layout.group_idx == 0 {
                    0usize
                } else if layout.group_idx + 1 == self.group_layouts.len() {
                    2usize
                } else {
                    1usize
                }
            } else if tiered_d_relation && layout.group_idx + 1 == self.group_layouts.len() {
                2usize
            } else {
                1usize
            };
            let is_meta_group = group_role == 2;
            let fold_weight = if group_role == 0 {
                row_layout
                    .w_fold
                    .map(|row| self.eq_tau1[row])
                    .unwrap_or(consistency_weight)
            } else if is_meta_group {
                row_layout
                    .meta_fold
                    .map(|row| self.eq_tau1[row])
                    .unwrap_or(consistency_weight)
            } else {
                consistency_weight
            };
            let eval_row = if group_role == 0 {
                row_layout.w_eval.start
            } else if is_meta_group {
                row_layout.meta_eval.start
            } else {
                row_layout.original_eval.start
                    + (layout.group_idx - usize::from(self.group_layouts.len() >= 3))
            };
            let public_weight = self.eq_tau1.get(eval_row).copied().unwrap_or(F::zero());
            let group_a_weights = if group_role == 0 {
                &self.eq_tau1[row_layout.w_a.clone()]
            } else if is_meta_group {
                &self.eq_tau1[row_layout.meta_a.clone()]
            } else {
                &self.eq_tau1[row_layout.original_a.clone()]
            };
            let g_open = gadget_row_scalars::<F>(spec.num_digits_open, self.log_basis);
            let g_commit = gadget_row_scalars::<F>(spec.num_digits_commit, self.log_basis);
            let fold_gadget = gadget_row_scalars::<F>(spec.num_digits_fold, self.log_basis);
            let group_blocks = layout.claim_count * spec.num_blocks;

            for (dig, &open_gadget) in g_open.iter().enumerate() {
                for local_blk in 0..group_blocks {
                    let claim_within = local_blk / spec.num_blocks;
                    let block_idx = local_blk % spec.num_blocks;
                    let claim_idx = layout.claim_start + claim_within;
                    let point_idx = self.claim_to_point[claim_idx];
                    let x_local = dig * group_blocks + local_blk;
                    let eq_x =
                        eq_weight_at_index(x_challenges, offset_w + layout.w_hat_start + x_local);
                    if !eq_x.is_zero() {
                        let opening_point = &opening_points[point_idx];
                        algebraic += eq_x
                            * (public_weight * self.gamma[claim_idx] * opening_point.b[block_idx]
                                + fold_weight * challenge_evals[layout.block_start + local_blk])
                            * open_gadget;
                    }
                }
            }

            for (a_idx, &a_weight) in group_a_weights.iter().enumerate() {
                for (digit_idx, &open_gadget) in g_open.iter().enumerate() {
                    let compound = a_idx * spec.num_digits_open + digit_idx;
                    for local_blk in 0..group_blocks {
                        let x_local = compound * group_blocks + local_blk;
                        let eq_x = eq_weight_at_index(
                            x_challenges,
                            offset_t + layout.t_hat_start + x_local,
                        );
                        if !eq_x.is_zero() {
                            algebraic += eq_x
                                * a_weight
                                * challenge_evals[layout.block_start + local_blk]
                                * open_gadget;
                        }
                    }
                }
            }

            let z_total_blocks = self.num_eval_rows * spec.block_len;
            for (dc, &commit_gadget) in g_commit.iter().enumerate() {
                for (df, &fold_g) in fold_gadget.iter().enumerate() {
                    let compound = dc * spec.num_digits_fold + df;
                    for global_blk in 0..z_total_blocks {
                        let point_idx = global_blk / spec.block_len;
                        if !(layout.claim_start..layout.claim_start + layout.claim_count)
                            .any(|claim_idx| self.claim_to_point[claim_idx] == point_idx)
                        {
                            continue;
                        }
                        let blk = global_blk % spec.block_len;
                        let x_local = compound * z_total_blocks + global_blk;
                        let eq_x = eq_weight_at_index(
                            x_challenges,
                            offset_z + layout.z_hat_start + x_local,
                        );
                        if !eq_x.is_zero() {
                            let opening_point = &opening_points[point_idx];
                            algebraic -=
                                eq_x * fold_weight * opening_point.a[blk] * commit_gadget * fold_g;
                        }
                    }
                }
            }
        }

        let levels = r_decomp_levels::<F>(self.log_basis);
        let r_gadget = gadget_row_scalars::<F>(levels, self.log_basis);
        let alpha_pow_d = alpha_pows[D - 1] * alpha;
        let denom = alpha_pow_d + F::one();
        let offset_r = w_len + t_len + z_len;
        for row_idx in 0..self.rows {
            for (level_idx, &r_g) in r_gadget.iter().enumerate() {
                let x_idx = row_idx * levels + level_idx;
                let eq_x = eq_weight_at_index(x_challenges, offset_r + x_idx);
                if !eq_x.is_zero() {
                    algebraic -= eq_x * self.eq_tau1[row_idx] * denom * r_g;
                }
            }
        }
        Ok(algebraic)
    }

    /// Materialize the algebraic/setup split over the padded M-eval x-domain.
    ///
    /// Test-only helper used by integration tests in `akita-pcs` to verify
    /// that the structured split recombines to the materialized M-eval
    /// table. Reuses [`Self::eval_split_at_point`] at every Boolean point.
    ///
    /// Gated behind `#[cfg(any(test, feature = "test-helpers"))]` so this
    /// O(2^x_bits) materializer does not appear in the production verifier
    /// surface; downstream test crates opt in via the `test-helpers`
    /// feature.
    ///
    /// # Errors
    ///
    /// Returns any error surfaced by [`Self::eval_split_at_point`].
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn split_eval_table<const D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        opening_points: &[RingOpeningPoint<F>],
        alpha: F,
    ) -> Result<Vec<PreparedMEvalSplit<F>>, AkitaError> {
        let x_bits = self.padded_x_bits();
        let x_len = 1usize
            .checked_shl(x_bits as u32)
            .ok_or_else(|| AkitaError::InvalidSetup("M-eval x table too large".to_string()))?;
        (0..x_len)
            .map(|idx| {
                let point = boolean_point::<F>(idx, x_bits);
                self.eval_split_at_point::<D>(&point, setup, opening_points, alpha)
            })
            .collect()
    }

    /// Materialize setup-polynomial weights for the setup part at `x`.
    ///
    /// The returned table is indexed as `row | col | coeff` in little-endian
    /// bit order and is padded to powers of two in row and column dimensions.
    /// Its inner product with `setup.shared_matrix.setup_polynomial_view()`
    /// equals `eval_split_at_point(...).setup`.
    ///
    /// This drives the verifier-side setup-claim-reduction sumcheck: pairing
    /// these weights with the shared setup polynomial reduces
    /// `m_setup(r_x)` to a single point claim on `S(r_i, r_col, r_k)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha` does not match this prepared M-eval.
    pub fn setup_weight_table_at_point<const D: usize>(
        &self,
        x_challenges: &[F],
        setup: &AkitaExpandedSetup<F>,
        alpha: F,
    ) -> Result<Vec<F>, AkitaError> {
        if !self.uses_homogeneous_outer_layout() {
            return self.setup_weight_table_at_point_grouped::<D>(x_challenges, setup, alpha);
        }
        let alpha_pows = self.alpha_pows_for_eval::<D>(alpha)?;
        let row_count = self.setup_polynomial_row_count();
        let col_count = self.setup_polynomial_col_count().max(1);
        let row_bits = row_count.next_power_of_two().trailing_zeros() as usize;
        let col_bits = col_count.next_power_of_two().trailing_zeros() as usize;
        let coeff_bits = D.trailing_zeros() as usize;
        let mut weights = vec![F::zero(); 1usize << (row_bits + col_bits + coeff_bits)];
        let add_weight = |weights: &mut [F], row: usize, col: usize, coeff: usize, weight: F| {
            if weight.is_zero() {
                return;
            }
            let idx = row | (col << row_bits) | (coeff << (row_bits + col_bits));
            weights[idx] += weight;
        };

        let fold_gadget = gadget_row_scalars::<F>(self.depth_fold, self.log_basis);

        let row_layout = &self.row_layout;
        let d_start = row_layout.original_d.start;
        let commitment_row_count = self.n_b * self.num_commitment_groups;
        let b_start = d_start + self.n_d;
        let a_start = b_start + commitment_row_count;
        let d_weights = &self.eq_tau1[d_start..(d_start + self.n_d)];
        let a_weights = &self.eq_tau1[a_start..(a_start + self.n_a)];

        let w_len = self.depth_open * self.total_blocks;
        let t_len = self.depth_open * self.n_a * self.total_blocks;
        let z_total_blocks = self.num_points * self.block_len;
        let z_len = self.depth_fold * self.depth_commit * z_total_blocks;

        let offset_z = if self.z_first { 0 } else { w_len + t_len };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first { z_len + w_len } else { w_len };

        let num_blocks = self.num_blocks;
        let per_claim_d_width = num_blocks * self.depth_open;
        for dig in 0..self.depth_open {
            for blk in 0..self.total_blocks {
                let claim_idx = blk / num_blocks;
                let block_idx = blk % num_blocks;
                let x_idx = dig * self.total_blocks + blk;
                let eq = eq_weight_at_index(x_challenges, offset_w + x_idx);
                if eq.is_zero() {
                    continue;
                }
                let d_phys_col = claim_idx * per_claim_d_width + block_idx * self.depth_open + dig;
                for (row, &row_weight) in d_weights.iter().enumerate() {
                    for (coeff, &alpha_pow) in alpha_pows.iter().enumerate() {
                        add_weight(
                            &mut weights,
                            row,
                            d_phys_col,
                            coeff,
                            eq * row_weight * alpha_pow,
                        );
                    }
                }
            }
        }

        let t_compound_per_block = self.n_a * self.depth_open;
        let t_cols_per_claim = t_compound_per_block * num_blocks;
        for compound_dig in 0..(self.n_a * self.depth_open) {
            let a_idx = compound_dig / self.depth_open;
            let digit_idx = compound_dig % self.depth_open;
            for blk in 0..self.total_blocks {
                let claim_idx = blk / num_blocks;
                let block_idx = blk % num_blocks;
                let (group_idx, claim_idx_within_group) = self.claim_to_group[claim_idx];
                let x_idx = compound_dig * self.total_blocks + blk;
                let eq = eq_weight_at_index(x_challenges, offset_t + x_idx);
                if eq.is_zero() {
                    continue;
                }
                let local_col = claim_idx_within_group * t_cols_per_claim
                    + block_idx * t_compound_per_block
                    + a_idx * self.depth_open
                    + digit_idx;
                let commitment_weights = &self.eq_tau1
                    [(b_start + group_idx * self.n_b)..(b_start + (group_idx + 1) * self.n_b)];
                for (row, &row_weight) in commitment_weights.iter().enumerate() {
                    for (coeff, &alpha_pow) in alpha_pows.iter().enumerate() {
                        add_weight(
                            &mut weights,
                            row,
                            local_col,
                            coeff,
                            eq * row_weight * alpha_pow,
                        );
                    }
                }
            }
        }

        let inner_width = self.inner_width;
        for compound_dig in 0..(self.depth_fold * self.depth_commit) {
            let dc = compound_dig / self.depth_fold;
            let df = compound_dig % self.depth_fold;
            for global_blk in 0..z_total_blocks {
                let point_idx = global_blk / self.block_len;
                let blk = global_blk % self.block_len;
                let phys_k = point_idx * inner_width + blk * self.depth_commit + dc;
                let x_idx = compound_dig * z_total_blocks + global_blk;
                let eq = eq_weight_at_index(x_challenges, offset_z + x_idx);
                if eq.is_zero() {
                    continue;
                }
                for (row, &row_weight) in a_weights.iter().enumerate() {
                    let scale = -(eq * fold_gadget[df] * row_weight);
                    for (coeff, &alpha_pow) in alpha_pows.iter().enumerate() {
                        add_weight(&mut weights, row, phys_k, coeff, scale * alpha_pow);
                    }
                }
            }
        }

        Ok(weights)
    }

    fn setup_weight_table_at_point_grouped<const D: usize>(
        &self,
        x_challenges: &[F],
        _setup: &AkitaExpandedSetup<F>,
        alpha: F,
    ) -> Result<Vec<F>, AkitaError> {
        let alpha_pows = self.alpha_pows_for_eval::<D>(alpha)?;
        let row_count = self.setup_polynomial_row_count();
        let col_count = self.setup_polynomial_col_count().max(1);
        let row_bits = row_count.next_power_of_two().trailing_zeros() as usize;
        let col_bits = col_count.next_power_of_two().trailing_zeros() as usize;
        let coeff_bits = D.trailing_zeros() as usize;
        let mut weights = vec![F::zero(); 1usize << (row_bits + col_bits + coeff_bits)];
        let add_weight = |weights: &mut [F], row: usize, col: usize, coeff: usize, weight: F| {
            if weight.is_zero() {
                return;
            }
            let idx = row | (col << row_bits) | (coeff << (row_bits + col_bits));
            weights[idx] += weight;
        };

        let row_layout = &self.row_layout;
        let d_start = row_layout.original_d.start;
        let tiered_d_relation = has_tiered_group(&self.group_layouts);
        let total_d_row_count = if tiered_d_relation {
            self.group_layouts
                .iter()
                .map(|layout| group_d_row_count(layout, self.n_d, true))
                .sum::<usize>()
        } else {
            self.n_d
        };
        let b_start = row_layout.original_b.start;
        let d_weights = &self.eq_tau1[d_start..(d_start + total_d_row_count)];
        let (w_len, t_len, z_len) = self.segment_lengths_grouped();
        let offset_z = if self.z_first { 0 } else { w_len + t_len };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first { z_len + w_len } else { w_len };

        let mut d_running_offset = 0usize;
        let mut b_running_offset = 0usize;
        for layout in &self.group_layouts {
            let spec = &layout.spec;
            let is_tiered = spec.tier.is_some_and(|t| t.is_tiered());
            let group_role = if tiered_d_relation && self.group_layouts.len() >= 3 {
                if layout.group_idx == 0 {
                    0usize
                } else if layout.group_idx + 1 == self.group_layouts.len() {
                    2usize
                } else {
                    1usize
                }
            } else if tiered_d_relation && layout.group_idx + 1 == self.group_layouts.len() {
                2usize
            } else {
                1usize
            };
            let is_meta_group = group_role == 2;
            let group_blocks = layout.claim_count * spec.num_blocks;
            let n_b_chunk = spec.b_key.row_len();
            let b_weights_count = group_b_row_count(layout);
            let b_base = if group_role == 0 {
                row_layout.w_b.start
            } else if is_meta_group {
                row_layout.meta_b.start
            } else {
                b_start + b_running_offset
            };
            let b_weights = &self.eq_tau1[b_base..b_base + b_weights_count];
            let d_weights_count = group_d_row_count(layout, self.n_d, tiered_d_relation);
            let group_d_weights = if tiered_d_relation {
                let base = if group_role == 0 {
                    row_layout.w_d.start
                } else if is_meta_group {
                    row_layout.meta_d.start
                } else {
                    row_layout.original_d.start + d_running_offset
                };
                &self.eq_tau1[base..base + d_weights_count]
            } else {
                d_weights
            };
            let group_a_weights = if group_role == 0 {
                &self.eq_tau1[row_layout.w_a.clone()]
            } else if is_meta_group {
                &self.eq_tau1[row_layout.meta_a.clone()]
            } else {
                &self.eq_tau1[row_layout.original_a.clone()]
            };

            for dig in 0..spec.num_digits_open {
                for local_blk in 0..group_blocks {
                    let claim_within = local_blk / spec.num_blocks;
                    let block_idx = local_blk % spec.num_blocks;
                    let x_local = dig * group_blocks + local_blk;
                    let eq =
                        eq_weight_at_index(x_challenges, offset_w + layout.w_hat_start + x_local);
                    if eq.is_zero() {
                        continue;
                    }
                    // See `eval_split_at_point_grouped` for the rationale.
                    let d_col = block_idx * spec.num_digits_open + dig;
                    let d_row_base = if tiered_d_relation && is_tiered {
                        claim_within * self.n_d
                    } else {
                        0
                    };
                    for (row, &row_weight) in group_d_weights[d_row_base..d_row_base + self.n_d]
                        .iter()
                        .enumerate()
                    {
                        for (coeff, &alpha_pow) in alpha_pows.iter().enumerate() {
                            add_weight(
                                &mut weights,
                                row,
                                d_col,
                                coeff,
                                eq * row_weight * alpha_pow,
                            );
                        }
                    }
                }
            }

            for a_idx in 0..self.n_a {
                for digit_idx in 0..spec.num_digits_open {
                    let compound = a_idx * spec.num_digits_open + digit_idx;
                    for local_blk in 0..group_blocks {
                        let claim_within = local_blk / spec.num_blocks;
                        let block_idx = local_blk % spec.num_blocks;
                        let x_local = compound * group_blocks + local_blk;
                        let eq = eq_weight_at_index(
                            x_challenges,
                            offset_t + layout.t_hat_start + x_local,
                        );
                        if eq.is_zero() {
                            continue;
                        }
                        let local_col = if is_tiered {
                            block_idx * self.n_a * spec.num_digits_open + compound
                        } else {
                            claim_within * spec.num_blocks * self.n_a * spec.num_digits_open
                                + block_idx * self.n_a * spec.num_digits_open
                                + compound
                        };
                        let b_row_base = if is_tiered {
                            claim_within * n_b_chunk
                        } else {
                            0
                        };
                        for (row_offset, &row_weight) in b_weights
                            [b_row_base..b_row_base + n_b_chunk]
                            .iter()
                            .enumerate()
                        {
                            for (coeff, &alpha_pow) in alpha_pows.iter().enumerate() {
                                add_weight(
                                    &mut weights,
                                    row_offset,
                                    local_col,
                                    coeff,
                                    eq * row_weight * alpha_pow,
                                );
                            }
                        }
                    }
                }
            }

            let fold_gadget = gadget_row_scalars::<F>(spec.num_digits_fold, self.log_basis);
            let z_total_blocks = self.num_eval_rows * spec.block_len;
            let inner_width = spec.block_len * spec.num_digits_commit;
            for dc in 0..spec.num_digits_commit {
                for (df, &fold_g) in fold_gadget.iter().enumerate() {
                    let compound = dc * spec.num_digits_fold + df;
                    for global_blk in 0..z_total_blocks {
                        let point_idx = global_blk / spec.block_len;
                        if !(layout.claim_start..layout.claim_start + layout.claim_count)
                            .any(|claim_idx| self.claim_to_point[claim_idx] == point_idx)
                        {
                            continue;
                        }
                        let blk = global_blk % spec.block_len;
                        // See block-A above for the rationale.
                        let phys_k = blk * spec.num_digits_commit + dc;
                        let _ = inner_width;
                        let x_local = compound * z_total_blocks + global_blk;
                        let eq = eq_weight_at_index(
                            x_challenges,
                            offset_z + layout.z_hat_start + x_local,
                        );
                        if eq.is_zero() {
                            continue;
                        }
                        for (row, &row_weight) in group_a_weights.iter().enumerate() {
                            let scale = -(eq * fold_g * row_weight);
                            for (coeff, &alpha_pow) in alpha_pows.iter().enumerate() {
                                add_weight(&mut weights, row, phys_k, coeff, scale * alpha_pow);
                            }
                        }
                    }
                }
            }

            if group_role == 1 {
                d_running_offset += d_weights_count;
                b_running_offset += b_weights_count;
            }
        }

        Ok(weights)
    }

    #[cfg(any(test, feature = "test-helpers"))]
    fn padded_x_bits(&self) -> usize {
        let levels = r_decomp_levels::<F>(self.log_basis);
        let w_len = self.depth_open * self.total_blocks;
        let t_len = self.depth_open * self.n_a * self.total_blocks;
        let z_total_blocks = self.num_points * self.block_len;
        let z_len = self.depth_fold * self.depth_commit * z_total_blocks;
        let r_tail_len = self.rows * levels;
        let total_cols = w_len + t_len + z_len + r_tail_len;
        total_cols.next_power_of_two().trailing_zeros() as usize
    }

    /// Padded row/column dimensions used by the setup polynomial view in the
    /// claim-reduction sumcheck. The setup-claim sumcheck binds variables in
    /// `(row | col | coeff)` bit order over those dimensions.
    #[inline]
    pub fn setup_polynomial_padded_dims(&self, _setup_max_stride: usize) -> (usize, usize, usize) {
        let (row_count, col_count_padded) = self.padded_dims_pair();
        let row_bits = row_count.next_power_of_two().trailing_zeros() as usize;
        let col_bits = col_count_padded.trailing_zeros() as usize;
        let coeff_bits = (self.alpha_pows.len()).trailing_zeros() as usize;
        (row_bits, col_bits, coeff_bits)
    }

    /// Setup polynomial view row count used by the claim-reduction path.
    ///
    /// The setup polynomial view is shaped `max(n_A, max_g n_B_g, n_D)`
    /// rows by `max(D-cols, B-cols, A-cols)` columns. For tier-marked
    /// groups the per-chunk B and D rows are shared (book §5.4 line 752
    /// "MLE evaluation cost is O(|D_chunk|) + O(log k), independent of
    /// k") so the row envelope is independent of the number of chunks.
    /// The meta-tier is committed via the standard Akita machinery
    /// (book line 695) and contributes its `n_B_meta` rows through
    /// `max_group_b` like any other group, requiring no separate
    /// envelope expansion.
    #[inline]
    pub fn setup_polynomial_row_count(&self) -> usize {
        self.padded_dims_pair().0
    }

    #[inline]
    fn padded_dims_pair(&self) -> (usize, usize) {
        akita_types::setup_polynomial_padded_dims_inner(
            akita_types::SetupPolynomialDimsOuter {
                n_a: self.n_a,
                n_b: self.n_b,
                n_d: self.n_d,
                num_blocks: self.num_blocks,
                block_len: self.block_len,
                num_digits_open: self.depth_open,
                num_digits_commit: self.depth_commit,
                num_digits_fold: self.depth_fold,
            },
            &self.group_layouts,
            self.num_eval_rows,
            self.num_points,
        )
    }

    /// Phase D-full Slice G tier shape that this `PreparedMEval` was
    /// constructed for. `un_tiered()` (`f = 1`) leaves the relation at
    /// the Slice F 5-row-family shape; `f ≥ 2` activates the book §5.4
    /// 10-check-group relation. Consumers (commit kernel, sumcheck
    /// drivers, opening-point derivation) dispatch on
    /// [`tier_setup_params.is_tiered`](akita_types::TieredSetupParams::is_tiered).
    #[inline]
    pub fn tier_setup_params(&self) -> akita_types::TieredSetupParams {
        self.tier_setup_params
    }

    /// Setup polynomial view column count used by claim reduction.
    ///
    /// Returned col count is **not** padded to a power of two; callers
    /// that need the padded value should use
    /// [`Self::setup_polynomial_padded_dims`].
    #[inline]
    pub fn setup_polynomial_col_count(&self) -> usize {
        // `padded_dims_pair` returns the next-power-of-two col count, but
        // every existing caller of `setup_polynomial_col_count` immediately
        // pads via `.next_power_of_two()`, so we return the already-padded
        // value to keep callers stable. The historical un-padded variant
        // had no remaining external consumers.
        self.padded_dims_pair().1
    }

    /// Evaluate the multilinear setup weight polynomial at the sumcheck-bound
    /// point `r_setup` *without* materializing the full weight table.
    ///
    /// Produces the same value as
    /// `multilinear_eval(&prepared.setup_weight_table_at_point(...), r_setup)`
    /// but avoids the per-level `O(2^(row_bits + col_bits + coeff_bits))`
    /// weight materialization plus the subsequent table evaluation. We
    /// factor the per-row and per-coefficient contributions out of the inner
    /// `(dig, blk)` loops, so the cost is dominated by
    /// `O(depth_open · total_blocks + n_a · depth_open · total_blocks +
    ///   depth_fold · depth_commit · z_total_blocks)` field multiplies and
    /// `O(2^row_bits + 2^col_bits + 2^coeff_bits)` for the small eq tables.
    ///
    /// `r_setup` must have length `row_bits + col_bits + coeff_bits` and be
    /// laid out as `r_row || r_col || r_coeff` in least-significant-bit
    /// first order.
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha` does not match this prepared M-eval, if
    /// `r_setup` has the wrong length, or if internal padded dimensions
    /// disagree with the bound point.
    pub fn eval_setup_weight_at_point<const D: usize>(
        &self,
        x_challenges: &[F],
        setup: &AkitaExpandedSetup<F>,
        alpha: F,
        r_setup: &[F],
    ) -> Result<F, AkitaError> {
        akita_field::op_counter::with_category(akita_field::op_counter::OpCategory::Setup, || {
            self.eval_setup_weight_at_point_inner::<D>(x_challenges, setup, alpha, r_setup)
        })
    }

    fn eval_setup_weight_at_point_inner<const D: usize>(
        &self,
        x_challenges: &[F],
        setup: &AkitaExpandedSetup<F>,
        alpha: F,
        r_setup: &[F],
    ) -> Result<F, AkitaError> {
        let alpha_pows = self.alpha_pows_for_eval::<D>(alpha)?;
        let max_stride = setup.seed.max_stride.max(1);
        let (row_bits, col_bits, coeff_bits) = self.setup_polynomial_padded_dims(max_stride);
        let expected_len = row_bits + col_bits + coeff_bits;
        if r_setup.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: r_setup.len(),
            });
        }
        if !self.uses_homogeneous_outer_layout() {
            return self.eval_setup_weight_at_point_grouped::<D>(
                x_challenges,
                setup,
                alpha,
                r_setup,
                row_bits,
                col_bits,
                coeff_bits,
            );
        }
        let r_row = &r_setup[..row_bits];
        let r_col = &r_setup[row_bits..row_bits + col_bits];
        let r_coeff = &r_setup[row_bits + col_bits..];

        let eq_row = EqPolynomial::evals(r_row);
        let eq_col = EqPolynomial::evals(r_col);
        let eq_coeff = EqPolynomial::evals(r_coeff);
        debug_assert_eq!(eq_coeff.len(), D);

        let coeff_factor: F = alpha_pows
            .iter()
            .zip(eq_coeff.iter())
            .map(|(a, e)| *a * *e)
            .sum();

        let d_start = 1 + self.num_eval_rows;
        let commitment_row_count = self.n_b * self.num_commitment_groups;
        let b_start = d_start + self.n_d;
        let a_start = b_start + commitment_row_count;
        let d_weights = &self.eq_tau1[d_start..(d_start + self.n_d)];
        let a_weights = &self.eq_tau1[a_start..(a_start + self.n_a)];

        let d_row_factor: F = d_weights
            .iter()
            .zip(eq_row.iter())
            .take(self.n_d)
            .map(|(w, e)| *w * *e)
            .sum();
        let a_row_factor: F = a_weights
            .iter()
            .zip(eq_row.iter())
            .take(self.n_a)
            .map(|(w, e)| *w * *e)
            .sum();
        let cw_row_factor: Vec<F> = (0..self.num_commitment_groups)
            .map(|group_idx| {
                let group_weights = &self.eq_tau1
                    [(b_start + group_idx * self.n_b)..(b_start + (group_idx + 1) * self.n_b)];
                group_weights
                    .iter()
                    .zip(eq_row.iter())
                    .take(self.n_b)
                    .map(|(w, e)| *w * *e)
                    .sum::<F>()
            })
            .collect();

        let fold_gadget = gadget_row_scalars::<F>(self.depth_fold, self.log_basis);

        let w_len = self.depth_open * self.total_blocks;
        let t_len = self.depth_open * self.n_a * self.total_blocks;
        let z_total_blocks = self.num_points * self.block_len;
        let z_len = self.depth_fold * self.depth_commit * z_total_blocks;

        let offset_z = if self.z_first { 0 } else { w_len + t_len };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first { z_len + w_len } else { w_len };

        let num_blocks = self.num_blocks;
        let per_claim_d_width = num_blocks * self.depth_open;

        let mut w_inner = F::zero();
        for dig in 0..self.depth_open {
            for blk in 0..self.total_blocks {
                let claim_idx = blk / num_blocks;
                let block_idx = blk % num_blocks;
                let x_idx = dig * self.total_blocks + blk;
                let eq_x = eq_weight_at_index(x_challenges, offset_w + x_idx);
                if eq_x.is_zero() {
                    continue;
                }
                let d_phys_col = claim_idx * per_claim_d_width + block_idx * self.depth_open + dig;
                w_inner += eq_x * eq_col[d_phys_col];
            }
        }
        let w_contribution = d_row_factor * coeff_factor * w_inner;

        let t_compound_per_block = self.n_a * self.depth_open;
        let t_cols_per_claim = t_compound_per_block * num_blocks;
        let mut t_contribution = F::zero();
        for compound_dig in 0..(self.n_a * self.depth_open) {
            let a_idx = compound_dig / self.depth_open;
            let digit_idx = compound_dig % self.depth_open;
            for blk in 0..self.total_blocks {
                let claim_idx = blk / num_blocks;
                let block_idx = blk % num_blocks;
                let (group_idx, claim_idx_within_group) = self.claim_to_group[claim_idx];
                let x_idx = compound_dig * self.total_blocks + blk;
                let eq_x = eq_weight_at_index(x_challenges, offset_t + x_idx);
                if eq_x.is_zero() {
                    continue;
                }
                let local_col = claim_idx_within_group * t_cols_per_claim
                    + block_idx * t_compound_per_block
                    + a_idx * self.depth_open
                    + digit_idx;
                t_contribution += eq_x * eq_col[local_col] * cw_row_factor[group_idx];
            }
        }
        let t_contribution = t_contribution * coeff_factor;

        let inner_width = self.inner_width;
        let mut z_inner = F::zero();
        for compound_dig in 0..(self.depth_fold * self.depth_commit) {
            let dc = compound_dig / self.depth_fold;
            let df = compound_dig % self.depth_fold;
            for global_blk in 0..z_total_blocks {
                let point_idx = global_blk / self.block_len;
                let blk = global_blk % self.block_len;
                let phys_k = point_idx * inner_width + blk * self.depth_commit + dc;
                let x_idx = compound_dig * z_total_blocks + global_blk;
                let eq_x = eq_weight_at_index(x_challenges, offset_z + x_idx);
                if eq_x.is_zero() {
                    continue;
                }
                z_inner += eq_x * eq_col[phys_k] * fold_gadget[df];
            }
        }
        let z_contribution = -(a_row_factor * coeff_factor * z_inner);

        Ok(w_contribution + t_contribution + z_contribution)
    }

    /// Heterogeneous-LP analog of [`Self::eval_setup_weight_at_point`].
    ///
    /// Computes the same value as
    /// `multilinear_eval(&setup_weight_table_at_point_grouped(...), r_setup)`
    /// without materialising the
    /// `2^(row_bits + col_bits + coeff_bits)` weight hypercube. The
    /// algebra mirrors `setup_weight_table_at_point_grouped`: for each
    /// group, each (dig, blk) loop deposits a rank-1 outer product
    /// `eq · row_weights · alpha_pows` into the weight table; the
    /// structured form factors that as
    /// `eq · eq_col[col] · row_factor · coeff_factor`, where the row
    /// and coeff factors are precomputed once per group (and once per
    /// claim within a tier-marked group, since the B/D row weights are
    /// per-chunk slices of the group's `eq_tau1` window).
    ///
    /// Cost per level:
    /// `O(Σ_g (depth_open_g · group_blocks_g · log col + n_a · depth_open_g ·
    ///   group_blocks_g + depth_commit_g · depth_fold_g · z_total_blocks))`
    /// field multiplies, plus `O(2^row_bits + 2^col_bits + 2^coeff_bits)`
    /// for the small eq tables — book §5.3 line 528–538's
    /// `O(log m_row + log d)` setup-side asymptotic on tiered paths.
    #[allow(clippy::too_many_arguments)]
    fn eval_setup_weight_at_point_grouped<const D: usize>(
        &self,
        x_challenges: &[F],
        _setup: &AkitaExpandedSetup<F>,
        alpha: F,
        r_setup: &[F],
        row_bits: usize,
        col_bits: usize,
        coeff_bits: usize,
    ) -> Result<F, AkitaError> {
        debug_assert_eq!(r_setup.len(), row_bits + col_bits + coeff_bits);
        let alpha_pows = self.alpha_pows_for_eval::<D>(alpha)?;
        let r_row = &r_setup[..row_bits];
        let r_col = &r_setup[row_bits..row_bits + col_bits];
        let r_coeff = &r_setup[row_bits + col_bits..];

        let eq_row = EqPolynomial::evals(r_row);
        let eq_col = EqPolynomial::evals(r_col);
        let eq_coeff = EqPolynomial::evals(r_coeff);
        debug_assert_eq!(eq_coeff.len(), D);

        let coeff_factor: F = alpha_pows
            .iter()
            .zip(eq_coeff.iter())
            .map(|(a, e)| *a * *e)
            .sum();

        let row_layout = &self.row_layout;
        let tiered_d_relation = has_tiered_group(&self.group_layouts);
        let b_start = row_layout.original_b.start;

        let (w_len, t_len, z_len) = self.segment_lengths_grouped();
        let offset_z = if self.z_first { 0 } else { w_len + t_len };
        let offset_w = if self.z_first { z_len } else { 0 };
        let offset_t = if self.z_first { z_len + w_len } else { w_len };

        let mut total = F::zero();
        let mut d_running_offset = 0usize;
        let mut b_running_offset = 0usize;

        for layout in &self.group_layouts {
            let spec = &layout.spec;
            let is_tiered = spec.tier.is_some_and(|t| t.is_tiered());
            let group_role = if tiered_d_relation && self.group_layouts.len() >= 3 {
                if layout.group_idx == 0 {
                    0usize
                } else if layout.group_idx + 1 == self.group_layouts.len() {
                    2usize
                } else {
                    1usize
                }
            } else if tiered_d_relation && layout.group_idx + 1 == self.group_layouts.len() {
                2usize
            } else {
                1usize
            };
            let is_meta_group = group_role == 2;
            let group_blocks = layout.claim_count * spec.num_blocks;
            let n_b_chunk = spec.b_key.row_len();
            let b_weights_count = group_b_row_count(layout);
            let b_base = if group_role == 0 {
                row_layout.w_b.start
            } else if is_meta_group {
                row_layout.meta_b.start
            } else {
                b_start + b_running_offset
            };
            let b_weights = &self.eq_tau1[b_base..b_base + b_weights_count];
            let d_weights_count = group_d_row_count(layout, self.n_d, tiered_d_relation);
            let group_d_weights = if tiered_d_relation {
                let base = if group_role == 0 {
                    row_layout.w_d.start
                } else if is_meta_group {
                    row_layout.meta_d.start
                } else {
                    row_layout.original_d.start + d_running_offset
                };
                &self.eq_tau1[base..base + d_weights_count]
            } else {
                let d_start = row_layout.original_d.start;
                &self.eq_tau1[d_start..d_start + self.n_d]
            };
            let group_a_weights = if group_role == 0 {
                &self.eq_tau1[row_layout.w_a.clone()]
            } else if is_meta_group {
                &self.eq_tau1[row_layout.meta_a.clone()]
            } else {
                &self.eq_tau1[row_layout.original_a.clone()]
            };

            // Per-claim D row-factors: tier-marked groups slice
            // group_d_weights as `claim_within * n_d .. (claim_within + 1) *
            // n_d`. Un-tiered groups always read the first n_d window.
            let d_chunk_count = if tiered_d_relation && is_tiered {
                layout.claim_count
            } else {
                1
            };
            let d_row_factors: Vec<F> = (0..d_chunk_count)
                .map(|claim_within| {
                    let base = claim_within * self.n_d;
                    group_d_weights[base..base + self.n_d]
                        .iter()
                        .zip(eq_row.iter())
                        .take(self.n_d)
                        .map(|(w, e)| *w * *e)
                        .sum::<F>()
                })
                .collect();

            // Per-claim B row-factors: tier-marked groups slice
            // b_weights as `claim_within * n_b_chunk .. (claim_within + 1) *
            // n_b_chunk`.
            let b_chunk_count = if is_tiered { layout.claim_count } else { 1 };
            let b_row_factors: Vec<F> = (0..b_chunk_count)
                .map(|claim_within| {
                    let base = claim_within * n_b_chunk;
                    b_weights[base..base + n_b_chunk]
                        .iter()
                        .zip(eq_row.iter())
                        .take(n_b_chunk)
                        .map(|(w, e)| *w * *e)
                        .sum::<F>()
                })
                .collect();

            // A row-factor: A is one shared block per group (book §5.4
            // line 728-729 — one combined Ajtai binding).
            let a_row_factor: F = group_a_weights
                .iter()
                .zip(eq_row.iter())
                .take(group_a_weights.len())
                .map(|(w, e)| *w * *e)
                .sum();

            // W block: D-matrix rows × ŵ digit columns.
            //
            // Phase 5 book §5.5 line 752 chunk-axis amortisation. For
            // tier-marked groups the per-(dig, chunk, blk) sum factors
            // by hoisting `dig` to an outer loop and computing the
            // inner (chunk, blk) sum via `eval_offset_eq_tensor` with
            // factors `[eq_col_per_dig, d_row_factors]`. Per-chunk
            // `d_factor` becomes a 1-D tensor factor instead of a
            // claim_within-indexed scalar; the chunk-axis bits of
            // x_local are folded by the offset-eq carry DP in
            // `O(num_blocks_chunk + k)` per dig (vs the prior
            // `O(num_blocks_chunk · k · log x_bits)` per dig).
            let mut w_inner = F::zero();
            let chunk_axis_amortised = is_tiered && d_chunk_count > 1;
            if chunk_axis_amortised {
                let c_base = offset_w + layout.w_hat_start;
                let mut f_blk: Vec<F> = vec![F::zero(); spec.num_blocks];
                for dig in 0..spec.num_digits_open {
                    // Per-dig column factor: `eq_col[blk * num_digits_open + dig]`
                    // for blk ∈ [0, num_blocks_chunk).
                    for (blk_idx, slot) in f_blk.iter_mut().enumerate() {
                        let d_col = blk_idx * spec.num_digits_open + dig;
                        *slot = if d_col < eq_col.len() {
                            eq_col[d_col]
                        } else {
                            F::zero()
                        };
                    }
                    let c_dig = c_base + dig * group_blocks;
                    w_inner += eval_offset_eq_tensor(
                        x_challenges,
                        c_dig,
                        F::one(),
                        &[&f_blk, &d_row_factors[..]],
                    );
                }
            } else {
                for dig in 0..spec.num_digits_open {
                    for local_blk in 0..group_blocks {
                        let claim_within = local_blk / spec.num_blocks;
                        let block_idx = local_blk % spec.num_blocks;
                        let x_local = dig * group_blocks + local_blk;
                        let eq = eq_weight_at_index(
                            x_challenges,
                            offset_w + layout.w_hat_start + x_local,
                        );
                        if eq.is_zero() {
                            continue;
                        }
                        let d_col = block_idx * spec.num_digits_open + dig;
                        if d_col >= eq_col.len() {
                            continue;
                        }
                        let d_factor = if d_chunk_count == 1 {
                            d_row_factors[0]
                        } else {
                            d_row_factors[claim_within]
                        };
                        w_inner += eq * eq_col[d_col] * d_factor;
                    }
                }
            }
            total += w_inner * coeff_factor;

            // T block: B-matrix rows × t̂ digit columns.
            //
            // Phase 5 book §5.5 line 752 chunk-axis amortisation
            // (same shape as the W block above): hoist `compound` to
            // outer loop and fold the chunk axis via
            // `eval_offset_eq_tensor` with factors
            // `[eq_col_per_compound, b_row_factors]`. For tier-marked
            // groups `local_col = block_idx * n_a * num_digits_open
            // + compound` is chunk-independent (book §5.4 line 752
            // shared B_chunk col indexing).
            let mut t_inner = F::zero();
            let t_chunk_amortised = is_tiered && b_chunk_count > 1;
            if t_chunk_amortised {
                let c_base = offset_t + layout.t_hat_start;
                let compound_count = self.n_a * spec.num_digits_open;
                let mut f_blk: Vec<F> = vec![F::zero(); spec.num_blocks];
                for compound in 0..compound_count {
                    for (blk_idx, slot) in f_blk.iter_mut().enumerate() {
                        let local_col = blk_idx * self.n_a * spec.num_digits_open + compound;
                        *slot = if local_col < eq_col.len() {
                            eq_col[local_col]
                        } else {
                            F::zero()
                        };
                    }
                    let c_compound = c_base + compound * group_blocks;
                    t_inner += eval_offset_eq_tensor(
                        x_challenges,
                        c_compound,
                        F::one(),
                        &[&f_blk, &b_row_factors[..]],
                    );
                }
            } else {
                for a_idx in 0..self.n_a {
                    for digit_idx in 0..spec.num_digits_open {
                        let compound = a_idx * spec.num_digits_open + digit_idx;
                        for local_blk in 0..group_blocks {
                            let claim_within = local_blk / spec.num_blocks;
                            let block_idx = local_blk % spec.num_blocks;
                            let x_local = compound * group_blocks + local_blk;
                            let eq = eq_weight_at_index(
                                x_challenges,
                                offset_t + layout.t_hat_start + x_local,
                            );
                            if eq.is_zero() {
                                continue;
                            }
                            let local_col = if is_tiered {
                                block_idx * self.n_a * spec.num_digits_open + compound
                            } else {
                                claim_within * spec.num_blocks * self.n_a * spec.num_digits_open
                                    + block_idx * self.n_a * spec.num_digits_open
                                    + compound
                            };
                            if local_col >= eq_col.len() {
                                continue;
                            }
                            let b_factor = if b_chunk_count == 1 {
                                b_row_factors[0]
                            } else {
                                b_row_factors[claim_within]
                            };
                            t_inner += eq * eq_col[local_col] * b_factor;
                        }
                    }
                }
            }
            total += t_inner * coeff_factor;

            // Z block: A-matrix rows × ẑ digit columns, sign-flipped.
            let fold_gadget = gadget_row_scalars::<F>(spec.num_digits_fold, self.log_basis);
            let z_total_blocks = self.num_eval_rows * spec.block_len;
            let mut z_inner = F::zero();
            for dc in 0..spec.num_digits_commit {
                for (df, &fold_g) in fold_gadget.iter().enumerate() {
                    let compound = dc * spec.num_digits_fold + df;
                    for global_blk in 0..z_total_blocks {
                        let point_idx = global_blk / spec.block_len;
                        if !(layout.claim_start..layout.claim_start + layout.claim_count)
                            .any(|claim_idx| self.claim_to_point[claim_idx] == point_idx)
                        {
                            continue;
                        }
                        let blk = global_blk % spec.block_len;
                        let phys_k = blk * spec.num_digits_commit + dc;
                        if phys_k >= eq_col.len() {
                            continue;
                        }
                        let x_local = compound * z_total_blocks + global_blk;
                        let eq = eq_weight_at_index(
                            x_challenges,
                            offset_z + layout.z_hat_start + x_local,
                        );
                        if eq.is_zero() {
                            continue;
                        }
                        z_inner += eq * fold_g * eq_col[phys_k];
                    }
                }
            }
            total -= z_inner * coeff_factor * a_row_factor;

            if group_role == 1 {
                d_running_offset += d_weights_count;
                b_running_offset += b_weights_count;
            }
        }

        Ok(total)
    }
}

#[cfg(any(test, feature = "test-helpers"))]
fn boolean_point<F: FieldCore>(index: usize, bits: usize) -> Vec<F> {
    (0..bits)
        .map(|bit| {
            if (index >> bit) & 1 == 1 {
                F::one()
            } else {
                F::zero()
            }
        })
        .collect()
}

fn eq_weight_at_index<F: FieldCore>(challenges: &[F], index: usize) -> F {
    if challenges.len() < usize::BITS as usize && index >= (1usize << challenges.len()) {
        return F::zero();
    }
    challenges
        .iter()
        .enumerate()
        .fold(F::one(), |acc, (bit, &challenge)| {
            if (index >> bit) & 1 == 1 {
                acc * challenge
            } else {
                acc * (F::one() - challenge)
            }
        })
}

#[inline]
fn summarize_strided_pow2_block_carries<F: FieldCore, const D: usize>(
    eq_low: &[F],
    offset_low: usize,
    row: &[CyclotomicRing<F, D>],
    alpha_pows: &[F],
    block_count: usize,
    block_stride: usize,
    lane_offset: usize,
) -> [F; 2] {
    debug_assert!(block_count.is_power_of_two());
    debug_assert_eq!(eq_low.len(), block_count);
    debug_assert!(offset_low < block_count);

    let inner_bits = block_count.trailing_zeros() as usize;
    let inner_mask = block_count - 1;
    let mut out = [F::zero(), F::zero()];
    for block_idx in 0..block_count {
        let sum = offset_low + block_idx;
        let carry = sum >> inner_bits;
        let low_idx = sum & inner_mask;
        let col = block_idx * block_stride + lane_offset;
        let value = eval_ring_at_pows(&row[col], alpha_pows);
        out[carry] += value * eq_low[low_idx];
    }
    out
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn eval_d_matrix_w_residual_direct<F: FieldCore, const D: usize>(
    x_challenges: &[F],
    offset_w: usize,
    num_blocks: usize,
    num_claims: usize,
    depth_open: usize,
    d_weights: &[F],
    d_view: RingMatrixView<'_, F, D>,
    alpha_pows: &[F],
) -> F {
    debug_assert!(num_blocks.is_power_of_two());
    let block_bits = num_blocks.trailing_zeros() as usize;
    let block_low_eq = EqPolynomial::evals(&x_challenges[..block_bits]);
    let block_offset_low = offset_w & (num_blocks - 1);
    let per_claim_d_width = num_blocks * depth_open;
    let carry_terms: Vec<[F; 2]> = cfg_into_iter!(0..(num_claims * depth_open))
        .map(|q| {
            let claim_idx = q % num_claims;
            let dig = q / num_claims;
            let lane_offset = claim_idx * per_claim_d_width + dig;
            let mut out = [F::zero(), F::zero()];
            for (di, &d_weight) in d_weights.iter().enumerate() {
                if d_weight.is_zero() {
                    continue;
                }
                let row = d_view.row(di);
                let [block_low0, block_low1] = summarize_strided_pow2_block_carries(
                    &block_low_eq,
                    block_offset_low,
                    row,
                    alpha_pows,
                    num_blocks,
                    depth_open,
                    lane_offset,
                );
                out[0] += d_weight * block_low0;
                out[1] += d_weight * block_low1;
            }
            out
        })
        .collect();
    eval_offset_eq_peeled_carry_terms(x_challenges, offset_w, block_bits, &carry_terms)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn eval_b_matrix_t_residual_direct<F: FieldCore, const D: usize>(
    x_challenges: &[F],
    offset_t: usize,
    num_blocks: usize,
    num_claims: usize,
    depth_open: usize,
    n_a: usize,
    n_b: usize,
    eq_tau1: &[F],
    b_start: usize,
    claim_to_group: &[(usize, usize)],
    b_view: RingMatrixView<'_, F, D>,
    alpha_pows: &[F],
) -> F {
    debug_assert!(num_blocks.is_power_of_two());
    let block_bits = num_blocks.trailing_zeros() as usize;
    let block_low_eq = EqPolynomial::evals(&x_challenges[..block_bits]);
    let block_offset_low = offset_t & (num_blocks - 1);
    let t_compound_per_block = n_a * depth_open;
    let t_cols_per_claim = t_compound_per_block * num_blocks;
    let carry_terms: Vec<[F; 2]> = cfg_into_iter!(0..(num_claims * n_a * depth_open))
        .map(|q| {
            let claim_idx = q % num_claims;
            let compound_dig = q / num_claims;
            let a_idx = compound_dig / depth_open;
            let digit_idx = compound_dig % depth_open;
            let (group_idx, claim_idx_within_group) = claim_to_group[claim_idx];
            let commitment_weights =
                &eq_tau1[(b_start + group_idx * n_b)..(b_start + (group_idx + 1) * n_b)];
            let lane_offset =
                claim_idx_within_group * t_cols_per_claim + a_idx * depth_open + digit_idx;
            let mut out = [F::zero(), F::zero()];
            for (row_idx, &eq_i) in commitment_weights.iter().enumerate() {
                if eq_i.is_zero() {
                    continue;
                }
                let row = b_view.row(row_idx);
                let [block_low0, block_low1] = summarize_strided_pow2_block_carries(
                    &block_low_eq,
                    block_offset_low,
                    row,
                    alpha_pows,
                    num_blocks,
                    t_compound_per_block,
                    lane_offset,
                );
                out[0] += eq_i * block_low0;
                out[1] += eq_i * block_low1;
            }
            out
        })
        .collect();
    eval_offset_eq_peeled_carry_terms(x_challenges, offset_t, block_bits, &carry_terms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::{
        SparseChallenge, SparseChallengeConfig, Stage1ChallengeShape, TensorStage1Challenges,
    };
    use akita_field::Fp64;

    type F = Fp64<4294967197>;
    const D: usize = 8;

    fn sparse(positions: &[u32], coeffs: &[i8]) -> SparseChallenge {
        SparseChallenge {
            positions: positions.to_vec(),
            coeffs: coeffs.to_vec(),
        }
    }

    fn test_level_params() -> LevelParams {
        let mut lp = LevelParams::params_only(
            D,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        );
        lp.stage1_challenge_shape = Stage1ChallengeShape::Tensor;
        lp.num_blocks = 4;
        lp.block_len = 1;
        lp.m_vars = 2;
        lp.r_vars = 0;
        lp.num_digits_commit = 1;
        lp.num_digits_open = 1;
        lp.num_digits_fold = 1;
        lp
    }

    fn tensor_stage1_challenges() -> Stage1Challenges {
        Stage1Challenges::Tensor(TensorStage1Challenges {
            left: vec![sparse(&[0, 6], &[1, -1]), sparse(&[1, 3], &[2, 1])],
            right: vec![sparse(&[0], &[1]), sparse(&[2], &[-1])],
            left_len: 2,
            right_len: 2,
            num_claims: 1,
        })
    }

    #[test]
    fn prepare_m_eval_keeps_tensor_challenges_compact() {
        let challenges = tensor_stage1_challenges();
        let lp = test_level_params();
        let rows = lp.m_row_count(1, 1);
        let tau1 = vec![F::from_u64(3); rows.next_power_of_two().trailing_zeros() as usize];
        let alpha = F::from_u64(9);
        let alpha_pows = scalar_powers(alpha, D);

        let prepared =
            prepare_m_eval::<F, D>(&challenges, alpha, &lp, &tau1, &[1], &[F::one()], 1, 1, &[])
                .expect("prepare_m_eval");

        assert!(prepared.challenge_evals_are_tensor());
        assert_eq!(
            prepared.tensor_alpha_pow_d_plus_one(),
            Some(alpha_pows[D - 1] * alpha + F::one())
        );
        assert_eq!(
            prepared
                .debug_expanded_challenge_evals::<D>()
                .expect("expanded tensor evals"),
            challenges
                .evals_at_pows::<F, D>(&alpha_pows)
                .expect("reference evals")
        );
    }

    #[test]
    fn tensor_block_carry_summaries_match_expanded_reference() {
        let challenges = tensor_stage1_challenges();
        let lp = test_level_params();
        let rows = lp.m_row_count(1, 1);
        let tau1 = vec![F::from_u64(5); rows.next_power_of_two().trailing_zeros() as usize];
        let alpha = F::from_u64(11);
        let alpha_pows = scalar_powers(alpha, D);
        let expanded = challenges
            .evals_at_pows::<F, D>(&alpha_pows)
            .expect("expanded tensor evals");
        let prepared =
            prepare_m_eval::<F, D>(&challenges, alpha, &lp, &tau1, &[1], &[F::one()], 1, 1, &[])
                .expect("prepare_m_eval");
        let x_cases = [
            vec![F::from_u64(2), F::from_u64(3)],
            vec![F::from_u64(7), -F::from_u64(4)],
            vec![F::zero(), F::one()],
        ];

        for x_low in x_cases {
            let eq_low = EqPolynomial::evals(&x_low);
            for offset_low in 0..lp.num_blocks {
                let got = prepared
                    .challenge_evals
                    .summarize_all_block_carries::<D>(
                        1,
                        &x_low,
                        &eq_low,
                        offset_low,
                        lp.num_blocks,
                        &alpha_pows,
                    )
                    .expect("tensor summary")
                    .remove(0);
                let expected =
                    summarize_pow2_block_carries(&eq_low, offset_low, &expanded[..lp.num_blocks]);
                assert_eq!(
                    got, expected,
                    "summary mismatch for x_low={x_low:?}, offset_low={offset_low}"
                );
            }
        }
    }

    #[test]
    fn prepared_m_eval_rejects_mixed_alpha() {
        let challenges = tensor_stage1_challenges();
        let lp = test_level_params();
        let rows = lp.m_row_count(1, 1);
        let tau1 = vec![F::from_u64(3); rows.next_power_of_two().trailing_zeros() as usize];
        let alpha = F::from_u64(9);
        let prepared =
            prepare_m_eval::<F, D>(&challenges, alpha, &lp, &tau1, &[1], &[F::one()], 1, 1, &[])
                .expect("prepare_m_eval");

        let err = prepared
            .alpha_pows_for_eval::<D>(alpha + F::one())
            .unwrap_err();
        assert!(format!("{err:?}").contains("different ring-switch alpha"));
    }

    fn small_expanded_setup(max_stride: usize, seed_offset: u64) -> AkitaExpandedSetup<F> {
        let total = max_stride * D;
        let data: Vec<F> = (0..total)
            .map(|i| F::from_u64((seed_offset + i as u64).wrapping_mul(31415) + 7))
            .collect();
        AkitaExpandedSetup {
            seed: akita_types::AkitaSetupSeed {
                max_num_vars: 8,
                max_num_batched_polys: 1,
                max_num_points: 1,
                max_stride,
                public_matrix_seed: [0u8; 32],
            },
            shared_matrix: akita_types::FlatMatrix::from_flat_data(data, D),
        }
    }

    #[test]
    fn structured_setup_weight_matches_materialized() {
        let challenges = tensor_stage1_challenges();
        let lp = test_level_params();
        let rows = lp.m_row_count(1, 1);
        let tau1: Vec<F> = (0..rows.next_power_of_two().trailing_zeros() as usize)
            .map(|i| F::from_u64(13 + i as u64))
            .collect();
        let alpha = F::from_u64(11);

        let prepared =
            prepare_m_eval::<F, D>(&challenges, alpha, &lp, &tau1, &[1], &[F::one()], 1, 1, &[])
                .expect("prepare_m_eval");
        let setup = small_expanded_setup(4, 0);
        // x_challenges has length `padded_x_bits`.
        let x_bits = prepared.padded_x_bits();
        let x_challenges: Vec<F> = (0..x_bits).map(|i| F::from_u64(3 + i as u64)).collect();

        let weights = prepared
            .setup_weight_table_at_point::<D>(&x_challenges, &setup, alpha)
            .expect("materialized weights");

        let r_setup_len = weights.len().trailing_zeros() as usize;
        let r_setup_cases = [
            (0..r_setup_len)
                .map(|i| F::from_u64(21 + i as u64))
                .collect::<Vec<_>>(),
            (0..r_setup_len)
                .map(|i| F::from_u64(57 + 3 * i as u64))
                .collect::<Vec<_>>(),
            (0..r_setup_len)
                .map(|i| {
                    if i.is_multiple_of(2) {
                        F::zero()
                    } else {
                        F::one()
                    }
                })
                .collect::<Vec<_>>(),
        ];
        for r_setup in &r_setup_cases {
            let expected =
                akita_sumcheck::multilinear_eval(&weights, r_setup).expect("materialized eval");
            let got = prepared
                .eval_setup_weight_at_point::<D>(&x_challenges, &setup, alpha, r_setup)
                .expect("structured eval");
            assert_eq!(
                got, expected,
                "structured w_setup eval disagrees at r_setup={r_setup:?}"
            );
        }
    }
}
