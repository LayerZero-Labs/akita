//! Verifier-side ring-switch replay.

#[cfg(feature = "zk")]
use super::slice_mle::{compute_b_blinding_part, compute_d_blinding_part};
use super::slice_mle::{
    compute_r_contribution, EStructuredSlicesEvaluator, SetupEvaluation, SetupEvaluator,
    SetupEvaluatorMode, StructuredSliceMleEvaluator, TStructuredSlicesEvaluator,
    ZDenseSlicesEvaluator, ZStructuredPow2SlicesEvaluator,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_challenges::Challenges;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, RandomSampling,
};
use akita_transcript::labels::{
    ABSORB_NEXT_LEVEL_WITNESS_BINDING, ABSORB_TERMINAL_W_REMAINDER, CHALLENGE_RING_SWITCH,
    CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_ext_challenge, Transcript};
#[cfg(feature = "zk")]
use akita_types::zk;
use akita_types::{
    embed_ring_subfield_scalar, gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup,
    FlatRingVec, LevelParams, MRowLayout, RingMultiplierOpeningPoint, RingRelationInstance,
    RingRelationSegmentLayout, RingSubfieldEncoding, SetupContributionPlanInputs,
    TerminalWitnessTranscriptParts,
};

use super::{validate_level_dispatch, validate_log_basis, validate_ring_dispatch};
pub(crate) use tensor_challenges::PreparedChallengeEvals;

mod tensor_challenges;
#[cfg(test)]
mod tests;

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: FieldCore> {
    /// Prepared data for deferred ring-switch row MLE evaluation.
    pub prepared_row_eval: RingSwitchDeferredRowEval<E>,
    /// Evaluation table of alpha powers over the ring-coordinate dimension.
    pub alpha_evals_y: Vec<E>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Stage-1 challenge handoff for intermediate folds.
    pub stage1_tau0: Option<Vec<E>>,
    /// Challenge tau1 for the stage-2 M-row combination.
    pub tau1: Vec<E>,
    /// Basis size `b = 2^log_basis`.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: E,
}

/// Precomputed challenge-derived data for deferred ring-switch row MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views and gadget vectors.
#[derive(Clone)]
pub struct RingSwitchDeferredRowEval<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    pub(crate) eq_tau1: Vec<F>,
    pub(crate) num_t_vectors: usize,
    pub(crate) num_blocks: usize,
    pub(crate) num_claims: usize,
    pub(crate) depth_open: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_fold: usize,
    #[cfg(feature = "zk")]
    pub(crate) d_blinding_segment_len: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_digit_planes_per_point: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_segment_len: usize,
    pub(crate) block_len: usize,
    pub(crate) inner_width: usize,
    pub(crate) log_basis: u32,
    pub(crate) n_a: usize,
    pub(crate) n_d: usize,
    pub(crate) m_row_layout: MRowLayout,
    pub(crate) n_b: usize,
    /// Tiered split factor `f` (`1` = single-tier).
    pub(crate) tier_split: usize,
    /// Second-tier `F` rank (`0` = single-tier); the sent-commitment length.
    pub(crate) n_f: usize,
    pub(crate) num_points: usize,
    pub(crate) rows: usize,
    pub(crate) claim_to_t_vector: Vec<usize>,
    pub(crate) num_polys_per_commitment_group: Vec<usize>,
    pub(crate) num_public_rows: usize,
    pub(crate) gamma: Vec<F>,
    pub(crate) claim_to_point: Vec<usize>,
    pub(crate) witness_segment_layout: RingRelationSegmentLayout,
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
fn jolt_cycle_marker(marker_id_str: &str, event_type: u32) {
    const JOLT_CYCLE_TRACK_CALL_ID: u32 = 0xC7C1E;
    let marker_id = marker_id_str.as_ptr() as usize as u32;
    let marker_len = marker_id_str.len() as u32;
    unsafe {
        core::arch::asm!(
            ".insn i 0x5B, 2, x0, x0, 0",
            in("x10") JOLT_CYCLE_TRACK_CALL_ID,
            in("x11") marker_id,
            in("x12") marker_len,
            in("x13") event_type,
            options(nostack, preserves_flags)
        );
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
fn jolt_start_cycle_tracking(marker_id: &str) {
    jolt_cycle_marker(marker_id, 1);
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[inline(always)]
fn jolt_start_cycle_tracking(_marker_id: &str) {}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[inline(always)]
fn jolt_end_cycle_tracking(marker_id: &str) {
    jolt_cycle_marker(marker_id, 2);
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
#[inline(always)]
fn jolt_end_cycle_tracking(_marker_id: &str) {}

struct RowEvalPointContext<'a, F: FieldCore, E: FieldCore> {
    layout: RingRelationSegmentLayout,
    alpha_pows: Vec<E>,
    g1_open: Vec<F>,
    fold_gadget: Vec<F>,
    eq_low: Vec<E>,
    z_block_low_eq: Vec<E>,
    offset_low_bits: usize,
    z_offset_low_bits: usize,
    block_offset_low: usize,
    x_low_challenges: &'a [E],
    high_challenges: &'a [E],
    z_high_challenges: &'a [E],
}

impl<E: FieldCore> RingSwitchDeferredRowEval<E> {
    /// `num_blocks * num_claims` (W/D challenge logical length).
    ///
    /// Prepare validates the product with checked arithmetic before building
    /// this struct; replay uses the unchecked product on those same fields.
    #[inline(always)]
    pub(crate) fn total_blocks(&self) -> usize {
        self.num_blocks * self.num_claims
    }

    /// Number of active D rows in the selected M-row layout.
    pub(crate) fn n_d_active(&self) -> usize {
        match self.m_row_layout {
            MRowLayout::WithDBlock => self.n_d,
            MRowLayout::WithoutDBlock => 0,
        }
    }

    pub(crate) fn create_setup_contribution_inputs(&self) -> SetupContributionPlanInputs<E> {
        SetupContributionPlanInputs {
            eq_tau1: self.eq_tau1.clone(),
            num_t_vectors: self.num_t_vectors,
            num_blocks: self.num_blocks,
            num_claims: self.num_claims,
            depth_open: self.depth_open,
            depth_commit: self.depth_commit,
            depth_fold: self.depth_fold,
            block_len: self.block_len,
            inner_width: self.inner_width,
            n_a: self.n_a,
            n_d: self.n_d,
            m_row_layout: self.m_row_layout,
            n_b: self.n_b,
            num_points: self.num_points,
            rows: self.rows,
            num_polys_per_commitment_group: self.num_polys_per_commitment_group.clone(),
            num_public_rows: self.num_public_rows,
            tier_split: self.tier_split,
            n_f: self.n_f,
        }
    }

    fn prepare_point_context<'a, F, const D: usize>(
        &self,
        x_challenges: &'a [E],
        alpha: E,
    ) -> Result<RowEvalPointContext<'a, F, E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    {
        let layout = self.witness_segment_layout;
        validate_log_basis(self.log_basis)?;
        let alpha_pows = scalar_powers(alpha, D);
        let g1_open = gadget_row_scalars::<F>(self.depth_open, self.log_basis);
        let fold_gadget = gadget_row_scalars::<F>(self.depth_fold, self.log_basis);

        let offset_low_bits = self.num_blocks.trailing_zeros() as usize;
        if offset_low_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: offset_low_bits,
                actual: x_challenges.len(),
            });
        }
        let eq_low = EqPolynomial::evals(&x_challenges[..offset_low_bits])?;
        let block_offset_low = layout.offset_e & (self.num_blocks - 1);
        debug_assert_eq!(block_offset_low, layout.offset_t & (self.num_blocks - 1));

        let z_offset_low_bits = self.block_len.trailing_zeros() as usize;
        if z_offset_low_bits > x_challenges.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_offset_low_bits,
                actual: x_challenges.len(),
            });
        }
        let z_block_low_eq = EqPolynomial::evals(&x_challenges[..z_offset_low_bits])?;

        Ok(RowEvalPointContext {
            layout,
            alpha_pows,
            g1_open,
            fold_gadget,
            eq_low,
            z_block_low_eq,
            offset_low_bits,
            z_offset_low_bits,
            block_offset_low,
            x_low_challenges: &x_challenges[..offset_low_bits],
            high_challenges: &x_challenges[offset_low_bits..],
            z_high_challenges: &x_challenges[z_offset_low_bits..],
        })
    }

    /// Evaluate the prepared ring-switch row table at the supplied point.
    ///
    /// # Errors
    ///
    /// Returns an error if the setup matrix cannot be viewed at `D` or an
    /// internal offset-eq evaluation receives inconsistent dimensions.
    #[inline]
    pub fn eval_at_point<F, const D: usize>(
        &self,
        x_challenges: &[E],
        setup: &AkitaExpandedSetup<F>,
        ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    {
        let _ring_bits = validate_ring_dispatch::<D>()?;
        let context = self.prepare_point_context::<F, D>(x_challenges, alpha)?;
        if ring_multiplier_points.len() != self.num_points {
            return Err(AkitaError::InvalidSize {
                expected: self.num_points,
                actual: ring_multiplier_points.len(),
            });
        }
        for point in ring_multiplier_points {
            if point.b_len() != self.num_blocks || point.a_len() < self.block_len {
                return Err(AkitaError::InvalidProof);
            }
        }

        let total_blocks = self.total_blocks();
        if let Some(c_alphas) = self.c_alphas.as_flat() {
            if c_alphas.len() != total_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: total_blocks,
                    actual: c_alphas.len(),
                });
            }
        }
        let challenge_block_summaries: Vec<[E; 2]> =
            self.c_alphas.summarize_all_block_carries::<F, D>(
                self.num_claims,
                context.x_low_challenges,
                &context.eq_low,
                context.block_offset_low,
                self.num_blocks,
            )?;
        let mut challenge_block_summaries_by_t_vector =
            vec![[E::zero(), E::zero()]; self.num_t_vectors];
        if self.claim_to_t_vector.len() != self.num_claims
            || self
                .claim_to_t_vector
                .iter()
                .any(|&idx| idx >= self.num_t_vectors)
        {
            return Err(AkitaError::InvalidProof);
        }
        for (claim_idx, &t_vector_idx) in self.claim_to_t_vector.iter().enumerate() {
            let [carry0, carry1] = challenge_block_summaries[claim_idx];
            challenge_block_summaries_by_t_vector[t_vector_idx][0] += carry0;
            challenge_block_summaries_by_t_vector[t_vector_idx][1] += carry1;
        }

        // ----- E-hat ---------------------------------------------------------
        let e_structured_contribution = {
            let _span = tracing::info_span!("e_structured").entered();
            let uses_ring_multipliers = ring_multiplier_points
                .iter()
                .any(|point| point.as_base().is_none());
            let row_coefficient_rings = if uses_ring_multipliers {
                Some(
                    self.gamma
                        .iter()
                        .copied()
                        .map(|coefficient| {
                            embed_ring_subfield_scalar::<F, E, D>(
                                coefficient,
                                AkitaError::InvalidProof,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                None
            };
            let public_block_summaries: Vec<[E; 2]> = (0..self.num_claims)
                .map(|claim_idx| {
                    let point_idx = self.claim_to_point[claim_idx];
                    if point_idx >= ring_multiplier_points.len() {
                        return Err(AkitaError::InvalidProof);
                    }
                    let point = &ring_multiplier_points[point_idx];
                    let coefficient_ring = row_coefficient_rings
                        .as_ref()
                        .map(|rings| &rings[claim_idx]);
                    summarize_pow2_multiplier_block_carries(
                        &context.eq_low,
                        context.block_offset_low,
                        point.b_len(),
                        |idx| {
                            point.eval_b_with_coefficient(
                                idx,
                                self.gamma[claim_idx],
                                coefficient_ring,
                                &context.alpha_pows,
                            )
                        },
                    )
                })
                .collect::<Result<_, _>>()?;
            let public_row_weights_by_claim: Vec<E> = self
                .claim_to_point
                .iter()
                .map(|&point_idx| {
                    point_idx
                        .checked_add(1)
                        .and_then(|idx| self.eq_tau1.get(idx))
                        .copied()
                        .ok_or(AkitaError::InvalidProof)
                })
                .collect::<Result<_, _>>()?;
            EStructuredSlicesEvaluator {
                high_challenges: context.high_challenges,
                offset_high: context.layout.offset_e >> context.offset_low_bits,
                gadget_vector: &context.g1_open,
                public_block_summaries: &public_block_summaries,
                challenge_block_summaries: &challenge_block_summaries,
                public_row_weights_by_claim: &public_row_weights_by_claim,
                challenge_weight: self.eq_tau1[0],
            }
            .evaluate()
        };

        // Canonical A-block start (tiered-aware): consistency | public | D |
        // COMMIT (F when tiered, else B) | B_inner (tiered) | A.
        let commit_rows_pg = if self.tier_split > 1 {
            self.n_f
        } else {
            self.n_b
        };
        let b_inner_rows_pg = if self.tier_split > 1 {
            self.tier_split * self.n_b
        } else {
            0
        };
        let a_start = 1
            + self.num_public_rows
            + self.n_d_active()
            + (commit_rows_pg + b_inner_rows_pg) * self.num_points;

        // ----- T -------------------------------------------------------------
        let t_structured_contribution = {
            let _span = tracing::info_span!("t_structured").entered();
            TStructuredSlicesEvaluator {
                high_challenges: context.high_challenges,
                offset_high: context.layout.offset_t >> context.offset_low_bits,
                gadget_vector: &context.g1_open,
                challenge_block_summaries: &challenge_block_summaries_by_t_vector,
                a_row_weights: &self.eq_tau1[a_start..self.rows],
            }
            .evaluate()
        };

        // ----- Fused D·ŵ + B·t̂ + A·ẑ ---------------------------------------
        let setup_contribution = {
            let _span = tracing::info_span!("setup_contribution").entered();
            jolt_start_cycle_tracking("setup_contribution");
            let result = if let Some(claim) = setup_claim {
                Ok(claim)
            } else {
                let setup_contribution_inputs = self.create_setup_contribution_inputs();
                let evaluator = SetupEvaluator::new(
                    &setup_contribution_inputs,
                    x_challenges,
                    Some(&context.eq_low),
                    Some(&context.z_block_low_eq),
                    &context.alpha_pows,
                    &context.fold_gadget,
                    context.layout.offset_e,
                    context.layout.offset_t,
                    context.layout.offset_z,
                    context.layout.offset_u,
                );
                match evaluator.evaluate::<D>(SetupEvaluatorMode::Direct { setup })? {
                    SetupEvaluation::Direct(value) => Ok(value),
                    #[cfg(test)]
                    SetupEvaluation::Recursive(_) => Err(AkitaError::InvalidSetup(
                        "setup evaluator returned recursive output for direct mode".into(),
                    )),
                }
            };
            jolt_end_cycle_tracking("setup_contribution");
            result?
        };

        // ----- Z (consistency-row) ------------------------------------------
        let z_structured_contribution = {
            let _span = tracing::info_span!("z_structured").entered();
            let g1_commit = gadget_row_scalars::<F>(self.depth_commit, self.log_basis);
            if self.block_len.is_power_of_two() {
                let z_offset_low = context.layout.offset_z & (self.block_len - 1);
                let a_block_summary: Vec<[E; 2]> = ring_multiplier_points
                    .iter()
                    .map(|ring_multiplier_point| {
                        summarize_pow2_multiplier_block_carries(
                            &context.z_block_low_eq,
                            z_offset_low,
                            self.block_len,
                            |idx| ring_multiplier_point.eval_a_at::<E>(idx, &context.alpha_pows),
                        )
                    })
                    .collect::<Result<_, _>>()?;
                ZStructuredPow2SlicesEvaluator {
                    high_challenges: context.z_high_challenges,
                    offset_high: context.layout.offset_z >> context.z_offset_low_bits,
                    g1_commit: &g1_commit,
                    fold_gadget: &context.fold_gadget,
                    a_block_summary: &a_block_summary,
                    consistency_weight: self.eq_tau1[0],
                }
                .evaluate()
            } else {
                let a_evals_by_point: Vec<Vec<E>> = ring_multiplier_points
                    .iter()
                    .map(|ring_multiplier_point| {
                        (0..self.block_len)
                            .map(|idx| {
                                ring_multiplier_point.eval_a_at::<E>(idx, &context.alpha_pows)
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<_, AkitaError>>()?;
                ZDenseSlicesEvaluator {
                    g1_commit: &g1_commit,
                    fold_gadget: &context.fold_gadget,
                    consistency_weight: self.eq_tau1[0],
                    a_evals_by_point: &a_evals_by_point,
                    full_vec_randomness: x_challenges,
                    offset_z: context.layout.offset_z,
                    block_len: self.block_len,
                }
                .evaluate()?
            }
        };

        // ----- r-tail --------------------------------------------------------
        let r_contribution = {
            let r_gadget =
                gadget_row_scalars::<F>(r_decomp_levels::<F>(self.log_basis), self.log_basis);
            let denom = context.alpha_pows[D - 1] * alpha + E::one();
            compute_r_contribution(
                self,
                x_challenges,
                context.layout.offset_r,
                denom,
                &r_gadget,
            )?
        };

        // ----- Tiered B_inner RHS: -recompose(û_concat) ----------------------
        // The B_inner block enforces `B'·t̂_slice - recompose(û) = 0`. The B'
        // matrix part is in `setup_contribution`; this is the witness-side
        // `-recompose(û)` term (a constant gadget map on the `û_concat`
        // columns), weighted by the B_inner row eq. Zero for single-tier.
        let u_recompose_contribution = if self.tier_split > 1 {
            let n_d_active = self.n_d_active();
            let f_start = 1 + self.num_public_rows + n_d_active;
            let b_inner_start = f_start + commit_rows_pg * self.num_points;
            let n_b_small = self.n_b;
            let inner_rows_pg = self.tier_split * n_b_small;
            let width_f = inner_rows_pg * self.depth_open;
            let offset_u = context.layout.offset_u;
            let mut acc = E::zero();
            for g in 0..self.num_points {
                for slice_row in 0..inner_rows_pg {
                    let row = b_inner_start + g * inner_rows_pg + slice_row;
                    let row_w = self.eq_tau1[row];
                    if row_w.is_zero() {
                        continue;
                    }
                    let base_col = offset_u + g * width_f + slice_row * self.depth_open;
                    let mut recomp = E::zero();
                    for (digit, &gd) in context.g1_open.iter().enumerate().take(self.depth_open) {
                        let eq_col = akita_algebra::offset_eq::eq_eval_at_index(
                            x_challenges,
                            base_col + digit,
                        );
                        recomp += eq_col.mul_base(gd);
                    }
                    acc -= row_w * recomp;
                }
            }
            acc
        } else {
            E::zero()
        };

        #[allow(unused_mut)]
        let mut total = e_structured_contribution
            + t_structured_contribution
            + z_structured_contribution
            + setup_contribution
            + r_contribution
            + u_recompose_contribution;

        #[cfg(feature = "zk")]
        {
            let b_blinding = compute_b_blinding_part::<F, E, D>(self, x_challenges, setup, alpha)?;
            let d_blinding = compute_d_blinding_part::<F, E, D>(self, x_challenges, setup, alpha)?;
            total = total + b_blinding + d_blinding;
        }

        Ok(total)
    }
}

#[inline]
pub(super) fn summarize_pow2_multiplier_block_carries<E, EvalAt>(
    eq_low: &[E],
    offset_low: usize,
    values_len: usize,
    mut eval_at: EvalAt,
) -> Result<[E; 2], AkitaError>
where
    E: FieldCore,
    EvalAt: FnMut(usize) -> Result<E, AkitaError>,
{
    if !values_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "peeled inner block length must be a power of two".to_string(),
        ));
    }
    if eq_low.len() != values_len {
        return Err(AkitaError::InvalidSize {
            expected: values_len,
            actual: eq_low.len(),
        });
    }
    if offset_low >= values_len {
        return Err(AkitaError::InvalidInput(
            "low offset must lie inside the peeled block".to_string(),
        ));
    }

    let inner_bits = values_len.trailing_zeros() as usize;
    let inner_mask = values_len - 1;
    let mut out = [E::zero(), E::zero()];

    for u in 0..values_len {
        let sum = offset_low + u;
        let carry = sum >> inner_bits;
        debug_assert!(
            carry < 2,
            "sum of two peeled indices must carry at most one bit"
        );
        let low_idx = sum & inner_mask;
        out[carry] += eq_low[low_idx] * eval_at(u)?;
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_ring_switch_row_eval_inner<F, E, const D: usize>(
    challenges: &Challenges,
    alpha: E,
    lp: &LevelParams,
    tau1: &[E],
    num_polys_per_commitment_group: &[usize],
    claim_to_commitment_group: &[usize],
    claim_poly_in_commitment_group: &[usize],
    gamma: &[E],
    num_public_rows: usize,
    m_row_layout: MRowLayout,
    opening_points_len: usize,
    ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
    claim_to_point: &[usize],
    witness_segment_layout: RingRelationSegmentLayout,
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    validate_level_dispatch::<D>(lp)?;
    let alpha_pows = scalar_powers(alpha, D);
    let num_claims = claim_to_point.len();
    if claim_to_commitment_group.len() != num_claims
        || claim_poly_in_commitment_group.len() != num_claims
    {
        return Err(AkitaError::InvalidProof);
    }
    let num_points = num_polys_per_commitment_group.len();
    for claim_idx in 0..num_claims {
        let group_idx = claim_to_commitment_group[claim_idx];
        if group_idx >= num_points
            || claim_poly_in_commitment_group[claim_idx]
                >= num_polys_per_commitment_group[group_idx]
        {
            return Err(AkitaError::InvalidProof);
        }
    }

    if gamma.len() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: gamma.len(),
        });
    }

    let log_basis = lp.log_basis;
    validate_log_basis(log_basis)?;
    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let depth_fold = lp.num_digits_fold(num_claims, F::modulus_bits())?;
    let num_blocks = lp.num_blocks;
    if num_blocks == 0 || !num_blocks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "num_blocks must be a non-zero power of two".to_string(),
        ));
    }
    if lp.block_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "block_len must be non-zero".to_string(),
        ));
    }
    if depth_commit == 0 || depth_open == 0 || depth_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "digit depths must be non-zero".to_string(),
        ));
    }
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    let num_t_vectors = num_polys_per_commitment_group
        .iter()
        .try_fold(0usize, |acc, &count| acc.checked_add(count))
        .ok_or_else(|| AkitaError::InvalidSetup("batched t-vector count overflow".to_string()))?;
    #[cfg(feature = "zk")]
    let d_blinding_segment_len = match m_row_layout {
        MRowLayout::WithDBlock => zk::blinding_digit_plane_count::<F>(n_d, D, log_basis),
        MRowLayout::WithoutDBlock => 0,
    };
    #[cfg(feature = "zk")]
    let b_blinding_digit_planes_per_point = zk::blinding_digit_plane_count::<F>(n_b, D, log_basis);
    #[cfg(feature = "zk")]
    let b_blinding_segment_len = num_points
        .checked_mul(b_blinding_digit_planes_per_point)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK blinding width overflow".to_string()))?;
    // Must match [`RingSwitchDeferredRowEval::total_blocks`] on the prepared value.
    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    if challenges.logical_len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.logical_len(),
        });
    }
    let block_len = lp.block_len;
    let inner_width = block_len
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
    if lp.a_key.col_len() < inner_width {
        return Err(AkitaError::InvalidSetup(
            "A-key column width is too small for verifier layout".to_string(),
        ));
    }
    if m_row_layout == MRowLayout::WithDBlock {
        let expected_d_width = depth_open
            .checked_mul(num_blocks)
            .and_then(|width| width.checked_mul(num_claims))
            .ok_or_else(|| AkitaError::InvalidSetup("D-matrix width overflow".to_string()))?;
        if lp.d_key.col_len() < expected_d_width {
            return Err(AkitaError::InvalidSetup(
                "D-key column width is too small for verifier layout".to_string(),
            ));
        }
    }
    let max_point_poly_count = num_polys_per_commitment_group
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let expected_b_width = max_point_poly_count
        .checked_mul(lp.a_key.row_len())
        .and_then(|width| width.checked_mul(depth_open))
        .and_then(|width| width.checked_mul(num_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("B-matrix width overflow".to_string()))?;
    // Tiered: the stored first-tier `B'` is the full B width divided by the
    // reuse factor `tier_split`.
    let expected_stored_b_width = if lp.f_key.is_some() {
        expected_b_width.div_ceil(lp.tier_split.max(1))
    } else {
        expected_b_width
    };
    if lp.b_key.col_len() < expected_stored_b_width {
        return Err(AkitaError::InvalidSetup(
            "B-key column width is too small for verifier layout".to_string(),
        ));
    }
    if opening_points_len != num_points {
        return Err(AkitaError::InvalidProof);
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= num_points)
    {
        return Err(AkitaError::InvalidProof);
    }
    if ring_multiplier_points.len() != opening_points_len
        || ring_multiplier_points
            .iter()
            .any(|point| point.a_len() < lp.block_len || point.b_len() != lp.num_blocks)
    {
        return Err(AkitaError::InvalidProof);
    }
    let rows = lp.m_row_count_for(num_points, num_public_rows, m_row_layout)?;

    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let c_alphas: PreparedChallengeEvals<E> = match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => PreparedChallengeEvals::Flat(
            sparse
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, E, D>(&alpha_pows))
                .collect::<Result<_, _>>()?,
        ),
        Challenges::Tensor { factored } => {
            if D < 2 {
                return Err(AkitaError::InvalidInput(
                    "tensor challenge factored evaluation requires D >= 2".to_string(),
                ));
            }
            let aggregate = factored
                .clone()
                .prepare_factored_aggregate_at_pows::<F, E, D>(alpha_pows.clone())?;
            if aggregate.num_claims() != num_claims {
                return Err(AkitaError::InvalidSize {
                    expected: num_claims,
                    actual: aggregate.num_claims(),
                });
            }
            let blocks_per_claim = aggregate.blocks_per_claim()?;
            if blocks_per_claim != lp.num_blocks {
                return Err(AkitaError::InvalidSize {
                    expected: lp.num_blocks,
                    actual: blocks_per_claim,
                });
            }
            PreparedChallengeEvals::Tensor { aggregate }
        }
    };

    let t_vector_offsets: Vec<usize> = num_polys_per_commitment_group
        .iter()
        .scan(0usize, |acc, &count| {
            let offset = *acc;
            *acc = acc.checked_add(count)?;
            Some(offset)
        })
        .collect();
    if t_vector_offsets.len() != num_points {
        return Err(AkitaError::InvalidSetup(
            "batched t-vector offsets overflow".to_string(),
        ));
    }
    let claim_to_t_vector: Vec<usize> = claim_to_commitment_group
        .iter()
        .zip(claim_poly_in_commitment_group.iter())
        .map(|(&group_idx, &poly_idx)| {
            t_vector_offsets[group_idx]
                .checked_add(poly_idx)
                .filter(|&idx| idx < num_t_vectors)
                .ok_or(AkitaError::InvalidProof)
        })
        .collect::<Result<_, _>>()?;

    Ok(RingSwitchDeferredRowEval {
        c_alphas,
        eq_tau1,
        num_t_vectors,
        num_blocks,
        num_claims,
        depth_open,
        depth_commit,
        depth_fold,
        #[cfg(feature = "zk")]
        d_blinding_segment_len,
        #[cfg(feature = "zk")]
        b_blinding_digit_planes_per_point,
        #[cfg(feature = "zk")]
        b_blinding_segment_len,
        block_len,
        inner_width,
        log_basis,
        n_a: lp.a_key.row_len(),
        n_d,
        m_row_layout,
        n_b,
        tier_split: lp.tier_split,
        n_f: lp.f_key.as_ref().map_or(0, |fk| fk.row_len()),
        num_points,
        rows,
        claim_to_t_vector,
        num_polys_per_commitment_group: num_polys_per_commitment_group.to_vec(),
        num_public_rows,
        gamma: gamma.to_vec(),
        claim_to_point: claim_to_point.to_vec(),
        witness_segment_layout,
    })
}

/// Fixed public relation inputs for verifier ring-switch replay.
pub struct RingSwitchReplay<'a, F: FieldCore, E, const D: usize> {
    pub relation: &'a RingRelationInstance<F, D>,
    pub row_coefficients: &'a [E],
    pub lp: &'a LevelParams,
}

/// Replay the verifier half of ring switching.
///
/// This handles multiple opening points, arbitrary claim-to-point mapping, and
/// point-local polynomial bundles. The recursive/single-point path is the
/// `opening_points = [pt]`, `claim_to_point = [0]`,
/// `num_polys_per_commitment_group = [1]`, `num_public_rows = 1` specialization.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or ring-switch row-eval
/// preparation fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    // `validate_ring_dispatch` is called inside `ring_switch_verifier_core`;
    // the outer wrapper just performs the witness absorb before delegating.
    transcript.absorb_and_record_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, w_commitment);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithDBlock)
}

/// Terminal variant of [`ring_switch_verifier`].
///
/// This owns the required terminal final-witness remainder absorb before
/// sampling ring-switch challenges.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or
/// ring-switch row-eval preparation fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier_terminal")]
#[inline(never)]
pub(crate) fn ring_switch_verifier_terminal<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    transcript: &mut T,
    terminal_parts: &TerminalWitnessTranscriptParts,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithoutDBlock)
}

#[tracing::instrument(skip_all, name = "ring_switch_verifier_core")]
#[inline(never)]
fn ring_switch_verifier_core<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    transcript: &mut T,
    m_row_layout: MRowLayout,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let num_public_rows = relation.num_public_rows();

    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let num_points = relation
        .commitment_routing()
        .num_polys_per_commitment_group()
        .len();

    if w_len == 0 || !w_len.is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch column count overflow".to_string()))?
        .trailing_zeros() as usize;
    let ring_bits = validate_ring_dispatch::<D>()?;
    let m_rows = lp.m_row_count_for(num_points, num_public_rows, m_row_layout)?;
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch row count overflow".to_string()))?
        .trailing_zeros() as usize;

    let stage1_tau0 = match m_row_layout {
        MRowLayout::WithDBlock => Some(
            (0..num_sc_vars)
                .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                .collect(),
        ),
        MRowLayout::WithoutDBlock => None,
    };
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let alpha_evals_y = scalar_powers(alpha, D);
    let prepared_row_eval = prepare_ring_switch_row_eval::<F, E, D>(replay, alpha, &tau1)?;

    Ok(RingSwitchVerifyOutput {
        prepared_row_eval,
        alpha_evals_y,
        col_bits,
        ring_bits,
        stage1_tau0,
        tau1,
        b: 1usize
            .checked_shl(lp.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("basis size overflow".to_string()))?,
        alpha,
    })
}

/// Prepare deferred verifier ring-switch row evaluation data from a fixed
/// [`RingRelationInstance`] and transcript-sampled row coefficients.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[tracing::instrument(skip_all, name = "prepare_ring_switch_row_eval")]
pub fn prepare_ring_switch_row_eval<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    alpha: E,
    tau1: &[E],
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let claim_to_point = relation.claim_to_point();
    relation.check_level_shape(lp)?;
    let witness_segment_layout = relation.segment_layout(lp)?;
    let routing = relation.commitment_routing();
    prepare_ring_switch_row_eval_inner::<F, E, D>(
        &relation.challenges,
        alpha,
        lp,
        tau1,
        routing.num_polys_per_commitment_group(),
        routing.claim_to_commitment_group(),
        routing.claim_poly_in_commitment_group(),
        replay.row_coefficients,
        relation.num_public_rows(),
        relation.m_row_layout(),
        relation.opening_points().len(),
        relation.ring_multiplier_points(),
        claim_to_point,
        witness_segment_layout,
    )
}
