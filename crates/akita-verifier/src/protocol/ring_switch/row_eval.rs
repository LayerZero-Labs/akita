#[cfg(feature = "zk")]
use super::super::slice_mle::{compute_b_blinding_part, compute_d_blinding_part};
use super::super::slice_mle::{
    compute_r_contribution, EStructuredSlicesEvaluator, SetupEvaluation, SetupEvaluator,
    SetupEvaluatorMode, StructuredSliceMleEvaluator, TStructuredSlicesEvaluator,
    ZDenseSlicesEvaluator, ZStructuredPow2SlicesEvaluator,
};
use super::super::{validate_log_basis, validate_ring_dispatch};
use super::{RingSwitchDeferredRowEval, RingSwitchSegmentLayout};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::scalar_powers;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_types::{
    embed_ring_subfield_scalar, gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup,
    MRowLayout, RingMultiplierOpeningPoint, RingOpeningPoint, RingSubfieldEncoding,
    SetupContributionPlanInputs,
};

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
    layout: RingSwitchSegmentLayout,
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

    pub(crate) fn segment_layout(&self) -> Result<RingSwitchSegmentLayout, AkitaError> {
        Ok(self.witness_segment_layout)
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
        let layout = self.segment_layout()?;
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
        opening_points: &[RingOpeningPoint<F>],
        ring_multiplier_points: &[RingMultiplierOpeningPoint<F, D>],
        alpha: E,
        setup_claim: Option<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    {
        let _ring_bits = validate_ring_dispatch::<D>()?;
        if ring_multiplier_points.len() != opening_points.len() {
            return Err(AkitaError::InvalidProof);
        }
        let context = self.prepare_point_context::<F, D>(x_challenges, alpha)?;
        if opening_points.len() != self.num_points {
            return Err(AkitaError::InvalidSize {
                expected: self.num_points,
                actual: opening_points.len(),
            });
        }
        for opening_point in opening_points {
            if opening_point.b.len() != self.num_blocks || opening_point.a.len() < self.block_len {
                return Err(AkitaError::InvalidProof);
            }
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
        // Per-commitment-group t-vector starting indices. Under the current
        // relation-routing contract these match opening-point groups.
        let t_vector_offsets: Vec<usize> = self
            .num_polys_per_commitment_group
            .iter()
            .scan(0usize, |acc, &count| {
                let offset = *acc;
                *acc += count;
                Some(offset)
            })
            .collect();
        for (claim_idx, &(group_idx, poly_idx)) in
            self.claim_to_commitment_group_poly.iter().enumerate()
        {
            let t_vector_idx = t_vector_offsets[group_idx] + poly_idx;
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

#[cfg(test)]
#[inline]
pub(crate) fn summarize_pow2_block_carries_base<F, E>(
    eq_low: &[E],
    offset_low: usize,
    values: &[F],
) -> Result<[E; 2], AkitaError>
where
    F: FieldCore,
    E: akita_field::ExtField<F>,
{
    if !values.len().is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "peeled inner block length must be a power of two".to_string(),
        ));
    }
    if eq_low.len() != values.len() {
        return Err(AkitaError::InvalidSize {
            expected: values.len(),
            actual: eq_low.len(),
        });
    }
    if offset_low >= values.len() {
        return Err(AkitaError::InvalidInput(
            "low offset must lie inside the peeled block".to_string(),
        ));
    }

    let inner_bits = values.len().trailing_zeros() as usize;
    let inner_mask = values.len() - 1;
    let mut out = [E::zero(), E::zero()];

    for (u, &value) in values.iter().enumerate() {
        let sum = offset_low + u;
        let carry = sum >> inner_bits;
        debug_assert!(
            carry < 2,
            "sum of two peeled indices must carry at most one bit"
        );
        let low_idx = sum & inner_mask;
        out[carry] += eq_low[low_idx].mul_base(value);
    }

    Ok(out)
}
