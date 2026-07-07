use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::eval_flat_ring_at_pows;
#[cfg(test)]
use akita_algebra::ring::eval_ring_at_pows;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_types::{
    gadget_row_scalars, AkitaExpandedSetup, SetupContributionPlan, SetupContributionPlanInputs,
    WitnessChunkLayout, WitnessLayout,
};

use crate::protocol::ring_switch::{RingSwitchDeferredRowEval, RingSwitchDeferredRowGroupEval};

pub(crate) enum SetupEvaluatorMode<'a, F: FieldCore, E: FieldCore> {
    Direct {
        setup: &'a AkitaExpandedSetup<F>,
    },
    GroupedDirect {
        setup: &'a AkitaExpandedSetup<F>,
        prepared: &'a RingSwitchDeferredRowEval<E>,
        alpha_pows_b: &'a [E],
        alpha_pows_d: &'a [E],
    },
    #[cfg(test)]
    Recursive {
        setup: &'a AkitaExpandedSetup<F>,
    },
}

pub(crate) enum SetupEvaluation<E> {
    Direct(E),
    #[cfg(test)]
    Recursive(E),
}

pub(crate) struct SetupEvaluator<'a, F: FieldCore, E: FieldCore> {
    inputs: &'a SetupContributionPlanInputs<E>,
    full_vec_randomness: &'a [E],
    eq_low: Option<&'a [E]>,
    z_block_low_eq: Option<&'a [E]>,
    alpha_pows: &'a [E],
    fold_gadget: &'a [F],
    chunk_layout: &'a WitnessLayout,
}

impl<'a, F, E> SetupEvaluator<'a, F, E>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        inputs: &'a SetupContributionPlanInputs<E>,
        full_vec_randomness: &'a [E],
        eq_low: Option<&'a [E]>,
        z_block_low_eq: Option<&'a [E]>,
        alpha_pows: &'a [E],
        fold_gadget: &'a [F],
        chunk_layout: &'a WitnessLayout,
    ) -> Self {
        Self {
            inputs,
            full_vec_randomness,
            eq_low,
            z_block_low_eq,
            alpha_pows,
            fold_gadget,
            chunk_layout,
        }
    }

    pub(crate) fn evaluate<const D: usize>(
        &self,
        mode: SetupEvaluatorMode<'_, F, E>,
    ) -> Result<SetupEvaluation<E>, AkitaError>
    where
        E: akita_field::MulBaseUnreduced<F>,
    {
        if self.alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.alpha_pows.len(),
            });
        }
        match mode {
            SetupEvaluatorMode::Direct { setup } => {
                let plan = self.prepare()?;
                let value = plan.evaluate_direct::<F, D>(setup, self.alpha_pows)?;
                Ok(SetupEvaluation::Direct(value))
            }
            SetupEvaluatorMode::GroupedDirect {
                setup,
                prepared,
                alpha_pows_b,
                alpha_pows_d,
            } => {
                let value =
                    self.evaluate_grouped_direct::<D>(setup, prepared, alpha_pows_b, alpha_pows_d)?;
                Ok(SetupEvaluation::Direct(value))
            }
            #[cfg(test)]
            SetupEvaluatorMode::Recursive { setup } => {
                let plan = self.prepare()?;
                let value = recursive_inner_product::<F, E, D>(&plan, setup, self.alpha_pows)?;
                Ok(SetupEvaluation::Recursive(value))
            }
        }
    }

    pub(crate) fn prepare(&self) -> Result<SetupContributionPlan<E>, AkitaError> {
        SetupContributionPlan::prepare(
            self.inputs,
            self.full_vec_randomness,
            self.eq_low,
            self.z_block_low_eq,
            self.fold_gadget,
            self.chunk_layout,
        )
    }

    fn evaluate_grouped_direct<const D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        prepared: &RingSwitchDeferredRowEval<E>,
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
    ) -> Result<E, AkitaError> {
        let mut acc = E::zero();
        let d_d = prepared.role_dims.d_d();
        if alpha_pows_d.len() != d_d {
            return Err(AkitaError::InvalidSize {
                expected: d_d,
                actual: alpha_pows_d.len(),
            });
        }
        let d_b = prepared.role_dims.d_b();
        if alpha_pows_b.len() != d_b {
            return Err(AkitaError::InvalidSize {
                expected: d_b,
                actual: alpha_pows_b.len(),
            });
        }

        if prepared.n_d_active != 0 {
            let d_view = setup.shared_matrix.ring_view_dyn(
                prepared.n_d_active,
                prepared.e_setup_cols,
                d_d,
            )?;
            for row_idx in 0..prepared.n_d_active {
                let row = d_view.row_flat(row_idx)?;
                let row_weight = *self
                    .inputs
                    .eq_tau1
                    .get(prepared.d_start + row_idx)
                    .ok_or(AkitaError::InvalidProof)?;
                if row_weight.is_zero() {
                    continue;
                }
                for group in &prepared.groups {
                    let chunk = Self::group_chunk(prepared, group)?;
                    acc += self.evaluate_grouped_d_setup_row::<F>(
                        group,
                        chunk,
                        row,
                        row_weight,
                        alpha_pows_d,
                    )?;
                }
            }
        }

        for group in &prepared.groups {
            let a_weights = checked_slice(
                &self.inputs.eq_tau1,
                group.a_row_start,
                group.n_a,
                "grouped A rows",
            )?;
            let b_weights = checked_slice(
                &self.inputs.eq_tau1,
                group.b_row_start,
                group.n_b,
                "grouped B rows",
            )?;

            let a_view = setup
                .shared_matrix
                .ring_view_dyn(group.n_a, group.inner_width, D)?;
            let chunk = Self::group_chunk(prepared, group)?;
            let z_weights = self.grouped_z_setup_weights::<F>(group, chunk)?;
            for (row_idx, &row_weight) in a_weights.iter().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let row = a_view.row_flat(row_idx)?;
                for (col, &z_weight) in z_weights.iter().enumerate() {
                    if z_weight.is_zero() {
                        continue;
                    }
                    let coeff_start = checked_mul(col, D, "grouped A coeff start")?;
                    let coeffs = checked_slice(row, coeff_start, D, "grouped A setup coeffs")?;
                    acc += row_weight
                        * z_weight
                        * eval_flat_ring_at_pows::<F, E>(coeffs, self.alpha_pows);
                }
            }

            let b_width =
                checked_mul(group.num_claims, group.t_cols_per_vector, "grouped B width")?;
            let b_view = setup.shared_matrix.ring_view_dyn(group.n_b, b_width, d_b)?;
            for (row_idx, &row_weight) in b_weights.iter().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let row = b_view.row_flat(row_idx)?;
                acc += self.evaluate_grouped_b_setup_row::<F>(
                    group,
                    chunk,
                    row,
                    row_weight,
                    alpha_pows_b,
                )?;
            }
        }

        Ok(acc)
    }

    fn group_chunk<'b>(
        prepared: &'b RingSwitchDeferredRowEval<E>,
        group: &RingSwitchDeferredRowGroupEval<E>,
    ) -> Result<&'b WitnessChunkLayout, AkitaError> {
        prepared
            .chunk_layout
            .chunks
            .get(group.chunk_range.clone())
            .and_then(|chunks| chunks.first())
            .ok_or(AkitaError::InvalidProof)
    }

    fn evaluate_grouped_d_setup_row<Base>(
        &self,
        group: &RingSwitchDeferredRowGroupEval<E>,
        chunk: &WitnessChunkLayout,
        row: &[Base],
        row_weight: E,
        alpha_pows_d: &[E],
    ) -> Result<E, AkitaError>
    where
        Base: FieldCore,
        E: ExtField<Base>,
    {
        let block_bits = group.num_blocks.trailing_zeros() as usize;
        if block_bits > self.full_vec_randomness.len() {
            return Err(AkitaError::InvalidSize {
                expected: block_bits,
                actual: self.full_vec_randomness.len(),
            });
        }
        let eq_low = EqPolynomial::evals(&self.full_vec_randomness[..block_bits])?;
        let high_challenges = &self.full_vec_randomness[block_bits..];
        let high_len = checked_mul(group.num_claims, group.depth_open, "grouped D high width")?;
        let eq_high = high_eq_window(high_challenges, chunk.offset_e >> block_bits, high_len);
        let low_mask = group.num_blocks - 1;
        let offset_low = chunk.offset_e & low_mask;
        let d_d = alpha_pows_d.len();
        let total_blocks = checked_mul(group.num_claims, group.num_blocks, "grouped D blocks")?;
        let mut acc = E::zero();
        for digit in 0..group.depth_open {
            for flat_block in 0..total_blocks {
                let claim_idx = flat_block / group.num_blocks;
                let block_idx = flat_block % group.num_blocks;
                let high_idx = checked_add(
                    checked_mul(digit, group.num_claims, "grouped D high digit")?,
                    claim_idx,
                    "grouped D high claim",
                )?;
                let eq_weight = pow2_offset_eq_weight(
                    &eq_low, &eq_high, offset_low, block_idx, high_idx, low_mask, block_bits,
                )?;
                if eq_weight.is_zero() {
                    continue;
                }
                let local_col = checked_add(
                    checked_mul(flat_block, group.depth_open, "grouped D local column")?,
                    digit,
                    "grouped D local column",
                )?;
                let setup_col =
                    checked_add(group.e_setup_offset, local_col, "grouped D setup column")?;
                let coeff_start = checked_mul(setup_col, d_d, "grouped D coeff start")?;
                let coeffs = checked_slice(row, coeff_start, d_d, "grouped D setup coeffs")?;
                acc += row_weight
                    * eq_weight
                    * eval_flat_ring_at_pows::<Base, E>(coeffs, alpha_pows_d);
            }
        }
        Ok(acc)
    }

    fn evaluate_grouped_b_setup_row<Base>(
        &self,
        group: &RingSwitchDeferredRowGroupEval<E>,
        chunk: &WitnessChunkLayout,
        row: &[Base],
        row_weight: E,
        alpha_pows_b: &[E],
    ) -> Result<E, AkitaError>
    where
        Base: FieldCore,
        E: ExtField<Base>,
    {
        let block_bits = group.num_blocks.trailing_zeros() as usize;
        if block_bits > self.full_vec_randomness.len() {
            return Err(AkitaError::InvalidSize {
                expected: block_bits,
                actual: self.full_vec_randomness.len(),
            });
        }
        let eq_low = EqPolynomial::evals(&self.full_vec_randomness[..block_bits])?;
        let high_challenges = &self.full_vec_randomness[block_bits..];
        let high_len = checked_mul(
            checked_mul(group.num_claims, group.depth_open, "grouped B high width")?,
            group.n_a,
            "grouped B high width",
        )?;
        let eq_high = high_eq_window(high_challenges, chunk.offset_t >> block_bits, high_len);
        let low_mask = group.num_blocks - 1;
        let offset_low = chunk.offset_t & low_mask;
        let d_b = alpha_pows_b.len();
        let t_compound_per_block =
            checked_mul(group.n_a, group.depth_open, "grouped B compound stride")?;
        let mut acc = E::zero();
        for a_idx in 0..group.n_a {
            for digit in 0..group.depth_open {
                let compound = checked_add(
                    checked_mul(a_idx, group.depth_open, "grouped B compound")?,
                    digit,
                    "grouped B compound",
                )?;
                for t_vector_idx in 0..group.num_claims {
                    let high_idx = checked_add(
                        checked_mul(compound, group.num_claims, "grouped B high compound")?,
                        t_vector_idx,
                        "grouped B high vector",
                    )?;
                    for block_idx in 0..group.num_blocks {
                        let eq_weight = pow2_offset_eq_weight(
                            &eq_low, &eq_high, offset_low, block_idx, high_idx, low_mask,
                            block_bits,
                        )?;
                        if eq_weight.is_zero() {
                            continue;
                        }
                        let phys_claim_offset = checked_add(
                            checked_mul(block_idx, t_compound_per_block, "grouped B block")?,
                            compound,
                            "grouped B block",
                        )?;
                        let local_col = checked_add(
                            checked_mul(
                                t_vector_idx,
                                group.t_cols_per_vector,
                                "grouped B vector column",
                            )?,
                            phys_claim_offset,
                            "grouped B local column",
                        )?;
                        let coeff_start = checked_mul(local_col, d_b, "grouped B coeff start")?;
                        let coeffs =
                            checked_slice(row, coeff_start, d_b, "grouped B setup coeffs")?;
                        acc += row_weight
                            * eq_weight
                            * eval_flat_ring_at_pows::<Base, E>(coeffs, alpha_pows_b);
                    }
                }
            }
        }
        Ok(acc)
    }

    fn grouped_z_setup_weights<Base>(
        &self,
        group: &RingSwitchDeferredRowGroupEval<E>,
        chunk: &WitnessChunkLayout,
    ) -> Result<Vec<E>, AkitaError>
    where
        Base: FieldCore + CanonicalField,
        E: ExtField<Base>,
    {
        let fold_gadget = gadget_row_scalars::<Base>(group.depth_fold, group.log_basis);
        let z_range = checked_mul(group.block_len, group.depth_commit, "grouped Z range")?;
        if group.block_len.is_power_of_two() {
            let z_bits = group.block_len.trailing_zeros() as usize;
            if z_bits > self.full_vec_randomness.len() {
                return Err(AkitaError::InvalidSize {
                    expected: z_bits,
                    actual: self.full_vec_randomness.len(),
                });
            }
            let eq_low = EqPolynomial::evals(&self.full_vec_randomness[..z_bits])?;
            let high_challenges = &self.full_vec_randomness[z_bits..];
            let high_len =
                checked_mul(group.depth_commit, group.depth_fold, "grouped Z high width")?;
            let eq_high = high_eq_window(high_challenges, chunk.offset_z >> z_bits, high_len);
            let low_mask = group.block_len - 1;
            let offset_low = chunk.offset_z & low_mask;
            (0..z_range)
                .map(|k| {
                    let block_idx = k / group.depth_commit;
                    let dc = k % group.depth_commit;
                    let mut weight = E::zero();
                    for (df, &fold) in fold_gadget.iter().enumerate() {
                        let high_idx = checked_add(
                            checked_mul(dc, group.depth_fold, "grouped Z high dc")?,
                            df,
                            "grouped Z high df",
                        )?;
                        weight -= pow2_offset_eq_weight(
                            &eq_low, &eq_high, offset_low, block_idx, high_idx, low_mask, z_bits,
                        )?
                        .mul_base(fold);
                    }
                    Ok(weight)
                })
                .collect()
        } else {
            let z_len = checked_mul(
                checked_mul(
                    group.depth_commit,
                    group.depth_fold,
                    "grouped dense Z length",
                )?,
                group.block_len,
                "grouped dense Z length",
            )?;
            let low_bits = z_len
                .saturating_sub(1)
                .checked_next_power_of_two()
                .map(|p| p.trailing_zeros() as usize)
                .unwrap_or(0)
                .max(1)
                .min(self.full_vec_randomness.len());
            let low_mask = 1usize
                .checked_shl(
                    u32::try_from(low_bits).map_err(|_| AkitaError::InvalidSize {
                        expected: usize::BITS as usize,
                        actual: low_bits,
                    })?,
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("grouped dense Z eq width overflow".into())
                })?
                - 1;
            let eq_low = EqPolynomial::evals(&self.full_vec_randomness[..low_bits])?;
            let offset_low = chunk.offset_z & low_mask;
            let offset_high = chunk.offset_z >> low_bits;
            let max_high = checked_add(chunk.offset_z, z_len, "grouped dense Z end")?
                .checked_sub(1)
                .ok_or(AkitaError::InvalidProof)?
                >> low_bits;
            let eq_high: Vec<E> = (offset_high..=max_high)
                .map(|idx| eq_eval_at_index(&self.full_vec_randomness[low_bits..], idx))
                .collect();
            (0..z_range)
                .map(|k| {
                    let block_idx = k / group.depth_commit;
                    let dc = k % group.depth_commit;
                    let mut weight = E::zero();
                    for (df, &fold) in fold_gadget.iter().enumerate() {
                        let x = checked_add(
                            block_idx,
                            checked_mul(
                                group.block_len,
                                checked_add(
                                    df,
                                    checked_mul(dc, group.depth_fold, "grouped dense Z dc")?,
                                    "grouped dense Z df",
                                )?,
                                "grouped dense Z offset",
                            )?,
                            "grouped dense Z offset",
                        )?;
                        let shifted = checked_add(offset_low, x, "grouped dense Z low")?;
                        let low_idx = shifted & low_mask;
                        let high_carry = shifted >> low_bits;
                        let low = *eq_low.get(low_idx).ok_or(AkitaError::InvalidProof)?;
                        let high = *eq_high.get(high_carry).ok_or(AkitaError::InvalidProof)?;
                        weight -= (low * high).mul_base(fold);
                    }
                    Ok(weight)
                })
                .collect()
        }
    }
}

fn high_eq_window<E: FieldCore>(
    high_challenges: &[E],
    offset_high: usize,
    hi_len: usize,
) -> Vec<E> {
    (0..=hi_len)
        .map(|k| eq_eval_at_index(high_challenges, offset_high + k))
        .collect()
}

#[inline(always)]
fn checked_add(lhs: usize, rhs: usize, context: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} overflow")))
}

#[inline(always)]
fn checked_mul(lhs: usize, rhs: usize, context: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} overflow")))
}

#[inline(always)]
fn checked_slice<'a, T>(
    slice: &'a [T],
    start: usize,
    len: usize,
    context: &'static str,
) -> Result<&'a [T], AkitaError> {
    let end = checked_add(start, len, context)?;
    slice.get(start..end).ok_or(AkitaError::InvalidProof)
}

#[inline(always)]
fn pow2_offset_eq_weight<E: FieldCore>(
    eq_low: &[E],
    eq_high: &[E],
    offset_low: usize,
    block_idx: usize,
    high_base_idx: usize,
    low_mask: usize,
    low_bits: usize,
) -> Result<E, AkitaError> {
    let shifted = checked_add(offset_low, block_idx, "offset equality index")?;
    let low_idx = shifted & low_mask;
    let carry = shifted >> low_bits;
    let high_idx = checked_add(high_base_idx, carry, "offset equality high index")?;
    let low = eq_low.get(low_idx).ok_or(AkitaError::InvalidProof)?;
    let high = eq_high.get(high_idx).ok_or(AkitaError::InvalidProof)?;
    Ok(*low * *high)
}

#[cfg(test)]
fn recursive_inner_product<F, E, const D: usize>(
    plan: &SetupContributionPlan<E>,
    setup: &AkitaExpandedSetup<F>,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let bar_omega = plan.materialize_bar_omega();
    let setup_len = setup.shared_matrix().total_ring_elements_at::<D>()?;
    if setup_len < bar_omega.len() {
        return Err(AkitaError::InvalidSize {
            expected: bar_omega.len(),
            actual: setup_len,
        });
    }
    let setup_view = setup.shared_matrix().ring_view::<D>(1, setup_len)?;
    Ok(setup_view
        .as_slice()
        .iter()
        .zip(bar_omega)
        .map(|(ring, weight)| eval_ring_at_pows(ring, alpha_pows) * weight)
        .sum())
}
