use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::eval_ring_at_pows;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
#[cfg(test)]
use akita_field::MulBase;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_types::AkitaExpandedSetup;

use super::super::structured_slice::POSSIBLE_CARRIES;
use crate::protocol::ring_switch::RingSwitchDeferredRowEval;

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

/// Flat coefficient weights for `<S_{<=N}, omega_S>`.
#[cfg(test)]
pub(crate) struct MaterializedSetupOmega<E> {
    pub bar_omega: Vec<E>,
    pub omega_s: Vec<E>,
}

#[cfg(test)]
impl<E: FieldCore> MaterializedSetupOmega<E> {
    pub(super) fn coefficient_weight(
        &self,
        lambda: usize,
        y: usize,
        ring_dim: usize,
    ) -> Result<E, AkitaError> {
        let idx = checked_mul(lambda, ring_dim, "omega_S coefficient offset")?
            .checked_add(y)
            .ok_or_else(|| AkitaError::InvalidSetup("omega_S coefficient index overflow".into()))?;
        self.omega_s.get(idx).copied().ok_or_else(|| {
            AkitaError::InvalidSetup("omega_S coefficient index is out of bounds".into())
        })
    }

    pub(super) fn inner_product<F, const D: usize>(
        &self,
        setup_entries: &[CyclotomicRing<F, D>],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBase<F>,
    {
        if setup_entries.len() < self.bar_omega.len() {
            return Err(AkitaError::InvalidSize {
                expected: self.bar_omega.len(),
                actual: setup_entries.len(),
            });
        }
        let expected_omega_len = checked_mul(self.bar_omega.len(), D, "omega_S length")?;
        if self.omega_s.len() != expected_omega_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_omega_len,
                actual: self.omega_s.len(),
            });
        }

        let mut total = E::zero();
        for (lambda, ring) in setup_entries.iter().enumerate().take(self.bar_omega.len()) {
            for (y, &coeff) in ring.coefficients().iter().enumerate() {
                total += self.coefficient_weight(lambda, y, D)?.mul_base(coeff);
            }
        }
        Ok(total)
    }
}

pub(crate) enum SetupEvaluatorMode<'a, F: FieldCore> {
    Direct {
        setup: &'a AkitaExpandedSetup<F>,
    },
    #[cfg(test)]
    Recursive,
}

pub(crate) enum SetupEvaluation<E> {
    Direct(E),
    #[cfg(test)]
    Recursive(MaterializedSetupOmega<E>),
}

pub(crate) struct SetupEvaluator<'a, F: FieldCore, E: FieldCore> {
    prepared: &'a RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &'a [E],
    eq_low: Option<&'a [E]>,
    z_block_low_eq: Option<&'a [E]>,
    alpha_pows: &'a [E],
    fold_gadget: &'a [F],
    offset_w: usize,
    offset_t: usize,
    offset_z: usize,
}

impl<'a, F, E> SetupEvaluator<'a, F, E>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        prepared: &'a RingSwitchDeferredRowEval<E>,
        full_vec_randomness: &'a [E],
        eq_low: Option<&'a [E]>,
        z_block_low_eq: Option<&'a [E]>,
        alpha_pows: &'a [E],
        fold_gadget: &'a [F],
        offset_w: usize,
        offset_t: usize,
        offset_z: usize,
    ) -> Self {
        Self {
            prepared,
            full_vec_randomness,
            eq_low,
            z_block_low_eq,
            alpha_pows,
            fold_gadget,
            offset_w,
            offset_t,
            offset_z,
        }
    }

    pub(crate) fn evaluate<const D: usize>(
        &self,
        mode: SetupEvaluatorMode<'_, F>,
    ) -> Result<SetupEvaluation<E>, AkitaError> {
        if self.alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.alpha_pows.len(),
            });
        }
        let plan = self.prepare()?;
        match mode {
            SetupEvaluatorMode::Direct { setup } => {
                let value = plan.evaluate_direct::<F, D>(setup, self.alpha_pows)?;
                Ok(SetupEvaluation::Direct(value))
            }
            #[cfg(test)]
            SetupEvaluatorMode::Recursive => {
                let omega = plan.materialize::<D>(self.alpha_pows)?;
                Ok(SetupEvaluation::Recursive(omega))
            }
        }
    }

    fn prepare(&self) -> Result<SetupEvalPlan<E>, AkitaError> {
        let prepared = self.prepared;
        if self.alpha_pows.is_empty() {
            return Err(AkitaError::InvalidSetup("alpha powers are empty".into()));
        }
        if prepared.num_blocks == 0 || !prepared.num_blocks.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "num_blocks must be a non-zero power of two".into(),
            ));
        }
        if prepared.block_len == 0
            || prepared.depth_open == 0
            || prepared.depth_commit == 0
            || prepared.depth_fold == 0
        {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator layout has zero width".into(),
            ));
        }
        if self.fold_gadget.len() < prepared.depth_fold {
            return Err(AkitaError::InvalidSize {
                expected: prepared.depth_fold,
                actual: self.fold_gadget.len(),
            });
        }
        if prepared.num_polys_per_point.len() != prepared.num_points {
            return Err(AkitaError::InvalidSize {
                expected: prepared.num_points,
                actual: prepared.num_polys_per_point.len(),
            });
        }

        let block_bits = prepared.num_blocks.trailing_zeros() as usize;
        if block_bits > self.full_vec_randomness.len() {
            return Err(AkitaError::InvalidSize {
                expected: block_bits,
                actual: self.full_vec_randomness.len(),
            });
        }
        let block_mask = prepared.num_blocks - 1;
        let block_offset_low = self.offset_w & block_mask;
        let w_offset_high = self.offset_w >> block_bits;
        let t_offset_high = self.offset_t >> block_bits;
        let high_challenges = &self.full_vec_randomness[block_bits..];
        let eq_low_storage;
        let eq_low = if let Some(eq_low) = self.eq_low {
            eq_low
        } else {
            eq_low_storage = EqPolynomial::evals(&self.full_vec_randomness[..block_bits])?;
            &eq_low_storage
        };
        if eq_low.len() < prepared.num_blocks {
            return Err(AkitaError::InvalidSize {
                expected: prepared.num_blocks,
                actual: eq_low.len(),
            });
        }

        let z_offset_low_bits = prepared.block_len.trailing_zeros() as usize;
        if z_offset_low_bits > self.full_vec_randomness.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_offset_low_bits,
                actual: self.full_vec_randomness.len(),
            });
        }
        let z_offset_low = self.offset_z & prepared.block_len.saturating_sub(1);
        let z_range = prepared.inner_width;
        let expected_z_range = checked_mul(prepared.block_len, prepared.depth_commit, "Z width")?;
        if z_range != expected_z_range {
            return Err(AkitaError::InvalidSize {
                expected: expected_z_range,
                actual: z_range,
            });
        }
        let z_dims_pow2 = prepared.block_len.is_power_of_two();

        let n_d_active = prepared.n_d_active();
        let d_start = checked_add(1, prepared.num_public_rows, "D row start")?;
        let b_start = checked_add(d_start, n_d_active, "B row start")?;
        let b_rows = checked_mul(prepared.n_b, prepared.num_points, "B row count")?;
        let a_start = checked_add(b_start, b_rows, "A row start")?;
        let a_end = checked_add(a_start, prepared.n_a, "A row end")?;
        if a_end > prepared.rows || prepared.rows > prepared.eq_tau1.len() {
            return Err(AkitaError::InvalidSetup(
                "M-row weights are inconsistent with setup evaluator layout".into(),
            ));
        }

        let stride_t = checked_mul(prepared.n_a, prepared.depth_open, "T stride")?;
        let cols_per_poly_t = checked_mul(stride_t, prepared.num_blocks, "T polynomial width")?;
        let b_per_claim_w = checked_mul(prepared.num_blocks, prepared.depth_open, "W claim width")?;
        let n_cols_w = checked_mul(prepared.num_claims, b_per_claim_w, "W column width")?;
        let max_group_poly_count = prepared
            .num_polys_per_point
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let n_cols_t = checked_mul(max_group_poly_count, cols_per_poly_t, "T column width")?;

        let d_required = checked_mul(n_d_active, n_cols_w, "D setup footprint")?;
        let b_required = checked_mul(prepared.n_b, n_cols_t, "B setup footprint")?;
        let a_required = checked_mul(prepared.n_a, z_range, "A setup footprint")?;
        let required = d_required.max(b_required).max(a_required);
        if required == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator requires a non-empty packed footprint".into(),
            ));
        }

        let mut group_offsets = Vec::with_capacity(prepared.num_polys_per_point.len());
        let mut next_offset = 0usize;
        for &group_poly_count in &prepared.num_polys_per_point {
            group_offsets.push(next_offset);
            next_offset = checked_add(next_offset, group_poly_count, "T vector offset")?;
        }
        if next_offset != prepared.num_t_vectors {
            return Err(AkitaError::InvalidSetup(
                "T vector count is inconsistent with point polynomial counts".into(),
            ));
        }

        let w_eq_slice: Vec<E> = if n_d_active == 0 {
            Vec::new()
        } else {
            jolt_start_cycle_tracking("setup_w_eq_slice");
            let w_hi_len =
                checked_mul(prepared.num_claims, prepared.depth_open, "W high-eq width")?;
            let eq_hi_w_table: Vec<E> = (0..=w_hi_len)
                .map(|k| eq_eval_at_index(high_challenges, w_offset_high + k))
                .collect();
            let slice = cfg_into_iter!(0..n_cols_w)
                .map(|current_index| {
                    let (low_eq_idx, high_eq_idx) = get_eq_indices_for_d(
                        current_index,
                        prepared.depth_open,
                        prepared.num_blocks,
                        prepared.num_claims,
                        b_per_claim_w,
                        block_offset_low,
                        block_mask,
                        block_bits,
                    );
                    eq_low[low_eq_idx] * eq_hi_w_table[high_eq_idx]
                })
                .collect();
            jolt_end_cycle_tracking("setup_w_eq_slice");
            slice
        };

        jolt_start_cycle_tracking("setup_t_eq_slices");
        let t_hi_len = checked_mul(
            checked_mul(
                prepared.num_t_vectors,
                prepared.depth_open,
                "T high-eq width",
            )?,
            prepared.n_a,
            "T high-eq width",
        )?;
        let eq_hi_t_table: Vec<E> = (0..=t_hi_len)
            .map(|k| eq_eval_at_index(high_challenges, t_offset_high + k))
            .collect();
        let t_eq_slice_per_group: Vec<Vec<E>> = (0..prepared.num_points)
            .map(|g| {
                let group_size = prepared.num_polys_per_point[g];
                cfg_into_iter!(0..n_cols_t)
                    .map(|c| {
                        let poly_idx = c / cols_per_poly_t;
                        if poly_idx >= group_size {
                            return E::zero();
                        }
                        let flat_t_vector = group_offsets[g] + poly_idx;
                        let (low_eq_idx, high_eq_idx) = get_eq_indices_for_b(
                            c,
                            flat_t_vector,
                            prepared.depth_open,
                            prepared.n_a,
                            prepared.num_blocks,
                            prepared.num_t_vectors,
                            stride_t,
                            block_offset_low,
                            block_mask,
                            block_bits,
                        );
                        eq_low[low_eq_idx] * eq_hi_t_table[high_eq_idx]
                    })
                    .collect()
            })
            .collect();
        jolt_end_cycle_tracking("setup_t_eq_slices");

        jolt_start_cycle_tracking("setup_z_eq_slice");
        let z_eq_slice = if z_dims_pow2 {
            let z_block_low_storage;
            let z_block_low_eq = if let Some(z_block_low_eq) = self.z_block_low_eq {
                z_block_low_eq
            } else {
                z_block_low_storage =
                    EqPolynomial::evals(&self.full_vec_randomness[..z_offset_low_bits])?;
                &z_block_low_storage
            };
            if z_block_low_eq.len() < prepared.block_len {
                return Err(AkitaError::InvalidSize {
                    expected: prepared.block_len,
                    actual: z_block_low_eq.len(),
                });
            }

            let z_offset_high = self.offset_z >> z_offset_low_bits;
            let z_block_mask = prepared.block_len.wrapping_sub(1);
            let z_high_challenges = &self.full_vec_randomness[z_offset_low_bits..];
            let num_q_z = checked_mul(
                checked_mul(prepared.num_points, prepared.depth_fold, "Z high-eq width")?,
                prepared.depth_commit,
                "Z high-eq width",
            )?;
            let eq_hi_z_table: Vec<E> = (0..=num_q_z)
                .map(|k| eq_eval_at_index(z_high_challenges, z_offset_high + k))
                .collect();
            let s_per_dc_per_carry: Vec<[E; POSSIBLE_CARRIES]> = (0..prepared.depth_commit)
                .map(|dc| {
                    let mut s = [E::zero(); POSSIBLE_CARRIES];
                    for (carry_slot, slot) in s.iter_mut().enumerate() {
                        let mut acc = E::zero();
                        for (df, &fg) in self
                            .fold_gadget
                            .iter()
                            .enumerate()
                            .take(prepared.depth_fold)
                        {
                            for pt in 0..prepared.num_points {
                                let k = pt
                                    + prepared.num_points * df
                                    + prepared.num_points * prepared.depth_fold * dc
                                    + carry_slot;
                                acc += eq_hi_z_table[k].mul_base(fg);
                            }
                        }
                        *slot = -acc;
                    }
                    s
                })
                .collect();
            cfg_into_iter!(0..z_range)
                .map(|c| {
                    let (low_eq_idx, depth_commit_idx, block_carry) = get_eq_indices_for_a(
                        c,
                        prepared.depth_commit,
                        z_offset_low,
                        z_block_mask,
                        z_offset_low_bits,
                    );
                    z_block_low_eq[low_eq_idx] * s_per_dc_per_carry[depth_commit_idx][block_carry]
                })
                .collect()
        } else {
            let z_total_blocks_dense = checked_mul(
                prepared.block_len,
                prepared.num_points,
                "dense Z block width",
            )?;
            let z_len_dense = checked_mul(
                checked_mul(prepared.depth_fold, prepared.depth_commit, "dense Z length")?,
                z_total_blocks_dense,
                "dense Z length",
            )?;
            let n_rand = self.full_vec_randomness.len();
            let k = z_len_dense
                .saturating_sub(1)
                .checked_next_power_of_two()
                .map(|p| p.trailing_zeros() as usize)
                .unwrap_or(0)
                .max(1)
                .min(n_rand);
            let mask = 1usize
                .checked_shl(u32::try_from(k).map_err(|_| AkitaError::InvalidSize {
                    expected: usize::BITS as usize,
                    actual: k,
                })?)
                .ok_or_else(|| AkitaError::InvalidSetup("dense Z eq width overflow".into()))?
                - 1;
            let offset_z_dense_low = self.offset_z & mask;
            let offset_z_dense_high = self.offset_z >> k;
            let eq_low_z_dense = EqPolynomial::evals(&self.full_vec_randomness[..k])?;
            let max_high = self
                .offset_z
                .checked_add(z_len_dense)
                .and_then(|end| end.checked_sub(1))
                .ok_or_else(|| AkitaError::InvalidSetup("dense Z high-eq bound overflow".into()))?
                >> k;
            let n_high = max_high - offset_z_dense_high + 1;
            let eq_high_z_dense: Vec<E> = (0..n_high)
                .map(|h| eq_eval_at_index(&self.full_vec_randomness[k..], offset_z_dense_high + h))
                .collect();

            cfg_into_iter!(0..z_range)
                .map(|c| {
                    let dc = c % prepared.depth_commit;
                    let blk = c / prepared.depth_commit;
                    let mut acc = E::zero();
                    for pt in 0..prepared.num_points {
                        for (df, &fg) in self
                            .fold_gadget
                            .iter()
                            .enumerate()
                            .take(prepared.depth_fold)
                        {
                            let x = blk
                                + prepared.block_len * pt
                                + prepared.block_len * prepared.num_points * df
                                + prepared.block_len
                                    * prepared.num_points
                                    * prepared.depth_fold
                                    * dc;
                            let sum = offset_z_dense_low + x;
                            let low_idx = sum & mask;
                            let high_idx = sum >> k;
                            let eq_val = eq_low_z_dense[low_idx] * eq_high_z_dense[high_idx];
                            acc += eq_val.mul_base(fg);
                        }
                    }
                    -acc
                })
                .collect()
        };
        jolt_end_cycle_tracking("setup_z_eq_slice");

        jolt_start_cycle_tracking("setup_b_weights");
        let b_weights_by_row: Vec<Vec<E>> = (0..prepared.n_b)
            .map(|row| {
                (0..prepared.num_points)
                    .map(|g| prepared.eq_tau1[b_start + g * prepared.n_b + row])
                    .collect()
            })
            .collect();
        jolt_end_cycle_tracking("setup_b_weights");

        let mut endpoints = Vec::with_capacity(n_d_active + prepared.n_b + prepared.n_a + 2);
        endpoints.push(0);
        endpoints.push(required);
        push_role_boundaries(&mut endpoints, n_d_active, n_cols_w, "D")?;
        push_role_boundaries(&mut endpoints, prepared.n_b, n_cols_t, "B")?;
        push_role_boundaries(&mut endpoints, prepared.n_a, z_range, "A")?;
        endpoints.sort_unstable();
        endpoints.dedup();

        Ok(SetupEvalPlan {
            required,
            d_stride: n_cols_w,
            b_stride: n_cols_t,
            z_range,
            d_required,
            b_required,
            a_required,
            w_eq_slice,
            t_eq_slice_per_group,
            z_eq_slice,
            d_weights: prepared.eq_tau1[d_start..(d_start + n_d_active)].to_vec(),
            b_weights_by_row,
            a_weights: prepared.eq_tau1[a_start..a_end].to_vec(),
            endpoints,
        })
    }
}

struct SetupEvalPlan<E> {
    required: usize,
    d_stride: usize,
    b_stride: usize,
    z_range: usize,
    d_required: usize,
    b_required: usize,
    a_required: usize,
    w_eq_slice: Vec<E>,
    t_eq_slice_per_group: Vec<Vec<E>>,
    z_eq_slice: Vec<E>,
    d_weights: Vec<E>,
    b_weights_by_row: Vec<Vec<E>>,
    a_weights: Vec<E>,
    endpoints: Vec<usize>,
}

impl<E: FieldCore> SetupEvalPlan<E> {
    #[cfg(test)]
    fn materialize<const D: usize>(
        &self,
        alpha_pows: &[E],
    ) -> Result<MaterializedSetupOmega<E>, AkitaError> {
        jolt_start_cycle_tracking("setup_bar_omega");
        let bar_omega = self.materialize_bar_omega();
        jolt_end_cycle_tracking("setup_bar_omega");

        jolt_start_cycle_tracking("setup_omega_s");
        let omega_len = checked_mul(bar_omega.len(), D, "omega_S length")?;
        let mut omega_s = Vec::with_capacity(omega_len);
        for &weight in &bar_omega {
            for &alpha_pow in alpha_pows {
                omega_s.push(weight * alpha_pow);
            }
        }
        jolt_end_cycle_tracking("setup_omega_s");

        Ok(MaterializedSetupOmega { bar_omega, omega_s })
    }

    #[cfg(test)]
    fn materialize_bar_omega(&self) -> Vec<E> {
        let mut bar_omega = vec![E::zero(); self.required];
        for segment in self.segments() {
            for lambda in segment.lo..segment.hi {
                bar_omega[lambda] = self.weight_at(lambda, &segment);
            }
        }
        bar_omega
    }

    #[cfg(test)]
    fn weight_at(&self, lambda: usize, segment: &SetupSegment<'_, E>) -> E {
        let mut weight = E::zero();
        if segment.has_d {
            weight += segment.d_weight * self.w_eq_slice[lambda - segment.d_start_abs];
        }
        if segment.has_b {
            for (g, t_eq_slice) in self.t_eq_slice_per_group.iter().enumerate() {
                weight += segment.b_weights[g] * t_eq_slice[lambda - segment.b_start_abs];
            }
        }
        if segment.has_a {
            weight += segment.a_weight * self.z_eq_slice[lambda - segment.a_start_abs];
        }
        weight
    }

    fn segments(&self) -> Vec<SetupSegment<'_, E>> {
        (0..self.endpoints.len().saturating_sub(1))
            .filter_map(|idx| {
                let lo = self.endpoints[idx];
                let hi = self.endpoints[idx + 1];
                if lo == hi {
                    return None;
                }

                let has_d = self.d_stride != 0 && lo < self.d_required;
                let d_row = if has_d { lo / self.d_stride } else { 0 };
                let d_start_abs = if has_d { d_row * self.d_stride } else { 0 };
                let d_weight = if has_d {
                    self.d_weights[d_row]
                } else {
                    E::zero()
                };

                let has_b = self.b_stride != 0 && lo < self.b_required;
                let b_row = if has_b { lo / self.b_stride } else { 0 };
                let b_start_abs = if has_b { b_row * self.b_stride } else { 0 };
                let b_weights: &[E] = if has_b {
                    &self.b_weights_by_row[b_row]
                } else {
                    &[]
                };

                let has_a = self.z_range != 0 && lo < self.a_required;
                let a_row = if has_a { lo / self.z_range } else { 0 };
                let a_start_abs = if has_a { a_row * self.z_range } else { 0 };
                let a_weight = if has_a {
                    self.a_weights[a_row]
                } else {
                    E::zero()
                };

                Some(SetupSegment {
                    lo,
                    hi,
                    has_d,
                    d_start_abs,
                    d_weight,
                    has_b,
                    b_start_abs,
                    b_weights,
                    has_a,
                    a_start_abs,
                    a_weight,
                })
            })
            .collect()
    }
}

impl<E> SetupEvalPlan<E>
where
    E: FieldCore,
{
    fn evaluate_direct<F, const D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let setup_len = setup.shared_matrix().total_ring_elements_at::<D>()?;
        if self.required > setup_len {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected verifier layout".into(),
            ));
        }
        let setup_view = setup.shared_matrix().ring_view::<D>(1, setup_len)?;
        let setup_flat = setup_view.as_slice();

        jolt_start_cycle_tracking("setup_inner_product_segments");
        let segments = self.segments();
        let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
            .map(|idx| -> Result<E, AkitaError> {
                let segment = &segments[idx];
                macro_rules! segment_sum {
                    ($has_d:literal, $has_b:literal, $has_a:literal) => {
                        packed_slice_inner_sum::<F, E, D, $has_d, $has_b, $has_a>(
                            segment.lo..segment.hi,
                            setup_flat,
                            alpha_pows,
                            segment.d_start_abs,
                            segment.d_weight,
                            &self.w_eq_slice,
                            segment.b_start_abs,
                            segment.b_weights,
                            &self.t_eq_slice_per_group,
                            segment.a_start_abs,
                            segment.a_weight,
                            &self.z_eq_slice,
                        )
                    };
                }

                Ok(match (segment.has_d, segment.has_b, segment.has_a) {
                    (true, true, true) => segment_sum!(true, true, true),
                    (true, true, false) => segment_sum!(true, true, false),
                    (true, false, true) => segment_sum!(true, false, true),
                    (false, true, true) => segment_sum!(false, true, true),
                    (true, false, false) => segment_sum!(true, false, false),
                    (false, true, false) => segment_sum!(false, true, false),
                    (false, false, true) => segment_sum!(false, false, true),
                    (false, false, false) => E::zero(),
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        jolt_end_cycle_tracking("setup_inner_product_segments");

        Ok(segment_sums.into_iter().sum())
    }
}

struct SetupSegment<'a, E> {
    lo: usize,
    hi: usize,
    has_d: bool,
    d_start_abs: usize,
    d_weight: E,
    has_b: bool,
    b_start_abs: usize,
    b_weights: &'a [E],
    has_a: bool,
    a_start_abs: usize,
    a_weight: E,
}

/// Sum a contiguous absolute slice of the packed setup prefix.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_slice_inner_sum<
    F,
    E,
    const D: usize,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    setup_flat: &[CyclotomicRing<F, D>],
    alpha_pows: &[E],
    d_start: usize,
    d_weight: E,
    w_eq: &[E],
    b_start: usize,
    b_weights: &[E],
    t_eq_per_group: &[Vec<E>],
    a_start: usize,
    a_weight: E,
    z_eq: &[E],
) -> E
where
    F: FieldCore,
    E: ExtField<F>,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, lambda| {
            let mut weight = E::zero();
            if HAS_D {
                weight += d_weight * w_eq[lambda - d_start];
            }
            if HAS_B {
                for (g, t_eq_slice) in t_eq_per_group.iter().enumerate() {
                    weight += b_weights[g] * t_eq_slice[lambda - b_start];
                }
            }
            if HAS_A {
                weight += a_weight * z_eq[lambda - a_start];
            }
            if !weight.is_zero() {
                acc += eval_ring_at_pows(&setup_flat[lambda], alpha_pows) * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

/// Translate a D-column (D-physical order `[digit, block, claim]`) into
/// the M-layout `(low_block_eq_idx, high_eq_idx)` pair.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_eq_indices_for_d(
    current_index: usize,
    num_digits: usize,
    num_blocks: usize,
    num_claims: usize,
    blocks_per_claim_w: usize,
    block_offset_low: usize,
    block_mask: usize,
    block_bits: usize,
) -> (usize, usize) {
    let digit_idx = current_index % num_digits;
    let block_idx = (current_index / num_digits) % num_blocks;
    let claim_idx = current_index / blocks_per_claim_w;
    let m_layout_high_idx = digit_idx * num_claims + claim_idx;
    let block_sum = block_offset_low + block_idx;
    let low_eq_idx = block_sum & block_mask;
    let block_carry = block_sum >> block_bits;
    let high_eq_idx = m_layout_high_idx + block_carry;
    (low_eq_idx, high_eq_idx)
}

/// Translate a B-column (B-physical order `[digit, a_row, block, t_vector]`)
/// into `(low_block_eq_idx, high_eq_idx)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_eq_indices_for_b(
    current_index: usize,
    flat_t_vector: usize,
    num_digits: usize,
    n_a: usize,
    num_blocks: usize,
    num_t_vectors: usize,
    stride_t: usize,
    block_offset_low: usize,
    block_mask: usize,
    block_bits: usize,
) -> (usize, usize) {
    let digit_idx = current_index % num_digits;
    let a_row_idx = (current_index / num_digits) % n_a;
    let block_idx = (current_index / stride_t) % num_blocks;
    let m_layout_high_idx =
        flat_t_vector + num_t_vectors * digit_idx + num_t_vectors * num_digits * a_row_idx;
    let block_sum = block_offset_low + block_idx;
    let low_eq_idx = block_sum & block_mask;
    let block_carry = block_sum >> block_bits;
    let high_eq_idx = m_layout_high_idx + block_carry;
    (low_eq_idx, high_eq_idx)
}

/// Translate an A-column (A-physical order `[dc, block]`) into the
/// `(low_block_eq_idx, dc_idx, block_carry)` triple.
#[inline(always)]
fn get_eq_indices_for_a(
    current_index: usize,
    depth_commit: usize,
    z_offset_low: usize,
    z_block_mask: usize,
    z_offset_low_bits: usize,
) -> (usize, usize, usize) {
    let block_idx = current_index / depth_commit;
    let depth_commit_idx = current_index % depth_commit;
    let block_sum = z_offset_low + block_idx;
    let low_eq_idx = block_sum & z_block_mask;
    let block_carry = block_sum >> z_offset_low_bits;
    (low_eq_idx, depth_commit_idx, block_carry)
}

#[inline(always)]
fn push_role_boundaries(
    endpoints: &mut Vec<usize>,
    rows: usize,
    stride: usize,
    name: &'static str,
) -> Result<(), AkitaError> {
    if rows == 0 || stride == 0 {
        return Ok(());
    }
    let mut boundary = 0usize;
    for _ in 0..rows {
        boundary = boundary
            .checked_add(stride)
            .ok_or_else(|| AkitaError::InvalidSetup(format!("packed {name} boundary overflow")))?;
        endpoints.push(boundary);
    }
    Ok(())
}

#[inline(always)]
fn checked_add(lhs: usize, rhs: usize, name: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{name} overflow")))
}

#[inline(always)]
fn checked_mul(lhs: usize, rhs: usize, name: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{name} overflow")))
}
