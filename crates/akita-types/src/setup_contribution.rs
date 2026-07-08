//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns the pure layout/weight derivation for the stage-3 setup
//! product. The prover consumes the materialized `bar_omega` vector, while the
//! verifier can evaluate the same plan directly against the packed setup.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::{eval_flat_ring_at_pows, eval_ring_at_pows};
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase};

use crate::layout::{LevelParams, MRowLayout};
use crate::proof::AkitaExpandedSetup;
use crate::WitnessChunkLayout;
const POSSIBLE_CARRIES: usize = 2;

/// Minimal setup-contribution data needed to derive `bar_omega`.
#[derive(Clone)]
pub struct SetupContributionPlanInputs<E: FieldCore> {
    pub eq_tau1: Vec<E>,
    pub num_t_vectors: usize,
    pub num_blocks: usize,
    pub num_claims: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub block_len: usize,
    pub inner_width: usize,
    pub n_a: usize,
    pub n_d: usize,
    pub m_row_layout: MRowLayout,
    pub n_b: usize,
    pub num_groups: usize,
    pub rows: usize,
    pub num_polys_per_group: Vec<usize>,
}

impl<E: FieldCore> SetupContributionPlanInputs<E> {
    /// Build challenge-free setup-contribution inputs from per-level params.
    ///
    /// Mirrors the prover's `create_setup_contribution_inputs` field derivation
    /// without materializing `eq_tau1`.
    ///
    /// # Errors
    ///
    /// Returns an error when level layout parameters are inconsistent.
    pub fn from_level_params(
        lp: &LevelParams,
        num_polys_per_group: &[usize],
        m_row_layout: MRowLayout,
        depth_fold: usize,
    ) -> Result<Self, AkitaError> {
        let num_polynomials: usize = num_polys_per_group.iter().copied().sum();
        let num_groups = num_polys_per_group.len().max(1);
        let depth_commit = lp.num_digits_commit;
        let depth_open = lp.num_digits_open;
        if lp.num_blocks == 0 || !lp.num_blocks.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "num_blocks must be a non-zero power of two".into(),
            ));
        }
        if lp.block_len == 0 || depth_commit == 0 || depth_open == 0 || depth_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator layout has zero width".into(),
            ));
        }
        let inner_width = lp
            .block_len
            .checked_mul(depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".into()))?;
        if lp.a_key.col_len() < inner_width {
            return Err(AkitaError::InvalidSetup(
                "A-key column width is too small for setup contribution layout".into(),
            ));
        }
        let expected_b_width = num_polynomials
            .checked_mul(lp.a_key.row_len())
            .and_then(|width| width.checked_mul(depth_open))
            .and_then(|width| width.checked_mul(lp.num_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("B-matrix width overflow".into()))?;
        if lp.b_key.col_len() < expected_b_width {
            return Err(AkitaError::InvalidSetup(
                "B-key column width is too small for setup contribution layout".into(),
            ));
        }
        let rows = lp.m_row_count_for(num_groups, m_row_layout)?;
        Ok(Self {
            eq_tau1: Vec::new(),
            num_t_vectors: num_polynomials,
            num_blocks: lp.num_blocks,
            num_claims: num_polynomials,
            depth_open,
            depth_commit,
            depth_fold,
            block_len: lp.block_len,
            inner_width,
            n_a: lp.a_key.row_len(),
            n_d: lp.d_key.row_len(),
            m_row_layout,
            n_b: lp.b_key.row_len(),
            num_groups,
            rows,
            num_polys_per_group: num_polys_per_group.to_vec(),
        })
    }

    /// Attach the τ₁ eq-polynomial expansion after [`Self::from_level_params`].
    ///
    /// # Errors
    ///
    /// Returns an error when `tau1` cannot be expanded or is shorter than `min_rows`.
    pub fn with_eq_tau1_from_tau(
        mut self,
        tau1: &[E],
        min_rows: usize,
    ) -> Result<Self, AkitaError> {
        self.eq_tau1 = EqPolynomial::evals(tau1)?;
        if self.eq_tau1.len() < min_rows {
            return Err(AkitaError::InvalidSize {
                expected: min_rows,
                actual: self.eq_tau1.len(),
            });
        }
        Ok(self)
    }
}

#[derive(Clone)]
pub struct SetupContributionGroupInputs {
    pub e_col_offset: usize,
    pub num_claims: usize,
    pub num_blocks: usize,
    pub block_len: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub log_basis: u32,
    pub n_a: usize,
    pub n_b: usize,
    pub t_cols_per_vector: usize,
    pub a_row_start: usize,
    pub b_row_start: usize,
    pub blocks_per_chunk: usize,
    pub chunks: Vec<WitnessChunkLayout>,
}

pub struct GroupedSetupContributionPlan<E> {
    groups: Vec<GroupSetupContributionPlan<E>>,
    d_rows: usize,
    d_physical_cols: usize,
}

struct GroupSetupContributionPlan<E> {
    e_col_offset: usize,
    t_cols: usize,
    z_cols: usize,
    n_a: usize,
    n_b: usize,
    e_eq_slice: Vec<E>,
    t_eq_slice: Vec<E>,
    z_eq_slice: Vec<E>,
    a_weights: Vec<E>,
    b_weights: Vec<E>,
    d_weights: Vec<E>,
}

/// Prepared setup-contribution weights.
pub struct SetupContributionPlan<E> {
    required: usize,
    d_stride: usize,
    b_stride: usize,
    z_range: usize,
    d_required: usize,
    b_required: usize,
    a_required: usize,
    e_eq_slice: Vec<E>,
    t_eq_slice_per_group: Vec<Vec<E>>,
    z_eq_slice: Vec<E>,
    d_weights: Vec<E>,
    b_weights_by_row: Vec<Vec<E>>,
    a_weights: Vec<E>,
    endpoints: Vec<usize>,
}

impl<E: FieldCore> SetupContributionPlan<E> {
    pub fn required(&self) -> usize {
        self.required
    }

    pub fn materialize_bar_omega(&self) -> Vec<E> {
        let segments = self.segments();
        let segment_values = cfg_into_iter!(segments)
            .map(|segment| {
                let values = cfg_into_iter!(segment.lo..segment.hi)
                    .map(|lambda| self.weight_at(lambda, &segment))
                    .collect::<Vec<_>>();
                (segment.lo, values)
            })
            .collect::<Vec<_>>();
        let mut bar_omega = vec![E::zero(); self.required];
        for (lo, values) in segment_values {
            for (offset, value) in values.into_iter().enumerate() {
                bar_omega[lo + offset] = value;
            }
        }
        bar_omega
    }

    pub fn evaluate_bar_omega_with_eq(&self, eq_lambda: &[E]) -> Result<E, AkitaError> {
        let lambda_len = self
            .required
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("setup omega lambda length overflow".into()))?;
        if eq_lambda.len() != lambda_len {
            return Err(AkitaError::InvalidSize {
                expected: lambda_len,
                actual: eq_lambda.len(),
            });
        }

        let segments = self.segments();
        let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
            .map(|idx| {
                let segment = &segments[idx];
                macro_rules! segment_sum {
                    ($has_d:literal, $has_b:literal, $has_a:literal) => {
                        bar_omega_segment_eval::<E, $has_d, $has_b, $has_a>(
                            segment.lo..segment.hi,
                            eq_lambda,
                            segment.d_start_abs,
                            segment.d_weight,
                            &self.e_eq_slice,
                            segment.b_start_abs,
                            segment.b_weights,
                            &self.t_eq_slice_per_group,
                            segment.a_start_abs,
                            segment.a_weight,
                            &self.z_eq_slice,
                        )
                    };
                }

                match (segment.has_d, segment.has_b, segment.has_a) {
                    (true, true, true) => segment_sum!(true, true, true),
                    (true, true, false) => segment_sum!(true, true, false),
                    (true, false, true) => segment_sum!(true, false, true),
                    (false, true, true) => segment_sum!(false, true, true),
                    (true, false, false) => segment_sum!(true, false, false),
                    (false, true, false) => segment_sum!(false, true, false),
                    (false, false, true) => segment_sum!(false, false, true),
                    (false, false, false) => segment_sum!(false, false, false),
                }
            })
            .collect();
        Ok(segment_sums.into_iter().sum())
    }

    pub fn evaluate_prepared_direct<F, const D: usize>(
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
                            &self.e_eq_slice,
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
                    (false, false, false) => segment_sum!(false, false, false),
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;

        Ok(segment_sums.into_iter().sum())
    }

    pub fn prepare_grouped<F>(
        inputs: &SetupContributionPlanInputs<E>,
        full_vec_randomness: &[E],
        groups: &[SetupContributionGroupInputs],
        d_row_start: usize,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<GroupedSetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let d_weights = if d_rows == 0 {
            Vec::new()
        } else {
            checked_slice(&inputs.eq_tau1, d_row_start, d_rows, "grouped D rows")?.to_vec()
        };
        let num_groups = groups.len();
        let groups = groups
            .iter()
            .map(|group| {
                validate_group_chunk_layout(group, num_groups)?;
                let t_cols =
                    checked_mul(group.num_claims, group.t_cols_per_vector, "grouped B width")?;
                let z_cols = checked_mul(group.block_len, group.depth_commit, "grouped Z range")?;
                let a_weights = checked_slice(
                    &inputs.eq_tau1,
                    group.a_row_start,
                    group.n_a,
                    "grouped A rows",
                )?
                .to_vec();
                let b_weights = checked_slice(
                    &inputs.eq_tau1,
                    group.b_row_start,
                    group.n_b,
                    "grouped B rows",
                )?
                .to_vec();
                Ok(GroupSetupContributionPlan {
                    e_col_offset: group.e_col_offset,
                    t_cols,
                    z_cols,
                    n_a: group.n_a,
                    n_b: group.n_b,
                    e_eq_slice: grouped_e_setup_weights(group, full_vec_randomness)?,
                    t_eq_slice: grouped_t_setup_weights(group, full_vec_randomness)?,
                    z_eq_slice: grouped_z_setup_weights::<F, E>(group, full_vec_randomness)?,
                    a_weights,
                    b_weights,
                    d_weights: d_weights.clone(),
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        Ok(GroupedSetupContributionPlan {
            groups,
            d_rows,
            d_physical_cols,
        })
    }

    /// Canonical setup-contribution weight for shared-vector entry `lambda`
    /// within `segment` — the multiplier the verifier applies to
    /// `eval_ring_at_pows(setup_flat[lambda])` (see `packed_slice_inner_sum`,
    /// which carries an identical const-generic copy for the hot direct scan).
    fn weight_at(&self, lambda: usize, segment: &SetupSegment<'_, E>) -> E {
        let mut weight = E::zero();
        if segment.has_d {
            weight += segment.d_weight * self.e_eq_slice[lambda - segment.d_start_abs];
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

    #[cfg(test)]
    fn from_single_grouped(grouped: &GroupedSetupContributionPlan<E>) -> Result<Self, AkitaError> {
        if grouped.groups.len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "flat setup contribution evaluation requires exactly one group".into(),
            ));
        }
        let group = &grouped.groups[0];
        if group.d_weights.len() != grouped.d_rows {
            return Err(AkitaError::InvalidSize {
                expected: grouped.d_rows,
                actual: group.d_weights.len(),
            });
        }
        if group.a_weights.len() != group.n_a {
            return Err(AkitaError::InvalidSize {
                expected: group.n_a,
                actual: group.a_weights.len(),
            });
        }
        if group.b_weights.len() != group.n_b {
            return Err(AkitaError::InvalidSize {
                expected: group.n_b,
                actual: group.b_weights.len(),
            });
        }
        if group.e_col_offset > grouped.d_physical_cols
            || group.e_eq_slice.len() > grouped.d_physical_cols - group.e_col_offset
        {
            return Err(AkitaError::InvalidSetup(
                "grouped D setup weights exceed physical D width".into(),
            ));
        }

        let d_stride = grouped.d_physical_cols;
        let b_stride = group.t_cols;
        let z_range = group.z_cols;
        if group.t_eq_slice.len() != b_stride {
            return Err(AkitaError::InvalidSize {
                expected: b_stride,
                actual: group.t_eq_slice.len(),
            });
        }
        if group.z_eq_slice.len() != z_range {
            return Err(AkitaError::InvalidSize {
                expected: z_range,
                actual: group.z_eq_slice.len(),
            });
        }

        let d_required = checked_mul(grouped.d_rows, d_stride, "D setup footprint")?;
        let b_required = checked_mul(group.n_b, b_stride, "B setup footprint")?;
        let a_required = checked_mul(group.n_a, z_range, "A setup footprint")?;
        let required = d_required.max(b_required).max(a_required);
        if required == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator requires a non-empty packed footprint".into(),
            ));
        }

        let mut e_eq_slice = vec![E::zero(); d_stride];
        let e_start = group.e_col_offset;
        let e_end = e_start + group.e_eq_slice.len();
        e_eq_slice[e_start..e_end].copy_from_slice(&group.e_eq_slice);

        let b_weights_by_row = group
            .b_weights
            .iter()
            .copied()
            .map(|weight| vec![weight])
            .collect();

        let mut endpoints = Vec::with_capacity(grouped.d_rows + group.n_b + group.n_a + 2);
        endpoints.push(0);
        endpoints.push(required);
        push_role_boundaries(&mut endpoints, grouped.d_rows, d_stride, "D")?;
        push_role_boundaries(&mut endpoints, group.n_b, b_stride, "B")?;
        push_role_boundaries(&mut endpoints, group.n_a, z_range, "A")?;
        endpoints.sort_unstable();
        endpoints.dedup();

        Ok(Self {
            required,
            d_stride,
            b_stride,
            z_range,
            d_required,
            b_required,
            a_required,
            e_eq_slice,
            t_eq_slice_per_group: vec![group.t_eq_slice.clone()],
            z_eq_slice: group.z_eq_slice.clone(),
            d_weights: group.d_weights.clone(),
            b_weights_by_row,
            a_weights: group.a_weights.clone(),
            endpoints,
        })
    }

    /// Build the chunk-aware setup-contribution plan.
    ///
    /// The packed-scan footprint (`required`, `d_stride`, `b_stride`, `z_range`)
    /// and α-evaluation count are **independent of the chunk count**: the
    /// chunk-partitioned `e`/`t` columns each map to exactly one chunk (a
    /// partition, so the footprint is unchanged) and the chunk-replicated `z`
    /// enters only through the additively combined `z_eq_slice` (`Z_comb`),
    /// summed over chunks. `num_chunks = 1` reproduces the historical plan
    /// exactly. Multi-chunk (`W > 1`) is supported only for the non-tiered,
    /// single-commitment-bundle core.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<F>(
        inputs: &SetupContributionPlanInputs<E>,
        full_vec_randomness: &[E],
        eq_low: Option<&[E]>,
        z_block_low_eq: Option<&[E]>,
        fold_gadget: &[F],
        chunk_layout: &crate::WitnessLayout,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore,
        E: MulBase<F>,
    {
        if inputs.num_blocks == 0 || !inputs.num_blocks.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "num_blocks must be a non-zero power of two".into(),
            ));
        }
        if inputs.block_len == 0
            || inputs.depth_open == 0
            || inputs.depth_commit == 0
            || inputs.depth_fold == 0
        {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator layout has zero width".into(),
            ));
        }
        if fold_gadget.len() < inputs.depth_fold {
            return Err(AkitaError::InvalidSize {
                expected: inputs.depth_fold,
                actual: fold_gadget.len(),
            });
        }
        if inputs.num_polys_per_group.len() != inputs.num_groups {
            return Err(AkitaError::InvalidSize {
                expected: inputs.num_groups,
                actual: inputs.num_polys_per_group.len(),
            });
        }

        // Chunk geometry: the `e`/`t` block peel is over `blocks_per_chunk`
        // (the single-chunk case has `blocks_per_chunk = num_blocks`).
        let num_chunks = chunk_layout.num_chunks();
        let blocks_per_chunk = chunk_layout.blocks_per_chunk;
        if num_chunks == 0
            || blocks_per_chunk == 0
            || !blocks_per_chunk.is_power_of_two()
            || chunk_layout.chunk_lengths.len() != num_chunks
        {
            return Err(AkitaError::InvalidSetup(
                "malformed witness chunk layout".into(),
            ));
        }
        if checked_mul(num_chunks, blocks_per_chunk, "chunk block coverage")? != inputs.num_blocks {
            return Err(AkitaError::InvalidSetup(
                "witness chunk windows do not tile num_blocks".into(),
            ));
        }
        if num_chunks > 1 && inputs.num_groups != 1 {
            return Err(AkitaError::InvalidSetup(
                "multi-chunk setup contribution requires a single commitment bundle".into(),
            ));
        }

        let block_bits = blocks_per_chunk.trailing_zeros() as usize;
        if block_bits > full_vec_randomness.len() {
            return Err(AkitaError::InvalidSize {
                expected: block_bits,
                actual: full_vec_randomness.len(),
            });
        }
        let block_mask = blocks_per_chunk - 1;
        let high_challenges = &full_vec_randomness[block_bits..];
        let eq_low_storage;
        let eq_low = if let Some(eq_low) = eq_low {
            eq_low
        } else {
            eq_low_storage = EqPolynomial::evals(&full_vec_randomness[..block_bits])?;
            &eq_low_storage
        };
        if eq_low.len() < blocks_per_chunk {
            return Err(AkitaError::InvalidSize {
                expected: blocks_per_chunk,
                actual: eq_low.len(),
            });
        }

        let z_offset_low_bits = inputs.block_len.trailing_zeros() as usize;
        if z_offset_low_bits > full_vec_randomness.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_offset_low_bits,
                actual: full_vec_randomness.len(),
            });
        }
        let z_range = inputs.inner_width;
        let expected_z_range = checked_mul(inputs.block_len, inputs.depth_commit, "Z width")?;
        if z_range != expected_z_range {
            return Err(AkitaError::InvalidSize {
                expected: expected_z_range,
                actual: z_range,
            });
        }

        let n_d_active = match inputs.m_row_layout {
            MRowLayout::WithDBlock => inputs.n_d,
            MRowLayout::WithoutDBlock => 0,
        };
        // Canonical row layout: consistency (1) | A | B | D.
        let a_start = 1usize;
        let b_start = checked_add(a_start, inputs.n_a, "B row start")?;
        let b_rows_total = checked_mul(inputs.n_b, inputs.num_groups, "B row count")?;
        let d_start = checked_add(b_start, b_rows_total, "D row start")?;
        let a_end = checked_add(d_start, n_d_active, "D row end")?;
        let stride_t = checked_mul(inputs.n_a, inputs.depth_open, "T stride")?;
        let cols_per_poly_t = checked_mul(stride_t, inputs.num_blocks, "T polynomial width")?;
        let b_per_claim_e = checked_mul(inputs.num_blocks, inputs.depth_open, "e-hat claim width")?;
        let n_cols_e = checked_mul(inputs.num_claims, b_per_claim_e, "e-hat column width")?;
        let max_group_poly_count = inputs
            .num_polys_per_group
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let n_cols_t = checked_mul(max_group_poly_count, cols_per_poly_t, "T column width")?;
        let d_required = checked_mul(n_d_active, n_cols_e, "D setup footprint")?;
        let a_required = checked_mul(inputs.n_a, z_range, "A setup footprint")?;
        let b_required = checked_mul(inputs.n_b, n_cols_t, "B setup footprint")?;
        let required = d_required.max(b_required).max(a_required);
        if required == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator requires a non-empty packed footprint".into(),
            ));
        }
        if a_end > inputs.rows || inputs.rows > inputs.eq_tau1.len() {
            return Err(AkitaError::InvalidSetup(
                "M-row weights are inconsistent with setup evaluator layout".into(),
            ));
        }

        let mut group_offsets = Vec::with_capacity(inputs.num_polys_per_group.len());
        let mut next_offset = 0usize;
        for &group_poly_count in &inputs.num_polys_per_group {
            group_offsets.push(next_offset);
            next_offset = checked_add(next_offset, group_poly_count, "T vector offset")?;
        }
        if next_offset != inputs.num_t_vectors {
            return Err(AkitaError::InvalidSetup(
                "T vector count is inconsistent with point polynomial counts".into(),
            ));
        }

        let e_eq_slice: Vec<E> = if n_d_active == 0 {
            Vec::new()
        } else {
            let e_hi_len =
                checked_mul(inputs.num_claims, inputs.depth_open, "e-hat high-eq width")?;
            // Per-chunk high-eq tables based at each chunk's `ê` high offset. A
            // single SIS column maps to exactly one chunk (the `e` partition),
            // so the footprint `n_cols_e` is unchanged.
            let eq_hi_e_tables: Vec<Vec<E>> = chunk_layout
                .chunks
                .iter()
                .map(|chunk| {
                    let high_base = chunk.offset_e >> block_bits;
                    (0..=e_hi_len)
                        .map(|k| eq_eval_at_index(high_challenges, high_base + k))
                        .collect::<Vec<E>>()
                })
                .collect();
            cfg_into_iter!(0..n_cols_e)
                .map(|current_index| {
                    let (chunk_idx, low_eq_idx, high_eq_idx) = get_eq_indices_for_d_chunked(
                        current_index,
                        chunk_layout,
                        inputs.depth_open,
                        inputs.num_blocks,
                        inputs.num_claims,
                        b_per_claim_e,
                        block_mask,
                        block_bits,
                    );
                    eq_low[low_eq_idx] * eq_hi_e_tables[chunk_idx][high_eq_idx]
                })
                .collect()
        };

        let t_hi_len = checked_mul(
            checked_mul(inputs.num_t_vectors, inputs.depth_open, "T high-eq width")?,
            inputs.n_a,
            "T high-eq width",
        )?;
        // Per-chunk high-eq tables based at each chunk's `t̂` high offset.
        let eq_hi_t_tables: Vec<Vec<E>> = chunk_layout
            .chunks
            .iter()
            .map(|chunk| {
                let high_base = chunk.offset_t >> block_bits;
                (0..=t_hi_len)
                    .map(|k| eq_eval_at_index(high_challenges, high_base + k))
                    .collect::<Vec<E>>()
            })
            .collect();
        let t_eq_slice_per_group: Vec<Vec<E>> = (0..inputs.num_groups)
            .map(|g| {
                let group_size = inputs.num_polys_per_group[g];
                cfg_into_iter!(0..n_cols_t)
                    .map(|c| {
                        let poly_idx = c / cols_per_poly_t;
                        if poly_idx >= group_size {
                            return E::zero();
                        }
                        let flat_t_vector = group_offsets[g] + poly_idx;
                        let (chunk_idx, low_eq_idx, high_eq_idx) = get_eq_indices_for_b_chunked(
                            c,
                            flat_t_vector,
                            chunk_layout,
                            inputs.depth_open,
                            inputs.n_a,
                            inputs.num_blocks,
                            inputs.num_t_vectors,
                            stride_t,
                            block_mask,
                            block_bits,
                        );
                        eq_low[low_eq_idx] * eq_hi_t_tables[chunk_idx][high_eq_idx]
                    })
                    .collect()
            })
            .collect();

        // Chunk-replicated `A·ẑ`: `Z_comb[c] = Σ_chunk z_weight(c, chunk.offset_z)`.
        // The output length (`z_range = inner_width`) is unchanged, so the
        // downstream packed scan and α-evaluation count are identical to the
        // single-chunk plan; only the precomputed weights are summed over chunks.
        // For `num_chunks = 1` this is the historical single z_eq_slice.
        let mut z_eq_slice = vec![E::zero(); z_range];
        for chunk in &chunk_layout.chunks {
            let per_chunk = build_z_eq_slice_for_offset::<F, E>(
                inputs,
                full_vec_randomness,
                z_block_low_eq,
                fold_gadget,
                chunk.offset_z,
                z_offset_low_bits,
                z_range,
            )?;
            for (dst, src) in z_eq_slice.iter_mut().zip(per_chunk) {
                *dst += src;
            }
        }

        let b_weights_by_row: Vec<Vec<E>> = (0..inputs.n_b)
            .map(|row| {
                (0..inputs.num_groups)
                    .map(|g| inputs.eq_tau1[b_start + g * inputs.n_b + row])
                    .collect()
            })
            .collect();

        let mut endpoints = Vec::with_capacity(n_d_active + inputs.n_b + inputs.n_a + 2);
        endpoints.push(0);
        endpoints.push(required);
        push_role_boundaries(&mut endpoints, n_d_active, n_cols_e, "D")?;
        push_role_boundaries(&mut endpoints, inputs.n_b, n_cols_t, "B")?;
        push_role_boundaries(&mut endpoints, inputs.n_a, z_range, "A")?;
        endpoints.sort_unstable();
        endpoints.dedup();

        Ok(Self {
            required,
            d_stride: n_cols_e,
            b_stride: n_cols_t,
            z_range,
            d_required,
            b_required,
            a_required,
            e_eq_slice,
            t_eq_slice_per_group,
            z_eq_slice,
            d_weights: inputs.eq_tau1[d_start..a_end].to_vec(),
            b_weights_by_row,
            a_weights: inputs.eq_tau1[a_start..(a_start + inputs.n_a)].to_vec(),
            endpoints,
        })
    }
}

impl<E: FieldCore> GroupedSetupContributionPlan<E> {
    pub fn evaluate_direct<F, const D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        if alpha_pows_a.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: alpha_pows_a.len(),
            });
        }

        if alpha_pows_b == alpha_pows_a && alpha_pows_d == alpha_pows_a {
            let setup_len = setup.shared_matrix().total_ring_elements_at::<D>()?;
            let setup_view = setup.shared_matrix().ring_view::<D>(1, setup_len)?;
            let setup_flat = setup_view.as_slice();
            let mut acc = E::zero();
            for group in &self.groups {
                acc += group.evaluate_packed_direct::<F, D>(
                    setup_flat,
                    alpha_pows_a,
                    self.d_rows,
                    self.d_physical_cols,
                )?;
            }
            return Ok(acc);
        }

        self.evaluate_direct_by_rows::<F, D>(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d)
    }

    fn evaluate_direct_by_rows<F, const D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let mut acc = E::zero();
        if self.d_rows != 0 {
            let d_view = setup.shared_matrix.ring_view_dyn(
                self.d_rows,
                self.d_physical_cols,
                alpha_pows_d.len(),
            )?;
            for group in &self.groups {
                for (row_idx, &row_weight) in group.d_weights.iter().enumerate() {
                    if row_weight.is_zero() {
                        continue;
                    }
                    let row = d_view.row_flat(row_idx)?;
                    acc += evaluate_weighted_setup_row::<F, E>(
                        row,
                        group.e_col_offset,
                        &group.e_eq_slice,
                        row_weight,
                        alpha_pows_d,
                    )?;
                }
            }
        }

        for group in &self.groups {
            let a_view = setup
                .shared_matrix
                .ring_view_dyn(group.n_a, group.z_cols, D)?;
            for (row_idx, &row_weight) in group.a_weights.iter().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let row = a_view.row_flat(row_idx)?;
                acc += evaluate_weighted_setup_row::<F, E>(
                    row,
                    0,
                    &group.z_eq_slice,
                    row_weight,
                    alpha_pows_a,
                )?;
            }

            let b_view =
                setup
                    .shared_matrix
                    .ring_view_dyn(group.n_b, group.t_cols, alpha_pows_b.len())?;
            for (row_idx, &row_weight) in group.b_weights.iter().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let row = b_view.row_flat(row_idx)?;
                acc += evaluate_weighted_setup_row::<F, E>(
                    row,
                    0,
                    &group.t_eq_slice,
                    row_weight,
                    alpha_pows_b,
                )?;
            }
        }

        Ok(acc)
    }
}

impl<E: FieldCore> GroupSetupContributionPlan<E> {
    fn evaluate_packed_direct<F, const D: usize>(
        &self,
        setup_flat: &[CyclotomicRing<F, D>],
        alpha_pows: &[E],
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let (required, segments) = self.packed_segments(d_rows, d_physical_cols)?;
        if required > setup_flat.len() {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected grouped verifier layout".into(),
            ));
        }

        let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
            .map(|idx| {
                let segment = &segments[idx];
                macro_rules! segment_sum {
                    ($has_d:literal, $has_b:literal, $has_a:literal) => {
                        packed_group_slice_inner_sum::<F, E, D, $has_d, $has_b, $has_a>(
                            segment.lo..segment.hi,
                            setup_flat,
                            alpha_pows,
                            segment.d_start_abs,
                            segment.d_weight,
                            &self.e_eq_slice,
                            segment.b_start_abs,
                            segment.b_weight,
                            &self.t_eq_slice,
                            segment.a_start_abs,
                            segment.a_weight,
                            &self.z_eq_slice,
                        )
                    };
                }

                match (segment.has_d, segment.has_b, segment.has_a) {
                    (true, true, true) => segment_sum!(true, true, true),
                    (true, true, false) => segment_sum!(true, true, false),
                    (true, false, true) => segment_sum!(true, false, true),
                    (false, true, true) => segment_sum!(false, true, true),
                    (true, false, false) => segment_sum!(true, false, false),
                    (false, true, false) => segment_sum!(false, true, false),
                    (false, false, true) => segment_sum!(false, false, true),
                    (false, false, false) => E::zero(),
                }
            })
            .collect();

        Ok(segment_sums.into_iter().sum())
    }

    fn packed_segments(
        &self,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<(usize, Vec<GroupSetupSegment<E>>), AkitaError> {
        if self.d_weights.len() != d_rows {
            return Err(AkitaError::InvalidSize {
                expected: d_rows,
                actual: self.d_weights.len(),
            });
        }
        if self.a_weights.len() != self.n_a {
            return Err(AkitaError::InvalidSize {
                expected: self.n_a,
                actual: self.a_weights.len(),
            });
        }
        if self.b_weights.len() != self.n_b {
            return Err(AkitaError::InvalidSize {
                expected: self.n_b,
                actual: self.b_weights.len(),
            });
        }
        if self.t_eq_slice.len() != self.t_cols {
            return Err(AkitaError::InvalidSize {
                expected: self.t_cols,
                actual: self.t_eq_slice.len(),
            });
        }
        if self.z_eq_slice.len() != self.z_cols {
            return Err(AkitaError::InvalidSize {
                expected: self.z_cols,
                actual: self.z_eq_slice.len(),
            });
        }
        let e_end = checked_add(
            self.e_col_offset,
            self.e_eq_slice.len(),
            "grouped D setup footprint",
        )?;
        if e_end > d_physical_cols {
            return Err(AkitaError::InvalidSetup(
                "grouped D setup weights exceed physical D width".into(),
            ));
        }

        let d_required = checked_mul(d_rows, d_physical_cols, "grouped D setup footprint")?;
        let b_required = checked_mul(self.n_b, self.t_cols, "grouped B setup footprint")?;
        let a_required = checked_mul(self.n_a, self.z_cols, "grouped A setup footprint")?;
        let required = d_required.max(b_required).max(a_required);

        let mut endpoints = Vec::new();
        endpoints.push(0);
        endpoints.push(required);
        push_group_d_boundaries(
            &mut endpoints,
            d_rows,
            d_physical_cols,
            self.e_col_offset,
            self.e_eq_slice.len(),
        )?;
        push_role_boundaries(&mut endpoints, self.n_b, self.t_cols, "B")?;
        push_role_boundaries(&mut endpoints, self.n_a, self.z_cols, "A")?;
        endpoints.sort_unstable();
        endpoints.dedup();

        let segments = (0..endpoints.len().saturating_sub(1))
            .filter_map(|idx| {
                let lo = endpoints[idx];
                let hi = endpoints[idx + 1];
                if lo == hi {
                    return None;
                }

                let has_d =
                    if d_physical_cols == 0 || self.e_eq_slice.is_empty() || lo >= d_required {
                        false
                    } else {
                        let d_col = lo % d_physical_cols;
                        d_col >= self.e_col_offset && d_col < e_end
                    };
                let d_row = if has_d { lo / d_physical_cols } else { 0 };
                let d_start_abs = if has_d {
                    d_row * d_physical_cols + self.e_col_offset
                } else {
                    0
                };
                let d_weight = if has_d {
                    self.d_weights[d_row]
                } else {
                    E::zero()
                };

                let has_b = self.t_cols != 0 && lo < b_required;
                let b_row = if has_b { lo / self.t_cols } else { 0 };
                let b_start_abs = if has_b { b_row * self.t_cols } else { 0 };
                let b_weight = if has_b {
                    self.b_weights[b_row]
                } else {
                    E::zero()
                };

                let has_a = self.z_cols != 0 && lo < a_required;
                let a_row = if has_a { lo / self.z_cols } else { 0 };
                let a_start_abs = if has_a { a_row * self.z_cols } else { 0 };
                let a_weight = if has_a {
                    self.a_weights[a_row]
                } else {
                    E::zero()
                };

                if !has_d && !has_b && !has_a {
                    return None;
                }

                Some(GroupSetupSegment {
                    lo,
                    hi,
                    has_d,
                    d_start_abs,
                    d_weight,
                    has_b,
                    b_start_abs,
                    b_weight,
                    has_a,
                    a_start_abs,
                    a_weight,
                })
            })
            .collect();

        Ok((required, segments))
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

struct GroupSetupSegment<E> {
    lo: usize,
    hi: usize,
    has_d: bool,
    d_start_abs: usize,
    d_weight: E,
    has_b: bool,
    b_start_abs: usize,
    b_weight: E,
    has_a: bool,
    a_start_abs: usize,
    a_weight: E,
}

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
    e_eq: &[E],
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
                weight += d_weight * e_eq[lambda - d_start];
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

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_group_slice_inner_sum<
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
    e_eq: &[E],
    b_start: usize,
    b_weight: E,
    t_eq: &[E],
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
                weight += d_weight * e_eq[lambda - d_start];
            }
            if HAS_B {
                weight += b_weight * t_eq[lambda - b_start];
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

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn bar_omega_segment_eval<E, const HAS_D: bool, const HAS_B: bool, const HAS_A: bool>(
    range: std::ops::Range<usize>,
    eq_lambda: &[E],
    d_start: usize,
    d_weight: E,
    e_eq: &[E],
    b_start: usize,
    b_weights: &[E],
    t_eq_per_group: &[Vec<E>],
    a_start: usize,
    a_weight: E,
    z_eq: &[E],
) -> E
where
    E: FieldCore,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, lambda| {
            let mut weight = E::zero();
            if HAS_D {
                weight += d_weight * e_eq[lambda - d_start];
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
                acc += eq_lambda[lambda] * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

fn validate_group_chunk_layout(
    group: &SetupContributionGroupInputs,
    num_groups: usize,
) -> Result<(), AkitaError> {
    if group.chunks.is_empty()
        || group.blocks_per_chunk == 0
        || !group.blocks_per_chunk.is_power_of_two()
    {
        return Err(AkitaError::InvalidSetup(
            "malformed grouped witness chunk layout".into(),
        ));
    }
    if checked_mul(
        group.chunks.len(),
        group.blocks_per_chunk,
        "grouped chunk block coverage",
    )? != group.num_blocks
    {
        return Err(AkitaError::InvalidSetup(
            "grouped witness chunk windows do not tile num_blocks".into(),
        ));
    }
    if group.chunks.len() > 1 && num_groups != 1 {
        return Err(AkitaError::InvalidSetup(
            "multi-chunk grouped setup contribution requires exactly one group".into(),
        ));
    }
    Ok(())
}

fn grouped_e_setup_weights<E: FieldCore>(
    group: &SetupContributionGroupInputs,
    full_vec_randomness: &[E],
) -> Result<Vec<E>, AkitaError> {
    let block_bits = group.blocks_per_chunk.trailing_zeros() as usize;
    if block_bits > full_vec_randomness.len() {
        return Err(AkitaError::InvalidSize {
            expected: block_bits,
            actual: full_vec_randomness.len(),
        });
    }
    let eq_low = EqPolynomial::evals(&full_vec_randomness[..block_bits])?;
    let high_challenges = &full_vec_randomness[block_bits..];
    let high_len = checked_mul(group.num_claims, group.depth_open, "grouped D high width")?;
    let eq_high_by_chunk: Vec<Vec<E>> = group
        .chunks
        .iter()
        .map(|chunk| high_eq_window(high_challenges, chunk.offset_e >> block_bits, high_len))
        .collect();
    let low_mask = group.blocks_per_chunk - 1;
    let total_blocks = checked_mul(group.num_claims, group.num_blocks, "grouped D blocks")?;
    let e_cols = checked_mul(total_blocks, group.depth_open, "grouped D columns")?;
    cfg_into_iter!(0..e_cols)
        .map(|local_col| {
            let flat_block = local_col / group.depth_open;
            let digit = local_col % group.depth_open;
            let claim_idx = flat_block / group.num_blocks;
            let global_block_idx = flat_block % group.num_blocks;
            let chunk_idx = global_block_idx / group.blocks_per_chunk;
            let block_idx = global_block_idx % group.blocks_per_chunk;
            let chunk = group
                .chunks
                .get(chunk_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let eq_high = eq_high_by_chunk
                .get(chunk_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let offset_low = chunk.offset_e & low_mask;
            let high_idx = checked_add(
                checked_mul(digit, group.num_claims, "grouped D high digit")?,
                claim_idx,
                "grouped D high claim",
            )?;
            pow2_offset_eq_weight(
                &eq_low, eq_high, offset_low, block_idx, high_idx, low_mask, block_bits,
            )
        })
        .collect()
}

fn grouped_t_setup_weights<E: FieldCore>(
    group: &SetupContributionGroupInputs,
    full_vec_randomness: &[E],
) -> Result<Vec<E>, AkitaError> {
    let block_bits = group.blocks_per_chunk.trailing_zeros() as usize;
    if block_bits > full_vec_randomness.len() {
        return Err(AkitaError::InvalidSize {
            expected: block_bits,
            actual: full_vec_randomness.len(),
        });
    }
    let eq_low = EqPolynomial::evals(&full_vec_randomness[..block_bits])?;
    let high_challenges = &full_vec_randomness[block_bits..];
    let high_len = checked_mul(
        checked_mul(group.num_claims, group.depth_open, "grouped B high width")?,
        group.n_a,
        "grouped B high width",
    )?;
    let eq_high_by_chunk: Vec<Vec<E>> = group
        .chunks
        .iter()
        .map(|chunk| high_eq_window(high_challenges, chunk.offset_t >> block_bits, high_len))
        .collect();
    let low_mask = group.blocks_per_chunk - 1;
    let t_compound_per_block =
        checked_mul(group.n_a, group.depth_open, "grouped B compound stride")?;
    let t_cols = checked_mul(group.num_claims, group.t_cols_per_vector, "grouped B width")?;
    cfg_into_iter!(0..t_cols)
        .map(|local_col| {
            let t_vector_idx = local_col / group.t_cols_per_vector;
            let phys_claim_offset = local_col % group.t_cols_per_vector;
            let global_block_idx = phys_claim_offset / t_compound_per_block;
            let chunk_idx = global_block_idx / group.blocks_per_chunk;
            let block_idx = global_block_idx % group.blocks_per_chunk;
            let chunk = group
                .chunks
                .get(chunk_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let eq_high = eq_high_by_chunk
                .get(chunk_idx)
                .ok_or(AkitaError::InvalidProof)?;
            let offset_low = chunk.offset_t & low_mask;
            let compound = phys_claim_offset % t_compound_per_block;
            let high_idx = checked_add(
                checked_mul(compound, group.num_claims, "grouped B high compound")?,
                t_vector_idx,
                "grouped B high vector",
            )?;
            pow2_offset_eq_weight(
                &eq_low, eq_high, offset_low, block_idx, high_idx, low_mask, block_bits,
            )
        })
        .collect()
}

fn grouped_z_setup_weights<F, E>(
    group: &SetupContributionGroupInputs,
    full_vec_randomness: &[E],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: MulBase<F>,
{
    let fold_gadget = crate::gadget_row_scalars::<F>(group.depth_fold, group.log_basis);
    let z_range = checked_mul(group.block_len, group.depth_commit, "grouped Z range")?;
    let mut z_eq_slice = vec![E::zero(); z_range];
    for chunk in &group.chunks {
        let per_chunk = grouped_z_setup_weights_for_offset::<F, E>(
            group,
            full_vec_randomness,
            &fold_gadget,
            chunk.offset_z,
            z_range,
        )?;
        for (dst, src) in z_eq_slice.iter_mut().zip(per_chunk) {
            *dst += src;
        }
    }
    Ok(z_eq_slice)
}

fn grouped_z_setup_weights_for_offset<F, E>(
    group: &SetupContributionGroupInputs,
    full_vec_randomness: &[E],
    fold_gadget: &[F],
    offset_z: usize,
    z_range: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: MulBase<F>,
{
    if group.block_len.is_power_of_two() {
        let z_bits = group.block_len.trailing_zeros() as usize;
        if z_bits > full_vec_randomness.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_bits,
                actual: full_vec_randomness.len(),
            });
        }
        let eq_low = EqPolynomial::evals(&full_vec_randomness[..z_bits])?;
        let high_challenges = &full_vec_randomness[z_bits..];
        let high_len = checked_mul(group.depth_commit, group.depth_fold, "grouped Z high width")?;
        let eq_high = high_eq_window(high_challenges, offset_z >> z_bits, high_len);
        let low_mask = group.block_len - 1;
        let offset_low = offset_z & low_mask;
        cfg_into_iter!(0..z_range)
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
            .min(full_vec_randomness.len());
        let low_mask = 1usize
            .checked_shl(
                u32::try_from(low_bits).map_err(|_| AkitaError::InvalidSize {
                    expected: usize::BITS as usize,
                    actual: low_bits,
                })?,
            )
            .ok_or_else(|| AkitaError::InvalidSetup("grouped dense Z eq width overflow".into()))?
            - 1;
        let eq_low = EqPolynomial::evals(&full_vec_randomness[..low_bits])?;
        let offset_low = offset_z & low_mask;
        let offset_high = offset_z >> low_bits;
        let max_high = checked_add(offset_z, z_len, "grouped dense Z end")?
            .checked_sub(1)
            .ok_or(AkitaError::InvalidProof)?
            >> low_bits;
        let eq_high: Vec<E> = (offset_high..=max_high)
            .map(|idx| eq_eval_at_index(&full_vec_randomness[low_bits..], idx))
            .collect();
        cfg_into_iter!(0..z_range)
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

fn evaluate_weighted_setup_row<Base, E>(
    row: &[Base],
    col_offset: usize,
    col_weights: &[E],
    row_weight: E,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    Base: FieldCore,
    E: ExtField<Base>,
{
    let ring_d = alpha_pows.len();
    let mut acc = E::zero();
    for (col, &col_weight) in col_weights.iter().enumerate() {
        if col_weight.is_zero() {
            continue;
        }
        let setup_col = checked_add(col_offset, col, "weighted setup column")?;
        let coeff_start = checked_mul(setup_col, ring_d, "weighted setup coeff start")?;
        let coeffs = checked_slice(row, coeff_start, ring_d, "weighted setup coeffs")?;
        acc += row_weight * col_weight * eval_flat_ring_at_pows::<Base, E>(coeffs, alpha_pows);
    }
    Ok(acc)
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

/// Chunk-aware `D·ê` column → `(chunk_idx, low_eq_idx, high_eq_idx)` mapping.
///
/// Decodes the SIS column to its logical `(digit, global_block, claim)` as
/// before, then routes the global block to its chunk; `high_eq_idx` is relative
/// to that chunk's `ê` high offset (the per-chunk high table base). A SIS column
/// maps to exactly one chunk, so the footprint is unchanged. `num_chunks = 1`
/// (`blocks_per_chunk = num_blocks`) reproduces the single-chunk mapping.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_eq_indices_for_d_chunked(
    current_index: usize,
    layout: &crate::WitnessLayout,
    num_digits: usize,
    num_blocks: usize,
    num_claims: usize,
    blocks_per_claim_e: usize,
    block_mask: usize,
    block_bits: usize,
) -> (usize, usize, usize) {
    let digit_idx = current_index % num_digits;
    let blk_g = (current_index / num_digits) % num_blocks;
    let claim_idx = current_index / blocks_per_claim_e;
    let chunk_idx = blk_g / layout.blocks_per_chunk;
    let block_local = blk_g % layout.blocks_per_chunk;
    let offset_e = layout.chunks[chunk_idx].offset_e;
    let m_layout_high_idx = digit_idx * num_claims + claim_idx;
    let block_sum = (offset_e & block_mask) + block_local;
    let low_eq_idx = block_sum & block_mask;
    let block_carry = block_sum >> block_bits;
    let high_eq_idx = m_layout_high_idx + block_carry;
    (chunk_idx, low_eq_idx, high_eq_idx)
}

/// Chunk-aware `B·t̂` column → `(chunk_idx, low_eq_idx, high_eq_idx)` mapping.
/// Same chunk routing as [`get_eq_indices_for_d_chunked`], with the extra
/// `a_row` / `flat_t_vector` high axes preserved.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_eq_indices_for_b_chunked(
    current_index: usize,
    flat_t_vector: usize,
    layout: &crate::WitnessLayout,
    num_digits: usize,
    n_a: usize,
    num_blocks: usize,
    num_t_vectors: usize,
    stride_t: usize,
    block_mask: usize,
    block_bits: usize,
) -> (usize, usize, usize) {
    let digit_idx = current_index % num_digits;
    let a_row_idx = (current_index / num_digits) % n_a;
    let blk_g = (current_index / stride_t) % num_blocks;
    let chunk_idx = blk_g / layout.blocks_per_chunk;
    let block_local = blk_g % layout.blocks_per_chunk;
    let offset_t = layout.chunks[chunk_idx].offset_t;
    let m_layout_high_idx =
        flat_t_vector + num_t_vectors * digit_idx + num_t_vectors * num_digits * a_row_idx;
    let block_sum = (offset_t & block_mask) + block_local;
    let low_eq_idx = block_sum & block_mask;
    let block_carry = block_sum >> block_bits;
    let high_eq_idx = m_layout_high_idx + block_carry;
    (chunk_idx, low_eq_idx, high_eq_idx)
}

/// Build the `A·ẑ` column-equality weights for one chunk's replicated `ẑ`
/// placed at `offset_z`. Returns a length-`z_range` (`= inner_width`) vector;
/// the caller sums these over chunks into `Z_comb`. This is the per-offset
/// extract of the historical single-chunk `z_eq_slice` build (pow2 + dense
/// fallback), with `offset_z` as the only chunk-varying input.
fn build_z_eq_slice_for_offset<F, E>(
    inputs: &SetupContributionPlanInputs<E>,
    full_vec_randomness: &[E],
    z_block_low_eq: Option<&[E]>,
    fold_gadget: &[F],
    offset_z: usize,
    z_offset_low_bits: usize,
    z_range: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: MulBase<F>,
{
    if inputs.block_len.is_power_of_two() {
        let z_offset_low = offset_z & inputs.block_len.saturating_sub(1);
        let z_block_low_storage;
        let z_block_low_eq = if let Some(z_block_low_eq) = z_block_low_eq {
            z_block_low_eq
        } else {
            z_block_low_storage = EqPolynomial::evals(&full_vec_randomness[..z_offset_low_bits])?;
            &z_block_low_storage
        };
        if z_block_low_eq.len() < inputs.block_len {
            return Err(AkitaError::InvalidSize {
                expected: inputs.block_len,
                actual: z_block_low_eq.len(),
            });
        }

        let z_offset_high = offset_z >> z_offset_low_bits;
        let z_block_mask = inputs.block_len.wrapping_sub(1);
        let z_high_challenges = &full_vec_randomness[z_offset_low_bits..];
        let num_q_z = checked_mul(
            checked_mul(inputs.num_groups, inputs.depth_fold, "Z high-eq width")?,
            inputs.depth_commit,
            "Z high-eq width",
        )?;
        let eq_hi_z_table: Vec<E> = (0..=num_q_z)
            .map(|k| eq_eval_at_index(z_high_challenges, z_offset_high + k))
            .collect();
        let s_per_dc_per_carry: Vec<[E; POSSIBLE_CARRIES]> = (0..inputs.depth_commit)
            .map(|dc| {
                let mut s = [E::zero(); POSSIBLE_CARRIES];
                for (carry_slot, slot) in s.iter_mut().enumerate() {
                    let mut acc = E::zero();
                    for (df, &fg) in fold_gadget.iter().enumerate().take(inputs.depth_fold) {
                        for pt in 0..inputs.num_groups {
                            let k = pt
                                + inputs.num_groups * df
                                + inputs.num_groups * inputs.depth_fold * dc
                                + carry_slot;
                            acc += eq_hi_z_table[k].mul_base(fg);
                        }
                    }
                    *slot = -acc;
                }
                s
            })
            .collect();
        Ok(cfg_into_iter!(0..z_range)
            .map(|c| {
                let (low_eq_idx, depth_commit_idx, block_carry) = get_eq_indices_for_a(
                    c,
                    inputs.depth_commit,
                    z_offset_low,
                    z_block_mask,
                    z_offset_low_bits,
                );
                z_block_low_eq[low_eq_idx] * s_per_dc_per_carry[depth_commit_idx][block_carry]
            })
            .collect())
    } else {
        let z_total_blocks_dense =
            checked_mul(inputs.block_len, inputs.num_groups, "dense Z block width")?;
        let z_len_dense = checked_mul(
            checked_mul(inputs.depth_fold, inputs.depth_commit, "dense Z length")?,
            z_total_blocks_dense,
            "dense Z length",
        )?;
        let n_rand = full_vec_randomness.len();
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
        let offset_z_dense_low = offset_z & mask;
        let offset_z_dense_high = offset_z >> k;
        let eq_low_z_dense = EqPolynomial::evals(&full_vec_randomness[..k])?;
        let max_high = offset_z
            .checked_add(z_len_dense)
            .and_then(|end| end.checked_sub(1))
            .ok_or_else(|| AkitaError::InvalidSetup("dense Z high-eq bound overflow".into()))?
            >> k;
        let n_high = max_high - offset_z_dense_high + 1;
        let eq_high_z_dense: Vec<E> = (0..n_high)
            .map(|h| eq_eval_at_index(&full_vec_randomness[k..], offset_z_dense_high + h))
            .collect();

        Ok(cfg_into_iter!(0..z_range)
            .map(|c| {
                let dc = c % inputs.depth_commit;
                let blk = c / inputs.depth_commit;
                let mut acc = E::zero();
                for pt in 0..inputs.num_groups {
                    for (df, &fg) in fold_gadget.iter().enumerate().take(inputs.depth_fold) {
                        let x = blk
                            + inputs.block_len * pt
                            + inputs.block_len * inputs.num_groups * df
                            + inputs.block_len * inputs.num_groups * inputs.depth_fold * dc;
                        let sum = offset_z_dense_low + x;
                        let low_idx = sum & mask;
                        let high_carry = sum >> k;
                        let eq_val = eq_low_z_dense[low_idx] * eq_high_z_dense[high_carry];
                        acc += eq_val.mul_base(fg);
                    }
                }
                -acc
            })
            .collect())
    }
}

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
fn push_group_d_boundaries(
    endpoints: &mut Vec<usize>,
    rows: usize,
    stride: usize,
    active_col_start: usize,
    active_cols: usize,
) -> Result<(), AkitaError> {
    if rows == 0 || stride == 0 {
        return Ok(());
    }
    let active_col_end = checked_add(active_col_start, active_cols, "grouped D active columns")?;
    let mut row_start = 0usize;
    for _ in 0..rows {
        let row_end = checked_add(row_start, stride, "packed D boundary")?;
        endpoints.push(row_end);
        if active_cols != 0 {
            endpoints.push(checked_add(
                row_start,
                active_col_start,
                "grouped D active boundary",
            )?);
            endpoints.push(checked_add(
                row_start,
                active_col_end,
                "grouped D active boundary",
            )?);
        }
        row_start = row_end;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gadget_row_scalars, AkitaSetupSeed, FlatMatrix, MRowLayout};
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    fn test_scalar(value: u128) -> F {
        F::from_canonical_u128(value)
    }

    #[test]
    fn dense_z_eq_slice_uses_relative_high_carry() {
        let block_len = 12;
        let depth_commit = 3;
        let depth_fold = 2;
        let num_points = 1;
        let z_range = block_len * depth_commit;
        let offset_z = 0;
        let full_vec_randomness = (0..9)
            .map(|idx| test_scalar(101 + idx as u128))
            .collect::<Vec<_>>();
        let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
        let inputs = SetupContributionPlanInputs {
            eq_tau1: vec![test_scalar(11), test_scalar(12)],
            num_t_vectors: 0,
            num_blocks: 4,
            num_claims: 1,
            depth_open: 16,
            depth_commit,
            depth_fold,
            block_len,
            inner_width: z_range,
            n_a: 1,
            n_d: 0,
            m_row_layout: MRowLayout::WithoutDBlock,
            n_b: 0,
            num_groups: num_points,
            rows: 2,
            num_polys_per_group: vec![0],
        };

        let chunk_layout = crate::WitnessLayout {
            blocks_per_chunk: 4,
            chunks: vec![crate::WitnessChunkLayout {
                offset_z,
                offset_e: 0,
                offset_t: 64,
                offset_u: None,
                offset_r: Some(0),
                global_block_base: 0,
            }],
            chunk_lengths: vec![crate::WitnessChunkLengths {
                z_len: z_range,
                e_len: 0,
                t_len: 0,
                u_len: None,
                r_len: Some(0),
            }],
        };
        let plan = SetupContributionPlan::prepare::<F>(
            &inputs,
            &full_vec_randomness,
            None,
            None,
            &fold_gadget,
            &chunk_layout,
        )
        .unwrap();

        let expected = (0..z_range)
            .map(|c| {
                let dc = c % depth_commit;
                let blk = c / depth_commit;
                let mut acc = F::zero();
                for pt in 0..num_points {
                    for (df, &fg) in fold_gadget.iter().enumerate().take(depth_fold) {
                        let x = blk
                            + block_len * pt
                            + block_len * num_points * df
                            + block_len * num_points * depth_fold * dc;
                        acc += eq_eval_at_index(&full_vec_randomness, offset_z + x) * fg;
                    }
                }
                -acc
            })
            .collect::<Vec<_>>();

        assert_eq!(plan.z_eq_slice, expected);
    }

    #[test]
    fn grouped_single_group_supports_multi_chunk_weights() {
        let num_blocks = 4;
        let blocks_per_chunk = 2;
        let num_claims = 3;
        let depth_open = 2;
        let depth_commit = 2;
        let depth_fold = 2;
        let block_len = 4;
        let n_a = 2;
        let n_b = 2;
        let n_d = 1;
        let log_basis = 4;
        let z_range = block_len * depth_commit;
        let e_len_per_chunk = num_claims * depth_open * blocks_per_chunk;
        let t_len_per_chunk = n_a * num_claims * depth_open * blocks_per_chunk;
        let chunk_stride = z_range + e_len_per_chunk + t_len_per_chunk;
        let chunks = (0..2)
            .map(|idx| {
                let base = idx * chunk_stride;
                let offset_e = base + z_range;
                let offset_t = offset_e + e_len_per_chunk;
                crate::WitnessChunkLayout {
                    offset_z: base,
                    offset_e,
                    offset_t,
                    offset_u: None,
                    offset_r: (idx == 1).then_some(offset_t + t_len_per_chunk),
                    global_block_base: idx * blocks_per_chunk,
                }
            })
            .collect::<Vec<_>>();
        let chunk_lengths = (0..2)
            .map(|idx| crate::WitnessChunkLengths {
                z_len: z_range,
                e_len: e_len_per_chunk,
                t_len: t_len_per_chunk,
                u_len: None,
                r_len: (idx == 1).then_some(0),
            })
            .collect::<Vec<_>>();
        let chunk_layout = crate::WitnessLayout {
            blocks_per_chunk,
            chunks: chunks.clone(),
            chunk_lengths,
        };
        let rows = 1 + n_a + n_b + n_d;
        let inputs = SetupContributionPlanInputs {
            eq_tau1: (0..rows.next_power_of_two())
                .map(|idx| test_scalar(11 + idx as u128))
                .collect(),
            num_t_vectors: num_claims,
            num_blocks,
            num_claims,
            depth_open,
            depth_commit,
            depth_fold,
            block_len,
            inner_width: z_range,
            n_a,
            n_d,
            m_row_layout: MRowLayout::WithDBlock,
            n_b,
            num_groups: 1,
            rows,
            num_polys_per_group: vec![num_claims],
        };
        let full_vec_randomness = (0..10)
            .map(|idx| test_scalar(101 + idx as u128))
            .collect::<Vec<_>>();
        let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
        let flat_plan = SetupContributionPlan::prepare::<F>(
            &inputs,
            &full_vec_randomness,
            None,
            None,
            &fold_gadget,
            &chunk_layout,
        )
        .unwrap();

        let grouped_plan = SetupContributionPlan::prepare_grouped::<F>(
            &inputs,
            &full_vec_randomness,
            &[SetupContributionGroupInputs {
                e_col_offset: 0,
                num_claims,
                num_blocks,
                block_len,
                depth_open,
                depth_commit,
                depth_fold,
                log_basis,
                n_a,
                n_b,
                t_cols_per_vector: n_a * depth_open * num_blocks,
                a_row_start: 1,
                b_row_start: 1 + n_a,
                blocks_per_chunk,
                chunks,
            }],
            rows - n_d,
            n_d,
            num_claims * num_blocks * depth_open,
        )
        .unwrap();
        let group = grouped_plan.groups.first().unwrap();

        assert_eq!(group.e_eq_slice, flat_plan.e_eq_slice);
        assert_eq!(group.t_eq_slice, flat_plan.t_eq_slice_per_group[0]);
        assert_eq!(group.z_eq_slice, flat_plan.z_eq_slice);

        let converted = SetupContributionPlan::from_single_grouped(&grouped_plan).unwrap();
        assert_eq!(converted.required, flat_plan.required);
        assert_eq!(converted.d_stride, flat_plan.d_stride);
        assert_eq!(converted.b_stride, flat_plan.b_stride);
        assert_eq!(converted.z_range, flat_plan.z_range);
        assert_eq!(converted.d_weights, flat_plan.d_weights);
        assert_eq!(converted.b_weights_by_row, flat_plan.b_weights_by_row);
        assert_eq!(converted.a_weights, flat_plan.a_weights);

        let setup_len = converted.required();
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 0,
                max_num_batched_polys: 0,
                gen_ring_dim: 1,
                max_setup_len: setup_len,
                public_matrix_seed: [0u8; 32],
            },
            FlatMatrix::from_flat_data(
                (0..setup_len)
                    .map(|idx| test_scalar(211 + idx as u128))
                    .collect(),
                1,
            ),
        );
        let alpha_pows = [test_scalar(3)];
        let expected = flat_plan
            .evaluate_prepared_direct::<F, 1>(&setup, &alpha_pows)
            .unwrap();
        let got_grouped = grouped_plan
            .evaluate_direct::<F, 1>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
            .unwrap();
        assert_eq!(got_grouped, expected);
    }

    #[test]
    fn grouped_packed_direct_matches_row_fallback_with_d_offset() {
        let grouped_plan = GroupedSetupContributionPlan {
            d_rows: 2,
            d_physical_cols: 5,
            groups: vec![GroupSetupContributionPlan {
                e_col_offset: 2,
                t_cols: 4,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                e_eq_slice: vec![test_scalar(2), test_scalar(3)],
                t_eq_slice: vec![
                    test_scalar(5),
                    test_scalar(7),
                    test_scalar(11),
                    test_scalar(13),
                ],
                z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
                a_weights: vec![test_scalar(29), test_scalar(31)],
                b_weights: vec![test_scalar(37), test_scalar(41)],
                d_weights: vec![test_scalar(43), test_scalar(47)],
            }],
        };
        let setup_len = 10;
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 0,
                max_num_batched_polys: 0,
                gen_ring_dim: 1,
                max_setup_len: setup_len,
                public_matrix_seed: [0u8; 32],
            },
            FlatMatrix::from_flat_data(
                (0..setup_len)
                    .map(|idx| test_scalar(211 + idx as u128))
                    .collect(),
                1,
            ),
        );
        let alpha_pows = [test_scalar(3)];
        let expected = grouped_plan
            .evaluate_direct_by_rows::<F, 1>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
            .unwrap();
        let got = grouped_plan
            .evaluate_direct::<F, 1>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
            .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn from_level_params_rejects_non_pow2_num_blocks() {
        let mut lp = LevelParams::log_basis_stub(3);
        lp.ring_dimension = 64;
        lp.role_dims = crate::CommitmentRingDims::uniform(64);
        lp.num_blocks = 3;
        lp.block_len = 8;
        lp.num_digits_commit = 2;
        lp.num_digits_open = 3;
        assert!(SetupContributionPlanInputs::<F>::from_level_params(
            &lp,
            &[2],
            MRowLayout::WithoutDBlock,
            2,
        )
        .is_err());
    }
}
