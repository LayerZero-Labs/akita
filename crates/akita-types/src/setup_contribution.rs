//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns the pure layout/weight derivation for the stage-3 setup
//! product. The prover consumes the materialized `bar_omega` vector, while the
//! verifier can evaluate the same plan directly against the packed setup.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{eq_eval_at_index, high_eq_window};
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, MulBase};

use crate::layout::{LevelParams, MRowLayout};
const POSSIBLE_CARRIES: usize = 2;

#[path = "setup_contribution_grouped.rs"]
mod setup_contribution_grouped;

pub use setup_contribution_grouped::{
    GroupedSetupContributionPlan, GroupedSetupContributionStatic, SetupContributionGroupInputs,
};

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

    /// Canonical setup-contribution weight for shared-vector entry `lambda`
    /// within `segment`: the multiplier the verifier applies to the shared-setup
    /// ring element at `lambda` before summing.
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

    /// Build the chunk-aware setup-contribution plan.
    ///
    /// The packed-scan footprint (`required`, `d_stride`, `b_stride`, `z_range`)
    /// and α-evaluation count are **independent of the chunk count**: the
    /// chunk-partitioned `e`/`t` columns each map to exactly one chunk (a
    /// partition, so the footprint is unchanged) and the chunk-replicated `z`
    /// enters only through the additively combined `z_eq_slice` (`Z_comb`),
    /// summed over chunks. `num_chunks = 1` reproduces the historical plan
    /// exactly. Multi-chunk (`W > 1`) requires a single commitment bundle.
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
            setup_e_col_weights::<E>(
                &chunk_layout.chunks,
                blocks_per_chunk,
                inputs.num_blocks,
                inputs.num_claims,
                inputs.depth_open,
                full_vec_randomness,
                Some(eq_low),
            )?
        };

        let t_eq_slice_per_group: Vec<Vec<E>> = (0..inputs.num_groups)
            .map(|g| {
                setup_t_col_weights::<E>(
                    &chunk_layout.chunks,
                    blocks_per_chunk,
                    inputs.depth_open,
                    inputs.n_a,
                    cols_per_poly_t,
                    max_group_poly_count,
                    inputs.num_polys_per_group[g],
                    group_offsets[g],
                    inputs.num_t_vectors,
                    full_vec_randomness,
                    Some(eq_low),
                )
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;

        // Chunk-replicated `A·ẑ`: `Z_comb[c] = Σ_chunk z_weight(c, chunk.offset_z)`.
        // The output length (`z_range = inner_width`) is unchanged, so the
        // downstream packed scan and α-evaluation count are identical to the
        // single-chunk plan; only the precomputed weights are summed over chunks.
        // For `num_chunks = 1` this is the historical single z_eq_slice.
        let mut z_eq_slice = vec![E::zero(); z_range];
        for chunk in &chunk_layout.chunks {
            let per_chunk = setup_z_col_weights_for_offset::<F, E>(
                inputs.block_len,
                inputs.depth_commit,
                inputs.depth_fold,
                inputs.num_groups,
                full_vec_randomness,
                z_block_low_eq,
                fold_gadget,
                chunk.offset_z,
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

/// Canonical `D·ê` column eq-weights, shared by the flat and grouped setup
/// builders. Produces `num_claims * num_blocks * depth_open` weights, with a
/// per-chunk high-eq window based at each chunk's `ê` offset. The `ê` block is
/// setup-shared across commitment groups, so callers pass the same scalars
/// regardless of group layout.
pub(crate) fn setup_e_col_weights<E: FieldCore>(
    chunks: &[crate::WitnessChunkLayout],
    blocks_per_chunk: usize,
    num_blocks: usize,
    num_claims: usize,
    depth_open: usize,
    full_vec_randomness: &[E],
    eq_low: Option<&[E]>,
) -> Result<Vec<E>, AkitaError> {
    let block_bits = blocks_per_chunk.trailing_zeros() as usize;
    if block_bits > full_vec_randomness.len() {
        return Err(AkitaError::InvalidSize {
            expected: block_bits,
            actual: full_vec_randomness.len(),
        });
    }
    let eq_low_storage;
    let eq_low = if let Some(precomputed) = eq_low {
        precomputed
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
    let high_challenges = &full_vec_randomness[block_bits..];
    let high_len = checked_mul(num_claims, depth_open, "D high width")?;
    let eq_high_by_chunk: Vec<Vec<E>> = chunks
        .iter()
        .map(|chunk| high_eq_window(high_challenges, chunk.offset_e >> block_bits, high_len))
        .collect();
    let low_mask = blocks_per_chunk - 1;
    let total_blocks = checked_mul(num_claims, num_blocks, "D blocks")?;
    let e_cols = checked_mul(total_blocks, depth_open, "D columns")?;
    Ok(cfg_into_iter!(0..e_cols)
        .map(|local_col| {
            let flat_block = local_col / depth_open;
            let digit = local_col % depth_open;
            let claim_idx = flat_block / num_blocks;
            let global_block_idx = flat_block % num_blocks;
            let chunk_idx = global_block_idx / blocks_per_chunk;
            let block_idx = global_block_idx % blocks_per_chunk;
            let chunk = &chunks[chunk_idx];
            let eq_high = &eq_high_by_chunk[chunk_idx];
            let offset_low = chunk.offset_e & low_mask;
            let shifted = offset_low + block_idx;
            let low_idx = shifted & low_mask;
            let carry = shifted >> block_bits;
            let high_idx = digit * num_claims + claim_idx + carry;
            eq_low[low_idx] * eq_high[high_idx]
        })
        .collect())
}

/// Canonical `B·t̂` column eq-weights, shared by the flat and grouped setup
/// builders. Emits `num_vectors * cols_per_vector` weights; columns for a
/// t-vector index `>= active_vectors` are zero (flat zero-padding to the widest
/// group). The high-eq axis is addressed as `(vector_base + vector_idx)` with
/// stride `high_vector_stride`: the flat builder packs all groups' t-vectors
/// into one high axis (`vector_base = group offset`, stride = total t-vectors),
/// while the grouped builder is per-group (`vector_base = 0`, stride = the
/// group's t-vector count).
#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_t_col_weights<E: FieldCore>(
    chunks: &[crate::WitnessChunkLayout],
    blocks_per_chunk: usize,
    depth_open: usize,
    n_a: usize,
    cols_per_vector: usize,
    num_vectors: usize,
    active_vectors: usize,
    vector_base: usize,
    high_vector_stride: usize,
    full_vec_randomness: &[E],
    eq_low: Option<&[E]>,
) -> Result<Vec<E>, AkitaError> {
    let block_bits = blocks_per_chunk.trailing_zeros() as usize;
    if block_bits > full_vec_randomness.len() {
        return Err(AkitaError::InvalidSize {
            expected: block_bits,
            actual: full_vec_randomness.len(),
        });
    }
    let eq_low_storage;
    let eq_low = if let Some(precomputed) = eq_low {
        precomputed
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
    let high_challenges = &full_vec_randomness[block_bits..];
    let high_len = checked_mul(
        checked_mul(high_vector_stride, depth_open, "B high width")?,
        n_a,
        "B high width",
    )?;
    let eq_high_by_chunk: Vec<Vec<E>> = chunks
        .iter()
        .map(|chunk| high_eq_window(high_challenges, chunk.offset_t >> block_bits, high_len))
        .collect();
    let low_mask = blocks_per_chunk - 1;
    let t_compound_per_block = checked_mul(n_a, depth_open, "B compound stride")?;
    let t_cols = checked_mul(num_vectors, cols_per_vector, "B width")?;
    Ok(cfg_into_iter!(0..t_cols)
        .map(|local_col| {
            let vector_idx = local_col / cols_per_vector;
            if vector_idx >= active_vectors {
                return E::zero();
            }
            let phys_claim_offset = local_col % cols_per_vector;
            let global_block_idx = phys_claim_offset / t_compound_per_block;
            let chunk_idx = global_block_idx / blocks_per_chunk;
            let block_idx = global_block_idx % blocks_per_chunk;
            let chunk = &chunks[chunk_idx];
            let eq_high = &eq_high_by_chunk[chunk_idx];
            let offset_low = chunk.offset_t & low_mask;
            let compound = phys_claim_offset % t_compound_per_block;
            let shifted = offset_low + block_idx;
            let low_idx = shifted & low_mask;
            let carry = shifted >> block_bits;
            let high_idx = compound * high_vector_stride + (vector_base + vector_idx) + carry;
            eq_low[low_idx] * eq_high[high_idx]
        })
        .collect())
}

/// Canonical `A·ẑ` column eq-weights for one chunk's replicated `ẑ` placed at
/// `offset_z`, shared by the flat and grouped setup builders. Callers sum the
/// result over chunks into `Z_comb`. `num_fold_groups` is the number of
/// point-blocks folded into a single `ẑ` slice: the flat builder folds all
/// commitment groups (`num_fold_groups = num_groups`), the grouped builder is
/// per-group (`num_fold_groups = 1`). Handles both the power-of-two `block_len`
/// fast path and the dense fallback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_z_col_weights_for_offset<F, E>(
    block_len: usize,
    depth_commit: usize,
    depth_fold: usize,
    num_fold_groups: usize,
    full_vec_randomness: &[E],
    z_block_low_eq: Option<&[E]>,
    fold_gadget: &[F],
    offset_z: usize,
    z_range: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: MulBase<F>,
{
    if block_len.is_power_of_two() {
        let z_bits = block_len.trailing_zeros() as usize;
        if z_bits > full_vec_randomness.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_bits,
                actual: full_vec_randomness.len(),
            });
        }
        let eq_low_storage;
        let eq_low = if let Some(precomputed) = z_block_low_eq {
            precomputed
        } else {
            eq_low_storage = EqPolynomial::evals(&full_vec_randomness[..z_bits])?;
            &eq_low_storage
        };
        if eq_low.len() < block_len {
            return Err(AkitaError::InvalidSize {
                expected: block_len,
                actual: eq_low.len(),
            });
        }
        let high_challenges = &full_vec_randomness[z_bits..];
        let high_len = checked_mul(
            checked_mul(depth_commit, depth_fold, "Z high width")?,
            num_fold_groups,
            "Z high width",
        )?;
        let eq_high = high_eq_window(high_challenges, offset_z >> z_bits, high_len);
        let low_mask = block_len - 1;
        let offset_low = offset_z & low_mask;
        let s_per_dc_per_carry: Vec<[E; POSSIBLE_CARRIES]> = (0..depth_commit)
            .map(|dc| {
                let mut s = [E::zero(); POSSIBLE_CARRIES];
                for (carry_slot, slot) in s.iter_mut().enumerate() {
                    let mut acc = E::zero();
                    for (df, &fold) in fold_gadget.iter().enumerate().take(depth_fold) {
                        for pt in 0..num_fold_groups {
                            let high_idx = pt
                                + num_fold_groups * df
                                + num_fold_groups * depth_fold * dc
                                + carry_slot;
                            acc += eq_high[high_idx].mul_base(fold);
                        }
                    }
                    *slot = -acc;
                }
                s
            })
            .collect();
        Ok(cfg_into_iter!(0..z_range)
            .map(|k| {
                let block_idx = k / depth_commit;
                let dc = k % depth_commit;
                let shifted = offset_low + block_idx;
                let low_idx = shifted & low_mask;
                let carry = shifted >> z_bits;
                let low = eq_low[low_idx];
                let high = s_per_dc_per_carry[dc][carry];
                low * high
            })
            .collect())
    } else {
        let z_len = checked_mul(
            checked_mul(depth_fold, depth_commit, "dense Z length")?,
            checked_mul(block_len, num_fold_groups, "dense Z block width")?,
            "dense Z length",
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
            .ok_or_else(|| AkitaError::InvalidSetup("dense Z eq width overflow".into()))?
            - 1;
        let eq_low = EqPolynomial::evals(&full_vec_randomness[..low_bits])?;
        let offset_low = offset_z & low_mask;
        let offset_high = offset_z >> low_bits;
        let max_high = checked_add(offset_z, z_len, "dense Z end")?
            .checked_sub(1)
            .ok_or(AkitaError::InvalidProof)?
            >> low_bits;
        let eq_high: Vec<E> = (offset_high..=max_high)
            .map(|idx| eq_eval_at_index(&full_vec_randomness[low_bits..], idx))
            .collect();
        cfg_into_iter!(0..z_range)
            .map(|k| {
                let block_idx = k / depth_commit;
                let dc = k % depth_commit;
                let mut weight = E::zero();
                for pt in 0..num_fold_groups {
                    for (df, &fold) in fold_gadget.iter().enumerate().take(depth_fold) {
                        let x = checked_add(
                            block_idx,
                            checked_mul(
                                block_len,
                                checked_add(
                                    pt,
                                    checked_mul(
                                        num_fold_groups,
                                        checked_add(
                                            df,
                                            checked_mul(depth_fold, dc, "dense Z df")?,
                                            "dense Z pt",
                                        )?,
                                        "dense Z df stride",
                                    )?,
                                    "dense Z pt stride",
                                )?,
                                "dense Z block stride",
                            )?,
                            "dense Z x",
                        )?;
                        let shifted = checked_add(offset_low, x, "dense Z low")?;
                        let low_idx = shifted & low_mask;
                        let high_carry = shifted >> low_bits;
                        let low = *eq_low.get(low_idx).ok_or(AkitaError::InvalidProof)?;
                        let high = *eq_high.get(high_carry).ok_or(AkitaError::InvalidProof)?;
                        weight -= (low * high).mul_base(fold);
                    }
                }
                Ok(weight)
            })
            .collect()
    }
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
    use super::setup_contribution_grouped::GroupSetupContributionPlan;
    use super::*;
    use crate::{gadget_row_scalars, AkitaExpandedSetup, AkitaSetupSeed, FlatMatrix, MRowLayout};
    use akita_algebra::ring::scalar_powers;
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
                offset_r: Some(0),
                global_block_base: 0,
            }],
            chunk_lengths: vec![crate::WitnessChunkLengths {
                z_len: z_range,
                e_len: 0,
                t_len: 0,
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
            None,
            None,
            Some(&fold_gadget),
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
        let expected = grouped_plan
            .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, 1)
            .unwrap();
        let got_grouped = grouped_plan
            .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
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
            .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, 1)
            .unwrap();
        let got = grouped_plan
            .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
            .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn grouped_multi_group_packed_matches_row_fallback() {
        let grouped_plan = GroupedSetupContributionPlan {
            d_rows: 2,
            d_physical_cols: 5,
            groups: vec![
                GroupSetupContributionPlan {
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
                },
                GroupSetupContributionPlan {
                    e_col_offset: 0,
                    t_cols: 4,
                    z_cols: 3,
                    n_a: 2,
                    n_b: 2,
                    e_eq_slice: vec![test_scalar(53), test_scalar(59)],
                    t_eq_slice: vec![
                        test_scalar(61),
                        test_scalar(67),
                        test_scalar(71),
                        test_scalar(73),
                    ],
                    z_eq_slice: vec![test_scalar(79), test_scalar(83), test_scalar(89)],
                    a_weights: vec![test_scalar(97), test_scalar(101)],
                    b_weights: vec![test_scalar(103), test_scalar(107)],
                    d_weights: vec![test_scalar(109), test_scalar(113)],
                },
            ],
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
            .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, 1)
            .unwrap();
        let got = grouped_plan
            .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
            .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn grouped_packed_direct_matches_row_fallback_with_nested_role_dims() {
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
        const D: usize = 4;
        const D_B: usize = 2;
        const D_D: usize = 2;
        let setup_len = 10;
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 0,
                max_num_batched_polys: 0,
                gen_ring_dim: D,
                max_setup_len: setup_len,
                public_matrix_seed: [0u8; 32],
            },
            FlatMatrix::from_flat_data(
                (0..setup_len * D)
                    .map(|idx| test_scalar(211 + idx as u128))
                    .collect(),
                D,
            ),
        );
        let alpha = test_scalar(3);
        let alpha_pows_a = scalar_powers(alpha, D);
        let alpha_pows_b = scalar_powers(alpha, D_B);
        let alpha_pows_d = scalar_powers(alpha, D_D);
        let expected = grouped_plan
            .evaluate_direct_by_rows::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d, D)
            .unwrap();
        let got = grouped_plan
            .evaluate_direct::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d)
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
