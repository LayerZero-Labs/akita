//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns the pure layout/weight derivation for the stage-3 setup
//! product. The prover consumes the materialized `bar_omega` vector, while the
//! verifier can evaluate the same plan directly against the packed setup.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::eval_ring_at_pows;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{AkitaError, ExtField, FieldCore, MulBase};

use crate::layout::MRowLayout;
use crate::proof::AkitaExpandedSetup;

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
    pub num_segments: usize,
    pub rows: usize,
    pub num_polys_per_segment: Vec<usize>,
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

    pub fn evaluate_direct<F, const D: usize>(
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
        if inputs.num_polys_per_segment.len() != inputs.num_segments {
            return Err(AkitaError::InvalidSize {
                expected: inputs.num_segments,
                actual: inputs.num_polys_per_segment.len(),
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
        if num_chunks > 1 && inputs.num_segments != 1 {
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

        let n_b_active = if inputs.m_row_layout.has_b_block() {
            inputs.n_b
        } else {
            0
        };
        let n_d_active = if inputs.m_row_layout.has_d_block() {
            inputs.n_d
        } else {
            0
        };
        // Canonical row layout: consistency (1) | A | optional B | optional D.
        let a_start = 1usize;
        let b_start = checked_add(a_start, inputs.n_a, "B row start")?;
        let b_rows_total = checked_mul(n_b_active, inputs.num_segments, "B row count")?;
        let d_start = checked_add(b_start, b_rows_total, "D row start")?;
        let d_end = checked_add(d_start, n_d_active, "D row end")?;
        if d_end > inputs.rows || inputs.rows > inputs.eq_tau1.len() {
            return Err(AkitaError::InvalidSetup(
                "M-row weights are inconsistent with setup evaluator layout".into(),
            ));
        }

        let stride_t = checked_mul(inputs.n_a, inputs.depth_open, "T stride")?;
        let cols_per_poly_t = checked_mul(stride_t, inputs.num_blocks, "T polynomial width")?;
        let b_per_claim_e = checked_mul(inputs.num_blocks, inputs.depth_open, "e-hat claim width")?;
        let n_cols_e = checked_mul(inputs.num_claims, b_per_claim_e, "e-hat column width")?;
        let max_group_poly_count = inputs
            .num_polys_per_segment
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let n_cols_t = checked_mul(max_group_poly_count, cols_per_poly_t, "T column width")?;

        let d_required = checked_mul(n_d_active, n_cols_e, "D setup footprint")?;
        let a_required = checked_mul(inputs.n_a, z_range, "A setup footprint")?;
        let b_required = checked_mul(n_b_active, n_cols_t, "B setup footprint")?;
        let required = d_required.max(b_required).max(a_required);
        if required == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator requires a non-empty packed footprint".into(),
            ));
        }

        let mut group_offsets = Vec::with_capacity(inputs.num_polys_per_segment.len());
        let mut next_offset = 0usize;
        for &group_poly_count in &inputs.num_polys_per_segment {
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
        let t_eq_slice_per_group: Vec<Vec<E>> = (0..inputs.num_segments)
            .map(|g| {
                let group_size = inputs.num_polys_per_segment[g];
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

        let b_weights_by_row: Vec<Vec<E>> = (0..n_b_active)
            .map(|row| {
                (0..inputs.num_segments)
                    .map(|g| inputs.eq_tau1[b_start + g * n_b_active + row])
                    .collect()
            })
            .collect();

        let mut endpoints = Vec::with_capacity(n_d_active + n_b_active + inputs.n_a + 2);
        endpoints.push(0);
        endpoints.push(required);
        push_role_boundaries(&mut endpoints, n_d_active, n_cols_e, "D")?;
        push_role_boundaries(&mut endpoints, n_b_active, n_cols_t, "B")?;
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
            d_weights: inputs.eq_tau1[d_start..d_end].to_vec(),
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
            checked_mul(inputs.num_segments, inputs.depth_fold, "Z high-eq width")?,
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
                        for pt in 0..inputs.num_segments {
                            let k = pt
                                + inputs.num_segments * df
                                + inputs.num_segments * inputs.depth_fold * dc
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
            checked_mul(inputs.block_len, inputs.num_segments, "dense Z block width")?;
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
                for pt in 0..inputs.num_segments {
                    for (df, &fg) in fold_gadget.iter().enumerate().take(inputs.depth_fold) {
                        let x = blk
                            + inputs.block_len * pt
                            + inputs.block_len * inputs.num_segments * df
                            + inputs.block_len * inputs.num_segments * inputs.depth_fold * dc;
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
fn checked_add(lhs: usize, rhs: usize, name: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{name} overflow")))
}

#[inline(always)]
fn checked_mul(lhs: usize, rhs: usize, name: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{name} overflow")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gadget_row_scalars, MRowLayout};
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
            num_segments: num_points,
            rows: 2,
            num_polys_per_segment: vec![0],
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
}
