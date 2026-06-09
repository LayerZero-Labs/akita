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
    pub num_points: usize,
    pub rows: usize,
    pub num_polys_per_commitment_group: Vec<usize>,
    pub num_public_rows: usize,
    /// Tiered split factor `f` (`1` = single-tier). When `> 1`, `n_b` is the
    /// stored first-tier `B'` rank, the stored `B'` width is `n_cols_t /
    /// tier_split` reused across `f` slices, and `n_f` is the second-tier `F`
    /// rank (the sent-commitment length).
    pub tier_split: usize,
    /// Second-tier `F` rank (`0` = single-tier).
    pub n_f: usize,
}

/// All tiered-commitment state for one setup-contribution level.
///
/// Present (`Some`) on a [`SetupContributionPlan`] only when `tier_split > 1`.
/// Bundling the first-tier `B'` and second-tier `F` dims + precomputed weight
/// tables keeps the single-tier call sites and the packed-scan kernel signature
/// free of tiered-specific parameters — a non-tiered reader just sees `None`.
struct TieredCommitmentData<E> {
    /// Split factor `f`.
    tier_split: usize,
    /// Stored B' rank (`n_b'`).
    n_b_small: usize,
    /// Stored B' width per stored row (`n_cols_t / tier_split`).
    b_inner_stride: usize,
    /// `n_b' · b_inner_stride` (stored B' prefix footprint).
    b_inner_required: usize,
    /// `[group][slice·n_b' + row] = eq_tau1[b_inner_start + g·(f·n_b') + slice·n_b' + row]`.
    b_inner_weights_by_group: Vec<Vec<E>>,
    /// Stored F width (`tier_split · n_b' · depth_open`).
    f_stride: usize,
    /// `n_f · f_stride` (F prefix footprint).
    f_required: usize,
    /// `[f_row][group] = eq_tau1[f_start + g·n_f + f_row]`.
    f_weights_by_row: Vec<Vec<E>>,
    /// `[group][col] = û_concat column MLE` over F's columns.
    u_eq_slice_per_group: Vec<Vec<E>>,
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
    /// Tiered-commitment tables/dims, or `None` for a single-tier plan.
    tiered: Option<TieredCommitmentData<E>>,
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

        // `Σ_λ eq_lambda[λ] · weight(λ)`, mirroring `evaluate_direct`'s
        // const-generic dispatch (the eq-weighted twin) so the tiered `B'`/`F`
        // bindings are included identically while the non-tiered hot loop stays
        // branch-free.
        let segments = self.segments();
        let tiered = self.tiered.as_ref();
        let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
            .map(|idx| {
                let segment = &segments[idx];
                let (has_b_active, b_start) = if tiered.is_some() {
                    (segment.has_b_inner, segment.b_inner_start_abs)
                } else {
                    (segment.has_b, segment.b_start_abs)
                };
                macro_rules! segment_sum {
                    ($has_d:literal, $has_b:literal, $has_a:literal) => {
                        bar_omega_segment_eval::<E, $has_d, $has_b, $has_a>(
                            segment.lo..segment.hi,
                            eq_lambda,
                            segment.d_start_abs,
                            segment.d_weight,
                            &self.e_eq_slice,
                            b_start,
                            segment.b_weights,
                            &self.t_eq_slice_per_group,
                            segment.a_start_abs,
                            segment.a_weight,
                            &self.z_eq_slice,
                            tiered,
                            segment.b_inner_row,
                            segment.has_f,
                            segment.f_row,
                        )
                    };
                }

                match (segment.has_d, has_b_active, segment.has_a) {
                    (true, true, true) => segment_sum!(true, true, true),
                    (true, true, false) => segment_sum!(true, true, false),
                    (true, false, true) => segment_sum!(true, false, true),
                    (false, true, true) => segment_sum!(false, true, true),
                    (true, false, false) => segment_sum!(true, false, false),
                    (false, true, false) => segment_sum!(false, true, false),
                    (false, false, true) => segment_sum!(false, false, true),
                    // A segment with no A/D/B may still carry the tiered F block.
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
        let tiered = self.tiered.as_ref();
        let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
            .map(|idx| -> Result<E, AkitaError> {
                let segment = &segments[idx];
                // The B-block is the full single-tier B (`has_b`) or, when
                // tiered, the first-tier B' (`has_b_inner`). Both flow through
                // the same `packed_slice_inner_sum` `HAS_B` arm — passing the
                // tiered data (`Some`) switches the B-weight (slice fold vs
                // linear) while A/D stay identical. The tiered second-tier `F`
                // (COMMIT) block is also folded into the same pass (gated by
                // `segment.has_f`), so every shared-vector entry — A, D, B'/B and
                // F alike — is evaluated by `eval_ring_at_pows` exactly once.
                let (has_b_active, b_start) = if tiered.is_some() {
                    (segment.has_b_inner, segment.b_inner_start_abs)
                } else {
                    (segment.has_b, segment.b_start_abs)
                };
                macro_rules! segment_sum {
                    ($has_d:literal, $has_b:literal, $has_a:literal) => {
                        packed_slice_inner_sum::<F, E, D, $has_d, $has_b, $has_a>(
                            segment.lo..segment.hi,
                            setup_flat,
                            alpha_pows,
                            segment.d_start_abs,
                            segment.d_weight,
                            &self.e_eq_slice,
                            b_start,
                            segment.b_weights,
                            &self.t_eq_slice_per_group,
                            segment.a_start_abs,
                            segment.a_weight,
                            &self.z_eq_slice,
                            tiered,
                            segment.b_inner_row,
                            segment.has_f,
                            segment.f_row,
                        )
                    };
                }

                Ok(match (segment.has_d, has_b_active, segment.has_a) {
                    (true, true, true) => segment_sum!(true, true, true),
                    (true, true, false) => segment_sum!(true, true, false),
                    (true, false, true) => segment_sum!(true, false, true),
                    (false, true, true) => segment_sum!(false, true, true),
                    (true, false, false) => segment_sum!(true, false, false),
                    (false, true, false) => segment_sum!(false, true, false),
                    (false, false, true) => segment_sum!(false, false, true),
                    // A segment with no A/D/B may still carry the tiered F block,
                    // so it is not necessarily empty.
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
    ///
    /// Single source of truth for the `bar_omega` paths
    /// (`materialize_bar_omega`, `evaluate_bar_omega_with_eq`), so they include
    /// the tiered first-tier `B'` (B_inner) and second-tier `F` bindings rather
    /// than omitting them. For single-tier plans the tiered blocks are inert
    /// (`self.tiered == None`), reproducing the former D/B/A weight exactly.
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
        // Tiered first-tier `B'` (B_inner) and second-tier `F` (COMMIT) blocks,
        // matching `packed_slice_inner_sum`. `has_b` (full single-tier B) and
        // `has_b_inner` (tiered B') are mutually exclusive — `has_b` is always
        // false under a tiered plan (`b_required == 0`) — so the two B branches
        // never both fire. Inert when `self.tiered == None`.
        if let Some(td) = self.tiered.as_ref() {
            if segment.has_b_inner {
                let col = lambda - segment.b_inner_start_abs;
                for (g, t_eq_slice) in self.t_eq_slice_per_group.iter().enumerate() {
                    weight += fold_reused_b_weight(
                        segment.b_inner_row,
                        col,
                        td.tier_split,
                        td.n_b_small,
                        td.b_inner_stride,
                        &td.b_inner_weights_by_group[g],
                        t_eq_slice,
                    );
                }
            }
            if segment.has_f {
                let col = lambda - segment.f_row * td.f_stride;
                for (g, u_eq) in td.u_eq_slice_per_group.iter().enumerate() {
                    let rw = td.f_weights_by_row[segment.f_row][g];
                    if !rw.is_zero() {
                        weight += rw * u_eq[col];
                    }
                }
            }
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

                // Tiered first-tier `B'` (B_inner) and second-tier `F` blocks.
                // `endpoints` carries both their row boundaries (pushed in
                // `prepare` for tiered plans), so each block's row index is
                // constant across `[lo, hi)`.
                let has_b_inner = match self.tiered.as_ref() {
                    Some(td) => td.b_inner_stride != 0 && lo < td.b_inner_required,
                    None => false,
                };
                let b_inner_stride = self.tiered.as_ref().map_or(0, |td| td.b_inner_stride);
                let b_inner_row = if has_b_inner { lo / b_inner_stride } else { 0 };
                let b_inner_start_abs = if has_b_inner {
                    b_inner_row * b_inner_stride
                } else {
                    0
                };
                let has_f = match self.tiered.as_ref() {
                    Some(td) => td.f_stride != 0 && lo < td.f_required,
                    None => false,
                };
                let f_stride = self.tiered.as_ref().map_or(0, |td| td.f_stride);
                let f_row = if has_f { lo / f_stride } else { 0 };

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
                    has_b_inner,
                    b_inner_start_abs,
                    b_inner_row,
                    has_f,
                    f_row,
                })
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare<F>(
        inputs: &SetupContributionPlanInputs<E>,
        full_vec_randomness: &[E],
        eq_low: Option<&[E]>,
        z_block_low_eq: Option<&[E]>,
        fold_gadget: &[F],
        offset_e: usize,
        offset_t: usize,
        offset_z: usize,
        offset_u: usize,
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
        if inputs.num_polys_per_commitment_group.len() != inputs.num_points {
            return Err(AkitaError::InvalidSize {
                expected: inputs.num_points,
                actual: inputs.num_polys_per_commitment_group.len(),
            });
        }

        let block_bits = inputs.num_blocks.trailing_zeros() as usize;
        if block_bits > full_vec_randomness.len() {
            return Err(AkitaError::InvalidSize {
                expected: block_bits,
                actual: full_vec_randomness.len(),
            });
        }
        let block_mask = inputs.num_blocks - 1;
        let block_offset_low = offset_e & block_mask;
        let e_offset_high = offset_e >> block_bits;
        let t_offset_high = offset_t >> block_bits;
        let high_challenges = &full_vec_randomness[block_bits..];
        let eq_low_storage;
        let eq_low = if let Some(eq_low) = eq_low {
            eq_low
        } else {
            eq_low_storage = EqPolynomial::evals(&full_vec_randomness[..block_bits])?;
            &eq_low_storage
        };
        if eq_low.len() < inputs.num_blocks {
            return Err(AkitaError::InvalidSize {
                expected: inputs.num_blocks,
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
        let z_offset_low = offset_z & inputs.block_len.saturating_sub(1);
        let z_range = inputs.inner_width;
        let expected_z_range = checked_mul(inputs.block_len, inputs.depth_commit, "Z width")?;
        if z_range != expected_z_range {
            return Err(AkitaError::InvalidSize {
                expected: expected_z_range,
                actual: z_range,
            });
        }
        let z_dims_pow2 = inputs.block_len.is_power_of_two();

        let tiered = inputs.tier_split > 1;
        if tiered && (inputs.n_f == 0 || inputs.num_points != 1) {
            return Err(AkitaError::InvalidSetup(
                "tiered setup contribution requires n_f > 0 and a single commitment group".into(),
            ));
        }
        let n_d_active = match inputs.m_row_layout {
            MRowLayout::WithDBlock => inputs.n_d,
            MRowLayout::WithoutDBlock => 0,
        };
        // Canonical row layout: consistency (1) | public | D (n_d_active) |
        // COMMIT (F when tiered, else B) | B_inner (tiered) | A.
        let d_start = checked_add(1, inputs.num_public_rows, "D row start")?;
        // COMMIT block start (the F block when tiered, the B block otherwise).
        let f_start = checked_add(d_start, n_d_active, "COMMIT row start")?;
        let commit_rows_pg = if tiered { inputs.n_f } else { inputs.n_b };
        let b_inner_rows_pg = if tiered {
            checked_mul(inputs.tier_split, inputs.n_b, "B_inner rows")?
        } else {
            0
        };
        let commit_rows = checked_mul(commit_rows_pg, inputs.num_points, "COMMIT row count")?;
        let b_inner_start = checked_add(f_start, commit_rows, "B_inner row start")?;
        let b_inner_rows_total =
            checked_mul(b_inner_rows_pg, inputs.num_points, "B_inner row count")?;
        let a_start = checked_add(b_inner_start, b_inner_rows_total, "A row start")?;
        let a_end = checked_add(a_start, inputs.n_a, "A row end")?;
        // Non-tiered alias used by the packed B scan.
        let b_start = f_start;
        if a_end > inputs.rows || inputs.rows > inputs.eq_tau1.len() {
            return Err(AkitaError::InvalidSetup(
                "M-row weights are inconsistent with setup evaluator layout".into(),
            ));
        }

        let stride_t = checked_mul(inputs.n_a, inputs.depth_open, "T stride")?;
        let cols_per_poly_t = checked_mul(stride_t, inputs.num_blocks, "T polynomial width")?;
        let b_per_claim_e = checked_mul(inputs.num_blocks, inputs.depth_open, "e-hat claim width")?;
        let n_cols_e = checked_mul(inputs.num_claims, b_per_claim_e, "e-hat column width")?;
        let max_group_poly_count = inputs
            .num_polys_per_commitment_group
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let n_cols_t = checked_mul(max_group_poly_count, cols_per_poly_t, "T column width")?;

        let d_required = checked_mul(n_d_active, n_cols_e, "D setup footprint")?;
        let a_required = checked_mul(inputs.n_a, z_range, "A setup footprint")?;
        // Packed B is disabled when tiered (the COMMIT block is F + B_inner,
        // scanned separately so the stored B' is read once per entry).
        let b_required = if tiered {
            0
        } else {
            checked_mul(inputs.n_b, n_cols_t, "B setup footprint")?
        };
        // Tiered stored-prefix footprints. `n_cols_t` is the full per-group B
        // width; the stored B' is `n_cols_t / tier_split` reused across slices.
        let (b_inner_stride, b_inner_required, f_stride, f_required) = if tiered {
            if n_cols_t == 0 || !n_cols_t.is_multiple_of(inputs.tier_split) {
                return Err(AkitaError::InvalidSetup(
                    "tiered B' width does not divide the per-group T width".into(),
                ));
            }
            let b_inner_stride = n_cols_t / inputs.tier_split;
            let b_inner_required =
                checked_mul(inputs.n_b, b_inner_stride, "B_inner setup footprint")?;
            let f_stride = checked_mul(
                checked_mul(inputs.tier_split, inputs.n_b, "F width")?,
                inputs.depth_open,
                "F width",
            )?;
            let f_required = checked_mul(inputs.n_f, f_stride, "F setup footprint")?;
            (b_inner_stride, b_inner_required, f_stride, f_required)
        } else {
            (0, 0, 0, 0)
        };
        let required = d_required
            .max(b_required)
            .max(a_required)
            .max(b_inner_required)
            .max(f_required);
        if required == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator requires a non-empty packed footprint".into(),
            ));
        }

        let mut group_offsets = Vec::with_capacity(inputs.num_polys_per_commitment_group.len());
        let mut next_offset = 0usize;
        for &group_poly_count in &inputs.num_polys_per_commitment_group {
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
            let eq_hi_e_table: Vec<E> = (0..=e_hi_len)
                .map(|k| eq_eval_at_index(high_challenges, e_offset_high + k))
                .collect();
            cfg_into_iter!(0..n_cols_e)
                .map(|current_index| {
                    let (low_eq_idx, high_eq_idx) = get_eq_indices_for_d(
                        current_index,
                        inputs.depth_open,
                        inputs.num_blocks,
                        inputs.num_claims,
                        b_per_claim_e,
                        block_offset_low,
                        block_mask,
                        block_bits,
                    );
                    eq_low[low_eq_idx] * eq_hi_e_table[high_eq_idx]
                })
                .collect()
        };

        let t_hi_len = checked_mul(
            checked_mul(inputs.num_t_vectors, inputs.depth_open, "T high-eq width")?,
            inputs.n_a,
            "T high-eq width",
        )?;
        let eq_hi_t_table: Vec<E> = (0..=t_hi_len)
            .map(|k| eq_eval_at_index(high_challenges, t_offset_high + k))
            .collect();
        let t_eq_slice_per_group: Vec<Vec<E>> = (0..inputs.num_points)
            .map(|g| {
                let group_size = inputs.num_polys_per_commitment_group[g];
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
                            inputs.depth_open,
                            inputs.n_a,
                            inputs.num_blocks,
                            inputs.num_t_vectors,
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

        let z_eq_slice = if z_dims_pow2 {
            let z_block_low_storage;
            let z_block_low_eq = if let Some(z_block_low_eq) = z_block_low_eq {
                z_block_low_eq
            } else {
                z_block_low_storage =
                    EqPolynomial::evals(&full_vec_randomness[..z_offset_low_bits])?;
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
                checked_mul(inputs.num_points, inputs.depth_fold, "Z high-eq width")?,
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
                            for pt in 0..inputs.num_points {
                                let k = pt
                                    + inputs.num_points * df
                                    + inputs.num_points * inputs.depth_fold * dc
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
                        inputs.depth_commit,
                        z_offset_low,
                        z_block_mask,
                        z_offset_low_bits,
                    );
                    z_block_low_eq[low_eq_idx] * s_per_dc_per_carry[depth_commit_idx][block_carry]
                })
                .collect()
        } else {
            let z_total_blocks_dense =
                checked_mul(inputs.block_len, inputs.num_points, "dense Z block width")?;
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

            cfg_into_iter!(0..z_range)
                .map(|c| {
                    let dc = c % inputs.depth_commit;
                    let blk = c / inputs.depth_commit;
                    let mut acc = E::zero();
                    for pt in 0..inputs.num_points {
                        for (df, &fg) in fold_gadget.iter().enumerate().take(inputs.depth_fold) {
                            let x = blk
                                + inputs.block_len * pt
                                + inputs.block_len * inputs.num_points * df
                                + inputs.block_len * inputs.num_points * inputs.depth_fold * dc;
                            let sum = offset_z_dense_low + x;
                            let low_idx = sum & mask;
                            // `eq_high_z_dense` starts at `offset_z_dense_high`,
                            // so the low-bit carry is the relative high-table index.
                            let high_carry = sum >> k;
                            let eq_val = eq_low_z_dense[low_idx] * eq_high_z_dense[high_carry];
                            acc += eq_val.mul_base(fg);
                        }
                    }
                    -acc
                })
                .collect()
        };

        // Packed B weights (single-tier only). For tiered levels the packed B
        // scan is disabled (`b_required == 0`) and the COMMIT/B_inner weights
        // are built below.
        let b_weights_by_row: Vec<Vec<E>> = if tiered {
            Vec::new()
        } else {
            (0..inputs.n_b)
                .map(|row| {
                    (0..inputs.num_points)
                        .map(|g| inputs.eq_tau1[b_start + g * inputs.n_b + row])
                        .collect()
                })
                .collect()
        };

        // Tiered second-tier weight tables, bundled into `TieredCommitmentData`
        // (`None` for single-tier plans).
        let tiered_data: Option<TieredCommitmentData<E>> = if tiered {
            let f_weights_by_row: Vec<Vec<E>> = (0..inputs.n_f)
                .map(|row| {
                    (0..inputs.num_points)
                        .map(|g| inputs.eq_tau1[f_start + g * inputs.n_f + row])
                        .collect()
                })
                .collect();
            // û_concat column MLE over F's columns: a flat contiguous witness
            // segment at `offset_u`, `f_stride` columns per commitment group.
            let u_eq_slice_per_group: Vec<Vec<E>> = (0..inputs.num_points)
                .map(|g| {
                    (0..f_stride)
                        .map(|c| eq_eval_at_index(full_vec_randomness, offset_u + g * f_stride + c))
                        .collect()
                })
                .collect();
            let inner_rows_pg = inputs.tier_split * inputs.n_b;
            // `[group][slice_row]` so each group's slice-row weights compose
            // directly with `fold_reused_b_weight`.
            let b_inner_weights_by_group: Vec<Vec<E>> = (0..inputs.num_points)
                .map(|g| {
                    (0..inner_rows_pg)
                        .map(|slice_row| {
                            inputs.eq_tau1[b_inner_start + g * inner_rows_pg + slice_row]
                        })
                        .collect()
                })
                .collect();
            Some(TieredCommitmentData {
                tier_split: inputs.tier_split,
                n_b_small: inputs.n_b,
                b_inner_stride,
                b_inner_required,
                b_inner_weights_by_group,
                f_stride,
                f_required,
                f_weights_by_row,
                u_eq_slice_per_group,
            })
        } else {
            None
        };

        let mut endpoints =
            Vec::with_capacity(n_d_active + inputs.n_b + inputs.n_a + inputs.n_f + 2);
        endpoints.push(0);
        endpoints.push(required);
        push_role_boundaries(&mut endpoints, n_d_active, n_cols_e, "D")?;
        if tiered {
            // First-tier `B'` (B_inner) and second-tier `F` row boundaries so the
            // fused A/D/B'/F pass sees a constant B'/F row across each segment.
            // These only sub-split the existing D/A segments, so the prover
            // `bar_omega` weights (which ignore B_inner/F) are unchanged.
            push_role_boundaries(&mut endpoints, inputs.n_b, b_inner_stride, "B_inner")?;
            push_role_boundaries(&mut endpoints, inputs.n_f, f_stride, "F")?;
        } else {
            push_role_boundaries(&mut endpoints, inputs.n_b, n_cols_t, "B")?;
        }
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
            d_weights: inputs.eq_tau1[d_start..(d_start + n_d_active)].to_vec(),
            b_weights_by_row,
            a_weights: inputs.eq_tau1[a_start..a_end].to_vec(),
            endpoints,
            tiered: tiered_data,
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
    // Tiered first-tier `B'` (B_inner) and second-tier `F` (COMMIT) blocks, both
    // fused into the same pass as A/D. Always inactive (`false`/`0`) for
    // single-tier plans (the full single-tier `B` uses `has_b` above instead).
    // `segments()` populates these only when `tier_split > 1`; the prover
    // `bar_omega` path (`weight_at`) ignores them, so the extra B_inner/F segment
    // boundaries leave `bar_omega` unchanged.
    has_b_inner: bool,
    b_inner_start_abs: usize,
    b_inner_row: usize,
    has_f: bool,
    f_row: usize,
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
    // Tiered-commitment data (A/D are identical in both modes):
    // - `tiered = None`: the B-block is the full single-tier B; its weight is
    //   the linear `b_weights[g] · t_eq[λ - b_start]`.
    // - `tiered = Some(td)`: the B-block (when `HAS_B`) is the first-tier B'
    //   reused across `td.tier_split` column-slices; its weight is the slice fold
    //   (`fold_reused_b_weight`), `b_start` is the B' row start, `b_inner_row` is
    //   the (segment-constant) B' row, and `b_weights` is unused. The second-tier
    //   `F` (COMMIT) block is added when `has_f` (`f_row` is the segment-constant
    //   F row): `f_weights_by_row[f_row][g] · u_eq[g][λ - f_row·f_stride]`. Both
    //   tiered blocks share this single pass with A/D so each entry is evaluated
    //   by `eval_ring_at_pows` exactly once.
    tiered: Option<&TieredCommitmentData<E>>,
    b_inner_row: usize,
    has_f: bool,
    f_row: usize,
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
                if let Some(td) = tiered {
                    let col = lambda - b_start;
                    for (g, t_eq_slice) in t_eq_per_group.iter().enumerate() {
                        weight += fold_reused_b_weight(
                            b_inner_row,
                            col,
                            td.tier_split,
                            td.n_b_small,
                            td.b_inner_stride,
                            &td.b_inner_weights_by_group[g],
                            t_eq_slice,
                        );
                    }
                } else {
                    for (g, t_eq_slice) in t_eq_per_group.iter().enumerate() {
                        weight += b_weights[g] * t_eq_slice[lambda - b_start];
                    }
                }
            }
            if HAS_A {
                weight += a_weight * z_eq[lambda - a_start];
            }
            // Tiered second-tier `F` (COMMIT) block, fused into the same pass.
            // `has_f` / `f_row` are segment-constant, so this is a predictable
            // branch; it is inert (`None` / `has_f == false`) for single-tier.
            if has_f {
                if let Some(td) = tiered {
                    let col = lambda - f_row * td.f_stride;
                    for (g, u_eq) in td.u_eq_slice_per_group.iter().enumerate() {
                        let rw = td.f_weights_by_row[f_row][g];
                        if !rw.is_zero() {
                            weight += rw * u_eq[col];
                        }
                    }
                }
            }
            if !weight.is_zero() {
                acc += eval_ring_at_pows(&setup_flat[lambda], alpha_pows) * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

/// Eq-weighted twin of [`packed_slice_inner_sum`]: `Σ_λ eq_lambda[λ] · weight(λ)`
/// over one segment, where `weight` is the identical A/D/B (or tiered B'/F)
/// combination. Used by `evaluate_bar_omega_with_eq` (verifier stage-3); kept
/// const-generic on `HAS_D`/`HAS_B`/`HAS_A` for a branch-free hot loop, with the
/// tiered `B'`/`F` blocks added under the same runtime gates as the direct scan.
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
    // Tiered blocks, identical to `packed_slice_inner_sum`: `None` / `has_f ==
    // false` for single-tier plans (inert, predictable branch).
    tiered: Option<&TieredCommitmentData<E>>,
    b_inner_row: usize,
    has_f: bool,
    f_row: usize,
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
                if let Some(td) = tiered {
                    let col = lambda - b_start;
                    for (g, t_eq_slice) in t_eq_per_group.iter().enumerate() {
                        weight += fold_reused_b_weight(
                            b_inner_row,
                            col,
                            td.tier_split,
                            td.n_b_small,
                            td.b_inner_stride,
                            &td.b_inner_weights_by_group[g],
                            t_eq_slice,
                        );
                    }
                } else {
                    for (g, t_eq_slice) in t_eq_per_group.iter().enumerate() {
                        weight += b_weights[g] * t_eq_slice[lambda - b_start];
                    }
                }
            }
            if HAS_A {
                weight += a_weight * z_eq[lambda - a_start];
            }
            if has_f {
                if let Some(td) = tiered {
                    let col = lambda - f_row * td.f_stride;
                    for (g, u_eq) in td.u_eq_slice_per_group.iter().enumerate() {
                        let rw = td.f_weights_by_row[f_row][g];
                        if !rw.is_zero() {
                            weight += rw * u_eq[col];
                        }
                    }
                }
            }
            if !weight.is_zero() {
                acc += eq_lambda[lambda] * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

/// Folded weight of a stored `B'` entry `(row, col)` when the first-tier matrix
/// `B'` (dimensions `n_b_small × width_small`) is reused across `tier_split`
/// equal column-slices of `t̂` (the tiered-commitment design).
///
/// The logical relation matrix is the block-diagonal `blockdiag(B', …, B')`
/// (`tier_split` copies); slice `j` occupies logical rows
/// `[j·n_b_small, (j+1)·n_b_small)` and logical columns
/// `[j·width_small, (j+1)·width_small)`. Scanning the *stored* `B'` once and
/// weighting entry `(row, col)` by this fold — the sum over slices of the
/// `B_inner` row weight times the `t̂` slice-column eq-MLE — yields exactly the
/// same setup contribution as scanning the full `blockdiag(B', …, B')` once per
/// logical entry, but with `tier_split×` fewer `eval_ring_at_pows` calls. This
/// is the verifier-speedup hinge proved by the `reused_b_fold_matches_blockdiag`
/// unit test in this module.
#[inline]
pub fn fold_reused_b_weight<E: FieldCore>(
    row: usize,
    col: usize,
    tier_split: usize,
    n_b_small: usize,
    width_small: usize,
    b_inner_row_weight: &[E],
    t_col_eq: &[E],
) -> E {
    let mut acc = E::zero();
    for j in 0..tier_split {
        let rw = b_inner_row_weight[j * n_b_small + row];
        let cw = t_col_eq[j * width_small + col];
        if !rw.is_zero() && !cw.is_zero() {
            acc += rw * cw;
        }
    }
    acc
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn get_eq_indices_for_d(
    current_index: usize,
    num_digits: usize,
    num_blocks: usize,
    num_claims: usize,
    blocks_per_claim_e: usize,
    block_offset_low: usize,
    block_mask: usize,
    block_bits: usize,
) -> (usize, usize) {
    let digit_idx = current_index % num_digits;
    let block_idx = (current_index / num_digits) % num_blocks;
    let claim_idx = current_index / blocks_per_claim_e;
    let m_layout_high_idx = digit_idx * num_claims + claim_idx;
    let block_sum = block_offset_low + block_idx;
    let low_eq_idx = block_sum & block_mask;
    let block_carry = block_sum >> block_bits;
    let high_eq_idx = m_layout_high_idx + block_carry;
    (low_eq_idx, high_eq_idx)
}

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
        let offset_z = 192;
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
            num_points,
            rows: 2,
            num_polys_per_commitment_group: vec![0],
            num_public_rows: 0,
            tier_split: 1,
            n_f: 0,
        };

        let plan = SetupContributionPlan::prepare::<F>(
            &inputs,
            &full_vec_randomness,
            None,
            None,
            &fold_gadget,
            0,
            64,
            offset_z,
            0,
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
    #[allow(clippy::needless_range_loop)]
    fn reused_b_fold_matches_blockdiag() {
        // The tiered setup contribution scans the stored B' once with folded
        // slice weights; this must equal scanning the full blockdiag(B',…,B')
        // once per logical entry. Verify the algebraic identity directly.
        let tier_split = 4usize;
        let n_b_small = 2usize;
        let width_small = 3usize;
        let alpha = test_scalar(7);

        // Stored B': n_b_small × width_small ring elements (D-coefficient rings).
        const TD: usize = 4;
        let alpha_pows: Vec<F> = {
            let mut acc = F::one();
            (0..TD)
                .map(|_| {
                    let v = acc;
                    acc *= alpha;
                    v
                })
                .collect()
        };
        let b_prime: Vec<CyclotomicRing<F, TD>> = (0..n_b_small * width_small)
            .map(|idx| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                    test_scalar((idx as u128 + 1) * 31 + k as u128 * 7)
                }))
            })
            .collect();

        // Per-(slice,row) and per-(slice,col) weights for one commitment group.
        let row_weight: Vec<F> = (0..tier_split * n_b_small)
            .map(|i| test_scalar(13 + i as u128 * 5))
            .collect();
        let col_eq: Vec<F> = (0..tier_split * width_small)
            .map(|c| test_scalar(101 + c as u128 * 3))
            .collect();

        // Folded scan: stored B' once, folded weight per entry.
        let mut folded = F::zero();
        for row in 0..n_b_small {
            for col in 0..width_small {
                let w = fold_reused_b_weight::<F>(
                    row,
                    col,
                    tier_split,
                    n_b_small,
                    width_small,
                    &row_weight,
                    &col_eq,
                );
                folded += eval_ring_at_pows(&b_prime[row * width_small + col], &alpha_pows) * w;
            }
        }

        // Naive scan over the materialized blockdiag(B',…,B').
        let logical_rows = tier_split * n_b_small;
        let logical_cols = tier_split * width_small;
        let mut naive = F::zero();
        for lrow in 0..logical_rows {
            for lcol in 0..logical_cols {
                let slice_r = lrow / n_b_small;
                let slice_c = lcol / width_small;
                if slice_r != slice_c {
                    continue; // off-diagonal block is zero
                }
                let row = lrow % n_b_small;
                let col = lcol % width_small;
                let entry = eval_ring_at_pows(&b_prime[row * width_small + col], &alpha_pows);
                naive += entry * row_weight[lrow] * col_eq[lcol];
            }
        }

        assert_eq!(
            folded, naive,
            "folded B' scan must equal full blockdiag scan"
        );
    }
}
