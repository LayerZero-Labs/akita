use super::*;
use crate::protocol::ring_relation::validate_chunked_witness_cfg;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_flat_ring_at_pows, scalar_powers};
use akita_challenges::Challenges;
use akita_field::cfg_into_iter;
use akita_types::{
    build_grouped_root_stage2_trace_table, build_trace_table_scaled,
    gadget_row_scalars, ring_relation_segment_lengths, trace_public_weights_recursive,
    trace_public_weights_root_terms, CommitmentRingDims, ConsistencyLayer, LevelParams,
    OpeningClaimsLayout, PreparedOpeningPoint, RelationQuotientLayout, RelationRowFamily,
    RelationRowLayout, RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationInstance,
    RingRelationOpeningCounts, TraceWeightLayout,     FOLD_CONSISTENCY_ROW,
};

pub use akita_types::compute_grouped_m_evals_x;

/// Produce the compact `Vec<i8>` eval table of `w` for the fused prover.
///
/// The compact witness stays in the raw flat [`build_w_coeffs`] order.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by the ring
/// dimension.
pub fn build_w_evals_compact(w: &[i8], d: usize) -> Result<(Vec<i8>, usize, usize), AkitaError> {
    if !w.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let live_ring_count = w.len() / d;
    let col_bits = live_ring_count.next_power_of_two().trailing_zeros() as usize;
    let ring_bits = d.trailing_zeros() as usize;
    Ok((w.to_vec(), col_bits, ring_bits))
}

/// Unified M-table evaluation for the batched CWSS protocol.
///
/// All claims share one ring-level opening point and one committed bundle.
///
/// # Errors
///
/// Returns an error if the batch shape, opening-point layout, challenge count,
/// or expanded matrix dimensions are inconsistent.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_relation_column_weights")]
pub fn compute_relation_column_weights<F, E>(
    setup: &AkitaExpandedSetup<F>,
    opening_point: &RingOpeningPoint<F>,
    ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
    challenges: &Challenges,
    alpha: E,
    alpha_pows: &[E],
    role_dims: CommitmentRingDims,
    lp: &LevelParams,
    tau1: &[E],
    num_polys: usize,
    num_commitment_groups: usize,
    gamma: &[E],
    m_row_layout: MRowLayout,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F>,
{
    let d_a = role_dims.d_a();
    let d_b = role_dims.d_b();
    let d_d = role_dims.d_d();
    if alpha_pows.len() != d_a {
        return Err(AkitaError::InvalidSize {
            expected: d_a,
            actual: alpha_pows.len(),
        });
    }
    let alpha_pows_a = alpha_pows;
    let alpha_pows_b = scalar_powers(alpha, d_b);
    let alpha_pows_d = scalar_powers(alpha, d_d);
    let num_claims = gamma.len();
    if opening_point.a.len() < lp.block_len || opening_point.b.len() != lp.num_blocks {
        return Err(AkitaError::InvalidInput(
            "batched prover opening-point layout mismatch".to_string(),
        ));
    }
    if ring_multiplier_point.a_len() < lp.block_len
        || ring_multiplier_point.b_len() != lp.num_blocks
    {
        return Err(AkitaError::InvalidInput(
            "batched prover ring-multiplier opening-point layout mismatch".to_string(),
        ));
    }
    if num_polys != num_claims {
        return Err(AkitaError::InvalidInput(
            "ring switch currently requires dense single-group claims".to_string(),
        ));
    }

    let depth_commit = lp.num_digits_commit;
    let depth_open = lp.num_digits_open;
    let log_basis = lp.log_basis;
    let num_blocks = lp.num_blocks;
    let num_t_vectors = num_polys;

    let total_blocks = num_blocks
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched block count overflow".to_string()))?;
    let t_total_blocks = num_blocks
        .checked_mul(num_t_vectors)
        .ok_or_else(|| AkitaError::InvalidSetup("batched t block count overflow".to_string()))?;
    if challenges.logical_len() != total_blocks {
        return Err(AkitaError::InvalidSize {
            expected: total_blocks,
            actual: challenges.logical_len(),
        });
    }
    let block_len = lp.block_len;
    let segment_lengths = ring_relation_segment_lengths::<F>(
        lp,
        RingRelationOpeningCounts {
            num_claims,
            num_t_vectors,
        },
        m_row_layout,
    )?;
    let w_len = segment_lengths.e_len;
    let t_len = segment_lengths.t_len;
    let z_len = segment_lengths.z_len;
    let depth_fold = lp.num_digits_fold(num_claims, lp.field_bits_for_cache())?;
    let inner_width = block_len * depth_commit;
    let z_base_len = inner_width;
    let n_a = lp.a_key.row_len();
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    // Terminal layout drops the D-block from the M-matrix entirely; offsets
    // and per-row gates must use 0 for the n_d position.
    let n_d_active = match m_row_layout {
        MRowLayout::WithDBlock => n_d,
        MRowLayout::WithoutDBlock => 0,
    };
    let levels = r_decomp_levels::<F>(log_basis);
    let opening_batch = OpeningClaimsLayout::new(8, num_claims)?;
    let row_layout = RelationRowLayout::for_scalar_level::<F>(
        lp,
        role_dims,
        m_row_layout,
        &opening_batch,
        num_commitment_groups,
    )?;
    let rows = row_layout.total_row_count();
    let quotient_layout = RelationQuotientLayout::from_row_layout(&row_layout, levels);
    quotient_layout.validate()?;
    let r_tail_len = quotient_layout.total_coeffs();
    // Chunked layout replicates the `z` segment once per window; `e`/`t` are
    // partitioned (their totals are unchanged). `num_chunks = 1` is the
    // single-chunk case.
    validate_chunked_witness_cfg(lp)?;
    let num_chunks = lp.witness_chunk.num_chunks;
    let z_cols_total = z_len
        .checked_mul(num_chunks)
        .ok_or_else(|| AkitaError::InvalidSetup("chunked Z width overflow".to_string()))?;
    let total_cols = w_len
        .checked_add(t_len)
        .and_then(|cols| cols.checked_add(z_cols_total))
        .and_then(|cols| cols.checked_add(r_tail_len))
        .ok_or_else(|| AkitaError::InvalidSetup("expanded M width overflow".to_string()))?;

    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let g1_open: Vec<E> = gadget_row_scalars::<F>(depth_open, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let g1_commit: Vec<E> = gadget_row_scalars::<F>(depth_commit, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let fold_gadget: Vec<E> = gadget_row_scalars::<F>(depth_fold, log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    let x_len = total_cols.next_power_of_two();
    let mut out = Vec::with_capacity(x_len);

    let c_alphas: Vec<E> = match challenges {
        Challenges::Sparse {
            challenges: sparse, ..
        } => sparse
            .iter()
            .map(|challenge| challenge.eval_at_pows::<F, E>(alpha_pows))
            .collect::<Result<_, _>>()?,
        Challenges::Tensor { factored: _ } => challenges.evals_at_pows::<F, E>(alpha_pows)?,
    };

    let d_message_width = total_blocks
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("D setup width overflow".to_string()))?;
    let d_width = d_message_width;
    let t_cols_per_vector = n_a
        .checked_mul(depth_open)
        .and_then(|len| len.checked_mul(num_blocks))
        .ok_or_else(|| AkitaError::InvalidSetup("B setup vector width overflow".to_string()))?;
    let b_message_width = num_polys
        .checked_mul(t_cols_per_vector)
        .ok_or_else(|| AkitaError::InvalidSetup("B setup width overflow".to_string()))?;
    let b_width = b_message_width;
    let a_width = inner_width;
    let d_view = setup.shared_matrix.ring_view_dyn(n_d, d_width, d_d)?;
    let b_view = setup.shared_matrix.ring_view_dyn(n_b, b_width, d_b)?;
    let a_view = setup.shared_matrix.ring_view_dyn(n_a, a_width, d_a)?;
    let d_rows: Vec<&[F]> = (0..n_d)
        .map(|r| d_view.row_flat(r))
        .collect::<Result<_, _>>()?;
    let b_rows: Vec<&[F]> = (0..n_b)
        .map(|r| b_view.row_flat(r))
        .collect::<Result<_, _>>()?;
    let a_rows: Vec<&[F]> = (0..n_a)
        .map(|r| a_view.row_flat(r))
        .collect::<Result<_, _>>()?;

    // Canonical row layout: EvaluationTrace | FoldEvaluation | FoldConsistency | B | D.
    let fold_evaluation_row = row_layout
        .family(RelationRowFamily::FoldEvaluation)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("relation row layout missing FoldEvaluation".to_string())
        })?
        .row_start;
    let a_start = FOLD_CONSISTENCY_ROW;
    let outer = row_layout
        .family(RelationRowFamily::OuterConsistency {
            layer: ConsistencyLayer::Base,
        })
        .ok_or_else(|| {
            AkitaError::InvalidSetup("relation row layout missing OuterConsistency".to_string())
        })?;
    let b_start = outer.row_start;
    let outer_row_count = outer.row_count;
    let d_start = row_layout
        .family(RelationRowFamily::OpeningConsistency {
            layer: ConsistencyLayer::Base,
        })
        .map(|family| family.row_start)
        .unwrap_or(rows);
    let a_weights = &eq_tau1[a_start..(a_start + n_a)];
    let consistency_weight = eq_tau1[fold_evaluation_row];
    let t_compound_per_block = n_a * depth_open;

    let w_segment: Vec<E> = cfg_into_iter!(0..w_len)
        .map(|x| {
            let dig = x / total_blocks;
            let blk = x % total_blocks;
            let d_phys_col = blk * depth_open + dig;
            let mut acc = consistency_weight * c_alphas[blk] * g1_open[dig];
            // Terminal layout: `n_d_active == 0`, so this loop is empty and
            // the D-block contribution is omitted.
            for (di, eq_i) in eq_tau1[d_start..(d_start + n_d_active)].iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i
                        * eval_flat_ring_at_pows(
                            &d_rows[di][d_phys_col * d_d..(d_phys_col + 1) * d_d],
                            &alpha_pows_d,
                        );
                }
            }
            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    let mut challenge_sums_by_t_block = vec![E::zero(); t_total_blocks];
    for claim_idx in 0..num_claims {
        let dst_offset = claim_idx * num_blocks;
        let src_offset = claim_idx * num_blocks;
        for block_idx in 0..num_blocks {
            challenge_sums_by_t_block[dst_offset + block_idx] += c_alphas[src_offset + block_idx];
        }
    }
    let t_segment: Vec<E> = cfg_into_iter!(0..t_len)
        .map(|x| {
            let compound_dig = x / t_total_blocks;
            let blk = x % t_total_blocks;
            let a_idx = compound_dig / depth_open;
            let digit_idx = compound_dig % depth_open;
            let t_vector_idx = blk / num_blocks;
            let block_idx = blk % num_blocks;
            let phys_claim_offset =
                block_idx * t_compound_per_block + a_idx * depth_open + digit_idx;
            let local_col = t_vector_idx * t_cols_per_vector + phys_claim_offset;
            let mut acc = a_weights[a_idx] * challenge_sums_by_t_block[blk] * g1_open[digit_idx];
            let commitment_weights = &eq_tau1[b_start..(b_start + outer_row_count)];
            for (outer_idx, eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    let matrix_row = outer_idx % n_b;
                    acc += *eq_i
                        * eval_flat_ring_at_pows(
                            &b_rows[matrix_row][local_col * d_b..(local_col + 1) * d_b],
                            &alpha_pows_b,
                        );
                }
            }
            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    let z_base: Vec<E> = cfg_into_iter!(0..z_base_len)
        .map(|k| {
            let local_k = k;
            let block_idx = local_k / depth_commit;
            let digit_idx = local_k % depth_commit;
            let a_eval = ring_multiplier_point.eval_a_at_dyn::<E>(block_idx, alpha_pows_a)?;
            let mut acc = consistency_weight * a_eval * g1_commit[digit_idx];
            for (a_idx, eq_i) in a_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i
                        * eval_flat_ring_at_pows(
                            &a_rows[a_idx][local_k * d_a..(local_k + 1) * d_a],
                            alpha_pows_a,
                        );
                }
            }
            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;

    let z_total_blocks = block_len;
    let z_segment: Vec<E> = cfg_into_iter!(0..z_len)
        .map(|x| {
            let compound_dig = x / z_total_blocks;
            let global_blk = x % z_total_blocks;
            let dc = compound_dig / depth_fold;
            let df = compound_dig % depth_fold;
            let blk = global_blk % block_len;
            let phys_k = blk * depth_commit + dc;
            -(z_base[phys_k] * fold_gadget[df])
        })
        .collect();

    let r_tail = quotient_layout.materialize_tail_weights::<F, E>(&eq_tau1, alpha)?;
    debug_assert_eq!(r_tail.len(), r_tail_len);

    // Chunked column layout `[z|e_i|t_i]…[r]`: `z` is replicated per window
    // and `e`/`t` are partitioned by global block (same per-cell values as the
    // flat segments, only repositioned).
    let blocks_per_chunk = num_blocks / num_chunks;
    for i in 0..num_chunks {
        out.extend_from_slice(&z_segment);
        let block_lo = i * blocks_per_chunk;
        for outer in w_segment.chunks_exact(num_blocks) {
            out.extend_from_slice(&outer[block_lo..block_lo + blocks_per_chunk]);
        }
        for outer in t_segment.chunks_exact(num_blocks) {
            out.extend_from_slice(&outer[block_lo..block_lo + blocks_per_chunk]);
        }
    }
    out.extend(r_tail);
    out.resize(x_len, E::zero());
    Ok(out)
}

/// Trace segment inputs for unified relation-weight construction.
#[derive(Clone)]
pub enum RelationWeightTraceBuild<F: FieldCore, E: FieldCore> {
    /// Root fold: batched public row coefficients and block openings.
    Root {
        ring_d: usize,
        num_blocks: usize,
        layout: TraceWeightLayout,
        opening_batch: OpeningClaimsLayout,
        prepared_point: PreparedOpeningPoint<F, E>,
        row_coefficients: Vec<E>,
        trace_claim_scales: Option<Vec<E>>,
    },
    /// Recursive suffix: singleton fold with scaled trace weights.
    Recursive {
        ring_d: usize,
        layout: TraceWeightLayout,
        prepared: PreparedOpeningPoint<F, E>,
        trace_scale: E,
    },
    /// Grouped root: dense trace table with per-group block geometry.
    GroupedRoot {
        ring_d: usize,
        lp: Box<LevelParams>,
        layout: TraceWeightLayout,
        opening_batch: OpeningClaimsLayout,
        prepared_points: Vec<PreparedOpeningPoint<F, E>>,
        row_coefficients: Vec<E>,
        trace_claim_scales: Option<Vec<E>>,
        live_x_cols: usize,
    },
}

/// Build the materialized relation-weight evaluation table for stage-2 sumcheck.
///
/// Fuses `M_alpha` column evaluations with the
/// [`EvaluationTrace`](akita_types::RelationRowFamily::EvaluationTrace) row via
/// `trace_weight` builders. Does not expose split `m_evals_x` / `alpha_evals_y`.
///
/// # Errors
///
/// Returns an error if matrix expansion, trace table construction, or the live
/// witness layout check fails.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "build_relation_weight_evals")]
pub fn build_relation_weight_evals<F, E>(
    setup: &AkitaExpandedSetup<F>,
    opening_point: &RingOpeningPoint<F>,
    ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
    challenges: &Challenges,
    alpha: E,
    role_dims: CommitmentRingDims,
    lp: &LevelParams,
    tau1: &[E],
    num_polys: usize,
    num_commitment_groups: usize,
    gamma: &[E],
    m_row_layout: MRowLayout,
    ring_bits: usize,
    live_x_cols: usize,
    trace: Option<RelationWeightTraceBuild<F, E>>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + ExtField<F>,
{
    let d_a = role_dims.d_a();
    let ring_alpha_evals_y = scalar_powers(alpha, d_a);
    let alpha_evals_y = scalar_powers(alpha, 1usize << ring_bits);
    let relation_column_weights = compute_relation_column_weights::<F, E>(
        setup,
        opening_point,
        ring_multiplier_point,
        challenges,
        alpha,
        &ring_alpha_evals_y,
        role_dims,
        lp,
        tau1,
        num_polys,
        num_commitment_groups,
        gamma,
        m_row_layout,
    )?;
    let y_len = alpha_evals_y.len();
    if relation_column_weights.len() < live_x_cols {
        return Err(AkitaError::InvalidSize {
            expected: live_x_cols,
            actual: relation_column_weights.len(),
        });
    }
    let trace_row_weight = {
        let eq_tau1 = EqPolynomial::evals(tau1)?;
        eq_tau1.first().copied().unwrap_or(E::zero())
    };
    let trace_dense = if let Some(trace) = trace {
        let table = match trace {
            RelationWeightTraceBuild::Root {
                ring_d,
                num_blocks,
                layout,
                opening_batch,
                prepared_point,
                row_coefficients,
                trace_claim_scales,
            } => akita_types::dispatch_ring_dim_result!(ring_d, |D| {
                let public_weights = trace_public_weights_root_terms::<F, E, D>(
                    num_blocks,
                    &opening_batch,
                    &prepared_point,
                    &row_coefficients,
                    trace_claim_scales.as_deref(),
                )?;
                build_trace_table_scaled(&layout, &public_weights, live_x_cols, E::one())
            })?,
            RelationWeightTraceBuild::Recursive {
                ring_d,
                layout,
                prepared,
                trace_scale,
            } => akita_types::dispatch_ring_dim_result!(ring_d, |D| {
                let public_weights =
                    trace_public_weights_recursive::<F, E, D>(&prepared, trace_scale)?;
                build_trace_table_scaled(&layout, &public_weights, live_x_cols, E::one())
            })?,
            RelationWeightTraceBuild::GroupedRoot {
                ring_d,
                lp,
                layout: _,
                opening_batch,
                prepared_points,
                row_coefficients,
                trace_claim_scales,
                live_x_cols: grouped_live_x_cols,
            } => akita_types::dispatch_ring_dim_result!(ring_d, |D| {
                build_grouped_root_stage2_trace_table::<F, E>(
                    ring_d,
                    lp.as_ref(),
                    &opening_batch,
                    &prepared_points,
                    &row_coefficients,
                    trace_claim_scales.as_deref(),
                    E::one(),
                    grouped_live_x_cols,
                )
            })?,
        };
        Some(
            table
                .materialize_dense(live_x_cols, y_len)
                .into_iter()
                .map(|v| v * trace_row_weight)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let trace_dense = trace_dense.as_deref();
    let mut out = Vec::with_capacity(live_x_cols * y_len);
    for (x, column_weight) in relation_column_weights
        .iter()
        .copied()
        .enumerate()
        .take(live_x_cols)
    {
        let trace_column = trace_dense.map(|trace| &trace[x * y_len..(x + 1) * y_len]);
        for (y, alpha_y) in alpha_evals_y.iter().copied().enumerate() {
            let trace = trace_column.map(|column| column[y]).unwrap_or(E::zero());
            out.push(alpha_y * column_weight + trace);
        }
    }
    Ok(out)
}

/// Build relation-weight evaluations from a full [`RingRelationInstance`].
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "build_relation_weight_evals_from_instance")]
pub fn build_relation_weight_evals_from_instance<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    role_dims: CommitmentRingDims,
    lp: &LevelParams,
    tau1: &[E],
    gamma: &[E],
    m_row_layout: MRowLayout,
    ring_bits: usize,
    live_x_cols: usize,
    trace: Option<RelationWeightTraceBuild<F, E>>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + ExtField<F>,
{
    let d_a = role_dims.d_a();
    let ring_alpha_evals_y = scalar_powers(alpha, d_a);
    let alpha_evals_y = scalar_powers(alpha, 1usize << ring_bits);
    let relation_column_weights = compute_grouped_m_evals_x::<F, E>(
        setup,
        instance,
        alpha,
        &ring_alpha_evals_y,
        role_dims,
        lp,
        tau1,
        gamma,
        m_row_layout,
    )?;
    let y_len = alpha_evals_y.len();
    if relation_column_weights.len() < live_x_cols {
        return Err(AkitaError::InvalidSize {
            expected: live_x_cols,
            actual: relation_column_weights.len(),
        });
    }
    let trace_row_weight = {
        let eq_tau1 = EqPolynomial::evals(tau1)?;
        eq_tau1.first().copied().unwrap_or(E::zero())
    };
    let trace_dense = if let Some(trace) = trace {
        let table = match trace {
            RelationWeightTraceBuild::Root {
                ring_d,
                num_blocks,
                layout,
                opening_batch,
                prepared_point,
                row_coefficients,
                trace_claim_scales,
            } => akita_types::dispatch_ring_dim_result!(ring_d, |D| {
                let public_weights = trace_public_weights_root_terms::<F, E, D>(
                    num_blocks,
                    &opening_batch,
                    &prepared_point,
                    &row_coefficients,
                    trace_claim_scales.as_deref(),
                )?;
                build_trace_table_scaled(&layout, &public_weights, live_x_cols, E::one())
            })?,
            RelationWeightTraceBuild::Recursive {
                ring_d,
                layout,
                prepared,
                trace_scale,
            } => akita_types::dispatch_ring_dim_result!(ring_d, |D| {
                let public_weights =
                    trace_public_weights_recursive::<F, E, D>(&prepared, trace_scale)?;
                build_trace_table_scaled(&layout, &public_weights, live_x_cols, E::one())
            })?,
            RelationWeightTraceBuild::GroupedRoot {
                ring_d,
                lp,
                layout: _,
                opening_batch,
                prepared_points,
                row_coefficients,
                trace_claim_scales,
                live_x_cols: grouped_live_x_cols,
            } => akita_types::dispatch_ring_dim_result!(ring_d, |D| {
                build_grouped_root_stage2_trace_table::<F, E>(
                    ring_d,
                    lp.as_ref(),
                    &opening_batch,
                    &prepared_points,
                    &row_coefficients,
                    trace_claim_scales.as_deref(),
                    E::one(),
                    grouped_live_x_cols,
                )
            })?,
        };
        Some(
            table
                .materialize_dense(live_x_cols, y_len)
                .into_iter()
                .map(|v| v * trace_row_weight)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let trace_dense = trace_dense.as_deref();
    let mut out = Vec::with_capacity(live_x_cols * y_len);
    for (x, column_weight) in relation_column_weights
        .iter()
        .copied()
        .enumerate()
        .take(live_x_cols)
    {
        let trace_column = trace_dense.map(|trace| &trace[x * y_len..(x + 1) * y_len]);
        for (y, alpha_y) in alpha_evals_y.iter().copied().enumerate() {
            let trace = trace_column.map(|column| column[y]).unwrap_or(E::zero());
            out.push(alpha_y * column_weight + trace);
        }
    }
    Ok(out)
}
