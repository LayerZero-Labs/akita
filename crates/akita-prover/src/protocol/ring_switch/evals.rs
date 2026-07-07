use super::*;
use crate::protocol::ring_relation::validate_chunked_witness_cfg;
use akita_algebra::ring::{eval_flat_ring_at_pows, scalar_powers};
use akita_types::{
    build_trace_table_scaled, trace_public_weights_recursive, trace_public_weights_root_terms,
    CommitmentRingDims, ConsistencyLayer, OpeningClaimsLayout, PreparedOpeningPoint,
    RelationQuotientLayout, RelationRowFamily, RelationRowLayout, TraceWeightLayout,
    FOLD_CONSISTENCY_ROW,
};

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
}

/// Accumulate per-column relation-family weights into the flat live witness
/// coefficient range.
///
/// Each live flat index is `segment * witness_coeff_len + coeff` with
/// `witness_coeff_len = role_dims.d_a()`. Coefficients past the live segment
/// count or past `witness_coeff_len` are not stored (implicit zero).
fn accumulate_flat_column_family<E: FieldCore>(
    out: &mut [E],
    relation_column_weights: &[E],
    live_segments: usize,
    witness_coeff_len: usize,
    witness_coeff_powers: &[E],
) -> Result<(), AkitaError> {
    if relation_column_weights.len() < live_segments {
        return Err(AkitaError::InvalidSize {
            expected: live_segments,
            actual: relation_column_weights.len(),
        });
    }
    let witness_live_len = live_segments
        .checked_mul(witness_coeff_len)
        .ok_or_else(|| AkitaError::InvalidSetup("witness live length overflow".into()))?;
    if out.len() != witness_live_len {
        return Err(AkitaError::InvalidSize {
            expected: witness_live_len,
            actual: out.len(),
        });
    }
    if witness_coeff_powers.len() != witness_coeff_len {
        return Err(AkitaError::InvalidSize {
            expected: witness_coeff_len,
            actual: witness_coeff_powers.len(),
        });
    }
    for (segment, column_weight) in relation_column_weights
        .iter()
        .copied()
        .enumerate()
        .take(live_segments)
    {
        let flat_base = segment * witness_coeff_len;
        for (coeff, coeff_power) in witness_coeff_powers.iter().copied().enumerate() {
            out[flat_base + coeff] += coeff_power * column_weight;
        }
    }
    Ok(())
}

fn accumulate_flat_trace_family<E: FieldCore>(
    out: &mut [E],
    trace_dense: &[E],
    trace_row_weight: E,
) -> Result<(), AkitaError> {
    if trace_dense.len() != out.len() {
        return Err(AkitaError::InvalidSize {
            expected: out.len(),
            actual: trace_dense.len(),
        });
    }
    for (dst, trace) in out.iter_mut().zip(trace_dense.iter().copied()) {
        *dst += trace_row_weight * trace;
    }
    Ok(())
}

/// Build the materialized relation-weight evaluation table for stage-2 sumcheck.
///
/// Walks relation row families through [`compute_relation_column_weights`]
/// (per-role ring dimensions for A/B/D matrix rows and quotient slices) and
/// embeds the result into the flat live witness range `0..witness_live_len`
/// with `witness_coeff_len = role_dims.d_a()`.
///
/// Fuses the [`EvaluationTrace`](akita_types::RelationRowFamily::EvaluationTrace)
/// row via `trace_weight` builders when `trace` is present.
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
    witness_live_len: usize,
    trace: Option<RelationWeightTraceBuild<F, E>>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Invertible,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + ExtField<F>,
{
    let witness_coeff_len = role_dims.d_a();
    if witness_coeff_len == 0 || !witness_live_len.is_multiple_of(witness_coeff_len) {
        return Err(AkitaError::InvalidSize {
            expected: witness_coeff_len,
            actual: witness_live_len,
        });
    }
    let live_segments = witness_live_len / witness_coeff_len;
    let witness_coeff_powers = scalar_powers(alpha, witness_coeff_len);
    let quotient_family_column_weights = compute_relation_column_weights::<F, E>(
        setup,
        opening_point,
        ring_multiplier_point,
        challenges,
        alpha,
        &witness_coeff_powers,
        role_dims,
        lp,
        tau1,
        num_polys,
        num_commitment_groups,
        gamma,
        m_row_layout,
    )?;
    let mut out = vec![E::zero(); witness_live_len];
    accumulate_flat_column_family(
        &mut out,
        &quotient_family_column_weights,
        live_segments,
        witness_coeff_len,
        &witness_coeff_powers,
    )?;
    let trace_row_weight = {
        let eq_tau1 = EqPolynomial::evals(tau1)?;
        eq_tau1.first().copied().unwrap_or(E::zero())
    };
    if let Some(trace) = trace {
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
                build_trace_table_scaled(&layout, &public_weights, live_segments, E::one())
            })?,
            RelationWeightTraceBuild::Recursive {
                ring_d,
                layout,
                prepared,
                trace_scale,
            } => akita_types::dispatch_ring_dim_result!(ring_d, |D| {
                let public_weights =
                    trace_public_weights_recursive::<F, E, D>(&prepared, trace_scale)?;
                build_trace_table_scaled(&layout, &public_weights, live_segments, E::one())
            })?,
        };
        let trace_dense = table.materialize_dense(live_segments, witness_coeff_len);
        accumulate_flat_trace_family(&mut out, &trace_dense, trace_row_weight)?;
    }
    Ok(out)
}

#[cfg(test)]
mod materialization_tests {
    use super::{accumulate_flat_column_family, accumulate_flat_trace_family};
    use akita_algebra::ring::scalar_powers;
    use akita_field::{FpExt2, NegOneNr, Prime128Offset275};
    use akita_types::CommitmentRingDims;

    type F = Prime128Offset275;
    type E = FpExt2<F, NegOneNr>;

    #[test]
    fn flat_materialization_uses_witness_coeff_len_from_d_a() {
        let witness_coeff_len = 128usize;
        let live_segments = 3usize;
        let witness_live_len = live_segments * witness_coeff_len;
        let alpha = E::from_u64(5);
        let witness_coeff_powers = scalar_powers(alpha, witness_coeff_len);
        let column_weights: Vec<E> = (0..live_segments)
            .map(|i| E::from_u64((i + 1) as u64))
            .collect();
        let mut evals = vec![E::zero(); witness_live_len];
        accumulate_flat_column_family(
            &mut evals,
            &column_weights,
            live_segments,
            witness_coeff_len,
            &witness_coeff_powers,
        )
        .expect("materialize");
        assert_eq!(evals.len(), witness_live_len);
        assert_eq!(
            evals[0],
            witness_coeff_powers[0] * column_weights[0],
            "first flat slot is segment 0 coeff 0"
        );
        assert_eq!(
            evals[witness_coeff_len],
            witness_coeff_powers[0] * column_weights[1],
            "segment 1 starts at coeff_len offset"
        );
    }

    #[test]
    fn nested_role_dims_live_len_is_segments_times_d_a() {
        let dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        };
        assert!(dims.nests());
        let witness_coeff_len = dims.d_a();
        let live_segments = 4usize;
        let witness_live_len = live_segments * witness_coeff_len;
        let witness_coeff_powers = vec![E::one(); witness_coeff_len];
        let column_weights = vec![E::from_u64(2); live_segments];
        let mut evals = vec![E::zero(); witness_live_len];
        accumulate_flat_column_family(
            &mut evals,
            &column_weights,
            live_segments,
            witness_coeff_len,
            &witness_coeff_powers,
        )
        .expect("nested role dims materialize");
        assert_eq!(evals.len(), witness_live_len);
        assert_eq!(evals[witness_coeff_len + 1], column_weights[1]);
    }

    #[test]
    fn rejects_trace_length_mismatch() {
        let mut out = vec![E::zero(); 8];
        let err = accumulate_flat_trace_family(&mut out, &[E::zero(); 7], E::one())
            .expect_err("trace length mismatch");
        assert!(matches!(err, akita_field::AkitaError::InvalidSize { .. }));
    }
}
