use super::*;
use crate::protocol::ring_relation::validate_chunked_witness_cfg;

/// Produce the compact `Vec<i8>` eval table of `w` for the fused prover.
///
/// The compact witness stays in the raw [`build_w_coeffs`] order:
/// `w[x * y_len + y]`, with x outer and y inner.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by the ring
/// dimension.
pub fn build_w_evals_compact(
    w: &[i8],
    d: usize,
    extension_degree: usize,
) -> Result<(Vec<i8>, usize, usize), AkitaError> {
    if !w.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let live_x_cols = w.len() / d;
    let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
    if extension_degree == 1 {
        let ring_bits = d.trailing_zeros() as usize;
        return Ok((w.to_vec(), col_bits, ring_bits));
    }
    let packed_len = d / extension_degree;
    if packed_len == 0 || !packed_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "packed recursive witness has invalid slot count".to_string(),
        ));
    }
    let half = d / (2 * extension_degree);
    let mut compact = Vec::with_capacity(live_x_cols * packed_len);
    for ring in w.chunks_exact(d) {
        compact.extend_from_slice(&ring[..half]);
        compact.extend((half..packed_len).map(|low| ring[d / 2 + low - half]));
    }
    Ok((compact, col_bits, packed_len.trailing_zeros() as usize))
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
#[tracing::instrument(skip_all, name = "compute_m_evals_x_batched")]
pub fn compute_m_evals_x<F, E, const D: usize>(
    setup: &AkitaExpandedSetup<F>,
    opening_point: &RingOpeningPoint<F>,
    ring_multiplier_point: &RingMultiplierOpeningPoint<F, D>,
    challenges: &Challenges,
    alpha: E,
    alpha_pows: &[E],
    lp: &LevelParams,
    tau1: &[E],
    num_polys: usize,
    gamma: &[E],
    m_row_layout: MRowLayout,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F>,
{
    if alpha_pows.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }
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
    let segment_lengths = ring_relation_segment_lengths::<F, D>(
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
    let depth_fold = lp.num_digits_fold(num_claims, F::modulus_bits())?;
    let inner_width = block_len * depth_commit;
    let z_base_len = inner_width;
    let n_a = lp.a_key.row_len();
    let n_b = lp.b_key.row_len();
    let n_d = lp.d_key.row_len();
    // Terminal layout drops the D-block from the M-matrix entirely; offsets
    // and per-row gates must use 0 for the n_d position.
    let n_d_active = match m_row_layout {
        MRowLayout::WithDBlock => n_d,
        MRowLayout::WithoutDBlock | MRowLayout::WithoutCommitmentBlocks => 0,
    };
    let rows = lp.m_row_count_for(1, m_row_layout)?;
    let levels = r_decomp_levels::<F>(log_basis);
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
        .and_then(|cols| cols.checked_add(rows.checked_mul(levels)?))
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
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, log_basis)
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
            .map(|challenge| challenge.eval_at_pows::<F, E, D>(alpha_pows))
            .collect::<Result<_, _>>()?,
        Challenges::Tensor { factored: _ } => challenges.evals_at_pows::<F, E, D>(alpha_pows)?,
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
    let d_view = setup.shared_matrix.ring_view::<D>(n_d, d_width)?;
    let b_view = setup.shared_matrix.ring_view::<D>(n_b, b_width)?;
    let a_view = setup.shared_matrix.ring_view::<D>(n_a, a_width)?;
    let d_rows: Vec<_> = d_view.rows().collect();
    let b_rows: Vec<_> = b_view.rows().collect();
    let a_rows: Vec<_> = a_view.rows().collect();

    // Canonical row layout: consistency (1) | A | B | D.
    let a_start = lp.a_start();
    let b_start = lp.b_start()?;
    let d_start = lp.d_start(1)?;
    let a_weights = &eq_tau1[a_start..(a_start + n_a)];
    let consistency_weight = eq_tau1[0];
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
                    acc += *eq_i * eval_ring_at_pows(&d_rows[di][d_phys_col], alpha_pows);
                }
            }
            acc
        })
        .collect();

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
            let commitment_weights = &eq_tau1[b_start..(b_start + n_b)];
            for (row_idx, eq_i) in commitment_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&b_rows[row_idx][local_col], alpha_pows);
                }
            }
            acc
        })
        .collect();

    let z_base: Vec<E> = cfg_into_iter!(0..z_base_len)
        .map(|k| {
            let local_k = k;
            let block_idx = local_k / depth_commit;
            let digit_idx = local_k % depth_commit;
            let a_eval = ring_multiplier_point.eval_a_at::<E>(block_idx, alpha_pows)?;
            let mut acc = consistency_weight * a_eval * g1_commit[digit_idx];
            for (a_idx, eq_i) in a_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += *eq_i * eval_ring_at_pows(&a_rows[a_idx][local_k], alpha_pows);
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

    let alpha_pow_d = alpha_pows[D - 1] * alpha;
    let denom = alpha_pow_d + E::one();
    let r_tail_len = rows * levels;
    let r_tail: Vec<E> = cfg_into_iter!(0..r_tail_len)
        .map(|idx| {
            let row_idx = idx / levels;
            let level_idx = idx % levels;
            -(eq_tau1[row_idx] * denom * r_gadget[level_idx])
        })
        .collect();

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
