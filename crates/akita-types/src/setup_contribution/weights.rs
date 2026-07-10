use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{eq_eval_at_index, high_eq_window};
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, MulBase};

const POSSIBLE_CARRIES: usize = 2;

/// Canonical `D·ê` column eq-weights, shared by the single-group and multi-group setup
/// builders. Produces `num_claims * num_blocks * depth_open` weights, with a
/// per-chunk high-eq window based at each chunk's `ê` offset. The `ê` block is
/// setup-shared across commitment groups, so callers pass the same scalars
/// regardless of group layout.
#[inline(always)]
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
    let high_len = num_claims
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("D high width overflow".into()))?;
    let eq_high_by_chunk: Vec<Vec<E>> = {
        let _span = tracing::info_span!(
            "setup_e_eq_high_by_chunk",
            num_chunks = chunks.len(),
            high_len,
            table_len = high_len + 1,
            total_len = chunks.len() * (high_len + 1)
        )
        .entered();
        cfg_iter!(chunks)
            .map(|chunk| high_eq_window(high_challenges, chunk.offset_e >> block_bits, high_len))
            .collect()
    };
    let low_mask = blocks_per_chunk - 1;
    let total_blocks = num_claims
        .checked_mul(num_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("D blocks overflow".into()))?;
    let e_cols = total_blocks
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("D columns overflow".into()))?;
    let _scan_span =
        tracing::info_span!("setup_e_col_scan", e_cols, num_chunks = chunks.len()).entered();
    // One `divmod` per block (not per element): `chunk_idx`/`block_idx` reduce to
    // shift/mask since `blocks_per_chunk` is a power of two, and the inner `digit`
    // loop is a per-block constant `eq_low[low_idx]` times a stride-`num_claims`
    // walk through `eq_high`.
    let mut out = vec![E::zero(); e_cols];
    // `chunks_mut(0)` panics unconditionally; `depth_open == 0` means no columns
    // (`e_cols == 0`), so the empty `out` is already the correct result. Guards the
    // verifier no-panic boundary even though callers validate `depth_open > 0`.
    if depth_open == 0 {
        return Ok(out);
    }
    cfg_chunks_mut!(out, depth_open)
        .enumerate()
        .for_each(|(flat_block, dst)| {
            let claim_idx = flat_block / num_blocks;
            let global_block_idx = flat_block % num_blocks;
            let chunk_idx = global_block_idx >> block_bits;
            let block_idx = global_block_idx & low_mask;
            let chunk = &chunks[chunk_idx];
            let eq_high = &eq_high_by_chunk[chunk_idx];
            let offset_low = chunk.offset_e & low_mask;
            let shifted = offset_low + block_idx;
            let low_idx = shifted & low_mask;
            let carry = shifted >> block_bits;
            let low_factor = eq_low[low_idx];
            let mut high_idx = claim_idx + carry;
            for slot in dst.iter_mut() {
                *slot = low_factor * eq_high[high_idx];
                high_idx += num_claims;
            }
        });
    Ok(out)
}

/// Canonical `B·t̂` column eq-weights, shared by the single-group and multi-group setup
/// builders. Emits `num_vectors * cols_per_vector` weights; columns for a
/// t-vector index `>= active_vectors` are zero (flat zero-padding to the widest
/// group). The high-eq axis is addressed as `(vector_base + vector_idx)` with
/// stride `high_vector_stride`: the single-group path packs all groups' t-vectors
/// into one high axis (`vector_base = group offset`, stride = total t-vectors),
/// while the multi-group builder is per-group (`vector_base = 0`, stride = the
/// group's t-vector count).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
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
    let high_len = high_vector_stride
        .checked_mul(depth_open)
        .and_then(|width| width.checked_mul(n_a))
        .ok_or_else(|| AkitaError::InvalidSetup("B high width overflow".into()))?;
    let eq_high_by_chunk: Vec<Vec<E>> = {
        let _span = tracing::info_span!(
            "setup_t_eq_high_by_chunk",
            num_chunks = chunks.len(),
            high_len,
            table_len = high_len + 1,
            total_len = chunks.len() * (high_len + 1)
        )
        .entered();
        cfg_iter!(chunks)
            .map(|chunk| high_eq_window(high_challenges, chunk.offset_t >> block_bits, high_len))
            .collect()
    };
    let low_mask = blocks_per_chunk - 1;
    let t_compound_per_block = n_a
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("B compound stride overflow".into()))?;
    let t_cols = num_vectors
        .checked_mul(cols_per_vector)
        .ok_or_else(|| AkitaError::InvalidSetup("B width overflow".into()))?;
    let _scan_span =
        tracing::info_span!("setup_t_col_scan", t_cols, num_chunks = chunks.len()).entered();

    if t_compound_per_block == 0 || !cols_per_vector.is_multiple_of(t_compound_per_block) {
        return Err(AkitaError::InvalidSetup(
            "setup T weights require cols_per_vector to be a nonzero multiple of n_a * depth_open"
                .into(),
        ));
    }
    let blocks_per_vector = cols_per_vector / t_compound_per_block;
    let mut out = vec![E::zero(); t_cols];
    cfg_chunks_mut!(out, t_compound_per_block)
        .enumerate()
        .for_each(|(flat_block, dst)| {
            let vector_idx = flat_block / blocks_per_vector;
            if vector_idx >= active_vectors {
                return;
            }
            let global_block_idx = flat_block % blocks_per_vector;
            let chunk_idx = global_block_idx >> block_bits;
            let block_idx = global_block_idx & low_mask;
            let chunk = &chunks[chunk_idx];
            let eq_high = &eq_high_by_chunk[chunk_idx];
            let offset_low = chunk.offset_t & low_mask;
            let shifted = offset_low + block_idx;
            let low_idx = shifted & low_mask;
            let carry = shifted >> block_bits;
            let low_factor = eq_low[low_idx];
            let mut high_idx = vector_base + vector_idx + carry;
            for slot in dst.iter_mut() {
                *slot = low_factor * eq_high[high_idx];
                high_idx += high_vector_stride;
            }
        });
    Ok(out)
}

/// Column weights for the A-row setup term `A * G_fold * z_hat`.
///
/// For the A column `k = blk * depth_commit + dc`, this function adds
///
/// ```text
/// z_weights[k] -=
///   sum_chunk sum_pt sum_df
///     G_fold[df] *
///     eq_x(chunk.offset_z + blk
///          + block_len * (pt + num_fold_groups * (df + depth_fold * dc))).
/// ```
///
/// The commit gadget `G_commit` is not present here. The matrix `A` is already
/// indexed by commit digits `(blk, dc)`, so this setup term is exactly
/// `A * G_fold * z_hat`. The opening row is the separate term that uses
/// `G_commit * G_fold`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn setup_z_col_weights<F, E>(
    chunks: &[crate::WitnessChunkLayout],
    block_len: usize,
    depth_commit: usize,
    depth_fold: usize,
    num_fold_groups: usize,
    full_vec_randomness: &[E],
    z_block_low_eq: Option<&[E]>,
    fold_gadget: &[F],
    z_weights: &mut [E],
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: MulBase<F>,
{
    if chunks.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "setup Z weights require at least one witness chunk".into(),
        ));
    }
    let z_range = block_len
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("setup Z range overflow".into()))?;
    if z_weights.len() != z_range {
        return Err(AkitaError::InvalidSize {
            expected: z_range,
            actual: z_weights.len(),
        });
    }
    // `depth_commit == 0` yields `z_range == 0` (empty `z_weights`) but would panic
    // both branches below: `chunks_mut(0)` in the pow2 path and `c % depth_commit`
    // (div-by-zero) in the dense path. Nothing to accumulate, so return early.
    if depth_commit == 0 {
        return Ok(());
    }
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
        let high_len = depth_commit
            .checked_mul(depth_fold)
            .and_then(|width| width.checked_mul(num_fold_groups))
            .ok_or_else(|| AkitaError::InvalidSetup("Z high width overflow".into()))?;
        let low_mask = block_len - 1;
        let chunk_summaries: Vec<(usize, Vec<[E; POSSIBLE_CARRIES]>)> = cfg_iter!(chunks)
            .map(|chunk| {
                let eq_high = high_eq_window(high_challenges, chunk.offset_z >> z_bits, high_len);
                let s_per_dc_per_carry = (0..depth_commit)
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
                (chunk.offset_z & low_mask, s_per_dc_per_carry)
            })
            .collect();
        // One task per commit block (mirrors the D·ê / B·t̂ scans): the per-chunk
        // `low_idx`/`carry`/`eq_low[low_idx]` depend only on `block_idx`, so we hoist
        // them out of the inner `dc` walk instead of recomputing them for every
        // `(block_idx, dc)` element.
        cfg_chunks_mut!(z_weights, depth_commit)
            .enumerate()
            .for_each(|(block_idx, dst)| {
                for (offset_low, s_per_dc_per_carry) in &chunk_summaries {
                    let shifted = *offset_low + block_idx;
                    let low_idx = shifted & low_mask;
                    let carry = shifted >> z_bits;
                    let low_factor = eq_low[low_idx];
                    for (slot, s) in dst.iter_mut().zip(s_per_dc_per_carry) {
                        *slot += low_factor * s[carry];
                    }
                }
            });
        Ok(())
    } else {
        let z_depth = depth_fold
            .checked_mul(depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("dense Z length overflow".into()))?;
        let z_block_width = block_len
            .checked_mul(num_fold_groups)
            .ok_or_else(|| AkitaError::InvalidSetup("dense Z block width overflow".into()))?;
        let z_len = z_depth
            .checked_mul(z_block_width)
            .ok_or_else(|| AkitaError::InvalidSetup("dense Z length overflow".into()))?;
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
        let chunk_tables: Vec<(usize, Vec<E>)> = cfg_iter!(chunks)
            .map(|chunk| {
                let offset_low = chunk.offset_z & low_mask;
                let offset_high = chunk.offset_z >> low_bits;
                let max_high = chunk
                    .offset_z
                    .checked_add(z_len)
                    .ok_or_else(|| AkitaError::InvalidSetup("dense Z end overflow".into()))?
                    .checked_sub(1)
                    .ok_or(AkitaError::InvalidProof)?
                    >> low_bits;
                let eq_high = (offset_high..=max_high)
                    .map(|idx| eq_eval_at_index(&full_vec_randomness[low_bits..], idx))
                    .collect();
                Ok((offset_low, eq_high))
            })
            .collect::<Result<_, AkitaError>>()?;
        cfg_iter_mut!(z_weights)
            .enumerate()
            .try_for_each(|(k, dst)| {
                let block_idx = k / depth_commit;
                let dc = k % depth_commit;
                let mut weight = E::zero();
                for (offset_low, eq_high) in &chunk_tables {
                    for pt in 0..num_fold_groups {
                        for (df, &fold) in fold_gadget.iter().enumerate().take(depth_fold) {
                            let x = block_idx
                                + block_len * (pt + num_fold_groups * (df + depth_fold * dc));
                            let shifted = *offset_low + x;
                            let low_idx = shifted & low_mask;
                            let high_carry = shifted >> low_bits;
                            weight -= (eq_low[low_idx] * eq_high[high_carry]).mul_base(fold);
                        }
                    }
                }
                *dst += weight;
                Ok(())
            })
    }
}
