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
    let eq_high_by_chunk: Vec<Vec<E>> = chunks
        .iter()
        .map(|chunk| high_eq_window(high_challenges, chunk.offset_e >> block_bits, high_len))
        .collect();
    let low_mask = blocks_per_chunk - 1;
    let total_blocks = num_claims
        .checked_mul(num_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("D blocks overflow".into()))?;
    let e_cols = total_blocks
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("D columns overflow".into()))?;
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
    let eq_high_by_chunk: Vec<Vec<E>> = chunks
        .iter()
        .map(|chunk| high_eq_window(high_challenges, chunk.offset_t >> block_bits, high_len))
        .collect();
    let low_mask = blocks_per_chunk - 1;
    let t_compound_per_block = n_a
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("B compound stride overflow".into()))?;
    let t_cols = num_vectors
        .checked_mul(cols_per_vector)
        .ok_or_else(|| AkitaError::InvalidSetup("B width overflow".into()))?;
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

/// Canonical `A·ẑ` column eq-weights for chunk-replicated `ẑ`, shared by the
/// single-group and multi-group setup builders. `num_fold_groups` is the number of
/// point-blocks folded into a single `ẑ` slice: the single-group path folds all
/// commitment groups (`num_fold_groups = num_groups`), the multi-group builder is
/// per-group (`num_fold_groups = 1`). Handles both the power-of-two `block_len`
/// fast path and the dense fallback.
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
        let chunk_summaries: Vec<(usize, Vec<[E; POSSIBLE_CARRIES]>)> = chunks
            .iter()
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
        cfg_iter_mut!(z_weights)
            .enumerate()
            .try_for_each(|(k, dst)| {
                let block_idx = k / depth_commit;
                let dc = k % depth_commit;
                let mut weight = E::zero();
                for (offset_low, s_per_dc_per_carry) in &chunk_summaries {
                    let shifted = *offset_low + block_idx;
                    let low_idx = shifted & low_mask;
                    let carry = shifted >> z_bits;
                    weight += eq_low[low_idx] * s_per_dc_per_carry[dc][carry];
                }
                *dst += weight;
                Ok(())
            })
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
        let chunk_tables: Vec<(usize, Vec<E>)> = chunks
            .iter()
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
