use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{eq_eval_at_index, high_eq_window};
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, MulBase};

use super::{checked_add, checked_mul};

const POSSIBLE_CARRIES: usize = 2;

/// Canonical `D·ê` column eq-weights, shared by the single-group and multi-group setup
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

/// Canonical `B·t̂` column eq-weights, shared by the single-group and multi-group setup
/// builders. Emits `num_vectors * cols_per_vector` weights; columns for a
/// t-vector index `>= active_vectors` are zero (flat zero-padding to the widest
/// group). The high-eq axis is addressed as `(vector_base + vector_idx)` with
/// stride `high_vector_stride`: the single-group path packs all groups' t-vectors
/// into one high axis (`vector_base = group offset`, stride = total t-vectors),
/// while the multi-group builder is per-group (`vector_base = 0`, stride = the
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
/// `offset_z`, shared by the single-group and multi-group setup builders. Callers sum the
/// result over chunks into `Z_comb`. `num_fold_groups` is the number of
/// point-blocks folded into a single `ẑ` slice: the single-group path folds all
/// commitment groups (`num_fold_groups = num_groups`), the multi-group builder is
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
