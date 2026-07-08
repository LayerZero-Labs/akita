//! Shared setup-contribution planning for prover and verifier.
//!
//! This module owns the pure layout/weight derivation for the stage-3 setup
//! product. The prover consumes the materialized `bar_omega` vector, while the
//! verifier can evaluate the same plan directly against the packed setup.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{eq_eval_at_index, high_eq_window};
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, MulBase};

use crate::layout::{LevelParams, RelationMatrixRowLayout};
const POSSIBLE_CARRIES: usize = 2;

#[path = "setup_contribution_plan.rs"]
mod setup_contribution_plan;

pub use setup_contribution_plan::{
    SetupContributionGroupInputs, SetupContributionPlan, SetupContributionStatic,
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
    pub relation_matrix_row_layout: RelationMatrixRowLayout,
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
        relation_matrix_row_layout: RelationMatrixRowLayout,
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
        let rows = lp.relation_matrix_row_count_for(num_groups, relation_matrix_row_layout)?;
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
            relation_matrix_row_layout,
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
    use super::setup_contribution_plan::SetupContributionGroupPlan;
    use super::*;
    use crate::{
        gadget_row_scalars, AkitaExpandedSetup, AkitaSetupSeed, FlatMatrix, RelationMatrixRowLayout,
    };
    use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
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
            relation_matrix_row_layout: RelationMatrixRowLayout::WithoutDBlock,
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
        let plan = SetupContributionPlan::prepare_single_group::<F>(
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

        assert_eq!(plan.groups[0].z_eq_slice, expected);
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
            relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
            n_b,
            num_groups: 1,
            rows,
            num_polys_per_group: vec![num_claims],
        };
        let full_vec_randomness = (0..10)
            .map(|idx| test_scalar(101 + idx as u128))
            .collect::<Vec<_>>();
        let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
        let plan = SetupContributionPlan::prepare::<F>(
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

        let setup_len = plan.required().unwrap();
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
        let expected = plan
            .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, 1)
            .unwrap();
        let got = plan
            .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
            .unwrap();
        assert_eq!(got, expected);

        let bar_omega = plan.materialize_bar_omega().unwrap();
        let setup_view = setup
            .shared_matrix()
            .ring_view::<1>(1, bar_omega.len())
            .unwrap();
        let tie: F = bar_omega
            .iter()
            .zip(setup_view.as_slice())
            .map(|(w, ring)| eval_ring_at_pows(ring, &alpha_pows) * *w)
            .sum();
        assert_eq!(tie, got);
    }

    #[test]
    fn grouped_packed_direct_matches_row_fallback_with_d_offset() {
        let grouped_plan = SetupContributionPlan {
            d_rows: 2,
            d_physical_cols: 5,
            groups: vec![SetupContributionGroupPlan {
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
        let grouped_plan = SetupContributionPlan {
            d_rows: 2,
            d_physical_cols: 5,
            groups: vec![
                SetupContributionGroupPlan {
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
                SetupContributionGroupPlan {
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

        let bar_omega = grouped_plan.materialize_bar_omega().unwrap();
        let setup_view = setup
            .shared_matrix()
            .ring_view::<1>(1, bar_omega.len())
            .unwrap();
        let tie: F = bar_omega
            .iter()
            .zip(setup_view.as_slice())
            .map(|(w, ring)| eval_ring_at_pows(ring, &alpha_pows) * *w)
            .sum();
        assert_eq!(tie, got);
    }

    #[test]
    fn grouped_packed_direct_matches_row_fallback_with_nested_role_dims() {
        let grouped_plan = SetupContributionPlan {
            d_rows: 2,
            d_physical_cols: 5,
            groups: vec![SetupContributionGroupPlan {
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
    fn grouped_packed_direct_accepts_d_footprint_at_nested_d_d() {
        // D-role columns are counted at d_d; comparing `required` against
        // total_ring_elements_at_dyn(d_a) falsely rejects valid setups when
        // d_d < d_a and the D footprint dominates.
        let grouped_plan = SetupContributionPlan {
            d_rows: 2,
            d_physical_cols: 11,
            groups: vec![SetupContributionGroupPlan {
                e_col_offset: 0,
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
        const D_A: usize = 64;
        const D_B: usize = 64;
        const D_D: usize = 32;
        let setup_ring_elements = 20usize;
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 0,
                max_num_batched_polys: 0,
                gen_ring_dim: D_A,
                max_setup_len: setup_ring_elements,
                public_matrix_seed: [0u8; 32],
            },
            FlatMatrix::from_flat_data(
                (0..setup_ring_elements * D_A)
                    .map(|idx| test_scalar(311 + idx as u128))
                    .collect(),
                D_A,
            ),
        );
        let alpha = test_scalar(3);
        let alpha_pows_a = scalar_powers(alpha, D_A);
        let alpha_pows_b = scalar_powers(alpha, D_B);
        let alpha_pows_d = scalar_powers(alpha, D_D);
        let expected = grouped_plan
            .evaluate_direct_by_rows::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d, D_A)
            .unwrap();
        let got = grouped_plan
            .evaluate_direct::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d)
            .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn grouped_multi_group_packed_matches_row_fallback_with_mismatched_t_cols() {
        let grouped_plan = SetupContributionPlan {
            d_rows: 2,
            d_physical_cols: 5,
            groups: vec![
                SetupContributionGroupPlan {
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
                SetupContributionGroupPlan {
                    e_col_offset: 0,
                    t_cols: 6,
                    z_cols: 3,
                    n_a: 2,
                    n_b: 2,
                    e_eq_slice: vec![test_scalar(53), test_scalar(59)],
                    t_eq_slice: vec![
                        test_scalar(61),
                        test_scalar(67),
                        test_scalar(71),
                        test_scalar(73),
                        test_scalar(79),
                        test_scalar(83),
                    ],
                    z_eq_slice: vec![test_scalar(89), test_scalar(97), test_scalar(101)],
                    a_weights: vec![test_scalar(103), test_scalar(107)],
                    b_weights: vec![test_scalar(109), test_scalar(113)],
                    d_weights: vec![test_scalar(127), test_scalar(131)],
                },
            ],
        };
        let setup_len = 12;
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
            RelationMatrixRowLayout::WithoutDBlock,
            2,
        )
        .is_err());
    }
}
