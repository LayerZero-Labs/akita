use super::*;
use crate::compute::{OperationCtx, RuntimeRingSwitchProveBackend};
use crate::protocol::ring_relation::validate_chunked_witness_cfg;
use crate::protocol::ring_relation_witness::{RingRelationGroupWitness, RingRelationWitness};
use crate::validation::validate_i8_setup_log_basis;
use akita_algebra::CyclotomicRing;
use akita_serialization::AkitaSerialize;
use akita_types::{
    dispatch_for_field, LevelParamsLike, RingRole, GROUPED_ROOT_MULTI_CHUNK_UNSUPPORTED,
};

/// Prover-side ring artifacts retained for segment-typed terminal encoding.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// closures borrow typed ring rows via [`Self::e_folded_trusted`],
/// [`Self::recomposed_block_trusted`], [`Self::z_folded_centered_trusted`], and
/// [`Self::r_trusted`].
pub struct RingSwitchTerminalArtifacts<F: FieldCore> {
    pub e_folded: RingVec<F>,
    pub recomposed_inner_rows: Vec<RingVec<F>>,
    z_folded_centered_flat: Vec<i32>,
    pub r: RingVec<F>,
    ring_dim: usize,
}

impl<F: FieldCore> RingSwitchTerminalArtifacts<F> {
    /// Construct from typed ring-switch output at a kernel boundary.
    pub fn from_parts<const D: usize>(
        e_folded: Vec<CyclotomicRing<F, D>>,
        recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
        z_folded_centered: Vec<[i32; D]>,
        r: Vec<CyclotomicRing<F, D>>,
    ) -> Self {
        Self {
            e_folded: RingVec::from_ring_elems(&e_folded),
            recomposed_inner_rows: recomposed_inner_rows
                .into_iter()
                .map(|block| RingVec::from_ring_elems(&block))
                .collect(),
            z_folded_centered_flat: z_folded_centered
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect(),
            r: RingVec::from_ring_elems(&r),
            ring_dim: D,
        }
    }

    /// Stored ring dimension (coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Flat centered fold-response coefficients (`ring_dim` field elements per row).
    pub fn z_folded_centered_flat(&self) -> &[i32] {
        &self.z_folded_centered_flat
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "ring switch terminal artifacts ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        if !self.z_folded_centered_flat.len().is_multiple_of(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.z_folded_centered_flat.len(),
            });
        }
        if !self.e_folded.can_decode_vec(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.e_folded.coeff_len(),
            });
        }
        if !self.r.can_decode_vec(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.r.coeff_len(),
            });
        }
        for block in &self.recomposed_inner_rows {
            if !block.can_decode_vec(D) {
                return Err(AkitaError::InvalidSize {
                    expected: D,
                    actual: block.coeff_len(),
                });
            }
        }
        Ok(())
    }

    /// Borrow folded `e` rows after [`Self::ensure_ring_dim`].
    pub fn e_folded_trusted<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        Ok(self.e_folded.as_ring_slice_trusted::<D>())
    }

    /// Borrow recomposed rows for one block after [`Self::ensure_ring_dim`].
    pub fn recomposed_block_trusted<const D: usize>(
        &self,
        block: usize,
    ) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        self.recomposed_inner_rows
            .get(block)
            .ok_or_else(|| {
                AkitaError::InvalidInput(format!(
                    "ring switch terminal artifacts block index {block} out of range"
                ))
            })
            .map(|rows| rows.as_ring_slice_trusted::<D>())
    }

    /// Borrow centered coefficient rows after [`Self::ensure_ring_dim`].
    pub fn z_folded_centered_trusted<const D: usize>(&self) -> Result<&[[i32; D]], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        let (chunks, rem) = self.z_folded_centered_flat.as_chunks::<D>();
        debug_assert!(rem.is_empty());
        Ok(chunks)
    }

    /// Borrow relation quotient `r` rows after [`Self::ensure_ring_dim`].
    pub fn r_trusted<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        self.ensure_ring_dim::<D>()?;
        Ok(self.r.as_ring_slice_trusted::<D>())
    }
}

/// Output of [`ring_switch_build_w`].
pub struct RingSwitchBuildOutput<F: FieldCore> {
    pub w: RecursiveWitnessFlat,
    pub terminal_artifacts: Option<RingSwitchTerminalArtifacts<F>>,
}

pub(crate) struct PreparedRingSwitchGroup<'a, F: FieldCore, const D: usize> {
    pub(crate) params: &'a dyn LevelParamsLike,
    pub(crate) e_hat: DigitBlocks,
    pub(crate) t_hat: DigitBlocks,
    pub(crate) recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    pub(crate) e_folded: Vec<CyclotomicRing<F, D>>,
    pub(crate) z_centered: Vec<[i32; D]>,
    pub(crate) z_inf: u32,
    pub(crate) z_folded_centered_per_chunk: Vec<Vec<[i32; D]>>,
}

fn concat_digit_blocks(blocks: &[DigitBlocks]) -> Result<DigitBlocks, AkitaError> {
    let Some(first) = blocks.first() else {
        return Err(AkitaError::InvalidInput(
            "grouped ring-switch requires at least one digit group".to_string(),
        ));
    };
    let stride = first.digit_stride();
    let mut digits = Vec::new();
    let mut block_sizes = Vec::new();
    for block in blocks {
        if block.digit_stride() != stride {
            return Err(AkitaError::InvalidInput(
                "grouped ring-switch digit groups have mixed ring dimensions".to_string(),
            ));
        }
        digits.extend_from_slice(block.digits());
        block_sizes.extend_from_slice(block.block_sizes());
    }
    DigitBlocks::new(digits, block_sizes, stride)
}

fn typed_z_folded_centered_per_chunk<const D: usize>(
    z_folded_centered_per_chunk: &[Vec<Vec<i32>>],
) -> Result<Vec<Vec<[i32; D]>>, AkitaError> {
    z_folded_centered_per_chunk
        .iter()
        .map(|chunk| {
            chunk
                .iter()
                .map(|row| {
                    row.as_slice()
                        .try_into()
                        .map_err(|_| AkitaError::InvalidSize {
                            expected: D,
                            actual: row.len(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()
}

/// Append one group's contiguous `[z_g ‖ e_g ‖ t_g]` stride to the group-major
/// witness. `build_w_coeffs_for_params` already emits a group's segments in
/// `z ‖ e ‖ t` order (with an empty `r`), so the whole run is appended verbatim.
fn append_group_witness_segments<F: CanonicalField, const D: usize>(
    out: &mut Vec<i8>,
    group: &PreparedRingSwitchGroup<'_, F, D>,
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<(), AkitaError> {
    let num_digits_fold = root_lp.num_digits_fold_for_params(
        group.params,
        num_claims,
        root_lp.field_bits_for_cache(),
    )?;
    let group_w = build_w_coeffs_for_params::<F, D>(
        &group.e_hat,
        &group.t_hat,
        &group.z_folded_centered_per_chunk,
        &[],
        group.params,
        num_digits_fold,
        group.z_folded_centered_per_chunk.len(),
    )?;
    let z_len = group
        .z_folded_centered_per_chunk
        .iter()
        .map(|chunk| chunk.len() * num_digits_fold * D)
        .sum::<usize>();
    let e_len = group.e_hat.typed_planes::<D>()?.len() * D;
    let t_len = group.t_hat.typed_planes::<D>()?.len() * D;
    let expected_len = z_len
        .checked_add(e_len)
        .and_then(|len| len.checked_add(t_len))
        .ok_or_else(|| AkitaError::InvalidSetup("group witness segment overflow".to_string()))?;
    let digits = group_w.as_i8_digits();
    if digits.len() != expected_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_len,
            actual: digits.len(),
        });
    }
    out.extend_from_slice(digits);
    Ok(())
}

/// Build the witness vector `w` from the ring-relation witness.
///
/// This is the first half of the ring switch: it computes `r` and assembles
/// `w` as a flat recursive witness. The resulting `w` is D-agnostic and can be
/// committed at any supported ring dimension by the recursive commitment path.
///
/// # Errors
///
/// Returns an error if the ring-relation witness is missing prover-side data.
#[tracing::instrument(skip_all, name = "ring_switch_build_w")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ring_switch_build_w<F, B>(
    instance: &RingRelationInstance<F>,
    witness: RingRelationWitness<F>,
    ring_switch_ctx: &OperationCtx<'_, F, B>,
    lp: &LevelParams,
    retain_terminal_artifacts: bool,
) -> Result<RingSwitchBuildOutput<F>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + AkitaSerialize,
    B: RuntimeRingSwitchProveBackend<F>,
{
    let dims = instance.role_dims();
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        dims.d_a(),
        |D| {
            validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
            witness.ensure_role_dim::<D>(RingRole::Opening)?;
            witness.ensure_role_dim::<D>(RingRole::Inner)?;
            let RingRelationWitness {
                groups,
                fold_grind_nonce: _,
            } = witness;
            let opening_batch = instance.opening_batch();
            let is_multi_group = groups.len() > 1;
            if groups.len() != opening_batch.num_groups() {
                return Err(AkitaError::InvalidInput(
                    "ring-switch witness count does not match opening batch".to_string(),
                ));
            }
            let final_group_index = lp.validate_root_opening_batch(opening_batch)?;
            let order = opening_batch.root_group_order()?;
            let mut owned = Vec::with_capacity(groups.len());
            for (group_index, group) in groups.into_iter().enumerate() {
                group.ensure_role_dim::<D>(RingRole::Opening)?;
                group.ensure_role_dim::<D>(RingRole::Inner)?;
                let group_lp = lp.root_group_params(opening_batch, group_index)?;
                let RingRelationGroupWitness {
                    z_folded_rings,
                    z_folded_centered_per_chunk,
                    e_hat,
                    e_folded,
                    hint,
                    ..
                } = group;
                e_hat.ensure_stride::<D>()?;
                let e_folded = e_folded.as_ring_slice_trusted::<D>().to_vec();
                let recomposed_inner_rows = crate::compute::recompose_flat_hint_inner_rows::<F, D>(
                    &hint,
                    group_lp.num_digits_open(),
                    group_lp.log_basis(),
                )?;
                let t_hat = hint.into_flat_parts()?;
                t_hat.ensure_stride::<D>()?;
                let z_folded_centered_per_chunk =
                    typed_z_folded_centered_per_chunk::<D>(&z_folded_centered_per_chunk)?;
                owned.push(PreparedRingSwitchGroup {
                    params: group_lp,
                    e_hat,
                    t_hat,
                    recomposed_inner_rows,
                    e_folded,
                    z_centered: z_folded_rings.centered_coeffs_owned::<D>(),
                    z_inf: z_folded_rings.centered_inf_norm,
                    z_folded_centered_per_chunk,
                });
            }
            let has_multi_chunk_witness = lp.witness_chunk.num_chunks > 1
                || owned
                    .iter()
                    .any(|group| group.z_folded_centered_per_chunk.len() > 1);
            if is_multi_group && has_multi_chunk_witness {
                return Err(AkitaError::InvalidSetup(
                    GROUPED_ROOT_MULTI_CHUNK_UNSUPPORTED.to_string(),
                ));
            }
            // Only the singleton suffix retains terminal artifacts; multi-group folds
            // are never terminal.
            if is_multi_group && retain_terminal_artifacts {
                return Err(AkitaError::InvalidInput(
                    "grouped root ring-switch does not produce terminal artifacts".to_string(),
                ));
            }
            validate_chunked_witness_cfg(lp)?;
            if dims.d_d() != D {
                return Err(AkitaError::InvalidSetup(format!(
                "mixed-role ring switch build requires d_d={} to match d_a={D} until nested views land",
                dims.d_d()
            )));
            }
            instance.ensure_ring_dim::<D>()?;

            // Shared relation quotient `r`: its consistency row (summed over all
            // groups) and D rows span every group, so a single trailing block owns
            // it. `groups.len() == 1` reproduces the scalar layout byte-for-byte.
            let e_hat_blocks = order
                .iter()
                .map(|&group_index| owned[group_index].e_hat.clone())
                .collect::<Vec<_>>();
            let e_hat_concat = concat_digit_blocks(&e_hat_blocks)?;
            let ring_multiplier_points = owned
                .iter()
                .enumerate()
                .map(|(group_index, _)| instance.group_ring_multiplier_point(group_index))
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let r = compute_grouped_relation_quotient::<F, B, D>(
                ring_switch_ctx,
                lp,
                opening_batch,
                &owned,
                &ring_multiplier_points,
                instance.group_challenges(),
                e_hat_concat.typed_planes::<D>()?,
                instance.rhs_trusted::<D>()?,
                instance.relation_matrix_row_layout(),
            )?;

            // Group-major witness: emit each group's contiguous `[z_g ‖ e_g ‖ t_g]`
            // stride in `root_group_order()`, then the single shared `r` tail.
            let mut out = Vec::new();
            for &group_index in &order {
                let group_layout = opening_batch.group_layout(group_index)?;
                append_group_witness_segments::<F, D>(
                    &mut out,
                    &owned[group_index],
                    lp,
                    group_layout.num_polynomials(),
                )?;
            }
            let levels = r_decomp_levels::<F>(lp.log_basis);
            emit_r_decomposition_tail::<F, D>(&mut out, &r, levels, lp.log_basis);
            let expected =
                lp.root_next_w_len::<F>(opening_batch, instance.relation_matrix_row_layout())?;
            if out.len() != expected {
                return Err(AkitaError::InvalidSize {
                    expected,
                    actual: out.len(),
                });
            }

            // Terminal artifacts are produced only for the singleton suffix, whose
            // sole group is the final group.
            let terminal_artifacts = if retain_terminal_artifacts {
                let group = owned
                    .get(final_group_index)
                    .ok_or(AkitaError::InvalidProof)?;
                Some(RingSwitchTerminalArtifacts::from_parts::<D>(
                    group.e_folded.clone(),
                    group.recomposed_inner_rows.clone(),
                    group.z_centered.clone(),
                    r,
                ))
            } else {
                None
            };
            Ok(RingSwitchBuildOutput {
                w: RecursiveWitnessFlat::from_i8_digits(out),
                terminal_artifacts,
            })
        }
    )
}

pub(super) fn balanced_decompose_centered_i32_i8_into<const D: usize>(
    centered: &[i32; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let levels = out.len();
    assert!(
        log_basis > 0 && log_basis <= 6,
        "log_basis must be in 1..=6 for i8 output"
    );
    assert!(
        (levels as u32).saturating_mul(log_basis) <= 128 + log_basis,
        "levels * log_basis must be <= 128 + log_basis"
    );

    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;

    for coeff_idx in 0..D {
        let mut c = centered[coeff_idx] as i128;
        for plane in out.iter_mut() {
            let d = c & mask;
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) >> log_basis;
            plane[coeff_idx] = balanced as i8;
        }
    }
}

/// Emit one chunk's window of a block-major digit segment (`ê` or `t̂`),
/// digit-major with the block index innermost, restricted to the global block
/// window `[block_lo, block_lo + blocks_per_chunk)`.
///
/// `flat` is indexed `flat[(outer · num_blocks + block) · planes_per_block +
/// compound_dig]`, where `outer` is the claim (for `ê`) or `t`-vector (for `t̂`)
/// axis. With `block_lo = 0` and `blocks_per_chunk = num_blocks` (the
/// single-chunk case) this reproduces [`akita_types::emit_witness_planes_block_inner`]
/// exactly.
fn emit_witness_planes_block_window<const D: usize>(
    out: &mut Vec<i8>,
    flat: &[[i8; D]],
    num_outer: usize,
    num_blocks: usize,
    planes_per_block: usize,
    block_lo: usize,
    blocks_per_chunk: usize,
) {
    for compound_dig in 0..planes_per_block {
        for outer in 0..num_outer {
            for bl in 0..blocks_per_chunk {
                let blk = outer * num_blocks + (block_lo + bl);
                out.extend_from_slice(&flat[blk * planes_per_block + compound_dig]);
            }
        }
    }
}

fn emit_digit_blocks_block_window<const D: usize>(
    out: &mut Vec<i8>,
    blocks: &DigitBlocks,
    num_outer: usize,
    num_blocks: usize,
    planes_per_block: usize,
    block_lo: usize,
    blocks_per_chunk: usize,
) -> Result<(), AkitaError> {
    emit_witness_planes_block_window(
        out,
        blocks.typed_planes::<D>()?,
        num_outer,
        num_blocks,
        planes_per_block,
        block_lo,
        blocks_per_chunk,
    );
    Ok(())
}

/// Decompose centered `z` fold response coeffs and emit digit-major planes.
fn emit_z_folded_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    z_folded_centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    num_digits_fold: usize,
    log_basis: u32,
) {
    let total_elems = z_folded_centered.len();
    let inner_width = block_len * depth_commit;
    debug_assert_eq!(
        total_elems % inner_width,
        0,
        "z_folded_rings length {total_elems} not divisible by inner_width {inner_width}",
    );

    let mut all_planes = vec![[0i8; D]; total_elems * num_digits_fold];
    for (k, z_j) in z_folded_centered.iter().enumerate() {
        balanced_decompose_centered_i32_i8_into(
            z_j,
            &mut all_planes[k * num_digits_fold..(k + 1) * num_digits_fold],
            log_basis,
        );
    }
    akita_types::emit_witness_z_folded_planes_inner::<D>(
        out,
        &all_planes,
        block_len,
        depth_commit,
        num_digits_fold,
        total_elems,
    );
}

fn emit_r_decomposition_tail<F: CanonicalField, const D: usize>(
    out: &mut Vec<i8>,
    r: &[CyclotomicRing<F, D>],
    levels: usize,
    log_basis: u32,
) {
    let mut r_planes = vec![[0i8; D]; levels];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(levels, log_basis, q);
    for ri in r {
        r_planes.fill([0i8; D]);
        ri.balanced_decompose_pow2_i8_into_with_params(&mut r_planes, &decompose_params);
        for plane in &r_planes {
            out.extend_from_slice(plane);
        }
    }
}

/// Build the committed witness polynomial from ring-domain digit planes.
///
/// Emits field-domain coefficients in digit-major order (block index innermost):
/// z-hat, e-hat + t-hat, r-hat.
///
/// Within each segment, the power-of-2 block index is the fastest-varying
/// (innermost) dimension.
///
/// [`DigitBlocks`] stores ring-domain data in block-major order (all digit
/// planes for one block contiguously), which is natural for ring-domain matvec
/// and recomposition. This function transposes opening digits to digit-major at
/// the ring-to-field boundary.
///
/// # Panics
///
/// Panics if the caller supplies digit blocks whose plane counts do not match
/// the fold layout in `lp`.
#[allow(clippy::too_many_arguments)]
fn build_w_coeffs_for_params<F: CanonicalField, const D: usize>(
    e_hat: &DigitBlocks,
    t_hat: &DigitBlocks,
    z_folded_centered_per_chunk: &[Vec<[i32; D]>],
    r: &[CyclotomicRing<F, D>],
    params: &(impl LevelParamsLike + ?Sized),
    num_digits_fold: usize,
    num_chunks: usize,
) -> Result<RecursiveWitnessFlat, AkitaError> {
    let log_basis = params.log_basis();
    let depth_open = params.num_digits_open();
    let depth_commit = params.num_digits_commit();
    let block_len = params.block_len();
    let num_blocks = params.num_blocks();
    let levels = r_decomp_levels::<F>(log_basis);

    // Chunk geometry: `num_chunks = 1` is the single-chunk (historical) layout.
    if num_chunks == 0 {
        return Err(AkitaError::InvalidSetup(
            "witness chunk count must be >= 1".to_string(),
        ));
    }
    if num_chunks > num_blocks {
        return Err(AkitaError::InvalidSetup(
            "witness chunk count exceeds num_blocks".to_string(),
        ));
    }
    if !num_blocks.is_multiple_of(num_chunks) {
        return Err(AkitaError::InvalidSetup(
            "witness chunk count must divide num_blocks".to_string(),
        ));
    }
    let blocks_per_chunk = num_blocks / num_chunks;
    if num_chunks > 1 && !blocks_per_chunk.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "witness chunk block window must be a power of two".to_string(),
        ));
    }

    let e_hat_planes = e_hat.typed_planes::<D>()?.len();
    let t_hat_planes = t_hat.typed_planes::<D>()?.len();
    let z_planes_total: usize = z_folded_centered_per_chunk
        .iter()
        .map(|z| z.len() * num_digits_fold)
        .sum();
    let z_count = e_hat_planes + t_hat_planes + z_planes_total;
    let r_hat_count = r.len() * levels;
    tracing::debug!(
        num_chunks,
        e_hat_planes,
        t_hat_planes,
        z_folded_planes = z_planes_total,
        r_elems = r.len(),
        r_planes = r_hat_count,
        total_ring = z_count + r_hat_count,
        total_field = (z_count + r_hat_count) * D,
        "build_w_coeffs"
    );
    let total_planes = z_count + r_hat_count;
    let total_elems = total_planes * D;

    let mut out = Vec::with_capacity(total_elems);

    let w_block_count = e_hat.block_count();
    assert_eq!(
        e_hat_planes,
        w_block_count * depth_open,
        "build_w_coeffs: e_hat block layout does not match open digit depth"
    );
    let e_num_outer = w_block_count.checked_div(num_blocks).unwrap_or(0);
    let t_block_count = t_hat.block_count();
    let t_planes_per_block = if t_block_count == 0 {
        0
    } else {
        assert_eq!(
            t_hat_planes % t_block_count,
            0,
            "build_w_coeffs: t_hat block layout must be uniform"
        );
        t_hat_planes / t_block_count
    };
    let t_num_outer = t_block_count.checked_div(num_blocks).unwrap_or(0);
    assert_eq!(
        z_folded_centered_per_chunk.len(),
        num_chunks,
        "build_w_coeffs: per-chunk fold count must equal num_chunks"
    );

    // Per window: z_i (replicated full fold), ê_i / t̂_i (partitioned block
    // windows). The single shared `r̂` tail follows the last window. With
    // `num_chunks = 1` this is byte-identical to the legacy `z ‖ e ‖ t ‖ r`
    // emission.
    for (chunk, z_i) in z_folded_centered_per_chunk.iter().enumerate() {
        emit_z_folded_block_inner(
            &mut out,
            z_i,
            block_len,
            depth_commit,
            num_digits_fold,
            log_basis,
        );
        let block_lo = chunk * blocks_per_chunk;
        emit_digit_blocks_block_window::<D>(
            &mut out,
            e_hat,
            e_num_outer,
            num_blocks,
            depth_open,
            block_lo,
            blocks_per_chunk,
        )?;
        emit_digit_blocks_block_window::<D>(
            &mut out,
            t_hat,
            t_num_outer,
            num_blocks,
            t_planes_per_block,
            block_lo,
            blocks_per_chunk,
        )?;
    }

    emit_r_decomposition_tail::<F, D>(&mut out, r, levels, log_basis);
    Ok(RecursiveWitnessFlat::from_i8_digits(out))
}
