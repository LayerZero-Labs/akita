use super::*;
use crate::compute::{OperationCtx, RuntimeRingSwitchProveBackend};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::protocol::ring_relation::{
    validate_chunked_witness_cfg, CompressionSourceId, CompressionWitnessMaterialization,
    RelationQuotientOutput,
};
use crate::protocol::ring_relation_witness::{RingRelationGroupWitness, RingRelationWitness};
use crate::validation::validate_i8_setup_log_basis;
use akita_algebra::CyclotomicRing;
use akita_serialization::AkitaSerialize;
use akita_types::{
    dispatch_for_field, emit_witness_e_planes, emit_witness_t_planes, emit_witness_z_planes,
    CommitmentRingDims, CompressionWitnessSpan, LevelParamsLike, PackedNegativeBinary, RingRole,
    RingVec, WitnessLayout,
};

pub(crate) struct PreparedRingSwitchGroup<'a, F: FieldCore> {
    pub(crate) params: &'a dyn LevelParamsLike,
    pub(crate) role_dims: CommitmentRingDims,
    pub(crate) e_hat: DigitBlocks,
    pub(crate) t_hat: DigitBlocks,
    /// Block-major native-A rows: `[block][A row][coefficient]`.
    pub(crate) recomposed_inner_rows: RingVec<F>,
    pub(crate) e_folded: RingVec<F>,
    pub(crate) z_centered: Vec<i32>,
    pub(crate) z_inf: u32,
    pub(crate) z_folded_centered_per_chunk: Vec<Vec<Vec<i32>>>,
}

fn concat_digit_blocks<'a>(
    blocks: impl IntoIterator<Item = &'a DigitBlocks>,
) -> Result<DigitBlocks, AkitaError> {
    let mut blocks = blocks.into_iter();
    let Some(first) = blocks.next() else {
        return Err(AkitaError::InvalidInput(
            "multi-group ring-switch requires at least one digit group".to_string(),
        ));
    };
    let stride = first.digit_stride();
    let mut digits = Vec::with_capacity(first.digits().len());
    let mut block_sizes = Vec::with_capacity(first.block_sizes().len());
    digits.extend_from_slice(first.digits());
    block_sizes.extend_from_slice(first.block_sizes());
    for block in blocks {
        if block.digit_stride() != stride {
            return Err(AkitaError::InvalidInput(
                "multi-group ring-switch digit groups have mixed ring dimensions".to_string(),
            ));
        }
        digits.extend_from_slice(block.digits());
        block_sizes.extend_from_slice(block.block_sizes());
    }
    DigitBlocks::new(digits, block_sizes, stride)
}

fn emit_packed_negative_binary(
    out: &mut [i8],
    span: &CompressionWitnessSpan,
    packed: &PackedNegativeBinary,
) -> Result<(), AkitaError> {
    if packed.map() != span.map() || span.range().len() != packed.map().padded_digit_count() {
        return Err(AkitaError::InvalidProof);
    }
    let range = span.range();
    let target = out.get_mut(range).ok_or(AkitaError::InvalidProof)?;
    for (linear, coefficient) in target
        .iter_mut()
        .take(packed.map().real_digit_count())
        .enumerate()
    {
        if packed.bytes()[linear / 8] >> (linear % 8) & 1 == 1 {
            *coefficient = -1;
        }
    }
    Ok(())
}

fn emit_compression_witness<F: FieldCore>(
    out: &mut [i8],
    layout: &WitnessLayout,
    compression: &CompressionWitnessMaterialization<F>,
) -> Result<(), AkitaError> {
    for layer in layout.compression_layers() {
        let map_index = layer.map_index();
        for (group_index, span) in layer.f_spans() {
            let source = compression.source(CompressionSourceId::Outer {
                group_index: *group_index,
            })?;
            let packed = source
                .witness
                .stages()
                .get(map_index)
                .ok_or(AkitaError::InvalidProof)?;
            emit_packed_negative_binary(out, span, packed)?;
        }
        let source = compression.source(CompressionSourceId::Opening)?;
        let packed = source
            .witness
            .stages()
            .get(map_index)
            .ok_or(AkitaError::InvalidProof)?;
        emit_packed_negative_binary(out, layer.h_span(), packed)?;
    }
    Ok(())
}

/// Emit one group's physical Z, E, and T planes through the canonical layout.
fn emit_group_witness_segments<F: CanonicalField>(
    out: &mut [i8],
    layout: &WitnessLayout,
    group_id: usize,
    group: &PreparedRingSwitchGroup<'_, F>,
    num_claims: usize,
) -> Result<(), AkitaError> {
    let num_digits_fold = group.params.num_digits_fold();
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        group.role_dims.d_a(),
        |D_G| {
            emit_group_native_a_segments::<F, D_G>(
                out,
                layout,
                group_id,
                group,
                num_claims,
                num_digits_fold,
            )
        }
    )?;
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        group.role_dims.d_a(),
        |D_A| {
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Opening),
                F,
                group.role_dims.d_d(),
                |D_D| {
                    emit_witness_e_planes::<D_A, D_D>(
                        out,
                        layout,
                        group_id,
                        num_claims,
                        group.params.num_digits_open(),
                        &group.e_hat,
                        group.params.num_live_blocks(),
                    )
                }
            )
        }
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_group_native_a_segments<F: CanonicalField, const D_GROUP: usize>(
    out: &mut [i8],
    layout: &WitnessLayout,
    group_id: usize,
    group: &PreparedRingSwitchGroup<'_, F>,
    num_claims: usize,
    num_digits_fold: usize,
) -> Result<(), AkitaError> {
    let units = layout.units_for_group(group_id)?;
    let unit_count = units.clone().count();
    if unit_count != group.z_folded_centered_per_chunk.len() {
        return Err(AkitaError::InvalidSize {
            expected: unit_count,
            actual: group.z_folded_centered_per_chunk.len(),
        });
    }
    for (unit, z_centered) in units.zip(&group.z_folded_centered_per_chunk) {
        let typed: Vec<[i32; D_GROUP]> = z_centered
            .iter()
            .map(|row| {
                row.as_slice()
                    .try_into()
                    .map_err(|_| AkitaError::InvalidSize {
                        expected: D_GROUP,
                        actual: row.len(),
                    })
            })
            .collect::<Result<_, _>>()?;
        let z_planes =
            decompose_z_folded_planes(&typed, num_digits_fold, group.params.log_basis_open())?;
        emit_witness_z_planes::<D_GROUP>(
            out,
            unit,
            group.params.num_positions_per_block(),
            group.params.num_digits_inner(),
            num_digits_fold,
            &z_planes,
        )?;
    }
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Outer),
        F,
        group.role_dims.d_b(),
        |D_B| {
            emit_witness_t_planes::<D_GROUP, D_B>(
                out,
                layout,
                group_id,
                num_claims,
                group.params.a_rows_len(),
                group.params.num_digits_outer(),
                &group.t_hat,
                group.params.num_live_blocks(),
            )
        }
    )
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
    lp: &CommittedGroupParams,
) -> Result<RecursiveWitnessFlat, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + AkitaSerialize,
    B: RuntimeRingSwitchProveBackend<F>,
{
    let opening_batch = instance.opening_batch();
    validate_i8_setup_log_basis(lp.log_basis_open, "for i8 prover opening decomposition")?;
    let RingRelationWitness {
        groups,
        fold_grind_nonce: _,
        compression,
    } = witness;
    if groups.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidInput(
            "ring-switch witness count does not match opening batch".to_string(),
        ));
    }
    lp.validate_opening_batch(opening_batch)?;
    let order = opening_batch.root_group_order()?;
    let mut owned = Vec::with_capacity(groups.len());
    for (group_index, group) in groups.into_iter().enumerate() {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_dims = lp.group_role_dims(opening_batch, group_index)?;
        if group.role_dims() != group_dims {
            return Err(AkitaError::InvalidInput(format!(
                        "ring-switch witness group {group_index} role dimensions disagree with level params"
                    )));
        }
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            group_dims.d_a(),
            |D_G| group.ensure_role_dim::<D_G>(RingRole::Inner)
        )?;
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            group_dims.d_d(),
            |D_G| group.ensure_role_dim::<D_G>(RingRole::Opening)
        )?;
        let RingRelationGroupWitness {
            z_folded_rings,
            z_folded_centered_per_chunk,
            e_hat,
            e_folded,
            hint,
            ..
        } = group;
        if hint.ring_dim() != group_dims.d_a() {
            return Err(AkitaError::InvalidSize {
                expected: group_dims.d_a(),
                actual: hint.ring_dim(),
            });
        }
        let inner_rows_by_polynomial = hint.into_rows();
        let polynomial_count = opening_batch.group_layout(group_index)?.num_polynomials();
        if inner_rows_by_polynomial.len() != polynomial_count {
            return Err(AkitaError::InvalidSize {
                expected: polynomial_count,
                actual: inner_rows_by_polynomial.len(),
            });
        }
        let expected_rings_per_polynomial = group_lp
            .num_live_blocks()
            .checked_mul(group_lp.a_rows_len())
            .ok_or_else(|| AkitaError::InvalidSetup("commitment hint row count overflow".into()))?;
        let t_hat = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            group_dims.d_a(),
            |D_G| {
                dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Outer),
                    F,
                    group_dims.d_b(),
                    |D_B| {
                        let mut blocks =
                            Vec::with_capacity(polynomial_count * group_lp.num_live_blocks());
                        for rows in &inner_rows_by_polynomial {
                            let typed_rows = rows.as_ring_slice::<D_G>()?;
                            if typed_rows.len() != expected_rings_per_polynomial {
                                return Err(AkitaError::InvalidSize {
                                    expected: expected_rings_per_polynomial,
                                    actual: typed_rows.len(),
                                });
                            }
                            blocks.extend(typed_rows.chunks_exact(group_lp.a_rows_len()));
                        }
                        decompose_commit_blocks_into::<F, D_G, D_B>(
                            &blocks,
                            group_lp.num_digits_outer(),
                            group_lp.log_basis_outer(),
                        )
                    }
                )
            }
        )?;
        let expected_coefficients = polynomial_count
            .checked_mul(expected_rings_per_polynomial)
            .and_then(|count| count.checked_mul(group_dims.d_a()))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("commitment hint coefficient count overflow".into())
            })?;
        let mut inner_rows = inner_rows_by_polynomial.into_iter();
        let mut inner_coefficients = inner_rows
            .next()
            .ok_or(AkitaError::InvalidProof)?
            .into_coeffs();
        inner_coefficients.reserve(expected_coefficients - inner_coefficients.len());
        for rows in inner_rows {
            inner_coefficients.extend(rows.into_coeffs());
        }
        let recomposed_inner_rows =
            RingVec::from_coeffs_with_ring_dim(inner_coefficients, group_dims.d_a())?;
        owned.push(PreparedRingSwitchGroup {
            params: group_lp,
            role_dims: group_dims,
            e_hat,
            t_hat,
            recomposed_inner_rows,
            e_folded,
            z_centered: z_folded_rings.centered_coeffs_flat().to_vec(),
            z_inf: z_folded_rings.centered_inf_norm,
            z_folded_centered_per_chunk,
        });
    }
    validate_chunked_witness_cfg(lp)?;
    for group_index in 0..opening_batch.num_groups() {
        let group_dims = lp.group_role_dims(opening_batch, group_index)?;
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            group_dims.d_a(),
            |D_G| {
                instance
                    .group_ring_multiplier_point(group_index)?
                    .ensure_ring_dim::<D_G>()
            }
        )?;
    }
    let witness_layout = instance.segment_layout(lp, None)?;

    // Relation quotient `r`: each group owns a native consistency/A/B
    // block, while the level owns the shared D tail. One trailing witness
    // segment carries all quotient rows in canonical relation order.
    let e_hat_concat_storage;
    let e_hat_concat = if let [group_index] = order.as_slice() {
        &owned[*group_index].e_hat
    } else {
        e_hat_concat_storage =
            concat_digit_blocks(order.iter().map(|&group_index| &owned[group_index].e_hat))?;
        &e_hat_concat_storage
    };
    let ring_multiplier_points = owned
        .iter()
        .enumerate()
        .map(|(group_index, _)| instance.group_ring_multiplier_point(group_index))
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let r = compute_multi_group_relation_quotient::<F, B>(
        ring_switch_ctx,
        lp,
        opening_batch,
        &owned,
        &ring_multiplier_points,
        instance.group_challenges(),
        e_hat_concat,
        instance.rhs(),
        compression.as_ref(),
    )
    .map_err(|err| {
        AkitaError::InvalidInput(format!("relation quotient preparation failed: {err:?}"))
    })?;

    let mut out = vec![0i8; witness_layout.live_coeff_len()];
    for &group_index in &order {
        let group_layout = opening_batch.group_layout(group_index)?;
        emit_group_witness_segments::<F>(
            &mut out,
            &witness_layout,
            group_index,
            &owned[group_index],
            group_layout.num_polynomials(),
        )?;
    }
    let levels = r_decomp_levels::<F>(lp.log_basis_open);
    emit_r_rows(&mut out, &witness_layout, &r, levels, lp.log_basis_open)?;
    if let Some(compression) = &compression {
        emit_compression_witness(&mut out, &witness_layout, compression)?;
    }
    let expected = witness_layout.live_coeff_len();
    if out.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: out.len(),
        });
    }

    // Every segment of the generated witness is balanced, but grouped
    // roots may mix decomposition bases. The whole-buffer certificate
    // must therefore carry the widest emitted basis: using only the
    // root basis could incorrectly trust a later narrower commit.
    let known_balanced_log_basis = owned
        .iter()
        .flat_map(|group| {
            [
                group.params.log_basis_inner(),
                group.params.log_basis_outer(),
                group.params.log_basis_open(),
            ]
        })
        .fold(lp.log_basis_open, u32::max);
    RecursiveWitnessFlat::from_witness_layout(out, &witness_layout, known_balanced_log_basis)
}

pub(super) fn balanced_decompose_centered_i32_i8_into<const D: usize>(
    centered: &[i32; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let levels = out.len();
    assert!(
        log_basis > 0 && log_basis <= 8,
        "log_basis must be in 1..=8 for i8 output"
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

/// Decompose centered Z fold responses into `(position, commit_digit, fold_digit)` planes.
fn decompose_z_folded_planes<const D: usize>(
    z_folded_centered: &[[i32; D]],
    num_digits_fold: usize,
    log_basis: u32,
) -> Result<Vec<[i8; D]>, AkitaError> {
    let plane_count = z_folded_centered
        .len()
        .checked_mul(num_digits_fold)
        .ok_or_else(|| AkitaError::InvalidSetup("Z plane count overflow".to_string()))?;
    let mut all_planes = vec![[0i8; D]; plane_count];
    for (k, z_j) in z_folded_centered.iter().enumerate() {
        balanced_decompose_centered_i32_i8_into(
            z_j,
            &mut all_planes[k * num_digits_fold..(k + 1) * num_digits_fold],
            log_basis,
        );
    }
    Ok(all_planes)
}

fn emit_r_rows<F: CanonicalField>(
    out: &mut [i8],
    layout: &WitnessLayout,
    r: &RelationQuotientOutput<F>,
    levels: usize,
    log_basis: u32,
) -> Result<(), AkitaError> {
    if layout.r_rows().len() != r.rows().len() || layout.quotient_depth() != levels {
        return Err(AkitaError::InvalidProof);
    }
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2Params::new(levels, log_basis, q);
    for (row_index, row) in r.rows().iter().enumerate() {
        let row_layout = layout
            .r_rows()
            .get(row_index)
            .ok_or(AkitaError::InvalidProof)?;
        if row_layout.ring_dim() != row.ring_dim() {
            return Err(AkitaError::InvalidSize {
                expected: row_layout.ring_dim(),
                actual: row.ring_dim(),
            });
        }
        let digits = match row.ring_dim() {
            8 => decompose_r_row::<F, 8>(row.coeffs(), levels, &decompose_params)?,
            16 => decompose_r_row::<F, 16>(row.coeffs(), levels, &decompose_params)?,
            32 => decompose_r_row::<F, 32>(row.coeffs(), levels, &decompose_params)?,
            64 => decompose_r_row::<F, 64>(row.coeffs(), levels, &decompose_params)?,
            128 => decompose_r_row::<F, 128>(row.coeffs(), levels, &decompose_params)?,
            256 => decompose_r_row::<F, 256>(row.coeffs(), levels, &decompose_params)?,
            512 => decompose_r_row::<F, 512>(row.coeffs(), levels, &decompose_params)?,
            1024 => decompose_r_row::<F, 1024>(row.coeffs(), levels, &decompose_params)?,
            actual => {
                return Err(AkitaError::InvalidSize {
                    expected: 512,
                    actual,
                })
            }
        };
        for digit in 0..levels {
            let start = digit * row.ring_dim();
            let end = start + row.ring_dim();
            let destination = layout.r_coefficient_index(row_index, digit, 0)?;
            let destination_end = destination
                .checked_add(row.ring_dim())
                .ok_or_else(|| AkitaError::InvalidSetup("R witness end overflow".into()))?;
            let plane = out
                .get_mut(destination..destination_end)
                .ok_or(AkitaError::InvalidProof)?;
            plane.copy_from_slice(&digits[start..end]);
        }
    }
    Ok(())
}

fn decompose_r_row<F: CanonicalField, const D: usize>(
    coeffs: &[F],
    levels: usize,
    params: &BalancedDecomposePow2Params,
) -> Result<Vec<i8>, AkitaError> {
    let coeffs: [F; D] = coeffs.try_into().map_err(|_| AkitaError::InvalidSize {
        expected: D,
        actual: coeffs.len(),
    })?;
    let ring = CyclotomicRing::<F, D>::from_coefficients(coeffs);
    let mut planes = vec![[0i8; D]; levels];
    ring.balanced_decompose_pow2_i8_into_with_params(&mut planes, params);
    let mut out = Vec::with_capacity(levels * D);
    for plane in planes {
        out.extend_from_slice(&plane);
    }
    Ok(out)
}
