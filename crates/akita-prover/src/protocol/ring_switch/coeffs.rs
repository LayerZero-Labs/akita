use super::*;
use crate::compute::{OperationCtx, RuntimeRingSwitchProveBackend};
use crate::protocol::ring_relation::validate_chunked_witness_cfg;
use crate::protocol::ring_relation::RelationQuotientOutput;
use crate::protocol::ring_relation_witness::{RingRelationGroupWitness, RingRelationWitness};
use crate::validation::validate_i8_setup_log_basis;
use akita_algebra::CyclotomicRing;
use akita_serialization::AkitaSerialize;
use akita_types::{
    dispatch_for_field, emit_witness_t_planes, emit_witness_z_planes, LevelParamsLike, RingRole,
    WitnessLayout,
};

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

/// Emit one group's physical Z, E, and T planes through the canonical layout.
fn emit_group_witness_segments<F: CanonicalField, const D: usize>(
    out: &mut [i8],
    layout: &WitnessLayout,
    group_id: usize,
    group: &PreparedRingSwitchGroup<'_, F, D>,
    root_lp: &CommittedGroupParams,
    num_claims: usize,
) -> Result<(), AkitaError> {
    let num_digits_fold = root_lp.num_digits_fold_for_params(
        group.params,
        num_claims,
        root_lp.field_bits_for_cache(),
    )?;
    let units = layout.units_for_group(group_id)?;
    if units.len() != group.z_folded_centered_per_chunk.len() {
        return Err(AkitaError::InvalidSize {
            expected: units.len(),
            actual: group.z_folded_centered_per_chunk.len(),
        });
    }
    for (unit, z_centered) in units.into_iter().zip(&group.z_folded_centered_per_chunk) {
        let z_planes =
            decompose_z_folded_planes(z_centered, num_digits_fold, group.params.log_basis_open())?;
        emit_witness_z_planes::<D>(
            out,
            unit,
            group.params.num_positions_per_block(),
            group.params.num_digits_inner(),
            num_digits_fold,
            &z_planes,
        )?;
    }
    emit_group_e_planes_padded::<D>(
        out,
        layout,
        group_id,
        num_claims,
        group.params.num_digits_open(),
        &group.e_hat,
        group.params.num_live_blocks(),
    )?;
    emit_witness_t_planes::<D>(
        out,
        layout,
        group_id,
        num_claims,
        group.params.a_rows_len(),
        group.params.num_digits_outer(),
        group.t_hat.typed_planes::<D>()?,
        group.params.num_live_blocks(),
    )?;
    Ok(())
}

fn emit_group_e_planes_padded<const D_A: usize>(
    out: &mut [i8],
    layout: &WitnessLayout,
    group_id: usize,
    num_claims: usize,
    depth_open: usize,
    e_hat: &DigitBlocks,
    source_num_live_blocks: usize,
) -> Result<(), AkitaError> {
    if e_hat.digit_stride() > D_A || !D_A.is_multiple_of(e_hat.digit_stride()) {
        return Err(AkitaError::InvalidSize {
            expected: D_A,
            actual: e_hat.digit_stride(),
        });
    }
    let expected = num_claims
        .checked_mul(source_num_live_blocks)
        .and_then(|n| n.checked_mul(depth_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness E source length overflow".into()))?;
    let expected_digits = expected
        .checked_mul(D_A)
        .ok_or_else(|| AkitaError::InvalidSetup("witness E digit length overflow".into()))?;
    if e_hat.digits().len() != expected_digits {
        return Err(AkitaError::InvalidSize {
            expected: expected_digits,
            actual: e_hat.digits().len(),
        });
    }
    let role_ratio = D_A / e_hat.digit_stride();
    if e_hat.block_count() != num_claims * source_num_live_blocks * role_ratio {
        return Err(AkitaError::InvalidProof);
    }
    for unit in layout.units_for_group(group_id)? {
        for claim in 0..num_claims {
            for global_block in unit.global_block_range() {
                for digit in 0..depth_open {
                    let logical_block = claim * source_num_live_blocks + global_block;
                    let mut plane = [0i8; D_A];
                    for role_subcol in 0..role_ratio {
                        let source_block = logical_block * role_ratio + role_subcol;
                        let source = source_block * depth_open + digit;
                        let source_plane = e_hat.plane(source).ok_or(AkitaError::InvalidProof)?;
                        let start = role_subcol * e_hat.digit_stride();
                        plane[start..start + e_hat.digit_stride()].copy_from_slice(source_plane);
                    }
                    write_padded_plane::<D_A>(
                        out,
                        unit.e_index(num_claims, depth_open, claim, global_block, digit)?,
                        &plane,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn write_padded_plane<const D_A: usize>(
    out: &mut [i8],
    ring_index: usize,
    plane: &[i8],
) -> Result<(), AkitaError> {
    if plane.len() > D_A {
        return Err(AkitaError::InvalidSize {
            expected: D_A,
            actual: plane.len(),
        });
    }
    let start = ring_index
        .checked_mul(D_A)
        .ok_or_else(|| AkitaError::InvalidSetup("witness plane offset overflow".into()))?;
    let end = start
        .checked_add(D_A)
        .ok_or_else(|| AkitaError::InvalidSetup("witness plane end overflow".into()))?;
    let dst = out.get_mut(start..end).ok_or(AkitaError::InvalidProof)?;
    dst.fill(0);
    dst[..plane.len()].copy_from_slice(plane);
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
    let dims = instance.role_dims();
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        dims.d_a(),
        |D| {
            validate_i8_setup_log_basis(lp.log_basis_open, "for i8 prover opening decomposition")?;
            witness.ensure_role_dim::<D>(RingRole::Inner)?;
            let RingRelationWitness {
                groups,
                fold_grind_nonce: _,
            } = witness;
            let opening_batch = instance.opening_batch();
            if groups.len() != opening_batch.num_groups() {
                return Err(AkitaError::InvalidInput(
                    "ring-switch witness count does not match opening batch".to_string(),
                ));
            }
            lp.validate_opening_batch(opening_batch)?;
            let order = opening_batch.root_group_order()?;
            let mut owned = Vec::with_capacity(groups.len());
            for (group_index, group) in groups.into_iter().enumerate() {
                group.ensure_role_dim::<D>(RingRole::Inner)?;
                let group_lp = lp.group_params(opening_batch, group_index)?;
                let RingRelationGroupWitness {
                    z_folded_rings,
                    z_folded_centered_per_chunk,
                    e_hat,
                    e_folded,
                    hint,
                    ..
                } = group;
                let e_folded = e_folded.as_ring_slice_trusted::<D>().to_vec();
                let t_hat = hint.into_flat_parts()?;
                t_hat.ensure_stride::<D>()?;
                let recomposed_inner_rows = crate::compute::recompose_inner_rows::<F, D>(
                    &t_hat,
                    group_lp.num_digits_outer(),
                    group_lp.log_basis_outer(),
                )?;
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
            validate_chunked_witness_cfg(lp)?;
            for group_index in 0..opening_batch.num_groups() {
                instance
                    .group_ring_multiplier_point(group_index)?
                    .ensure_ring_dim::<D>()?;
            }
            let witness_layout = instance.segment_layout(lp, None)?;

            // Shared relation quotient `r`: its consistency row (summed over all
            // groups) and D rows span every group, so a single trailing block owns
            // it. `groups.len() == 1` reproduces the scalar layout byte-for-byte.
            let e_hat_concat_storage;
            let e_hat_concat = if let [group_index] = order.as_slice() {
                &owned[*group_index].e_hat
            } else {
                e_hat_concat_storage = concat_digit_blocks(
                    order.iter().map(|&group_index| &owned[group_index].e_hat),
                )?;
                &e_hat_concat_storage
            };
            let ring_multiplier_points = owned
                .iter()
                .enumerate()
                .map(|(group_index, _)| instance.group_ring_multiplier_point(group_index))
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let r = compute_multi_group_relation_quotient::<F, B, D>(
                ring_switch_ctx,
                lp,
                opening_batch,
                &owned,
                &ring_multiplier_points,
                instance.group_challenges(),
                e_hat_concat,
                instance.rhs(),
                dims,
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!("relation quotient preparation failed: {err:?}"))
            })?;

            let physical_len = witness_layout
                .total_len()
                .checked_mul(D)
                .ok_or_else(|| AkitaError::InvalidSetup("witness length overflow".to_string()))?;
            let mut out = vec![0i8; physical_len];
            for &group_index in &order {
                let group_layout = opening_batch.group_layout(group_index)?;
                emit_group_witness_segments::<F, D>(
                    &mut out,
                    &witness_layout,
                    group_index,
                    &owned[group_index],
                    lp,
                    group_layout.num_polynomials(),
                )?;
            }
            let levels = r_decomp_levels::<F>(lp.log_basis_open);
            emit_r_rows_padded::<F, D>(&mut out, &witness_layout, &r, levels, lp.log_basis_open)?;
            let expected = lp.output_witness_len::<F>(opening_batch)?;
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
            RecursiveWitnessFlat::from_witness_layout::<D>(
                out,
                &witness_layout,
                known_balanced_log_basis,
            )
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

fn emit_r_rows_padded<F: CanonicalField, const D_A: usize>(
    out: &mut [i8],
    layout: &WitnessLayout,
    r: &RelationQuotientOutput<F>,
    levels: usize,
    log_basis: u32,
) -> Result<(), AkitaError> {
    let expected_len = r
        .rows()
        .len()
        .checked_mul(levels)
        .ok_or_else(|| AkitaError::InvalidSetup("R witness width overflow".to_string()))?;
    if layout.r_range().len() != expected_len {
        return Err(AkitaError::InvalidProof);
    }
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(levels, log_basis, q);
    for (row_index, row) in r.rows().iter().enumerate() {
        let digits = match row.ring_dim() {
            16 => decompose_r_row::<F, 16>(row.coeffs(), levels, &decompose_params)?,
            32 => decompose_r_row::<F, 32>(row.coeffs(), levels, &decompose_params)?,
            64 => decompose_r_row::<F, 64>(row.coeffs(), levels, &decompose_params)?,
            128 => decompose_r_row::<F, 128>(row.coeffs(), levels, &decompose_params)?,
            256 => decompose_r_row::<F, 256>(row.coeffs(), levels, &decompose_params)?,
            actual => {
                return Err(AkitaError::InvalidSize {
                    expected: 256,
                    actual,
                })
            }
        };
        for digit in 0..levels {
            let start = digit * row.ring_dim();
            let end = start + row.ring_dim();
            write_padded_plane::<D_A>(
                out,
                layout.r_index(levels, row_index, digit)?,
                &digits[start..end],
            )?;
        }
    }
    Ok(())
}

fn decompose_r_row<F: CanonicalField, const D: usize>(
    coeffs: &[F],
    levels: usize,
    params: &BalancedDecomposePow2I8Params,
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
