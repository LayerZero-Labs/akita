use super::*;
use crate::backend::packed_digits::PackedSignedDigitWriter;
#[cfg(feature = "response-model-diagnostics")]
use crate::backend::packed_digits::PackedSignedDigits;
use crate::compute::{OperationCtx, RuntimeRingSwitchProveBackend};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::protocol::ring_relation::{
    validate_chunked_witness_cfg, CompressionSourceId, CompressionWitnessMaterialization,
    RelationQuotientOutput,
};
use crate::protocol::ring_relation_witness::{
    FoldChunkCoefficients, GroupFoldedOpening, RelationDQuotientWitness, RingRelationGroupWitness,
    RingRelationWitness,
};
use crate::validation::validate_i8_setup_log_basis;
use crate::DecomposeFoldWitness;
use akita_algebra::balanced_decompose_coefficients_pow2_i8_into;
use akita_serialization::AkitaSerialize;
use akita_types::{
    dispatch_for_field, emit_witness_e_planes, emit_witness_t_planes, emit_witness_z_planes,
    CommitmentRingDims, CompressionWitnessSpan, DigitBlocks, PackedNegativeBinary, RingRole,
    RingVec, WitnessLayout, WitnessUnitLayout,
};
use jolt_field::solinas::parallel::*;
use jolt_field::Field;

pub(crate) struct PreparedRingSwitchGroup<F: Field> {
    pub(crate) params: akita_types::GroupOpenPhaseParams,
    pub(crate) role_dims: CommitmentRingDims,
    pub(crate) e_hat: DigitBlocks,
    pub(crate) t_hat: DigitBlocks,
    /// Block-major native-A rows: `[block][A row][coefficient]`.
    pub(crate) recomposed_inner_rows: RingVec<F>,
    pub(crate) folded_opening: GroupFoldedOpening<F>,
    pub(crate) z_centered: Vec<i32>,
    pub(crate) z_inf: u32,
    pub(crate) z_folded_coefficients: FoldChunkCoefficients,
}

fn emit_packed_negative_binary(
    out: &mut PackedSignedDigitWriter,
    span: &CompressionWitnessSpan,
    packed: &PackedNegativeBinary,
) -> Result<(), AkitaError> {
    if packed.map() != span.map() || span.range().len() != packed.map().padded_digit_count() {
        return Err(AkitaError::InvalidProof);
    }
    let range = span.range();
    const CHUNK: usize = 4096;
    let mut scratch = [0i8; CHUNK];
    let mut written = 0usize;
    while written < range.len() {
        let count = CHUNK.min(range.len() - written);
        scratch[..count].fill(0);
        for (offset, coefficient) in scratch[..count].iter_mut().enumerate() {
            let linear = written + offset;
            if linear < packed.map().real_digit_count()
                && packed.bytes()[linear / 8] >> (linear % 8) & 1 == 1
            {
                *coefficient = -1;
            }
        }
        out.write_at(range.start + written, &scratch[..count])?;
        written += count;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum WitnessTailEvent {
    Quotient {
        row_index: usize,
    },
    Compression {
        source: CompressionSourceId,
        map_index: usize,
    },
}

enum PreparedWitnessTail<F: Field> {
    QuotientLift(RelationQuotientOutput<F>),
    ReducedEvaluation,
}

#[cfg(test)]
std::thread_local! {
    static QUOTIENT_DECOMPOSITION_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn reset_quotient_decomposition_calls() {
    QUOTIENT_DECOMPOSITION_CALLS.set(0);
}

#[cfg(test)]
pub(super) fn quotient_decomposition_calls() -> usize {
    QUOTIENT_DECOMPOSITION_CALLS.get()
}

#[cfg(feature = "response-model-diagnostics")]
fn integer_range_l2_sq(witness: &PackedSignedDigits, range: std::ops::Range<usize>) -> u128 {
    witness
        .view()
        .slice(range)
        .expect("diagnostic range comes from the witness layout")
        .iter()
        .fold(0u128, |sum, value| {
            let magnitude = u128::from(value.unsigned_abs());
            // A witness is indexed by `usize`, while each i8 square is at most
            // 2^14. Consequently this sum cannot overflow u128 on a supported
            // host, even for the largest addressable witness.
            sum + magnitude * magnitude
        })
}

#[cfg(feature = "response-model-diagnostics")]
fn trace_witness_source_moments(
    witness: &PackedSignedDigits,
    layout: &WitnessLayout,
    lp: &CommittedGroupParams,
) {
    if !tracing::enabled!(
        target: "akita_prover::protocol::fold_response_model",
        tracing::Level::INFO
    ) {
        return;
    }

    let mut z_coeffs = 0usize;
    let mut e_coeffs = 0usize;
    let mut t_coeffs = 0usize;
    let mut z_l2_sq = 0u128;
    let mut e_l2_sq = 0u128;
    let mut t_l2_sq = 0u128;
    for unit in layout.units() {
        for (range, coeffs, energy) in [
            (unit.z_range(), &mut z_coeffs, &mut z_l2_sq),
            (unit.e_range(), &mut e_coeffs, &mut e_l2_sq),
            (unit.t_range(), &mut t_coeffs, &mut t_l2_sq),
        ] {
            *coeffs += range.len();
            *energy += integer_range_l2_sq(witness, range);
        }
    }

    let mut r_coeffs = 0usize;
    let mut r_l2_sq = 0u128;
    for row in layout.r_rows() {
        let range = row.range();
        r_coeffs += range.len();
        r_l2_sq += integer_range_l2_sq(witness, range);
    }

    let mut compression_coeffs = 0usize;
    let mut compression_l2_sq = 0u128;
    for layer in layout.compression_layers() {
        for (_, span) in layer.f_spans() {
            let range = span.range();
            compression_coeffs += range.len();
            compression_l2_sq += integer_range_l2_sq(witness, range);
        }
        let range = layer.h_span().range();
        compression_coeffs += range.len();
        compression_l2_sq += integer_range_l2_sq(witness, range);
    }

    let alignment_coeffs = layout
        .compression_alignment_ranges()
        .iter()
        .map(std::ops::Range::len)
        .sum::<usize>();
    let classified_coeffs =
        z_coeffs + e_coeffs + t_coeffs + r_coeffs + compression_coeffs + alignment_coeffs;
    debug_assert_eq!(classified_coeffs, witness.len());
    let source_l2_sq = z_l2_sq + e_l2_sq + t_l2_sq + r_l2_sq + compression_l2_sq;

    tracing::info!(
        target: "akita_prover::protocol::fold_response_model",
        source_coeffs = witness.len(),
        source_l2_sq,
        z_coeffs,
        z_l2_sq,
        e_coeffs,
        e_l2_sq,
        t_coeffs,
        t_l2_sq,
        r_coeffs,
        r_l2_sq,
        compression_coeffs,
        compression_l2_sq,
        alignment_coeffs,
        log_basis_inner = lp.inner().digits.log_basis,
        log_basis_outer = lp.outer().digits.log_basis,
        log_basis_open = lp.open().digits.log_basis,
        num_digits_inner = lp.inner().digits.num_digits,
        num_digits_outer = lp.outer().digits.num_digits,
        num_digits_open = lp.open().digits.num_digits,
        num_digits_fold = lp.num_digits_fold(),
        d_a = lp.role_dims().d_a(),
        d_b = lp.role_dims().d_b(),
        d_d = lp.role_dims().d_d(),
        compressed = lp.payload_mode.is_compressed(),
        "recursive witness source moments"
    );
}

/// Emit one physical `[Z | E | T]` ownership unit directly into packed storage.
fn emit_witness_unit<F: Field + CanonicalEncoding>(
    out: &mut PackedSignedDigitWriter,
    unit: &WitnessUnitLayout,
    group: &PreparedRingSwitchGroup<F>,
    num_claims: usize,
    expected_chunks: usize,
) -> Result<(), AkitaError> {
    let num_digits_fold = group.params.num_digits_fold();
    {
        let _span = tracing::info_span!("ring_switch_emit_native_a_segments").entered();
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            group.role_dims.d_a(),
            |D_G| {
                emit_unit_z_segment::<D_G>(out, unit, group, num_digits_fold, expected_chunks)
            }
        )?;
    }
    {
        let _span = tracing::info_span!("ring_switch_emit_e_segments").entered();
        let opening_width = unit.e_geometry().physical_coefficient_width();
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            group.role_dims.d_d(),
            |D_D| {
                emit_witness_e_planes::<D_D>(
                    out,
                    unit,
                    opening_width,
                    num_claims,
                    group.params.num_digits_open(),
                    &group.e_hat,
                    group.params.num_live_blocks(),
                )
            }
        )?;
    }
    {
        let _span = tracing::info_span!("ring_switch_emit_t_segments").entered();
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            group.role_dims.d_a(),
            |D_A| {
                dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Outer),
                    F,
                    group.role_dims.d_b(),
                    |D_B| {
                        emit_witness_t_planes::<D_A, D_B>(
                            out,
                            unit,
                            num_claims,
                            group.params.a_rows_len(),
                            group.params.num_digits_outer(),
                            &group.t_hat,
                            group.params.num_live_blocks(),
                        )
                    }
                )
            }
        )?;
    }
    Ok(())
}

fn emit_unit_z_segment<const D: usize>(
    out: &mut PackedSignedDigitWriter,
    unit: &WitnessUnitLayout,
    group: &PreparedRingSwitchGroup<impl Field + CanonicalEncoding>,
    num_digits_fold: usize,
    expected_chunks: usize,
) -> Result<(), AkitaError> {
    let z_centered = group.z_folded_coefficients.chunk(
        &group.z_centered,
        expected_chunks,
        unit.chunk_index(),
    )?;
    let z_planes = {
        let _span = tracing::info_span!("ring_switch_decompose_z_planes").entered();
        decompose_z_folded_planes::<D>(z_centered, num_digits_fold, group.params.log_basis_open())?
    };
    let expected_planes = group
        .params
        .num_positions_per_block()
        .checked_mul(group.params.num_digits_inner())
        .and_then(|count| count.checked_mul(num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z plane count overflow".into()))?;
    let range = unit.z_range();
    if z_planes.len() != expected_planes || z_planes.as_flattened().len() != range.len() {
        return Err(AkitaError::InvalidSize {
            expected: range.len(),
            actual: z_planes.as_flattened().len(),
        });
    }
    let _span = tracing::info_span!("ring_switch_emit_z_planes").entered();
    emit_witness_z_planes::<D>(
        out,
        unit,
        group.params.num_positions_per_block(),
        group.params.num_digits_inner(),
        num_digits_fold,
        &z_planes,
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
    F: Field + CanonicalEncoding + Ring + AkitaSerialize,
    B: RuntimeRingSwitchProveBackend<F>,
{
    let opening_batch = instance.opening_batch();
    validate_i8_setup_log_basis(
        lp.open().digits.log_basis,
        "for i8 prover opening decomposition",
    )?;
    let RingRelationWitness {
        groups,
        d_quotients,
        compression,
    } = witness;
    if groups.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidInput(
            "ring-switch witness count does not match opening batch".to_string(),
        ));
    }
    lp.validate_opening_batch(opening_batch)?;
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
            z_folded_coefficients,
            e_hat,
            folded_opening,
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
        let z_inf = z_folded_rings.centered_inf_norm();
        let DecomposeFoldWitness {
            centered_coeffs_flat: z_centered,
            ..
        } = z_folded_rings;
        owned.push(PreparedRingSwitchGroup {
            params: group_lp,
            role_dims: group_dims,
            e_hat,
            t_hat,
            recomposed_inner_rows,
            folded_opening,
            z_centered,
            z_inf,
            z_folded_coefficients,
        });
    }
    validate_chunked_witness_cfg(lp)?;
    for group_index in 0..opening_batch.num_groups() {
        let group_dims = lp.group_role_dims(opening_batch, group_index)?;
        let opening = instance
            .group_openings()
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if let Ok(ring_multiplier_point) = opening.evaluation_trace_multiplier_point() {
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                F,
                group_dims.d_a(),
                |D_G| ring_multiplier_point.ensure_ring_dim::<D_G>()
            )?;
        }
    }
    let witness_layout = instance.segment_layout(lp, None)?;

    // Relation quotient `r`: each group owns a native consistency/A/B
    // block, while the level owns the shared D tail. One trailing witness
    // segment carries all quotient rows in canonical relation order.
    let tail = match (lp.ring_relation_mode, d_quotients) {
        (
            akita_types::RingRelationMode::QuotientLift,
            RelationDQuotientWitness::QuotientLift(d_quotients),
        ) => PreparedWitnessTail::QuotientLift(
            compute_multi_group_relation_quotient::<F, B>(
                ring_switch_ctx,
                lp,
                opening_batch,
                &owned,
                instance.group_openings(),
                instance.extension_degree(),
                &d_quotients,
                instance.rhs(),
                compression.as_ref(),
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!("relation quotient preparation failed: {err:?}"))
            })?,
        ),
        (
            akita_types::RingRelationMode::ReducedEvaluation,
            RelationDQuotientWitness::ReducedEvaluation,
        ) => PreparedWitnessTail::ReducedEvaluation,
        _ => return Err(AkitaError::InvalidProof),
    };

    // Every segment of the generated witness is balanced, but grouped roots
    // may mix decomposition bases. Z and the quotient tail use the opening
    // basis, E uses the opening basis, and T uses the outer basis. The inner
    // basis controls how many Z planes exist; it is not the basis used to
    // decompose their coefficients. The whole-buffer certificate and physical
    // packed width therefore use the widest basis that emits coefficients.
    let known_balanced_log_basis = owned
        .iter()
        .flat_map(|group| {
            [
                group.params.log_basis_outer(),
                group.params.log_basis_open(),
            ]
        })
        .fold(lp.open().digits.log_basis, u32::max);
    let packed_width = u8::try_from(known_balanced_log_basis).map_err(|_| {
        AkitaError::InvalidSetup("recursive witness basis does not fit i8 storage".into())
    })?;
    let mut out = {
        let _span = tracing::info_span!("ring_switch_allocate_output").entered();
        PackedSignedDigitWriter::new(witness_layout.live_coeff_len(), packed_width)?
    };
    for unit in witness_layout.units() {
        let group_index = unit.group_index();
        let _span = tracing::info_span!(
            "ring_switch_emit_witness_unit",
            group_index,
            chunk_index = unit.chunk_index()
        )
        .entered();
        let group_layout = opening_batch.group_layout(group_index)?;
        emit_witness_unit::<F>(
            &mut out,
            unit,
            &owned[group_index],
            group_layout.num_polynomials(),
            witness_layout.num_chunks_for_group(group_index),
        )?;
    }
    {
        let _span = tracing::info_span!("ring_switch_emit_tail").entered();
        match tail {
            PreparedWitnessTail::QuotientLift(r) => emit_witness_tail(
                &mut out,
                &witness_layout,
                &r,
                r_decomp_levels::<F>(lp.open().digits.log_basis),
                lp.open().digits.log_basis,
                compression.as_ref(),
            )?,
            PreparedWitnessTail::ReducedEvaluation => {
                emit_reduced_witness_tail(&mut out, &witness_layout, compression.as_ref())?
            }
        }
    }
    let expected = witness_layout.live_coeff_len();
    if out.position() > expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: out.position(),
        });
    }
    let out = out.finish()?;
    #[cfg(feature = "response-model-diagnostics")]
    trace_witness_source_moments(&out, &witness_layout, lp);
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
    z_folded_centered: &[i32],
    num_digits_fold: usize,
    log_basis: u32,
) -> Result<Vec<[i8; D]>, AkitaError> {
    let (rows, remainder) = z_folded_centered.as_chunks::<D>();
    if !remainder.is_empty() {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: z_folded_centered.len(),
        });
    }
    let plane_count = rows
        .len()
        .checked_mul(num_digits_fold)
        .ok_or_else(|| AkitaError::InvalidSetup("Z plane count overflow".to_string()))?;
    let mut all_planes = vec![[0i8; D]; plane_count];
    cfg_iter!(rows)
        .zip(cfg_chunks_mut!(&mut all_planes, num_digits_fold))
        .for_each(|(z_j, planes)| {
            balanced_decompose_centered_i32_i8_into(z_j, planes, log_basis);
        });
    Ok(all_planes)
}

fn emit_witness_tail<F: Field + CanonicalEncoding>(
    out: &mut PackedSignedDigitWriter,
    layout: &WitnessLayout,
    r: &RelationQuotientOutput<F>,
    levels: usize,
    log_basis: u32,
    compression: Option<&CompressionWitnessMaterialization<F>>,
) -> Result<(), AkitaError> {
    if layout.r_rows().len() != r.rows().len() || layout.quotient_depth() != Some(levels) {
        return Err(AkitaError::InvalidProof);
    }
    let q = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let decompose_params = BalancedDecomposePow2Params::new(levels, log_basis, q);
    let mut events = layout
        .r_rows()
        .iter()
        .enumerate()
        .map(|(row_index, row)| (row.range().start, WitnessTailEvent::Quotient { row_index }))
        .collect::<Vec<_>>();
    for layer in layout.compression_layers() {
        for (group_index, span) in layer.f_spans() {
            events.push((
                span.range().start,
                WitnessTailEvent::Compression {
                    source: CompressionSourceId::Outer {
                        group_index: *group_index,
                    },
                    map_index: layer.map_index(),
                },
            ));
        }
        events.push((
            layer.h_span().range().start,
            WitnessTailEvent::Compression {
                source: CompressionSourceId::Opening,
                map_index: layer.map_index(),
            },
        ));
    }
    events.sort_unstable_by_key(|(start, _)| *start);

    for (_, event) in events {
        match event {
            WitnessTailEvent::Quotient { row_index } => {
                #[cfg(test)]
                QUOTIENT_DECOMPOSITION_CALLS.with(|calls| calls.set(calls.get() + 1));
                let row = r.rows().get(row_index).ok_or(AkitaError::InvalidProof)?;
                let row_layout = layout
                    .r_rows()
                    .get(row_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let geometry = row_layout.geometry();
                if geometry != row.geometry() {
                    return Err(AkitaError::InvalidSize {
                        expected: geometry.physical_coefficient_width(),
                        actual: row.coeffs().len(),
                    });
                }
                let expected_len = levels
                    .checked_mul(geometry.physical_coefficient_width())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("R witness row length overflow".into())
                    })?;
                let range = row_layout.range();
                if range.len() != expected_len {
                    return Err(AkitaError::InvalidSize {
                        expected: expected_len,
                        actual: range.len(),
                    });
                }
                let mut digits = vec![0i8; expected_len];
                balanced_decompose_coefficients_pow2_i8_into(
                    row.coeffs(),
                    &mut digits,
                    &decompose_params,
                );
                out.write_at(range.start, &digits)?;
            }
            WitnessTailEvent::Compression { source, map_index } => {
                emit_compression_witness_event(
                    out,
                    layout,
                    compression.ok_or(AkitaError::InvalidProof)?,
                    source,
                    map_index,
                )?;
            }
        }
    }
    Ok(())
}

fn emit_compression_witness_event<F: Field + CanonicalEncoding>(
    out: &mut PackedSignedDigitWriter,
    layout: &WitnessLayout,
    compression: &CompressionWitnessMaterialization<F>,
    source: CompressionSourceId,
    map_index: usize,
) -> Result<(), AkitaError> {
    let layer = layout
        .compression_layers()
        .get(map_index)
        .ok_or(AkitaError::InvalidProof)?;
    let span = match source {
        CompressionSourceId::Outer { group_index } => layer
            .f_spans()
            .iter()
            .find_map(|(candidate, span)| (*candidate == group_index).then_some(span))
            .ok_or(AkitaError::InvalidProof)?,
        CompressionSourceId::Opening => layer.h_span(),
    };
    let packed = compression
        .source(source)?
        .witness
        .stages()
        .get(map_index)
        .ok_or(AkitaError::InvalidProof)?;
    emit_packed_negative_binary(out, span, packed)
}

fn emit_reduced_witness_tail<F: Field + CanonicalEncoding>(
    out: &mut PackedSignedDigitWriter,
    layout: &WitnessLayout,
    compression: Option<&CompressionWitnessMaterialization<F>>,
) -> Result<(), AkitaError> {
    if !layout.r_rows().is_empty() || layout.quotient_depth().is_some() {
        return Err(AkitaError::InvalidProof);
    }
    for layer in layout.compression_layers() {
        let compression = compression.ok_or(AkitaError::InvalidProof)?;
        for (group_index, _) in layer.f_spans() {
            emit_compression_witness_event(
                out,
                layout,
                compression,
                CompressionSourceId::Outer {
                    group_index: *group_index,
                },
                layer.map_index(),
            )?;
        }
        emit_compression_witness_event(
            out,
            layout,
            compression,
            CompressionSourceId::Opening,
            layer.map_index(),
        )?;
    }
    Ok(())
}
