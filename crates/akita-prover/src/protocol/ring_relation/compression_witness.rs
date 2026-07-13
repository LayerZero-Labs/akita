//! Materialization of the heterogeneous F/H compression witness suffix.
//!
//! This module consumes the canonical [`RelationLayout`] graph. It does not
//! replay catalog geometry: row families select native rings and matrix shapes,
//! segment spans select storage, and gadget edges select digit bases.

use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::layout::{
    CoeffSpan, RelationRowFamily, RelationRowInputs, RelationRowRhs, RelationSegmentId,
};
use akita_types::{
    compression_digit_depth, dispatch_for_field, CompressionAlphabet, CompressionSourceId,
    RelationLayout, RelationRowId,
};

use crate::compute::{
    CompressionRowsItem, CompressionRowsMode, CompressionRowsPlan, CyclicRowsComputeBackend,
    OperationCtx,
};
use crate::validation::validate_i8_setup_log_basis;

/// Unpadded compression suffix and terminal compression images.
#[allow(dead_code)] // Reached by the atomic schedule/proof cutover.
pub(crate) struct CompressionWitnessMaterialization<F> {
    /// Physical field-coefficient offset at which this suffix is appended.
    pub(crate) suffix_start: usize,
    /// Canonical i8 digits for every compression input and quotient span.
    pub(crate) suffix_digits: Vec<i8>,
    pub(crate) terminal_payloads: Vec<(CompressionSourceId, Vec<F>)>,
}

fn compression_families(layout: &RelationLayout) -> Vec<&RelationRowFamily> {
    layout
        .row_plan()
        .families()
        .iter()
        .filter(|family| matches!(family.id(), RelationRowId::Compression { .. }))
        .collect()
}

fn source_family_len(
    layout: &RelationLayout,
    source: CompressionSourceId,
) -> Result<usize, AkitaError> {
    let family = layout
        .row_plan()
        .families()
        .iter()
        .find(|family| {
            matches!(
                (source, family.id()),
                (
                    CompressionSourceId::CurrentOuter,
                    RelationRowId::B {
                        group: akita_types::RelationGroupId::Current
                    }
                ) | (CompressionSourceId::Opening, RelationRowId::D)
            ) || matches!(
                (source, family.id()),
                (
                    CompressionSourceId::PrecommittedOuter { index },
                    RelationRowId::B {
                        group: akita_types::RelationGroupId::Precommitted {
                            index: group_index
                        }
                    }
                ) if index == group_index
            )
        })
        .ok_or_else(|| AkitaError::InvalidSetup("compression source family is absent".into()))?;
    family
        .rows()
        .len()
        .checked_mul(family.native_ring_dim())
        .ok_or_else(|| AkitaError::InvalidSetup("compression source image length overflow".into()))
}

fn input_log_basis(
    layout: &RelationLayout,
    source: CompressionSourceId,
    map: usize,
    input: RelationSegmentId,
) -> Result<u32, AkitaError> {
    let edge = if map == 0 {
        layout.row_plan().families().iter().find_map(|family| {
            let matches_source = match (source, family.id()) {
                (
                    CompressionSourceId::CurrentOuter,
                    RelationRowId::B {
                        group: akita_types::RelationGroupId::Current,
                    },
                )
                | (CompressionSourceId::Opening, RelationRowId::D) => true,
                (
                    CompressionSourceId::PrecommittedOuter { index },
                    RelationRowId::B {
                        group: akita_types::RelationGroupId::Precommitted { index: group_index },
                    },
                ) => index == group_index,
                _ => false,
            };
            if !matches_source {
                return None;
            }
            match family.inputs() {
                RelationRowInputs::B {
                    compression_input, ..
                }
                | RelationRowInputs::D {
                    compression_input, ..
                } => *compression_input,
                _ => None,
            }
        })
    } else {
        layout.row_plan().families().iter().find_map(|family| {
            if family.id()
                != (RelationRowId::Compression {
                    source,
                    map: map - 1,
                })
            {
                return None;
            }
            match family.inputs() {
                RelationRowInputs::Compression { successor, .. } => *successor,
                _ => None,
            }
        })
    }
    .ok_or_else(|| AkitaError::InvalidSetup("compression input gadget edge is absent".into()))?;
    if edge.segment() != input {
        return Err(AkitaError::InvalidSetup(
            "compression input gadget edge targets the wrong segment".into(),
        ));
    }
    Ok(edge.log_basis())
}

fn is_negative_binary(
    layout: &RelationLayout,
    input: RelationSegmentId,
) -> Result<bool, AkitaError> {
    let span = layout.segment(input)?.span();
    let start = span.start();
    let end = start
        .checked_add(span.len())
        .ok_or_else(|| AkitaError::InvalidSetup("compression input span overflow".into()))?;
    Ok(layout.negative_binary_support().iter().any(|support| {
        let support_end = support.start().saturating_add(support.len());
        support.start() <= start && end <= support_end
    }))
}

fn decompose_input<F: FieldCore + CanonicalField>(
    values: &[F],
    depth: usize,
    log_basis: u32,
    negative_binary: bool,
) -> Result<Vec<i8>, AkitaError> {
    if depth == 0 || log_basis == 0 || (negative_binary && log_basis != 1) {
        return Err(AkitaError::InvalidSetup(
            "compression digit decomposition parameters are invalid".into(),
        ));
    }
    let len = values
        .len()
        .checked_mul(depth)
        .ok_or_else(|| AkitaError::InvalidSetup("compression digit length overflow".into()))?;
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let mask = (1u128 << log_basis) - 1;
    let mut digits = Vec::with_capacity(len);
    for digit in 0..depth {
        let shift = (digit as u32)
            .checked_mul(log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("compression digit shift overflow".into()))?;
        for value in values {
            let canonical = value.to_canonical_u128();
            let magnitude = if negative_binary && canonical != 0 {
                modulus - canonical
            } else {
                canonical
            };
            let raw = if shift >= 128 {
                0
            } else {
                (magnitude >> shift) & mask
            };
            let signed = if negative_binary {
                -(raw as i8)
            } else {
                raw as i8
            };
            digits.push(signed);
        }
    }
    Ok(digits)
}

fn place(
    dst: &mut [i8],
    suffix_start: usize,
    span: CoeffSpan,
    src: &[i8],
) -> Result<(), AkitaError> {
    if span.len() != src.len() {
        return Err(AkitaError::InvalidSize {
            expected: span.len(),
            actual: src.len(),
        });
    }
    let start = span.start().checked_sub(suffix_start).ok_or_else(|| {
        AkitaError::InvalidSetup("physical compression span precedes suffix".into())
    })?;
    let end = start.checked_add(span.len()).ok_or_else(|| {
        AkitaError::InvalidSetup("physical compression suffix span overflow".into())
    })?;
    let target = dst.get_mut(start..end).ok_or_else(|| {
        AkitaError::InvalidSetup("physical compression span exceeds witness".into())
    })?;
    target.copy_from_slice(src);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_bucket<F, B, const D: usize>(
    ctx: &OperationCtx<'_, F, B>,
    layout: &RelationLayout,
    families: &[&RelationRowFamily],
    digits: &[Vec<i8>],
    suffix_start: usize,
    suffix: &mut [i8],
    next_images: &mut [(CompressionSourceId, Vec<F>)],
    terminal_payloads: &mut [(CompressionSourceId, Option<Vec<F>>)],
    level_log_basis: u32,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    B: CyclicRowsComputeBackend<F>,
{
    let first_family = families
        .first()
        .ok_or_else(|| AkitaError::InvalidSetup("compression execution bucket is empty".into()))?;
    let first_digits = digits
        .first()
        .ok_or_else(|| AkitaError::InvalidSetup("compression digit bucket is empty".into()))?;
    let row_count = first_family.rows().len();
    let column_count = first_digits.len() / D;
    let typed = digits
        .iter()
        .map(|flat| {
            let (rings, remainder) = flat.as_slice().as_chunks::<D>();
            if !remainder.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "compression digit span is not divisible by its native ring".into(),
                ));
            }
            Ok(rings)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    if typed.iter().any(|rows| rows.len() != column_count) {
        return Err(AkitaError::InvalidSetup(
            "compression digit ring view is malformed".into(),
        ));
    }
    let items = typed
        .iter()
        .zip(families)
        .map(|(rows, family)| {
            let input = compression_input(family)?;
            Ok(CompressionRowsItem {
                digits: rows,
                digit_abs_bound: if is_negative_binary(layout, input)? {
                    1
                } else {
                    let (source, map) = compression_id(family)?;
                    (1u64 << input_log_basis(layout, source, map, input)?) - 1
                },
                mode: CompressionRowsMode::EagerPaired,
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let outputs = ctx.backend().compression_rows(
        ctx.prepared(),
        CompressionRowsPlan {
            row_count,
            column_count,
            items: &items,
        },
    )?;
    if outputs.len() != families.len() {
        return Err(AkitaError::InvalidSetup(
            "compression backend output count disagrees with its bucket".into(),
        ));
    }
    let quotient_levels = akita_types::r_decomp_levels::<F>(level_log_basis);
    let modulus = (-F::one()).to_canonical_u128() + 1;
    let quotient_params =
        BalancedDecomposePow2I8Params::new(quotient_levels, level_log_basis, modulus);
    for (item_index, (family, output)) in families.iter().zip(outputs).enumerate() {
        let (source, _) = compression_id(family)?;
        let input_id = compression_input(family)?;
        let input_span = layout.physical_compression_segment_span(input_id)?;
        place(suffix, suffix_start, input_span, &digits[item_index])?;

        let neg = output.u_neg.ok_or_else(|| {
            AkitaError::InvalidSetup("compression negacyclic output is absent".into())
        })?;
        let quotient = output.quotient.ok_or_else(|| {
            AkitaError::InvalidSetup("compression quotient output is absent".into())
        })?;
        let neg_flat = neg
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .collect::<Vec<_>>();
        let mut quotient_digits = Vec::with_capacity(quotient.len() * quotient_levels * D);
        let mut planes = vec![[0i8; D]; quotient_levels];
        for ring in &quotient {
            planes.fill([0; D]);
            ring.balanced_decompose_pow2_i8_into_with_params(&mut planes, &quotient_params);
            quotient_digits.extend(planes.iter().flat_map(|plane| plane.iter()).copied());
        }
        place(
            suffix,
            suffix_start,
            layout.physical_compression_segment_span(family.quotient())?,
            &quotient_digits,
        )?;
        if let RelationRowRhs::TerminalPayload { coeffs } = family.rhs() {
            if coeffs != neg_flat.len() {
                return Err(AkitaError::InvalidSetup(
                    "terminal compression payload length disagrees with its row family".into(),
                ));
            }
            let payload = terminal_payloads
                .iter_mut()
                .find(|(candidate, _)| *candidate == source)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "terminal compression source has no canonical payload slot".into(),
                    )
                })?;
            if payload.1.replace(neg_flat).is_some() {
                return Err(AkitaError::InvalidSetup(
                    "compression source produced more than one terminal payload".into(),
                ));
            }
        } else {
            let image = next_images
                .iter_mut()
                .find(|(candidate, _)| *candidate == source)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("compression source image is absent".into())
                })?;
            image.1 = neg_flat;
        }
    }
    Ok(())
}

fn compression_id(family: &RelationRowFamily) -> Result<(CompressionSourceId, usize), AkitaError> {
    match family.id() {
        RelationRowId::Compression { source, map } => Ok((source, map)),
        _ => Err(AkitaError::InvalidSetup(
            "compression bucket contains a non-compression row family".into(),
        )),
    }
}

fn compression_input(family: &RelationRowFamily) -> Result<RelationSegmentId, AkitaError> {
    match family.inputs() {
        RelationRowInputs::Compression { input, .. } => Ok(*input),
        _ => Err(AkitaError::InvalidSetup(
            "compression row family has non-compression inputs".into(),
        )),
    }
}

/// Materialize every compression input and quotient into the physical witness.
#[allow(dead_code)] // Reached by the atomic schedule/proof cutover.
pub(crate) fn materialize_compression_witness<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    layout: &RelationLayout,
    source_images: &[(CompressionSourceId, &[F])],
    level_log_basis: u32,
) -> Result<CompressionWitnessMaterialization<F>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    B: CyclicRowsComputeBackend<F>,
{
    let families = compression_families(layout);
    if families.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "compression witness has no row families".into(),
        ));
    }
    let canonical_sources = families
        .iter()
        .filter_map(|family| match family.id() {
            RelationRowId::Compression { source, map: 0 } => Some(source),
            _ => None,
        })
        .collect::<Vec<_>>();
    if canonical_sources.is_empty()
        || canonical_sources
            .iter()
            .enumerate()
            .any(|(index, source)| canonical_sources[..index].contains(source))
    {
        return Err(AkitaError::InvalidSetup(
            "compression layout has an invalid canonical map-zero source set".into(),
        ));
    }
    let mut images = Vec::with_capacity(source_images.len());
    for &(source, coeffs) in source_images {
        if !canonical_sources.contains(&source) {
            return Err(AkitaError::InvalidInput(
                "extraneous compression source image".into(),
            ));
        }
        if images.iter().any(|(candidate, _)| *candidate == source) {
            return Err(AkitaError::InvalidInput(
                "duplicate compression source image".into(),
            ));
        }
        let expected = source_family_len(layout, source)?;
        if coeffs.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: coeffs.len(),
            });
        }
        images.push((source, coeffs.to_vec()));
    }
    if canonical_sources
        .iter()
        .any(|source| !images.iter().any(|(candidate, _)| candidate == source))
    {
        return Err(AkitaError::InvalidInput(
            "canonical compression source image is absent".into(),
        ));
    }
    validate_i8_setup_log_basis(level_log_basis, "for compression witness quotients")?;
    let first_segment = layout
        .segments()
        .iter()
        .find(|segment| matches!(segment.id(), RelationSegmentId::CompressionInput { .. }))
        .ok_or_else(|| AkitaError::InvalidSetup("compression input segment is absent".into()))?;
    let suffix_start = layout
        .physical_compression_segment_span(first_segment.id())?
        .start();
    let suffix_len = layout
        .segments()
        .iter()
        .filter(|segment| {
            matches!(
                segment.id(),
                RelationSegmentId::CompressionInput { .. }
                    | RelationSegmentId::CompressionQuotient { .. }
            )
        })
        .try_fold(0usize, |total, segment| {
            total.checked_add(segment.span().len()).ok_or_else(|| {
                AkitaError::InvalidSetup("compression suffix length overflow".into())
            })
        })?;
    let mut suffix = vec![0i8; suffix_len];
    let mut terminal_payloads = Vec::new();
    for family in &families {
        let (source, map) = compression_id(family)?;
        if map != 0 {
            continue;
        }
        if terminal_payloads
            .iter()
            .any(|(candidate, _)| *candidate == source)
        {
            return Err(AkitaError::InvalidSetup(
                "compression layout has duplicate map-zero source families".into(),
            ));
        }
        terminal_payloads.push((source, None));
    }
    let mut cursor = 0;
    while cursor < families.len() {
        let (_, map) = compression_id(families[cursor])?;
        let d = families[cursor].native_ring_dim();
        let rows = families[cursor].rows().len();
        let input = compression_input(families[cursor])?;
        if d == 0 || !layout.segment(input)?.span().len().is_multiple_of(d) {
            return Err(AkitaError::InvalidSetup(
                "compression input has invalid native-ring projection".into(),
            ));
        }
        let cols = layout.segment(input)?.span().len() / d;
        let mut end = cursor + 1;
        while end < families.len() {
            let (_, candidate_map) = compression_id(families[end])?;
            let candidate_input = compression_input(families[end])?;
            if candidate_map != map
                || families[end].native_ring_dim() != d
                || families[end].rows().len() != rows
                || layout.segment(candidate_input)?.span().len() / d != cols
            {
                break;
            }
            end += 1;
        }
        let bucket = &families[cursor..end];
        let mut bucket_digits = Vec::with_capacity(bucket.len());
        for family in bucket {
            let (source, map) = compression_id(family)?;
            let input = compression_input(family)?;
            let span_len = layout.segment(input)?.span().len();
            let log_basis = input_log_basis(layout, source, map, input)?;
            let negative = is_negative_binary(layout, input)?;
            validate_i8_setup_log_basis(log_basis, "for compression input")?;
            let values = &images
                .iter()
                .find(|(candidate, _)| *candidate == source)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("compression source image is absent".into())
                })?
                .1;
            let depth = span_len
                .checked_div(values.len())
                .filter(|_| !values.is_empty() && span_len.is_multiple_of(values.len()))
                .ok_or(AkitaError::InvalidSize {
                    expected: span_len,
                    actual: values.len(),
                })?;
            let expected_depth = compression_digit_depth(
                if negative {
                    CompressionAlphabet::NegativeBinary
                } else {
                    CompressionAlphabet::OpeningBase { log_basis }
                },
                F::modulus_bits(),
                level_log_basis,
            )?;
            if depth != expected_depth {
                return Err(AkitaError::InvalidSetup(
                    "compression input depth disagrees with its authenticated alphabet".into(),
                ));
            }
            if values.len().checked_mul(depth) != Some(span_len) {
                return Err(AkitaError::InvalidSize {
                    expected: span_len / depth,
                    actual: values.len(),
                });
            }
            bucket_digits.push(decompose_input(values, depth, log_basis, negative)?);
        }
        dispatch_for_field!(akita_types::ProtocolDispatchSlot::Compression, F, d, |D| {
            execute_bucket::<F, B, D>(
                ctx,
                layout,
                bucket,
                &bucket_digits,
                suffix_start,
                &mut suffix,
                &mut images,
                &mut terminal_payloads,
                level_log_basis,
            )
        })?;
        cursor = end;
    }
    let terminal_payloads = terminal_payloads
        .into_iter()
        .map(|(source, payload)| {
            payload.map(|payload| (source, payload)).ok_or_else(|| {
                AkitaError::InvalidSetup("compression source produced no terminal payload".into())
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    Ok(CompressionWitnessMaterialization {
        suffix_start,
        suffix_digits: suffix,
        terminal_payloads,
    })
}

#[cfg(test)]
mod tests;
