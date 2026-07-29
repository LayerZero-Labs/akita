//! Opt-in execution of compressed commitments without protocol effects.

use crate::compute::{CompressionRowsPlan, DigitRowsComputeBackend, OperationCtx};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_planner::compression_diagnostic::{
    plan_compression_diagnostic, CompressionDiagnosticMap,
};
use akita_types::sis::SisModulusProfileId;
use akita_types::{
    dispatch_for_field, field_modulus, protocol_dispatch_tier, ProtocolRingDispatchTierId, RingVec,
};
use std::collections::BTreeMap;

/// Origin of one live B/D image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompressionDiagnosticSourceKind {
    Outer { group_index: usize },
    Opening,
}

/// One live B/D image handed to the diagnostic executor.
pub(crate) struct CompressionDiagnosticSource<'a, F> {
    pub(crate) kind: CompressionDiagnosticSourceKind,
    pub(crate) coefficients: &'a [F],
}

/// Aggregate facts emitted after shadow compression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CompressionDiagnosticReport {
    pub(crate) sources: usize,
    pub(crate) maps: usize,
    pub(crate) terminal_bytes: usize,
}

struct WorkItem<F> {
    source: CompressionDiagnosticSourceKind,
    field_bytes: usize,
    maps: Vec<CompressionDiagnosticMap>,
    coefficients: Vec<F>,
}

fn modulus_profile<F: CanonicalField>() -> Result<SisModulusProfileId, AkitaError> {
    let profile = match protocol_dispatch_tier::<F>() {
        ProtocolRingDispatchTierId::Fp128 => SisModulusProfileId::Q128OffsetA7F7,
        ProtocolRingDispatchTierId::Fp64 => SisModulusProfileId::Q64Offset59,
        ProtocolRingDispatchTierId::Fp32 => SisModulusProfileId::Q32Offset99,
    };
    if !profile.matches_modulus(field_modulus::<F>()) {
        return Err(AkitaError::InvalidSetup(format!(
            "compression diagnostic has no SIS profile for field modulus {}",
            field_modulus::<F>()
        )));
    }
    Ok(profile)
}

fn negative_binary_digits<F: CanonicalField, const D: usize>(
    coefficients: &[F],
    input_width: usize,
) -> Result<Vec<[i8; D]>, AkitaError> {
    let field_bits = usize::try_from(F::modulus_bits())
        .map_err(|_| AkitaError::InvalidSetup("field bit length overflow".into()))?;
    let digit_coefficients = coefficients
        .len()
        .checked_mul(field_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("compression digit length overflow".into()))?;
    let capacity = input_width
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("compression digit capacity overflow".into()))?;
    if digit_coefficients > capacity {
        return Err(AkitaError::InvalidSetup(
            "compression map input width is undersized".into(),
        ));
    }
    let modulus = field_modulus::<F>();
    let mut digits = vec![[0i8; D]; input_width];
    for bit in 0..field_bits {
        for (coefficient_index, coefficient) in coefficients.iter().enumerate() {
            let canonical = coefficient.to_canonical_u128();
            let magnitude = if canonical == 0 {
                0
            } else {
                modulus - canonical
            };
            let linear = bit * coefficients.len() + coefficient_index;
            digits[linear / D][linear % D] = -(((magnitude >> bit) & 1) as i8);
        }
    }
    Ok(digits)
}

fn execute_group<F, B, const D: usize>(
    ctx: &OperationCtx<'_, F, B>,
    items: &mut [WorkItem<F>],
    item_indices: &[usize],
    map_index: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    let first_map = items
        .get(*item_indices.first().ok_or_else(|| {
            AkitaError::InvalidSetup("compression execution group is empty".into())
        })?)
        .and_then(|item| item.maps.get(map_index))
        .copied()
        .ok_or_else(|| AkitaError::InvalidSetup("compression map is absent".into()))?;
    let digit_vectors = item_indices
        .iter()
        .map(|&item_index| {
            let item = items.get(item_index).ok_or_else(|| {
                AkitaError::InvalidSetup("compression item index is invalid".into())
            })?;
            negative_binary_digits::<F, D>(&item.coefficients, first_map.input_width)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let digit_views = digit_vectors.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let outputs = ctx.backend().compression_rows(
        ctx.prepared(),
        CompressionRowsPlan {
            output_rank: first_map.output_rank,
            digit_vectors: &digit_views,
        },
    )?;
    if outputs.len() != item_indices.len() {
        return Err(AkitaError::InvalidSetup(
            "compression backend returned the wrong batch length".into(),
        ));
    }
    for (&item_index, rows) in item_indices.iter().zip(outputs) {
        let coefficients = RingVec::from_ring_elems(&rows).coeffs().to_vec();
        if coefficients.len() != first_map.output_coefficients {
            return Err(AkitaError::InvalidSetup(
                "compression backend returned the wrong image length".into(),
            ));
        }
        items[item_index].coefficients = coefficients;
    }
    Ok(())
}

fn execute_stage<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    items: &mut [WorkItem<F>],
    map_index: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    let mut groups = BTreeMap::<(usize, usize, usize), Vec<usize>>::new();
    for (item_index, item) in items.iter().enumerate() {
        if let Some(map) = item.maps.get(map_index) {
            groups
                .entry((map.ring_dimension, map.input_width, map.output_rank))
                .or_default()
                .push(item_index);
        }
    }
    for ((ring_dimension, _, _), item_indices) in groups {
        dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Compression,
            F,
            ring_dimension,
            |D| execute_group::<F, B, D>(ctx, items, &item_indices, map_index)
        )?;
    }
    Ok(())
}

/// Compute compressed commitments for live B/D images and retain only metrics.
pub(crate) fn compute_shadow_compressed_commitments<F, B>(
    ctx: &OperationCtx<'_, F, B>,
    sources: &[CompressionDiagnosticSource<'_, F>],
) -> Result<CompressionDiagnosticReport, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    let profile = modulus_profile::<F>()?;
    let mut items = Vec::new();
    for source in sources {
        if source.coefficients.is_empty() {
            continue;
        }
        let plan = plan_compression_diagnostic(profile, source.coefficients.len())?;
        items.push(WorkItem {
            source: source.kind,
            field_bytes: plan.field_bytes,
            maps: plan.maps,
            coefficients: source.coefficients.to_vec(),
        });
    }
    let map_count = items.iter().map(|item| item.maps.len()).sum();
    let max_maps = items.iter().map(|item| item.maps.len()).max().unwrap_or(0);
    for map_index in 0..max_maps {
        execute_stage(ctx, &mut items, map_index)?;
    }
    let terminal_bytes = items.iter().try_fold(0usize, |total, item| {
        let bytes = item
            .coefficients
            .len()
            .checked_mul(item.field_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal byte length overflow".into()))?;
        tracing::debug!(
            source = ?item.source,
            maps = item.maps.len(),
            terminal_bytes = bytes,
            "computed shadow compressed commitment"
        );
        total
            .checked_add(bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal byte total overflow".into()))
    })?;
    Ok(CompressionDiagnosticReport {
        sources: items.len(),
        maps: map_count,
        terminal_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7;

    #[test]
    fn negative_binary_digits_reconstruct_the_input() {
        type F = Prime128OffsetA7F7;
        let values = [F::zero(), F::one(), F::from_u64(17), -F::from_u64(9)];
        let digits = negative_binary_digits::<F, 8>(&values, 64).expect("digits");
        for (coefficient_index, expected) in values.iter().copied().enumerate() {
            let mut actual = F::zero();
            let mut power = F::one();
            for bit in 0..F::modulus_bits() as usize {
                actual += F::from_i8(
                    digits[(bit * values.len() + coefficient_index) / 8]
                        [(bit * values.len() + coefficient_index) % 8],
                ) * power;
                power += power;
            }
            assert_eq!(actual, expected);
        }
    }
}
