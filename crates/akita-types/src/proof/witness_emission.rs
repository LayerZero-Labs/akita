//! Canonical physical emission of recursive witness coefficient planes.

use akita_error::AkitaError;

use crate::proof::DigitBlocks;
use crate::WitnessUnitLayout;

/// Destination for canonical witness coefficient emission.
pub trait WitnessCoefficientSink {
    /// Write one contiguous coefficient plane at its physical witness offset.
    fn write_coefficients(&mut self, start: usize, coefficients: &[i8]) -> Result<(), AkitaError>;
}

impl WitnessCoefficientSink for [i8] {
    fn write_coefficients(&mut self, start: usize, coefficients: &[i8]) -> Result<(), AkitaError> {
        let end = start
            .checked_add(coefficients.len())
            .ok_or_else(|| AkitaError::InvalidSetup("witness coefficient end overflow".into()))?;
        self.get_mut(start..end)
            .ok_or(AkitaError::InvalidProof)?
            .copy_from_slice(coefficients);
        Ok(())
    }
}

impl WitnessCoefficientSink for Vec<i8> {
    fn write_coefficients(&mut self, start: usize, coefficients: &[i8]) -> Result<(), AkitaError> {
        self.as_mut_slice().write_coefficients(start, coefficients)
    }
}

/// Emit one group's role-native E planes at canonical witness addresses.
#[allow(clippy::too_many_arguments)]
pub fn emit_witness_e_planes<const D_ROLE: usize>(
    out: &mut impl WitnessCoefficientSink,
    unit: &WitnessUnitLayout,
    source_physical_width: usize,
    num_claims: usize,
    depth_open: usize,
    digits: &DigitBlocks,
    source_num_live_blocks: usize,
) -> Result<(), AkitaError> {
    if !source_physical_width.is_multiple_of(D_ROLE) {
        return Err(AkitaError::InvalidSetup(
            "witness E dimensions must satisfy D_ROLE | D_A".into(),
        ));
    }
    digits.ensure_stride::<D_ROLE>()?;
    let role_subcolumns = source_physical_width / D_ROLE;
    let expected = num_claims
        .checked_mul(source_num_live_blocks)
        .and_then(|n| n.checked_mul(role_subcolumns))
        .and_then(|n| n.checked_mul(depth_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness E source length overflow".into()))?;
    if digits.total_planes() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: digits.total_planes(),
        });
    }
    let flat = digits.typed_planes::<D_ROLE>()?;
    if unit.e_geometry().physical_coefficient_width() != source_physical_width {
        return Err(AkitaError::InvalidSetup(
            "witness E source width disagrees with resolved geometry".into(),
        ));
    }
    let planes_per_block = role_subcolumns
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("witness E source stride overflow".into()))?;
    if unit.num_live_blocks() == 0 {
        return Ok(());
    }
    for claim in 0..num_claims {
        let source =
            unit_claim_planes(unit, claim, source_num_live_blocks, planes_per_block, flat)?;
        let destination = unit.e_coefficient_index(
            D_ROLE,
            num_claims,
            depth_open,
            claim,
            unit.global_block_start(),
            0,
            0,
            0,
        )?;
        out.write_coefficients(destination, source.as_flattened())?;
    }
    Ok(())
}

/// Emit one group's role-native T planes at canonical witness addresses.
#[allow(clippy::too_many_arguments)]
pub fn emit_witness_t_planes<const D_A: usize, const D_ROLE: usize>(
    out: &mut impl WitnessCoefficientSink,
    unit: &WitnessUnitLayout,
    num_claims: usize,
    n_a: usize,
    depth_outer: usize,
    digits: &DigitBlocks,
    source_num_live_blocks: usize,
) -> Result<(), AkitaError> {
    if !D_A.is_multiple_of(D_ROLE) {
        return Err(AkitaError::InvalidSetup(
            "witness T dimensions must satisfy D_ROLE | D_A".into(),
        ));
    }
    digits.ensure_stride::<D_ROLE>()?;
    let role_subcolumns = D_A / D_ROLE;
    let expected = num_claims
        .checked_mul(source_num_live_blocks)
        .and_then(|n| n.checked_mul(n_a))
        .and_then(|n| n.checked_mul(role_subcolumns))
        .and_then(|n| n.checked_mul(depth_outer))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T source length overflow".into()))?;
    if digits.total_planes() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: digits.total_planes(),
        });
    }
    let flat = digits.typed_planes::<D_ROLE>()?;
    let planes_per_block = n_a
        .checked_mul(role_subcolumns)
        .and_then(|n| n.checked_mul(depth_outer))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T source stride overflow".into()))?;
    if unit.num_live_blocks() == 0 {
        return Ok(());
    }
    for claim in 0..num_claims {
        let source =
            unit_claim_planes(unit, claim, source_num_live_blocks, planes_per_block, flat)?;
        let destination = unit.t_coefficient_index(
            D_A,
            D_ROLE,
            num_claims,
            n_a,
            depth_outer,
            claim,
            unit.global_block_start(),
            0,
            0,
            0,
            0,
        )?;
        out.write_coefficients(destination, source.as_flattened())?;
    }
    Ok(())
}

/// The source planes of `claim` for the blocks `unit` owns.
///
/// Inside a claim, source and unit planes share the order (block, row,
/// subcolumn, digit), and a unit owns consecutive blocks, so these planes fill
/// one contiguous run of the unit's range from the claim's first owned plane.
fn unit_claim_planes<'a, const D: usize>(
    unit: &WitnessUnitLayout,
    claim: usize,
    source_num_live_blocks: usize,
    planes_per_block: usize,
    flat: &'a [[i8; D]],
) -> Result<&'a [[i8; D]], AkitaError> {
    let blocks = unit.global_block_range();
    if blocks.end > source_num_live_blocks {
        return Err(AkitaError::InvalidSize {
            expected: source_num_live_blocks,
            actual: blocks.end,
        });
    }
    let start = claim
        .checked_mul(source_num_live_blocks)
        .and_then(|block| block.checked_add(blocks.start))
        .and_then(|block| block.checked_mul(planes_per_block));
    let count = blocks.len().checked_mul(planes_per_block);
    start
        .zip(count)
        .and_then(|(start, count)| flat.get(start..start.checked_add(count)?))
        .ok_or_else(|| AkitaError::InvalidSetup("witness source planes exceed the source".into()))
}
