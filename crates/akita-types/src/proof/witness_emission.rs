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
    for claim in 0..num_claims {
        for global_block in unit.global_block_range() {
            let semantic = claim * source_num_live_blocks + global_block;
            for role_subcolumn in 0..role_subcolumns {
                for digit in 0..depth_open {
                    let source = (semantic * role_subcolumns + role_subcolumn) * depth_open + digit;
                    let destination = unit.e_coefficient_index(
                        D_ROLE,
                        num_claims,
                        depth_open,
                        claim,
                        global_block,
                        role_subcolumn,
                        digit,
                        0,
                    )?;
                    out.write_coefficients(destination, &flat[source])?;
                }
            }
        }
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
    for claim in 0..num_claims {
        for global_block in unit.global_block_range() {
            for a_row in 0..n_a {
                for role_subcolumn in 0..role_subcolumns {
                    for digit in 0..depth_outer {
                        let source = (claim * source_num_live_blocks + global_block)
                            * planes_per_block
                            + (a_row * role_subcolumns + role_subcolumn) * depth_outer
                            + digit;
                        let destination = unit.t_coefficient_index(
                            D_A,
                            D_ROLE,
                            num_claims,
                            n_a,
                            depth_outer,
                            claim,
                            global_block,
                            a_row,
                            role_subcolumn,
                            digit,
                            0,
                        )?;
                        out.write_coefficients(destination, &flat[source])?;
                    }
                }
            }
        }
    }
    Ok(())
}
