//! Canonical physical emission of recursive witness coefficient planes.

use akita_error::AkitaError;

use crate::proof::DigitBlocks;
use crate::{WitnessLayout, WitnessUnitLayout};

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

/// Emit one ownership unit's replicated Z planes at canonical addresses.
pub fn emit_witness_z_planes<const D_SOURCE: usize>(
    out: &mut impl WitnessCoefficientSink,
    unit: &WitnessUnitLayout,
    num_positions_per_block: usize,
    depth_commit: usize,
    depth_fold: usize,
    all_planes: &[[i8; D_SOURCE]],
) -> Result<(), AkitaError> {
    let expected = num_positions_per_block
        .checked_mul(depth_commit)
        .and_then(|n| n.checked_mul(depth_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z source length overflow".into()))?;
    if all_planes.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: all_planes.len(),
        });
    }
    for position in 0..num_positions_per_block {
        for commit_digit in 0..depth_commit {
            for fold_digit in 0..depth_fold {
                let source = (position * depth_commit + commit_digit) * depth_fold + fold_digit;
                out.write_coefficients(
                    unit.z_coefficient_index(
                        D_SOURCE,
                        num_positions_per_block,
                        depth_commit,
                        depth_fold,
                        position,
                        commit_digit,
                        fold_digit,
                        0,
                    )?,
                    &all_planes[source],
                )?;
            }
        }
    }
    Ok(())
}

/// Emit the shared R planes at canonical witness addresses.
pub fn emit_witness_r_planes<const D: usize>(
    out: &mut impl WitnessCoefficientSink,
    layout: &WitnessLayout,
    quotient_depth: usize,
    planes: &[[i8; D]],
) -> Result<(), AkitaError> {
    if layout.r_rows().iter().any(|row| {
        row.geometry().polynomial_modulus_dimension() != D
            || row.geometry().coordinate_plane_count() != 1
    }) || Some(quotient_depth) != layout.quotient_depth()
    {
        return Err(AkitaError::InvalidSetup(
            "witness R source shape is malformed".into(),
        ));
    }
    let expected = layout
        .r_rows()
        .len()
        .checked_mul(quotient_depth)
        .ok_or_else(|| AkitaError::InvalidSetup("witness R source shape overflow".into()))?;
    if planes.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: planes.len(),
        });
    }
    for row in 0..layout.r_rows().len() {
        for digit in 0..quotient_depth {
            out.write_coefficients(
                layout.r_coefficient_index(row, digit, 0, 0)?,
                &planes[row * quotient_depth + digit],
            )?;
        }
    }
    Ok(())
}
