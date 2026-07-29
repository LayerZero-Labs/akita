//! Standalone parameter selection for shadow compressed commitments.
//!
//! This module does not participate in schedule search or catalog emission.
//! It selects only the negative-binary F/H ladder exercised by the prover's
//! opt-in diagnostic mode.

use akita_field::AkitaError;
use akita_types::sis::compression::{
    min_compression_secure_rank, COMPRESSION_SIS_COEFF_LINF_BOUND,
};
use akita_types::sis::{SisModulusProfileId, DEFAULT_SIS_SECURITY_POLICY};

/// Maximum uncompressed B/D image size handled by the diagnostic.
pub const MAX_COMPRESSION_INPUT_BYTES: usize = 16 * 1024;

/// Target terminal compressed commitment size.
pub const COMPRESSION_TARGET_BYTES: usize = 128;

/// Maximum number of F/H maps in the diagnostic ladder.
pub const MAX_COMPRESSION_MAPS: usize = 3;

/// One selected negative-binary F/H map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressionDiagnosticMap {
    /// Ring dimension of this F/H matrix.
    pub ring_dimension: usize,
    /// Number of input ring columns after negative-binary decomposition.
    pub input_width: usize,
    /// SIS-secure output module rank.
    pub output_rank: usize,
    /// Number of field coefficients in this map's image.
    pub output_coefficients: usize,
}

/// Complete shadow-compression plan for one B or D image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressionDiagnosticPlan {
    /// Exact SIS modulus profile.
    pub modulus_profile: SisModulusProfileId,
    /// Canonical field bit length.
    pub field_bits: usize,
    /// Canonical field byte length.
    pub field_bytes: usize,
    /// Number of coefficients in the unsliced source image.
    pub source_coefficients: usize,
    /// Selected negative-binary maps for the complete source image.
    pub maps: Vec<CompressionDiagnosticMap>,
}

fn profile_geometry(profile: SisModulusProfileId) -> (usize, usize) {
    match profile {
        SisModulusProfileId::Q128OffsetA7F7 => (128, 16),
        SisModulusProfileId::Q64Offset59 => (64, 32),
        SisModulusProfileId::Q32Offset99 => (32, 64),
    }
}

fn select_maps(
    profile: SisModulusProfileId,
    field_bits: usize,
    field_bytes: usize,
    first_ring_dimension: usize,
    source_coefficients: usize,
) -> Result<Vec<CompressionDiagnosticMap>, AkitaError> {
    let mut input_coefficients = source_coefficients;
    let mut maps = Vec::with_capacity(MAX_COMPRESSION_MAPS);
    for map_index in 0..MAX_COMPRESSION_MAPS {
        let ring_dimension =
            first_ring_dimension
                .checked_shr(u32::try_from(map_index).map_err(|_| {
                    AkitaError::InvalidSetup("compression map index overflow".into())
                })?)
                .filter(|&dimension| dimension > 0)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("compression ring dimension underflow".into())
                })?;
        let digit_coefficients = input_coefficients
            .checked_mul(field_bits)
            .ok_or_else(|| AkitaError::InvalidSetup("compression digit length overflow".into()))?;
        let input_width = digit_coefficients.div_ceil(ring_dimension);
        let output_rank = min_compression_secure_rank(
            DEFAULT_SIS_SECURITY_POLICY,
            profile,
            u32::try_from(ring_dimension).map_err(|_| {
                AkitaError::InvalidSetup("compression ring dimension overflow".into())
            })?,
            COMPRESSION_SIS_COEFF_LINF_BOUND,
            u64::try_from(input_width)
                .map_err(|_| AkitaError::InvalidSetup("compression width overflow".into()))?,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "no compression SIS rank for profile={profile:?} d={ring_dimension} width={input_width}"
            ))
        })?;
        let output_coefficients = output_rank
            .checked_mul(ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("compression output length overflow".into()))?;
        let output_bytes = output_coefficients
            .checked_mul(field_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("compression output bytes overflow".into()))?;
        maps.push(CompressionDiagnosticMap {
            ring_dimension,
            input_width,
            output_rank,
            output_coefficients,
        });
        if output_bytes == COMPRESSION_TARGET_BYTES {
            return Ok(maps);
        }
        if output_bytes < COMPRESSION_TARGET_BYTES {
            return Err(AkitaError::InvalidSetup(
                "compression ladder undershot its terminal byte target".into(),
            ));
        }
        input_coefficients = output_coefficients;
    }
    Err(AkitaError::InvalidSetup(format!(
        "compression ladder did not reach {COMPRESSION_TARGET_BYTES} bytes"
    )))
}

/// Select the standalone negative-binary diagnostic plan for one B/D image.
///
/// # Errors
///
/// Returns an error for an empty or larger-than-16-KiB source, or when the
/// narrow compression SIS table cannot price every map.
pub fn plan_compression_diagnostic(
    modulus_profile: SisModulusProfileId,
    source_coefficients: usize,
) -> Result<CompressionDiagnosticPlan, AkitaError> {
    if source_coefficients == 0 {
        return Err(AkitaError::InvalidInput(
            "compression diagnostic source must be nonempty".into(),
        ));
    }
    let (field_bits, standard_first_ring_dimension) = profile_geometry(modulus_profile);
    let field_bytes = field_bits.div_ceil(8);
    let source_bytes = source_coefficients
        .checked_mul(field_bytes)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("compression source byte length overflow".into())
        })?;
    if source_bytes > MAX_COMPRESSION_INPUT_BYTES {
        return Err(AkitaError::InvalidInput(format!(
            "compression diagnostic source is {source_bytes} bytes, exceeding the {MAX_COMPRESSION_INPUT_BYTES}-byte maximum"
        )));
    }
    let first_ring_dimension = if source_bytes > MAX_COMPRESSION_INPUT_BYTES / 2 {
        standard_first_ring_dimension
            .checked_mul(2)
            .ok_or_else(|| AkitaError::InvalidSetup("compression ring dimension overflow".into()))?
    } else {
        standard_first_ring_dimension
    };
    let maps = select_maps(
        modulus_profile,
        field_bits,
        field_bytes,
        first_ring_dimension,
        source_coefficients,
    )?;
    Ok(CompressionDiagnosticPlan {
        modulus_profile,
        field_bits,
        field_bytes,
        source_coefficients,
        maps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_through_eight_kib_inputs_use_two_maps() {
        for (profile, field_bytes) in [
            (SisModulusProfileId::Q128OffsetA7F7, 16),
            (SisModulusProfileId::Q64Offset59, 8),
            (SisModulusProfileId::Q32Offset99, 4),
        ] {
            for input_kib in [1, 2, 4, 8] {
                let plan = plan_compression_diagnostic(profile, input_kib * 1024 / field_bytes)
                    .expect("plan");
                assert_eq!(plan.maps.len(), 2);
                assert!(plan.maps.iter().all(|map| map.output_rank == 1));
                assert_eq!(plan.maps[0].output_coefficients * field_bytes, 256);
                assert_eq!(plan.maps[1].output_coefficients * field_bytes, 128);
            }
        }
    }

    #[test]
    fn sixteen_kib_inputs_use_three_maps() {
        for (profile, field_bytes) in [
            (SisModulusProfileId::Q128OffsetA7F7, 16),
            (SisModulusProfileId::Q64Offset59, 8),
            (SisModulusProfileId::Q32Offset99, 4),
        ] {
            let plan = plan_compression_diagnostic(profile, 16 * 1024 / field_bytes).expect("plan");
            assert_eq!(plan.maps.len(), 3);
            assert_eq!(
                plan.maps
                    .iter()
                    .map(|map| map.ring_dimension)
                    .collect::<Vec<_>>(),
                match profile {
                    SisModulusProfileId::Q128OffsetA7F7 => [32, 16, 8],
                    SisModulusProfileId::Q64Offset59 => [64, 32, 16],
                    SisModulusProfileId::Q32Offset99 => [128, 64, 32],
                }
            );
            assert!(plan.maps.iter().all(|map| map.output_rank == 1));
            assert_eq!(
                plan.maps
                    .iter()
                    .map(|map| map.output_coefficients * field_bytes)
                    .collect::<Vec<_>>(),
                [512, 256, 128]
            );
            let terminal = plan.maps.last().expect("terminal map");
            assert_eq!(
                terminal.output_coefficients * field_bytes,
                COMPRESSION_TARGET_BYTES
            );
        }
    }

    #[test]
    fn non_power_of_two_inputs_over_eight_kib_use_the_rank_one_ladder() {
        for (profile, field_bytes) in [
            (SisModulusProfileId::Q128OffsetA7F7, 16),
            (SisModulusProfileId::Q64Offset59, 8),
            (SisModulusProfileId::Q32Offset99, 4),
        ] {
            for source_bytes in [9 * 1024, 12 * 1024, 15 * 1024] {
                let plan =
                    plan_compression_diagnostic(profile, source_bytes / field_bytes).expect("plan");
                assert_eq!(plan.maps.len(), 3);
                assert!(plan.maps.iter().all(|map| map.output_rank == 1));
            }
        }
    }

    #[test]
    fn sources_above_sixteen_kib_are_rejected_not_sliced() {
        assert!(plan_compression_diagnostic(SisModulusProfileId::Q128OffsetA7F7, 1025).is_err());
    }
}
