//! Canonical SIS-occurrence derivation for expanded schedules.

use super::FoldSchedule;
use crate::sis::{
    InnerCommitMatrixParams, InnerCommitSecurityRoute, SisMatrixRole, SisModulusProfileId,
};
use crate::{CompressionChainPlan, CompressionMapPlan, GroupOpenPhaseParams};
use akita_error::AkitaError;

/// Collision bound attached to one scheduled SIS occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleSisBound {
    /// Coefficient-`L∞` collision bound.
    Linf(u128),
    /// Squared `L2` bound on the complete scalar collision vector.
    L2Squared(u128),
}

/// Protocol role of one SIS occurrence selected or derived by a schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleSisRole {
    /// Inner commitment matrix A.
    Inner,
    /// Outer commitment matrix B.
    Outer,
    /// Shared opening matrix D.
    Open,
    /// Rank-one compressed-commitment map.
    Compression,
}

impl From<SisMatrixRole> for ScheduleSisRole {
    fn from(role: SisMatrixRole) -> Self {
        match role {
            SisMatrixRole::Inner => Self::Inner,
            SisMatrixRole::Outer => Self::Outer,
            SisMatrixRole::Open => Self::Open,
        }
    }
}

/// Exact inert parameters of one SIS matrix occurrence in an expanded schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduleSisOccurrence {
    /// Stable human-readable location within the schedule.
    pub location: String,
    /// Protocol matrix role.
    pub role: ScheduleSisRole,
    /// Module output rank.
    pub output_rank: usize,
    /// Module input width.
    pub input_width: usize,
    /// Ring dimension.
    pub ring_dimension: usize,
    /// Exact modulus profile.
    pub modulus_profile: SisModulusProfileId,
    /// Collision bound selected by the schedule's audited security route.
    pub bound: ScheduleSisBound,
}

impl FoldSchedule {
    /// Derive every direct commitment and compression SIS occurrence selected
    /// by this schedule.
    ///
    /// This is the canonical occurrence topology shared by offline security
    /// diagnostics. It validates the complete schedule before returning any
    /// descriptors, so an occurrence report can never authenticate an
    /// inadmissible schedule shape.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError`] when the schedule is structurally invalid or a
    /// compressed payload cannot derive its canonical chain.
    pub fn sis_occurrences(&self) -> Result<Vec<ScheduleSisOccurrence>, AkitaError> {
        self.validate_structure()?;
        let mut occurrences = Vec::new();

        append_nonterminal_occurrences("root fold", &self.root.params, &mut occurrences)?;
        for (index, fold) in self.recursive_folds.iter().enumerate() {
            append_nonterminal_occurrences(
                &format!("recursive fold {index}"),
                &fold.params,
                &mut occurrences,
            )?;
        }
        occurrences.push(inner_occurrence(
            "terminal fold A".to_string(),
            &self.terminal.inner.matrix,
        ));
        Ok(occurrences)
    }
}

fn append_nonterminal_occurrences(
    location: &str,
    params: &crate::CommittedGroupParams,
    occurrences: &mut Vec<ScheduleSisOccurrence>,
) -> Result<(), AkitaError> {
    for (index, group) in params.preceding_group_iter().enumerate() {
        let group_location = if group.setup_natural_len.is_some() {
            format!("{location} setup prefix")
        } else {
            format!("{location} precommitted group {index}")
        };
        append_group_occurrences(&group_location, group, occurrences);
    }
    append_group_occurrences(
        &format!("{location} final group"),
        params.own_group(),
        occurrences,
    );
    occurrences.push(open_occurrence(
        format!("{location} shared D"),
        &params.open_matrix,
    ));

    if !params.payload_mode.is_compressed() {
        return Ok(());
    }

    append_group_compression_occurrences(
        &format!("{location} final group B"),
        params.own_group(),
        occurrences,
    )?;
    for (index, group) in params.preceding_group_iter().enumerate() {
        let group_location = if group.setup_natural_len.is_some() {
            format!("{location} setup prefix B")
        } else {
            format!("{location} precommitted group {index} B")
        };
        append_group_compression_occurrences(&group_location, group, occurrences)?;
    }
    append_compression_chain_occurrences(
        &format!("{location} shared D"),
        params.open_matrix.sis_modulus_profile(),
        params.opening_payload_geometry()?.source_coefficients(),
        occurrences,
    )
}

fn append_group_occurrences(
    location: &str,
    group: &GroupOpenPhaseParams,
    occurrences: &mut Vec<ScheduleSisOccurrence>,
) {
    occurrences.push(inner_occurrence(
        format!("{location} A"),
        &group.profile.inner.matrix,
    ));
    occurrences.push(outer_occurrence(
        format!("{location} B"),
        &group.profile.outer.matrix,
    ));
}

fn append_group_compression_occurrences(
    location: &str,
    group: &GroupOpenPhaseParams,
    occurrences: &mut Vec<ScheduleSisOccurrence>,
) -> Result<(), AkitaError> {
    let matrix = &group.profile.outer.matrix;
    let source_coefficients = group
        .profile
        .outer_slice_count
        .complete_source_coefficients(matrix.output_rank(), matrix.ring_dimension())?;
    append_compression_chain_occurrences(
        location,
        matrix.sis_modulus_profile(),
        source_coefficients,
        occurrences,
    )
}

fn append_compression_chain_occurrences(
    location: &str,
    modulus_profile: SisModulusProfileId,
    source_coefficients: usize,
    occurrences: &mut Vec<ScheduleSisOccurrence>,
) -> Result<(), AkitaError> {
    let plan = CompressionChainPlan::for_complete_source(modulus_profile, source_coefficients)?;
    occurrences.extend(plan.maps().iter().enumerate().map(|(index, map)| {
        compression_occurrence(format!("{location} compression map {index}"), *map)
    }));
    Ok(())
}

fn inner_occurrence(location: String, matrix: &InnerCommitMatrixParams) -> ScheduleSisOccurrence {
    let (modulus_profile, bound) = match matrix.security_route() {
        InnerCommitSecurityRoute::Linf(key) => (
            key.modulus_profile,
            ScheduleSisBound::Linf(key.coeff_linf_bound),
        ),
        InnerCommitSecurityRoute::L2 { table_key, .. } => (
            table_key.modulus_profile,
            ScheduleSisBound::L2Squared(table_key.collision_l2_sq),
        ),
    };
    ScheduleSisOccurrence {
        location,
        role: ScheduleSisRole::Inner,
        output_rank: matrix.output_rank(),
        input_width: matrix.input_width(),
        ring_dimension: matrix.ring_dimension(),
        modulus_profile,
        bound,
    }
}

fn outer_occurrence(
    location: String,
    matrix: &crate::OuterCommitMatrixParams,
) -> ScheduleSisOccurrence {
    ScheduleSisOccurrence {
        location,
        role: ScheduleSisRole::Outer,
        output_rank: matrix.output_rank(),
        input_width: matrix.input_width(),
        ring_dimension: matrix.ring_dimension(),
        modulus_profile: matrix.sis_modulus_profile(),
        bound: ScheduleSisBound::Linf(matrix.coeff_linf_bound()),
    }
}

fn open_occurrence(
    location: String,
    matrix: &crate::OpenCommitMatrixParams,
) -> ScheduleSisOccurrence {
    ScheduleSisOccurrence {
        location,
        role: ScheduleSisRole::Open,
        output_rank: matrix.output_rank(),
        input_width: matrix.input_width(),
        ring_dimension: matrix.ring_dimension(),
        modulus_profile: matrix.sis_modulus_profile(),
        bound: ScheduleSisBound::Linf(matrix.coeff_linf_bound()),
    }
}

fn compression_occurrence(location: String, map: CompressionMapPlan) -> ScheduleSisOccurrence {
    ScheduleSisOccurrence {
        location,
        role: ScheduleSisRole::Compression,
        output_rank: map.output_rank(),
        input_width: map.input_width(),
        ring_dimension: map.ring_dimension(),
        modulus_profile: map.modulus_profile(),
        bound: ScheduleSisBound::Linf(crate::sis::compression::COMPRESSION_SIS_COEFF_LINF_BOUND),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sis::{sis_l2_table_key_for_collision_sq, SisL2TableDigest};
    use crate::{PhysicalL2NormProofShape, DEFAULT_SIS_SECURITY_POLICY};

    #[test]
    fn inner_occurrence_preserves_the_euclidean_route() {
        let table_key = sis_l2_table_key_for_collision_sq(
            DEFAULT_SIS_SECURITY_POLICY,
            SisL2TableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            1u128 << 50,
        )
        .expect("generated L2 key");
        let matrix = InnerCommitMatrixParams::try_new_l2_with_min_rank(
            table_key,
            21,
            1u128 << 30,
            PhysicalL2NormProofShape::Direct {
                physical_response_len: 21 * 64,
            },
        )
        .expect("audited L2 matrix");

        let occurrence = inner_occurrence("test A".to_string(), &matrix);
        assert_eq!(occurrence.role, ScheduleSisRole::Inner);
        assert_eq!(occurrence.bound, ScheduleSisBound::L2Squared(1u128 << 50));
        assert_eq!(occurrence.output_rank, matrix.output_rank());
        assert_eq!(occurrence.input_width, 21);
        assert_eq!(occurrence.ring_dimension, 64);
    }
}
