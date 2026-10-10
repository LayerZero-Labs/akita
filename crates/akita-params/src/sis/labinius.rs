//! LaBinius binary-source SIS identities and exact collision accounting.
//!
//! The opt-in root profile and shape admit binary root geometry against the
//! certified runtime width table and exact challenge/response bounds. This does
//! not select a protocol or admit a complete schedule. Challenge sampling and
//! its certified multiplication bound belong to `akita-challenges`.

mod commitment_lift;
mod fold_response;
mod generated_commitment_prime_width_table;
mod proof_prime;
mod root_profile;
mod root_shape;
mod width_table;

pub use commitment_lift::{labinius_image_digit_bound, LabiniusCommitmentLift};
pub use fold_response::{LabiniusFoldResponse, LABINIUS_BALANCED_LOG_BASIS};
pub use proof_prime::{check_labinius_proof_prime_units, labinius_proof_prime_residue_degree};
pub use root_profile::LabiniusRootProfile;
pub use root_shape::{
    LabiniusRootEncoding, LabiniusRootShape, LabiniusSignedRange, LABINIUS_MIN_DERIVATION_BIAS_BITS,
};
pub use width_table::{
    labinius_min_secure_rank, LabiniusWidthCell, LabiniusWidthCutoff,
    LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE, LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE_DIGEST,
};

use akita_error::AkitaError;

use super::norm_bound::source_comparison_inf_norm;

/// Closed commitment moduli. The proof prime is not part of this identity: it
/// is admitted separately by [`LabiniusRootShape::derive_encoding`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LabiniusCommitmentModulus {
    /// The prime `2^25 + 14561 = 1 + 23328 * 1439`, which splits the conductors
    /// 1944, 3888, 5832 and 7776.
    ///
    /// Its certified cells are [`LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE`].
    Q25Plus14561,
}

impl LabiniusCommitmentModulus {
    /// Exact prime modulus `q`.
    pub const fn modulus(self) -> u32 {
        match self {
            Self::Q25Plus14561 => 33_568_993,
        }
    }

    /// Stable profile label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Q25Plus14561 => "q25+14561",
        }
    }
}

/// Binary commitment-ring degree whose scalar components are priced together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LabiniusRingDegree {
    /// `Phi_243`, scalar degree 162.
    D162,
    /// First packed `Phi_243` tower level.
    D324,
    /// Second packed `Phi_243` tower level.
    D648,
    /// `Phi_729`, scalar degree 486.
    D486,
    /// First packed `Phi_729` tower level.
    D972,
    /// Second packed `Phi_729` tower level.
    D1944,
}

impl LabiniusRingDegree {
    /// Physical polynomial degree used by scalar SIS scalarization.
    pub const fn degree(self) -> u32 {
        match self {
            Self::D162 => 162,
            Self::D324 => 324,
            Self::D648 => 648,
            Self::D486 => 486,
            Self::D972 => 972,
            Self::D1944 => 1_944,
        }
    }

    /// Degree of one binary scalar component.
    pub const fn scalar_degree(self) -> u32 {
        match self {
            Self::D162 | Self::D324 | Self::D648 => 162,
            Self::D486 | Self::D972 | Self::D1944 => 486,
        }
    }

    /// Signed-interleaving component count over the scalar ring.
    pub const fn packing_degree(self) -> u32 {
        self.degree() / self.scalar_degree()
    }
}

/// Source-comparison class identity.
///
/// The digest names the exact matrix view and coefficient addressing. Equal
/// rings with different views must not be compared as occurrences of one
/// source class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabiniusSourceComparisonId {
    /// Commitment modulus `q` of the SIS instance.
    pub commitment_modulus: LabiniusCommitmentModulus,
    /// Actual binary commitment ring.
    pub ring_degree: LabiniusRingDegree,
    /// Versioned digest of the matrix view/addressing semantics.
    pub matrix_view_digest: [u8; 32],
}

/// Certified mixed-envelope inputs for one source occurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceOccurrenceBound {
    numerator_bound: u128,
    slack_operator_bound: u128,
}

impl SourceOccurrenceBound {
    /// Binary weak extraction from an accepted response interval.
    ///
    /// `challenge_multiplication_bound` is the independently certified
    /// `Gamma_inf` from the challenge profile. Extraction doubles the slack
    /// difference, hence `K = 2 * Gamma_inf`; `B` is the full accepted response
    /// diameter, including every chunk that enters the A relation.
    pub fn binary_extracted(
        challenge_multiplication_bound: u128,
        accepted_response_diameter: u128,
    ) -> Option<Self> {
        Some(Self {
            numerator_bound: accepted_response_diameter,
            slack_operator_bound: challenge_multiplication_bound.checked_mul(2)?,
        })
    }

    /// Canonical known source reference. Its slack is exactly one, so no extra
    /// factor two is introduced.
    pub const fn canonical_reference(source_coefficient_bound: u128) -> Self {
        Self {
            numerator_bound: source_coefficient_bound,
            slack_operator_bound: 1,
        }
    }

    /// Certified numerator coefficient bound `B_u`.
    pub const fn numerator_bound(self) -> u128 {
        self.numerator_bound
    }

    /// Certified induced slack multiplication bound `K_u`.
    pub const fn slack_operator_bound(self) -> u128 {
        self.slack_operator_bound
    }
}

/// Compare two occurrences only when their source semantics and matrix view
/// are identical, and enforce the integer no-wrap condition `eta < q`.
pub fn checked_source_comparison_bound(
    left_id: LabiniusSourceComparisonId,
    left: SourceOccurrenceBound,
    right_id: LabiniusSourceComparisonId,
    right: SourceOccurrenceBound,
) -> Result<u128, AkitaError> {
    if left_id != right_id {
        return Err(AkitaError::InvalidSetup(
            "binary source comparison identities do not match".into(),
        ));
    }
    let eta = source_comparison_inf_norm(
        left.numerator_bound,
        left.slack_operator_bound,
        right.numerator_bound,
        right.slack_operator_bound,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("binary source collision bound overflow".into()))?;
    if eta >= u128::from(left_id.commitment_modulus.modulus()) {
        return Err(AkitaError::InvalidSetup(format!(
            "binary source collision bound {eta} does not satisfy eta < q"
        )));
    }
    Ok(eta)
}

/// Maximum no-wrap collision bound over every pair in one source class.
///
/// The diagonal pairs are included: a class containing one occurrence still
/// needs the collision bound obtained by comparing that occurrence with
/// itself. Callers must place only occurrences with the supplied source and
/// matrix-view identity in this slice.
pub fn checked_source_comparison_class_bound(
    id: LabiniusSourceComparisonId,
    occurrences: &[SourceOccurrenceBound],
) -> Result<u128, AkitaError> {
    if occurrences.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "binary source comparison class must be nonempty".into(),
        ));
    }
    let mut maximum = 0;
    for (left_index, &left) in occurrences.iter().enumerate() {
        for &right in &occurrences[left_index..] {
            maximum = maximum.max(checked_source_comparison_bound(id, left, id, right)?);
        }
    }
    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> LabiniusSourceComparisonId {
        LabiniusSourceComparisonId {
            commitment_modulus: LabiniusCommitmentModulus::Q25Plus14561,
            ring_degree: LabiniusRingDegree::D648,
            matrix_view_digest: [7; 32],
        }
    }

    #[test]
    fn packed_ring_degrees_preserve_scalar_component_geometry() {
        assert_eq!(LabiniusRingDegree::D162.packing_degree(), 1);
        assert_eq!(LabiniusRingDegree::D324.packing_degree(), 2);
        assert_eq!(LabiniusRingDegree::D648.packing_degree(), 4);
        assert_eq!(LabiniusRingDegree::D486.packing_degree(), 1);
        assert_eq!(LabiniusRingDegree::D972.packing_degree(), 2);
        assert_eq!(LabiniusRingDegree::D1944.packing_degree(), 4);
    }

    #[test]
    fn two_extractions_match_four_gamma_times_the_interval_diameter() {
        let extracted =
            SourceOccurrenceBound::binary_extracted(92, 4_095).expect("bounded extraction");
        let eta = checked_source_comparison_bound(identity(), extracted, identity(), extracted)
            .expect("no-wrap collision");
        assert_eq!(eta, 4 * 92 * 4_095);
    }

    #[test]
    fn canonical_reference_has_unit_slack_without_an_extra_factor_two() {
        let extracted = SourceOccurrenceBound::binary_extracted(94, 1_000).unwrap();
        let canonical = SourceOccurrenceBound::canonical_reference(1);
        let eta =
            checked_source_comparison_bound(identity(), extracted, identity(), canonical).unwrap();
        assert_eq!(eta, 1_000 + 188);
    }

    #[test]
    fn source_class_maximizes_over_mixed_pairs() {
        let class = [
            SourceOccurrenceBound {
                numerator_bound: 10,
                slack_operator_bound: 1,
            },
            SourceOccurrenceBound {
                numerator_bound: 1,
                slack_operator_bound: 100,
            },
        ];
        assert_eq!(
            checked_source_comparison_class_bound(identity(), &class,),
            Ok(1_001)
        );
        assert!(checked_source_comparison_class_bound(identity(), &[],).is_err());
    }

    #[test]
    fn mismatched_views_and_the_exact_no_wrap_boundary_are_rejected() {
        let occurrence = SourceOccurrenceBound::binary_extracted(2, 3).unwrap();
        let id = identity();
        let mut other = id;
        other.matrix_view_digest[0] ^= 1;
        assert!(checked_source_comparison_bound(id, occurrence, other, occurrence,).is_err());

        let zero = SourceOccurrenceBound::canonical_reference(0);
        let modulus = u128::from(id.commitment_modulus.modulus());
        assert_eq!(
            checked_source_comparison_bound(
                id,
                SourceOccurrenceBound::canonical_reference(modulus - 1),
                id,
                zero,
            ),
            Ok(modulus - 1)
        );
        assert!(checked_source_comparison_bound(
            id,
            SourceOccurrenceBound::canonical_reference(modulus),
            id,
            zero,
        )
        .is_err());
        assert!(checked_source_comparison_bound(
            id,
            SourceOccurrenceBound {
                numerator_bound: u128::MAX,
                slack_operator_bound: 2,
            },
            id,
            SourceOccurrenceBound::canonical_reference(1),
        )
        .is_err());
    }
}
