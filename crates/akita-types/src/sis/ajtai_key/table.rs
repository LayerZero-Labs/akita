use super::super::coverage::{inner_coeff_linf_bounds, sis_role_cell, GADGET_COEFF_LINF_ANCHORS};
use super::super::generated_sis_table::{
    sis_max_widths as generated_sis_max_widths, SIS_TABLE_DIGEST,
};
#[cfg(test)]
use super::InnerCommitMatrixParams;

/// Digest of the generated scalar table and its coverage certificate.
///
/// The bytes are fixed width and are part of every runtime SIS identity. The
/// value is replaced by the generator when the checked-in table changes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct SisTableDigest(pub [u8; 32]);

impl Default for SisTableDigest {
    fn default() -> Self {
        Self::CURRENT
    }
}

impl SisTableDigest {
    /// Stable wire tag for the digest field.
    pub const TAG: u8 = 1;

    /// Digest committed by the current generated artifact.
    pub const CURRENT: Self = Self(SIS_TABLE_DIGEST);
}

/// Matrix role whose coefficient and ring geometry is being priced.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum SisMatrixRole {
    /// Inner commitment matrix (A).
    Inner,
    /// Outer commitment matrix (B).
    Outer,
    /// Opening commitment matrix (D).
    Open,
}

impl SisMatrixRole {
    /// Stable wire/catalog tag.
    pub const fn tag(self) -> u8 {
        match self {
            Self::Inner => 1,
            Self::Outer => 2,
            Self::Open => 3,
        }
    }

    /// Stable name used in generated provenance.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Inner => "Inner",
            Self::Outer => "Outer",
            Self::Open => "Open",
        }
    }

    /// Parse the stable wire/catalog tag.
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Inner),
            2 => Some(Self::Outer),
            3 => Some(Self::Open),
            _ => None,
        }
    }
}

/// Policy identity used by SIS sizing and generated artifacts.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum SisSecurityPolicyId {
    /// Corrected ADPS16 quantum LGSA estimator at a 128-bit target.
    #[default]
    Quantum128BitADPS16,
}

impl SisSecurityPolicyId {
    /// Stable wire/catalog tag for this policy.
    pub const fn tag(self) -> u8 {
        match self {
            Self::Quantum128BitADPS16 => 1,
        }
    }

    /// Descriptive policy name used in diagnostics and generated metadata.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Quantum128BitADPS16 => "Quantum128BitADPS16",
        }
    }

    /// Parse the stable wire/catalog tag.
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Quantum128BitADPS16),
            _ => None,
        }
    }
}

#[cfg(test)]
mod policy_id_tests {
    use super::SisSecurityPolicyId;

    #[test]
    fn policy_has_the_unversioned_stable_tag() {
        let policy = SisSecurityPolicyId::Quantum128BitADPS16;
        assert_eq!(policy.tag(), 1);
        assert_eq!(SisSecurityPolicyId::from_tag(1), Some(policy));
        assert_eq!(SisSecurityPolicyId::from_tag(2), None);
    }
}

/// Exact SIS modulus profile used to select generated security floors.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum SisModulusProfileId {
    /// Representative q = 2^32 - 99.
    Q32Offset99,
    /// Representative q = 2^64 - 59.
    Q64Offset59,
    /// Representative q = 2^128 - (2^32 - 22537).
    #[default]
    Q128OffsetA7F7,
}

impl SisModulusProfileId {
    /// Exact modulus represented by this profile.
    pub const fn modulus(self) -> u128 {
        match self {
            Self::Q32Offset99 => 4_294_967_197,
            Self::Q64Offset59 => 18_446_744_073_709_551_557,
            Self::Q128OffsetA7F7 => 340_282_366_920_938_463_463_374_607_427_473_266_697,
        }
    }

    /// Bit width of the represented field modulus.
    pub const fn field_bits(self) -> u32 {
        128 - (self.modulus() - 1).leading_zeros()
    }

    /// Stable serialized tag.
    pub const fn tag(self) -> u8 {
        match self {
            Self::Q32Offset99 => 1,
            Self::Q64Offset59 => 2,
            Self::Q128OffsetA7F7 => 3,
        }
    }

    /// Parse the stable serialized tag.
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Q32Offset99),
            2 => Some(Self::Q64Offset59),
            3 => Some(Self::Q128OffsetA7F7),
            _ => None,
        }
    }

    /// Stable descriptor name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Q32Offset99 => "Q32Offset99",
            Self::Q64Offset59 => "Q64Offset59",
            Self::Q128OffsetA7F7 => "Q128OffsetA7F7",
        }
    }

    /// Validate an exact field modulus against this profile.
    pub const fn matches_modulus(self, modulus: u128) -> bool {
        self.modulus() == modulus
    }
}

/// Default policy used by production presets.
pub const DEFAULT_SIS_SECURITY_POLICY: SisSecurityPolicyId =
    SisSecurityPolicyId::Quantum128BitADPS16;

/// Policies with checked-in SIS table support.
pub const SUPPORTED_SIS_SECURITY_POLICIES: &[SisSecurityPolicyId] = &[DEFAULT_SIS_SECURITY_POLICY];

/// Canonical key for a generated SIS floor row.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct SisTableKey {
    /// SIS security policy.
    pub policy: SisSecurityPolicyId,
    /// Digest of the generated scalar table.
    pub table_digest: SisTableDigest,
    /// Exact SIS modulus profile.
    pub modulus_profile: SisModulusProfileId,
    /// Matrix role.
    pub role: SisMatrixRole,
    /// Ring dimension.
    pub ring_dimension: u32,
    /// Exact role coefficient-`L∞` bound.
    pub coeff_linf_bound: u128,
}

/// Resolve a raw coefficient-`L∞` bound to a generated role cell.
///
/// A-role collisions first use the exact protocol-derived target, then round
/// upward to the smallest audited A cell for the selected profile and ring
/// dimension. One-response targets are themselves audited cells. Chunked
/// responses can land between those cells, so this conservative rounding lets
/// them reuse the existing table without adding a chunk-count coverage axis.
/// B and D round up to the smallest exact gadget anchor.
#[must_use]
pub fn ceil_supported_linf_bound(
    policy: SisSecurityPolicyId,
    table_digest: SisTableDigest,
    sis_modulus_profile: SisModulusProfileId,
    role: SisMatrixRole,
    d: u32,
    linf: u128,
) -> Option<u128> {
    if linf == 0 {
        return None;
    }
    let bound = match role {
        SisMatrixRole::Inner => inner_coeff_linf_bounds(sis_modulus_profile, d)
            .into_iter()
            .find(|&candidate| linf <= candidate)?,
        SisMatrixRole::Outer | SisMatrixRole::Open => GADGET_COEFF_LINF_ANCHORS
            .iter()
            .copied()
            .find(|&candidate| linf <= candidate)?,
    };
    sis_role_cell(role, sis_modulus_profile, d, bound)?;
    sis_max_widths(policy, table_digest, sis_modulus_profile, d, bound)?;
    Some(bound)
}

/// Canonical generated-table key for a raw coefficient-`L∞` bound.
///
/// Returns `None` for an unsupported security floor, family/dimension pair, or
/// coefficient bound.
#[must_use]
pub fn sis_table_key_for_linf_bound(
    policy: SisSecurityPolicyId,
    table_digest: SisTableDigest,
    sis_modulus_profile: SisModulusProfileId,
    role: SisMatrixRole,
    d: u32,
    linf: u128,
) -> Option<SisTableKey> {
    let coeff_linf_bound =
        ceil_supported_linf_bound(policy, table_digest, sis_modulus_profile, role, d, linf)?;
    Some(SisTableKey {
        policy,
        table_digest,
        modulus_profile: sis_modulus_profile,
        role,
        ring_dimension: d,
        coeff_linf_bound,
    })
}

/// Certified scalar cutoff kind retained for offline CSV and audit tooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarCutoff {
    /// The accepted value and its immediate successor were certified.
    Exact(u64),
    /// The search reached the configured cap at this value.
    AtLeast(u64),
}

impl ScalarCutoff {
    /// Largest accepted scalar column count represented by this cutoff.
    pub const fn value(self) -> u64 {
        match self {
            Self::Exact(value) | Self::AtLeast(value) => value,
        }
    }
}

fn sis_max_widths(
    policy: SisSecurityPolicyId,
    table_digest: SisTableDigest,
    modulus_profile: SisModulusProfileId,
    d: u32,
    coeff_linf_bound: u128,
) -> Option<&'static [u64]> {
    if policy != DEFAULT_SIS_SECURITY_POLICY {
        return None;
    }
    if table_digest != SisTableDigest::CURRENT {
        return None;
    }
    generated_sis_max_widths(policy, modulus_profile, d, coeff_linf_bound)
}

/// Minimum generated SIS-secure module rank that supports `width` ring columns
/// at an already rounded-up coefficient-`L∞` bucket.
///
/// Returns `None` when no generated SIS-floor row covers the configuration.
pub fn min_secure_rank(key: SisTableKey, width: u64) -> Option<usize> {
    let role_cell = sis_role_cell(
        key.role,
        key.modulus_profile,
        key.ring_dimension,
        key.coeff_linf_bound,
    )?;
    let widths = sis_max_widths(
        key.policy,
        key.table_digest,
        key.modulus_profile,
        key.ring_dimension,
        key.coeff_linf_bound,
    )?;
    let max_module_rank = usize::try_from(role_cell.max_module_rank).ok()?;
    let widths = &widths[..max_module_rank.min(widths.len())];
    widths
        .iter()
        .position(|&max_width| width <= max_width)
        .map(|rank_index| rank_index + 1)
}

#[cfg(test)]
#[path = "table_tests.rs"]
mod tests;
