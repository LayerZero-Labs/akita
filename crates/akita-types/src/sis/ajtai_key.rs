//! Ajtai-commitment key sizing: exact SIS profiles, role-specific matrix
//! parameter types, secure-rank lookup, and coefficient-`L∞` bucket rounding.
//!
//! This is the single home for "given a width and a rounded-up coefficient
//! bound at a security floor, what is the minimum SIS-secure module rank, and what audited
//! commit-matrix parameters does it yield". The generated SIS-floor tables it consults
//! live in the private sibling module `super::generated_sis_table`.

use akita_field::AkitaError;

use super::generated_l2_sis_table::{
    sis_max_widths as generated_l2_sis_max_widths, TABLE_DIGEST as L2_TABLE_DIGEST,
};
use super::generated_sis_table::sis_max_widths as generated_sis_max_widths;
use crate::descriptor_bytes::{push_u128, push_usize, sis_modulus_profile_tag};

/// Digest of the generated scalar table and its coverage certificate.
///
/// The bytes are fixed width and are part of every runtime SIS identity. The
/// value is replaced by the generator when the checked-in table changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    pub const CURRENT: Self = Self([
        0xb4, 0x65, 0x7f, 0x62, 0x90, 0x61, 0x5c, 0xf3, 0x58, 0x55, 0x77, 0xd7, 0xad, 0x51, 0x9f,
        0x9d, 0xc5, 0x5d, 0x4b, 0x8d, 0xcc, 0x63, 0x16, 0x11, 0x1b, 0x26, 0x70, 0x42, 0xac, 0x3b,
        0x92, 0x94,
    ]);

    /// Additive q128 Inner/512 coverage generated directly for `D = 512`.
    ///
    /// Existing schedules intentionally remain on [`Self::CURRENT`].
    pub const Q128_INNER_D512: Self = Self([
        0xc2, 0x02, 0x7a, 0x80, 0xd8, 0x4b, 0x01, 0xdb, 0xbf, 0xfa, 0xe5, 0x71, 0xcb, 0x9b, 0xf0,
        0xe9, 0x68, 0x6d, 0xb6, 0xe7, 0x62, 0xc5, 0xa4, 0x20, 0x2d, 0x5e, 0x53, 0xa3, 0x06, 0xe6,
        0xca, 0xce,
    ]);
}

/// Digest of the separate generated Euclidean SIS table and its boundary
/// evidence.
///
/// This identity is distinct from [`SisTableDigest`]. An L2-selected schedule
/// binds both its squared collision bucket and this digest without changing the
/// coefficient-L∞ fallback table identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SisL2TableDigest(pub [u8; 32]);

impl Default for SisL2TableDigest {
    fn default() -> Self {
        Self::CURRENT
    }
}

impl SisL2TableDigest {
    /// Stable wire tag for the L2 digest field.
    pub const TAG: u8 = 1;

    /// SHA-256 digest of the current generated L2 table's `audit.csv`.
    pub const CURRENT: Self = Self(L2_TABLE_DIGEST);
}

/// Matrix role whose coefficient and ring geometry is being priced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SisSecurityPolicyId {
    /// ADPS16 quantum LGSA estimator at a 128-bit target.
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

/// Exact SIS modulus profile used to select generated security floors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

    /// Infinity-norm expansion of the current trace-subfield embedding.
    ///
    /// The 128-bit profile is the base-field path. The 32- and 64-bit profiles
    /// use the paired-lane trace embedding and therefore carry the certified
    /// factor-of-two expansion.
    pub const fn ring_subfield_embedding_norm_bound(self) -> u32 {
        match self {
            Self::Q128OffsetA7F7 => 1,
            Self::Q32Offset99 | Self::Q64Offset59 => 2,
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

/// Coefficient-`L∞` collision buckets for norm-bound sizing.
///
/// Keep in lockstep with `COEFF_LINF_BUCKETS` in
/// `crates/akita-sis-estimator/src/width_table.rs`.
pub const COEFF_LINF_BUCKETS: &[u128] = &[
    2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131_071,
    262_143, 524_287, 1_048_575, 2_097_151, 4_194_303, 8_388_607, 16_777_215, 33_554_431,
    67_108_863,
];

/// Canonical key for a generated SIS floor row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    /// Rounded coefficient-`L∞` bound.
    pub coeff_linf_bound: u128,
}

/// Canonical key for one generated Euclidean SIS floor row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SisL2TableKey {
    /// SIS security policy.
    pub policy: SisSecurityPolicyId,
    /// Digest of the separate generated Euclidean table.
    pub table_digest: SisL2TableDigest,
    /// Exact SIS modulus profile.
    pub modulus_profile: SisModulusProfileId,
    /// Ring dimension.
    pub ring_dimension: u32,
    /// Rounded squared L2 norm of the complete scalar collision vector.
    pub collision_l2_sq: u128,
}

/// One reachable role coverage cell used by generation and runtime checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SisRoleCell {
    /// Matrix role.
    pub role: SisMatrixRole,
    /// Exact modulus profile.
    pub modulus_profile: SisModulusProfileId,
    /// Ring dimension.
    pub ring_dimension: u32,
    /// Exact role coefficient bound cell.
    pub coeff_linf_bound: u128,
    /// Maximum supported module rank.
    pub max_module_rank: u32,
    /// Largest required ring width from the planner domain.
    pub required_max_width: u64,
}

/// Exact gadget anchors used by B and D.
pub const GADGET_COEFF_LINF_ANCHORS: &[u128] = &[3, 7, 15, 31, 63, 127, 255];

/// Ring dimensions supported by A for every SIS modulus profile.
///
/// Q128 has the additional profile-specific `D = 512` cell enforced by
/// [`sis_role_cell`].
pub const A_ROLE_RING_DIMS: &[u32] = &[64, 128, 256];

/// Admitted B/D commitment-matrix dimensions.
pub const BD_ROLE_RING_DIMS: &[u32] = &[64, 128, 256];

/// Production matrix roles with checked-in coverage.
pub const SIS_MATRIX_ROLES: &[SisMatrixRole] = &[
    SisMatrixRole::Inner,
    SisMatrixRole::Outer,
    SisMatrixRole::Open,
];

/// Return whether the exact role cell is part of the canonical coverage.
///
/// The function is deliberately role aware. It does not form a product of
/// independent dimension and bound lists for one shared table.
#[must_use]
pub fn sis_role_cell(
    role: SisMatrixRole,
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
    coeff_linf_bound: u128,
) -> Option<SisRoleCell> {
    let (dimension_supported, bounds) = match role {
        SisMatrixRole::Inner => (
            A_ROLE_RING_DIMS.contains(&ring_dimension)
                || (modulus_profile == SisModulusProfileId::Q128OffsetA7F7
                    && ring_dimension == 512),
            COEFF_LINF_BUCKETS,
        ),
        SisMatrixRole::Outer | SisMatrixRole::Open => (
            BD_ROLE_RING_DIMS.contains(&ring_dimension),
            GADGET_COEFF_LINF_ANCHORS,
        ),
    };
    if !dimension_supported || !bounds.contains(&coeff_linf_bound) {
        return None;
    }
    Some(SisRoleCell {
        role,
        modulus_profile,
        ring_dimension,
        coeff_linf_bound,
        max_module_rank: 20,
        required_max_width: 6_400_000_000_000,
    })
}

/// Smallest coefficient-`L∞` bucket with `B >= linf`.
#[must_use]
pub fn ceil_coeff_linf_bucket(linf: u128) -> Option<u128> {
    if linf == 0 {
        return None;
    }
    COEFF_LINF_BUCKETS
        .iter()
        .copied()
        .find(|&bucket| linf <= bucket)
}

/// Round a raw coefficient-`L∞` bound up to a generated table bucket.
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
    let bucket = match role {
        SisMatrixRole::Inner => ceil_coeff_linf_bucket(linf)?,
        SisMatrixRole::Outer | SisMatrixRole::Open => GADGET_COEFF_LINF_ANCHORS
            .iter()
            .copied()
            .find(|&candidate| linf <= candidate)?,
    };
    sis_role_cell(role, sis_modulus_profile, d, bucket)?;
    sis_max_widths(policy, table_digest, sis_modulus_profile, d, bucket)?;
    Some(bucket)
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

/// Certified scalar cutoff kind retained for offline CSV / audit tooling.
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
    if table_digest == SisTableDigest::CURRENT && d == 512 {
        return None;
    }
    if table_digest != SisTableDigest::CURRENT && table_digest != SisTableDigest::Q128_INNER_D512 {
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
    for (i, &max_width) in widths.iter().take(max_module_rank).enumerate() {
        if width <= max_width {
            return Some(i + 1);
        }
    }
    None
}

/// Round a complete scalar collision-vector squared L2 norm to the generated
/// ADPS16 quantum table ladder.
#[must_use]
pub fn ceil_supported_l2_collision_sq(collision_l2_sq: u128) -> Option<u128> {
    if collision_l2_sq == 0 {
        return None;
    }
    let bucket = collision_l2_sq.checked_next_power_of_two()?.max(2);
    (bucket <= (1u128 << 84)).then_some(bucket)
}

/// Canonical generated-table key for a raw complete squared L2 collision norm.
///
/// Returns `None` for an unsupported policy, digest, family, dimension, or
/// collision bucket.
#[must_use]
pub fn sis_l2_table_key_for_collision_sq(
    policy: SisSecurityPolicyId,
    table_digest: SisL2TableDigest,
    modulus_profile: SisModulusProfileId,
    ring_dimension: u32,
    collision_l2_sq: u128,
) -> Option<SisL2TableKey> {
    if policy != DEFAULT_SIS_SECURITY_POLICY || table_digest != SisL2TableDigest::CURRENT {
        return None;
    }
    let collision_l2_sq = ceil_supported_l2_collision_sq(collision_l2_sq)?;
    generated_l2_sis_max_widths(modulus_profile, ring_dimension, collision_l2_sq)?;
    Some(SisL2TableKey {
        policy,
        table_digest,
        modulus_profile,
        ring_dimension,
        collision_l2_sq,
    })
}

/// Minimum module rank under the generated 128-bit quantum ADPS16 Euclidean
/// SIS model.
///
/// `key.collision_l2_sq` is the squared norm of the complete scalar collision
/// vector, not a per-ring-row bound.
#[must_use]
pub fn min_secure_l2_rank(key: SisL2TableKey, width: u64) -> Option<usize> {
    if width == 0
        || key.policy != DEFAULT_SIS_SECURITY_POLICY
        || key.table_digest != SisL2TableDigest::CURRENT
    {
        return None;
    }
    let widths =
        generated_l2_sis_max_widths(key.modulus_profile, key.ring_dimension, key.collision_l2_sq)?;
    widths
        .iter()
        .position(|&max_width| width <= max_width)
        .map(|index| index + 1)
}

#[derive(Debug, Clone, Copy)]
struct AuditedCommitMatrixFields {
    output_rank: usize,
    input_width: usize,
    sis_table_key: SisTableKey,
}

/// Schedule-owned shape of the integer norm proof for one L2-selected A
/// matrix.
///
/// The prover does not serialize block or limb-pair identifiers. The checked
/// shape derives consecutive blocks and the complete upper-triangular pair
/// sequence from these counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PhysicalL2NormProofShape {
    /// One direct square-sum claim over every physical response coefficient.
    Direct { physical_response_len: usize },
    /// Blockwise balanced-limb Gram claims used when the direct integer sum
    /// could wrap the base field.
    LimbGram {
        physical_response_len: usize,
        block_len: usize,
        limb_count: usize,
    },
}

impl PhysicalL2NormProofShape {
    /// Derive the canonical no-wrap proof shape for one physical response
    /// domain and its existing balanced digit decomposition.
    pub fn derive(
        modulus_profile: SisModulusProfileId,
        physical_response_len: usize,
        fold_basis: usize,
        fold_digit_count: usize,
    ) -> Result<Self, AkitaError> {
        if physical_response_len == 0
            || fold_digit_count == 0
            || fold_basis < 2
            || !fold_basis.is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(
                "L2 norm shape requires a nonempty response and balanced power-of-two digits"
                    .into(),
            ));
        }
        let direct = Self::Direct {
            physical_response_len,
        };
        if direct
            .validate_integer_soundness(modulus_profile, fold_basis, fold_digit_count)
            .is_ok()
        {
            return Ok(direct);
        }
        let modulus = modulus_profile.modulus();
        let digit_abs = (fold_basis / 2) as u128;
        if modulus > i128::MAX as u128 {
            return Err(AkitaError::InvalidSetup(
                "L2 norm response is too wide for direct proof and its modulus has no centered-limb path"
                    .into(),
            ));
        }
        let digit_square = digit_abs
            .checked_mul(digit_abs)
            .ok_or_else(|| AkitaError::InvalidSetup("L2 limb digit square overflow".into()))?;
        let max_block = modulus
            .checked_div(2)
            .and_then(|half| half.checked_sub(1))
            .and_then(|limit| limit.checked_div(digit_square))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("L2 limb alphabet cannot fit a centered block".into())
            })?;
        let block_len = usize::try_from(max_block)
            .unwrap_or(usize::MAX)
            .min(physical_response_len);
        let shape = Self::LimbGram {
            physical_response_len,
            block_len,
            limb_count: fold_digit_count,
        };
        shape.validate_integer_soundness(modulus_profile, fold_basis, fold_digit_count)?;
        Ok(shape)
    }

    /// Validate that a scheduled shape covers the exact digit response and
    /// rules out field wraparound using only public bounds.
    pub fn validate_integer_soundness(
        self,
        modulus_profile: SisModulusProfileId,
        fold_basis: usize,
        fold_digit_count: usize,
    ) -> Result<(), AkitaError> {
        self.validate()?;
        if fold_digit_count == 0 || fold_basis < 2 || !fold_basis.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "L2 norm shape has an invalid balanced digit decomposition".into(),
            ));
        }
        let modulus = modulus_profile.modulus();
        let digit_abs = (fold_basis / 2) as u128;
        match self {
            Self::Direct {
                physical_response_len,
            } => {
                let mut max_response = 0u128;
                let mut power = 1u128;
                for _ in 0..fold_digit_count {
                    max_response = max_response
                        .checked_add(digit_abs.checked_mul(power).ok_or_else(|| {
                            AkitaError::InvalidSetup("direct norm response bound overflow".into())
                        })?)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("direct norm response bound overflow".into())
                        })?;
                    power = power.checked_mul(fold_basis as u128).ok_or_else(|| {
                        AkitaError::InvalidSetup("direct norm basis power overflow".into())
                    })?;
                }
                let worst = (physical_response_len as u128)
                    .checked_mul(max_response.checked_mul(max_response).ok_or_else(|| {
                        AkitaError::InvalidSetup("direct norm worst-case overflow".into())
                    })?)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("direct norm worst-case overflow".into())
                    })?;
                if worst >= modulus {
                    return Err(AkitaError::InvalidSetup(
                        "direct norm shape does not rule out field wraparound".into(),
                    ));
                }
            }
            Self::LimbGram {
                block_len,
                limb_count,
                ..
            } => {
                if limb_count != fold_digit_count || modulus > i128::MAX as u128 {
                    return Err(AkitaError::InvalidSetup(
                        "L2 limb-Gram shape disagrees with its field or digit count".into(),
                    ));
                }
                let claim_abs_bound =
                    (block_len as u128)
                        .checked_mul(digit_abs.checked_mul(digit_abs).ok_or_else(|| {
                            AkitaError::InvalidSetup("L2 limb bound overflow".into())
                        })?)
                        .ok_or_else(|| AkitaError::InvalidSetup("L2 limb bound overflow".into()))?;
                if claim_abs_bound >= modulus / 2 {
                    return Err(AkitaError::InvalidSetup(
                        "L2 limb block does not rule out centered-lift ambiguity".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Validate nonzero, bounded arithmetic for the schedule-derived shape.
    pub fn validate(self) -> Result<(), AkitaError> {
        let physical_response_len = self.physical_response_len();
        if physical_response_len == 0 {
            return Err(AkitaError::InvalidSetup(
                "L2 norm proof requires a nonempty physical response".into(),
            ));
        }
        if let Self::LimbGram {
            block_len,
            limb_count,
            ..
        } = self
        {
            if block_len == 0 || limb_count == 0 || block_len > physical_response_len {
                return Err(AkitaError::InvalidSetup(
                    "L2 limb-Gram shape has invalid block or limb count".into(),
                ));
            }
            self.subclaim_count().ok_or_else(|| {
                AkitaError::InvalidSetup("L2 limb-Gram subclaim count overflow".into())
            })?;
        }
        Ok(())
    }

    /// Number of physical centered coefficients certified by the proof.
    #[must_use]
    pub const fn physical_response_len(self) -> usize {
        match self {
            Self::Direct {
                physical_response_len,
            }
            | Self::LimbGram {
                physical_response_len,
                ..
            } => physical_response_len,
        }
    }

    /// Number of integer claims carried by the norm proof.
    #[must_use]
    pub fn subclaim_count(self) -> Option<usize> {
        match self {
            Self::Direct { .. } => Some(0),
            Self::LimbGram {
                physical_response_len,
                block_len,
                limb_count,
            } => {
                let blocks = physical_response_len.div_ceil(block_len);
                let pairs = limb_count.checked_mul(limb_count.checked_add(1)?)? / 2;
                blocks.checked_mul(pairs)
            }
        }
    }

    /// Number of final Stage 1 evaluations bound into Stage 2.
    #[must_use]
    pub const fn virtual_evaluation_count(self) -> usize {
        match self {
            Self::Direct { .. } => 1,
            Self::LimbGram { limb_count, .. } => limb_count,
        }
    }

    fn append_descriptor_bytes(self, bytes: &mut Vec<u8>) {
        match self {
            Self::Direct {
                physical_response_len,
            } => {
                bytes.push(1);
                push_usize(bytes, physical_response_len);
            }
            Self::LimbGram {
                physical_response_len,
                block_len,
                limb_count,
            } => {
                bytes.push(2);
                push_usize(bytes, physical_response_len);
                push_usize(bytes, block_len);
                push_usize(bytes, limb_count);
            }
        }
    }
}

/// The single selected security route for an A commitment matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum InnerCommitSecurityRoute {
    /// Existing coefficient-L-infinity sizing and digit-range proof.
    Linf(SisTableKey),
    /// Complete physical L2 sizing with the scheduled integer norm proof.
    L2 {
        table_key: SisL2TableKey,
        response_l2_sq_cap: u128,
        norm_proof_shape: PhysicalL2NormProofShape,
    },
}

impl InnerCommitSecurityRoute {
    /// Exact modulus profile selected by this route.
    #[must_use]
    pub const fn modulus_profile(self) -> SisModulusProfileId {
        match self {
            Self::Linf(key) => key.modulus_profile,
            Self::L2 { table_key, .. } => table_key.modulus_profile,
        }
    }

    /// Exact policy selected by this route.
    #[must_use]
    pub const fn policy(self) -> SisSecurityPolicyId {
        match self {
            Self::Linf(key) => key.policy,
            Self::L2 { table_key, .. } => table_key.policy,
        }
    }

    /// A-role ring dimension selected by this route.
    #[must_use]
    pub const fn ring_dimension(self) -> u32 {
        match self {
            Self::Linf(key) => key.ring_dimension,
            Self::L2 { table_key, .. } => table_key.ring_dimension,
        }
    }
}

/// Parameters for the inner commitment matrix (A).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InnerCommitMatrixParams {
    pub(crate) output_rank: usize,
    pub(crate) input_width: usize,
    pub(crate) security_route: InnerCommitSecurityRoute,
}

impl InnerCommitMatrixParams {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        policy: SisSecurityPolicyId,
        table_digest: SisTableDigest,
        sis_modulus_profile: SisModulusProfileId,
        output_rank: usize,
        input_width: usize,
        coeff_linf_bound: u128,
        ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        let fields = audit_commit_matrix_fields(
            SisMatrixRole::Inner,
            policy,
            table_digest,
            sis_modulus_profile,
            output_rank,
            input_width,
            coeff_linf_bound,
            ring_dimension,
        )?;
        Ok(Self {
            output_rank: fields.output_rank,
            input_width: fields.input_width,
            security_route: InnerCommitSecurityRoute::Linf(fields.sis_table_key),
        })
    }

    pub fn try_new_with_min_rank(key: SisTableKey, input_width: usize) -> Result<Self, AkitaError> {
        let fields = min_rank_commit_matrix_fields(SisMatrixRole::Inner, key, input_width)?;
        Ok(Self {
            output_rank: fields.output_rank,
            input_width: fields.input_width,
            security_route: InnerCommitSecurityRoute::Linf(fields.sis_table_key),
        })
    }

    /// Construct the minimum-rank A matrix for one checked Euclidean route.
    pub fn try_new_l2_with_min_rank(
        table_key: SisL2TableKey,
        input_width: usize,
        response_l2_sq_cap: u128,
        norm_proof_shape: PhysicalL2NormProofShape,
    ) -> Result<Self, AkitaError> {
        if input_width == 0 || response_l2_sq_cap == 0 {
            return Err(AkitaError::InvalidSetup(
                "L2 A matrix requires nonzero width and response cap".into(),
            ));
        }
        norm_proof_shape.validate()?;
        let width = u64::try_from(input_width)
            .map_err(|_| AkitaError::InvalidSetup("A matrix input width exceeds u64".into()))?;
        let output_rank = min_secure_l2_rank(table_key, width).ok_or_else(|| {
            AkitaError::InvalidSetup("A matrix has no audited L2 SIS rank".into())
        })?;
        Ok(Self {
            output_rank,
            input_width,
            security_route: InnerCommitSecurityRoute::L2 {
                table_key,
                response_l2_sq_cap,
                norm_proof_shape,
            },
        })
    }

    /// Rebuild this matrix for a layout-derived input width while preserving
    /// the selected security route and explicit output rank.
    pub fn try_with_input_width(self, input_width: usize) -> Result<Self, AkitaError> {
        match self.security_route {
            InnerCommitSecurityRoute::Linf(key) => {
                if key.coeff_linf_bound == 0 {
                    return Ok(Self::new_unchecked(
                        key.policy,
                        key.table_digest,
                        key.modulus_profile,
                        self.output_rank,
                        input_width,
                        key.coeff_linf_bound,
                        key.ring_dimension as usize,
                    ));
                }
                Self::try_new(
                    key.policy,
                    key.table_digest,
                    key.modulus_profile,
                    self.output_rank,
                    input_width,
                    key.coeff_linf_bound,
                    key.ring_dimension as usize,
                )
            }
            InnerCommitSecurityRoute::L2 {
                table_key,
                response_l2_sq_cap,
                norm_proof_shape,
            } => {
                let width = u64::try_from(input_width).map_err(|_| {
                    AkitaError::InvalidSetup("A matrix input width exceeds u64".into())
                })?;
                let floor = min_secure_l2_rank(table_key, width).ok_or_else(|| {
                    AkitaError::InvalidSetup("A matrix has no audited L2 SIS rank".into())
                })?;
                if self.output_rank < floor {
                    return Err(AkitaError::InvalidSetup(format!(
                        "A matrix output_rank {} is below L2 SIS floor {floor}",
                        self.output_rank
                    )));
                }
                let out = Self {
                    output_rank: self.output_rank,
                    input_width,
                    security_route: InnerCommitSecurityRoute::L2 {
                        table_key,
                        response_l2_sq_cap,
                        norm_proof_shape,
                    },
                };
                out.validate()?;
                Ok(out)
            }
        }
    }

    /// Re-audit the selected route against its generated table and rank floor.
    pub fn validate(&self) -> Result<(), AkitaError> {
        match self.security_route {
            InnerCommitSecurityRoute::Linf(key) => {
                let fields = audit_commit_matrix_fields(
                    SisMatrixRole::Inner,
                    key.policy,
                    key.table_digest,
                    key.modulus_profile,
                    self.output_rank,
                    self.input_width,
                    key.coeff_linf_bound,
                    key.ring_dimension as usize,
                )?;
                if fields.sis_table_key != key {
                    return Err(AkitaError::InvalidSetup(
                        "A matrix L-infinity table key is not canonical".into(),
                    ));
                }
            }
            InnerCommitSecurityRoute::L2 {
                table_key,
                response_l2_sq_cap,
                norm_proof_shape,
            } => {
                norm_proof_shape.validate()?;
                let width = u64::try_from(self.input_width).map_err(|_| {
                    AkitaError::InvalidSetup("A matrix input width exceeds u64".into())
                })?;
                if response_l2_sq_cap == 0
                    || min_secure_l2_rank(table_key, width)
                        .is_none_or(|rank| rank > self.output_rank)
                {
                    return Err(AkitaError::InvalidSetup(
                        "A matrix L2 route is below its audited SIS floor".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn new_unchecked(
        policy: SisSecurityPolicyId,
        table_digest: SisTableDigest,
        sis_modulus_profile: SisModulusProfileId,
        output_rank: usize,
        input_width: usize,
        coeff_linf_bound: u128,
        ring_dimension: usize,
    ) -> Self {
        Self {
            output_rank,
            input_width,
            security_route: InnerCommitSecurityRoute::Linf(SisTableKey {
                policy,
                table_digest,
                modulus_profile: sis_modulus_profile,
                role: SisMatrixRole::Inner,
                ring_dimension: ring_dimension as u32,
                coeff_linf_bound,
            }),
        }
    }

    #[must_use]
    pub const fn output_rank(&self) -> usize {
        self.output_rank
    }

    #[must_use]
    pub const fn input_width(&self) -> usize {
        self.input_width
    }

    #[must_use]
    pub const fn security_route(&self) -> InnerCommitSecurityRoute {
        self.security_route
    }

    #[must_use]
    pub const fn security_policy(&self) -> SisSecurityPolicyId {
        self.security_route.policy()
    }

    #[must_use]
    pub const fn sis_modulus_profile(&self) -> SisModulusProfileId {
        self.security_route.modulus_profile()
    }

    #[must_use]
    pub const fn ring_dimension(&self) -> usize {
        self.security_route.ring_dimension() as usize
    }

    /// Coefficient table key for an L-infinity-selected matrix.
    #[must_use]
    pub const fn sis_table_key(&self) -> Option<SisTableKey> {
        match self.security_route {
            InnerCommitSecurityRoute::Linf(key) => Some(key),
            InnerCommitSecurityRoute::L2 { .. } => None,
        }
    }

    /// Rounded coefficient bound for an L-infinity-selected matrix.
    #[must_use]
    pub const fn coeff_linf_bound(&self) -> Option<u128> {
        match self.security_route {
            InnerCommitSecurityRoute::Linf(key) => Some(key.coeff_linf_bound),
            InnerCommitSecurityRoute::L2 { .. } => None,
        }
    }

    #[must_use]
    pub fn max_secure_collision_linf(&self) -> Option<u128> {
        let key = self.sis_table_key()?;
        COEFF_LINF_BUCKETS
            .iter()
            .copied()
            .take_while(|&bound| {
                min_secure_rank(
                    SisTableKey {
                        coeff_linf_bound: bound,
                        ..key
                    },
                    self.input_width as u64,
                )
                .is_some_and(|rank| rank <= self.output_rank)
            })
            .last()
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(sis_modulus_profile_tag(self.sis_modulus_profile()));
        bytes.push(self.security_policy().tag());
        bytes.push(SisMatrixRole::Inner.tag());
        match self.security_route {
            InnerCommitSecurityRoute::Linf(key) => {
                // Preserve the established Linf descriptor byte sequence.
                bytes.extend_from_slice(&key.table_digest.0);
                bytes.extend_from_slice(&key.ring_dimension.to_le_bytes());
                push_usize(bytes, self.output_rank);
                push_usize(bytes, self.input_width);
                push_u128(bytes, key.coeff_linf_bound);
            }
            InnerCommitSecurityRoute::L2 {
                table_key,
                response_l2_sq_cap,
                norm_proof_shape,
            } => {
                bytes.extend_from_slice(b"akita-l2-route-v1");
                bytes.extend_from_slice(&table_key.table_digest.0);
                bytes.extend_from_slice(&table_key.ring_dimension.to_le_bytes());
                push_usize(bytes, self.output_rank);
                push_usize(bytes, self.input_width);
                push_u128(bytes, table_key.collision_l2_sq);
                push_u128(bytes, response_l2_sq_cap);
                norm_proof_shape.append_descriptor_bytes(bytes);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn audit_commit_matrix_fields(
    expected_role: SisMatrixRole,
    policy: SisSecurityPolicyId,
    table_digest: SisTableDigest,
    sis_modulus_profile: SisModulusProfileId,
    output_rank: usize,
    input_width: usize,
    coeff_linf_bound: u128,
    ring_dimension: usize,
) -> Result<AuditedCommitMatrixFields, AkitaError> {
    if output_rank == 0 || input_width == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "{} matrix requires nonzero output_rank and input_width",
            expected_role.name()
        )));
    }
    let ring_dimension = u32::try_from(ring_dimension).map_err(|_| {
        AkitaError::InvalidSetup(format!(
            "{} matrix ring dimension exceeds u32",
            expected_role.name()
        ))
    })?;
    let input_width_u64 = u64::try_from(input_width).map_err(|_| {
        AkitaError::InvalidSetup(format!(
            "{} matrix input width exceeds u64",
            expected_role.name()
        ))
    })?;
    let key = sis_table_key_for_linf_bound(
        policy,
        table_digest,
        sis_modulus_profile,
        expected_role,
        ring_dimension,
        coeff_linf_bound,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "{} matrix has no audited SIS table key for policy={} profile={sis_modulus_profile:?} d={ring_dimension} coeff_linf_bound={coeff_linf_bound}",
            expected_role.name(),
            policy.name()
        ))
    })?;
    let floor = min_secure_rank(key, input_width_u64).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "{} matrix has no audited SIS rank for input_width={input_width}",
            expected_role.name()
        ))
    })?;
    if output_rank < floor {
        return Err(AkitaError::InvalidSetup(format!(
            "{} matrix output_rank {output_rank} is below SIS floor {floor}",
            expected_role.name()
        )));
    }
    Ok(AuditedCommitMatrixFields {
        output_rank,
        input_width,
        sis_table_key: key,
    })
}

fn min_rank_commit_matrix_fields(
    expected_role: SisMatrixRole,
    key: SisTableKey,
    input_width: usize,
) -> Result<AuditedCommitMatrixFields, AkitaError> {
    if key.role != expected_role || input_width == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "{} matrix has mismatched role or zero input_width",
            expected_role.name()
        )));
    }
    let input_width_u64 = u64::try_from(input_width).map_err(|_| {
        AkitaError::InvalidSetup(format!(
            "{} matrix input width exceeds u64",
            expected_role.name()
        ))
    })?;
    let output_rank = min_secure_rank(key, input_width_u64).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "{} matrix has no audited SIS rank for input_width={input_width}",
            expected_role.name()
        ))
    })?;
    Ok(AuditedCommitMatrixFields {
        output_rank,
        input_width,
        sis_table_key: key,
    })
}

macro_rules! define_commit_matrix_params {
    ($name:ident, $role:expr, $description:literal) => {
        #[doc = $description]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name {
            pub(crate) output_rank: usize,
            pub(crate) input_width: usize,
            pub(crate) sis_table_key: SisTableKey,
        }

        impl $name {
            #[allow(clippy::too_many_arguments)]
            pub fn try_new(
                policy: SisSecurityPolicyId,
                table_digest: SisTableDigest,
                sis_modulus_profile: SisModulusProfileId,
                output_rank: usize,
                input_width: usize,
                coeff_linf_bound: u128,
                ring_dimension: usize,
            ) -> Result<Self, AkitaError> {
                let fields = audit_commit_matrix_fields(
                    $role,
                    policy,
                    table_digest,
                    sis_modulus_profile,
                    output_rank,
                    input_width,
                    coeff_linf_bound,
                    ring_dimension,
                )?;
                Ok(Self {
                    output_rank: fields.output_rank,
                    input_width: fields.input_width,
                    sis_table_key: fields.sis_table_key,
                })
            }

            pub fn try_new_with_min_rank(
                key: SisTableKey,
                input_width: usize,
            ) -> Result<Self, AkitaError> {
                let fields = min_rank_commit_matrix_fields($role, key, input_width)?;
                Ok(Self {
                    output_rank: fields.output_rank,
                    input_width: fields.input_width,
                    sis_table_key: fields.sis_table_key,
                })
            }

            /// Re-audit all security-sensitive matrix fields against the
            /// canonical SIS table and rank floor.
            pub fn validate(&self) -> Result<(), AkitaError> {
                let fields = audit_commit_matrix_fields(
                    $role,
                    self.security_policy(),
                    self.sis_table_key.table_digest,
                    self.sis_modulus_profile(),
                    self.output_rank(),
                    self.input_width(),
                    self.coeff_linf_bound(),
                    self.ring_dimension(),
                )?;
                if fields.sis_table_key != self.sis_table_key {
                    return Err(AkitaError::InvalidSetup(format!(
                        "{} matrix SIS table key is not canonical",
                        $role.name()
                    )));
                }
                Ok(())
            }

            #[allow(clippy::too_many_arguments)]
            pub const fn new_unchecked(
                policy: SisSecurityPolicyId,
                table_digest: SisTableDigest,
                sis_modulus_profile: SisModulusProfileId,
                output_rank: usize,
                input_width: usize,
                coeff_linf_bound: u128,
                ring_dimension: usize,
            ) -> Self {
                Self {
                    output_rank,
                    input_width,
                    sis_table_key: SisTableKey {
                        policy,
                        table_digest,
                        modulus_profile: sis_modulus_profile,
                        role: $role,
                        ring_dimension: ring_dimension as u32,
                        coeff_linf_bound,
                    },
                }
            }

            #[inline]
            pub fn output_rank(&self) -> usize {
                self.output_rank
            }

            #[inline]
            pub fn input_width(&self) -> usize {
                self.input_width
            }

            #[inline]
            pub fn security_policy(&self) -> SisSecurityPolicyId {
                self.sis_table_key.policy
            }

            #[inline]
            pub fn coeff_linf_bound(&self) -> u128 {
                self.sis_table_key.coeff_linf_bound
            }

            #[inline]
            pub fn sis_modulus_profile(&self) -> SisModulusProfileId {
                self.sis_table_key.modulus_profile
            }

            #[inline]
            pub fn sis_table_key(&self) -> SisTableKey {
                self.sis_table_key
            }

            #[inline]
            pub fn ring_dimension(&self) -> usize {
                self.sis_table_key.ring_dimension as usize
            }

            #[must_use]
            pub fn max_secure_collision_linf(&self) -> Option<u128> {
                COEFF_LINF_BUCKETS
                    .iter()
                    .copied()
                    .take_while(|&bound| {
                        let key = SisTableKey {
                            coeff_linf_bound: bound,
                            ..self.sis_table_key
                        };
                        min_secure_rank(key, self.input_width as u64)
                            .is_some_and(|rank| rank <= self.output_rank)
                    })
                    .last()
            }

            pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
                bytes.push(sis_modulus_profile_tag(self.sis_modulus_profile()));
                bytes.push(self.security_policy().tag());
                bytes.push(self.sis_table_key.role.tag());
                bytes.extend_from_slice(&self.sis_table_key.table_digest.0);
                bytes.extend_from_slice(&self.sis_table_key.ring_dimension.to_le_bytes());
                push_usize(bytes, self.output_rank());
                push_usize(bytes, self.input_width());
                push_u128(bytes, self.coeff_linf_bound());
            }
        }
    };
}

define_commit_matrix_params!(
    OuterCommitMatrixParams,
    SisMatrixRole::Outer,
    "Parameters for the outer commitment matrix (B)."
);
define_commit_matrix_params!(
    OpenCommitMatrixParams,
    SisMatrixRole::Open,
    "Parameters for the opening commitment matrix (D)."
);

#[cfg(test)]
#[path = "ajtai_key_tests.rs"]
mod tests;
