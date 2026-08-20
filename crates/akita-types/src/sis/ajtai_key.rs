//! Ajtai-commitment key sizing: exact SIS profiles, role-specific matrix
//! parameter types, secure-rank lookup, and coefficient-`L∞` bucket rounding.
//!
//! This is the single home for "given a width and a rounded-up coefficient
//! bound at a security floor, what is the minimum SIS-secure module rank, and what audited
//! commit-matrix parameters does it yield". The generated SIS-floor tables it consults
//! live in the private sibling module `super::generated_sis_table`.

use akita_field::AkitaError;

use super::coverage::{sis_role_cell, GADGET_COEFF_LINF_ANCHORS};
use super::generated_sis_table::{sis_max_widths as generated_sis_max_widths, SIS_TABLE_DIGEST};
use super::l2_table::{min_secure_l2_rank, SisL2TableKey};
use super::physical_l2::{InnerCommitSecurityRoute, PhysicalL2NormProofShape};
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
    pub const CURRENT: Self = Self(SIS_TABLE_DIGEST);
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

#[derive(Debug, Clone, Copy)]
struct AuditedCommitMatrixFields {
    output_rank: usize,
    input_width: usize,
    sis_table_key: SisTableKey,
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

/// Parameters for the inner commitment matrix (A).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InnerCommitMatrixParams {
    pub(crate) output_rank: usize,
    pub(crate) input_width: usize,
    pub(crate) security_route: InnerCommitSecurityRoute,
}

fn validate_l2_response_shape(
    table_key: SisL2TableKey,
    input_width: usize,
    norm_proof_shape: PhysicalL2NormProofShape,
) -> Result<(), AkitaError> {
    norm_proof_shape.validate()?;
    let physical_response_len = input_width
        .checked_mul(table_key.ring_dimension as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("L2 physical response length overflow".into()))?;
    if norm_proof_shape.physical_response_len() != physical_response_len {
        return Err(AkitaError::InvalidSetup(format!(
            "L2 norm shape response length {} does not match A matrix physical length {physical_response_len}",
            norm_proof_shape.physical_response_len()
        )));
    }
    Ok(())
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
        validate_l2_response_shape(table_key, input_width, norm_proof_shape)?;
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

    /// Rebuild this matrix for a layout-derived width while preserving its route.
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
                validate_l2_response_shape(table_key, self.input_width, norm_proof_shape)?;
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

    /// Input dimension after expanding module coordinates into ring coefficients.
    #[must_use]
    pub fn raw_input_dimension(&self) -> Option<usize> {
        self.input_width.checked_mul(self.ring_dimension())
    }

    /// Output dimension after expanding module coordinates into ring coefficients.
    #[must_use]
    pub fn raw_output_dimension(&self) -> Option<usize> {
        self.output_rank.checked_mul(self.ring_dimension())
    }

    #[must_use]
    pub const fn sis_table_key(&self) -> Option<SisTableKey> {
        match self.security_route {
            InnerCommitSecurityRoute::Linf(key) => Some(key),
            InnerCommitSecurityRoute::L2 { .. } => None,
        }
    }

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

mod sealed {
    /// Prevents downstream crates from inventing a table-keyed matrix role.
    pub trait Sealed {}
}

/// A commitment-matrix role whose identity is a plain audited SIS table key.
///
/// Only the B and D roles qualify. The A role carries an
/// [`InnerCommitSecurityRoute`] instead, so its `sis_table_key` and
/// `coeff_linf_bound` are optional and its `validate` branches on the route;
/// forcing it into this generic would reintroduce exactly the `Option`-shaped
/// tag that the parameter-consolidation plan rejects. What the three roles share
/// is the audit code, which they already do:
/// [`audit_commit_matrix_fields`] and [`min_rank_commit_matrix_fields`] serve all
/// three.
pub trait LinfMatrixRole: sealed::Sealed {
    /// The protocol role this marker stands for.
    const ROLE: SisMatrixRole;
}

/// Marker for the outer commitment matrix (B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Outer;

/// Marker for the opening commitment matrix (D).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Open;

impl sealed::Sealed for Outer {}
impl sealed::Sealed for Open {}

impl LinfMatrixRole for Outer {
    const ROLE: SisMatrixRole = SisMatrixRole::Outer;
}

impl LinfMatrixRole for Open {
    const ROLE: SisMatrixRole = SisMatrixRole::Open;
}

/// One audited L-infinity Ajtai matrix identity.
///
/// Replaces two byte-identical macro expansions. The role moves from the type
/// *name* into a type *parameter*, so the ~170 lines of constructors, accessors,
/// validation, and encoding exist once instead of twice.
///
/// This does not change what the type system enforces. The macro already emitted
/// two distinct structs, so a B matrix could never be passed where a D matrix was
/// required; `LinfCommitMatrix<Outer>` and `LinfCommitMatrix<Open>` are equally
/// distinct. What changes is that the shared behaviour has one definition, and
/// that [`LinfMatrixRole`] is sealed, so the set of table-keyed roles cannot grow
/// outside this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LinfCommitMatrix<R: LinfMatrixRole> {
    pub(crate) output_rank: usize,
    pub(crate) input_width: usize,
    pub(crate) sis_table_key: SisTableKey,
    role: core::marker::PhantomData<R>,
}

/// Parameters for the outer commitment matrix (B).
///
/// Kept as a permanent alias: the name documents the protocol role at every
/// signature and at each generated static-table construction site.
pub type OuterCommitMatrixParams = LinfCommitMatrix<Outer>;

/// Parameters for the opening commitment matrix (D).
///
/// Kept as a permanent alias, for the same reason as
/// [`OuterCommitMatrixParams`].
pub type OpenCommitMatrixParams = LinfCommitMatrix<Open>;

impl<R: LinfMatrixRole> LinfCommitMatrix<R> {
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
            R::ROLE,
            policy,
            table_digest,
            sis_modulus_profile,
            output_rank,
            input_width,
            coeff_linf_bound,
            ring_dimension,
        )?;
        Ok(Self::from_audited(fields))
    }

    pub fn try_new_with_min_rank(key: SisTableKey, input_width: usize) -> Result<Self, AkitaError> {
        // `min_rank_commit_matrix_fields` rejects `key.role != R::ROLE`, so a
        // caller cannot smuggle another role's key into this slot.
        let fields = min_rank_commit_matrix_fields(R::ROLE, key, input_width)?;
        Ok(Self::from_audited(fields))
    }

    #[inline]
    fn from_audited(fields: AuditedCommitMatrixFields) -> Self {
        Self {
            output_rank: fields.output_rank,
            input_width: fields.input_width,
            sis_table_key: fields.sis_table_key,
            role: core::marker::PhantomData,
        }
    }

    /// Re-audit all security-sensitive matrix fields against the
    /// canonical SIS table and rank floor.
    pub fn validate(&self) -> Result<(), AkitaError> {
        let fields = audit_commit_matrix_fields(
            R::ROLE,
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
                R::ROLE.name()
            )));
        }
        Ok(())
    }

    /// Assemble a matrix identity without auditing it.
    ///
    /// `const` because the generated schedule tables construct these in `static`
    /// position. `PhantomData` is zero-sized and `const`-constructible, so the
    /// emitted literals keep the shape the emitter already writes.
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
                role: R::ROLE,
                ring_dimension: ring_dimension as u32,
                coeff_linf_bound,
            },
            role: core::marker::PhantomData,
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

    /// Input dimension after expanding module coordinates into raw ring coefficients.
    #[inline]
    pub fn raw_input_dimension(&self) -> Option<usize> {
        self.input_width.checked_mul(self.ring_dimension())
    }

    /// Output dimension after expanding module coordinates into raw ring coefficients.
    #[inline]
    pub fn raw_output_dimension(&self) -> Option<usize> {
        self.output_rank.checked_mul(self.ring_dimension())
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

    /// Byte-identical to the two macro expansions it replaces.
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

#[cfg(test)]
#[path = "ajtai_key/artifact_tests.rs"]
mod artifact_tests;

#[cfg(test)]
#[path = "ajtai_key_tests.rs"]
mod l2_tests;

#[cfg(test)]
#[path = "ajtai_key/table_tests.rs"]
mod tests;

#[cfg(test)]
mod linf_matrix_tests {
    use super::*;

    /// The generated tables build these in `static` position, so the constructor
    /// must stay usable in a const context with the `PhantomData` field present.
    const STATIC_OUTER: OuterCommitMatrixParams = OuterCommitMatrixParams::new_unchecked(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        128,
        7,
        64,
    );

    #[test]
    fn role_markers_stamp_their_protocol_role() {
        assert_eq!(<Outer as LinfMatrixRole>::ROLE, SisMatrixRole::Outer);
        assert_eq!(<Open as LinfMatrixRole>::ROLE, SisMatrixRole::Open);
        assert_eq!(STATIC_OUTER.sis_table_key().role, SisMatrixRole::Outer);
    }

    #[test]
    fn const_construction_is_zero_sized_in_the_role_parameter() {
        // `PhantomData` must not widen the value: a role-parameterised matrix has
        // to stay the same size as the untagged fields it carries, or the
        // generated tables grow for nothing.
        assert_eq!(
            core::mem::size_of::<OuterCommitMatrixParams>(),
            core::mem::size_of::<OpenCommitMatrixParams>()
        );
        assert_eq!(core::mem::size_of::<Outer>(), 0);
        assert_eq!(core::mem::size_of::<Open>(), 0);
    }

    #[test]
    fn min_rank_construction_rejects_another_roles_key() {
        // The D slot must not accept a B-role table key. This is the check the
        // sealed role parameter makes unambiguous.
        let outer_key = STATIC_OUTER.sis_table_key();
        assert!(OpenCommitMatrixParams::try_new_with_min_rank(outer_key, 128).is_err());
        assert!(OuterCommitMatrixParams::try_new_with_min_rank(outer_key, 128).is_ok());
    }

    #[test]
    fn descriptor_bytes_carry_the_role_tag() {
        // The role tag is part of the descriptor, so two roles with otherwise
        // identical fields must not encode alike.
        let open = OpenCommitMatrixParams::new_unchecked(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            128,
            7,
            64,
        );
        let mut outer_bytes = Vec::new();
        STATIC_OUTER.append_descriptor_bytes(&mut outer_bytes);
        let mut open_bytes = Vec::new();
        open.append_descriptor_bytes(&mut open_bytes);
        assert_eq!(outer_bytes.len(), open_bytes.len());
        assert_ne!(outer_bytes, open_bytes);
    }
}
