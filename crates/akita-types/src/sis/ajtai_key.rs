//! Ajtai-commitment key sizing: exact SIS profiles, explicit matrix roles, and
//! the `AjtaiKeyParams`
//! type, the secure-rank lookup, and coefficient-`L∞` bucket rounding.
//!
//! This is the single home for "given a width and a rounded-up coefficient
//! bound at a security floor, what is the minimum SIS-secure module rank, and what audited
//! `AjtaiKeyParams` does it yield". The generated SIS-floor tables it consults
//! live in the private sibling module `super::generated_sis_table`.

use akita_field::AkitaError;

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
}

/// Matrix role whose coefficient and ring geometry is being priced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SisMatrixRole {
    /// Fold witness and quotient matrix.
    A,
    /// Digit commitment matrix.
    B,
    /// Opening digit matrix.
    D,
}

impl SisMatrixRole {
    /// Stable wire/catalog tag.
    pub const fn tag(self) -> u8 {
        match self {
            Self::A => 1,
            Self::B => 2,
            Self::D => 3,
        }
    }

    /// Stable name used in generated provenance.
    pub const fn name(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::D => "D",
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

    /// Stable serialized tag.
    pub const fn tag(self) -> u8 {
        match self {
            Self::Q32Offset99 => 1,
            Self::Q64Offset59 => 2,
            Self::Q128OffsetA7F7 => 3,
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

/// Current planner ring dimensions for A. The list starts at 64 and leaves
/// room for larger A dimensions without forcing them onto B or D.
pub const A_ROLE_RING_DIMS: &[u32] = &[64, 128, 256];

/// Current planner ring dimensions for B and D, including the Q128 d=32 case.
pub const BD_ROLE_RING_DIMS: &[u32] = &[32, 64, 128, 256];

/// Production matrix roles with checked-in coverage.
pub const SIS_MATRIX_ROLES: &[SisMatrixRole] =
    &[SisMatrixRole::A, SisMatrixRole::B, SisMatrixRole::D];

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
    let (dims, bounds) = match role {
        SisMatrixRole::A => (A_ROLE_RING_DIMS, COEFF_LINF_BUCKETS),
        SisMatrixRole::B | SisMatrixRole::D => (BD_ROLE_RING_DIMS, GADGET_COEFF_LINF_ANCHORS),
    };
    if !dims.contains(&ring_dimension) || !bounds.contains(&coeff_linf_bound) {
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

fn supports_family_dimension(sis_modulus_profile: SisModulusProfileId, d: u32) -> bool {
    matches!(
        (sis_modulus_profile, d),
        (SisModulusProfileId::Q32Offset99, 32)
            | (SisModulusProfileId::Q32Offset99, 64)
            | (SisModulusProfileId::Q32Offset99, 128)
            | (SisModulusProfileId::Q32Offset99, 256)
            | (SisModulusProfileId::Q64Offset59, 32)
            | (SisModulusProfileId::Q64Offset59, 64)
            | (SisModulusProfileId::Q64Offset59, 128)
            | (SisModulusProfileId::Q64Offset59, 256)
            | (SisModulusProfileId::Q128OffsetA7F7, 32)
            | (SisModulusProfileId::Q128OffsetA7F7, 64)
            | (SisModulusProfileId::Q128OffsetA7F7, 128)
            | (SisModulusProfileId::Q128OffsetA7F7, 256)
    )
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
    if linf == 0 || !supports_family_dimension(sis_modulus_profile, d) {
        return None;
    }
    let bucket = match role {
        SisMatrixRole::A => ceil_coeff_linf_bucket(linf)?,
        SisMatrixRole::B | SisMatrixRole::D => GADGET_COEFF_LINF_ANCHORS
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
    if table_digest != SisTableDigest::CURRENT || policy != DEFAULT_SIS_SECURITY_POLICY {
        return None;
    }
    generated_sis_max_widths(policy, modulus_profile, d, coeff_linf_bound)
}

/// Minimum generated SIS-secure module rank that supports `width` ring columns
/// at an already rounded-up coefficient-`L∞` bucket.
///
/// Returns `None` when no generated SIS-floor row covers the configuration.
pub fn min_secure_rank(key: SisTableKey, width: u64) -> Option<usize> {
    let _role_cell = sis_role_cell(
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
    for (i, &max_w) in widths.iter().enumerate() {
        if width <= max_w {
            return Some(i + 1);
        }
    }
    None
}

/// Parameters for a single Ajtai commitment matrix.
///
/// Each matrix in the protocol (A, B, D) is characterised by its row count
/// (security rank), column count (message width), and the generated SIS-floor
/// key used for security sizing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AjtaiKeyParams {
    pub(crate) row_len: usize,
    pub(crate) col_len: usize,
    pub(crate) sis_table_key: SisTableKey,
}

impl AjtaiKeyParams {
    /// Create a new SIS-secure `AjtaiKeyParams`, auditing the
    /// `(row_len, col_len, sis_table_key)` tuple against the generated
    /// coefficient-`L∞` SIS-floor tables.
    ///
    /// The check is strict and has no silent-permissive fallback: a zero field,
    /// an unsupported collision bucket, a `col_len` outside the audited range,
    /// or a `row_len` below the audited SIS-secure floor is reported as
    /// `AkitaError::InvalidSetup(message)`. Used by callers that must gracefully
    /// reject SIS-insecure candidates (e.g. the planner's outer loop).
    ///
    /// # Errors
    ///
    /// Returns an error if any field is zero, if the SIS-floor tables do not
    /// cover the configuration, or if `row_len` is below the audited floor.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        policy: SisSecurityPolicyId,
        table_digest: SisTableDigest,
        sis_modulus_profile: SisModulusProfileId,
        role: SisMatrixRole,
        row_len: usize,
        col_len: usize,
        coeff_linf_bound: u128,
        ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if row_len == 0 {
            return Err(AkitaError::InvalidSetup(
                "AjtaiKeyParams: row_len = 0".to_string(),
            ));
        }
        if col_len == 0 {
            return Err(AkitaError::InvalidSetup(
                "AjtaiKeyParams: col_len = 0".to_string(),
            ));
        }
        let Some(key) = sis_table_key_for_linf_bound(
            policy,
            table_digest,
            sis_modulus_profile,
            role,
            ring_dimension as u32,
            coeff_linf_bound,
        ) else {
            return Err(AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: no audited SIS table key for \
                     policy={} profile={sis_modulus_profile:?} \
                     d={ring_dimension} coeff_linf_bound={coeff_linf_bound}",
                policy.name()
            )));
        };
        let floor = min_secure_rank(key, col_len as u64).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: no audited SIS rank for \
                     policy={} profile={sis_modulus_profile:?} \
                     d={ring_dimension} coeff_linf_bound={} col_len={col_len}",
                policy.name(),
                key.coeff_linf_bound
            ))
        })?;
        if row_len < floor {
            return Err(AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: row_len {row_len} < SIS floor {floor} \
                 (policy={}, profile={sis_modulus_profile:?}, \
                 d={ring_dimension}, coeff_linf_bound={}, col_len={col_len})",
                policy.name(),
                key.coeff_linf_bound
            )));
        }
        Ok(Self {
            row_len,
            col_len,
            sis_table_key: key,
        })
    }

    /// Create a SIS-secure `AjtaiKeyParams`, sizing `row_len` to the audited
    /// SIS floor for `col_len` columns under `key`.
    ///
    /// Computes the minimum SIS-secure module rank with [`min_secure_rank`] and
    /// forwards it (along with the fields carried by `key`) to
    /// [`try_new`](Self::try_new). Callers that would otherwise thread an
    /// explicit `min_secure_rank` result next to `try_new` should use this
    /// instead.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when no generated SIS-floor row
    /// covers `(key, col_len)`, or when [`try_new`](Self::try_new) rejects the
    /// resulting tuple.
    pub fn try_new_with_min_rank(key: SisTableKey, col_len: usize) -> Result<Self, AkitaError> {
        if col_len == 0 {
            return Err(AkitaError::InvalidSetup(
                "AjtaiKeyParams: col_len = 0".to_string(),
            ));
        }
        let row_len = min_secure_rank(key, col_len as u64).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: no audited SIS rank for \
                 policy={} profile={:?} d={} coeff_linf_bound={} col_len={col_len}",
                key.policy.name(),
                key.modulus_profile,
                key.ring_dimension,
                key.coeff_linf_bound
            ))
        })?;
        Ok(Self {
            row_len,
            col_len,
            sis_table_key: key,
        })
    }

    /// Create a new `AjtaiKeyParams` without enforcing SIS security.
    ///
    /// Use this only for intermediate construction steps that carry
    /// incomplete data (`params_only` placeholders with `col_len = 0` or a
    /// zero coefficient bucket, iterative SIS fixed-point loops, etc.) and for
    /// synthetic test/descriptor/proof-size layouts that intentionally carry
    /// degenerate ranks. Production-facing schedule layouts are built through
    /// [`try_new`](Self::try_new), which audits the SIS floor against the final
    /// width as the key is constructed.
    #[allow(clippy::too_many_arguments)]
    pub fn new_unchecked(
        policy: SisSecurityPolicyId,
        table_digest: SisTableDigest,
        sis_modulus_profile: SisModulusProfileId,
        role: SisMatrixRole,
        row_len: usize,
        col_len: usize,
        coeff_linf_bound: u128,
        ring_dimension: usize,
    ) -> Self {
        Self {
            row_len,
            col_len,
            sis_table_key: SisTableKey {
                policy,
                table_digest,
                modulus_profile: sis_modulus_profile,
                role,
                ring_dimension: ring_dimension as u32,
                coeff_linf_bound,
            },
        }
    }

    /// Number of rows.
    #[inline]
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// Number of columns.
    #[inline]
    pub fn col_len(&self) -> usize {
        self.col_len
    }

    /// SIS policy used to size and validate this key.
    #[inline]
    pub fn security_policy(&self) -> SisSecurityPolicyId {
        self.sis_table_key.policy
    }

    /// Rounded coefficient-`L∞` bucket for SIS sizing.
    #[inline]
    pub fn coeff_linf_bound(&self) -> u128 {
        self.sis_table_key.coeff_linf_bound
    }

    /// Exact SIS modulus profile used to validate this key.
    #[inline]
    pub fn sis_modulus_profile(&self) -> SisModulusProfileId {
        self.sis_table_key.modulus_profile
    }

    /// Full generated-table key used to validate this key.
    #[inline]
    pub fn sis_table_key(&self) -> SisTableKey {
        self.sis_table_key
    }

    /// Largest audited coefficient-`L∞` bucket supported by this fixed matrix.
    ///
    /// This inverts the generated SIS floor table using the matrix's exact
    /// role, ring dimension, input width, output rank, policy, profile, and
    /// table digest. It does not run the lattice estimator.
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
                min_secure_rank(key, self.col_len as u64).is_some_and(|rank| rank <= self.row_len)
            })
            .last()
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(sis_modulus_profile_tag(self.sis_modulus_profile()));
        bytes.push(self.security_policy().tag());
        bytes.push(self.sis_table_key.role.tag());
        bytes.extend_from_slice(&self.sis_table_key.table_digest.0);
        bytes.extend_from_slice(&self.sis_table_key.ring_dimension.to_le_bytes());
        push_usize(bytes, self.row_len());
        push_usize(bytes, self.col_len());
        push_u128(bytes, self.coeff_linf_bound());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_shape_rejects_linf_bucket() {
        assert_eq!(
            ceil_supported_linf_bound(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q32Offset99,
                SisMatrixRole::A,
                31,
                7,
            ),
            None
        );
    }

    #[test]
    fn fixed_matrix_capacity_inverts_the_checked_sis_table() {
        let key = SisTableKey {
            policy: DEFAULT_SIS_SECURITY_POLICY,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            role: SisMatrixRole::A,
            ring_dimension: 64,
            coeff_linf_bound: 32_767,
        };
        let matrix = AjtaiKeyParams::try_new_with_min_rank(key, 64).expect("audited matrix");
        let capacity = matrix
            .max_secure_collision_linf()
            .expect("fixed matrix capacity");
        assert!(capacity >= key.coeff_linf_bound);
        for &larger in COEFF_LINF_BUCKETS.iter().filter(|&&bound| bound > capacity) {
            let larger_key = SisTableKey {
                coeff_linf_bound: larger,
                ..key
            };
            assert!(
                min_secure_rank(larger_key, matrix.col_len() as u64)
                    .is_none_or(|rank| rank > matrix.row_len()),
                "capacity must be the largest bucket supported by the fixed matrix"
            );
        }
    }

    #[test]
    fn floor_slices_have_family_specific_rank_caps() {
        let bucket = 15;
        if generated_sis_max_widths(
            DEFAULT_SIS_SECURITY_POLICY,
            SisModulusProfileId::Q32Offset99,
            32,
            bucket,
        )
        .is_some()
        {
            assert!(generated_sis_max_widths(
                DEFAULT_SIS_SECURITY_POLICY,
                SisModulusProfileId::Q32Offset99,
                32,
                bucket,
            )
            .is_some());
        }
    }

    #[test]
    fn linf_key_rounds_to_coefficient_bucket() {
        let linf = 1_048_575u128;
        let key = sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q32Offset99,
            SisMatrixRole::A,
            128,
            linf,
        );
        if let Some(key) = key {
            assert_eq!(key.coeff_linf_bound, linf);
            assert_eq!(key.policy, DEFAULT_SIS_SECURITY_POLICY);
        }
    }

    #[test]
    fn coeff_linf_bucket_ladder_matches_main_ceiling() {
        assert_eq!(ceil_coeff_linf_bucket(1_048_574), Some(1_048_575));
        assert_eq!(ceil_coeff_linf_bucket(1_048_575), Some(1_048_575));
        assert_eq!(ceil_coeff_linf_bucket(1_048_576), Some(2_097_151));
    }
}
