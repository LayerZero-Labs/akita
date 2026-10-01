//! Generated Euclidean SIS table lookup for complete physical L2 collisions.

use super::ajtai_key::{SisModulusProfileId, SisSecurityPolicyId, DEFAULT_SIS_SECURITY_POLICY};
use super::generated_l2_sis_table::{sis_max_widths as generated_l2_sis_max_widths, TABLE_DIGEST};

/// Digest of the separate generated Euclidean SIS table and its boundary
/// evidence.
///
/// This identity is distinct from [`super::ajtai_key::SisTableDigest`]. An L2-selected schedule
/// binds both its squared collision bucket and this digest without changing the
/// coefficient-L∞ fallback table identity.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct SisL2TableDigest(pub [u8; 32]);

impl Default for SisL2TableDigest {
    fn default() -> Self {
        Self::CURRENT
    }
}

impl SisL2TableDigest {
    /// Stable wire tag for the L2 digest field.
    pub const TAG: u8 = 1;

    /// SHA-256 digest of the audit CSV emitted while generating the current L2 table.
    pub const CURRENT: Self = Self(TABLE_DIGEST);
}

/// Canonical key for one generated Euclidean SIS floor row.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct SisL2TableKey {
    pub policy: SisSecurityPolicyId,
    pub table_digest: SisL2TableDigest,
    pub modulus_profile: SisModulusProfileId,
    pub ring_dimension: u32,
    /// Rounded squared L2 norm of the complete scalar collision vector.
    pub collision_l2_sq: u128,
}

const MIN_COLLISION_SQ_BUCKET: u128 = 1u128 << 1;
const MAX_COLLISION_SQ_BUCKET: u128 = 1u128 << 84;

/// Round a complete scalar collision-vector squared L2 norm to the generated
/// ADPS16 quantum table ladder.
#[must_use]
pub fn ceil_supported_l2_collision_sq(collision_l2_sq: u128) -> Option<u128> {
    if collision_l2_sq == 0 {
        return None;
    }
    let bucket = collision_l2_sq
        .checked_next_power_of_two()?
        .max(MIN_COLLISION_SQ_BUCKET);
    (bucket <= MAX_COLLISION_SQ_BUCKET).then_some(bucket)
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
