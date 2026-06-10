//! Ajtai-commitment key sizing: the SIS modulus family, the `AjtaiKeyParams`
//! type, the secure-rank lookup, and collision-bucket rounding.
//!
//! This is the single home for "given a width and a rounded-up collision
//! bound, what is the minimum SIS-secure module rank, and what audited
//! `AjtaiKeyParams` does it yield". The generated SIS-floor tables it consults
//! live in the private sibling module `super::generated_sis_table`.

use akita_field::AkitaError;

use super::generated_sis_table::sis_max_widths;
use crate::descriptor_bytes::{push_u128, push_usize, sis_family_tag};

/// SIS modulus family used to select generated security floors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SisModulusFamily {
    /// Representative q = 2^32 - 99.
    Q32,
    /// Representative q = 2^64 - 59.
    Q64,
    /// Representative q = 2^128 - (2^32 - 22537).
    #[default]
    Q128,
}

/// Smallest power-of-two squared-collision bucket exponent in the generated
/// ladder. Keep in lockstep with `MIN_LOG_BUCKET` in `scripts/gen_sis_table.py`.
pub const MIN_LOG_BUCKET: u32 = 1;

/// Largest power-of-two squared-collision bucket exponent in the generated
/// ladder. Keep in lockstep with `MAX_LOG_BUCKET` in `scripts/gen_sis_table.py`.
pub const MAX_LOG_BUCKET: u32 = 84;

/// Coefficient-`L∞` collision buckets for norm-bound sizing.
///
/// Complements the power-of-two `collision_l2_sq` ladder in `generated_sis_table/`:
/// derived keys `K = d · B²` for `B` in this table are the default lookup for
/// collisions that enter through an `L∞` envelope (A/B/D norm_bound). Keep in
/// lockstep with `COEFF_LINF_BUCKETS` in `scripts/gen_sis_table.py`.
pub const COEFF_LINF_BUCKETS: &[u128] = &[
    2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131_071,
    262_143, 524_287, 1_048_575, 2_097_151, 4_194_303, 8_388_607, 16_777_215, 33_554_431,
    67_108_863,
];

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

/// Derived squared-Euclidean table key `K = d · B²` for `B = ceil_coeff_linf_bucket(linf)`.
///
/// Returns `None` when `linf` is zero, arithmetic overflows, or the stitched table
/// has no row for `(sis_family, d, K)`.
#[must_use]
pub fn derived_collision_l2_sq_key(
    sis_family: SisModulusFamily,
    d: u32,
    linf: u128,
) -> Option<u128> {
    let bucket = ceil_coeff_linf_bucket(linf)?;
    let key = bucket.checked_mul(bucket)?.checked_mul(u128::from(d))?;
    sis_max_widths(sis_family, d, key).map(|_| key)
}

/// Default L2 collision bucket for an `L∞`-originating envelope.
///
/// Prefers the derived key `d · ceil(linf)²` when tabulated; otherwise rounds the
/// raw `d · linf²` up to the next generated power-of-two bucket.
#[must_use]
pub fn collision_l2_sq_for_linf_envelope(
    sis_family: SisModulusFamily,
    d: u32,
    linf: u128,
) -> Option<u128> {
    if let Some(key) = derived_collision_l2_sq_key(sis_family, d, linf) {
        return Some(key);
    }
    let raw = linf.checked_mul(linf)?.checked_mul(u128::from(d))?;
    ceil_supported_collision(sis_family, d, raw)
}

fn supports_family_dimension(sis_family: SisModulusFamily, d: u32) -> bool {
    matches!(
        (sis_family, d),
        (SisModulusFamily::Q32, 32)
            | (SisModulusFamily::Q32, 64)
            | (SisModulusFamily::Q32, 128)
            | (SisModulusFamily::Q32, 256)
            | (SisModulusFamily::Q64, 32)
            | (SisModulusFamily::Q64, 64)
            | (SisModulusFamily::Q64, 128)
            | (SisModulusFamily::Q64, 256)
            | (SisModulusFamily::Q128, 32)
            | (SisModulusFamily::Q128, 64)
            | (SisModulusFamily::Q128, 128)
            | (SisModulusFamily::Q128, 256)
    )
}

/// Minimum generated SIS-secure module rank that supports `width` ring columns
/// at an **already rounded-up** collision bucket `collision_l2_sq_rounded_up`
/// (see [`ceil_supported_collision`]).
///
/// Returns `None` when no generated SIS-floor row covers the configuration.
pub fn min_secure_rank(
    sis_family: SisModulusFamily,
    d: u32,
    collision_l2_sq_rounded_up: u128,
    width: u64,
) -> Option<usize> {
    let widths = sis_max_widths(sis_family, d, collision_l2_sq_rounded_up)?;
    for (i, &max_w) in widths.iter().enumerate() {
        if width <= max_w {
            return Some(i + 1);
        }
    }
    None
}

/// Round a raw per-ring-row squared Euclidean collision bound up to the next
/// generated power-of-two bucket for `(sis_family, d)`.
///
/// Returns `None` for an unsupported `(sis_family, d)`, a zero collision, or a
/// collision above the largest tabulated bucket.
pub fn ceil_supported_collision(
    sis_family: SisModulusFamily,
    d: u32,
    collision_l2_sq: u128,
) -> Option<u128> {
    if collision_l2_sq == 0 || !supports_family_dimension(sis_family, d) {
        return None;
    }
    let min_bucket = 1u128.checked_shl(MIN_LOG_BUCKET)?;
    let max_bucket = 1u128.checked_shl(MAX_LOG_BUCKET)?;
    let bucket = if collision_l2_sq <= min_bucket {
        min_bucket
    } else if collision_l2_sq.is_power_of_two() {
        collision_l2_sq
    } else {
        collision_l2_sq.checked_next_power_of_two()?
    };
    if bucket > max_bucket {
        return None;
    }
    sis_max_widths(sis_family, d, bucket)?;
    Some(bucket)
}

/// Parameters for a single Ajtai commitment matrix.
///
/// Each matrix in the protocol (A, B, D) is characterised by its row count
/// (security rank), column count (message width), and the worst-case per-ring-row
/// squared Euclidean collision bound used for SIS security sizing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AjtaiKeyParams {
    pub(crate) row_len: usize,
    pub(crate) col_len: usize,
    pub(crate) collision_l2_sq: u128,
    pub(crate) sis_family: SisModulusFamily,
}

impl AjtaiKeyParams {
    /// Create a new SIS-secure `AjtaiKeyParams`, auditing the
    /// `(row_len, col_len, collision_l2_sq)` triple against the generated 128-bit
    /// SIS-floor tables for `(sis_family, ring_dimension)`.
    ///
    /// The check is strict and has no silent-permissive fallback: a zero field,
    /// an unsupported collision bucket, a `col_len` outside the audited range,
    /// or a `row_len` below the audited SIS-secure floor is reported as
    /// `AkitaError::InvalidSetup(message)`. Used by callers that must gracefully
    /// reject SIS-insecure candidates (e.g. the planner's outer loop).
    ///
    /// # Errors
    ///
    /// Returns an error if any of `row_len`, `col_len`, or `collision_l2_sq` is
    /// zero, if the SIS-floor tables do not cover the configuration, or if
    /// `row_len` is below the audited floor.
    pub fn try_new(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_l2_sq: u128,
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
        if collision_l2_sq == 0 {
            return Err(AkitaError::InvalidSetup(
                "AjtaiKeyParams: collision_l2_sq = 0".to_string(),
            ));
        }
        let floor = min_secure_rank(
            sis_family,
            ring_dimension as u32,
            collision_l2_sq,
            col_len as u64,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: no audited SIS rank for \
                     family={sis_family:?} d={ring_dimension} \
                     collision_l2_sq={collision_l2_sq} col_len={col_len}"
            ))
        })?;
        if row_len < floor {
            return Err(AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: row_len {row_len} < SIS floor {floor} \
                 (family={sis_family:?}, d={ring_dimension}, \
                 collision_l2_sq={collision_l2_sq}, col_len={col_len})"
            )));
        }
        Ok(Self {
            row_len,
            col_len,
            collision_l2_sq,
            sis_family,
        })
    }

    /// Create a new `AjtaiKeyParams` without enforcing SIS security.
    ///
    /// Use this only for intermediate construction steps that carry
    /// incomplete data (`params_only` placeholders with `col_len = 0` or
    /// `collision_l2_sq = 0`, iterative SIS fixed-point loops, etc.).
    /// Production-facing layouts must reach [`try_new`](Self::try_new) before
    /// they're emitted into a schedule or setup.
    pub fn new_unchecked(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_l2_sq: u128,
        ring_dimension: usize,
    ) -> Self {
        let _ = ring_dimension;
        Self {
            row_len,
            col_len,
            collision_l2_sq,
            sis_family,
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

    /// Rounded-up per-ring-row squared Euclidean collision bucket for SIS sizing.
    #[inline]
    pub fn collision_l2_sq(&self) -> u128 {
        self.collision_l2_sq
    }

    /// SIS modulus family used to validate this key.
    #[inline]
    pub fn sis_family(&self) -> SisModulusFamily {
        self.sis_family
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(sis_family_tag(self.sis_family()));
        push_usize(bytes, self.row_len());
        push_usize(bytes, self.col_len());
        push_u128(bytes, self.collision_l2_sq());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_supported_collision_rounds_to_power_of_two() {
        assert_eq!(
            ceil_supported_collision(SisModulusFamily::Q32, 32, 5),
            sis_max_widths(SisModulusFamily::Q32, 32, 8).map(|_| 8)
        );
        assert_eq!(
            ceil_supported_collision(SisModulusFamily::Q32, 32, 8),
            Some(8)
        );
    }

    #[test]
    fn floor_slices_have_family_specific_rank_caps() {
        let bucket = 1u128 << 4;
        if sis_max_widths(SisModulusFamily::Q32, 32, bucket).is_some() {
            assert_eq!(
                sis_max_widths(SisModulusFamily::Q32, 32, bucket).map(<[u64]>::len),
                Some(20)
            );
        }
    }

    #[test]
    fn derived_key_prefers_exact_d_linf_bucket_sq_over_pow2_ceil() {
        let linf = 1_048_575u128;
        let derived = derived_collision_l2_sq_key(SisModulusFamily::Q32, 128, linf).unwrap();
        let pow2 =
            ceil_supported_collision(SisModulusFamily::Q32, 128, 128u128 * linf * linf).unwrap();
        assert_ne!(derived, pow2);
        assert_eq!(
            min_secure_rank(SisModulusFamily::Q32, 128, derived, 32_768),
            Some(10)
        );
        assert_eq!(
            min_secure_rank(SisModulusFamily::Q32, 128, pow2, 32_768),
            None
        );
    }

    #[test]
    fn coeff_linf_bucket_ladder_matches_main_ceiling() {
        assert_eq!(ceil_coeff_linf_bucket(1_048_574), Some(1_048_575));
        assert_eq!(ceil_coeff_linf_bucket(1_048_575), Some(1_048_575));
        assert_eq!(ceil_coeff_linf_bucket(1_048_576), Some(2_097_151));
    }
}
