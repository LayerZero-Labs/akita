//! Ajtai-commitment key sizing: the SIS modulus family, the `AjtaiKeyParams`
//! type, the secure-rank lookup, and coefficient-`L∞` collision buckets.
//!
//! Security floors live in `super::generated_sis_linf_table`.

use akita_field::AkitaError;

use super::generated_sis_linf_table::sis_max_widths;
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

/// Coefficient-`L∞` collision buckets for norm-bound sizing.
///
/// Keep in lockstep with `COEFF_LINF_BUCKETS` in `scripts/gen_sis_linf_table.py`.
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

/// Audited coefficient-`L∞` table key for a raw per-ring-row collision envelope.
#[must_use]
pub fn collision_linf_bucket_for_envelope(linf: u128) -> Option<u128> {
    ceil_coeff_linf_bucket(linf)
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

/// Minimum generated SIS-secure module rank for `width` ring columns at an audited
/// coefficient-`L∞` collision bucket `collision_linf`.
#[must_use]
pub fn min_secure_rank(
    sis_family: SisModulusFamily,
    d: u32,
    collision_linf: u128,
    width: u64,
) -> Option<usize> {
    if !supports_family_dimension(sis_family, d) {
        return None;
    }
    let widths = sis_max_widths(sis_family, d, collision_linf)?;
    for (i, &max_w) in widths.iter().enumerate() {
        if width <= max_w {
            return Some(i + 1);
        }
    }
    None
}

/// Parameters for a single Ajtai commitment matrix.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AjtaiKeyParams {
    pub(crate) row_len: usize,
    pub(crate) col_len: usize,
    pub(crate) collision_linf: u128,
    pub(crate) sis_family: SisModulusFamily,
}

impl AjtaiKeyParams {
    /// Create a new SIS-secure `AjtaiKeyParams`, auditing the
    /// `(row_len, col_len, collision_linf)` triple against the generated floors.
    ///
    /// # Errors
    ///
    /// Returns an error if any field is zero, the table misses the bucket, or
    /// `row_len` is below the audited SIS floor.
    pub fn try_new(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_linf: u128,
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
        if collision_linf == 0 {
            return Err(AkitaError::InvalidSetup(
                "AjtaiKeyParams: collision_linf = 0".to_string(),
            ));
        }
        let floor = min_secure_rank(
            sis_family,
            ring_dimension as u32,
            collision_linf,
            col_len as u64,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: no audited SIS rank for \
                     family={sis_family:?} d={ring_dimension} \
                     collision_linf={collision_linf} col_len={col_len}"
            ))
        })?;
        if row_len < floor {
            return Err(AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: row_len {row_len} < SIS floor {floor} \
                 (family={sis_family:?}, d={ring_dimension}, \
                 collision_linf={collision_linf}, col_len={col_len})"
            )));
        }
        Ok(Self {
            row_len,
            col_len,
            collision_linf,
            sis_family,
        })
    }

    /// Create a new `AjtaiKeyParams` without enforcing SIS security.
    pub fn new_unchecked(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_linf: u128,
        ring_dimension: usize,
    ) -> Self {
        let _ = ring_dimension;
        Self {
            row_len,
            col_len,
            collision_linf,
            sis_family,
        }
    }

    #[inline]
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    #[inline]
    pub fn col_len(&self) -> usize {
        self.col_len
    }

    /// Audited coefficient-`L∞` collision bucket for SIS sizing.
    #[inline]
    pub fn collision_linf(&self) -> u128 {
        self.collision_linf
    }

    #[inline]
    pub fn sis_family(&self) -> SisModulusFamily {
        self.sis_family
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(sis_family_tag(self.sis_family()));
        push_usize(bytes, self.row_len());
        push_usize(bytes, self.col_len());
        push_u128(bytes, self.collision_linf());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coeff_linf_bucket_ladder_matches_main_ceiling() {
        assert_eq!(ceil_coeff_linf_bucket(1_048_574), Some(1_048_575));
        assert_eq!(ceil_coeff_linf_bucket(1_048_575), Some(1_048_575));
        assert_eq!(ceil_coeff_linf_bucket(1_048_576), Some(2_097_151));
    }

    #[test]
    fn fp128_d64_dense_l1_counterexample_bucket() {
        let linf = 997_248u128;
        let bucket = ceil_coeff_linf_bucket(linf).unwrap();
        assert_eq!(bucket, 1_048_575);
    }

    #[test]
    fn min_secure_rank_smoke_q128_d64_production_bucket() {
        // Representative production geometry: bucket 1_048_575, modest width.
        let rank = min_secure_rank(SisModulusFamily::Q128, 64, 1_048_575, 4_096);
        assert!(
            rank.is_some(),
            "generated_sis_linf_table must cover (Q128, d=64, B=1_048_575)"
        );
        assert!(rank.unwrap() >= 1);
    }

    #[test]
    fn fp128_d64_dense_l1_counterexample_rank() {
        // Spec counterexample: direct L∞ needs rank 5 at width 3_790, bucket 1_048_575.
        assert_eq!(
            min_secure_rank(SisModulusFamily::Q128, 64, 1_048_575, 3_790),
            Some(5),
        );
        assert!(AjtaiKeyParams::try_new(SisModulusFamily::Q128, 4, 3_790, 1_048_575, 64).is_err(),);
    }
}
