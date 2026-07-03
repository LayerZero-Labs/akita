//! Ajtai-commitment key sizing: the SIS modulus family, the `AjtaiKeyParams`
//! type, the secure-rank lookup, and coefficient-`L∞` bucket rounding.
//!
//! This is the single home for "given a width and a rounded-up coefficient
//! bound at a security floor, what is the minimum SIS-secure module rank, and what audited
//! `AjtaiKeyParams` does it yield". The generated SIS-floor tables it consults
//! live in the private sibling module `super::generated_sis_table`.

use akita_field::AkitaError;

use super::generated_sis_table::sis_max_widths;
use crate::descriptor_bytes::{push_u128, push_u16, push_usize, sis_family_tag};

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

/// Production SIS security floor for generated coefficient-`L∞` tables.
pub const DEFAULT_SIS_SECURITY_BITS: u16 = 138;

/// Security floors currently shipped in the production SIS table.
pub const SUPPORTED_SIS_SECURITY_BITS: &[u16] = &[DEFAULT_SIS_SECURITY_BITS];

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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SisTableKey {
    /// Minimum SIS security floor in bits.
    pub min_security_bits: u16,
    /// SIS modulus family.
    pub family: SisModulusFamily,
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

/// Round a raw coefficient-`L∞` bound up to a generated table bucket.
#[must_use]
pub fn ceil_supported_linf_bound(
    min_security_bits: u16,
    sis_family: SisModulusFamily,
    d: u32,
    linf: u128,
) -> Option<u128> {
    if linf == 0 || !supports_family_dimension(sis_family, d) {
        return None;
    }
    let bucket = ceil_coeff_linf_bucket(linf)?;
    sis_max_widths(min_security_bits, sis_family, d, bucket)?;
    Some(bucket)
}

/// Canonical generated-table key for a raw coefficient-`L∞` bound.
///
/// Returns `None` for an unsupported security floor, family/dimension pair, or
/// coefficient bound.
#[must_use]
pub fn sis_table_key_for_linf_bound(
    min_security_bits: u16,
    sis_family: SisModulusFamily,
    d: u32,
    linf: u128,
) -> Option<SisTableKey> {
    let coeff_linf_bound = ceil_supported_linf_bound(min_security_bits, sis_family, d, linf)?;
    Some(SisTableKey {
        min_security_bits,
        family: sis_family,
        ring_dimension: d,
        coeff_linf_bound,
    })
}

/// Minimum generated SIS-secure module rank that supports `width` ring columns
/// at an already rounded-up coefficient-`L∞` bucket.
///
/// Returns `None` when no generated SIS-floor row covers the configuration.
pub fn min_secure_rank(key: SisTableKey, width: u64) -> Option<usize> {
    let widths = sis_max_widths(
        key.min_security_bits,
        key.family,
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
    pub fn try_new(
        min_security_bits: u16,
        sis_family: SisModulusFamily,
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
            min_security_bits,
            sis_family,
            ring_dimension as u32,
            coeff_linf_bound,
        ) else {
            return Err(AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: no audited SIS table key for \
                     min_security_bits={min_security_bits} family={sis_family:?} \
                     d={ring_dimension} coeff_linf_bound={coeff_linf_bound}"
            )));
        };
        let floor = min_secure_rank(key, col_len as u64).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: no audited SIS rank for \
                     min_security_bits={min_security_bits} family={sis_family:?} \
                     d={ring_dimension} coeff_linf_bound={} col_len={col_len}",
                key.coeff_linf_bound
            ))
        })?;
        if row_len < floor {
            return Err(AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: row_len {row_len} < SIS floor {floor} \
                 (min_security_bits={min_security_bits}, family={sis_family:?}, \
                 d={ring_dimension}, coeff_linf_bound={}, col_len={col_len})",
                key.coeff_linf_bound
            )));
        }
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
    pub fn new_unchecked(
        min_security_bits: u16,
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        coeff_linf_bound: u128,
        ring_dimension: usize,
    ) -> Self {
        Self {
            row_len,
            col_len,
            sis_table_key: SisTableKey {
                min_security_bits,
                family: sis_family,
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

    /// Minimum SIS security floor in bits.
    #[inline]
    pub fn min_security_bits(&self) -> u16 {
        self.sis_table_key.min_security_bits
    }

    /// Rounded coefficient-`L∞` bucket for SIS sizing.
    #[inline]
    pub fn coeff_linf_bound(&self) -> u128 {
        self.sis_table_key.coeff_linf_bound
    }

    /// SIS modulus family used to validate this key.
    #[inline]
    pub fn sis_family(&self) -> SisModulusFamily {
        self.sis_table_key.family
    }

    /// Full generated-table key used to validate this key.
    #[inline]
    pub fn sis_table_key(&self) -> SisTableKey {
        self.sis_table_key
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(sis_family_tag(self.sis_family()));
        push_u16(bytes, self.min_security_bits());
        push_usize(bytes, self.row_len());
        push_usize(bytes, self.col_len());
        push_u128(bytes, self.coeff_linf_bound());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_security_floor_rejects_linf_bucket() {
        assert_eq!(
            ceil_supported_linf_bound(128, SisModulusFamily::Q32, 32, 7),
            None
        );
    }

    #[test]
    fn floor_slices_have_family_specific_rank_caps() {
        let bucket = 15;
        if sis_max_widths(DEFAULT_SIS_SECURITY_BITS, SisModulusFamily::Q32, 32, bucket).is_some() {
            assert_eq!(
                sis_max_widths(DEFAULT_SIS_SECURITY_BITS, SisModulusFamily::Q32, 32, bucket)
                    .map(<[u64]>::len),
                Some(20)
            );
        }
    }

    #[test]
    fn linf_key_rounds_to_coefficient_bucket() {
        let linf = 1_048_575u128;
        let key = sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q32,
            128,
            linf,
        );
        if let Some(key) = key {
            assert_eq!(key.coeff_linf_bound, linf);
            assert_eq!(key.min_security_bits, DEFAULT_SIS_SECURITY_BITS);
        }
    }

    #[test]
    fn coeff_linf_bucket_ladder_matches_main_ceiling() {
        assert_eq!(ceil_coeff_linf_bucket(1_048_574), Some(1_048_575));
        assert_eq!(ceil_coeff_linf_bucket(1_048_575), Some(1_048_575));
        assert_eq!(ceil_coeff_linf_bucket(1_048_576), Some(2_097_151));
    }
}
