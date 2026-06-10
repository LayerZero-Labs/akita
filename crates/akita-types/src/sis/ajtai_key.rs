//! Ajtai-commitment key sizing: the SIS modulus family, the `AjtaiKeyParams`
//! type, the secure-rank lookup, and collision-bucket rounding.
//!
//! This is the single home for "given a width and a rounded-up collision
//! bound, what is the minimum SIS-secure module rank, and what audited
//! `AjtaiKeyParams` does it yield". The generated SIS-floor tables it consults
//! live in the private sibling `super::generated_sis_table`.

use akita_field::AkitaError;

use super::generated_sis_table::sis_max_widths;
use crate::descriptor_bytes::{push_u32, push_usize, sis_family_tag};

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

/// Minimum generated SIS-secure module rank that supports `width` ring columns
/// at an **already rounded-up** collision bucket `collision_inf_rounded_up`
/// (see [`ceil_supported_collision`]).
///
/// Returns `None` when no generated SIS-floor row covers the configuration.
pub fn min_secure_rank(
    sis_family: SisModulusFamily,
    d: u32,
    collision_inf_rounded_up: u32,
    width: u64,
) -> Option<usize> {
    let widths = sis_max_widths(sis_family, d, collision_inf_rounded_up)?;
    for (i, &max_w) in widths.iter().enumerate() {
        if width <= max_w {
            return Some(i + 1);
        }
    }
    None
}

/// Round a raw collision infinity-norm up to the next generated collision
/// bucket for `(sis_family, d)`. Returns `None` for an unsupported
/// `(sis_family, d)` or a collision above the largest tabulated bucket.
pub fn ceil_supported_collision(
    sis_family: SisModulusFamily,
    d: u32,
    collision_inf: u32,
) -> Option<u32> {
    const BUCKETS: &[u32] = &[
        2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071,
        262143, 524287, 1048575, 2097151, 4194303, 8388607, 16777215, 33554431, 67108863,
    ];
    let supported = matches!(
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
    );
    if !supported {
        return None;
    }
    BUCKETS
        .iter()
        .copied()
        .find(|&bucket| collision_inf <= bucket)
}

/// Parameters for a single Ajtai commitment matrix.
///
/// Each matrix in the protocol (A, B, D) is characterised by its row count
/// (security rank), column count (message width), and the worst-case L∞
/// collision bound used for SIS security sizing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AjtaiKeyParams {
    pub(crate) row_len: usize,
    pub(crate) col_len: usize,
    pub(crate) collision_inf: u32,
    pub(crate) sis_family: SisModulusFamily,
}

impl AjtaiKeyParams {
    /// Create a new SIS-secure `AjtaiKeyParams`, auditing the
    /// `(row_len, col_len, collision_inf)` triple against the generated 128-bit
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
    /// Returns an error if any of `row_len`, `col_len`, or `collision_inf` is
    /// zero, if the SIS-floor tables do not cover the configuration, or if
    /// `row_len` is below the audited floor.
    pub fn try_new(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
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
        if collision_inf == 0 {
            return Err(AkitaError::InvalidSetup(
                "AjtaiKeyParams: collision_inf = 0".to_string(),
            ));
        }
        let floor = min_secure_rank(
            sis_family,
            ring_dimension as u32,
            collision_inf,
            col_len as u64,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: no audited SIS rank for \
                     family={sis_family:?} d={ring_dimension} \
                     collision_inf={collision_inf} col_len={col_len}"
            ))
        })?;
        if row_len < floor {
            return Err(AkitaError::InvalidSetup(format!(
                "AjtaiKeyParams: row_len {row_len} < SIS floor {floor} \
                 (family={sis_family:?}, d={ring_dimension}, \
                 collision_inf={collision_inf}, col_len={col_len})"
            )));
        }
        Ok(Self {
            row_len,
            col_len,
            collision_inf,
            sis_family,
        })
    }

    /// Create a new `AjtaiKeyParams` without enforcing SIS security.
    ///
    /// Use this only for intermediate construction steps that carry
    /// incomplete data (`params_only` placeholders with `col_len = 0` or
    /// `collision_inf = 0`, iterative SIS fixed-point loops, etc.) and for
    /// synthetic test/descriptor/proof-size layouts that intentionally carry
    /// degenerate ranks. Production-facing schedule layouts are built through
    /// [`try_new`](Self::try_new), which audits the SIS floor against the final
    /// width as the key is constructed.
    pub fn new_unchecked(
        sis_family: SisModulusFamily,
        row_len: usize,
        col_len: usize,
        collision_inf: u32,
        ring_dimension: usize,
    ) -> Self {
        let _ = ring_dimension;
        Self {
            row_len,
            col_len,
            collision_inf,
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

    /// Worst-case L∞ collision bound for SIS security sizing.
    #[inline]
    pub fn collision_inf(&self) -> u32 {
        self.collision_inf
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
        push_u32(bytes, self.collision_inf());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshaped_floor_slices_preserve_legacy_values() {
        let cases: &[(SisModulusFamily, u32, u32, &[u64])] = &[
            (SisModulusFamily::Q32, 32, 7, &[6, 15, 140, 959]),
            (SisModulusFamily::Q32, 64, 7, &[7, 479, 11_966, 177_951]),
            (
                SisModulusFamily::Q32,
                128,
                7,
                &[239, 88_975, 8_446_449, 396_975_954],
            ),
            (
                SisModulusFamily::Q32,
                256,
                7,
                &[44_487, 198_487_977, 10_000_000_000, 10_000_000_000],
            ),
            (SisModulusFamily::Q64, 32, 7, &[15, 959, 23_932, 355_903]),
            (
                SisModulusFamily::Q64,
                64,
                7,
                &[479, 177_951, 16_892_899, 793_951_920],
            ),
            (
                SisModulusFamily::Q64,
                128,
                7,
                &[88_975, 396_975_960, 50_000_000_000, 50_000_000_000],
            ),
            (
                SisModulusFamily::Q64,
                256,
                7,
                &[198_487_980, 10_000_000_000, 10_000_000_000, 10_000_000_000],
            ),
            (
                SisModulusFamily::Q128,
                32,
                7,
                &[959, 355_903, 33_785_799, 1_587_903_841],
            ),
            (
                SisModulusFamily::Q128,
                64,
                7,
                &[177_951, 793_951_920, 10_000_000_000, 10_000_000_000],
            ),
            (
                SisModulusFamily::Q128,
                128,
                7,
                &[396_975_960, 50_000_000_000, 50_000_000_000, 50_000_000_000],
            ),
            (
                SisModulusFamily::Q128,
                256,
                7,
                &[
                    10_000_000_000,
                    10_000_000_000,
                    10_000_000_000,
                    10_000_000_000,
                ],
            ),
        ];

        for &(family, d, collision, expected) in cases {
            let actual = sis_max_widths(family, d, collision).expect("SIS row should exist");
            assert_eq!(&actual[..expected.len()], expected);
        }
    }

    #[test]
    fn floor_slices_have_family_specific_rank_caps() {
        assert_eq!(
            sis_max_widths(SisModulusFamily::Q32, 32, 7).map(<[u64]>::len),
            Some(20)
        );
        assert_eq!(min_secure_rank(SisModulusFamily::Q32, 32, 127, 16), Some(5));
    }
}
