//! Standalone Johnson-Lindenstrauss projection primitives (prototype).

use akita_field::{field_modulus, AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_JL_PROJECTION, CHALLENGE_JL_SEED};
use akita_transcript::Transcript;

use crate::sampler::xof::XofCursor;

mod hooks;
mod kernels;
pub mod mle;
mod packed_byte;
mod panel;
#[cfg(any(test, feature = "jl-test-fixtures"))]
pub(crate) mod testutil;

const JL_PRG_DOMAIN: &[u8] = b"akita/jl-projection-prg";
const JL_SAMPLE_DOMAIN_VERSION: u64 = 1;

/// Default JL row count used by tests and as the prototype's reference size.
pub const DEFAULT_JL_ROWS: usize = 256;

/// Maximum absolute balanced-digit magnitude for JL witness coefficients (`lb ≤ 6`).
pub const MAX_JL_DIGIT: i32 = 32;

/// Byte length of one packed row of `cols` binary-sign entries (1 bit each).
pub(crate) fn row_bytes_for(cols: usize) -> Result<usize, AkitaError> {
    if cols == 0 {
        return Err(AkitaError::InvalidInput(
            "JL matrix requires a non-zero column count".to_string(),
        ));
    }
    Ok(cols.div_ceil(8))
}

#[inline]
pub(crate) fn jl_geometry_overflow() -> AkitaError {
    AkitaError::InvalidInput("JL matrix dimensions overflow".to_string())
}

#[inline]
fn jl_digit_within_bound(d: i32) -> bool {
    (-MAX_JL_DIGIT..=MAX_JL_DIGIT).contains(&d)
}

/// Dense binary-sign JL projection matrix with entries in `{-1, +1}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JlProjectionMatrix {
    n_rows: usize,
    cols: usize,
    row_bytes: usize,
    packed_rows: Vec<u8>,
}

impl JlProjectionMatrix {
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub(crate) fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    #[inline]
    pub(crate) fn packed_rows(&self) -> &[u8] {
        &self.packed_rows
    }

    #[inline]
    pub(crate) fn row_slice(&self, row_idx: usize) -> &[u8] {
        let start = row_idx * self.row_bytes;
        &self.packed_rows[start..start + self.row_bytes]
    }

    pub fn sample<F, T>(transcript: &mut T, n_rows: usize, cols: usize) -> Result<Self, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
    {
        if n_rows == 0 {
            return Err(AkitaError::InvalidInput(
                "JL matrix requires a non-zero row count".to_string(),
            ));
        }
        let row_bytes = row_bytes_for(cols)?;

        let mut absorb = Vec::with_capacity(24);
        absorb.extend_from_slice(&(n_rows as u64).to_le_bytes());
        absorb.extend_from_slice(&(cols as u64).to_le_bytes());
        absorb.extend_from_slice(&JL_SAMPLE_DOMAIN_VERSION.to_le_bytes());
        transcript.append_bytes(ABSORB_JL_PROJECTION, &absorb);
        let seed = transcript.challenge_bytes(CHALLENGE_JL_SEED, 32);

        let mut cursor = XofCursor::from_seed_with_domain(JL_PRG_DOMAIN, &seed);
        let packed_len = n_rows
            .checked_mul(row_bytes)
            .ok_or_else(jl_geometry_overflow)?;
        if packed_len > isize::MAX as usize {
            return Err(jl_geometry_overflow());
        }
        let mut packed_rows = vec![0u8; packed_len];
        for row_idx in 0..n_rows {
            let start = row_idx
                .checked_mul(row_bytes)
                .ok_or_else(jl_geometry_overflow)?;
            cursor.fill_bytes(&mut packed_rows[start..start + row_bytes]);
        }

        Ok(Self {
            n_rows,
            cols,
            row_bytes,
            packed_rows,
        })
    }

    pub fn project<F>(&self, coeffs: &[F]) -> Result<JlImage, AkitaError>
    where
        F: FieldCore + CanonicalField,
    {
        if coeffs.len() != self.cols {
            return Err(AkitaError::InvalidInput(format!(
                "JL projection expects {} coefficients, got {}",
                self.cols,
                coeffs.len()
            )));
        }

        let digits = center_coefficients(coeffs)?;
        self.project_digits(&digits)
    }

    pub fn project_digits(&self, digits: &[i32]) -> Result<JlImage, AkitaError> {
        validate_digit_witness(digits, self.cols)?;
        let coords = kernels::project_rows_fast(
            self.n_rows,
            self.row_bytes,
            &self.packed_rows,
            digits,
            self.cols,
            use_parallel_projection(self.n_rows, self.cols),
        );
        Ok(JlImage { coords })
    }
}

/// Center a flat coefficient slice to balanced `i32` digits.
pub fn center_coefficients<F: CanonicalField>(coeffs: &[F]) -> Result<Vec<i32>, AkitaError> {
    let q = field_modulus::<F>();
    let half_q = q / 2;
    coeffs
        .iter()
        .map(|c| center_to_i32(c.to_canonical_u128(), q, half_q))
        .collect()
}

pub(crate) fn validate_digit_witness(digits: &[i32], cols: usize) -> Result<(), AkitaError> {
    if digits.len() != cols {
        return Err(AkitaError::InvalidInput(format!(
            "JL projection expects {cols} centered coefficients, got {}",
            digits.len()
        )));
    }
    if digits.iter().any(|&d| !jl_digit_within_bound(d)) {
        return Err(AkitaError::InvalidInput(format!(
            "JL witness digit exceeds balanced bound |d| <= {MAX_JL_DIGIT}"
        )));
    }
    if cols > i32::MAX as usize / MAX_JL_DIGIT as usize {
        return Err(AkitaError::InvalidInput(
            "JL column count too large for unchecked i32 row accumulation".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn center_to_i32(canonical: u128, q: u128, half_q: u128) -> Result<i32, AkitaError> {
    let magnitude = if canonical > half_q {
        q - canonical
    } else {
        canonical
    };
    let magnitude = i32::try_from(magnitude).map_err(|_| {
        AkitaError::InvalidInput(
            "JL centered coefficient exceeds i32 range (not a small balanced digit)".to_string(),
        )
    })?;
    Ok(if canonical > half_q {
        -magnitude
    } else {
        magnitude
    })
}

#[inline]
pub(crate) fn use_parallel_projection(n_rows: usize, cols: usize) -> bool {
    panel::parallel_jl_enabled(n_rows, cols)
}

/// Integer image `p = J · c` of a JL projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JlImage {
    coords: Vec<i32>,
}

impl JlImage {
    pub fn coords(&self) -> &[i32] {
        &self.coords
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

pub use hooks::{project_digits_reference, project_digits_scalar};

#[cfg(test)]
mod tests;
