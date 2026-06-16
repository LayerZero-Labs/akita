//! Standalone Johnson-Lindenstrauss projection primitives (prototype).
//!
//! This module hosts the dense, field-granular JL projection used by the
//! reveal-tail prototype: the verifier samples a dense ternary matrix
//! `J ∈ {-1, 0, +1}^{n_rows × cols}` from the Fiat-Shamir transcript, the
//! prover projects a centered integer coefficient vector to the integer image
//! `p = J · c`, and the image norm is checked over the integers.
//!
//! The projection acts on the *integer coefficient vector* of base-field
//! elements; ring structure is irrelevant to it, so the public API takes a flat
//! `&[F]` coefficient slice and the caller flattens any ring layout. This keeps
//! `akita-challenges` at its field + transcript dependency layer.
//!
//! Scope: matrix sampling, integer projection, checked Euclidean-norm helpers,
//! and an injective signed-coordinate field embedding. The nonce-regrind
//! completeness loop, the consistency sumcheck, and any protocol wiring are
//! deferred (see `specs/akita-jl-tail-projection-prototype.md`).

#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_JL_PROJECTION, CHALLENGE_JL_SEED};
use akita_transcript::Transcript;

use crate::sampler::xof::XofCursor;

mod kernels;
pub mod mle;

pub use mle::{build_jl_row_weights, eval_jl_mle_at, eval_mle_from_weights};

/// PRG domain separator for the JL matrix stream. Distinct from the
/// sparse-challenge PRG domain so the two streams cannot collide on a shared
/// transcript seed.
const JL_PRG_DOMAIN: &[u8] = b"akita/jl-projection-prg";

/// Version tag bound into the matrix-sampling context buffer. Bumping it
/// changes every sampled matrix, separating proofs across geometry revisions.
const JL_SAMPLE_DOMAIN_VERSION: u64 = 1;

/// Default JL row count used by tests and as the prototype's reference size.
/// The secure `n_rows` derivation is deferred (spec D6); callers parameterize.
pub const DEFAULT_JL_ROWS: usize = 256;

/// Maximum absolute balanced-digit magnitude for JL witness coefficients (`lb ≤ 6`).
pub const MAX_JL_DIGIT: i32 = 32;

/// Minimum `n_rows * cols` before the `parallel` feature fans projection out
/// over rows. Below this, rayon scheduling overhead dominates (see
/// `benches/jl_projection.rs`).
const JL_PARALLEL_ELEMS_THRESHOLD: usize = 1 << 16;

/// Byte length of one packed row of `cols` ternary entries (2 bits each).
fn row_bytes_for(cols: usize) -> Result<usize, AkitaError> {
    if cols == 0 {
        return Err(AkitaError::InvalidInput(
            "JL matrix requires a non-zero column count".to_string(),
        ));
    }
    Ok((cols * 2).div_ceil(8))
}

/// Dense ternary JL projection matrix with entries in `{-1, 0, +1}`, packed two
/// bits per entry in a single contiguous row-major buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JlProjectionMatrix {
    n_rows: usize,
    cols: usize,
    row_bytes: usize,
    packed_rows: Vec<u8>,
}

impl JlProjectionMatrix {
    /// Number of projection rows (the image dimension).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns (the projected coefficient-vector length).
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    fn row_slice(&self, row_idx: usize) -> &[u8] {
        let start = row_idx * self.row_bytes;
        &self.packed_rows[start..start + self.row_bytes]
    }

    /// Packed row bytes for MLE kernels (`pub(crate)` for `jl::mle`).
    #[inline]
    pub(crate) fn row_bytes_slice(&self, row_idx: usize) -> &[u8] {
        self.row_slice(row_idx)
    }

    /// Sample a dense ternary matrix deterministically from the transcript.
    ///
    /// Absorbs a context buffer (`n_rows`, `cols`, domain version), draws a
    /// 32-byte seed, and expands `n_rows` packed rows from a single
    /// JL-domain-separated XOF stream. Prover and verifier in the same
    /// transcript state reconstruct an identical matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_rows` or `cols` is zero.
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
        let mut packed_rows = vec![0u8; n_rows * row_bytes];
        for row_idx in 0..n_rows {
            let start = row_idx * row_bytes;
            cursor.fill_bytes(&mut packed_rows[start..start + row_bytes]);
        }

        Ok(Self {
            n_rows,
            cols,
            row_bytes,
            packed_rows,
        })
    }

    /// Project a flat coefficient slice to its exact integer image `J · c`.
    ///
    /// Coefficients are centered to balanced `i32` digits and projected through
    /// the fast kernel. Any input whose centered magnitude exceeds the balanced
    /// digit bound [`MAX_JL_DIGIT`] (e.g. a full-magnitude fp128 element) is
    /// rejected at the boundary as a non-digit witness.
    ///
    /// # Errors
    ///
    /// Returns an error if `coeffs.len() != cols`, if any centered magnitude
    /// exceeds `i32`, or if any centered digit exceeds [`MAX_JL_DIGIT`].
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

    /// Project a pre-centered balanced-digit vector with the fast `i32` kernel.
    ///
    /// Column-panel accumulation parallelizes over panels when the `parallel`
    /// feature is enabled and `n_rows * cols` exceeds the internal threshold.
    /// Runtime dispatch selects NEON (aarch64), AVX-512 (x86_64 when F/DQ/BW
    /// are present), AVX2, or the scalar fast kernel.
    ///
    /// # Errors
    ///
    /// Returns an error on a shape mismatch or if any digit exceeds
    /// [`MAX_JL_DIGIT`].
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

    /// Checked `i64` reference projection for tests and differential benches.
    ///
    /// # Errors
    ///
    /// Returns an error on shape mismatch, digit-bound violation, or `i64`
    /// overflow during accumulation.
    #[doc(hidden)]
    pub fn project_digits_reference(&self, digits: &[i32]) -> Result<JlImage, AkitaError> {
        validate_digit_witness(digits, self.cols)?;
        let centered: Vec<i64> = digits.iter().map(|&d| i64::from(d)).collect();
        let project_row = |row_idx: usize| {
            kernels::project_row_reference(self.row_slice(row_idx), &centered, self.cols)
        };
        let coords = if use_parallel_projection(self.n_rows, self.cols) {
            akita_field::cfg_into_iter!(0..self.n_rows)
                .map(project_row)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..self.n_rows)
                .map(project_row)
                .collect::<Result<Vec<_>, _>>()?
        };
        let coords: Vec<i32> = coords
            .into_iter()
            .map(|c| {
                i32::try_from(c).map_err(|_| {
                    AkitaError::InvalidInput(
                        "JL reference coordinate exceeds i32 range for digit witness".to_string(),
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(JlImage { coords })
    }

    /// Reconstruct a matrix from explicit sign rows. Test-only constructor for
    /// projection-vs-reference and packing round-trip checks.
    #[cfg(test)]
    fn from_sign_rows(signs: &[Vec<i8>]) -> Result<Self, AkitaError> {
        let n_rows = signs.len();
        if n_rows == 0 {
            return Err(AkitaError::InvalidInput(
                "JL matrix requires a non-zero row count".to_string(),
            ));
        }
        let cols = signs[0].len();
        let row_bytes = row_bytes_for(cols)?;
        if signs.iter().any(|row| row.len() != cols) {
            return Err(AkitaError::InvalidInput(
                "JL matrix row length mismatch".to_string(),
            ));
        }

        let mut packed_rows = vec![0u8; n_rows * row_bytes];
        for (row_idx, row) in signs.iter().enumerate() {
            let row_start = row_idx * row_bytes;
            for (col_idx, &sign) in row.iter().enumerate() {
                let pair: u8 = match sign {
                    -1 => 0b00,
                    0 => 0b01,
                    1 => 0b11,
                    _ => {
                        return Err(AkitaError::InvalidInput(
                            "JL matrix entries must be in {-1, 0, +1}".to_string(),
                        ))
                    }
                };
                packed_rows[row_start + (col_idx >> 2)] |= pair << ((col_idx & 0b11) << 1);
            }
        }

        Ok(Self {
            n_rows,
            cols,
            row_bytes,
            packed_rows,
        })
    }

    /// Ternary sign at `(row_idx, col_idx)`. Test-only accessor.
    #[cfg(test)]
    fn sign_at(&self, row_idx: usize, col_idx: usize) -> Option<i8> {
        if row_idx >= self.n_rows || col_idx >= self.cols {
            return None;
        }
        let shift = (col_idx & 0b11) << 1;
        let pair = (self.row_slice(row_idx)[col_idx >> 2] >> shift) & 0b11;
        Some(kernels::pair_to_sign(pair))
    }
}

/// Center a flat coefficient slice to balanced `i32` digits.
///
/// # Errors
///
/// Returns an error if any centered magnitude exceeds `i32` (non-digit input).
pub fn center_coefficients<F: CanonicalField>(coeffs: &[F]) -> Result<Vec<i32>, AkitaError> {
    let q = field_modulus::<F>();
    let half_q = q / 2;
    coeffs
        .iter()
        .map(|c| center_to_i32(c.to_canonical_u128(), q, half_q))
        .collect()
}

fn validate_digit_witness(digits: &[i32], cols: usize) -> Result<(), AkitaError> {
    if digits.len() != cols {
        return Err(AkitaError::InvalidInput(format!(
            "JL projection expects {cols} centered coefficients, got {}",
            digits.len()
        )));
    }
    if digits.iter().any(|&d| d.abs() > MAX_JL_DIGIT) {
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

/// Recover the field modulus `q` as a `u128` for a base prime field.
#[inline]
fn field_modulus<F: FieldCore + CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

/// Center a canonical residue in `[0, q)` to its balanced representative.
fn center_to_i32(canonical: u128, q: u128, half_q: u128) -> Result<i32, AkitaError> {
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
fn use_parallel_projection(n_rows: usize, cols: usize) -> bool {
    cfg!(feature = "parallel") && n_rows.saturating_mul(cols) >= JL_PARALLEL_ELEMS_THRESHOLD
}

/// Integer image `p = J · c` of a JL projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JlImage {
    coords: Vec<i32>,
}

impl JlImage {
    /// Signed integer coordinates of the image.
    pub fn coords(&self) -> &[i32] {
        &self.coords
    }

    /// Number of image coordinates (the matrix row count).
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Whether the image has no coordinates.
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Squared Euclidean norm over the integers, with checked accumulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the running sum exceeds `u128`.
    pub fn l2_norm_sq_checked(&self) -> Result<u128, AkitaError> {
        let mut acc: u128 = 0;
        for &c in &self.coords {
            let mag = u128::from(c.unsigned_abs());
            let sq = mag * mag;
            acc = acc.checked_add(sq).ok_or_else(|| {
                AkitaError::InvalidInput("JL image squared norm exceeds u128".to_string())
            })?;
        }
        Ok(acc)
    }

    /// Accept the image iff `||p||_2^2 <= bound_sq` over the integers.
    ///
    /// # Errors
    ///
    /// Returns an error if the norm overflows or exceeds `bound_sq`.
    pub fn check_l2(&self, bound_sq: u128) -> Result<(), AkitaError> {
        let norm_sq = self.l2_norm_sq_checked()?;
        if norm_sq > bound_sq {
            return Err(AkitaError::InvalidInput(format!(
                "JL image squared L2 norm {norm_sq} exceeds bound {bound_sq}"
            )));
        }
        Ok(())
    }

    /// Embed each coordinate into the base field, requiring an injective signed
    /// representative (`|p_j| < q/2`). Rejects coordinates that would alias
    /// modulo `q` with a different Euclidean norm.
    ///
    /// # Errors
    ///
    /// Returns an error if any coordinate falls outside the injective window.
    pub fn embed_into_field<F>(&self) -> Result<Vec<F>, AkitaError>
    where
        F: FieldCore + CanonicalField,
    {
        let q = field_modulus::<F>();
        let half_q = q / 2;
        self.coords
            .iter()
            .map(|&c| {
                let mag = u128::from(c.unsigned_abs());
                if mag > half_q {
                    return Err(AkitaError::InvalidInput(format!(
                        "JL image coordinate {c} outside injective signed window (|c| <= {half_q})"
                    )));
                }
                let elem = F::from_canonical_u128_reduced(mag);
                Ok(if c < 0 { -elem } else { elem })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
