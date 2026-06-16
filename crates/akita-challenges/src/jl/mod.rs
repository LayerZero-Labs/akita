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

use akita_field::cfg_into_iter;
#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_JL_PROJECTION, CHALLENGE_JL_SEED};
use akita_transcript::Transcript;

use crate::sampler::xof::XofCursor;

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

/// Minimum `n_rows * cols` before the `parallel` feature fans projection out
/// over rows. Below this, rayon scheduling overhead dominates (see
/// `benches/jl_projection.rs`).
const JL_PARALLEL_ELEMS_THRESHOLD: usize = 1 << 16;

/// Map a packed 2-bit pair to its ternary sign: `0b00 -> -1`, `0b11 -> +1`,
/// `0b01`/`0b10 -> 0`.
#[inline]
fn pair_to_sign(pair: u8) -> i8 {
    ((pair == 0b11) as i8) - ((pair == 0b00) as i8)
}

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
    /// JL is only ever applied to small balanced digits, so coefficients and
    /// coordinates are accumulated as `i64`: each coefficient is centered to its
    /// balanced representative in `[-(q-1)/2, (q-1)/2]` and accumulated over the
    /// integers with no modular reduction. Accumulation is checked, so a
    /// coefficient or coordinate that would escape `i64` (i.e. a non-digit input
    /// such as a full-magnitude fp128 element) is rejected rather than wrapped
    /// or saturated, keeping the path panic-free without a wider integer type.
    ///
    /// # Errors
    ///
    /// Returns an error if `coeffs.len() != cols` or if any centered coefficient
    /// or coordinate overflows `i64`.
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

        let q = field_modulus::<F>();
        let half_q = q / 2;
        let centered: Vec<i64> = coeffs
            .iter()
            .map(|c| center_to_i64(c.to_canonical_u128(), q, half_q))
            .collect::<Result<_, _>>()?;

        self.project_centered(&centered)
    }

    /// Project a pre-centered digit coefficient vector to its integer image.
    ///
    /// Callers that already hold balanced `i64` digits (e.g. from a witness
    /// flattening pass) can skip repeated field centering. Row accumulation
    /// parallelizes over `n_rows` when the `parallel` feature is enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if `centered.len() != cols` or if any coordinate
    /// overflows `i64`.
    pub fn project_centered(&self, centered: &[i64]) -> Result<JlImage, AkitaError> {
        if centered.len() != self.cols {
            return Err(AkitaError::InvalidInput(format!(
                "JL projection expects {} centered coefficients, got {}",
                self.cols,
                centered.len()
            )));
        }

        let coords = if use_parallel_projection(self.n_rows, self.cols) {
            cfg_into_iter!(0..self.n_rows)
                .map(|row_idx| project_row(self.row_slice(row_idx), centered, self.cols))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (0..self.n_rows)
                .map(|row_idx| project_row(self.row_slice(row_idx), centered, self.cols))
                .collect::<Result<Vec<_>, _>>()?
        };

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
        Some(pair_to_sign(pair))
    }
}

/// Center a flat coefficient slice to balanced `i64` digits.
///
/// # Errors
///
/// Returns an error if any centered magnitude exceeds `i64` (non-digit input).
pub fn center_coefficients<F: CanonicalField>(coeffs: &[F]) -> Result<Vec<i64>, AkitaError> {
    let q = field_modulus::<F>();
    let half_q = q / 2;
    coeffs
        .iter()
        .map(|c| center_to_i64(c.to_canonical_u128(), q, half_q))
        .collect()
}

/// Recover the field modulus `q` as a `u128` for a base prime field.
#[inline]
fn field_modulus<F: FieldCore + CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

/// Center a canonical residue in `[0, q)` to its balanced representative.
///
/// A JL input is a small balanced digit, so the centered value fits `i64` with
/// room to spare. The conversion is still checked: a non-digit input whose
/// centered magnitude exceeds `i64` (e.g. a full-magnitude fp128 element) is
/// rejected, which both documents the digit contract and keeps projection
/// panic-free without a wider integer type.
fn center_to_i64(canonical: u128, q: u128, half_q: u128) -> Result<i64, AkitaError> {
    let magnitude = if canonical > half_q {
        q - canonical
    } else {
        canonical
    };
    let magnitude = i64::try_from(magnitude).map_err(|_| {
        AkitaError::InvalidInput(
            "JL centered coefficient exceeds i64 range (not a small balanced digit)".to_string(),
        )
    })?;
    Ok(if canonical > half_q {
        -magnitude
    } else {
        magnitude
    })
}

/// Add `sign(pair) * value` into `acc` with checked `i64` arithmetic.
#[inline]
fn accum_pair(acc: i64, pair: u8, value: i64) -> Result<i64, AkitaError> {
    let sign = pair_to_sign(pair);
    if sign == 0 {
        return Ok(acc);
    }
    let term = if sign < 0 {
        value
            .checked_neg()
            .ok_or_else(|| AkitaError::InvalidInput(jl_overflow_msg()))?
    } else {
        value
    };
    acc.checked_add(term)
        .ok_or_else(|| AkitaError::InvalidInput(jl_overflow_msg()))
}

/// Accumulate one projection coordinate `sum_col sign(col) * centered[col]`
/// with checked arithmetic, rejecting `i64` overflow.
///
/// Processes four packed column pairs per byte on the full-byte prefix.
fn project_row(row: &[u8], centered: &[i64], cols: usize) -> Result<i64, AkitaError> {
    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    let mut coeff_idx = 0usize;
    let mut acc: i64 = 0;

    for &byte in row.iter().take(full_bytes) {
        acc = accum_pair(acc, byte & 0b11, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_pair(acc, (byte >> 2) & 0b11, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_pair(acc, (byte >> 4) & 0b11, centered[coeff_idx])?;
        coeff_idx += 1;
        acc = accum_pair(acc, (byte >> 6) & 0b11, centered[coeff_idx])?;
        coeff_idx += 1;
    }

    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let pair = (byte >> (lane << 1)) & 0b11;
            acc = accum_pair(acc, pair, centered[coeff_idx])?;
            coeff_idx += 1;
        }
    }
    debug_assert_eq!(coeff_idx, cols);
    Ok(acc)
}

fn jl_overflow_msg() -> String {
    "JL projection coordinate exceeds i64 range".to_string()
}

#[inline]
fn use_parallel_projection(n_rows: usize, cols: usize) -> bool {
    cfg!(feature = "parallel") && n_rows.saturating_mul(cols) >= JL_PARALLEL_ELEMS_THRESHOLD
}

/// Integer image `p = J · c` of a JL projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JlImage {
    coords: Vec<i64>,
}

impl JlImage {
    /// Signed integer coordinates of the image.
    pub fn coords(&self) -> &[i64] {
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
    /// Coordinates are `i64`, so each square fits `u128`; the `u128` accumulator
    /// is the one place a width past `i64` is warranted (squaring), and it runs
    /// `O(n_rows)`, off the hot projection path. The running sum is still
    /// checked so an out-of-contract image rejects rather than wraps.
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
    /// representative (`|p_j| < q/2`).
    ///
    /// The window check prevents two integers differing by a multiple of `q`
    /// (with different Euclidean norms) from sharing one field residue, which a
    /// later field-consistency check could not distinguish.
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
