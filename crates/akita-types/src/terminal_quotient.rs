//! L2 terminal quotient certificate (experimental, `l2-terminal-quotient`).
//!
//! The terminal A rows assert `sum_j A_ij z_j = H_i` in `F[X]/(X^D + 1)`.
//! For the unreduced product `P_i = sum_j A_ij z_j` (degree at most
//! `2D - 2`), write `P_i = L_i + X^D U_i` with `deg L_i < D`. Then
//! `P_i mod (X^D + 1) = L_i - U_i`, so a true row satisfies the polynomial
//! identity `P_i - H_i = (X^D + 1) U_i` with `deg U_i <= D - 2`. The prover
//! sends every `U_i` as its quotient, and the verifier checks the identity at
//! fresh points drawn after the quotient is absorbed.

use akita_error::AkitaError;
use jolt_field::{Field, Ring};

/// Quotient coefficients per terminal A row: `D - 1`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `ring_dimension < 2`, where the
/// quotient would be empty.
pub fn terminal_quotient_len(ring_dimension: usize) -> Result<usize, AkitaError> {
    ring_dimension
        .checked_sub(1)
        .filter(|&len| len > 0)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("terminal quotient needs ring dimension at least 2".into())
        })
}

/// The honest quotients `U_i`, row-major and low degree first, for the
/// terminal rows `A_ij` (ring `i * n_cols + j` of `setup`, `D` coefficients
/// each) applied to the centered response `z` (`n_cols` rings of `D`
/// coefficients each).
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the setup is shorter than the
/// terminal A prefix or the ring dimension is below 2, and
/// [`AkitaError::InvalidProof`] when `z` is not a whole number of rings.
pub fn terminal_a_row_quotients<F>(
    setup: &[F],
    z: &[i16],
    ring_dimension: usize,
    num_rows: usize,
) -> Result<Vec<F>, AkitaError>
where
    F: Field + Ring,
{
    let d = ring_dimension;
    let quotient_len = terminal_quotient_len(d)?;
    if !z.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidProof);
    }
    let num_cols = z.len() / d;
    let prefix = num_rows
        .checked_mul(num_cols)
        .and_then(|rings| rings.checked_mul(d))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal A prefix overflow".into()))?;
    let setup = setup.get(..prefix).ok_or_else(|| {
        AkitaError::InvalidSetup("setup is shorter than the terminal A prefix".into())
    })?;
    let z_fields: Vec<F> = z
        .iter()
        .map(|&value| F::from_i64(i64::from(value)))
        .collect();
    let total = num_rows
        .checked_mul(quotient_len)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal quotient length overflow".into()))?;
    let mut quotients = vec![F::zero(); total];
    if num_cols == 0 {
        return Ok(quotients);
    }
    for (row, quotient) in setup
        .chunks_exact(num_cols * d)
        .zip(quotients.chunks_exact_mut(quotient_len))
    {
        for (a, zz) in row.chunks_exact(d).zip(z_fields.chunks_exact(d)) {
            for (a_index, a_value) in a.iter().enumerate() {
                // X^a_index * X^z_index lands at degree a_index + z_index;
                // the high half (degree >= D) is U at a_index + z_index - D.
                for (z_index, z_value) in zz.iter().enumerate().skip(d - a_index) {
                    if let Some(slot) = quotient.get_mut(a_index + z_index - d) {
                        *slot += *a_value * *z_value;
                    }
                }
            }
        }
    }
    Ok(quotients)
}
