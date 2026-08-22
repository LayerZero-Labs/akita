use akita_error::AkitaError;

use crate::backend::packed_digits::PackedSignedDigits;

/// Produce the compact packed eval table of `w` for the fused prover.
///
/// The compact witness stays in the raw `build_w_coeffs` order:
/// `w[x * y_len + y]`, with x outer and y inner. Only exact live columns are
/// returned; the Boolean-domain suffix is represented implicitly by the
/// Stage 1 and Stage 2 prefix kernels.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by the ring
/// dimension.
pub(crate) fn build_w_evals_compact(
    w: impl Into<PackedSignedDigits>,
    d: usize,
    extension_degree: usize,
    opening_source_len: usize,
) -> Result<(PackedSignedDigits, usize, usize), AkitaError> {
    let w = w.into();
    if d == 0 || !w.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let live_physical_cols = w.len() / d;
    if live_physical_cols > opening_source_len {
        return Err(AkitaError::InvalidSize {
            expected: opening_source_len,
            actual: live_physical_cols,
        });
    }
    let opening_x_cols = akita_types::opening_domain_len(opening_source_len)?;
    let col_bits = opening_x_cols.trailing_zeros() as usize;
    if extension_degree == 1 {
        let ring_bits = d.trailing_zeros() as usize;
        return Ok((w, col_bits, ring_bits));
    }
    let packed_len = d / extension_degree;
    if packed_len == 0 || !packed_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "packed recursive witness has invalid slot count".to_string(),
        ));
    }
    let half = d / (2 * extension_degree);
    let mut compact = vec![0i8; live_physical_cols * packed_len];
    let mut ring = vec![0i8; d];
    for physical_index in 0..live_physical_cols {
        w.view()
            .slice(physical_index * d..(physical_index + 1) * d)?
            .decode_range(0, &mut ring)?;
        let opening_index =
            akita_types::checked_opening_source_index(opening_source_len, physical_index)?;
        let dst = &mut compact[opening_index * packed_len..(opening_index + 1) * packed_len];
        dst[..half].copy_from_slice(&ring[..half]);
        for (slot, low) in (half..packed_len).enumerate() {
            dst[half + slot] = ring[d / 2 + low - half];
        }
    }
    Ok((
        PackedSignedDigits::from_i8_digits_auto(compact),
        col_bits,
        packed_len.trailing_zeros() as usize,
    ))
}
