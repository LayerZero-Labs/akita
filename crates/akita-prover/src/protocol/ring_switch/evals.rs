use akita_field::AkitaError;

/// Unified singleton/multi-group relation matrix column evaluation.
///
/// Canonical dense-table implementation lives in `akita-types`; the prover
/// ring-switch finalize path uses it to feed stage-2 proving.
pub use akita_types::compute_relation_weight_evals;

/// Produce the compact `Vec<i8>` eval table of `w` for the fused prover.
///
/// The compact witness stays in the raw `build_w_coeffs` order:
/// `w[x * y_len + y]`, with x outer and y inner.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by the ring
/// dimension.
pub fn build_w_evals_compact(
    w: &[i8],
    d: usize,
    extension_degree: usize,
    opening_source_len: usize,
) -> Result<(Vec<i8>, usize, usize), AkitaError> {
    if !w.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let live_x_cols = w.len() / d;
    if live_x_cols > opening_source_len {
        return Err(AkitaError::InvalidSize {
            expected: opening_source_len,
            actual: live_x_cols,
        });
    }
    let opening_x_cols = akita_types::opening_domain_len(opening_source_len)?;
    let col_bits = opening_x_cols.trailing_zeros() as usize;
    if extension_degree == 1 {
        let ring_bits = d.trailing_zeros() as usize;
        let mut compact = vec![0i8; opening_x_cols * d];
        for (physical_index, ring) in w.chunks_exact(d).enumerate() {
            let opening_index =
                akita_types::checked_opening_source_index(opening_source_len, physical_index)?;
            compact[opening_index * d..(opening_index + 1) * d].copy_from_slice(ring);
        }
        return Ok((compact, col_bits, ring_bits));
    }
    let packed_len = d / extension_degree;
    if packed_len == 0 || !packed_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "packed recursive witness has invalid slot count".to_string(),
        ));
    }
    let half = d / (2 * extension_degree);
    let mut compact = vec![0i8; opening_x_cols * packed_len];
    for (physical_index, ring) in w.chunks_exact(d).enumerate() {
        let opening_index =
            akita_types::checked_opening_source_index(opening_source_len, physical_index)?;
        let dst = &mut compact[opening_index * packed_len..(opening_index + 1) * packed_len];
        dst[..half].copy_from_slice(&ring[..half]);
        for (slot, low) in (half..packed_len).enumerate() {
            dst[half + slot] = ring[d / 2 + low - half];
        }
    }
    Ok((compact, col_bits, packed_len.trailing_zeros() as usize))
}
