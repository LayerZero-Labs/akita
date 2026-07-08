use akita_field::AkitaError;

/// Unified singleton/grouped relation matrix column evaluation.
///
/// Canonical dense-table implementation lives in `akita-types`; the prover
/// ring-switch finalize path uses it to feed stage-2 proving.
pub use akita_types::compute_relation_matrix_col_evals;

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
) -> Result<(Vec<i8>, usize, usize), AkitaError> {
    if !w.len().is_multiple_of(d) {
        return Err(AkitaError::InvalidSize {
            expected: d,
            actual: w.len(),
        });
    }
    let live_x_cols = w.len() / d;
    let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
    if extension_degree == 1 {
        let ring_bits = d.trailing_zeros() as usize;
        return Ok((w.to_vec(), col_bits, ring_bits));
    }
    let packed_len = d / extension_degree;
    if packed_len == 0 || !packed_len.is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "packed recursive witness has invalid slot count".to_string(),
        ));
    }
    let half = d / (2 * extension_degree);
    let mut compact = Vec::with_capacity(live_x_cols * packed_len);
    for ring in w.chunks_exact(d) {
        compact.extend_from_slice(&ring[..half]);
        compact.extend((half..packed_len).map(|low| ring[d / 2 + low - half]));
    }
    Ok((compact, col_bits, packed_len.trailing_zeros() as usize))
}
