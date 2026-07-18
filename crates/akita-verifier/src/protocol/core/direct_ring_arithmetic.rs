//! Checked plain ring arithmetic shared by direct verifier paths.
//!
//! These kernels deliberately operate on validated setup matrix views. They
//! are verifier soundness code: callers own protocol layout and shape checks;
//! this module owns only the canonical arithmetic over those checked slices.

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};

pub(super) fn decompose_rows_i8<F, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Result<Vec<[i8; D]>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if !(1..=6).contains(&log_basis)
        || num_digits
            .checked_mul(log_basis as usize)
            .is_none_or(|bits| bits > 128 + log_basis as usize)
    {
        return Err(AkitaError::InvalidSetup(
            "i8 decomposition parameters exceed the supported width".to_string(),
        ));
    }
    let output_len = rows
        .len()
        .checked_mul(num_digits)
        .ok_or_else(|| AkitaError::InvalidSetup("i8 decomposition length overflow".into()))?;
    let mut out = vec![[0i8; D]; output_len];
    for (dst_chunk, row) in out.chunks_mut(num_digits).zip(rows.iter()) {
        row.balanced_decompose_pow2_i8_into(dst_chunk, log_basis);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime32Offset99 as F;

    #[test]
    fn decomposition_rejects_parameters_that_would_panic() {
        let row = [CyclotomicRing::<F, 2>::one()];
        assert!(decompose_rows_i8(&row, 1, 0).is_err());
        assert!(decompose_rows_i8(&row, 1, 7).is_err());
        assert!(decompose_rows_i8(&row, 66, 2).is_err());
    }
}
