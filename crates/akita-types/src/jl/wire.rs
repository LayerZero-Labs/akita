//! Verifier-wire validation for revealed JL image coordinates.

use akita_field::{AkitaError, CanonicalField, FieldCore};

/// Embed and optionally norm-check verifier-wire JL image coordinates.
///
/// # Errors
///
/// Returns an error if the coordinate count does not match `n_rows`, if the
/// checked integer L2 norm exceeds `bound_sq`, or if any signed coordinate lies
/// outside the field's injective signed window.
pub fn embed_jl_image_coords<F>(
    image_coords: &[i32],
    n_rows: usize,
    bound_sq: Option<u128>,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if image_coords.len() != n_rows {
        return Err(AkitaError::InvalidSize {
            expected: n_rows,
            actual: image_coords.len(),
        });
    }
    if let Some(bound_sq) = bound_sq {
        check_l2_norm(image_coords, bound_sq)?;
    }
    let half_q = field_modulus::<F>() / 2;
    image_coords
        .iter()
        .map(|&coord| embed_signed_i32::<F>(coord, half_q))
        .collect()
}

fn check_l2_norm(coords: &[i32], bound_sq: u128) -> Result<(), AkitaError> {
    let mut norm_sq = 0u128;
    for &coord in coords {
        let mag = u128::from(coord.unsigned_abs());
        let sq = mag * mag;
        norm_sq = norm_sq.checked_add(sq).ok_or_else(|| {
            AkitaError::InvalidInput("JL image squared norm exceeds u128".to_string())
        })?;
    }
    if norm_sq > bound_sq {
        return Err(AkitaError::InvalidInput(format!(
            "JL image squared L2 norm {norm_sq} exceeds bound {bound_sq}"
        )));
    }
    Ok(())
}

/// Embed a signed integer coordinate into the base field injective window.
#[inline]
pub fn embed_signed_i32<F>(coord: i32, half_q: u128) -> Result<F, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let mag = u128::from(coord.unsigned_abs());
    if mag > half_q {
        return Err(AkitaError::InvalidInput(format!(
            "JL image coordinate {coord} outside injective signed window (|c| <= {half_q})"
        )));
    }
    let elem = F::from_canonical_u128_reduced(mag);
    Ok(if coord < 0 { -elem } else { elem })
}

/// Base-field modulus as a `u128`.
#[inline]
pub fn field_modulus<F>() -> u128
where
    F: FieldCore + CanonicalField,
{
    (-F::one()).to_canonical_u128() + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp64;

    type F = Fp64<4294967197>;

    #[test]
    fn image_embedding_checks_shape_norm_and_signed_window() {
        let ok = embed_jl_image_coords::<F>(&[-3, 4], 2, Some(25)).unwrap();
        assert_eq!(ok.len(), 2);
        assert!(matches!(
            embed_jl_image_coords::<F>(&[-3, 4], 3, Some(25)),
            Err(AkitaError::InvalidSize { .. })
        ));
        assert!(embed_jl_image_coords::<F>(&[-3, 4], 2, Some(24)).is_err());
        assert!(embed_jl_image_coords::<F>(&[i32::MAX], 1, None).is_err());
    }
}
