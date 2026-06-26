//! Verifier-wire validation for revealed JL image coordinates.

use akita_field::{field_modulus, AkitaError, CanonicalField, FieldCore};

/// Squared Euclidean norm over the integers with checked accumulation.
///
/// # Errors
///
/// Returns an error if the running sum exceeds `u128`.
pub fn jl_l2_norm_sq_checked(coords: &[i32]) -> Result<u128, AkitaError> {
    let mut acc: u128 = 0;
    for &c in coords {
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
pub fn check_jl_l2_norm(coords: &[i32], bound_sq: u128) -> Result<(), AkitaError> {
    let norm_sq = jl_l2_norm_sq_checked(coords)?;
    if norm_sq > bound_sq {
        return Err(AkitaError::InvalidInput(format!(
            "JL image squared L2 norm {norm_sq} exceeds bound {bound_sq}"
        )));
    }
    Ok(())
}

/// Embed and optionally norm-check verifier-wire JL image coordinates.
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
        check_jl_l2_norm(image_coords, bound_sq)?;
    }
    let half_q = field_modulus::<F>() / 2;
    image_coords
        .iter()
        .map(|&coord| embed_signed_i32::<F>(coord, half_q))
        .collect()
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

    #[test]
    fn l2_norm_helpers_match_expectations() {
        assert_eq!(jl_l2_norm_sq_checked(&[3, -4]).unwrap(), 25);
        assert!(check_jl_l2_norm(&[3, -4], 25).is_ok());
        assert!(check_jl_l2_norm(&[3, -4], 24).is_err());
    }
}
