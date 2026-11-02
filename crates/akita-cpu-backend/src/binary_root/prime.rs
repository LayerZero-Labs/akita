//! Honest construction of order-three prime evaluation planes.

use akita_algebra::ring::trinomial::module_map::unpack_signed_components;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BINARY_SETUP_P;
use akita_types::binary_source::{BinarySourceDescriptor, PrimePlaneManifest, PRIME_PLANE_DIGITS};
use jolt_field::{CanonicalEncoding, Field};

/// Build the committed prime planes before drawing column fold challenges.
///
/// Sources are polynomial/column/ring-row ordered. Returned natural digits
/// are unique-plane/column/digit/coefficient ordered. Each coefficient is first
/// reduced modulo the proof prime and centered; 43 balanced digits cover that
/// entire centered interval. Canonical nonnegative residues do not all fit the
/// positive endpoint of that digit interval.
pub fn prime_plane_digits<F: Field + CanonicalEncoding>(
    descriptor: &BinarySourceDescriptor,
    manifest: &PrimePlaneManifest,
    sources: &[[i128; 648]],
) -> Result<Vec<i8>, AkitaError> {
    if F::from_u128(BINARY_SETUP_P) != F::zero() || F::from_u128(BINARY_SETUP_P - 1) != -F::one() {
        return Err(AkitaError::InvalidSetup(
            "prime plane field differs from the proof prime".into(),
        ));
    }
    let columns = descriptor.index_map().columns();
    let scalar_rows = checked::exact_div(descriptor.index_map().padded_carriers(), columns)
        .ok_or(AkitaError::InvalidProof)?;
    let ring_rows = checked::exact_div(scalar_rows, 4).ok_or(AkitaError::InvalidProof)?;
    let expected = checked::product([descriptor.polynomials().len(), columns, ring_rows])
        .ok_or(AkitaError::InvalidProof)?;
    if sources.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: sources.len(),
        });
    }
    let width = checked::product([manifest.planes().len(), columns, PRIME_PLANE_DIGITS, 162])
        .ok_or(AkitaError::InvalidProof)?;
    let mut digits = Vec::new();
    digits
        .try_reserve_exact(width)
        .map_err(|_| AkitaError::InvalidProof)?;
    digits.resize(width, 0);
    for (plane, claim) in manifest.planes().iter().enumerate() {
        let (rows, _, _) = claim.weights::<F>(descriptor)?;
        for column in 0..columns {
            let mut partial = [F::zero(); 162];
            for (row, row_weight) in rows.iter().enumerate() {
                if *row_weight == F::zero() {
                    continue;
                }
                let element = checked::mul_add(claim.polynomial, columns, column)
                    .and_then(|x| checked::mul_add(x, ring_rows, row / 4))
                    .ok_or(AkitaError::InvalidProof)?;
                let components = unpack_signed_components(
                    sources.get(element).ok_or(AkitaError::InvalidProof)?,
                )?;
                for (output, &coefficient) in partial
                    .iter_mut()
                    .zip(components.get(row % 4).ok_or(AkitaError::InvalidProof)?)
                {
                    *output += *row_weight * F::from_i128(coefficient);
                }
            }
            for (coefficient, value) in partial.into_iter().enumerate() {
                let canonical = value.to_u128_checked().ok_or(AkitaError::InvalidProof)?;
                let mut centered = if canonical > BINARY_SETUP_P / 2 {
                    -i128::try_from(BINARY_SETUP_P - canonical)
                        .map_err(|_| AkitaError::InvalidProof)?
                } else {
                    i128::try_from(canonical).map_err(|_| AkitaError::InvalidProof)?
                };
                for digit in 0..PRIME_PLANE_DIGITS {
                    let remainder = centered.rem_euclid(8);
                    let balanced = if remainder >= 4 {
                        remainder - 8
                    } else {
                        remainder
                    };
                    // Division before subtraction avoids overflowing at either
                    // endpoint of the centered i128 coefficient interval.
                    centered = centered / 8 + (centered % 8 - balanced) / 8;
                    let address = checked::mul_add(plane, columns, column)
                        .and_then(|x| checked::mul_add(x, PRIME_PLANE_DIGITS, digit))
                        .and_then(|x| checked::mul_add(x, 162, coefficient))
                        .ok_or(AkitaError::InvalidProof)?;
                    *digits.get_mut(address).ok_or(AkitaError::InvalidProof)? = balanced as i8;
                }
                if centered != 0 {
                    return Err(AkitaError::Internal(
                        "centered prime coefficient exceeds 43 balanced digits".into(),
                    ));
                }
            }
        }
    }
    Ok(digits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, PrimePlaneClaim};
    use jolt_field::{One, Prime128Offset275, Ring, Zero};

    type F = Prime128Offset275;

    #[test]
    fn prime_plane_centered_digit_bound_and_roundtrip() {
        let source = BinarySourceDescriptor::new(
            BinaryIndexMap::new(128, 4, 4, 1).unwrap(),
            vec![[1; 32]],
            BinaryCommitmentProfiles {
                a: [2; 32],
                b: [3; 32],
            },
        )
        .unwrap();
        let claim = PrimePlaneClaim {
            polynomial: 0,
            point: vec![0; 10],
            tensor_component: 0,
            value: 0,
        };
        let manifest = PrimePlaneManifest::new(&source, &[claim]).unwrap();
        let half = i128::try_from(BINARY_SETUP_P / 2).unwrap();
        for integer in [0, 1, -1, half, -half, i128::MAX, i128::MIN + 1] {
            let mut element = [0i128; 648];
            element[0] = integer;
            let digits = prime_plane_digits::<F>(&source, &manifest, &[element]).unwrap();
            assert_eq!(digits.len(), PRIME_PLANE_DIGITS * 162);
            assert!(digits.iter().all(|digit| (-4..=3).contains(digit)));
            let mut value = F::zero();
            let mut power = F::one();
            for digit in digits.chunks_exact(162) {
                value += power * F::from_i128(i128::from(digit[0]));
                assert!(digit.iter().skip(1).all(|&x| x == 0));
                power *= F::from_u64(8);
            }
            assert_eq!(value, F::from_i128(integer));
        }
    }
}
