//! Honest construction of order-three prime evaluation planes.

use akita_algebra::ring::trinomial::module_map::unpack_signed_components;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BINARY_SETUP_P;
use akita_types::binary_source::{BinarySourceDescriptor, PrimePlaneManifest, PRIME_PLANE_DIGITS};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{cfg_chunks, cfg_chunks_mut, CanonicalEncoding, Field};

/// Digits of one plane column: [`PRIME_PLANE_DIGITS`] rows of 162 coefficients.
const COLUMN_DIGITS: usize = PRIME_PLANE_DIGITS * 162;

/// Build the committed prime planes before drawing column fold challenges.
///
/// Sources are polynomial/column/ring-row ordered. Returned natural digits
/// are polynomial/column/digit/coefficient ordered. Each coefficient is first
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
    manifest.validate(descriptor)?;
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
    let Some(prime) = manifest.claim() else {
        return Ok(digits);
    };
    let (rows, _, _) = prime.weights::<F>(descriptor)?;
    for (plane, claim) in manifest.planes().iter().enumerate() {
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
            let address = checked::mul_add(plane, columns, column)
                .and_then(|x| checked::product([x, COLUMN_DIGITS]))
                .ok_or(AkitaError::InvalidProof)?;
            let column_digits = checked::range(address, COLUMN_DIGITS)
                .and_then(|range| digits.get_mut(range))
                .ok_or(AkitaError::InvalidProof)?;
            write_prime_column(column_digits, &partial)?;
        }
    }
    Ok(digits)
}

/// Construct prime planes directly from column-contiguous packed carriers.
/// Shared subset sums lift source bits into the proof prime without expanding
/// source ring coefficients or multiplying each bit by a row weight.
pub(crate) fn packed_prime_plane_digits<F: Field + CanonicalEncoding>(
    descriptor: &BinarySourceDescriptor,
    manifest: &PrimePlaneManifest,
    sources: &[u128],
) -> Result<Vec<i8>, AkitaError> {
    if F::from_u128(BINARY_SETUP_P) != F::zero() || F::from_u128(BINARY_SETUP_P - 1) != -F::one() {
        return Err(AkitaError::InvalidSetup(
            "prime plane field differs from the proof prime".into(),
        ));
    }
    manifest.validate(descriptor)?;
    let index = descriptor.index_map();
    let columns = index.columns();
    let expected = checked::product([descriptor.polynomials().len(), index.padded_carriers()])
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
    let plane_width = checked::product([columns, COLUMN_DIGITS]).ok_or(AkitaError::InvalidProof)?;
    let Some(prime) = manifest.claim() else {
        return Ok(digits);
    };
    let (rows, _, _) = prime.weights::<F>(descriptor)?;
    for (claim, plane_digits) in manifest
        .planes()
        .iter()
        .zip(digits.chunks_exact_mut(plane_width))
    {
        let start = checked::product([claim.polynomial, index.padded_carriers()])
            .ok_or(AkitaError::InvalidProof)?;
        let range =
            checked::range(start, index.padded_carriers()).ok_or(AkitaError::InvalidProof)?;
        let partials = akita_labinius_prover::lowered::weighted_source_bits::<
            akita_algebra::binary::BinaryField128,
            F,
        >(
            sources.get(range).ok_or(AkitaError::InvalidProof)?,
            &rows,
            1,
            columns,
            162,
            0,
        )?;
        if partials.len() != checked::product([columns, 162]).ok_or(AkitaError::InvalidProof)? {
            return Err(AkitaError::InvalidProof);
        }
        // Columns own disjoint digit windows, so the parallel form writes the
        // same digits.
        cfg_chunks_mut!(plane_digits, COLUMN_DIGITS)
            .zip(cfg_chunks!(partials, 162))
            .try_for_each(|(column_digits, partial)| write_prime_column(column_digits, partial))?;
    }
    Ok(digits)
}

/// Write the balanced base-8 digits of one plane column's coefficients into
/// `column_digits`: digit-major, [`PRIME_PLANE_DIGITS`] rows of 162.
fn write_prime_column<F: Field + CanonicalEncoding>(
    column_digits: &mut [i8],
    partial: &[F],
) -> Result<(), AkitaError> {
    if column_digits.len() != COLUMN_DIGITS || partial.len() != 162 {
        return Err(AkitaError::InvalidProof);
    }
    for (coefficient, value) in partial.iter().copied().enumerate() {
        let canonical = value.to_u128_checked().ok_or(AkitaError::InvalidProof)?;
        let mut centered = if canonical > BINARY_SETUP_P / 2 {
            -i128::try_from(BINARY_SETUP_P - canonical).map_err(|_| AkitaError::InvalidProof)?
        } else {
            i128::try_from(canonical).map_err(|_| AkitaError::InvalidProof)?
        };
        for row in column_digits.chunks_exact_mut(162) {
            // The low three bits of the two's complement are the Euclidean
            // remainder modulo 8.
            let remainder = (centered & 7) as i8;
            let balanced = if remainder >= 4 {
                remainder - 8
            } else {
                remainder
            };
            // `|centered| <= (P - 1) / 2 < 2^127 - 4` only shrinks, so the
            // subtraction cannot overflow, and the shift divides a multiple
            // of 8 exactly.
            centered = (centered - i128::from(balanced)) >> 3;
            *row.get_mut(coefficient).ok_or(AkitaError::InvalidProof)? = balanced;
        }
        if centered != 0 {
            return Err(AkitaError::Internal(
                "centered prime coefficient exceeds 43 balanced digits".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::binary_source::{
        BinaryCommitmentProfiles, BinaryIndexMap, PrimeGroupClaim, PrimePlaneClaim,
    };
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
        let claim = PrimeGroupClaim {
            point: vec![0; 10],
            evaluations: vec![PrimePlaneClaim {
                polynomial: 0,
                value: 0,
            }],
        };
        let manifest = PrimePlaneManifest::new(&source, Some(claim)).unwrap();
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

    #[test]
    fn column_digits_match_euclidean_division() {
        let half = BINARY_SETUP_P / 2;
        let mut canonical = vec![0, 1, 2, 3, 4, 5, 7, 8, half - 1, half, half + 1];
        canonical.extend([BINARY_SETUP_P - 1, BINARY_SETUP_P - 4, BINARY_SETUP_P - 5]);
        let mut state = 0x9e37_79b9_7f4a_7c15_u128;
        while canonical.len() < 162 {
            state = state.wrapping_mul(0x2545_f491_4f6c_dd1d).wrapping_add(1);
            canonical.push(state % BINARY_SETUP_P);
        }
        let partial = canonical
            .iter()
            .map(|&x| F::from_u128(x))
            .collect::<Vec<_>>();
        let mut digits = vec![0i8; COLUMN_DIGITS];
        write_prime_column(&mut digits, &partial).unwrap();
        for (coefficient, &value) in canonical.iter().enumerate() {
            let mut centered = if value > half {
                -i128::try_from(BINARY_SETUP_P - value).unwrap()
            } else {
                i128::try_from(value).unwrap()
            };
            for digit in 0..PRIME_PLANE_DIGITS {
                let remainder = centered.rem_euclid(8);
                let balanced = if remainder >= 4 {
                    remainder - 8
                } else {
                    remainder
                };
                centered = centered / 8 + (centered % 8 - balanced) / 8;
                assert_eq!(i128::from(digits[digit * 162 + coefficient]), balanced);
            }
            assert_eq!(centered, 0);
        }
        assert!(write_prime_column(&mut digits[1..], &partial).is_err());
        assert!(write_prime_column(&mut digits, &partial[1..]).is_err());
    }
}
