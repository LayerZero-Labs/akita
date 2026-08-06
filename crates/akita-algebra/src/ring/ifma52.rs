//! Limb-major prepared matrices over canonical 50-bit IFMA residues.

use crate::ntt::ifma52::{
    forward, ifma52_enabled, inverse, pointwise_accumulate, Ifma52Prime, Ifma52Twiddles,
};
use crate::{AkitaError, CanonicalField, CrtCapacity, CyclotomicRing, FieldCore};

/// Parameters for one fixed-size IFMA52 CRT profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ifma52Params<const K: usize, const D: usize> {
    primes: [Ifma52Prime; K],
    twiddles: [Ifma52Twiddles<D>; K],
    gamma: [[u64; K]; K],
    use_ifma: bool,
}

impl<const K: usize, const D: usize> Ifma52Params<K, D> {
    /// Validate prime moduli and compute transform and Garner tables.
    pub fn new(moduli: [u64; K]) -> Result<Self, AkitaError> {
        if K == 0 {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 CRT profile must contain a prime".into(),
            ));
        }
        let primes: [Ifma52Prime; K] = moduli
            .map(Ifma52Prime::new)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| AkitaError::InvalidSetup("IFMA52 prime count mismatch".into()))?;
        let twiddles: [Ifma52Twiddles<D>; K] = primes
            .iter()
            .copied()
            .map(Ifma52Twiddles::compute)
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| AkitaError::InvalidSetup("IFMA52 twiddle count mismatch".into()))?;
        let mut gamma = [[0; K]; K];
        for index in 1..K {
            for prior in 0..index {
                gamma[index][prior] = modular_inverse(
                    primes[prior].modulus % primes[index].modulus,
                    primes[index].modulus,
                )?;
            }
        }
        Ok(Self {
            primes,
            twiddles,
            gamma,
            use_ifma: ifma52_enabled(),
        })
    }

    /// Exact CRT capacity of this profile.
    pub fn crt_capacity(&self) -> CrtCapacity {
        CrtCapacity::from_prime_moduli(self.primes.iter().map(|prime| prime.modulus as u128))
    }

    fn reconstruct<F: FieldCore + CanonicalField>(
        &self,
        canonical: &[[u64; D]; K],
    ) -> Result<CyclotomicRing<F, D>, AkitaError> {
        if K == 0 {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 CRT profile must contain a prime".into(),
            ));
        }
        let coefficients = std::array::from_fn(|coefficient| {
            let mut digits = [0i64; K];
            let first = canonical[0][coefficient];
            digits[0] = center(first, self.primes[0].modulus);
            for index in 1..K {
                let modulus = i128::from(self.primes[index].modulus);
                let mut digit = i128::from(center(
                    canonical[index][coefficient],
                    self.primes[index].modulus,
                ));
                for prior in 0..index {
                    digit = (digit - i128::from(digits[prior])).rem_euclid(modulus);
                    digit = (digit * i128::from(self.gamma[index][prior])).rem_euclid(modulus);
                }
                if digit > modulus / 2 {
                    digit -= modulus;
                }
                digits[index] = digit as i64;
            }

            let mut result = F::from_i64(digits[0]);
            let mut partial_product = F::from_u64(self.primes[0].modulus);
            for index in 1..K {
                result += F::from_i64(digits[index]) * partial_product;
                if index + 1 < K {
                    partial_product *= F::from_u64(self.primes[index].modulus);
                }
            }
            result
        });
        Ok(CyclotomicRing::from_coefficients(coefficients))
    }
}

/// A prepared row-major matrix within each contiguous CRT limb.
///
/// `limbs[k][row * width + column]` is one transformed ring. Keeping the CRT
/// limb outside the flat matrix makes the limb-major mat-vec loop contiguous.
#[derive(Debug)]
pub struct Ifma52NttMatrix<const K: usize, const D: usize> {
    limbs: [Vec<[u64; D]>; K],
}

impl<const K: usize, const D: usize> Ifma52NttMatrix<K, D> {
    /// Prepare a flat row-major coefficient matrix in negacyclic NTT form.
    pub fn prepare<F: FieldCore + CanonicalField>(
        rings: &[CyclotomicRing<F, D>],
        params: &Ifma52Params<K, D>,
    ) -> Self {
        let mut limbs: [Vec<[u64; D]>; K] =
            std::array::from_fn(|_| Vec::with_capacity(rings.len()));
        for ring in rings {
            let centered = ring.centered_coefficients_i128();
            for (limb, (prime, twiddles)) in limbs
                .iter_mut()
                .zip(params.primes.iter().zip(&params.twiddles))
            {
                let mut transformed =
                    centered.map(|value| value.rem_euclid(i128::from(prime.modulus)) as u64);
                forward(&mut transformed, *prime, twiddles, params.use_ifma);
                limb.push(transformed);
            }
        }
        Self { limbs }
    }

    /// In-memory byte footprint of the prepared matrix.
    #[must_use]
    pub fn cache_bytes(&self) -> usize {
        self.limbs.iter().map(Vec::len).sum::<usize>() * D * core::mem::size_of::<u64>()
    }

    /// Number of flat matrix entries in each CRT limb.
    #[must_use]
    pub fn len(&self) -> usize {
        self.limbs.first().map_or(0, Vec::len)
    }

    /// Whether the prepared matrix has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Multiply by one exact signed-i16 vector.
    pub fn mat_vec_i16<F: FieldCore + CanonicalField>(
        &self,
        num_rows: usize,
        rhs: &[[i16; D]],
        params: &Ifma52Params<K, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let num_cols = rhs.len();
        let required = num_rows
            .checked_mul(num_cols)
            .ok_or(AkitaError::InvalidProof)?;
        if self.limbs.iter().any(|limb| limb.len() < required) {
            return Err(AkitaError::InvalidSetup(
                "prepared IFMA52 matrix prefix is undersized".into(),
            ));
        }
        if num_rows == 0 || num_cols == 0 {
            return Ok(vec![CyclotomicRing::zero(); num_rows]);
        }

        let mut accumulators = vec![[[0; D]; K]; num_rows];
        let tile_width = (64 * 1024 / (D * core::mem::size_of::<u64>())).max(1);
        for (prime_index, ((matrix_limb, prime), twiddles)) in self
            .limbs
            .iter()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
            .enumerate()
        {
            let mut transformed_rhs = Vec::with_capacity(tile_width);
            for tile_start in (0..num_cols).step_by(tile_width) {
                let tile_end = (tile_start + tile_width).min(num_cols);
                transformed_rhs.clear();
                transformed_rhs.extend(rhs[tile_start..tile_end].iter().map(|digits| {
                    let mut transformed = digits.map(|digit| {
                        i128::from(digit).rem_euclid(i128::from(prime.modulus)) as u64
                    });
                    forward(&mut transformed, *prime, twiddles, params.use_ifma);
                    transformed
                }));
                for (row, accumulator) in accumulators.iter_mut().enumerate() {
                    let row_start = row * num_cols + tile_start;
                    let row_end = row * num_cols + tile_end;
                    for (matrix_entry, rhs_entry) in
                        matrix_limb[row_start..row_end].iter().zip(&transformed_rhs)
                    {
                        pointwise_accumulate(
                            &mut accumulator[prime_index],
                            matrix_entry,
                            rhs_entry,
                            *prime,
                            params.use_ifma,
                        );
                    }
                }
            }
        }

        accumulators
            .into_iter()
            .map(|mut accumulator| {
                for (limb, (prime, twiddles)) in accumulator
                    .iter_mut()
                    .zip(params.primes.iter().zip(&params.twiddles))
                {
                    inverse(limb, *prime, twiddles, params.use_ifma);
                }
                params.reconstruct(&accumulator)
            })
            .collect()
    }
}

fn center(value: u64, modulus: u64) -> i64 {
    if value > modulus / 2 {
        value as i64 - modulus as i64
    } else {
        value as i64
    }
}

fn modular_inverse(value: u64, modulus: u64) -> Result<u64, AkitaError> {
    let (mut old_remainder, mut remainder) = (i128::from(modulus), i128::from(value));
    let (mut old_coefficient, mut coefficient) = (0i128, 1i128);
    while remainder != 0 {
        let quotient = old_remainder / remainder;
        (old_remainder, remainder) = (remainder, old_remainder - quotient * remainder);
        (old_coefficient, coefficient) = (coefficient, old_coefficient - quotient * coefficient);
    }
    if old_remainder != 1 {
        return Err(AkitaError::InvalidSetup(
            "IFMA52 CRT primes are not pairwise coprime".into(),
        ));
    }
    Ok(old_coefficient.rem_euclid(i128::from(modulus)) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::ifma52::IFMA52_PRIMES;
    use akita_field::Prime64Offset59;

    #[test]
    fn limb_major_i16_matvec_matches_ring_arithmetic() {
        const D: usize = 64;
        type F = Prime64Offset59;
        let params =
            Ifma52Params::<2, D>::new([IFMA52_PRIMES[0], IFMA52_PRIMES[1]]).expect("params");
        let matrix = (0..6)
            .map(|entry| {
                CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_i64((entry * 13 + coefficient * 7) as i64 - 40)
                }))
            })
            .collect::<Vec<_>>();
        let rhs = (0..3)
            .map(|column| std::array::from_fn(|index| (column * 9 + index) as i16 - 20))
            .collect::<Vec<_>>();
        let prepared = Ifma52NttMatrix::prepare(&matrix, &params);
        let actual = prepared.mat_vec_i16::<F>(2, &rhs, &params).expect("matvec");
        let expected = matrix
            .chunks_exact(3)
            .map(|row| {
                row.iter()
                    .zip(&rhs)
                    .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
                        sum + *lhs
                            * CyclotomicRing::from_coefficients(
                                rhs.map(|value| F::from_i64(value.into())),
                            )
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
