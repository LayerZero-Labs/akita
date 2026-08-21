//! Limb-major prepared matrices over canonical 50-bit IFMA residues.

use crate::ntt::crt::{modular_inverse, GarnerData};
use crate::ntt::ifma52::{
    forward, ifma52_enabled, inverse, pointwise_dot_accumulate, Ifma52Prime, Ifma52Twiddles,
};
use crate::{
    CanonicalEncoding, CenteredMontLut, CrtCapacity, CrtNttParamSet, CyclotomicCrtNtt,
    CyclotomicRing, Field,
};
use akita_error::AkitaError;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Ifma52I16Tail<const K: usize> {
    modulus: i64,
    residue_weight: i64,
    digit_weights: [i64; K],
}

/// Parameters for one fixed-size IFMA52 CRT profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ifma52Params<const K: usize, const D: usize> {
    primes: [Ifma52Prime; K],
    twiddles: [Ifma52Twiddles<D>; K],
    garner: GarnerData<K>,
    use_ifma: bool,
    i16_tail: Option<Ifma52I16Tail<K>>,
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
        let garner = GarnerData::try_from_moduli(moduli)?;
        Ok(Self {
            primes,
            twiddles,
            garner,
            use_ifma: ifma52_enabled(),
            i16_tail: None,
        })
    }

    /// Extend this profile with one exactness-only i16 prime.
    pub fn with_i16_tail(mut self, tail_modulus: i16) -> Result<Self, AkitaError> {
        let modulus = i64::from(tail_modulus);
        if modulus <= 2 {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 i16 tail modulus must be positive".into(),
            ));
        }
        let mut residue_weight = 1i64;
        let mut digit_weights = [0; K];
        for index in (0..K).rev() {
            let inverse =
                modular_inverse(self.primes[index].modulus % modulus as u64, modulus as u64)?
                    as i64;
            residue_weight = (residue_weight * inverse) % modulus;
            digit_weights[index] = (-residue_weight).rem_euclid(modulus);
        }
        self.i16_tail = Some(Ifma52I16Tail {
            modulus,
            residue_weight,
            digit_weights,
        });
        Ok(self)
    }

    /// Exact CRT capacity of this profile.
    pub fn crt_capacity(&self) -> CrtCapacity {
        let capacity =
            CrtCapacity::from_prime_moduli(self.primes.iter().map(|prime| prime.modulus as u128));
        self.i16_tail.as_ref().map_or(capacity.clone(), |tail| {
            capacity.with_prime_modulus(tail.modulus as u128)
        })
    }

    /// Whether this profile includes an exactness-only i16 CRT limb.
    #[must_use]
    pub const fn has_i16_tail(&self) -> bool {
        self.i16_tail.is_some()
    }

    #[inline]
    fn reconstruct<F: Field + CanonicalEncoding>(
        &self,
        canonical: &[[u64; D]; K],
        tail_canonical: Option<&[i16; D]>,
    ) -> Result<CyclotomicRing<F, D>, AkitaError> {
        if K == 0 {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 CRT profile must contain a prime".into(),
            ));
        }
        if self.i16_tail.is_some() != tail_canonical.is_some() {
            return Err(AkitaError::InvalidSetup(
                "IFMA52 reconstruction tail does not match its parameters".into(),
            ));
        }
        let mut field_product = F::one();
        let field_weights: [F; K] = std::array::from_fn(|index| {
            let weight = field_product;
            field_product *= F::from_u64(self.primes[index].modulus);
            weight
        });
        let tail_field_weight = field_product;
        let coefficients = std::array::from_fn(|coefficient| {
            let moduli = self.primes.map(|prime| prime.modulus);
            let residues = std::array::from_fn(|limb| i128::from(canonical[limb][coefficient]));
            let digits = self.garner.centered_mixed_radix(residues, moduli);

            let mut result = F::zero();
            for (digit, weight) in digits.iter().zip(field_weights) {
                result += F::from_i128(*digit) * weight;
            }
            if let (Some(tail), Some(tail_canonical)) = (&self.i16_tail, tail_canonical) {
                let tail_digit = i128::from(tail_canonical[coefficient])
                    * i128::from(tail.residue_weight)
                    + digits
                        .iter()
                        .zip(tail.digit_weights)
                        .map(|(digit, weight)| *digit * i128::from(weight))
                        .sum::<i128>();
                let tail_modulus = i128::from(tail.modulus);
                let mut tail_digit = tail_digit.rem_euclid(tail_modulus);
                if tail_digit > tail_modulus / 2 {
                    tail_digit -= tail_modulus;
                }
                result += F::from_i64(tail_digit as i64) * tail_field_weight;
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
    params: Ifma52Params<K, D>,
}

impl<const K: usize, const D: usize> Ifma52NttMatrix<K, D> {
    /// Prepare a flat row-major coefficient matrix in negacyclic NTT form.
    pub fn prepare<F: Field + CanonicalEncoding>(
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
        Self {
            limbs,
            params: params.clone(),
        }
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

    /// Exact CRT capacity of the parameters bound to this matrix.
    #[must_use]
    pub fn crt_capacity(&self) -> CrtCapacity {
        self.params.crt_capacity()
    }

    /// Whether the bound parameters require an exactness-only i16 tail.
    #[must_use]
    pub const fn has_i16_tail(&self) -> bool {
        self.params.has_i16_tail()
    }

    /// Multiply by one exact signed-i16 vector.
    #[inline]
    pub fn mat_vec_i16<F: Field + CanonicalEncoding>(
        &self,
        num_rows: usize,
        rhs: &[[i16; D]],
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        if self.params.i16_tail.is_some() {
            return Err(AkitaError::InvalidSetup(
                "prepared IFMA52 tail does not match its parameters".into(),
            ));
        }
        if num_rows == 0 || rhs.is_empty() {
            return Ok(vec![CyclotomicRing::zero(); num_rows]);
        }
        self.mat_vec_i16_canonical(num_rows, rhs)?
            .iter()
            .map(|canonical| self.params.reconstruct(canonical, None))
            .collect()
    }

    /// Multiply by one exact signed-i16 vector with an i16 CRT tail.
    #[inline]
    pub fn mat_vec_i16_with_tail<F: Field + CanonicalEncoding>(
        &self,
        tail_matrix: &[CyclotomicCrtNtt<i16, 1, D>],
        num_rows: usize,
        rhs: &[[i16; D]],
        tail_params: &CrtNttParamSet<i16, 1, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let num_cols = rhs.len();
        let required = num_rows
            .checked_mul(num_cols)
            .ok_or(AkitaError::InvalidProof)?;
        if self.params.i16_tail.is_none() {
            return Err(AkitaError::InvalidSetup(
                "prepared IFMA52 tail does not match its parameters".into(),
            ));
        }
        if tail_matrix.len() < required {
            return Err(AkitaError::InvalidSetup(
                "prepared IFMA52 i16-tail matrix prefix is undersized".into(),
            ));
        }
        if num_rows == 0 || num_cols == 0 {
            return Ok(vec![CyclotomicRing::zero(); num_rows]);
        }

        let accumulators = self.mat_vec_i16_canonical(num_rows, rhs)?;
        let rhs_abs_bound = rhs
            .iter()
            .flatten()
            .map(|digit| i32::from(*digit).unsigned_abs())
            .max()
            .unwrap_or(0) as i32;
        let lut = CenteredMontLut::new(tail_params, rhs_abs_bound);
        let mut tail_accumulators = vec![CyclotomicCrtNtt::zero(); num_rows];
        for (column, digits) in rhs.iter().enumerate() {
            if digits.iter().all(|digit| *digit == 0) {
                continue;
            }
            let centered = digits.map(i32::from);
            let transformed =
                CyclotomicCrtNtt::from_centered_i32_with_lut(&centered, tail_params, &lut);
            for (accumulator, row) in tail_accumulators
                .iter_mut()
                .zip(tail_matrix.chunks_exact(num_cols))
            {
                accumulator.add_assign_pointwise_mul(&row[column], &transformed, tail_params);
            }
        }
        let tail_canonical = tail_accumulators
            .iter()
            .map(|accumulator| accumulator.centered_coefficients_with_params(tail_params)[0])
            .collect::<Vec<_>>();

        accumulators
            .iter()
            .zip(&tail_canonical)
            .map(|(canonical, tail)| self.params.reconstruct(canonical, Some(tail)))
            .collect()
    }

    #[inline(always)]
    fn mat_vec_i16_canonical(
        &self,
        num_rows: usize,
        rhs: &[[i16; D]],
    ) -> Result<Vec<[[u64; D]; K]>, AkitaError> {
        let num_cols = rhs.len();
        let required = num_rows
            .checked_mul(num_cols)
            .ok_or(AkitaError::InvalidProof)?;
        if self.limbs.iter().any(|limb| limb.len() < required) {
            return Err(AkitaError::InvalidSetup(
                "prepared IFMA52 matrix prefix is undersized".into(),
            ));
        }

        let mut accumulators = vec![[[0; D]; K]; num_rows];
        if num_rows == 0 || num_cols == 0 {
            return Ok(accumulators);
        }
        let tile_width = (64 * 1024 / (D * core::mem::size_of::<u64>())).max(1);
        for (prime_index, ((matrix_limb, prime), twiddles)) in self
            .limbs
            .iter()
            .zip(self.params.primes.iter())
            .zip(self.params.twiddles.iter())
            .enumerate()
        {
            let mut transformed_rhs = Vec::with_capacity(tile_width);
            for tile_start in (0..num_cols).step_by(tile_width) {
                let tile_end = (tile_start + tile_width).min(num_cols);
                transformed_rhs.clear();
                transformed_rhs.extend(rhs[tile_start..tile_end].iter().map(|digits| {
                    let mut transformed = digits.map(|digit| prime.canonical_i16(digit));
                    forward(&mut transformed, *prime, twiddles, self.params.use_ifma);
                    transformed
                }));
                for (row, accumulator) in accumulators.iter_mut().enumerate() {
                    let row_start = row * num_cols + tile_start;
                    let row_end = row * num_cols + tile_end;
                    pointwise_dot_accumulate(
                        &mut accumulator[prime_index],
                        &matrix_limb[row_start..row_end],
                        &transformed_rhs,
                        *prime,
                        self.params.use_ifma,
                    );
                }
            }
        }

        for accumulator in &mut accumulators {
            for (limb, (prime, twiddles)) in accumulator
                .iter_mut()
                .zip(self.params.primes.iter().zip(&self.params.twiddles))
            {
                inverse(limb, *prime, twiddles, self.params.use_ifma);
            }
        }
        Ok(accumulators)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::ifma52::IFMA52_PRIMES;
    use crate::ntt::tables::I16_TAIL_PRIME;
    use jolt_field::{One, Prime128OffsetA7F7, Prime64Offset59, Ring, Zero};

    fn assert_limb_major_i16_matvec<const D: usize>() {
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
        let actual = prepared.mat_vec_i16::<F>(2, &rhs).expect("matvec");
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

    #[test]
    fn limb_major_i16_matvec_matches_ring_arithmetic_at_all_ifma_dimensions() {
        assert_limb_major_i16_matvec::<64>();
        assert_limb_major_i16_matvec::<128>();
        assert_limb_major_i16_matvec::<256>();
        assert_limb_major_i16_matvec::<512>();
    }

    fn assert_mixed_ifma_i16_tail_matvec<const D: usize>() {
        type F = Prime64Offset59;
        let params = Ifma52Params::<1, D>::new([IFMA52_PRIMES[0]])
            .expect("params")
            .with_i16_tail(I16_TAIL_PRIME.p)
            .expect("tail params");
        let tail_params = CrtNttParamSet::new([I16_TAIL_PRIME]);
        let matrix = (0..6)
            .map(|entry| {
                CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_u64(
                        IFMA52_PRIMES[0] + (entry as u64 + 1) * 65_537 + coefficient as u64 * 4_099,
                    )
                }))
            })
            .collect::<Vec<_>>();
        let rhs = (0..3)
            .map(|column| {
                std::array::from_fn(|coefficient| ((column * 3 + coefficient * 2) % 5) as i16 - 2)
            })
            .collect::<Vec<_>>();
        let prepared = Ifma52NttMatrix::prepare(&matrix, &params);
        let tail_matrix = matrix
            .iter()
            .map(|ring| CyclotomicCrtNtt::from_ring(ring, &tail_params))
            .collect::<Vec<_>>();
        let actual = prepared
            .mat_vec_i16_with_tail::<F>(&tail_matrix, 2, &rhs, &tail_params)
            .expect("mixed matvec");
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

    #[test]
    fn mixed_ifma_i16_tail_matvec_matches_ring_arithmetic_at_all_dimensions() {
        assert_mixed_ifma_i16_tail_matvec::<64>();
        assert_mixed_ifma_i16_tail_matvec::<128>();
        assert_mixed_ifma_i16_tail_matvec::<256>();
        assert_mixed_ifma_i16_tail_matvec::<512>();
    }

    fn assert_q128_ifma_i16_tail_matvec<const D: usize>() {
        type F = Prime128OffsetA7F7;
        let params = Ifma52Params::<3, D>::new(IFMA52_PRIMES)
            .expect("params")
            .with_i16_tail(I16_TAIL_PRIME.p)
            .expect("tail params");
        let tail_params = CrtNttParamSet::new([I16_TAIL_PRIME]);
        let matrix = (0..6)
            .map(|entry| {
                CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                    let low = (entry as u64 + 1) * 65_537 + coefficient as u64 * 4_099;
                    F::from_i128((u128::from(low) << 80) as i128) - F::from_u64(low.rotate_left(17))
                }))
            })
            .collect::<Vec<_>>();
        let rhs = (0..3)
            .map(|column| {
                std::array::from_fn(|coefficient| {
                    if (column + coefficient) % 2 == 0 {
                        i16::MAX
                    } else {
                        i16::MIN
                    }
                })
            })
            .collect::<Vec<_>>();
        let prepared = Ifma52NttMatrix::prepare(&matrix, &params);
        let tail_matrix = matrix
            .iter()
            .map(|ring| CyclotomicCrtNtt::from_ring(ring, &tail_params))
            .collect::<Vec<_>>();
        let actual = prepared
            .mat_vec_i16_with_tail::<F>(&tail_matrix, 2, &rhs, &tail_params)
            .expect("Q128 mixed matvec");
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

    #[test]
    fn q128_ifma_i16_tail_matvec_matches_ring_arithmetic_at_all_dimensions() {
        assert_q128_ifma_i16_tail_matvec::<64>();
        assert_q128_ifma_i16_tail_matvec::<128>();
        assert_q128_ifma_i16_tail_matvec::<256>();
        assert_q128_ifma_i16_tail_matvec::<512>();
    }

    #[test]
    fn q128_tail_reconstruction_handles_maximum_centered_digits() {
        const D: usize = 64;
        type F = Prime128OffsetA7F7;
        let params = Ifma52Params::<3, D>::new(IFMA52_PRIMES)
            .expect("params")
            .with_i16_tail(I16_TAIL_PRIME.p)
            .expect("tail params");
        let digits = IFMA52_PRIMES.map(|prime| (prime / 2) as i64);
        let residue = |modulus: u64| {
            let modulus = u128::from(modulus);
            let mut residue = 0u128;
            let mut weight = 1u128;
            for (digit, prime) in digits.iter().zip(IFMA52_PRIMES) {
                residue = (residue + (*digit as u128 * weight) % modulus) % modulus;
                weight = (weight * u128::from(prime)) % modulus;
            }
            residue as u64
        };
        let canonical = IFMA52_PRIMES.map(|prime| [residue(prime); D]);
        let tail_modulus = I16_TAIL_PRIME.p as u64;
        let tail_residue = residue(tail_modulus) as i64;
        let tail_centered = if tail_residue > i64::from(I16_TAIL_PRIME.p) / 2 {
            tail_residue - i64::from(I16_TAIL_PRIME.p)
        } else {
            tail_residue
        } as i16;
        let tail = [tail_centered; D];

        let mut field_weight = F::one();
        let mut expected = F::zero();
        for (digit, prime) in digits.into_iter().zip(IFMA52_PRIMES) {
            expected += F::from_i64(digit) * field_weight;
            field_weight *= F::from_u64(prime);
        }
        assert_eq!(
            params
                .reconstruct::<F>(&canonical, Some(&tail))
                .expect("reconstruction"),
            CyclotomicRing::from_coefficients([expected; D])
        );
    }
}
