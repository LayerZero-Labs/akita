//! Public-input batched CRT reconstruction.

#[cfg(test)]
#[path = "radix_sum_tests.rs"]
mod tests;

use super::{CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt};
use crate::ntt::butterfly::inverse_ntt;
use crate::{CyclotomicRing, MontCoeff, PrimeWidth};
use akita_error::AkitaError;

/// Radix bits of [`CyclotomicCrtNtt::radix_sum_to_ring`] planes.
pub const RADIX_SUM_BITS: u32 = 32;
/// Maximum planes whose shifted residues fit in `u128`.
pub const RADIX_SUM_PLANES: usize = 4;

impl<W: PrimeWidth, const K: usize, const D: usize> CyclotomicCrtNtt<W, K, D> {
    /// `Σ_l 2^{32·l} · X_l` in `F`, where `X_l` is the centered integer whose
    /// residues `planes[l]` holds: the weighted sum of up to four planes,
    /// reconstructed by explicit CRT with one field image per coefficient
    /// for all of them, where [`Self::to_ring`] reconstructs each plane by
    /// Garner and lifts it into `F` on its own.
    ///
    /// With `P` the product of the primes and `M_k = P/p_k`, each residue
    /// `x_k` gives `y_k = x_k · M_k^{-1} mod p_k`, and
    /// `X = Σ_k y_k·M_k − u·P` for `u = round(Σ_k y_k/p_k)`. Summed over the
    /// planes, `Σ_l 2^{32·l}·y_{l,k}` and `Σ_l 2^{32·l}·u_l` are exact in
    /// `u128` (every prime is below 2^31, so the shifted terms do not
    /// overlap), which leaves `K + 1` field products per coefficient.
    ///
    /// WARNING: every `X_l` must satisfy `|X_l| < P/4`. `u` is rounded in
    /// `f64`, and the margin keeps `Σ_k y_k/p_k` within `1/4` of it; a plane
    /// past the bound yields a wrong value, not an error. Size the planes
    /// with [`CrtCapacity::supports`](crate::CrtCapacity::supports) at four
    /// times their coefficient bound.
    ///
    /// Not constant time: the residues are read through `f64` rounding.
    /// The planes must be public.
    pub fn radix_sum_to_ring<F: CrtNttConvertibleField>(
        planes: &[Self],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Result<CyclotomicRing<F, D>, AkitaError> {
        if planes.len() > RADIX_SUM_PLANES {
            return Err(AkitaError::InvalidSize {
                expected: RADIX_SUM_PLANES,
                actual: planes.len(),
            });
        }
        let moduli = params.garner.moduli();
        let mut cofactors = [F::one(); K];
        for (k, cofactor) in cofactors.iter_mut().enumerate() {
            for (_, &modulus) in moduli.iter().enumerate().filter(|&(j, _)| j != k) {
                *cofactor *= F::from_u64(modulus);
            }
        }
        let modulus = moduli
            .iter()
            .fold(F::one(), |product, &p| product * F::from_u64(p));
        let reciprocals = moduli.map(|p| 1.0 / p as f64);
        let inverses = params
            .garner
            .crt_inverse()
            .map(|inverse| MontCoeff::from_raw(W::from_i64(inverse as i64)));
        let mut sums = [[0u128; K]; D];
        let mut quotients = [0u128; D];
        for (plane, shift) in planes.iter().zip((0..).step_by(RADIX_SUM_BITS as usize)) {
            let mut fractions = [0f64; D];
            for (k, (((limb, prime), twiddles), inverse)) in plane
                .limbs
                .iter()
                .zip(params.primes.iter())
                .zip(params.twiddles.iter())
                .zip(inverses)
                .enumerate()
            {
                let mut limb = *limb;
                inverse_ntt(&mut limb, *prime, twiddles, params.kernel_plan);
                for ((sum, fraction), residue) in sums.iter_mut().zip(&mut fractions).zip(limb) {
                    // `residue` is `x·R`; the inverse is plain, so the
                    // Montgomery product leaves `x·M_k^{-1}`.
                    let y = prime.normalize(prime.mul(residue, inverse)).raw().to_i64() as u64;
                    sum[k] += u128::from(y) << shift;
                    *fraction += y as f64 * reciprocals[k];
                }
            }
            for (quotient, fraction) in quotients.iter_mut().zip(fractions) {
                *quotient += (fraction.round() as u128) << shift;
            }
        }
        Ok(CyclotomicRing::from_coefficients(std::array::from_fn(
            |index| {
                sums[index]
                    .iter()
                    .zip(&cofactors)
                    .fold(F::zero(), |acc, (&sum, &cofactor)| {
                        acc + F::from_u128_reduced(sum) * cofactor
                    })
                    - F::from_u128_reduced(quotients[index]) * modulus
            },
        )))
    }
}
