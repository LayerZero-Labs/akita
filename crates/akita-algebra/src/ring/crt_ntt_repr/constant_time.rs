//! Constant-time CRT-NTT conversions.

use super::{CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt};
use crate::ntt::butterfly::{forward_ntt, inverse_ntt};
use crate::ntt::crt::digits_to_field_ct;
use crate::{CyclotomicRing, PrimeWidth, LIFT_BIAS};

impl<W: PrimeWidth, const K: usize, const D: usize> CyclotomicCrtNtt<W, K, D> {
    /// [`Self::to_ring`] with no branch, table lookup or memory address that
    /// depends on the coefficients: masked centering and Garner digits, and a
    /// biased lift of each digit into `F` ([`crate::lift_biased`]).
    ///
    /// Constant time holds if the target's integer and `F` arithmetic and the
    /// inverse NTT kernel are themselves data independent.
    pub fn to_ring_ct<F: CrtNttConvertibleField>(
        &self,
        params: &CrtNttParamSet<W, K, D>,
    ) -> CyclotomicRing<F, D> {
        let mut digits = [[0i64; D]; K];
        for (k, ((column, prime), twiddles)) in digits
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
            .enumerate()
        {
            let mut limb = self.limbs[k];
            inverse_ntt(&mut limb, *prime, twiddles, params.kernel_plan);
            for (dst, src) in column.iter_mut().zip(limb.iter()) {
                *dst = prime.center(prime.to_canonical(*src)).to_i64();
            }
        }
        params.garner.centered_mixed_radix_ct(&mut digits);
        let (weights, _) = params.garner.field_weights::<F>();
        let bias =
            weights.iter().fold(F::zero(), |sum, &weight| sum + weight) * F::from_u64(LIFT_BIAS);
        CyclotomicRing::from_coefficients(std::array::from_fn(|index| {
            digits_to_field_ct(
                &std::array::from_fn(|limb| digits[limb][index]),
                &weights,
                bias,
            )
        }))
    }
}

impl<const K: usize, const D: usize> CyclotomicCrtNtt<i32, K, D> {
    /// Overwrite `self` with the negacyclic CRT+NTT image of centered `i32`
    /// coefficients, in time and with memory addresses independent of their
    /// values: one Montgomery product per residue, no lookup table, and no
    /// strategy chosen from the magnitudes.
    ///
    /// Every `i32` is exact: with `R = 2^32`, its Montgomery product with
    /// `R² mod p` stays below `p · 2^31` in magnitude.
    ///
    /// Timing assumes data-independent target integer arithmetic and forward
    /// NTT kernels. Parameters and CPU dispatch are public.
    pub fn assign_centered_i32_ct(
        &mut self,
        coeffs: &[i32; D],
        params: &CrtNttParamSet<i32, K, D>,
    ) {
        for ((limb, prime), tw) in self
            .limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            for (dst, &coefficient) in limb.iter_mut().zip(coeffs) {
                *dst = prime.from_canonical(coefficient);
            }
            forward_ntt(limb, *prime, tw, params.kernel_plan);
        }
    }
}
