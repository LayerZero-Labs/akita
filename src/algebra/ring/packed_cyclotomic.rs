//! Packed additive negacyclic ring helpers for one-hot kernels.

use super::{CyclotomicRing, SparseChallenge};
use crate::algebra::fields::packed_additive::PackedAdditive;
use crate::algebra::fields::{HasAdditivePacking, ReduceTo};
use crate::FieldCore;
use std::array::from_fn;

/// Negacyclic ring with coefficients packed across the coefficient dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedNegacyclicRing<P: PackedAdditive, const D: usize> {
    coeffs: Vec<P>,
}

impl<P: PackedAdditive, const D: usize> PackedNegacyclicRing<P, D> {
    #[inline]
    fn packed_len() -> usize {
        assert!(P::WIDTH > 0, "packed width must be positive");
        assert_eq!(
            D % P::WIDTH,
            0,
            "ring degree must be divisible by packed width"
        );
        D / P::WIDTH
    }

    /// Packed zero ring element.
    #[inline]
    pub fn zero() -> Self {
        Self {
            coeffs: vec![P::zero(); Self::packed_len()],
        }
    }

    /// Pack a full scalar coefficient array.
    #[inline]
    pub fn from_coefficients(coeffs: [P::Scalar; D]) -> Self {
        let width = P::WIDTH;
        let packed = coeffs
            .chunks_exact(width)
            .map(|chunk| P::from_fn(|lane| chunk[lane]))
            .collect();
        Self { coeffs: packed }
    }

    /// Convert to a scalar coefficient array.
    #[inline]
    pub fn unpack_coefficients(&self) -> [P::Scalar; D] {
        let width = P::WIDTH;
        from_fn(|idx| self.coeffs[idx / width].extract(idx % width))
    }

    /// Reduce a packed additive ring back to the base field.
    #[inline]
    pub fn reduce<F>(&self) -> CyclotomicRing<F, D>
    where
        F: FieldCore,
        P::Scalar: ReduceTo<F>,
    {
        let width = P::WIDTH;
        let coeffs = from_fn(|idx| self.coeffs[idx / width].extract(idx % width).reduce());
        CyclotomicRing::from_coefficients(coeffs)
    }

    /// Accumulate another packed ring coefficient-wise.
    #[inline]
    pub fn accumulate_from(&mut self, rhs: &Self) {
        for (dst, src) in self.coeffs.iter_mut().zip(rhs.coeffs.iter()) {
            *dst += *src;
        }
    }

    #[inline]
    fn shift_accumulate_signed_into(&self, dst: &mut Self, k: usize, subtract: bool) {
        let width = P::WIDTH;
        let num_packs = Self::packed_len();
        let k = k % D;
        let pack_shift = k / width;
        let lane_shift = k % width;

        if lane_shift == 0 {
            for (src_pack_idx, src_pack) in self.coeffs.iter().copied().enumerate() {
                let raw_dst = src_pack_idx + pack_shift;
                let dst_pack = raw_dst % num_packs;
                let wrapped = raw_dst >= num_packs;
                let neg = subtract ^ wrapped;
                if neg {
                    dst.coeffs[dst_pack] -= src_pack;
                } else {
                    dst.coeffs[dst_pack] += src_pack;
                }
            }
            return;
        }

        for (src_pack_idx, src_pack) in self.coeffs.iter().copied().enumerate() {
            let raw_dst0 = src_pack_idx + pack_shift;
            let dst0 = raw_dst0 % num_packs;
            let neg0 = subtract ^ (raw_dst0 >= num_packs);
            let contrib0 = P::from_fn(|lane| {
                if lane < lane_shift {
                    P::Scalar::default()
                } else {
                    let mut value = src_pack.extract(lane - lane_shift);
                    if neg0 {
                        value = -value;
                    }
                    value
                }
            });
            dst.coeffs[dst0] += contrib0;

            let raw_dst1 = raw_dst0 + 1;
            let dst1 = raw_dst1 % num_packs;
            let neg1 = subtract ^ (raw_dst1 >= num_packs);
            let contrib1 = P::from_fn(|lane| {
                if lane >= lane_shift {
                    P::Scalar::default()
                } else {
                    let mut value = src_pack.extract(width - lane_shift + lane);
                    if neg1 {
                        value = -value;
                    }
                    value
                }
            });
            dst.coeffs[dst1] += contrib1;
        }
    }

    /// Fused negacyclic shift + accumulate: `dst += self * X^k`.
    #[inline]
    pub fn shift_accumulate_into(&self, dst: &mut Self, k: usize) {
        self.shift_accumulate_signed_into(dst, k, false);
    }

    /// Fused negacyclic shift + subtract: `dst -= self * X^k`.
    #[inline]
    pub fn shift_sub_into(&self, dst: &mut Self, k: usize) {
        self.shift_accumulate_signed_into(dst, k, true);
    }

    /// Fused multiply-by-monomial-sum + accumulate.
    #[inline]
    pub fn mul_by_monomial_sum_into(&self, dst: &mut Self, nonzero_positions: &[usize]) {
        for &k in nonzero_positions {
            self.shift_accumulate_into(dst, k);
        }
    }

    /// Fused sparse challenge multiply-accumulate for small integer coefficients.
    #[inline]
    pub fn mul_by_sparse_into(&self, challenge: &SparseChallenge, dst: &mut Self) {
        for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
            match coeff {
                1 => self.shift_accumulate_into(dst, pos as usize),
                -1 => self.shift_sub_into(dst, pos as usize),
                c if c > 0 => {
                    for _ in 0..(c as usize) {
                        self.shift_accumulate_into(dst, pos as usize);
                    }
                }
                c => {
                    for _ in 0..((-c) as usize) {
                        self.shift_sub_into(dst, pos as usize);
                    }
                }
            }
        }
    }

    /// Pack a canonical ring element into additive-wide packed coefficients.
    #[inline]
    pub fn from_ring<F>(ring: &CyclotomicRing<F, D>) -> Self
    where
        F: FieldCore + HasAdditivePacking,
        P: PackedAdditive<Scalar = F::AdditiveWide>,
    {
        let width = P::WIDTH;
        let packed = ring
            .coefficients()
            .chunks_exact(width)
            .map(|chunk| P::from_fn(|lane| F::AdditiveWide::from(chunk[lane])))
            .collect();
        Self { coeffs: packed }
    }
}

impl<P: PackedAdditive<Scalar = i32>, const D: usize> PackedNegacyclicRing<P, D> {
    /// Pack a sparse challenge into signed `i32` coefficients.
    pub fn from_sparse_challenge(challenge: &SparseChallenge) -> Self {
        let mut coeffs = [0i32; D];
        for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
            coeffs[pos as usize] = coeff as i32;
        }
        Self::from_coefficients(coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::PackedNegacyclicRing;
    use crate::algebra::fields::{HasAdditivePacking, Prime128M8M4M1M0, SignedPacking};
    use crate::algebra::ring::{CyclotomicRing, SparseChallenge};
    use crate::FieldSampling;
    use rand::{rngs::StdRng, SeedableRng};
    use std::array::from_fn;

    fn scalar_shift_accumulate<const D: usize>(src: &[i32; D], dst: &mut [i32; D], k: usize) {
        let k = k % D;
        for (i, value) in src.iter().copied().enumerate() {
            let target = i + k;
            if target < D {
                dst[target] += value;
            } else {
                dst[target - D] -= value;
            }
        }
    }

    #[test]
    fn signed_packed_shift_matches_scalar() {
        const D: usize = 16;
        let src = from_fn(|i| i as i32 - 4);
        let packed = PackedNegacyclicRing::<SignedPacking, D>::from_coefficients(src);
        let mut packed_dst = PackedNegacyclicRing::<SignedPacking, D>::zero();
        packed.shift_accumulate_into(&mut packed_dst, 5);

        let mut scalar_dst = [0i32; D];
        scalar_shift_accumulate(&src, &mut scalar_dst, 5);
        assert_eq!(packed_dst.unpack_coefficients(), scalar_dst);
    }

    #[test]
    fn sparse_challenge_accumulate_matches_scalar() {
        const D: usize = 16;
        let challenge = SparseChallenge {
            positions: vec![1, 3, 11],
            coeffs: vec![1, -1, 2],
        };
        let packed = PackedNegacyclicRing::<SignedPacking, D>::from_sparse_challenge(&challenge);
        let mut packed_dst = PackedNegacyclicRing::<SignedPacking, D>::zero();
        packed.mul_by_sparse_into(&challenge, &mut packed_dst);

        let src = packed.unpack_coefficients();
        let mut scalar_dst = [0i32; D];
        for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
            for _ in 0..(coeff.unsigned_abs() as usize) {
                if coeff > 0 {
                    scalar_shift_accumulate(&src, &mut scalar_dst, pos as usize);
                } else {
                    let mut tmp = [0i32; D];
                    scalar_shift_accumulate(&src, &mut tmp, pos as usize);
                    for (dst, value) in scalar_dst.iter_mut().zip(tmp.iter()) {
                        *dst -= *value;
                    }
                }
            }
        }
        assert_eq!(packed_dst.unpack_coefficients(), scalar_dst);
    }

    #[test]
    fn fp128_additive_packed_roundtrip_matches_scalar() {
        const D: usize = 16;
        type F = Prime128M8M4M1M0;
        type P = <F as HasAdditivePacking>::AdditivePacking;

        let mut rng = StdRng::seed_from_u64(11);
        let ring = CyclotomicRing::from_coefficients(from_fn(|_| F::sample(&mut rng)));
        let packed = PackedNegacyclicRing::<P, D>::from_ring(&ring);
        assert_eq!(packed.reduce::<F>(), ring);

        let mut packed_dst = PackedNegacyclicRing::<P, D>::zero();
        packed.shift_accumulate_into(&mut packed_dst, 7);
        let shifted = ring.negacyclic_shift(7);
        assert_eq!(packed_dst.reduce::<F>(), shifted);
    }
}
