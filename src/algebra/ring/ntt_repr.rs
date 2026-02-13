//! NTT-domain representation of cyclotomic ring elements.

use crate::algebra::fields::{Fp32, Fp64};
use crate::algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
use crate::algebra::ntt::crt::QData;
use crate::algebra::ntt::prime::{MontCoeff, NttPrime};
use crate::Field;

use super::cyclotomic::CyclotomicRing;

/// NTT-domain representation of a cyclotomic ring element.
///
/// Stores `K` arrays of `D` [`MontCoeff`] values, one per CRT prime.
/// Multiplication is pointwise per prime — O(K*D) vs O(D^2) for coefficient form.
///
/// Use [`CyclotomicNtt::from_ring`] and [`CyclotomicNtt::to_ring`] to convert
/// between coefficient and NTT domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CyclotomicNtt<const K: usize, const D: usize> {
    pub(crate) limbs: [[MontCoeff; D]; K],
}

/// Field types that can safely convert to and from the CRT/NTT representation.
///
/// This trait is intentionally narrower than [`Field`]: NTT conversion needs a
/// canonical integer representative so we can reduce coefficients mod small CRT
/// primes without going through serialization hacks.
pub trait NttConvertibleField: Field {
    /// Reduce this field element modulo a small prime `p`.
    fn mod_small_prime(self, p: i16) -> i16;

    /// Reconstruct from a residue in `[0, q)` after CRT combination.
    fn from_q_residue_u128(x: u128) -> Self;
}

impl<const MODULUS: u32> NttConvertibleField for Fp32<MODULUS> {
    #[inline]
    fn mod_small_prime(self, p: i16) -> i16 {
        (self.to_canonical_u32() % (p as u32)) as i16
    }

    #[inline]
    fn from_q_residue_u128(x: u128) -> Self {
        let narrowed = u32::try_from(x).expect("CRT residue does not fit in u32");
        Self::from_u64(narrowed as u64)
    }
}

impl<const MODULUS: u64> NttConvertibleField for Fp64<MODULUS> {
    #[inline]
    fn mod_small_prime(self, p: i16) -> i16 {
        (self.to_canonical_u64() % (p as u64)) as i16
    }

    #[inline]
    fn from_q_residue_u128(x: u128) -> Self {
        let narrowed = u64::try_from(x).expect("CRT residue does not fit in u64");
        Self::from_u64(narrowed)
    }
}

impl<const K: usize, const D: usize> CyclotomicNtt<K, D> {
    /// The additive identity (all zeros in every CRT limb).
    pub fn zero() -> Self {
        Self {
            limbs: [[MontCoeff::from_raw(0); D]; K],
        }
    }

    /// Convert a coefficient-form ring element into NTT domain.
    ///
    /// For each CRT prime:
    /// 1. Reduce each ring coefficient mod the prime.
    /// 2. Convert to Montgomery form.
    /// 3. Apply the forward NTT butterfly.
    pub fn from_ring<F: NttConvertibleField>(
        ring: &CyclotomicRing<F, D>,
        primes: &[NttPrime; K],
        twiddles: &[NttTwiddles<D>; K],
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(0); D]; K];
        for ((limb, prime), tw) in limbs.iter_mut().zip(primes.iter()).zip(twiddles.iter()) {
            // Reduce each coefficient mod p and convert to Montgomery form.
            for (dst, src) in limb.iter_mut().zip(ring.coeffs.iter()) {
                let val = src.mod_small_prime(prime.p);
                *dst = prime.from_canonical(val);
            }
            forward_ntt(limb, *prime, tw);
        }
        Self { limbs }
    }

    /// Convert from NTT domain back to coefficient form.
    ///
    /// For each CRT prime:
    /// 1. Apply the inverse NTT butterfly.
    /// 2. Convert from Montgomery to canonical form.
    ///
    /// Then performs CRT reconstruction to recover the ring element.
    ///
    /// # Panics
    ///
    /// Panics if `q` or CRT limb constants do not fit in `u128`.
    pub fn to_ring<F: NttConvertibleField, const L: usize>(
        &self,
        primes: &[NttPrime; K],
        twiddles: &[NttTwiddles<D>; K],
        qdata: &QData<K, L>,
    ) -> CyclotomicRing<F, D> {
        // Inverse NTT each limb (work on a copy).
        let mut canonical = [[0i16; D]; K];
        for (k, ((can, prime), tw)) in canonical
            .iter_mut()
            .zip(primes.iter())
            .zip(twiddles.iter())
            .enumerate()
        {
            let mut limb = self.limbs[k];
            inverse_ntt(&mut limb, *prime, tw);
            for (dst, src) in can.iter_mut().zip(limb.iter()) {
                *dst = prime.to_canonical(*src);
            }
        }

        // CRT reconstruction: first lift from residues mod p_i to an integer mod P
        // (P = product p_i), then reduce that lift mod q.
        let q = qdata.q_u128().expect("q must fit in u128");
        let q_i128 = i128::try_from(q).expect("q must fit in i128");
        let prime_moduli: [i128; K] = std::array::from_fn(|k| primes[k].p as i128);
        let big_p = prime_moduli.iter().fold(1i128, |acc, p| {
            acc.checked_mul(*p)
                .expect("product of CRT primes must fit i128")
        });
        let crt_m: [i128; K] = std::array::from_fn(|k| big_p / prime_moduli[k]);
        let crt_inv: [i128; K] =
            std::array::from_fn(|k| mod_inverse(crt_m[k] % prime_moduli[k], prime_moduli[k]));

        let mut coeffs = [F::zero(); D];
        for (d, coeff) in coeffs.iter_mut().enumerate() {
            let mut acc: i128 = 0;
            for k in 0..K {
                let ck = canonical[k][d] as i128;
                acc = (acc + ck * crt_m[k] * crt_inv[k]) % big_p;
            }
            let lifted = (acc % big_p + big_p) % big_p;
            let residue = ((lifted % q_i128) + q_i128) % q_i128;
            *coeff = F::from_q_residue_u128(residue as u128);
        }

        CyclotomicRing::from_coefficients(coeffs)
    }
    /// Add another NTT element and reduce each coefficient with the matching
    /// prime to maintain valid Montgomery ranges.
    pub fn add_reduced(&self, rhs: &Self, primes: &[NttPrime; K]) -> Self {
        let mut out = self.clone();
        for (k, (limb, rhs_limb)) in out.limbs.iter_mut().zip(rhs.limbs.iter()).enumerate() {
            let prime = primes[k];
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                let sum = MontCoeff::from_raw(a.raw().wrapping_add(b.raw()));
                *a = prime.reduce(sum);
            }
        }
        out
    }

    /// Subtract another NTT element and reduce each coefficient with the
    /// matching prime to maintain valid Montgomery ranges.
    pub fn sub_reduced(&self, rhs: &Self, primes: &[NttPrime; K]) -> Self {
        let mut out = self.clone();
        for (k, (limb, rhs_limb)) in out.limbs.iter_mut().zip(rhs.limbs.iter()).enumerate() {
            let prime = primes[k];
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                let diff = MontCoeff::from_raw(a.raw().wrapping_sub(b.raw()));
                *a = prime.reduce(diff);
            }
        }
        out
    }

    /// Negate each NTT coefficient and reduce with the matching prime.
    pub fn neg_reduced(&self, primes: &[NttPrime; K]) -> Self {
        let mut out = self.clone();
        for (k, limb) in out.limbs.iter_mut().enumerate() {
            let prime = primes[k];
            for a in limb.iter_mut() {
                let neg = MontCoeff::from_raw(a.raw().wrapping_neg());
                *a = prime.reduce(neg);
            }
        }
        out
    }

    /// Pointwise multiplication in NTT domain.
    ///
    /// Each limb is multiplied pointwise using the corresponding prime's
    /// Montgomery multiplication.
    pub fn pointwise_mul(&self, rhs: &Self, primes: &[NttPrime; K]) -> Self {
        let mut out = [[MontCoeff::from_raw(0); D]; K];
        for (k, ((o, a), b)) in out
            .iter_mut()
            .zip(self.limbs.iter())
            .zip(rhs.limbs.iter())
            .enumerate()
        {
            primes[k].pointwise_mul(o, a, b);
        }
        Self { limbs: out }
    }
}

fn mod_inverse(a: i128, modulus: i128) -> i128 {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (modulus, ((a % modulus) + modulus) % modulus);

    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }

    assert_eq!(r, 1, "CRT inverse does not exist");
    (t % modulus + modulus) % modulus
}
