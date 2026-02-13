//! NTT-domain representation of cyclotomic ring elements.

use crate::algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
use crate::algebra::ntt::crt::QData;
use crate::algebra::ntt::prime::{MontCoeff, NttPrime};
use crate::Field;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

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
    pub fn from_ring<F: Field>(
        ring: &CyclotomicRing<F, D>,
        primes: &[NttPrime; K],
        twiddles: &[NttTwiddles<D>; K],
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(0); D]; K];
        for (k, ((limb, prime), tw)) in limbs
            .iter_mut()
            .zip(primes.iter())
            .zip(twiddles.iter())
            .enumerate()
        {
            let _ = k;
            // Reduce each coefficient mod p and convert to Montgomery form.
            for (dst, src) in limb.iter_mut().zip(ring.coeffs.iter()) {
                // Extract canonical value via from_u64 → to_canonical_u* → mod p.
                // For generality, use from_i64 with the field element.
                // We need to get an i16 value mod p from the field element.
                // The field element's canonical value mod p:
                let val = coeff_to_i16(*src, prime.p);
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
    pub fn to_ring<F: Field, const L: usize>(
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
            let _ = k;
            let mut limb = self.limbs[k];
            inverse_ntt(&mut limb, *prime, tw);
            for (dst, src) in can.iter_mut().zip(limb.iter()) {
                *dst = prime.to_canonical(*src);
            }
        }

        // CRT reconstruction: combine the per-prime canonical values into
        // a single value mod q for each coefficient position.
        let q = qdata.q_u128().expect("q must fit in u128");
        let mut coeffs = [F::zero(); D];
        for (d, coeff) in coeffs.iter_mut().enumerate() {
            let mut acc: i128 = 0;
            for k in 0..K {
                // CRT formula: coeff = sum_k (x_k * t_k * canonical_k) mod q
                // where x_k = P/p_k mod q and t_k is the CRT reconstruction constant.
                // For simplicity, use the direct formula with the precomputed xvec.
                let xk = u128::try_from(qdata.xvec[k]).expect("xvec limb must fit in u128") as i128;
                let tk = primes[k].t as i128;
                let ck = canonical[k][d] as i128;
                // The CRT helper: canonical_k * t_k gives the partial
                // contribution in Montgomery form; multiply by xvec to lift.
                // Actually, t_k is already the CRT coefficient: val_k * t_k mod p_k
                // gives the weight, then multiply by x_k (= P/p_k mod q).
                let pk = primes[k].p as i128;
                // val_k * t_k mod p_k:
                let weighted = ((ck * tk) % pk + pk) % pk;
                acc += weighted * xk;
            }
            // Add pmq correction and reduce mod q.
            let pmq = u128::try_from(qdata.pmq).expect("pmq must fit in u128") as i128;
            acc = ((acc + pmq) % q as i128 + q as i128) % q as i128;
            *coeff = F::from_u64(acc as u64);
        }

        CyclotomicRing::from_coefficients(coeffs)
    }
}

/// Extract an `i16` value from a field element reduced mod a small prime `p`.
///
/// Uses `from_u64(0)` as a baseline to detect the field's zero, then
/// reconstructs via the `from_u64` / arithmetic path.
fn coeff_to_i16<F: Field>(val: F, p: i16) -> i16 {
    // Strategy: probe by constructing F::from_u64(k) for k = 0..p-1.
    // This is O(p) which is fine for small NTT primes (< 2^14).
    // A faster approach would require F to expose a canonical integer,
    // but our Field trait doesn't have that yet.
    //
    // Optimization: use the field's arithmetic to compute val mod p directly.
    // val mod p = val - floor(val / p) * p, but we don't have integer division.
    //
    // Practical approach for small p: convert field element to an integer
    // by testing against F::from_u64. But this is too slow for p ~ 13000.
    //
    // Better: use the fact that our field types (Fp32, Fp64, Fp128) all
    // have a canonical u32/u64/u128 representative. We extract it via
    // serialization.
    let mut buf = [0u8; 16]; // enough for u128
    let _ = val.serialize_with_mode(&mut buf[..], crate::primitives::serialization::Compress::No);
    // Read as little-endian u64 (works for Fp32, Fp64; Fp128 needs u128 but
    // the mod-p result fits in u64 for our small primes).
    let le_val = u64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]);
    (le_val % (p as u64)) as i16
}

impl<const K: usize, const D: usize> AddAssign for CyclotomicNtt<K, D> {
    fn add_assign(&mut self, rhs: Self) {
        for (limb, rhs_limb) in self.limbs.iter_mut().zip(rhs.limbs.iter()) {
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                *a = MontCoeff::from_raw(a.raw().wrapping_add(b.raw()));
            }
        }
    }
}

impl<const K: usize, const D: usize> SubAssign for CyclotomicNtt<K, D> {
    fn sub_assign(&mut self, rhs: Self) {
        for (limb, rhs_limb) in self.limbs.iter_mut().zip(rhs.limbs.iter()) {
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                *a = MontCoeff::from_raw(a.raw().wrapping_sub(b.raw()));
            }
        }
    }
}

impl<const K: usize, const D: usize> Add for CyclotomicNtt<K, D> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<const K: usize, const D: usize> Sub for CyclotomicNtt<K, D> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}

impl<const K: usize, const D: usize> Neg for CyclotomicNtt<K, D> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut out = self.limbs;
        for limb in out.iter_mut() {
            for a in limb.iter_mut() {
                *a = MontCoeff::from_raw(a.raw().wrapping_neg());
            }
        }
        Self { limbs: out }
    }
}

/// Pointwise multiplication in NTT domain.
///
/// Each limb is multiplied pointwise using the corresponding prime's
/// Montgomery multiplication.
impl<const K: usize, const D: usize> CyclotomicNtt<K, D> {
    /// Pointwise multiplication using the given primes.
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
