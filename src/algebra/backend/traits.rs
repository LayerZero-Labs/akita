//! Backend traits for CRT+NTT execution semantics.

use crate::algebra::ntt::butterfly::NttTwiddles;
use crate::algebra::ntt::crt::QData;
use crate::algebra::ntt::prime::{MontCoeff, NttPrime};
use crate::algebra::ring::CrtNttConvertibleField;

/// Per-prime arithmetic primitives used by CRT+NTT domains.
pub trait NttPrimeOps<const D: usize> {
    /// Convert canonical coefficient to backend prime representation.
    fn from_canonical(prime: NttPrime, value: i16) -> MontCoeff;

    /// Convert backend prime representation back to canonical coefficient.
    fn to_canonical(prime: NttPrime, value: MontCoeff) -> i16;

    /// Range-reduce one backend prime coefficient.
    fn reduce(prime: NttPrime, value: MontCoeff) -> MontCoeff;

    /// Pointwise multiplication in backend prime representation.
    fn pointwise_mul(
        prime: NttPrime,
        out: &mut [MontCoeff; D],
        lhs: &[MontCoeff; D],
        rhs: &[MontCoeff; D],
    );
}

/// Forward/inverse transform kernels for one NTT limb.
pub trait NttTransform<const D: usize> {
    /// Forward transform from coefficient limb to NTT limb.
    fn forward_ntt(limb: &mut [MontCoeff; D], prime: NttPrime, twiddles: &NttTwiddles<D>);

    /// Inverse transform from NTT limb to coefficient limb.
    fn inverse_ntt(limb: &mut [MontCoeff; D], prime: NttPrime, twiddles: &NttTwiddles<D>);
}

/// CRT reconstruction contract from per-prime canonical coefficients.
pub trait CrtReconstruct<const K: usize, const D: usize, const L: usize> {
    /// Reconstruct coefficient-domain values from canonical CRT residues.
    fn reconstruct<F: CrtNttConvertibleField>(
        primes: &[NttPrime; K],
        canonical_limbs: &[[i16; D]; K],
        qdata: &QData<K, L>,
    ) -> [F; D];
}

/// Convenience composition trait for full ring backend capability.
pub trait RingBackend<const K: usize, const D: usize, const L: usize>:
    NttPrimeOps<D> + NttTransform<D> + CrtReconstruct<K, D, L>
{
}

impl<T, const K: usize, const D: usize, const L: usize> RingBackend<K, D, L> for T where
    T: NttPrimeOps<D> + NttTransform<D> + CrtReconstruct<K, D, L>
{
}
