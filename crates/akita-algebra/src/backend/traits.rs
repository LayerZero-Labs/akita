//! Backend traits for CRT+NTT execution semantics.
//!
//! All traits are generic over `W: PrimeWidth` to support both
//! `i16` (primes < 2^14) and `i32` (primes < 2^30) NTT backends.

use crate::ntt::butterfly::NttTwiddles;
use crate::ntt::crt::GarnerData;
use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};
use crate::ring::CrtNttConvertibleField;

/// Per-prime arithmetic primitives used by CRT+NTT domains.
pub trait NttPrimeOps<W: PrimeWidth, const D: usize> {
    /// Convert canonical coefficient to backend prime representation.
    fn from_canonical(prime: NttPrime<W>, value: W) -> MontCoeff<W>;

    /// Convert backend prime representation back to canonical coefficient.
    fn to_canonical(prime: NttPrime<W>, value: MontCoeff<W>) -> W;

    /// Range-reduce one coefficient from `(-2p, 2p)` to `(-p, p)`.
    fn reduce_range(prime: NttPrime<W>, value: MontCoeff<W>) -> MontCoeff<W>;

    /// Pointwise multiplication in backend prime representation.
    fn pointwise_mul(
        prime: NttPrime<W>,
        out: &mut [MontCoeff<W>; D],
        lhs: &[MontCoeff<W>; D],
        rhs: &[MontCoeff<W>; D],
    );
}

/// Forward/inverse transform kernels for one NTT limb.
pub trait NttTransform<W: PrimeWidth, const D: usize> {
    /// Forward transform from coefficient limb to NTT limb.
    fn forward_ntt(limb: &mut [MontCoeff<W>; D], prime: NttPrime<W>, twiddles: &NttTwiddles<W, D>);

    /// Inverse transform from NTT limb to coefficient limb.
    fn inverse_ntt(limb: &mut [MontCoeff<W>; D], prime: NttPrime<W>, twiddles: &NttTwiddles<W, D>);
}

/// CRT reconstruction from per-prime canonical coefficients via Garner's algorithm.
pub trait CrtReconstruct<W: PrimeWidth, const K: usize, const D: usize> {
    /// Reconstruct coefficient-domain values from canonical CRT residues.
    fn reconstruct<F: CrtNttConvertibleField>(
        primes: &[NttPrime<W>; K],
        canonical_limbs: &[[W; D]; K],
        garner: &GarnerData<W, K>,
    ) -> [F; D];
}

/// Convenience composition trait for full ring backend capability.
pub trait RingBackend<W: PrimeWidth, const K: usize, const D: usize>:
    NttPrimeOps<W, D> + NttTransform<W, D> + CrtReconstruct<W, K, D>
{
}

impl<T, W: PrimeWidth, const K: usize, const D: usize> RingBackend<W, K, D> for T where
    T: NttPrimeOps<W, D> + NttTransform<W, D> + CrtReconstruct<W, K, D>
{
}
