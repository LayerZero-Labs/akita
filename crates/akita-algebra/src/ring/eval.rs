//! Scalar evaluation helpers for cyclotomic ring elements.

use super::CyclotomicRing;
use akita_field::unreduced::HasUnreducedOps;
use akita_field::{FieldCore, MulBase, MulBaseUnreduced, Zero};

/// Return the first `len` powers of `alpha`, starting with one.
pub fn scalar_powers<F: FieldCore>(alpha: F, len: usize) -> Vec<F> {
    let mut out = vec![F::zero(); len];
    let mut power = F::one();
    for val in out.iter_mut() {
        *val = power;
        power *= alpha;
    }
    out
}

/// Evaluate a cyclotomic ring element at the scalar `alpha`.
pub fn eval_ring_at<F: FieldCore, const D: usize>(r: &CyclotomicRing<F, D>, alpha: &F) -> F {
    let mut acc = F::zero();
    let mut power = F::one();
    for coeff in r.coefficients() {
        acc += *coeff * power;
        power *= *alpha;
    }
    acc
}

/// Evaluate a ring element against precomputed powers of `alpha`.
///
/// Ring coefficients live in `F`; the scalar powers may live in any field `E`
/// that supports multiplication by `F`. The ordinary base-field case is `E = F`.
///
/// # Panics
///
/// Panics in debug builds if `alpha_pows.len() != D`.
#[inline]
pub fn eval_ring_at_pows<F, E, const D: usize>(r: &CyclotomicRing<F, D>, alpha_pows: &[E]) -> E
where
    F: FieldCore,
    E: FieldCore + MulBase<F>,
{
    debug_assert_eq!(alpha_pows.len(), D);
    eval_flat_ring_at_pows(r.coefficients(), alpha_pows)
}

/// Evaluate a flat ring element (raw coefficients at a runtime ring
/// dimension) against precomputed powers of `alpha`.
///
/// This is the runtime-dimension form of [`eval_ring_at_pows`]: the ring
/// dimension is `alpha_pows.len()` and `coeffs` must hold exactly one ring
/// element of that dimension.
///
/// # Panics
///
/// Panics in debug builds if `coeffs.len() != alpha_pows.len()`.
#[inline]
pub fn eval_flat_ring_at_pows<F, E>(coeffs: &[F], alpha_pows: &[E]) -> E
where
    F: FieldCore,
    E: FieldCore + MulBase<F>,
{
    debug_assert_eq!(alpha_pows.len(), coeffs.len());
    coeffs
        .iter()
        .zip(alpha_pows.iter())
        .fold(E::zero(), |acc, (coeff, alpha_pow)| {
            acc + alpha_pow.mul_base(*coeff)
        })
}

/// Fast (deferred-reduction) counterpart of [`eval_flat_ring_at_pows`].
///
/// This is the runtime-dimension form of [`eval_ring_at_pows_fast`].
///
/// # Panics
///
/// Panics in debug builds if `coeffs.len() != alpha_pows.len()`.
#[inline]
pub fn eval_flat_ring_at_pows_fast<F, E>(coeffs: &[F], alpha_pows: &[E]) -> E
where
    F: FieldCore,
    E: MulBaseUnreduced<F>,
{
    debug_assert_eq!(alpha_pows.len(), coeffs.len());
    let accum = coeffs.iter().zip(alpha_pows.iter()).fold(
        <E as HasUnreducedOps>::ProductAccum::zero(),
        |acc, (coeff, alpha_pow)| acc + alpha_pow.mul_base_to_product_accum(*coeff),
    );
    <E as HasUnreducedOps>::reduce_product_accum(accum)
}

/// Fast (deferred-reduction) counterpart of [`eval_ring_at_pows`].
///
/// Same signature and result as [`eval_ring_at_pows`], but accumulates all `D`
/// widening `E × F` products into a single [`HasUnreducedOps::ProductAccum`] and
/// reduces **once** instead of reducing after every coefficient. On a 128-bit
/// prime the modular reduction is a large fraction of each multiply, so this
/// turns ~`D` reductions into one.
///
/// Bit-identical to [`eval_ring_at_pows`] as long as the running product-sum
/// stays within the accumulator's carry headroom. For `Fp128` each `u128`
/// accumulator limb holds a 64-bit product word, so the sum of up to ~`2^64`
/// products is exact — `D ≈ 64` is trivially within bounds (validated by
/// `deferred_matches_per_term_fp128_d64`). This is why callers can use it even
/// though `Fp128` keeps `DELAYED_PRODUCT_SUM_IS_EXACT` at its conservative
/// `false` default.
///
/// # Panics
///
/// Panics in debug builds if `alpha_pows.len() != D`.
#[inline]
pub fn eval_ring_at_pows_fast<F, E, const D: usize>(r: &CyclotomicRing<F, D>, alpha_pows: &[E]) -> E
where
    F: FieldCore,
    E: MulBaseUnreduced<F>,
{
    debug_assert_eq!(alpha_pows.len(), D);
    let accum = r.coefficients().iter().zip(alpha_pows.iter()).fold(
        <E as HasUnreducedOps>::ProductAccum::zero(),
        |acc, (coeff, alpha_pow)| acc + alpha_pow.mul_base_to_product_accum(*coeff),
    );
    <E as HasUnreducedOps>::reduce_product_accum(accum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;
    const D: usize = 64;

    fn sample(seed: u128) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15_1234_5678_9ABC_DEF1)
                .wrapping_add((i as u128).wrapping_mul(0x100_0000_01B3));
            F::from_canonical_u128(x & ((1u128 << 120) - 1))
        }))
    }

    /// The deferred-reduction dot product must equal the per-term reduce path
    /// bit-for-bit at `D = 64` (validates the `Fp128` accumulator headroom that
    /// `DELAYED_PRODUCT_SUM_IS_EXACT = false` leaves formally unblessed).
    #[test]
    fn deferred_matches_per_term_fp128_d64() {
        for seed in 0..128u128 {
            let ring = sample(seed.wrapping_add(1));
            let alpha = F::from_canonical_u128(
                seed.wrapping_mul(0x1234_5678_9ABC).wrapping_add(7) & ((1u128 << 120) - 1),
            );
            let mut pows = [F::zero(); D];
            let mut p = F::one();
            for slot in pows.iter_mut() {
                *slot = p;
                p *= alpha;
            }
            assert_eq!(
                eval_ring_at_pows(&ring, &pows),
                eval_ring_at_pows_fast(&ring, &pows),
                "deferred reduction diverged from per-term at seed {seed}"
            );
            assert_eq!(
                eval_flat_ring_at_pows(ring.coefficients(), &pows),
                eval_flat_ring_at_pows_fast(ring.coefficients(), &pows),
                "flat deferred reduction diverged from per-term at seed {seed}"
            );
        }
    }
}
