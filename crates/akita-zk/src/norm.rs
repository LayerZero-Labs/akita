//! Balanced coefficient bounds over Akita cyclotomic rings.

use crate::error::ZkResult;
use akita_algebra::CyclotomicRing;
use akita_field::{
    pseudo_mersenne_modulus, AkitaError, CanonicalField, FieldCore, PseudoMersenneField,
};
use rand_core::RngCore;

/// Return the modulus for a pseudo-Mersenne Akita field.
///
/// # Errors
///
/// Returns an error if the field metadata does not describe a supported
/// pseudo-Mersenne modulus.
pub fn field_modulus<F>() -> ZkResult<u128>
where
    F: PseudoMersenneField,
{
    pseudo_mersenne_modulus(F::MODULUS_BITS, F::MODULUS_OFFSET).ok_or_else(|| {
        AkitaError::InvalidInput("unsupported pseudo-Mersenne field modulus".to_string())
    })
}

/// Return the balanced absolute value of a field element.
///
/// The result is `min(x, q - x)` for canonical representative `x`.
///
/// # Errors
///
/// Returns an error if the field modulus metadata is unsupported.
pub fn centered_abs_u128<F>(x: F) -> ZkResult<u128>
where
    F: CanonicalField + PseudoMersenneField,
{
    let q = field_modulus::<F>()?;
    let canonical = x.to_canonical_u128();
    if canonical <= q / 2 {
        Ok(canonical)
    } else {
        Ok(q - canonical)
    }
}

/// Return the balanced signed lift of a field element.
///
/// # Errors
///
/// Returns an error if the field modulus metadata is unsupported, or if the
/// balanced representative does not fit in `i128`.
pub fn centered_i128<F>(x: F) -> ZkResult<i128>
where
    F: CanonicalField + PseudoMersenneField,
{
    let q = field_modulus::<F>()?;
    let canonical = x.to_canonical_u128();
    if canonical <= q / 2 {
        i128::try_from(canonical).map_err(|_| {
            AkitaError::InvalidInput("positive centered representative exceeds i128".to_string())
        })
    } else {
        let magnitude = q - canonical;
        let signed = i128::try_from(magnitude).map_err(|_| {
            AkitaError::InvalidInput("negative centered representative exceeds i128".to_string())
        })?;
        Ok(-signed)
    }
}

/// Convert a small centered integer into a field element.
///
/// # Errors
///
/// Returns an error if the magnitude does not fit in `u128`.
pub fn field_from_centered_i128<F>(value: i128) -> ZkResult<F>
where
    F: CanonicalField,
{
    if value >= 0 {
        Ok(F::from_canonical_u128_reduced(value as u128))
    } else {
        let magnitude = value.checked_neg().ok_or_else(|| {
            AkitaError::InvalidInput("centered value magnitude overflow".to_string())
        })? as u128;
        Ok(-F::from_canonical_u128_reduced(magnitude))
    }
}

/// Check a ring element's balanced coefficient infinity norm.
///
/// # Errors
///
/// Returns an error if the field modulus metadata is unsupported.
pub fn ring_within_infinity_bound<F, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    bound: u128,
) -> ZkResult<bool>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
{
    for &coeff in ring.coefficients() {
        if centered_abs_u128(coeff)? > bound {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Check a ring vector's balanced coefficient infinity norm.
///
/// # Errors
///
/// Returns an error if the field modulus metadata is unsupported.
pub fn ring_vec_within_infinity_bound<F, const D: usize>(
    vector: &[CyclotomicRing<F, D>],
    bound: u128,
) -> ZkResult<bool>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
{
    for ring in vector {
        if !ring_within_infinity_bound(ring, bound)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Sample a centered integer uniformly from `[-bound, bound]`.
///
/// # Errors
///
/// Returns an error if the interval is too large for the current sampler.
pub fn sample_centered_i128<R>(rng: &mut R, bound: u128) -> ZkResult<i128>
where
    R: RngCore + ?Sized,
{
    let width = bound
        .checked_mul(2)
        .and_then(|x| x.checked_add(1))
        .ok_or_else(|| AkitaError::InvalidInput("box width overflow".to_string()))?;
    let width_u64 = u64::try_from(width).map_err(|_| {
        AkitaError::InvalidInput("box sampler currently supports width <= u64::MAX".to_string())
    })?;
    let zone = u64::MAX - (u64::MAX % width_u64);
    loop {
        let sample = rng.next_u64();
        if sample < zone {
            let offset = (sample % width_u64) as u128;
            return Ok(offset as i128 - bound as i128);
        }
    }
}

/// Sample a ring element with coefficients uniform in `[-bound, bound]`.
///
/// # Errors
///
/// Returns an error if the interval is too large for the current sampler.
pub fn sample_ring_box<F, R, const D: usize>(
    rng: &mut R,
    bound: u128,
) -> ZkResult<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField,
    R: RngCore + ?Sized,
{
    let mut coeffs = [F::zero(); D];
    for coeff in &mut coeffs {
        *coeff = field_from_centered_i128(sample_centered_i128(rng, bound)?)?;
    }
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

/// Sample a ring vector with coefficients uniform in `[-bound, bound]`.
///
/// # Errors
///
/// Returns an error if the interval is too large for the current sampler.
pub fn sample_ring_vec_box<F, R, const D: usize>(
    rng: &mut R,
    len: usize,
    bound: u128,
) -> ZkResult<Vec<CyclotomicRing<F, D>>>
where
    F: FieldCore + CanonicalField,
    R: RngCore + ?Sized,
{
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(sample_ring_box(rng, bound)?);
    }
    Ok(out)
}
