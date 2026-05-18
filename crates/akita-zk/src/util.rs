use crate::error::ZkResult;
use akita_field::AkitaError;
use rand_core::RngCore;

pub(crate) fn open_unit_f64<R>(rng: &mut R) -> f64
where
    R: RngCore + ?Sized,
{
    const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
    (((rng.next_u64() >> 11) as f64) + 0.5) * SCALE
}

pub(crate) fn ceil_f64_to_u128(value: f64) -> ZkResult<u128> {
    if !value.is_finite() || value < 0.0 || value > u128::MAX as f64 {
        return Err(AkitaError::InvalidInput(
            "cannot convert derived f64 bound to u128".to_string(),
        ));
    }
    Ok(value.ceil() as u128)
}
