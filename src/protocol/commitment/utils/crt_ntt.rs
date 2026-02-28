//! Protocol-facing CRT+NTT parameter dispatch.

use crate::algebra::ntt::tables::{
    q128_primes, q64_primes, MAX_CRT_RING_DEGREE, Q128_MODULUS, Q128_NUM_PRIMES, Q32_MODULUS,
    Q32_NUM_PRIMES, Q32_PRIMES, Q64_MODULUS, Q64_NUM_PRIMES, RING_DEGREE,
};
use crate::algebra::ring::CrtNttParamSet;
use crate::error::HachiError;
use crate::CanonicalField;

use super::norm::detect_field_modulus;

/// Supported protocol CRT+NTT parameter families.
pub(crate) enum ProtocolCrtNttParams<const D: usize> {
    Q32(CrtNttParamSet<i16, Q32_NUM_PRIMES, D>),
    Q64(CrtNttParamSet<i32, Q64_NUM_PRIMES, D>),
    Q128(CrtNttParamSet<i32, Q128_NUM_PRIMES, D>),
}

/// Select a CRT+NTT parameter set from field modulus and ring degree.
///
/// Dispatch policy:
/// - `q <= 2^32-99` and `D <= 64`: Q32 (`i16`)
/// - `q <= 2^64-59` and `D <= 1024`: Q64 (`i32`, conservative K=5)
/// - `q == 2^128-275` and `D <= 1024`: Q128 (`i32`, K=5)
/// - otherwise: explicit setup error
pub(crate) fn select_crt_ntt_params<F: CanonicalField, const D: usize>(
) -> Result<ProtocolCrtNttParams<D>, HachiError> {
    if !D.is_power_of_two() {
        return Err(HachiError::InvalidSetup(format!(
            "CRT+NTT requires power-of-two ring degree, got D={D}"
        )));
    }
    if D > MAX_CRT_RING_DEGREE {
        return Err(HachiError::InvalidSetup(format!(
            "CRT+NTT supports D <= {MAX_CRT_RING_DEGREE}, got D={D}"
        )));
    }

    let modulus = detect_field_modulus::<F>();

    if modulus == Q128_MODULUS {
        return Ok(ProtocolCrtNttParams::Q128(CrtNttParamSet::new(
            q128_primes(),
        )));
    }

    if modulus <= Q32_MODULUS as u128 {
        if D <= RING_DEGREE {
            return Ok(ProtocolCrtNttParams::Q32(CrtNttParamSet::new(Q32_PRIMES)));
        }
        return Ok(ProtocolCrtNttParams::Q64(CrtNttParamSet::new(q64_primes())));
    }

    if modulus <= Q64_MODULUS as u128 {
        return Ok(ProtocolCrtNttParams::Q64(CrtNttParamSet::new(q64_primes())));
    }

    Err(HachiError::InvalidSetup(format!(
        "no CRT+NTT parameter set for modulus {modulus} and D={D}; supported ranges: <= {Q64_MODULUS} (with Q32/Q64 dispatch) or exactly {Q128_MODULUS}"
    )))
}
