//! Runtime-to-const-generic dispatch shared by prover and verifier.

use crate::layout::{CommitmentRingDims, RingRole};
use crate::LevelParams;
use akita_field::AkitaError;

/// Validate that a const-generic ring dimension is supported for dispatch.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `D` is zero or not a power of two.
#[inline]
pub fn validate_ring_dispatch<const D: usize>() -> Result<usize, AkitaError> {
    if D == 0 || !D.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "ring dimension must be a non-zero power of two".to_string(),
        ));
    }
    Ok(D.trailing_zeros() as usize)
}

/// Validate that schedule level params match the dispatched A-role (`d_a`).
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `lp.role_dims().d_a() != D`.
#[inline]
pub fn validate_level_dispatch<const D: usize>(lp: &LevelParams) -> Result<usize, AkitaError> {
    validate_role_dispatch::<D>(lp.role_dims(), RingRole::Inner)
}

/// Validate that `dims.dim_for(role) == D` for a kernel dispatch arm.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on dimension mismatch.
#[inline]
pub fn validate_role_dispatch<const D: usize>(
    dims: CommitmentRingDims,
    role: RingRole,
) -> Result<usize, AkitaError> {
    let ring_bits = validate_ring_dispatch::<D>()?;
    if dims.dim_for(role) != D {
        return Err(AkitaError::InvalidSetup(format!(
            "role {:?} ring dimension {} does not match dispatch D={D}",
            role,
            dims.dim_for(role)
        )));
    }
    Ok(ring_bits)
}

/// Bridge a runtime ring dimension to a const-generic `D` context.
///
/// Returns an [`AkitaError`](akita_field::AkitaError) instead of panicking so it
/// is safe to use across verifier-reachable paths.
#[macro_export]
macro_rules! dispatch_ring_dim_result {
    ($d:expr, |$D:ident| $body:expr) => {{
        let __d = $d;
        match __d {
            32 => {
                const $D: usize = 32;
                $body
            }
            64 => {
                const $D: usize = 64;
                $body
            }
            128 => {
                const $D: usize = 128;
                $body
            }
            256 => {
                const $D: usize = 256;
                $body
            }
            _ => Err(akita_field::AkitaError::InvalidInput(format!(
                "unsupported ring dimension: {__d}"
            ))),
        }
    }};
}

#[cfg(test)]
mod tests {
    use akita_field::AkitaError;

    #[test]
    fn dispatch_ring_dim_result_accepts_supported_dimensions() {
        for d in [32usize, 64, 128, 256] {
            let got: Result<usize, AkitaError> = crate::dispatch_ring_dim_result!(d, |D| Ok(D));
            assert_eq!(got.expect("supported ring dimension"), d);
        }
    }

    #[test]
    fn dispatch_ring_dim_result_rejects_unsupported_dimensions() {
        let err: AkitaError = crate::dispatch_ring_dim_result!(16usize, |D| Ok(D))
            .expect_err("unsupported ring dimension must be rejected");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }
}
