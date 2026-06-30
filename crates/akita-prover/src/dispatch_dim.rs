//! Runtime ring-dimension → const-generic dispatcher for the prover.
//!
//! [`dispatch_ring_d!`] is the **single source of truth** for the supported
//! ring-dimension set `{32, 64, 128, 256}` inside the prover crate.  Every
//! kernel that is generic over `const D: usize` should be reached through
//! this macro rather than through ad-hoc `match ring_d` blocks scattered
//! across call sites.
//!
//! ## Design
//!
//! The macro expands a runtime `ring_d: usize` into a `match` with one arm
//! per supported dimension.  Each arm introduces a *const* binding `D` so
//! the body can instantiate const-generic functions/types directly:
//!
//! ```rust,ignore
//! dispatch_ring_d!(level.ring_dimension, |D| {
//!     backend.dense_commit_rows::<D>(prepared, plan)
//! })
//! ```
//!
//! The `_` arm returns [`AkitaError::InvalidInput`] — it never panics.
//!
//! ## Usage rules
//!
//! - Invoke **once per kernel-entry boundary** (fold / ring-switch / opening /
//!   tensor / commit).  Do **not** thread `D` through API types or pass it as
//!   a runtime integer across call boundaries; each entry point dispatches
//!   independently.
//! - The body expression must evaluate to `Result<T, AkitaError>` so the
//!   error arm type-checks.  If your kernel returns a plain `T`, wrap it:
//!   `dispatch_ring_d!(d, |D| Ok(my_kernel::<D>(...)))`.

/// Bridge a runtime ring dimension to a const-generic `D` kernel.
///
/// # Syntax
///
/// ```rust,ignore
/// let result: Result<T, AkitaError> = dispatch_ring_d!(ring_d, |D| {
///     some_function_generic_over_d::<D>(args)
/// });
/// ```
///
/// `ring_d` is evaluated once and matched against `{32, 64, 128, 256}`.
/// The matching arm introduces `const D: usize` and evaluates the body
/// expression.  An unrecognised dimension returns
/// `Err(AkitaError::InvalidInput("unsupported ring dimension: …"))`.
///
/// # Errors
///
/// Returns `Err` when `ring_d` is not in `{32, 64, 128, 256}`.
#[macro_export]
macro_rules! dispatch_ring_d {
    ($ring_d:expr, |$D:ident| $body:expr) => {{
        let __ring_d: usize = $ring_d;
        match __ring_d {
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
                "unsupported ring dimension: {__ring_d} \
                 (supported: 32, 64, 128, 256)"
            ))),
        }
    }};
}

#[cfg(test)]
mod tests {
    use akita_field::AkitaError;

    /// Each supported dimension must route to the correct const value.
    #[test]
    fn dispatch_ring_d_routes_to_correct_monomorphization() {
        for d in [32usize, 64, 128, 256] {
            let got: Result<usize, AkitaError> = crate::dispatch_ring_d!(d, |D| Ok(D));
            assert_eq!(
                got.expect("supported dimension"),
                d,
                "arm mismatch for d={d}"
            );
        }
    }

    /// Unsupported values must return `Err`, never panic.
    #[test]
    fn dispatch_ring_d_rejects_unsupported_dimension_48() {
        let err = crate::dispatch_ring_d!(48usize, |D| Ok(D))
            .expect_err("48 is not a supported ring dimension");
        assert!(
            matches!(err, AkitaError::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );
    }

    #[test]
    fn dispatch_ring_d_rejects_zero() {
        let err = crate::dispatch_ring_d!(0usize, |D| Ok(D))
            .expect_err("0 is not a supported ring dimension");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }

    #[test]
    fn dispatch_ring_d_rejects_16() {
        let err = crate::dispatch_ring_d!(16usize, |D| Ok(D))
            .expect_err("16 is not a supported ring dimension");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }

    #[test]
    fn dispatch_ring_d_rejects_512() {
        let err = crate::dispatch_ring_d!(512usize, |D| Ok(D))
            .expect_err("512 is not a supported ring dimension");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }

    /// The body can use D to size a heap allocation (not a fixed-size array,
    /// since different arms produce different array types which can't unify).
    #[test]
    fn dispatch_ring_d_body_can_use_d_for_heap_allocation() {
        let got: Result<Vec<u8>, AkitaError> =
            crate::dispatch_ring_d!(128usize, |D| Ok(vec![0u8; D]));
        assert_eq!(got.expect("128 is supported").len(), 128);
    }
}
