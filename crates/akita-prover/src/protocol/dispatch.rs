//! Runtime-to-const-generic dispatch for ring dimension D.
//!
//! The supported D values (all powers of 2 that admit a CRT+NTT decomposition)
//! are: 32, 64, 128, 256.

/// Bridge a runtime `d: usize` to a const-generic `D` context.
///
/// Calls `$body` with the matched const `D`. Inside `$body`, `D` is available
/// as a const generic parameter (via the generated function).
///
/// # Supported dimensions
///
/// 32, 64, 128, 256.
///
/// # Panics
///
/// Panics at runtime if `d` is not one of the supported values.
///
/// # Examples
///
/// ```
/// use akita_prover::dispatch_ring_dim;
/// let ring_dim: usize = 256;
/// let result = dispatch_ring_dim!(ring_dim, |D| D * 2);
/// assert_eq!(result, 512);
/// ```
#[macro_export]
macro_rules! dispatch_ring_dim {
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
            _ => panic!("unsupported ring dimension: {__d}"),
        }
    }};
}

/// Bridge a runtime `d: usize` to a const-generic `D` context, returning an
/// [`AkitaError`](akita_field::AkitaError) for unsupported dimensions.
#[macro_export]
macro_rules! dispatch_ring_dim_result {
    (
        $d:expr,
        $current_D:ident,
        $current_prepared:expr,
        |$D:ident, $prepared:ident| $body:block,
        $prepare:expr
    ) => {{
        let __d = $d;
        if __d == $current_D {
            let $prepared = $current_prepared;
            $body
        } else {
            match __d {
                32 => {
                    const $D: usize = 32;
                    let __prepared = $prepare;
                    let $prepared = &__prepared;
                    $body
                }
                64 => {
                    const $D: usize = 64;
                    let __prepared = $prepare;
                    let $prepared = &__prepared;
                    $body
                }
                128 => {
                    const $D: usize = 128;
                    let __prepared = $prepare;
                    let $prepared = &__prepared;
                    $body
                }
                256 => {
                    const $D: usize = 256;
                    let __prepared = $prepare;
                    let $prepared = &__prepared;
                    $body
                }
                _ => Err(akita_field::AkitaError::InvalidInput(format!(
                    "unsupported ring dimension: {__d}"
                ))),
            }
        }
    }};

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

/// The set of supported ring dimensions for [`dispatch_ring_dim!`].
pub const SUPPORTED_RING_DIMS: &[usize] = &[32, 64, 128, 256];

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    #[test]
    fn dispatch_ring_dim_basic() {
        for &d in super::SUPPORTED_RING_DIMS {
            let result = dispatch_ring_dim!(d, |D| D);
            assert_eq!(result, d);
        }
    }

    #[test]
    #[should_panic(expected = "unsupported ring dimension")]
    fn dispatch_ring_dim_unsupported_panics() {
        let _ = dispatch_ring_dim!(42, |D| D);
    }

    #[test]
    fn dispatch_ring_dim_result_rejects_unsupported_dimension() {
        let result: Result<usize, akita_field::AkitaError> =
            dispatch_ring_dim_result!(42, |D| Ok(D));
        assert!(matches!(
            result,
            Err(akita_field::AkitaError::InvalidInput(message))
                if message.contains("unsupported ring dimension: 42")
        ));
    }
}
