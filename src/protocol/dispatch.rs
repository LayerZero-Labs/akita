//! Runtime-to-const-generic dispatch for ring dimension D.
//!
//! The supported D values (all powers of 2 that admit a CRT+NTT decomposition)
//! are: 64, 128, 256, 512, 1024.

/// Bridge a runtime `d: usize` to a const-generic `D` context.
///
/// Calls `$body` with the matched const `D`. Inside `$body`, `D` is available
/// as a const generic parameter (via the generated function).
///
/// # Supported dimensions
///
/// 64, 128, 256, 512, 1024.
///
/// # Panics
///
/// Panics at runtime if `d` is not one of the supported values.
///
/// # Examples
///
/// ```
/// use hachi_pcs::dispatch_ring_dim;
/// let ring_dim: usize = 256;
/// let result = dispatch_ring_dim!(ring_dim, |D| D * 2);
/// assert_eq!(result, 512);
/// ```
#[macro_export]
macro_rules! dispatch_ring_dim {
    ($d:expr, |$D:ident| $body:expr) => {{
        let __d = $d;
        match __d {
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
            512 => {
                const $D: usize = 512;
                $body
            }
            1024 => {
                const $D: usize = 1024;
                $body
            }
            _ => panic!("unsupported ring dimension: {__d}"),
        }
    }};
}

/// The set of supported ring dimensions for [`dispatch_ring_dim!`].
pub const SUPPORTED_RING_DIMS: &[usize] = &[64, 128, 256, 512, 1024];

/// Returns true if `d` is one of the [`SUPPORTED_RING_DIMS`].
#[inline]
pub fn is_supported_ring_dim(d: usize) -> bool {
    SUPPORTED_RING_DIMS.contains(&d)
}

#[cfg(test)]
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
}
