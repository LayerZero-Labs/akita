//! Conditional parallelism utilities.
//!
//! When the `parallel` feature is enabled, the `cfg_iter!` family of macros
//! expand to rayon's parallel iterators. Otherwise they fall back to standard
//! sequential iterators.

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

/// Returns `.par_iter()` when `parallel` is enabled, `.iter()` otherwise.
#[macro_export]
macro_rules! cfg_iter {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.par_iter();
        #[cfg(not(feature = "parallel"))]
        let it = $e.iter();
        it
    }};
}

/// Returns `.par_iter_mut()` when `parallel` is enabled, `.iter_mut()` otherwise.
#[macro_export]
macro_rules! cfg_iter_mut {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.par_iter_mut();
        #[cfg(not(feature = "parallel"))]
        let it = $e.iter_mut();
        it
    }};
}

/// Returns `.into_par_iter()` when `parallel` is enabled, `.into_iter()` otherwise.
#[macro_export]
macro_rules! cfg_into_iter {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let it = $e.into_iter();
        it
    }};
}

/// Returns `.par_chunks(n)` when `parallel` is enabled, `.chunks(n)` otherwise.
#[macro_export]
macro_rules! cfg_chunks {
    ($e:expr, $n:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.par_chunks($n);
        #[cfg(not(feature = "parallel"))]
        let it = $e.chunks($n);
        it
    }};
}
