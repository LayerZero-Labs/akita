//! NTT-friendly small-prime arithmetic and CRT helpers.

pub mod butterfly;
pub mod crt;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub mod prime;
#[cfg(target_arch = "x86_64")]
pub(crate) mod sse41;
pub mod tables;

pub use butterfly::NttTwiddles;
pub use crt::{GarnerData, LimbQ, RADIX_BITS};
pub use prime::{MontCoeff, NttPrime, PrimeWidth};
