//! NTT-friendly small-prime arithmetic and CRT helpers.

pub mod butterfly;
pub mod crt;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod prime;
pub mod tables;
#[cfg(target_arch = "x86_64")]
pub mod x86;

pub use butterfly::NttTwiddles;
pub use crt::{GarnerData, LimbQ, RADIX_BITS};
pub use prime::{MontCoeff, NttPrime, PrimeWidth};
