//! NTT-friendly small-prime arithmetic and CRT helpers.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx;
pub mod butterfly;
pub mod crt;
pub mod forward_dif_tail_policy;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod prime;
pub mod tables;

pub use butterfly::NttTwiddles;
pub use crt::{GarnerData, LimbQ, RADIX_BITS};
pub use forward_dif_tail_policy::forward_dif_tail_eligible;
pub use prime::{MontCoeff, NttPrime, PrimeWidth};
