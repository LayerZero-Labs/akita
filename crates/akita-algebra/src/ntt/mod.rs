//! NTT-friendly small-prime arithmetic and CRT helpers.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx;
mod batched_four_point_policy;
pub mod butterfly;
pub mod crt;
mod digit_validation;
pub mod ifma52;
#[cfg(target_arch = "aarch64")]
pub mod neon;
mod plan;
pub mod prime;
pub mod tables;

pub(crate) use batched_four_point_policy::batched_four_point_eligible;
pub use butterfly::NttTwiddles;
pub use crt::{CrtCapacity, GarnerData, LimbQ, RADIX_BITS};
pub use digit_validation::i16_values_in_balanced_range;
pub use plan::NttKernelPlan;
pub use prime::{MontCoeff, NttPrime, PrimeWidth};
