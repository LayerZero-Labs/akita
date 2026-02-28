//! NTT-friendly small-prime arithmetic and CRT helpers.

pub mod butterfly;
pub mod crt;
pub mod prime;
pub mod tables;

pub use butterfly::NttTwiddles;
pub use crt::{LimbQ, QData, RADIX_BITS};
pub use prime::{MontCoeff, NttPrime};
