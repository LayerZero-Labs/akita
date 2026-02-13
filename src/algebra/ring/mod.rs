//! Cyclotomic ring types and NTT representations.

pub mod crt_ntt_repr;
pub mod cyclotomic;

pub use crt_ntt_repr::{CrtNttConvertibleField, CyclotomicCrtNtt};
pub use cyclotomic::CyclotomicRing;
