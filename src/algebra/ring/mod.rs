//! Cyclotomic ring types and NTT representations.

pub mod crt_ntt_repr;
pub mod cyclotomic;
pub mod sparse_challenge;

pub use crt_ntt_repr::{CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt};
pub use cyclotomic::{CyclotomicRing, WideCyclotomicRing};
pub use sparse_challenge::{SparseChallenge, SparseChallengeConfig};
