//! Concrete algebra backends and arithmetic building blocks.
//!
//! This module includes:
//! - Generic prime fields and extensions (`fields`)
//! - Module and polynomial containers (`module`, `poly`)
//! - Low-level NTT arithmetic scaffolding (`ntt`)

pub mod fields;
pub mod module;
pub mod ntt;
pub mod poly;
pub mod ring;

// Flat re-exports for convenience.
pub use fields::{Fp128, Fp2, Fp2Config, Fp32, Fp4, Fp4Config, Fp64, U256};
pub use module::VectorModule;
pub use ntt::tables;
pub use ntt::{LimbQ, MontCoeff, NttPrime, QData, RADIX_BITS};
pub use ring::{CyclotomicNtt, CyclotomicRing, NttConvertibleField};
