//! Concrete algebra backends and arithmetic building blocks.
//!
//! This module includes:
//! - Generic prime fields and extensions (`fields`)
//! - Module and polynomial containers (`module`, `poly`)
//! - Low-level NTT and CRT+NTT arithmetic scaffolding (`ntt`)

pub mod backend;
pub mod domains;
pub mod fields;
pub mod module;
pub mod ntt;
pub mod poly;
pub mod ring;

// Flat re-exports for convenience.
pub use backend::{CrtReconstruct, NttPrimeOps, NttTransform, RingBackend, ScalarBackend};
pub use domains::{CoeffDomain, CrtNttDomain};
pub use fields::{
    is_pow2_offset, pow2_offset, pseudo_mersenne_modulus, Fp128, Fp2, Fp2Config, Fp32, Fp4,
    Fp4Config, Fp64, Pow2Offset128Field, Pow2Offset24Field, Pow2Offset32Field, Pow2Offset40Field,
    Pow2Offset48Field, Pow2Offset56Field, Pow2Offset64Field, Pow2OffsetPrimeSpec, Prime128M13M4P0,
    Prime128M13M4P0Params, Prime128M37P3P0, Prime128M37P3P0Params, Prime128M52M3P0,
    Prime128M52M3P0Params, Prime128M54P4P0, Prime128M54P4P0Params, Prime128M8M4M1M0,
    Prime128M8M4M1M0Params, SolinasFp128, SolinasParams, POW2_OFFSET_IMPLEMENTED_MAX_BITS,
    POW2_OFFSET_MAX, POW2_OFFSET_PRIMES, POW2_OFFSET_TABLE, U256,
};
pub use module::VectorModule;
pub use ntt::tables;
pub use ntt::{LimbQ, MontCoeff, NttPrime, QData, RADIX_BITS};
pub use ring::{CrtNttConvertibleField, CyclotomicCrtNtt, CyclotomicRing};
