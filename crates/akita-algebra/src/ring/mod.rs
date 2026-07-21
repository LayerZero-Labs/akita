//! Cyclotomic ring types and NTT representations.

pub mod crt_ntt_repr;
pub mod cyclotomic;
pub mod eval;

pub use crt_ntt_repr::{
    mat_vec_i16_with_tail, CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet,
    CyclotomicCrtNtt, DigitMontLut, I16TailParams,
};
pub use cyclotomic::{CyclotomicRing, WideCyclotomicRing};
pub use eval::{
    eval_flat_ring_at_pows, eval_flat_ring_at_pows_fast, eval_ring_at, eval_ring_at_pows,
    eval_ring_at_pows_fast, scalar_powers,
};
