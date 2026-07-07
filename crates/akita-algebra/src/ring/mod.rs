//! Cyclotomic ring types and NTT representations.

pub mod crt_ntt_repr;
pub mod cyclotomic;
pub mod eval;
pub mod partial_split_ntt;

pub use crt_ntt_repr::{
    CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut,
};
pub use cyclotomic::{CyclotomicRing, WideCyclotomicRing};
pub use eval::{
    eval_flat_ring_at_pows, eval_flat_ring_at_pows_fast, eval_ring_at, eval_ring_at_pows,
    scalar_powers,
};
pub use partial_split_ntt::{
    PackedPartialSplitEval16, PackedPartialSplitNtt16, PartialSplitEval16, PartialSplitNtt16,
};
