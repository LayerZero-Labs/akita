//! Cyclotomic ring types and NTT representations.

pub mod crt_ntt_repr;
pub mod cyclotomic;
pub mod eval;
pub mod partial_split_ntt;
pub mod sparse_challenge;

pub use crt_ntt_repr::{
    CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut,
};
pub use cyclotomic::{CyclotomicRing, WideCyclotomicRing};
pub use eval::{eval_ring_at, eval_ring_at_pows, scalar_powers, trace};
pub use partial_split_ntt::{
    PackedPartialSplitEval16, PackedPartialSplitNtt16, PartialSplitEval16, PartialSplitNtt16,
};
pub use sparse_challenge::{
    sample_quaternary, sample_ternary, SparseChallenge, SparseChallengeConfig,
};
