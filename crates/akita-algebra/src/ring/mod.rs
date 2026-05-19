//! Cyclotomic ring types and NTT representations.

pub mod crt_ntt_cache;
pub mod crt_ntt_repr;
pub mod cyclotomic;
pub mod eval;
pub mod ntt_matvec;
pub mod partial_split_ntt;

pub use crt_ntt_cache::{
    build_ntt_slot, select_crt_ntt_params, NttSlotCache, ProtocolCrtNttParams,
};
pub use crt_ntt_repr::{
    CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut,
};
pub use cyclotomic::{CyclotomicRing, WideCyclotomicRing};
pub use eval::{eval_ring_at, eval_ring_at_pows, scalar_powers, trace};
pub use ntt_matvec::{
    decompose_rows_i8, decompose_rows_i8_into, mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_single_i8,
    mat_vec_mul_ntt_single_i8_cyclic,
};
pub use partial_split_ntt::{
    PackedPartialSplitEval16, PackedPartialSplitNtt16, PartialSplitEval16, PartialSplitNtt16,
};
