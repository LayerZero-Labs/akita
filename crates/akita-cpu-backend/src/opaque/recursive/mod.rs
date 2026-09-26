//! Recursive prover-only state for later Akita prove levels.
//!
//! Owns the D-agnostic recursive witness vector `w`, its zero-copy D-specific
//! views, and the setup-prefix source adapter.

mod commit;
pub(in crate::opaque) mod opening;
mod relation_witness;
mod witness;

pub(crate) use crate::opaque::sumcheck::digit_range;

#[cfg(test)]
pub(crate) use digit_range::direct_range_leaf::pad_compact_witness;
pub use digit_range::{DigitRangeProver, LowBasisRangeCheckProver};
pub(crate) use relation_witness::build_w_evals_compact;
pub(crate) use witness::cpu_extension_opening_session_from_witnesses;
pub(crate) use witness::OpaqueRecursiveWitness;
pub(crate) use witness::{
    CpuPreparedOpeningHandle, CpuRelationHandle, CpuStage1SessionHandle, CpuStage2SessionHandle,
    CpuWitnessHandle, CpuWitnessOpeningHandle, RecursiveWitnessFlat,
};

pub(in crate::opaque) use witness::prepare_recursive_witness_opening;

pub(crate) use witness::suffix_witness_coefficient_packing_partials;
