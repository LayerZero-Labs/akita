//! Recursive witness helpers for later Akita prove levels.
//!
//! Recursive levels carry a flat digit witness that is re-chunked under the
//! current ring dimension on demand.

#![allow(missing_docs, clippy::missing_errors_doc, clippy::missing_panics_doc)]

mod coefficient_packing;
mod handles_and_relation;
mod opening_and_flat;
mod recursive_kernels;
mod stage_sessions;
mod tensor;

#[cfg(test)]
mod tests;

pub(crate) use coefficient_packing::suffix_witness_coefficient_packing_partials;
pub(crate) use handles_and_relation::{
    CpuPreparedOpeningHandle, CpuRelationHandle, CpuWitnessHandle, CpuWitnessOpeningHandle,
    OpaqueRecursiveWitness, RecursiveWitnessFlat,
};
pub(crate) use opening_and_flat::{SuffixWitnessBatchView, SuffixWitnessView};
pub(in crate::opaque) use recursive_kernels::prepare_recursive_witness_opening;
#[cfg(test)]
pub(crate) use stage_sessions::{cpu_extension_opening_session, ConsumerStage2Session};
pub(crate) use stage_sessions::{
    cpu_extension_opening_session_from_witnesses, CpuStage1SessionHandle, CpuStage2SessionHandle,
};
