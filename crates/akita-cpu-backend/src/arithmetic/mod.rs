//! CPU-private physical arithmetic, setup caches, and resource controls.

pub(crate) mod backend;
pub(crate) mod coefficient_packing;
mod commitment;
mod commitment_stage;
#[cfg(test)]
mod commitment_tests;
mod compression;
mod compression_cache;
mod compression_stage;
mod controls;
#[cfg(test)]
mod cyclic_rows;
mod digit_rows;
mod exact_i16;
#[cfg(test)]
mod exact_i16_tests;
pub(crate) mod field_reduction;
#[cfg(test)]
mod kernel_tests;
mod prepared;
#[cfg(test)]
mod prepared_tests;
pub(crate) mod requirements;
pub(crate) mod ring_switch;
pub(crate) mod ring_switch_relation;
mod stack;
#[cfg(test)]
mod streamed_tests;

#[cfg(test)]
pub(crate) use backend::CyclicRowsComputeBackend;
pub(crate) use backend::{
    CompressionComputeBackend, CompressionRowsProducts, ComputeBackendSetup,
    DigitRowsComputeBackend, NttCacheOwnerId,
};
pub(crate) use commitment_stage::{CpuInnerCommitOperation, CpuOuterCommitOperation};
pub(crate) use compression_stage::CpuCompressionOperation;
pub(crate) use field_reduction::tensor_pack_recursive_witness;
pub(crate) use prepared::CpuPreparedSetup;
pub use prepared::{PreparedCrtNttProfile, PreparedNttCacheMetric};
pub(crate) use requirements::{
    NttExecutionRequirements, NttOperationCluster, RoutedNttRequirement,
};
pub(crate) use ring_switch_relation::RingSwitchRelationView;
pub(crate) use stack::OperationCtx;

pub(crate) use crate::opaque::CpuBackend;
