mod assembly;
mod coefficient_packing;
mod commitment_material;
mod compression_witness;
mod d_rows;
mod finalize;
mod prepared_openings;
mod relation_quotient;

pub(crate) use assembly::{
    begin_cpu_recursive_witness, finish_cpu_recursive_witness, CpuRecursiveWitnessAssemblyState,
};
#[cfg(test)]
pub(crate) use coefficient_packing::{
    fold_coefficient_packing_group, materialize_coefficient_packing_d_input,
};
pub(crate) use commitment_material::{
    CpuCommitmentMaterial, CpuCommitmentMaterialHandle, OpaqueCompressionState,
    OpaqueInnerRelationState,
};
#[cfg(test)]
pub(crate) use compression_witness::{
    materialize_compression_witness, CompressionSourceId, CompressionSourceWitness,
    CompressionWitnessMaterialization,
};
pub(crate) use finalize::{
    balanced_decompose_centered_i32_i8_into, CpuRecursiveWitnessBuilder,
    CpuRecursiveWitnessUnitPlan,
};
#[cfg(test)]
pub(crate) use finalize::{
    cpu_recursive_witness_build, quotient_decomposition_calls, reset_quotient_decomposition_calls,
    RelationDQuotientWitness, RingRelationGroupWitness, RingRelationWitness,
};
pub(crate) use prepared_openings::{PreparedOpeningWitness, PublicPreparedRelationOpening};
#[cfg(test)]
pub(crate) use relation_quotient::{multi_group_quotient_calls, reset_multi_group_quotient_calls};
