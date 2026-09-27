//! Owning CPU execution backend for the Akita PCS.
//!
//! This crate owns prover-side polynomial backends, setup artifacts, recursive
//! witness construction, ring-switch handoff, and Akita-specific sumcheck
//! provers. Config and schedule policy live in `akita-config`.

pub(crate) mod arithmetic;
pub(crate) mod commitment;
pub(crate) mod kernels;
pub(crate) mod opaque;
pub(crate) mod setup;
pub(crate) mod sources;
mod validation;

use akita_algebra::CyclotomicRing;
use akita_types::RingVec;
use jolt_field::Field;

pub(crate) use akita_prover::protocol;
pub use commitment::{
    GroupContext, PortableCompressionState, SetupPrefixProverRegistry, SetupPrefixSlot,
};
pub use opaque::standalone;
pub use opaque::{
    evaluate_root_polynomial, CpuBackend, PreparedCrtNttProfile, PreparedNttCacheMetric,
    RootPolyMeta, RootPolyShape, RootPolynomialEvaluator,
};
pub use opaque::{CommitOutput, CommitmentHandle, CpuSource, SourceHandle};
pub use setup::AkitaProverSetup;
pub use sources::{DensePoly, OneHotIndex, OneHotPoly, OneHotSource};

/// Extension contracts for application-owned CPU polynomial sources.
///
/// Sources imported through this module are still consumed and retained behind
/// the owning backend's opaque handles; only their source-specific streaming
/// kernels remain application-defined.
#[doc(hidden)]
#[allow(missing_docs)]
pub mod custom_source {
    pub use crate::arithmetic::{ComputeBackendSetup, CpuPreparedSetup};
    pub use crate::commitment::{
        cpu_external_inner_commitment_capability, cpu_external_inner_prepared_setup,
        AvailablePolynomialTypes, BackendKindId, CommitSourceClass, CommitSourceDescriptor,
        CommitmentSource, ExternalInnerCommitmentCapability, ExternalInnerCommitmentInput,
        ExternalInnerCommitmentOperation, ExternalOperationIdentity, PolynomialRepresentation,
        PolynomialTypeSelection, PreparedExternalInnerCommitment,
    };
    pub use crate::opaque::{
        aggregate_decompose_fold_witnesses, CommitInnerPlan, CpuFoldResponses,
        DecomposeFoldBatchPlan, DecomposeFoldPlan, DecomposeFoldWitness, OpeningBatchKernel,
        OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
        SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
        SubringCoefficientPackingPlan,
    };
    pub use crate::sources::onehot::OneHotBatchView;
    pub use crate::sources::poly::{
        RootOpeningSource, RootPolyMeta, RootPolyShape, SourceCoefficients,
    };
    pub use crate::sources::poly_helpers::fill_rotated_challenge;
}

/// Arithmetic entry points exercised by the crate's standalone kernel benchmarks.
#[doc(hidden)]
pub mod benchmark_support {
    pub use crate::kernels::linear::{
        decompose_rows_i8_into, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_i8_dense,
        mat_vec_mul_ntt_i8_dense_single_row,
    };
    pub use crate::sources::poly_helpers::balanced_ring_decompose_fold_partitioned;
}

/// Prover-side output of the inner Ajtai commit step.
///
/// Ring dimension is stored by the single flat A-ring buffer. Public commit
/// parameters own the source-block and row boundaries.
pub(crate) fn typed_inner_rows<F: Field, const D: usize>(
    recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
) -> RingVec<F> {
    let coefficient_count = recomposed_inner_rows
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        .checked_mul(D)
        .expect("trusted inner commitment output length must fit usize");
    let mut coefficients = Vec::with_capacity(coefficient_count);
    for block in recomposed_inner_rows {
        for row in block {
            coefficients.extend_from_slice(row.coefficients());
        }
    }
    RingVec::from_coeffs_with_ring_dim(coefficients, D)
        .expect("typed inner commitment rows have valid ring storage")
}

#[cfg(test)]
mod tests;
