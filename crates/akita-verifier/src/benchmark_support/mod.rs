//! Benchmark fixtures that drive production verifier kernels.
//!
//! The evaluation-trace fixture and [`coefficient_setup`] are also compiled
//! for unit tests; the relation evaluator fixture needs the
//! `benchmark-support` feature.

mod evaluation_trace;
#[cfg(feature = "benchmark-support")]
mod relation;

use akita_params::FlatMatrix;
use akita_types::{AkitaExpandedSetup, AkitaSetupDescriptor};
use jolt_field::Field;

pub use evaluation_trace::{evaluation_trace_benchmark_case, EvaluationTraceBenchmarkCase};
#[cfg(feature = "benchmark-support")]
pub use relation::{
    relation_evaluator_benchmark_case, relation_evaluator_benchmark_case_with_chunks,
    RelationEvaluatorBenchmarkCase,
};

/// Expanded setup whose matrix is exactly `coefficients`.
pub(crate) fn coefficient_setup<F: Field>(coefficients: Vec<F>) -> AkitaExpandedSetup<F> {
    let descriptor = AkitaSetupDescriptor {
        max_num_vars: 0,
        max_num_batched_polys: 0,
        num_field_elements: coefficients.len(),
        setup_seed: [0; 32].into(),
    };
    // SAFETY: fixtures only evaluate these coefficients; none relies on binding.
    unsafe {
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            descriptor,
            FlatMatrix::from_flat_data(coefficients),
        )
    }
}
