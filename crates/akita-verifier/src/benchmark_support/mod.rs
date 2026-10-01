//! Benchmark fixtures that drive production verifier kernels.
//!
//! The evaluation-trace fixture is also compiled for unit tests; the relation
//! evaluator fixture needs the `benchmark-support` feature.

mod evaluation_trace;
#[cfg(feature = "benchmark-support")]
mod relation;

pub use evaluation_trace::{evaluation_trace_benchmark_case, EvaluationTraceBenchmarkCase};
#[cfg(feature = "benchmark-support")]
pub use relation::{
    relation_evaluator_benchmark_case, relation_evaluator_benchmark_case_with_chunks,
    RelationEvaluatorBenchmarkCase,
};
