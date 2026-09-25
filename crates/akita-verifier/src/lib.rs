//! Verifier-facing API surface for the Akita PCS.
//!
//! This crate owns verifier replay for already-selected Akita proof schedules.
//! It deliberately avoids prover polynomial backends, commit hints, recursive
//! witness construction, and planner search.
//!
//! Downstream verifier-only integrations should pair this crate with
//! `akita-types` for proof/setup/claim shapes and `akita-config` for concrete
//! runtime schedule policy. The broader `akita-pcs` crate is an umbrella for
//! end-to-end examples and also re-exports prover-facing APIs.
//!
//! # Public surface
//!
//! Only the entry points actually consumed by downstream crates are public.
//! Replay internals (per-level/root verifiers, ring-switch replay, the stage-2
//! verifier, schedule context, prepared-claim shapes) are crate-private. If
//! you need to reach into them, that's a signal to either move the consumer
//! into this crate or expose a narrower entry point.
//!
//! Two replay primitives are public but are not part of the verifier's
//! intended downstream API: [`RelationMatrixEvaluator`], which the
//! `benchmark-support` relation-evaluator bench drives directly, and
//! [`AkitaStage1Verifier`], which downstream range-proof lanes reuse.

mod prepared_cache;
mod protocol;
mod stages;

pub use prepared_cache::build_riscv64_terminal_ntt_cache;
pub use protocol::{batched_verify, RelationMatrixEvaluator};
#[cfg(any(test, feature = "benchmark-support"))]
pub use protocol::{evaluation_trace_benchmark_case, EvaluationTraceBenchmarkCase};
#[cfg(feature = "benchmark-support")]
pub use protocol::{
    relation_evaluator_benchmark_case, relation_evaluator_benchmark_case_with_chunks,
    RelationEvaluatorBenchmarkCase,
};
pub use stages::stage1::AkitaStage1Verifier;
