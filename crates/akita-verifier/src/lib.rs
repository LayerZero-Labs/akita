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
//! These items are kept public solely so a small number of integration tests
//! in `akita-pcs` can exercise specific replay primitives in isolation:
//! [`prepare_relation_matrix_evaluator`], [`RelationMatrixEvaluator`],
//! [`RingSwitchReplay`], and [`AkitaStage1Verifier`]. They are not part of the
//! verifier's intended downstream API.

mod protocol;
mod stages;

pub use protocol::{
    batched_verify, prepare_relation_matrix_evaluator, RelationMatrixEvaluator, RingSwitchReplay,
};
pub use stages::stage1::AkitaStage1Verifier;
