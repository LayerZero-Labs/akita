//! Pure protocol description for the Akita sumcheck stack.
//!
//! This crate holds only protocol *description*: the concrete identifier types,
//! per-level-stage formula constructors that build concrete sumcheck descriptors,
//! and the per-level protocol plan (instances, batching, carried openings, and
//! the transcript schedule) with its feature gates. Every building block (the
//! `Source`/`Term`/`Expr`/`SumcheckInstanceDescriptor` algebra and the generic
//! descriptor evaluator) is imported from `akita-sumcheck`; this crate holds no
//! engine code.
//!
//! **Vocabulary:** use named **level stages** (norm, fold, setup), not stage
//! numbers. A norm stage contains **norm tree nodes** (each an eq-factored
//! sumcheck; legacy `AkitaStage1Proof::stages` becomes `::nodes`).
//! [`plan::StagePlan`] is a **scheduled sumcheck batch**, not a level stage.
//! See [`naming`] and
//! `specs/akita-sumcheck-level-naming.md`.
//!
//! The same description is consumed by both sides of the protocol. The verifier
//! evaluates a stage descriptor via the panic-free
//! `SumcheckInstanceDescriptor::try_evaluate` helper; the prover walks the same
//! descriptor over witness oracles. Because both sides build their plan from
//! [`plan::plan_level`], a pure function of `(const D, LevelParams, next_w_len,
//! ProtocolGates)`, the Fiat-Shamir ordering, batching, and per-instance proof
//! format agree by construction.
//!
//! Verifier-reachable: every entry point here is fallible and panic-free.

pub mod ids;
pub mod naming;
pub mod plan;
pub mod stage2;

pub use ids::{AkitaChallengeId, AkitaOpeningId, AkitaPublicId};
pub use plan::{
    plan_level, BatchingScheme, CarriedOpeningPlan, LevelProtocolPlan, LevelRole, ProtocolGates,
    StagePlan, TranscriptEvent, TranscriptSchedule,
};
pub use stage2::{
    matches_stage2_intermediate_descriptor, stage2_descriptor, stage2_relation_subclaim,
    stage2_summand, stage2_virtual_subclaim, AkitaSubClaim, AkitaSumcheckDescriptor, AkitaSummand,
};
