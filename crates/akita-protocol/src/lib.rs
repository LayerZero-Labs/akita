//! Pure protocol description for the Akita sumcheck stack.
//!
//! This crate holds only protocol *description*: the concrete identifier types,
//! the per-stage formula constructors that build concrete sumcheck descriptors,
//! and the per-level protocol plan (instances, batching, carried openings, and
//! the transcript schedule) with its feature gates. Every building block (the
//! `Source`/`Term`/`Expr`/`SumcheckInstanceDescriptor` algebra and the generic
//! descriptor evaluator) is imported from `akita-sumcheck`; this crate holds no
//! engine code.
//!
//! The same description is consumed by both sides of the protocol. The verifier
//! computes its `expected_output_claim` by evaluating a stage descriptor (the
//! generic, panic-free `SumcheckInstanceDescriptor::try_evaluate` helper); the
//! prover runs the same descriptor through its kernel. Because both sides build
//! their plan from [`plan::plan_level`], a pure function of `(LevelParams,
//! ProtocolGates)`, the Fiat-Shamir ordering, batching, and per-instance proof
//! format agree by construction.
//!
//! Verifier-reachable: every entry point here is fallible and panic-free.

pub mod ids;
pub mod plan;
pub mod stage2;

pub use ids::{AkitaChallengeId, AkitaOpeningId, AkitaPublicId};
pub use plan::{
    plan_level, BatchingScheme, CarriedOpeningPlan, L2CertificateMode, LevelProtocolPlan,
    ProtocolGates, StagePlan, TranscriptEvent, TranscriptSchedule,
};
pub use stage2::{stage2_descriptor, stage2_expr, AkitaSumcheckDescriptor};
