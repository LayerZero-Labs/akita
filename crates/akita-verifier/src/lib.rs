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
//! Three items are kept public solely so a small number of integration tests
//! in `akita-pcs` can exercise specific replay primitives in isolation:
//! [`prepare_ring_switch_row_eval`], [`RingSwitchDeferredRowEval`], and
//! [`AkitaStage1Verifier`]. They are not part of the verifier's intended
//! downstream API.

mod proof;
mod protocol;
mod stages;

pub use akita_types::{CommitmentVerifier, CommittedOpenings, VerifierClaims};
pub use proof::direct_witness_opening_matches;
pub use protocol::{
    prepare_ring_switch_row_eval, verify_batched_with_policy,
    verify_root_direct_commitments_with_params, RingSwitchDeferredRowEval,
    RootDirectBlindingPayload,
};
pub use stages::AkitaStage1Verifier;

/// Re-exports of the tiered M-row brute-force reference evaluator from
/// `specs/tiered_commit.md` §3. Intended for cross-crate integration
/// tests and as the soundness oracle for the future optimised tiered
/// evaluator (Phase 4b-opt). Not part of the verifier's downstream API.
pub mod tier1_reference {
    pub use crate::protocol::slice_mle::tier1_reference::{
        compute_tier1_and_f_contribution_reference, BPhysicalLayout, Tier1AndFInputs,
    };
}
