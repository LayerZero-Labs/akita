//! Verifier-facing API surface for the Akita PCS.
//!
//! This crate owns the verifier trait and claim shapes that downstream
//! verifier-only consumers need. It deliberately avoids prover polynomial
//! backends, commit hints, recursive witness construction, and planner search.

pub mod ring_switch;
pub mod stage1;
pub mod stage2;

use akita_field::{CanonicalField, FieldCore, HachiError};
use akita_transcript::Transcript;
use akita_types::{AppendToTranscript, BasisMode};

pub use ring_switch::{
    prepare_m_eval, ring_switch_verifier, PreparedMEval, RingSwitchVerifyOutput,
};
pub use stage1::HachiStage1Verifier;
pub use stage2::{relation_claim_from_rows, HachiStage2Verifier, Stage2MEvalSource};

/// Opening-point coordinates used by batched verification inputs.
pub type OpeningPoints<'a, F> = &'a [F];

/// One committed opening group verified at an opening point.
#[derive(Debug, Clone)]
pub struct CommittedOpenings<'a, F, C> {
    /// Claimed openings for the committed polynomial group.
    pub openings: &'a [F],
    /// Commitment for `openings`.
    pub commitment: &'a C,
}

/// Batched verifier input grouped by opening point.
pub type VerifierClaims<'a, F, C> = Vec<(OpeningPoints<'a, F>, Vec<CommittedOpenings<'a, F, C>>)>;

/// Verifier-side commitment-scheme interface used by Akita protocol code.
///
/// Generic over field `F` and cyclotomic ring degree `D`.
///
/// This surface is intentionally proof/claim/setup oriented. It does not name
/// prover polynomial backends or prover-side hints, so verifier-only crates can
/// depend on it without importing commitment/proving machinery.
pub trait CommitmentVerifier<F, const D: usize>: Clone + Send + Sync + 'static
where
    F: FieldCore + CanonicalField,
{
    /// Verifier setup parameters.
    type VerifierSetup: Clone + Send + Sync;
    /// Commitment object.
    type Commitment: Clone + PartialEq + Send + Sync + AppendToTranscript<F>;
    /// Batched (potentially multi-point) evaluation/opening proof object.
    ///
    /// A "singleton" opening is the 1x1 special case: a single polynomial,
    /// a single commitment group, and a single opening point.
    type BatchedProof: Clone + Send + Sync;

    /// Verify a fused batched opening proof over one or more opening points.
    ///
    /// The root layout is derived deterministically from the opening points.
    ///
    /// Same-point batching is the special case `opening_points.len() == 1`.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    #[allow(clippy::too_many_arguments)]
    fn batched_verify<'a, T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: VerifierClaims<'a, F, Self::Commitment>,
        basis: BasisMode,
    ) -> Result<(), HachiError>;

    /// Protocol identifier.
    fn protocol_name() -> &'static [u8];
}
