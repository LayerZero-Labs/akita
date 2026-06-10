//! Shared commitment-scheme API contracts.

use crate::{AppendToTranscript, BasisMode, SetupContributionMode};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;

/// Opening-point coordinates used by batched verification inputs.
pub type OpeningPoints<'a, F> = &'a [F];

/// Commitment plus its claimed openings at one opening point.
///
/// Every opening point cites exactly one commitment. The commitment may bundle
/// multiple polynomials, and `openings[i]` is the claimed evaluation of the
/// i-th polynomial in that bundle at the opening point.
#[derive(Debug, Clone)]
pub struct CommittedOpenings<'a, F, C> {
    /// Claimed openings for the bundled polynomials.
    pub openings: &'a [F],
    /// Commitment for `openings`.
    pub commitment: &'a C,
}

/// Batched verifier input: one commitment plus its claimed openings per point.
pub type VerifierClaims<'a, F, C> = Vec<(OpeningPoints<'a, F>, CommittedOpenings<'a, F, C>)>;

/// Verifier-side commitment-scheme interface used by Akita protocol code.
///
/// Generic over base field `F` and cyclotomic ring degree `D`.
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
    /// Public opening point and claimed-evaluation field.
    type ClaimField: ExtField<F>;
    /// Batched (potentially multi-point) evaluation/opening proof object.
    ///
    /// A "singleton" opening is the 1x1 special case: a single polynomial,
    /// a single commitment, and a single opening point.
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
        claims: VerifierClaims<'a, Self::ClaimField, Self::Commitment>,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<(), AkitaError>;

    /// Protocol identifier.
    fn protocol_name() -> &'static [u8];
}
