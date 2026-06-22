//! Shared commitment-scheme API contracts.

use crate::{AppendToTranscript, BasisMode, SetupContributionMode};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;

/// Opening-point coordinates used by batched verification inputs.
pub type OpeningPoints<'a, F> = &'a [F];

/// One PCS commitment and the claimed openings of its bundled polynomials.
///
/// `openings[i]` is the claimed evaluation of `polynomials[i]` at the batch's
/// shared opening point.
#[derive(Debug, Clone)]
pub struct CommittedOpenings<'a, F, C> {
    /// Claimed evaluations for the bundled polynomials at the shared point.
    pub openings: &'a [F],
    /// Commitment covering `openings`.
    pub commitment: &'a C,
}

/// Batched verifier input: one shared opening point plus one commitment bundle.
///
/// Shape: `(shared_point, CommittedOpenings)`.
///
/// # Protocol contract
///
/// - **Single opening point.** All claims in the batch share `shared_point`.
///   To open the same polynomials at different points, run separate prove/verify
///   calls.
/// - **Batched prove/verify.** One commitment object may bundle `N`
///   polynomials, all opened at `shared_point`.
pub type VerifierClaims<'a, F, C> = (OpeningPoints<'a, F>, CommittedOpenings<'a, F, C>);

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
    /// Public opening point, claimed-evaluation, and proof scalar field.
    type ExtField: ExtField<F>;
    /// Batched single-point evaluation/opening proof object.
    ///
    /// A "singleton" opening is the 1x1 special case: a single polynomial,
    /// a single commitment, and a single opening point.
    type BatchedProof: Clone + Send + Sync;

    /// Verify a fused batched opening proof at one shared opening point.
    ///
    /// The root layout and Fiat-Shamir batching are derived from the normalized
    /// [`OpeningBatch`](crate::OpeningBatch) built from `claims` (single shared point, no multipoint).
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    #[allow(clippy::too_many_arguments)]
    fn batched_verify<'a, T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: VerifierClaims<'a, Self::ExtField, Self::Commitment>,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<(), AkitaError>;

    /// Protocol identifier.
    fn protocol_name() -> &'static [u8];
}
