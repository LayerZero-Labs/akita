//! Shared commitment-scheme API contracts.

use crate::{AppendToTranscript, BasisMode, OpeningClaims, SetupContributionMode};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;
use std::borrow::Cow;

/// Opening-point coordinates used by batched verification inputs.
pub type OpeningPoints<'a, F> = Cow<'a, [F]>;

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
    /// [`OpeningClaims`] built from `claims` (single shared point, no multipoint).
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    #[allow(clippy::too_many_arguments)]
    fn batched_verify<T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: OpeningClaims<'_, Self::ExtField, &Self::Commitment>,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<(), AkitaError>;

    /// Protocol identifier.
    fn protocol_name() -> &'static [u8];
}
