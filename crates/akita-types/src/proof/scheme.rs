//! Shared commitment-scheme API contracts.

use crate::{BasisMode, OpeningClaims};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;
use std::borrow::Cow;

/// Opening-point coordinates used by batched verification inputs.
pub type OpeningPoints<'a, F> = Cow<'a, [F]>;

/// Verifier-side commitment-scheme interface used by Akita protocol code.
///
/// Generic over base field `F` only. Ring dimension is schedule-derived at
/// replay time. Protocol-facing commitments are [`Commitment`](crate::proof::Commitment)
/// wrappers over flat [`RingVec`](crate::proof::RingVec) storage.
///
/// This surface is intentionally proof/claim/setup oriented. It does not name
/// prover polynomial backends or prover-side hints, so verifier-only crates can
/// depend on it without importing commitment/proving machinery.
pub trait CommitmentVerifier<F>: Clone + Send + Sync + 'static
where
    F: FieldCore + CanonicalField,
{
    /// Verifier setup parameters.
    type VerifierSetup: Clone + Send + Sync;
    /// Protocol-facing commitment storage for public claims.
    type Commitment: Clone + PartialEq + Send + Sync;
    /// Batched single-point evaluation/opening proof object.
    ///
    /// A "singleton" opening is the 1x1 special case: a single polynomial,
    /// a single commitment, and a single opening point.
    type BatchedProof: Clone + Send + Sync;
    /// Public opening point, claimed-evaluation, and proof scalar field.
    type ExtField: ExtField<F>;

    /// Verify a fused batched opening proof at one shared opening point.
    ///
    /// The root layout and Fiat-Shamir batching are derived from the normalized
    /// [`OpeningClaims`] built from `claims` (single shared point, no multipoint).
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    fn batched_verify<T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: OpeningClaims<'_, Self::ExtField, &Self::Commitment>,
        basis: BasisMode,
    ) -> Result<(), AkitaError>;

    /// Protocol identifier.
    fn protocol_name() -> &'static [u8];
}
