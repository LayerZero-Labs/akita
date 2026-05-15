//! Shared commitment-scheme API contracts.

use crate::{AppendToTranscript, BasisMode};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;

/// Opening-point coordinates used by batched verification inputs.
pub type OpeningPoints<'a, F> = &'a [F];

/// One opening claim: at point `point`, polynomial `poly_indices[k]` of the
/// single committed bundle evaluates to `openings[k]` (`k = 0..l_i`).
#[derive(Debug, Clone)]
pub struct PointClaim<'a, F> {
    /// Opening point coordinates.
    pub point: &'a [F],
    /// Claimed evaluations at `point` (length `l_i`).
    pub openings: &'a [F],
    /// Global polynomial indices into the single committed bundle (length
    /// `l_i`). Each value must be in `[0, num_committed_polys)`.
    pub poly_indices: Vec<usize>,
}

impl<'a, F> PointClaim<'a, F> {
    /// Construct a claim from explicit pieces.
    pub fn new(point: &'a [F], openings: &'a [F], poly_indices: impl Into<Vec<usize>>) -> Self {
        Self {
            point,
            openings,
            poly_indices: poly_indices.into(),
        }
    }

    /// Construct a claim that opens polynomials `0..openings.len()` of the
    /// committed bundle in order. Convenience for the common
    /// "single-commitment, all-polynomials, one-point" shape.
    pub fn all(point: &'a [F], openings: &'a [F]) -> Self {
        Self::new(point, openings, (0..openings.len()).collect::<Vec<_>>())
    }
}

/// Verifier input for a fused batched opening.
///
/// All claims share a single underlying commitment over the bundle of
/// polynomials. Each [`PointClaim`] picks out a subset of those polynomials by
/// global index and gives their claimed evaluations at one opening point.
#[derive(Debug, Clone)]
pub struct VerifierClaims<'a, F, C> {
    /// The single commitment over the entire polynomial bundle.
    pub commitment: &'a C,
    /// Per-opening-point claims.
    pub points: Vec<PointClaim<'a, F>>,
}

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
    /// A "singleton" opening is the 1x1 special case: a single polynomial
    /// opened at a single point.
    type BatchedProof: Clone + Send + Sync;

    /// Verify a fused batched opening proof over one or more opening points.
    ///
    /// The root layout is derived deterministically from the opening points.
    ///
    /// Same-point batching is the special case `claims.points.len() == 1`.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    fn batched_verify<'a, T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        claims: VerifierClaims<'a, Self::ClaimField, Self::Commitment>,
        basis: BasisMode,
    ) -> Result<(), AkitaError>;

    /// Protocol identifier.
    fn protocol_name() -> &'static [u8];
}
