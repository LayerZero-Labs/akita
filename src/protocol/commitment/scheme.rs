//! Commitment-scheme trait surface for Hachi protocol code.

use super::transcript_append::AppendToTranscript;
use crate::error::HachiError;
use crate::protocol::hachi_poly_ops::HachiPolyOps;
use crate::protocol::opening_point::BasisMode;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

/// Commitment-scheme interface used by Hachi protocol code.
///
/// Generic over field `F` and cyclotomic ring degree `D`.
/// Caller-provided root polynomials are provided as `impl HachiPolyOps<F, D>`.
/// Recursive `w` witnesses are internal to the protocol and no longer modelled
/// through this trait.
pub trait CommitmentScheme<F, const D: usize>: Clone + Send + Sync + 'static
where
    F: FieldCore + CanonicalField,
{
    /// Prover setup parameters.
    type ProverSetup: Clone + Send + Sync;
    /// Verifier setup parameters.
    type VerifierSetup: Clone + Send + Sync;
    /// Commitment object.
    type Commitment: Clone + PartialEq + Send + Sync + AppendToTranscript<F>;
    /// Evaluation/opening proof object.
    type Proof: Clone + Send + Sync;
    /// Batched same-point evaluation/opening proof object.
    type BatchedProof: Clone + Send + Sync;
    /// Prover-side hint produced for one commitment group.
    type CommitHint: Clone + Send + Sync;
    /// Prover-side hint collection for same-point grouped openings.
    type BatchedCommitHint: Clone + Send + Sync;

    /// Build prover setup for maximum polynomial dimension, batch capacity,
    /// and distinct opening-point count.
    ///
    /// # Panics
    ///
    /// Panics if internal setup fails (programming error, not adversarial input).
    fn setup_prover(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Self::ProverSetup;

    /// Derive verifier setup from prover setup.
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Commit to polynomials.
    ///
    /// The root layout is derived automatically from the polynomial dimension.
    /// All polynomials in `polys` are aggregated into one commitment. Callers
    /// that need multiple commitments should call this method repeatedly, once
    /// per commitment group.
    ///
    /// # Errors
    ///
    /// Returns an error when setup/parameter constraints are not satisfied.
    fn commit<P: HachiPolyOps<F, D>>(
        polys: &[P],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), HachiError>;

    /// Produce an opening proof at `opening_point`.
    ///
    /// The root layout is derived from `opening_point.len()`. Recursive
    /// w-opening levels derive their own layouts internally.
    ///
    /// `basis` selects the polynomial representation (see [`BasisMode`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the opening point is invalid or proof generation fails.
    #[allow(clippy::too_many_arguments)]
    fn prove<T: Transcript<F>, P: HachiPolyOps<F, D>>(
        setup: &Self::ProverSetup,
        poly: &P,
        opening_point: &[F],
        hint: Self::CommitHint,
        transcript: &mut T,
        commitment: &Self::Commitment,
        basis: BasisMode,
    ) -> Result<Self::Proof, HachiError>;

    /// Produce a fused batched opening proof for one or more opening points.
    ///
    /// The outer slice indexes opening points. For each point, the prover
    /// receives grouped batches: `poly_groups_by_point[j][g]` is one commitment
    /// group at point `j`.
    ///
    /// Same-point batching is the special case `opening_points.len() == 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if any opening point is invalid or proof generation
    /// fails.
    #[allow(clippy::too_many_arguments)]
    fn batched_prove<T: Transcript<F>, P: HachiPolyOps<F, D>>(
        setup: &Self::ProverSetup,
        poly_groups_by_point: &[&[&[P]]],
        opening_points: &[&[F]],
        hints_by_point: Vec<Self::BatchedCommitHint>,
        transcript: &mut T,
        commitments_by_point: &[&[Self::Commitment]],
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, HachiError>;

    /// Verify an opening proof.
    ///
    /// The root layout is derived deterministically from `opening_point.len()`.
    ///
    /// `basis` must match the mode used by the prover (see [`BasisMode`]).
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    #[allow(clippy::too_many_arguments)]
    fn verify<T: Transcript<F>>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_point: &[F],
        opening: &F,
        commitment: &Self::Commitment,
        basis: BasisMode,
    ) -> Result<(), HachiError>;

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
    fn batched_verify<T: Transcript<F>>(
        proof: &Self::BatchedProof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_points: &[&[F]],
        opening_groups_by_point: &[&[&[F]]],
        commitments_by_point: &[&[Self::Commitment]],
        basis: BasisMode,
    ) -> Result<(), HachiError>;

    /// Protocol identifier.
    fn protocol_name() -> &'static [u8];
}
