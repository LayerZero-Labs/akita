//! Prover-side commitment-scheme trait surface for Akita protocol code.

use crate::compute::{
    AkitaRootPoly, CommitmentComputeBackend, ProverComputeBackend, RootCommitKernel,
    RootCommitSource, RootTensorSource, TensorProjectionKernel,
};
use crate::{AkitaPolyOps, ProverClaims, RootTensorProjectionPoly};
use akita_field::unreduced::HasWide;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_transcript::Transcript;
use akita_types::{BasisMode, RingSubfieldEncoding, SetupContributionMode};

/// Prover-side commitment-scheme interface used by Akita protocol code.
///
/// Generic over base field `F` and cyclotomic ring degree `D`.
/// Caller-provided root polynomials are provided as `impl AkitaRootPoly<F, D>`
/// for commit paths; prove still accepts `impl AkitaPolyOps<F, D>` until the
/// flow cutover lands.
/// Recursive `w` witnesses are internal to the protocol and no longer modelled
/// through this trait.
pub trait CommitmentProver<F, const D: usize>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
{
    /// Prover setup parameters.
    type ProverSetup: Clone + Send + Sync;
    /// Verifier setup derived from prover setup.
    type VerifierSetup: Clone + Send + Sync;
    /// Commitment object produced by the scheme.
    type Commitment: Clone + Send + Sync;
    /// Public opening point and claimed-evaluation field.
    type ClaimField: ExtField<F>;
    /// Extension field used for root tensor projection during commit.
    type TensorField: ExtField<F> + RingSubfieldEncoding<F> + 'static;
    /// Prover-side hint produced for one opening-point commitment.
    type CommitHint: Clone + Send + Sync;
    /// Batched proof object produced by the scheme.
    type BatchedProof: Clone + Send + Sync;
    /// Build prover setup for maximum polynomial dimension, batch capacity,
    /// and distinct opening-point count.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested capacity, field tower, or generated
    /// setup is invalid.
    fn setup_prover(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<Self::ProverSetup, AkitaError>;

    /// Derive verifier setup from prover setup.
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Commit a single opening-point bundle.
    ///
    /// All polynomials in `polys` are aggregated into one commitment using a
    /// layout derived from the singleton incidence view. The returned
    /// commitment is compatible with a subsequent `batched_prove` call **only
    /// when this is the sole opening point in that call**. For multipoint
    /// batched proofs callers must use [`Self::batched_commit`] so that every
    /// per-point commitment shares the same root layout the prove path will
    /// select for the full multipoint incidence.
    ///
    /// # Errors
    ///
    /// Returns an error when setup/parameter constraints are not satisfied.
    fn commit<P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        polys: &[P],
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        P: AkitaRootPoly<F, D>,
        B: CommitmentComputeBackend<F>
            + for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>
            + for<'a> TensorProjectionKernel<
                <P as RootTensorSource<F, D>>::TensorView<'a>,
                F,
                Self::TensorField,
                D,
            >
            + for<'a> RootCommitKernel<
                <RootTensorProjectionPoly<F, D> as RootCommitSource<F, D>>::CommitView<'a>,
                F,
                D,
            >;

    /// Commit one polynomial bundle per opening point under a shared root
    /// layout matched to the corresponding multipoint batched prove.
    ///
    /// `polys_per_point[i]` is the bundle that will be opened at opening
    /// point `i` in a subsequent [`Self::batched_prove`] call. Bundle sizes
    /// may differ across points; the implementation must derive its shared
    /// commitment layout from the full multipoint incidence so the produced
    /// commitments are compatible with the prove root.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation, layout selection, or any
    /// per-point commitment fails.
    #[allow(clippy::type_complexity)]
    fn batched_commit<P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        polys_per_point: &[&[P]],
    ) -> Result<Vec<(Self::Commitment, Self::CommitHint)>, AkitaError>
    where
        P: AkitaRootPoly<F, D>,
        B: CommitmentComputeBackend<F>
            + for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>
            + for<'a> TensorProjectionKernel<
                <P as RootTensorSource<F, D>>::TensorView<'a>,
                F,
                Self::TensorField,
                D,
            >
            + for<'a> RootCommitKernel<
                <RootTensorProjectionPoly<F, D> as RootCommitSource<F, D>>::CommitView<'a>,
                F,
                D,
            >;

    /// Produce a fused batched opening proof for one or more opening points.
    ///
    /// The outer vector indexes opening points. Each point carries one
    /// commitment plus the polynomials it bundles.
    ///
    /// A singleton opening is the 1x1 special case (one polynomial, one
    /// commitment, one opening point). Same-point batching is the special
    /// case `opening_points.len() == 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if any opening point is invalid or proof generation
    /// fails.
    #[allow(clippy::too_many_arguments)]
    fn batched_prove<'a, T, P, B>(
        setup: &Self::ProverSetup,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
        claims: ProverClaims<'a, Self::ClaimField, P, Self::Commitment, Self::CommitHint>,
        transcript: &mut T,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<Self::BatchedProof, AkitaError>
    where
        T: Transcript<F>,
        P: AkitaPolyOps<F, D>,
        B: ProverComputeBackend<F>;
}
