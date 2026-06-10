//! Prover-side commitment-scheme trait surface for Akita protocol code.

use crate::compute::{
    ProverComputeStack, RootCommitBackend, RootCommitPoly, RootCommitPolys, RootProveFlowBackend,
    RootProvePoly,
};
use crate::ProverClaims;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
    RandomSampling,
};
use akita_transcript::Transcript;
use akita_types::{BasisMode, RingSubfieldEncoding, SetupContributionMode};

/// Prover-side commitment-scheme interface used by Akita protocol code.
///
/// Generic over base field `F` and cyclotomic ring degree `D`.
/// Caller-provided root polynomials use [`RootCommitPoly`] with a backend `B` that
/// implements [`RootCommitBackend`] for that `P`. Prove accepts [`RootProvePoly`]
/// with a backend implementing [`crate::compute::RootProveBackend`] for the same `P`.
/// Recursive `w` witnesses are internal to the protocol and no longer modelled
/// through this trait.
pub trait CommitmentProver<F, const D: usize>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + RandomSampling + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
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
    type TensorField: ExtField<F> + RingSubfieldEncoding<F>;
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

    /// Build prover setup for recursive setup-contribution mode.
    ///
    /// # Errors
    ///
    /// Returns an error if base setup construction or recursive setup-prefix
    /// population fails.
    fn setup_prover_recursion(
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
    /// **Parameter order:** `bundle` precedes `backend` and `prepared` so the
    /// compiler fixes the polynomial type `P` from [`RootCommitPolys`] before
    /// proving `B: RootCommitBackend<F, P, …>`. Pass `&CpuBackend` as
    /// `backend`; do not pin `CpuBackend` in generic commit signatures.
    ///
    /// # Errors
    ///
    /// Returns an error when setup/parameter constraints are not satisfied.
    fn commit<P, B>(
        setup: &Self::ProverSetup,
        bundle: RootCommitPolys<'_, P>,
        backend: &B,
        prepared: &B::PreparedSetup<D>,
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        P: RootCommitPoly<F, D>,
        B: RootCommitBackend<F, P, Self::TensorField, D>;

    /// Commit one polynomial bundle per opening point under a shared root
    /// layout matched to the corresponding multipoint batched prove.
    ///
    /// `polys_per_point[i]` is the bundle that will be opened at opening
    /// point `i` in a subsequent [`Self::batched_prove`] call. Bundle sizes
    /// may differ across points; the implementation must derive its shared
    /// commitment layout from the full multipoint incidence so the produced
    /// commitments are compatible with the prove root.
    ///
    /// Like [`Self::commit`], `polys_per_point` precedes `backend` and
    /// `prepared` so `P` is fixed before `B: RootCommitBackend` is checked.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation, layout selection, or any
    /// per-point commitment fails.
    #[allow(clippy::type_complexity)]
    fn batched_commit<P, B>(
        setup: &Self::ProverSetup,
        polys_per_point: &[&[P]],
        backend: &B,
        prepared: &B::PreparedSetup<D>,
    ) -> Result<Vec<(Self::Commitment, Self::CommitHint)>, AkitaError>
    where
        P: RootCommitPoly<F, D>,
        B: RootCommitBackend<F, P, Self::TensorField, D>;

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
    /// **Parameter order:** `claims` precedes `stack` so the compiler fixes the
    /// polynomial type `P` before proving `B: RootProveFlowBackend<F, P, …>`.
    /// Build a uniform stack with [`ProverComputeStack::uniform`] when every
    /// operation cluster shares one backend and prepared setup.
    #[allow(clippy::too_many_arguments)]
    fn batched_prove<'a, T, P, B>(
        setup: &Self::ProverSetup,
        claims: ProverClaims<'a, Self::ClaimField, P, Self::Commitment, Self::CommitHint>,
        stack: &ProverComputeStack<'a, F, D, B, B, B, B>,
        transcript: &mut T,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<Self::BatchedProof, AkitaError>
    where
        T: Transcript<F>,
        P: RootProvePoly<F, D>,
        B: RootProveFlowBackend<F, P, Self::ClaimField, Self::TensorField, D>;
}
