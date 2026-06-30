//! Prover-side commitment-scheme trait surface for Akita protocol code.

use crate::compute::ComputeBackendSetup;
use crate::compute::{
    LevelProveStacks, RecursiveProveBackend, RootCommitBackend, RootCommitPoly, RootProvePoly,
    UniformProverStack,
};
use crate::CommittedGroupHandle;
use crate::ProverOpeningBatch;
use crate::ProverTranscriptGrind;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
    RandomSampling,
};
use akita_transcript::Transcript;
use akita_types::{BasisMode, FpExtEncoding, SetupContributionMode};

/// Prover-side commitment-scheme interface used by Akita protocol code.
///
/// Generic over base field `F` and cyclotomic ring degree `D`.
/// Caller-provided root polynomials are source-typed and must satisfy the
/// prover-facing root polynomial traits (`RootProvePoly` and related capability
/// traits).
/// Recursive `w` witnesses are internal to the protocol and no longer modelled
/// through this trait.
pub trait CommitmentProver<F, const D: usize>
where
    F: FieldCore + CanonicalField,
{
    /// Prover setup parameters.
    type ProverSetup: Clone + Send + Sync;
    /// Verifier setup derived from prover setup.
    type VerifierSetup: Clone + Send + Sync;
    /// Commitment object produced by the scheme.
    type Commitment: Clone + Send + Sync;
    /// Public opening point, claimed-evaluation, and proof scalar field.
    type ExtField: ExtField<F>;
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
        max_num_polys_per_commitment_group: usize,
    ) -> Result<Self::ProverSetup, AkitaError>;

    /// Build prover setup for recursive setup-contribution mode.
    ///
    /// # Errors
    ///
    /// Returns an error if base setup construction or recursive setup-prefix
    /// population fails.
    fn setup_prover_recursion(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
    ) -> Result<Self::ProverSetup, AkitaError>;

    /// Derive verifier setup from prover setup.
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Commit a single opening-point bundle.
    ///
    /// All polynomials in `polys` are aggregated into one commitment using a
    /// layout derived from the single shared opening-batch shape.
    ///
    /// # Errors
    ///
    /// Returns an error when setup/parameter constraints are not satisfied.
    fn commit<P, B>(
        setup: &Self::ProverSetup,
        polys: &[P],
        stack: &UniformProverStack<'_, F, B, D>,
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        Self::ExtField: FpExtEncoding<F>,
        P: RootCommitPoly<F, D>,
        B: RootCommitBackend<F, P, Self::ExtField, D>;

    /// Commit the polynomial bundle used by a batched prove.
    ///
    /// The input bundle produces one commitment. All polynomials share one
    /// public opening point in the subsequent [`Self::batched_prove`] call.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation, layout selection, or any
    /// per-point commitment fails.
    fn batched_commit<P, B>(
        setup: &Self::ProverSetup,
        polys: &[P],
        stack: &UniformProverStack<'_, F, B, D>,
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        Self::ExtField: FpExtEncoding<F>,
        P: RootCommitPoly<F, D>,
        B: RootCommitBackend<F, P, Self::ExtField, D>;

    /// Commit one standalone one-hot commitment group with conservative B rank.
    ///
    /// The returned metadata freezes the group layout for a later grouped final
    /// plan. Grouped opening proofs remain unsupported until the next rollout
    /// phase.
    ///
    /// # Errors
    ///
    /// Returns an error if the group is empty, dense, exceeds setup capacity, or
    /// cannot be conservatively planned.
    fn commit_group<P, B>(
        setup: &Self::ProverSetup,
        polys: &[P],
        stack: &UniformProverStack<'_, F, B, D>,
    ) -> Result<CommittedGroupHandle<Self::Commitment, Self::CommitHint>, AkitaError>
    where
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        Self::ExtField: FpExtEncoding<F>,
        P: RootCommitPoly<F, D>,
        B: RootCommitBackend<F, P, Self::ExtField, D>;

    /// Produce a fused batched opening proof for one shared opening point.
    ///
    /// A singleton opening is the 1x1 special case (one polynomial, one
    /// commitment, one opening point).
    ///
    /// # Errors
    ///
    /// Returns an error if any opening point is invalid or proof generation
    /// fails.
    #[allow(clippy::too_many_arguments)]
    fn batched_prove<'a, T, P, B>(
        setup: &Self::ProverSetup,
        claims: ProverOpeningBatch<'a, Self::ExtField, P, F>,
        stacks: &'a impl LevelProveStacks<'a, F, D, Commit = B, Opening = B, Tensor = B, RingSwitch = B>,
        transcript: &mut T,
        basis: BasisMode,
        setup_contribution_mode: SetupContributionMode,
    ) -> Result<Self::BatchedProof, AkitaError>
    where
        T: Transcript<F> + ProverTranscriptGrind<F>,
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
        P: RootProvePoly<F, D>,
        B: RecursiveProveBackend<F, P, Self::ExtField, D> + ComputeBackendSetup<F> + 'a,
        <B as ComputeBackendSetup<F>>::PreparedSetup<D>: 'a;
}
