//! Prover-side commitment-scheme trait surface for Akita protocol code.

use crate::compute::ComputeBackendSetup;
use crate::compute::{
    LevelProveStacks, RecursiveProveBackend, RuntimeRootCommitBackend, RuntimeRootCommitPoly,
    RuntimeRootProvePoly, UniformProverStack,
};
use crate::ProverOpeningData;
use crate::ProverTranscriptGrind;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
    RandomSampling,
};
use akita_transcript::Transcript;
use akita_types::{BasisMode, FpExtEncoding, PolynomialGroupLayout};

/// Prover-side commitment-scheme interface used by Akita protocol code.
///
/// Generic over base field `F` only. The cyclotomic ring dimension enters at
/// kernel boundaries via schedule-derived dispatch inside the prover; commit
/// and prove methods are D-free and bound on the `Runtime*` capability
/// bundles.
pub trait CommitmentProver<F>
where
    F: FieldCore + CanonicalField,
{
    /// Prover setup parameters.
    type ProverSetup: Clone + Send + Sync;
    /// Verifier setup derived from prover setup.
    type VerifierSetup: Clone + Send + Sync;
    /// Protocol-facing commitment storage.
    type Commitment: Clone + PartialEq + Send + Sync;
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
        stack: &UniformProverStack<'_, F, B>,
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        Self::ExtField: FpExtEncoding<F>,
        P: RuntimeRootCommitPoly<F>,
        B: RuntimeRootCommitBackend<F, P, Self::ExtField>;

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
        stack: &UniformProverStack<'_, F, B>,
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        Self::ExtField: FpExtEncoding<F>,
        P: RuntimeRootCommitPoly<F>,
        B: RuntimeRootCommitBackend<F, P, Self::ExtField>;

    /// Commit the final polynomial bundle for a multi-group root commitment.
    ///
    /// `precommitteds` contains schedule keys for prior commitment groups in
    /// transcript order. The implementation derives the final group shape from
    /// `polys`, freezes precommitted layouts, and resolves the multi-group root
    /// commitment layout internally.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation, multi-group layout selection, or
    /// commitment execution fails.
    fn commit_final_group<P, B>(
        setup: &Self::ProverSetup,
        polys: &[P],
        stack: &UniformProverStack<'_, F, B>,
        precommitteds: Vec<PolynomialGroupLayout>,
    ) -> Result<(Self::Commitment, Self::CommitHint), AkitaError>
    where
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        Self::ExtField: FpExtEncoding<F>,
        P: RuntimeRootCommitPoly<F>,
        B: RuntimeRootCommitBackend<F, P, Self::ExtField>;

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
        claims: ProverOpeningData<'a, Self::ExtField, P, F>,
        stacks: &'a impl LevelProveStacks<'a, F, Commit = B, Opening = B, Tensor = B, RingSwitch = B>,
        transcript: &mut T,
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, AkitaError>
    where
        T: Transcript<F> + ProverTranscriptGrind<F>,
        F: FromPrimitiveInt + HasWide + RandomSampling + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
        P: RuntimeRootProvePoly<F>,
        B: RecursiveProveBackend<F, P, Self::ExtField> + ComputeBackendSetup<F> + 'a,
        <B as ComputeBackendSetup<F>>::PreparedSetup: 'a;
}
