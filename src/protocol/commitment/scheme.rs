//! Commitment-scheme trait surface for Hachi protocol code.

use super::config::CommitmentConfig;
use super::transcript_append::AppendToTranscript;
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, Polynomial};

/// Output type for batched commitments.
pub(crate) type BatchCommitOutput<C, H> = Result<Vec<(C, H)>, HachiError>;

/// Generic commitment-scheme interface used by Hachi protocol code.
pub trait CommitmentScheme<F>: Clone + Send + Sync + 'static
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
    /// Optional prover-side hint produced at commitment time.
    type OpeningProofHint: Clone + Send + Sync;

    /// Build prover setup for maximum polynomial dimension.
    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup;

    /// Derive verifier setup from prover setup.
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Commit to one polynomial.
    ///
    /// # Errors
    ///
    /// Returns an error when setup/parameter constraints are not satisfied.
    fn commit<P: Polynomial<F>>(
        poly: &P,
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::OpeningProofHint), HachiError>;

    /// Commit to many polynomials.
    ///
    /// # Errors
    ///
    /// Returns an error if any per-polynomial commitment fails.
    fn batch_commit<P: Polynomial<F>>(
        polys: &[P],
        setup: &Self::ProverSetup,
    ) -> BatchCommitOutput<Self::Commitment, Self::OpeningProofHint> {
        polys.iter().map(|p| Self::commit(p, setup)).collect()
    }

    /// Produce an opening proof at `opening_point`.
    ///
    /// # Errors
    ///
    /// Returns an error if the opening point is invalid or proof generation fails.
    fn prove<T: Transcript<F>, P: Polynomial<F>>(
        setup: &Self::ProverSetup,
        poly: &P,
        opening_point: &[F],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut T,
    ) -> Result<Self::Proof, HachiError>;

    /// Verify an opening proof.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    fn verify<T: Transcript<F>>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_point: &[F],
        opening: &F,
        commitment: &Self::Commitment,
    ) -> Result<(), HachiError>;

    /// Homomorphic commitment combination.
    fn combine_commitments(commitments: &[Self::Commitment], coeffs: &[F]) -> Self::Commitment;

    /// Homomorphic hint combination.
    fn combine_hints(hints: Vec<Self::OpeningProofHint>, coeffs: &[F]) -> Self::OpeningProofHint;

    /// Protocol identifier.
    fn protocol_name() -> &'static [u8];
}

/// Streaming extension for chunked commitment workflows.
pub trait StreamingCommitmentScheme<F>: CommitmentScheme<F>
where
    F: FieldCore + CanonicalField,
{
    /// Intermediate chunk state.
    type ChunkState: Clone + Send + Sync + PartialEq + std::fmt::Debug;

    /// Process one chunk of field elements.
    fn process_chunk(setup: &Self::ProverSetup, chunk: &[F]) -> Self::ChunkState;

    /// Process one chunk of one-hot values.
    fn process_chunk_onehot(
        setup: &Self::ProverSetup,
        onehot_k: usize,
        chunk: &[Option<usize>],
    ) -> Self::ChunkState;

    /// Aggregate chunk states into one commitment + hint.
    fn aggregate_chunks(
        setup: &Self::ProverSetup,
        onehot_k: Option<usize>,
        chunks: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint);
}

/// Ring-native commitment interface for §4.1 implementation work.
pub trait RingCommitmentScheme<F, const D: usize, Cfg>: Clone + Send + Sync + 'static
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    /// Prover setup parameters.
    type ProverSetup: Clone + Send + Sync;
    /// Verifier setup parameters.
    type VerifierSetup: Clone + Send + Sync;
    /// Ring-native commitment type.
    type Commitment: Clone + PartialEq + Send + Sync;
    /// Opening witness type returned by `commit_ring_blocks`.
    type Opening: Clone + Send + Sync;

    /// Construct commitment setup for at most `max_num_vars` variables.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent with `Cfg`.
    fn setup(max_num_vars: usize) -> Result<(Self::ProverSetup, Self::VerifierSetup), HachiError>;

    /// Commit to ring blocks arranged as `2^R` vectors of length `2^M`.
    ///
    /// # Errors
    ///
    /// Returns an error if block layout mismatches config or commitment fails.
    fn commit_ring_blocks(
        f_blocks: &[Vec<CyclotomicRing<F, D>>],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::Opening), HachiError>;
}
