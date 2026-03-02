//! Commitment-scheme trait surface for Hachi protocol code.

use super::config::{CommitmentConfig, HachiCommitmentLayout};
use super::transcript_append::AppendToTranscript;
use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, Polynomial};

/// Output type for batched commitments.
pub(crate) type BatchCommitOutput<C, H> = Result<Vec<(C, H)>, HachiError>;

/// Witness data produced alongside a ring-native commitment.
///
/// Contains the commitment itself plus `t_hat` (basis-decomposed inner Ajtai
/// output) from the two-layer Ajtai construction (§4.1). The decomposed input
/// vectors `s` are NOT stored; they are recomputed from `ring_coeffs` during
/// proving to avoid multi-GB memory usage at production parameters.
pub struct CommitWitness<C, F: FieldCore, const D: usize> {
    /// The ring commitment (outer Ajtai output `u = B · t̂`).
    pub commitment: C,
    /// Per-block basis-decomposed inner Ajtai output vectors.
    pub t_hat: Vec<Vec<CyclotomicRing<F, D>>>,
}

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
    ///
    /// # Panics
    ///
    /// Panics if internal setup fails (programming error, not adversarial input).
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
        commitment: &Self::Commitment,
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

    /// Construct commitment setup for at most `max_num_vars` variables.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent with `Cfg`.
    fn setup(max_num_vars: usize) -> Result<(Self::ProverSetup, Self::VerifierSetup), HachiError>;

    /// Read the runtime layout carried by `setup`.
    ///
    /// # Errors
    ///
    /// Returns an error when setup metadata is inconsistent.
    fn layout(setup: &Self::ProverSetup) -> Result<HachiCommitmentLayout, HachiError>;

    /// Commit to ring blocks arranged as `2^R` vectors of length `2^M`.
    ///
    /// Returns `(commitment, s, t_hat)` where `s` and `t_hat` are the
    /// decomposed witness vectors from §4.1.
    ///
    /// # Errors
    ///
    /// Returns an error if block layout mismatches config or commitment fails.
    fn commit_ring_blocks(
        f_blocks: &[Vec<CyclotomicRing<F, D>>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError>;

    /// Commit to a flat coefficient table `(f_i)_{i∈{0,1}^ℓ}` in ring form.
    ///
    /// The input uses sequential block layout: ring elements
    /// `[0, block_len)` form block 0, `[block_len, 2*block_len)` form
    /// block 1, and so on. This matches the sequential variable ordering
    /// where M variables (position in block) are lower-order and R variables
    /// (block selection) are higher-order.
    ///
    /// # Errors
    ///
    /// Returns an error if `f_coeffs.len()` does not match the configured block
    /// layout or if the underlying commitment routine fails.
    fn commit_coeffs(
        f_coeffs: &[CyclotomicRing<F, D>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let layout = Self::layout(setup)?;
        let num_blocks = layout.num_blocks;
        let block_len = layout.block_len;
        let expected_len = num_blocks
            .checked_mul(block_len)
            .ok_or_else(|| HachiError::InvalidSetup("coefficient length overflow".to_string()))?;
        if f_coeffs.len() != expected_len {
            return Err(HachiError::InvalidSize {
                expected: expected_len,
                actual: f_coeffs.len(),
            });
        }

        let blocks: Vec<Vec<CyclotomicRing<F, D>>> = f_coeffs
            .chunks_exact(block_len)
            .map(|chunk| chunk.to_vec())
            .collect();

        Self::commit_ring_blocks(&blocks, setup)
    }

    /// Commit to a regular one-hot witness.
    ///
    /// The witness represents `T` chunks of `onehot_k` field elements, each
    /// chunk containing exactly one 1 and all other entries 0. `indices[c]`
    /// gives the hot position in chunk `c` (must be in `[0, onehot_k)`).
    ///
    /// Requires `D` and `onehot_k` to be "nicely matched": one must divide
    /// the other.
    ///
    /// The default implementation materializes the full one-hot field vector,
    /// packs it into ring elements via coefficient embedding, and delegates
    /// to `commit_coeffs`. Implementations may override this with a
    /// sparse-aware path that avoids all inner ring multiplications.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent or any index is out
    /// of range.
    fn commit_onehot(
        onehot_k: usize,
        indices: &[Option<usize>],
        setup: &Self::ProverSetup,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError> {
        let num_chunks = indices.len();
        let total_field_elems = num_chunks
            .checked_mul(onehot_k)
            .ok_or_else(|| HachiError::InvalidInput("T*K overflow".into()))?;
        if total_field_elems % D != 0 {
            return Err(HachiError::InvalidInput(format!(
                "T*K={total_field_elems} is not divisible by D={D}"
            )));
        }

        // Materialize the full one-hot vector as ring elements.
        let total_ring_elems = total_field_elems / D;
        let mut ring_coeffs = vec![CyclotomicRing::<F, D>::zero(); total_ring_elems];
        for (c, opt) in indices.iter().enumerate() {
            let Some(&idx) = opt.as_ref() else { continue };
            if idx >= onehot_k {
                return Err(HachiError::InvalidInput(format!(
                    "index {idx} out of range for chunk size K={onehot_k} at position {c}"
                )));
            }
            let field_pos = c * onehot_k + idx;
            let ring_idx = field_pos / D;
            let coeff_idx = field_pos % D;
            ring_coeffs[ring_idx].coeffs[coeff_idx] = F::one();
        }

        Self::commit_coeffs(&ring_coeffs, setup)
    }
}
