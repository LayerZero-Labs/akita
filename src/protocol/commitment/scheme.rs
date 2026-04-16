//! Commitment-scheme trait surface for Hachi protocol code.

use super::config::CommitmentConfig;
use super::transcript_append::AppendToTranscript;
use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::hachi_poly_ops::{HachiPolyOps, OneHotIndex};
use crate::protocol::opening_point::BasisMode;
use crate::protocol::params::LevelParams;
use crate::protocol::proof::FlatDigitBlocks;
use crate::protocol::setup::HachiProverSetup;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

/// Witness data produced alongside a ring-native commitment.
///
/// Contains the commitment itself plus `t_hat` (basis-decomposed inner Ajtai
/// output) from the two-layer Ajtai construction (§4.1). The decomposed input
/// vectors `s` are NOT stored; they are recomputed from the polynomial during
/// proving via `HachiPolyOps`.
pub struct CommitWitness<C, F: FieldCore, const D: usize> {
    /// The ring commitment (outer Ajtai output `u = B · t̂`).
    pub commitment: C,
    /// Basis-decomposed inner Ajtai output vectors in flat column-major order
    /// plus block boundaries.
    pub t_hat: FlatDigitBlocks<D>,
    _marker: std::marker::PhantomData<F>,
}

impl<C, F: FieldCore, const D: usize> CommitWitness<C, F, D> {
    /// Construct a new commit witness.
    pub fn new(commitment: C, t_hat: FlatDigitBlocks<D>) -> Self {
        Self {
            commitment,
            t_hat,
            _marker: std::marker::PhantomData,
        }
    }
}

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

    /// Build prover setup for maximum polynomial dimension and batch capacity.
    ///
    /// # Panics
    ///
    /// Panics if internal setup fails (programming error, not adversarial input).
    fn setup_prover(max_num_vars: usize, max_num_batched_polys: usize) -> Self::ProverSetup;

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

/// Ring-native commitment interface for §4.1 implementation work.
pub trait RingCommitmentScheme<F, const D: usize, Cfg>: Clone + Send + Sync + 'static
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig<Field = F>,
{
    /// Ring-native commitment type.
    type Commitment: Clone + PartialEq + Send + Sync;

    /// Read the runtime layout carried by `setup`.
    ///
    /// # Errors
    ///
    /// Returns an error when setup metadata is inconsistent.
    fn layout(setup: &HachiProverSetup<F, D>) -> Result<LevelParams, HachiError>;

    /// Commit to ring blocks arranged as `2^R` vectors of length `2^M`.
    ///
    /// # Errors
    ///
    /// Returns an error if block layout mismatches config or commitment fails.
    fn commit_ring_blocks(
        f_blocks: &[Vec<CyclotomicRing<F, D>>],
        setup: &HachiProverSetup<F, D>,
    ) -> Result<CommitWitness<Self::Commitment, F, D>, HachiError>;

    /// Commit to a flat coefficient table `(f_i)_{i∈{0,1}^ℓ}` in ring form.
    ///
    /// # Errors
    ///
    /// Returns an error if `f_coeffs.len()` does not match the configured block
    /// layout or if the underlying commitment routine fails.
    fn commit_coeffs(
        f_coeffs: &[CyclotomicRing<F, D>],
        setup: &HachiProverSetup<F, D>,
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
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent or any index is out
    /// of range.
    fn commit_onehot<I: OneHotIndex>(
        onehot_k: usize,
        indices: &[Option<I>],
        setup: &HachiProverSetup<F, D>,
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

        let total_ring_elems = total_field_elems / D;
        let mut ring_coeffs = vec![CyclotomicRing::<F, D>::zero(); total_ring_elems];
        for (c, opt) in indices.iter().enumerate() {
            let Some(&idx_raw) = opt.as_ref() else {
                continue;
            };
            let idx = idx_raw.as_usize();
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
