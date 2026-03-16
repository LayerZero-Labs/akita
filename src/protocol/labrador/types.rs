//! Core Labrador witness/statement/proof types.

use crate::algebra::ring::CyclotomicRing;
use crate::algebra::SparseChallenge;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::labrador::constraints::LabradorConstraint;
use crate::protocol::labrador::setup::LabradorSetupMatrices;
use crate::{cfg_fold_reduce, CanonicalField, FieldCore};
use std::sync::Arc;

/// Witness object for a Labrador statement, holding the `s_i` row vectors.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LabradorWitness<F: FieldCore, const D: usize> {
    rows: Vec<Vec<CyclotomicRing<F, D>>>,
}

impl<F: FieldCore, const D: usize> LabradorWitness<F, D> {
    /// Build a witness from row vectors, all of which must share the same length.
    ///
    /// # Panics
    ///
    /// Panics if any two rows differ in length.
    pub fn new(rows: Vec<Vec<CyclotomicRing<F, D>>>) -> Self {
        if let Some(first_len) = rows.first().map(|r| r.len()) {
            assert!(
                rows.iter().all(|r| r.len() == first_len),
                "all witness rows must have the same length"
            );
        }
        Self { rows }
    }

    /// Build a witness without asserting uniform row length.
    ///
    /// Use only where the protocol produces rows of mixed length
    /// (e.g. z-decomposition rows plus an auxiliary row).
    pub(crate) fn new_unchecked(rows: Vec<Vec<CyclotomicRing<F, D>>>) -> Self {
        Self { rows }
    }

    /// Borrow the underlying row slices.
    pub fn rows(&self) -> &[Vec<CyclotomicRing<F, D>>] {
        &self.rows
    }
}

impl<F: FieldCore + CanonicalField, const D: usize> LabradorWitness<F, D> {
    /// Squared coefficient norm summed over every ring element in the witness.
    pub fn norm(&self) -> u128 {
        cfg_fold_reduce!(
            (0..self.rows.len()),
            || 0u128,
            |acc, i| {
                let row_sum = self.rows[i]
                    .iter()
                    .map(|ring| ring.coeff_norm_sq())
                    .fold(0u128, |a, v| a.saturating_add(v));
                acc.saturating_add(row_sum)
            },
            |a, b| a.saturating_add(b)
        )
    }
}

/// Compact recipe for the next-level Labrador statement.
///
/// This keeps the dominant recursive structure factored so the next level can
/// aggregate it directly without first materializing a full sparse constraint
/// vector. Explicit constraints are only reconstructed when they are actually
/// needed (for example, at terminal verification).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorReducedConstraintPlan<F: FieldCore, const D: usize> {
    /// Number of virtual input rows reduced at the previous level.
    pub row_count: usize,
    /// Length of each decomposed z-row in the next witness.
    pub max_len: usize,
    /// Reduction parameters that define the next witness layout.
    pub config: LabradorReductionConfig,
    /// Amortization challenges from the previous level.
    pub challenges: Vec<SparseChallenge>,
    /// Amortized `sum_i c_i * phi_i` relation carried into the next level
    /// (formerly `combined_phi`).
    pub amortized_phi: Vec<CyclotomicRing<F, D>>,
    /// Aggregated right-hand side for the diagonal relation
    /// (formerly `b_total`).
    pub aggregated_rhs: CyclotomicRing<F, D>,
    /// Commitment matrices needed to replay the reduced statement.
    pub setup: Arc<LabradorSetupMatrices<F, D>>,
}

/// Public statement reduced to Labrador recursion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorStatement<F: FieldCore, const D: usize> {
    /// Opening-side payload for the current round (formerly `u1`).
    ///
    /// This is an outer commitment in standard rounds and the raw opening-side
    /// digits in tail mode.
    pub inner_opening_payload: Vec<CyclotomicRing<F, D>>,
    /// Linear-garbage-side payload for the current round (formerly `u2`).
    ///
    /// This is an outer commitment in standard rounds and the raw
    /// linear-garbage digits in tail mode.
    pub linear_garbage_payload: Vec<CyclotomicRing<F, D>>,
    /// Amortization challenges (per input witness row).
    pub challenges: Vec<SparseChallenge>,
    /// Sparse constraints checked by reducer/verifier.
    pub constraints: Vec<LabradorConstraint<F, D>>,
    /// Compact recursive statement representation used between Labrador levels.
    pub reduced_constraints: Option<Box<LabradorReducedConstraintPlan<F, D>>>,
    /// Squared witness norm bound (formerly `beta_sq`).
    pub witness_norm_bound_sq: u128,
}

/// Per-level reduction parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabradorReductionConfig {
    /// Number of witness-side digit parts (formerly `f`).
    pub witness_digit_parts: usize,
    /// Bit width of each witness-side digit (formerly `b`).
    pub witness_digit_bits: usize,
    /// Number of auxiliary digit parts (formerly `fu`).
    pub aux_digit_parts: usize,
    /// Bit width of each auxiliary digit (formerly `bu`).
    pub aux_digit_bits: usize,
    /// Inner commitment rank (formerly `kappa`).
    pub inner_commit_rank: usize,
    /// Outer commitment rank (formerly `kappa1`, `0` in tail mode).
    pub outer_commit_rank: usize,
    /// Tail-mode marker.
    pub tail: bool,
}

/// One recursive Labrador level proof payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorLevelProof<F: FieldCore, const D: usize> {
    /// Whether this level uses tail semantics.
    pub tail: bool,
    /// Input row lengths (`n[i]` in C).
    pub input_row_lengths: Vec<usize>,
    /// Configuration selected for this level.
    pub config: LabradorReductionConfig,
    /// Virtual row length after reshaping (formerly `nn`).
    pub virtual_row_len: usize,
    /// Per-original-row split counts from the fold plan (formerly `nu`).
    pub row_split_counts: Vec<usize>,
    /// Opening-side payload for this level (formerly `u1`).
    pub inner_opening_payload: Vec<CyclotomicRing<F, D>>,
    /// Linear-garbage-side payload for this level (formerly `u2`).
    pub linear_garbage_payload: Vec<CyclotomicRing<F, D>>,
    /// JL projection vector.
    pub jl_projection: [i64; 256],
    /// JL nonce used to regenerate projection matrix.
    pub jl_nonce: u64,
    /// JL lift residuals with constant term zeroed in the proof
    /// (formerly `bb`).
    pub jl_lift_residuals: Vec<CyclotomicRing<F, D>>,
    /// Output witness norm bound after reduction (formerly `norm_sq`).
    pub next_witness_norm_sq: u128,
}

/// Full recursive Labrador proof plus final clear opening witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorProof<F: FieldCore, const D: usize> {
    /// Recursive level payloads.
    pub levels: Vec<LabradorLevelProof<F, D>>,
    /// Final clear witness opened at recursion termination.
    pub final_opening_witness: LabradorWitness<F, D>,
}

impl<F: FieldCore, const D: usize> LabradorLevelProof<F, D> {
    /// Serialized size of this level in bytes.
    pub fn size(&self) -> usize {
        let ring_bytes = std::mem::size_of::<CyclotomicRing<F, D>>();
        let ring_count = self.inner_opening_payload.len()
            + self.linear_garbage_payload.len()
            + self.jl_lift_residuals.len();
        ring_count * ring_bytes
            + self.jl_projection.len() * std::mem::size_of::<i64>()
            + std::mem::size_of::<u64>() // jl_nonce
            + std::mem::size_of::<u128>() // next_witness_norm_sq
    }
}

impl<F: FieldCore, const D: usize> LabradorProof<F, D> {
    /// Construct an empty proof (used when Labrador is disabled).
    pub fn empty() -> Self {
        Self {
            levels: Vec::new(),
            final_opening_witness: LabradorWitness { rows: Vec::new() },
        }
    }

    /// Total serialized size of the proof in bytes.
    pub fn size(&self) -> usize {
        let ring_bytes = std::mem::size_of::<CyclotomicRing<F, D>>();
        let levels_size: usize = self.levels.iter().map(|l| l.size()).sum();
        let witness_rings: usize = self
            .final_opening_witness
            .rows
            .iter()
            .map(|r| r.len())
            .sum();
        levels_size + witness_rings * ring_bytes
    }
}
