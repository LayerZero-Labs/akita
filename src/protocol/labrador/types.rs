//! Core Labrador witness/statement/proof types.

use crate::algebra::ring::CyclotomicRing;
use crate::protocol::labrador::constraints::LabradorConstraint;
use crate::protocol::labrador::setup::LabradorSetup;
use crate::{CanonicalField, FieldCore};
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
        self.rows
            .iter()
            .flat_map(|row| row.iter())
            .map(|ring| ring.coeff_norm_sq())
            .fold(0u128, |acc, v| acc.saturating_add(v))
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
    pub challenges: Vec<CyclotomicRing<F, D>>,
    /// Combined `sum_i c_i * phi_i` relation carried into the next level.
    pub combined_phi: Vec<CyclotomicRing<F, D>>,
    /// Aggregated right-hand side for the diagonal relation.
    pub b_total: CyclotomicRing<F, D>,
    /// Commitment matrices needed to replay the reduced statement.
    pub setup: Arc<LabradorSetup<F, D>>,
}

/// Public statement reduced to Labrador recursion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorStatement<F: FieldCore, const D: usize> {
    /// Outer commitment for opening relation.
    pub u1: Vec<CyclotomicRing<F, D>>,
    /// Outer commitment for linear-garbage relation.
    pub u2: Vec<CyclotomicRing<F, D>>,
    /// Amortization challenges (per input witness row).
    pub challenges: Vec<CyclotomicRing<F, D>>,
    /// Sparse constraints checked by reducer/verifier.
    pub constraints: Vec<LabradorConstraint<F, D>>,
    /// Compact recursive statement representation used between Labrador levels.
    pub reduced_constraints: Option<Box<LabradorReducedConstraintPlan<F, D>>>,
    /// Squared norm bound.
    pub beta_sq: u128,
}

/// Per-level reduction parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabradorReductionConfig {
    /// Witness decomposition parts.
    pub f: usize,
    /// Witness decomposition basis log2.
    pub b: usize,
    /// Commitment decomposition parts.
    pub fu: usize,
    /// Commitment decomposition basis log2.
    pub bu: usize,
    /// Inner commitment rank.
    pub kappa: usize,
    /// Outer commitment rank (`0` in tail mode).
    pub kappa1: usize,
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
    /// Virtual row length after nu-reshaping.
    pub nn: usize,
    /// Per-original-row split counts from the fold plan.
    pub nu: Vec<usize>,
    /// First outer commitment.
    pub u1: Vec<CyclotomicRing<F, D>>,
    /// Second outer commitment.
    pub u2: Vec<CyclotomicRing<F, D>>,
    /// JL projection vector.
    pub jl_projection: [i64; 256],
    /// JL nonce used to regenerate projection matrix.
    pub jl_nonce: u64,
    /// Lift polynomials (constant term zeroed in proof).
    pub bb: Vec<CyclotomicRing<F, D>>,
    /// Output witness norm bound after reduction.
    pub norm_sq: u128,
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
        let ring_count = self.u1.len() + self.u2.len() + self.bb.len();
        ring_count * ring_bytes
            + self.jl_projection.len() * std::mem::size_of::<i64>()
            + std::mem::size_of::<u64>() // jl_nonce
            + std::mem::size_of::<u128>() // norm_sq
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
