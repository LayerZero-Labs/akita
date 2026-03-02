//! Core Labrador witness/statement/proof types.

use crate::algebra::ring::CyclotomicRing;
use crate::FieldCore;

/// One witness row carried through Labrador recursion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorWitnessRow<F: FieldCore, const D: usize> {
    /// Row elements in `R_q`.
    pub s: Vec<CyclotomicRing<F, D>>,
    /// Squared norm contribution of this row.
    pub norm_sq: u128,
}

/// Full witness object for one Labrador statement.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LabradorWitness<F: FieldCore, const D: usize> {
    /// Witness rows.
    pub rows: Vec<LabradorWitnessRow<F, D>>,
}

/// Identifies a contiguous witness slice participating in a sparse constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorConstraintEntry {
    /// Row index into `LabradorWitness::rows`.
    pub row: usize,
    /// Start offset within the row.
    pub offset: usize,
    /// Number of elements consumed from the row.
    pub len: usize,
}

/// Sparse linear constraint:
/// `sum_i multiplicities[i] * <coefficients[i], witness[entries[i]]> = target`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorConstraint<F: FieldCore, const D: usize> {
    /// Witness slices used by the constraint.
    pub entries: Vec<LabradorConstraintEntry>,
    /// Multiplicity per entry (C analogue: `sparsecnst.mult`).
    pub multiplicities: Vec<usize>,
    /// Coefficient vectors paired with `entries`.
    pub coefficients: Vec<Vec<CyclotomicRing<F, D>>>,
    /// Right-hand side target vector.
    pub target: Vec<CyclotomicRing<F, D>>,
}

/// Public statement reduced to Labrador recursion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorStatement<F: FieldCore, const D: usize> {
    /// Outer commitment for opening relation.
    pub u1: Vec<CyclotomicRing<F, D>>,
    /// Outer commitment for linear-garbage relation.
    pub u2: Vec<CyclotomicRing<F, D>>,
    /// Sparse constraints checked by reducer/verifier.
    pub constraints: Vec<LabradorConstraint<F, D>>,
    /// Squared norm bound.
    pub beta_sq: u128,
    /// Statement hash binding.
    pub hash: [u8; 16],
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
    /// Input row chunk counts (`nu[i]` in C).
    pub input_row_chunks: Vec<usize>,
    /// Configuration selected for this level.
    pub config: LabradorReductionConfig,
    /// First outer commitment.
    pub u1: Vec<CyclotomicRing<F, D>>,
    /// Second outer commitment.
    pub u2: Vec<CyclotomicRing<F, D>>,
    /// JL projection vector.
    pub jl_projection: [i32; 256],
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

impl<F: FieldCore, const D: usize> LabradorProof<F, D> {
    /// Construct an empty proof (used when Labrador is disabled).
    pub fn empty() -> Self {
        Self {
            levels: Vec::new(),
            final_opening_witness: LabradorWitness { rows: Vec::new() },
        }
    }
}
