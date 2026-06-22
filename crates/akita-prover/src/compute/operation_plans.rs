use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallenge, TensorChallenges};
use akita_field::FieldCore;

// ===========================================================================
// Open, source-typed operation boundary (PO1)
//
// Everything below this banner is the *new* prover compute boundary. It sits
// ABOVE the fixed representation-named row helpers above (`dense_commit_rows`,
// `onehot_commit_rows`, `ring_switch_relation_rows`, ...), which survive only
// as lower-level standard kernels. The new layer is open by *source type* `S`
// instead of closed over Akita's built-in plan shapes:
//
// - operation kernels (`RootCommitKernel`, `OpeningFoldKernel`, ...) take the
//   borrowed representation view as a generic type parameter `S`, so a
//   downstream crate can define its own local view type and implement the
//   relevant kernel for `CpuBackend` without modifying an Akita-owned enum;
// - root polynomials expose those views through capability traits
//   (`RootCommitSource`, `RootOpeningSource`, ...) whose associated view types
//   become the `S` a kernel runs over;
// - a prover run threads operation *contexts* (`OperationCtx`) bundled into a
//   `ProverComputeStack`, each carrying a backend plus its validated prepared
//   setup, so commitment / opening / tensor / ring-switch work can run on
//   independent backends while the protocol still sees canonical Akita outputs.
//
// PO1 establishes this surface additively: the kernel traits are skeletons with
// no Akita impls yet (the six representation nodes implement them in their own
// backend files), and the monolithic `ProverComputeBackend`/`AkitaPolyOps`
// boundary is intentionally left in place for PO4 to remove.
// ===========================================================================

/// Scalar operation parameters for an inner Ajtai commit.
///
/// The polynomial data lives in the borrowed commit source view (`S`); this
/// plan carries only the shape parameters the kernel needs to size its work.
#[derive(Debug, Clone, Copy)]
pub struct CommitInnerPlan {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Root block length in ring elements.
    pub block_len: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Number of balanced digits used when opening (recomposition width).
    pub num_digits_open: usize,
    /// Logarithm of the gadget basis.
    pub log_basis: u32,
}

/// Base-multiplier fold parameters for a fused evaluate-and-fold opening.
#[derive(Debug, Clone, Copy)]
pub struct OpeningFoldBasePlan<'a, F: FieldCore> {
    /// Outer evaluation scalars applied to the folded blocks.
    pub eval_outer_scalars: &'a [F],
    /// Per-block fold scalars.
    pub fold_scalars: &'a [F],
    /// Block length in ring elements.
    pub block_len: usize,
}

/// Ring-multiplier fold parameters for a fused evaluate-and-fold opening.
#[derive(Debug, Clone, Copy)]
pub struct OpeningFoldRingPlan<'a, F: FieldCore, const D: usize> {
    /// Outer evaluation ring multipliers applied to the folded blocks.
    pub eval_outer_scalars: &'a [CyclotomicRing<F, D>],
    /// Per-block fold ring multipliers.
    pub fold_scalars: &'a [CyclotomicRing<F, D>],
    /// Block length in ring elements.
    pub block_len: usize,
}

/// Fold parameters for a fused evaluate-and-fold opening.
///
/// The base/ring split preserves the current distinction between base
/// multiplier points (scalar folds) and ring multiplier points (sparse
/// ring-multiplier accumulation).
#[derive(Debug, Clone, Copy)]
pub enum OpeningFoldPlan<'a, F: FieldCore, const D: usize> {
    /// Base multiplier point: scalar fold weights.
    Base(OpeningFoldBasePlan<'a, F>),
    /// Ring multiplier point: ring-element fold weights.
    Ring(OpeningFoldRingPlan<'a, F, D>),
}

/// Fused evaluate-and-fold output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningFoldOutput<F: FieldCore, const D: usize> {
    /// Evaluation of the polynomial at the opening point.
    pub eval: CyclotomicRing<F, D>,
    /// Folded witness rows in ring form.
    pub folded: Vec<CyclotomicRing<F, D>>,
}

/// Decompose + challenge-fold parameters for one opening.
#[derive(Debug, Clone, Copy)]
pub struct DecomposeFoldPlan<'a> {
    /// Sparse fold challenges, outermost first.
    pub challenges: &'a [SparseChallenge],
    /// Block length in ring elements.
    pub block_len: usize,
    /// Number of balanced digits.
    pub num_digits: usize,
    /// Logarithm of the gadget basis.
    pub log_basis: u32,
}

/// Tensor-shaped batched decompose + fold parameters at one opening point.
#[derive(Debug, Clone, Copy)]
pub struct DecomposeFoldTensorBatchPlan<'a> {
    /// Tensor-structured fold challenges.
    pub tensor: &'a TensorChallenges,
    /// Block length in ring elements.
    pub block_len: usize,
    /// Number of balanced digits.
    pub num_digits: usize,
    /// Logarithm of the gadget basis.
    pub log_basis: u32,
}

/// Batched decompose + fold parameters at one opening point.
///
/// Both the sparse-challenge and tensor-shaped fused batched paths are exposed
/// so a representation can keep its fast batched kernel rather than folding
/// each polynomial independently and aggregating later.
#[derive(Debug, Clone, Copy)]
pub enum DecomposeFoldBatchPlan<'a> {
    /// Sparse-challenge batched fold.
    Sparse(DecomposeFoldPlan<'a>),
    /// Tensor-shaped batched fold.
    Tensor(DecomposeFoldTensorBatchPlan<'a>),
}

/// Scalar operation parameters for the fused ring-switch relation rows.
///
/// The decomposed witness data (`e_hat`, `t_hat`, centered `z` segment) and the
/// centered infinity norm live in the borrowed relation source view (`S`).
#[derive(Debug, Clone, Copy)]
pub struct RingSwitchRelationPlan {
    /// Number of D-side cyclic rows to produce.
    pub n_d: usize,
    /// Number of B-side cyclic rows to produce.
    pub n_b: usize,
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// Logarithm of the gadget basis used to produce `e_hat` and `t_hat`.
    pub log_basis: u32,
}

/// Scalar operation parameters for additional public-row quotient rows.
#[derive(Debug, Clone, Copy)]
pub struct RingSwitchQuotientPlan {
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
}
