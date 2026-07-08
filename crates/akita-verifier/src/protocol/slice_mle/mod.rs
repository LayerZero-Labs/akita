//! Evaluate the relation matrix MLE from its non-zero slices.
//!
//! The verifier needs the multilinear-extension evaluation of a virtual
//! table `M` at a random point `r`. The naive approach is to materialize
//! the full equality table `eq(r, ·)`: that costs `O(|M|)` field operations
//! and `O(|M|)` memory, where `|M|` is linear in the witness size. Both are
//! too expensive.
//!
//! `M` is mostly zero. Only a handful of contiguous **slices** of `M` are
//! non-trivial. The MLE evaluation decomposes additively over those slices,
//! so we can evaluate each slice in isolation against the same `r` and sum
//! the results — each slice is orders of magnitude smaller than `M`.

mod setup_contribution;
mod structured_slice;

pub(super) use akita_algebra::offset_eq::high_eq_window;
pub(crate) use setup_contribution::SetupContributionEvaluator;
pub(crate) use setup_contribution::{SetupContributionEvalMode, SetupContributionEvaluation};
pub(super) use structured_slice::{
    compute_r_contribution, EStructuredSlicesEvaluator, StructuredSliceMleEvaluator,
    TStructuredSlicesEvaluator, ZDenseSlicesEvaluator, ZStructuredPow2SlicesEvaluator,
};
