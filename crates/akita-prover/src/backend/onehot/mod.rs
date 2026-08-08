//! One-hot polynomial: sparse witness with at most one nonzero field
//! element per chunk of size `onehot_k`.
//!
//! [`OneHotPoly`] implements the four prover operations (ring evaluation, per-block
//! fold, decompose+fold, and inner-Ajtai commit) by iterating only over the
//! nonzero monomial positions.
//!
//! # Module layout
//!
//! The module is organized as private kernel and polynomial modules.
//!
//!   - [`OneHotIndex`]: a tiny trait implemented for `u8`/`u16`/`u32`/
//!     `usize` so callers can hand [`OneHotPoly::new`] a `Vec<Option<I>>`
//!     at the narrowest width that fits their hot positions.
//!   - One hot block views use the same [`SparseRingBlockEntry`] and
//!     [`FlatBlocks<E>`] representation as sparse ring polynomials. Each hot
//!     coefficient is one entry with value `1`.
//!   - [`OneHotPoly<F, I>`]: the caller-facing polynomial. Storage is
//!     D-free; ring-shaped ops take the kernel dispatch dimension as a
//!     method-level const generic.

use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use akita_algebra::ring::cyclotomic::WideCyclotomicRing;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::parallel::*;
use akita_field::unreduced::{HasCommitAccum, HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
};
use akita_types::{FpExtEncoding, RingMatrixView};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use super::flat_blocks::FlatBlocks;
use super::sparse_ring::{SparseRingBlockEntry, SparseRingCoeff};
use crate::backend::poly_helpers::{build_decompose_fold_witness, fill_rotated_challenge};
use crate::compute::{CommitmentComputeBackend, OneHotBlockSource, OneHotCommitRowsPlan};
use crate::{CommitInnerWitness, DecomposeFoldWitness, SparseRingPoly};

/// Wide accumulators use 16-bit chunks in `i32` limbs, so they can safely
/// absorb at most 32,768 unit-scale additions before overflow.
#[cfg(test)]
pub(super) const MAX_WIDE_SHIFT_ACCUMULATIONS: usize = 1 << 15;

mod accumulate;
mod blocks;
mod column_sweep;
mod decompose_fold;
mod entries;
mod fold;
mod inner_ajtai;
mod ops;
mod poly;
#[cfg(test)]
pub(crate) mod test_helpers;
#[cfg(test)]
mod tests;

pub use blocks::LazyOneHotBlocks;
#[cfg(test)]
pub(crate) use column_sweep::column_sweep_ajtai_onehot;
pub(crate) use column_sweep::column_sweep_ajtai_onehot_multi;
pub use entries::OneHotIndex;
#[cfg(test)]
use inner_ajtai::{inner_ajtai_wide_onehot, inner_ajtai_wide_single_chunk_tiled};
pub use ops::{OneHotBatchView, OneHotView};
pub use poly::OneHotPoly;
