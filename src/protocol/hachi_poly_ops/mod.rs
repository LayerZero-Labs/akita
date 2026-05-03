//! Root polynomial backends for the Hachi commitment scheme.
//!
//! [`HachiPolyOps`](akita_prover::HachiPolyOps) lives in `akita-prover` and
//! exposes the operations the Hachi commit/prove paths need from a
//! caller-provided root polynomial, rather than raw coefficient access. The
//! concrete implementations in this module handle those root operations in
//! their own optimal way:
//!
//! - [`DensePoly`] — standard dense algorithms (decompose + NTT matvec).
//! - [`OneHotPoly`] — sparse monomial tricks, avoids all inner ring
//!   multiplications.
//! - [`MultilinearPolynomail`] — borrowed wrapper that lets one batch mix dense
//!   and one-hot multilinear polynomials under one shared scheme config/layout.
//!
//! Recursive levels do not use [`HachiPolyOps`](akita_prover::HachiPolyOps).
//! They operate on `RecursiveWitnessFlat` / `RecursiveWitnessView`, which
//! model the D-agnostic `w` witness produced by ring switching.
//!
//! # Module layout
//!
//! - `dense` — [`DensePoly`] and its
//!   [`HachiPolyOps`](akita_prover::HachiPolyOps) impl.
//! - `multilinear_polynomail` — [`MultilinearPolynomail`], the canonical
//!   representation-erasing wrapper for mixed root batches.
//! - `onehot` — [`OneHotPoly`], [`OneHotIndex`], and column-sweep Ajtai
//!   commit helpers.
//! - `recursive_witness` — recursive `w` owner/view types and digit-native
//!   operations for later folding levels.
//! - `helpers` — shared internal helpers: decomposition, sparse
//!   multiply-accumulate, position-partitioned accumulation.
//! - `decompose_fold_neon` — AArch64 NEON kernel for the sparse-mul-acc
//!   hot loop (conditionally compiled).
//!
//! # Extensibility
//!
//! This trait is coupled to power-of-2 cyclotomic rings
//! ([`CyclotomicRing<F, D>`]).  When non-power-of-2 rings are added, the trait
//! signature will change.  Additional operation methods may be added as the
//! protocol evolves.

#[cfg(target_arch = "aarch64")]
mod decompose_fold_neon;
mod dense;
mod helpers;
mod multilinear_polynomail;
mod onehot;
mod recursive_witness;

pub use dense::DensePoly;
pub use multilinear_polynomail::MultilinearPolynomail;
#[cfg(test)]
pub(crate) use onehot::OneHotBlocks;
pub use onehot::{OneHotIndex, OneHotPoly};
pub(crate) use recursive_witness::{RecursiveWitnessFlat, RecursiveWitnessView};
