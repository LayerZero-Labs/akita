//! Root polynomial backends for the Hachi commitment scheme.
//!
//! [`HachiPolyOps`](akita_prover::HachiPolyOps) lives in `akita-prover` and
//! exposes the operations the Hachi commit/prove paths need from a
//! caller-provided root polynomial, rather than raw coefficient access. The
//! concrete implementations in this module handle those root operations in
//! their own optimal way:
//!
//! - [`OneHotPoly`] — sparse monomial tricks, avoids all inner ring
//!   multiplications.
//! - [`MultilinearPolynomail`] — borrowed wrapper that lets one batch mix
//!   `akita_prover::DensePoly` and one-hot multilinear polynomials under one
//!   shared scheme config/layout.
//!
//! # Module layout
//!
//! - `multilinear_polynomail` — [`MultilinearPolynomail`], the canonical
//!   representation-erasing wrapper for mixed root batches.
//! - `onehot` — [`OneHotPoly`], [`OneHotIndex`], and column-sweep Ajtai
//!   commit helpers.
//!
//! Shared decomposition and sparse accumulation helpers live in
//! `akita-prover::poly_helpers`. Recursive `w` owner/view types live in
//! `akita-prover`.
//!
//! # Extensibility
//!
//! This trait is coupled to power-of-2 cyclotomic rings
//! ([`CyclotomicRing<F, D>`]).  When non-power-of-2 rings are added, the trait
//! signature will change.  Additional operation methods may be added as the
//! protocol evolves.

mod multilinear_polynomail;
mod onehot;

pub use multilinear_polynomail::MultilinearPolynomail;
#[cfg(test)]
pub(crate) use onehot::OneHotBlocks;
pub use onehot::{OneHotIndex, OneHotPoly};
