//! Prover-only Metal kernels for Akita.
//!
//! Field arithmetic, the device runtime and the error model come from
//! `jolt-metal`; this crate adds Akita's ring and commitment kernels on top.
//! Every kernel is bit-exact with the CPU path it replaces: the CPU backend
//! stays the reference, and each kernel has a differential test against it.
//!
//! On targets without Metal the crate compiles against `jolt-metal`'s
//! uninhabited backend and [`AkitaMetal::new`] returns an error of class
//! [`ErrorClass::Unavailable`], so callers keep the CPU path. The crate never
//! enters a verifier dependency graph. Design:
//! `specs/akita-compute-backend-metal.md`.

#![deny(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::panic_in_result_fn,
    clippy::unreachable
)]

mod crt;
pub mod decompose;
mod error;
mod library;
pub mod matvec;
pub mod ntt;
pub mod onehot;

pub use error::AkitaMetalError;
pub use jolt_metal::{ErrorClass, MetalError};
pub use library::{AkitaMetal, HEADERS, RING_DEGREES};
