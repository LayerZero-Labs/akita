//! Protocol-level Fiat-Shamir challenge samplers.
//!
//! This crate is split into three layers:
//!
//! - [`SparseChallenge`] — the dependency-light data type representing one
//!   sampled sparse polynomial in `F[X]/(X^D + 1)`. Most workspace consumers
//!   only ever import this type; see [`crate::challenge`].
//! - [`SparseChallengeConfig`] — the policy enum that selects which sampling
//!   family is used (`Uniform`, `ExactShell`, `BoundedL1Ball`) and exposes
//!   policy questions like `l1_mass()` / `max_abs_coeff()` / `validate()` to
//!   `akita-config`, `akita-types`, and `akita-planner`; see [`crate::config`].
//! - [`sample_sparse_challenges`] — the transcript-driven sampler that turns
//!   a config plus a Fiat-Shamir transcript into challenges; see
//!   [`crate::sampler`].
//!
//! Each [`SparseChallengeConfig`] variant has a dedicated implementation under
//! [`sampler`]: `sampler::uniform`, `sampler::exact_shell`, and
//! `sampler::bounded_l1`. Crate-internal helpers (the SHAKE256-backed
//! [`sampler::xof`] cursor, the WAYS-table types, etc.) are not part of the
//! public API.

mod challenge;
mod config;
mod sampler;

pub use challenge::SparseChallenge;
pub use config::SparseChallengeConfig;
pub use sampler::sample_sparse_challenges;
