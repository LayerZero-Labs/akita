//! Shared borrowed witness/polynomial view vocabulary for Akita.
//!
//! This crate is the single, lowest-layer definition of how a sumcheck or
//! polyops opening is resolved to a concrete multilinear witness table. It is
//! the Akita analog of Jolt's `jolt-witness`: vocabulary only, with no
//! algorithms and no protocol logic.
//!
//! Two items make up the vocabulary:
//!
//! - [`PolynomialView`]: a borrowed multilinear-evaluation view (an eval slice
//!   plus its `num_vars` shape). [`SumcheckEngine`] reads this view while
//!   evaluating a descriptor summand; polyops standard views use the same type.
//! - [`WitnessProvider`]: a fallible, panic-free trait that resolves an opening
//!   identifier to a [`PolynomialView`].
//!
//! Both lanes of the sumcheck/polyops stack consume this single vocabulary, so
//! there is exactly one witness-view layer (never invent a second one).
//!
//! The crate sits below `akita-sumcheck` and `akita-prover` in the dependency
//! graph and depends only on `akita-field`. It is verifier-reachable, so every
//! shape check is panic-free and returns [`akita_field::AkitaError`].

#![warn(missing_docs)]
#![warn(unreachable_pub)]

mod provider;
mod view;

pub use provider::WitnessProvider;
pub use view::PolynomialView;
