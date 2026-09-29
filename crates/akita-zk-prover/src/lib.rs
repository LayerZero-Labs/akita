//! Prover-side building blocks for zero-knowledge Akita openings.
//!
//! This crate is part of an opt-in extension that is still being assembled
//! (LayerZero-Labs/akita#120). It exposes no zero-knowledge mode: nothing on
//! the Akita proof path depends on it, and a caller cannot select it.
//! Randomness here comes from a caller-supplied cryptographic RNG and never
//! enters Akita's deterministic nonce searches.

pub mod ring;
