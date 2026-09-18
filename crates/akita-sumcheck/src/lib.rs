//! Generic sumcheck proof types, traits, and transcript drivers.
//!
//! This crate owns only protocol-independent sumcheck machinery. Akita-specific
//! stage provers, verifier instances, and two-round-prefix skip proofs stay in
//! the PCS protocol crate until their role-specific APIs are split.

mod accum;
mod batched_sumcheck;
mod compact_fold;
mod native;
mod single;
mod traits;
mod types;

pub use akita_algebra::poly::{fold_evals_in_place, multilinear_eval};
pub use akita_algebra::uni_poly::{CompressedUniPoly, UniPoly};

pub use accum::reduce_signed_accum;
pub use batched_sumcheck::{
    check_batched_output_claim, compute_batched_expected_output_claim, prove_batched_sumcheck,
    verify_batched_sumcheck, verify_batched_sumcheck_rounds, BatchedSumcheckRoundResult,
};
pub use compact_fold::CompactPairFoldLut;
pub use native::{prove_sumcheck_native, verify_sumcheck_native};
pub use single::{
    advance_eq_factored_claim, prove_eq_factored_sumcheck, prove_sumcheck,
    verify_eq_factored_sumcheck, verify_sumcheck, verify_sumcheck_rounds,
};
pub use traits::{
    EqFactoredSumcheckInstanceProver, SumcheckInstanceProver, SumcheckInstanceVerifier,
};
pub use types::{
    uniform_sumcheck_shape, EqFactoredSumcheckProof, EqFactoredSumcheckProofShape,
    EqFactoredUniPoly, SumcheckProof, SumcheckProofShape,
};
