//! Generic sumcheck proof types, traits, and transcript drivers.
//!
//! This crate owns only protocol-independent sumcheck machinery. Akita-specific
//! stage provers, verifier instances, and two-round-prefix skip proofs stay in
//! the PCS protocol crate until their role-specific APIs are split.

pub mod accum;
pub mod batched_sumcheck;
pub mod compact_fold;
pub mod descriptor;
pub mod drivers;
pub mod traits;
pub mod types;

pub use akita_algebra::poly::{
    fold_evals_in_place, multilinear_eval, multilinear_eval_small, range_check_eval,
};
pub use akita_algebra::uni_poly::{CompressedUniPoly, UniPoly};

pub use accum::reduce_signed_accum;
pub use batched_sumcheck::{
    check_batched_output_claim, compute_batched_expected_output_claim, prove_batched_sumcheck,
    verify_batched_sumcheck, verify_batched_sumcheck_rounds, BatchedSumcheckRoundResult,
};
pub use compact_fold::CompactPairFoldLut;
pub use drivers::{
    advance_eq_factored_claim, check_sumcheck_output_claim, EqFactoredSumcheckInstanceProverExt,
    EqFactoredSumcheckInstanceVerifierExt, SumcheckInstanceProverExt, SumcheckInstanceVerifierExt,
};
#[cfg(feature = "zk")]
pub use drivers::{
    EqFactoredMaskedProveOutput, MaskedProveOutput, ZkEqFactoredFinalRelation,
    ZkEqFactoredSumcheckInstanceProverExt, ZkEqFactoredSumcheckInstanceVerifierExt,
    ZkSumcheckFinalRelation, ZkSumcheckInstanceProverExt, ZkSumcheckInstanceVerifierExt,
};
pub use traits::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceVerifier,
    EqFactoredSumcheckRoundState, SumcheckInstanceProver, SumcheckInstanceVerifier,
};
pub use types::{
    uniform_sumcheck_shape, EqFactoredSumcheckProof, EqFactoredSumcheckProofShape,
    EqFactoredUniPoly, SumcheckProof, SumcheckProofShape,
};
#[cfg(feature = "zk")]
pub use types::{EqFactoredSumcheckProofMasked, SumcheckProofMasked};
