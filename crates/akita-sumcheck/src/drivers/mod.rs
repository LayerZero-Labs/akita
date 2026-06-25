//! Sumcheck proof driver traits and verifier replay.
//!
//! Clear prove loops live in [`crate::sink`]; these extensions delegate there
//! and retain verify-side replay plus ZK masked prove paths.

mod eq_factored;
mod standard;

pub use eq_factored::{
    advance_eq_factored_claim, EqFactoredSumcheckInstanceProverExt,
    EqFactoredSumcheckInstanceVerifierExt,
};
#[cfg(feature = "zk")]
pub use eq_factored::{
    EqFactoredMaskedProveOutput, ZkEqFactoredFinalRelation, ZkEqFactoredSumcheckInstanceProverExt,
    ZkEqFactoredSumcheckInstanceVerifierExt,
};
pub use standard::{
    check_sumcheck_output_claim, SumcheckInstanceProverExt, SumcheckInstanceVerifierExt,
};
#[cfg(feature = "zk")]
pub use standard::{
    MaskedProveOutput, ZkSumcheckFinalRelation, ZkSumcheckInstanceProverExt,
    ZkSumcheckInstanceVerifierExt,
};
