//! Sumcheck proof driver functions.
//!
//! Contains the generic prove/verify loops for standard and eq-factored
//! sumchecks.

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
