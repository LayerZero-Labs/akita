//! Generic sumcheck proof types, traits, and transcript drivers.
//!
//! This crate owns only protocol-independent sumcheck machinery. Akita-specific
//! stage provers, verifier instances, and two-round-prefix skip proofs stay in
//! the PCS protocol crate until their role-specific APIs are split.

pub mod accum;
pub mod batched_sumcheck;
pub mod compact_fold;
pub mod drivers;
pub mod extension_opening_reduction;
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
pub use extension_opening_reduction::{
    check_extension_opening_reduction_output, check_tensor_extension_opening_claim,
    extension_opening_reduction_claim, extension_opening_reduction_eval_at_point,
    tensor_column_partials_from_base_evals, tensor_equality_factor_eval_at_point,
    tensor_equality_factor_evals, tensor_logical_claim_from_partials, tensor_opening_split,
    tensor_packed_witness_evals, tensor_partials_from_base_evals, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, BatchedExtensionOpeningReductionProver,
    BatchedExtensionOpeningReductionTerm, ExtensionOpeningFactorTerm,
    ExtensionOpeningReductionFactor, ExtensionOpeningReductionProver,
    ExtensionOpeningReductionRoundResult, ExtensionOpeningReductionSumcheck,
    ExtensionOpeningReductionVerifier, ExtensionOpeningTensorPartials,
    SparseExtensionOpeningWitness, EXTENSION_OPENING_REDUCTION_DEGREE,
    SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS,
};
pub use traits::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceVerifier,
    EqFactoredSumcheckRoundState, SumcheckInstanceProver, SumcheckInstanceVerifier,
};
pub use types::{
    EqFactoredSumcheckProof, EqFactoredSumcheckProofShape, EqFactoredUniPoly, SumcheckProof,
    SumcheckProofShape,
};
#[cfg(feature = "zk")]
pub use types::{EqFactoredSumcheckProofMasked, FullUniPoly, SumcheckProofMasked};
