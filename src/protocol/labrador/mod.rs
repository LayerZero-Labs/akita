//! Labrador recursive proof sub-protocol.
//!
//! This module hosts the Labrador recursive proof sub-protocol used by Hachi's
//! handoff path.

pub mod aggregation;
pub mod challenge;
pub mod comkey;
pub mod commit;
pub mod config;
mod constraints;
pub mod fold;
pub mod guardrails;
pub(crate) mod hachi_opening;
pub(crate) mod hachi_statement;
pub mod johnson_lindenstrauss;
pub mod prover;
pub mod setup;
pub mod transcript;
pub mod types;
pub mod utils;
pub mod verifier;

pub use comkey::{derive_labrador_comkey_seed, LabradorComKeySeed};
pub use commit::{commit_linear_only, LabradorCommitmentArtifacts};
pub use config::{
    estimate_module_sis_euclidean, plan_fold, plan_handoff, select_config, select_config_with_mode,
    sis_secure, LabradorFoldPlan, SisEuclideanLatticeEstimate,
};
pub use constraints::{LabradorConstraint, LabradorConstraintTerm};
pub use fold::{prove_level, LabradorFoldResult};
pub use johnson_lindenstrauss::{
    collapse, project, restore_constant_term, zero_constant_term_for_proof, LabradorJlMatrix,
};
pub use prover::{prove, prove_with_plan};
pub use setup::LabradorSetup;
pub use types::{
    LabradorLevelProof, LabradorProof, LabradorReductionConfig, LabradorStatement, LabradorWitness,
};
pub use verifier::{verify, LabradorVerifyResult};
