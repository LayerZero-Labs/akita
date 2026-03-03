//! Labrador recursive proof sub-protocol.
//!
//! This module will host the Greyhound/Labrador integration used by Hachi's
//! recursive handoff path.

pub mod challenge;
pub mod comkey;
pub mod commit;
pub mod config;
pub mod fold;
pub mod guardrails;
pub mod johnson_lindenstrauss;
pub mod prover;
pub mod transcript;
pub mod types;
pub mod utils;
pub mod verifier;

pub use commit::{commit_linear_only, LabradorCommitmentArtifacts};
pub use config::{select_config, sis_secure};
pub use fold::{prove_level, LabradorFoldResult};
pub use johnson_lindenstrauss::{
    collapse, project, restore_constant_term, zero_constant_term_for_proof, LabradorJlMatrix,
};
pub use prover::{prove, prove_with_config};
pub use types::{
    LabradorConstraint, LabradorLevelProof, LabradorProof, LabradorReductionConfig,
    LabradorStatement, LabradorWitness,
};
pub use verifier::{verify, LabradorVerifyResult};
