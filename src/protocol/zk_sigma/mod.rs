//! Standalone zero-knowledge Sigma protocol with aborts.
//!
//! This module is intentionally independent from Hachi recursion. It proves
//! linear relations and product-of-linear quadratic relations over committed
//! field-coordinate witnesses. Hachi can later supply a ring/Ajtai commitment
//! backend and relation adapter without changing this transcript shape.

mod aborts;
mod commitment;
mod proof;
mod prover;
mod relation;
mod serialization;
mod statement;
mod transcript;
mod verifier;

pub use commitment::{CommitmentBackend, MatrixCommitmentKey};
pub use proof::{QuadraticMask, ZkSigmaProof};
pub use prover::{prove, MaskSampler};
pub use relation::{LinearExpression, LinearRelation, QuadraticRelation};
pub use statement::{ZkSigmaStatement, ZkSigmaWitness};
pub use verifier::verify;
