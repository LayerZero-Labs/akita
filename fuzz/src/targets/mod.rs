//! Engine-independent target entry points.

pub mod artifact;
pub mod boundary;
pub mod decompose;
pub mod deserialize;
pub mod field;
pub mod legacy;
pub mod multilinear;
pub mod pcs;
pub mod ring;
pub mod sumcheck;
pub mod transcript;

/// Every engine-independent target, keyed by its libFuzzer binary name.
/// A target's engine-independent entry point.
pub type Target = fn(&[u8]);

pub const ALL: &[(&str, Target)] = &[
    ("field_arith", field::run),
    ("ring_ntt", ring::run),
    ("decompose", decompose::run),
    ("multilinear", multilinear::run),
    ("sumcheck_roundtrip", sumcheck::run),
    ("sumcheck_rounds", legacy::sumcheck_rounds),
    ("transcript_roundtrip", transcript::run),
    ("transcript_labels", legacy::transcript_labels),
    ("serialization_vec", legacy::serialization_vec),
    ("public_deserialize", deserialize::run),
    ("schedule_artifact", artifact::run),
    ("pcs_dense", pcs::dense),
    ("pcs_onehot", pcs::onehot),
    ("pcs_batch", pcs::batch),
    ("pcs_recursive", pcs::recursive),
    ("pcs_reject", pcs::reject),
    ("pcs_parallel", pcs::parallel),
    ("verifier_boundary", boundary::verifier),
    ("prover_boundary", boundary::prover),
];

pub fn by_name(name: &str) -> Option<Target> {
    ALL.iter()
        .find(|(candidate, _)| *candidate == name)
        .map(|(_, run)| *run)
}
