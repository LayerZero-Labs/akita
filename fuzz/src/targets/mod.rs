//! Engine-independent target entry points.

pub mod artifact;
#[cfg(feature = "end-to-end")]
pub mod boundary;
pub mod challenges;
pub mod decompose;
pub mod deserialize;
pub mod field;
pub mod legacy;
pub mod multilinear;
#[cfg(feature = "end-to-end")]
pub mod pcs;
pub mod ring;
pub mod sumcheck;
pub mod transcript;

/// A target's engine-independent entry point.
pub type Target = fn(&[u8]);

/// Targets that need no end-to-end harness.
const PRIMITIVE: &[(&str, Target)] = &[
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
    ("fold_challenges", challenges::run),
    ("schedule_artifact", artifact::run),
];

/// Commit/prove/verify targets over the catalog families.
#[cfg(feature = "end-to-end")]
const END_TO_END: &[(&str, Target)] = &[
    ("pcs_dense", pcs::dense),
    ("pcs_onehot", pcs::onehot),
    ("pcs_batch", pcs::batch),
    ("pcs_recursive", pcs::recursive),
    ("pcs_reject", pcs::reject),
    ("pcs_parallel", pcs::parallel),
    ("pcs_liveness", pcs::liveness),
    ("pcs_shared", pcs::shared),
    ("verifier_boundary", boundary::verifier),
    ("prover_boundary", boundary::prover),
    ("terminal_cache", boundary::terminal_cache),
];

#[cfg(not(feature = "end-to-end"))]
const END_TO_END: &[(&str, Target)] = &[];

/// Every target in this build, keyed by its libFuzzer binary name.
pub fn all() -> impl Iterator<Item = &'static (&'static str, Target)> {
    PRIMITIVE.iter().chain(END_TO_END)
}

pub fn by_name(name: &str) -> Option<Target> {
    all()
        .find(|(candidate, _)| *candidate == name)
        .map(|(_, run)| *run)
}
