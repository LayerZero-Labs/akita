//! Cross-revision byte goldens captured on the authorized build host.
//!
//! Capture on the fixed base revision, then compare on the candidate. Capture
//! uses create_new, so rerunning it cannot replace an existing golden.

use super::*;
use std::io::Write;

#[allow(clippy::too_many_arguments)]
pub(super) fn check(
    scheme: &AkitaCommitmentScheme<BinaryTestSuffix>,
    setup: &akita_cpu_backend::AkitaProverSetup<F>,
    commitment: &CommittedGroup<F>,
    ordinary: Option<&CommittedGroup<F>>,
    proof: &[u8],
    continuation: &[u8; 32],
    mixed: bool,
    width: usize,
) {
    let Some(directory) = std::env::var_os("AKITA_BINARY_BYTE_GOLDENS") else {
        return;
    };
    let required = akita_config::SetupRequirements::from_catalog::<BinaryTestSuffix>(
        scheme.schedules(),
        if mixed { 23 } else { 18 },
        if mixed { 2 } else { 1 },
    )
    .unwrap()
    .matrix_capacity()
    .num_field_elements;
    let mut descriptor = setup.expanded.descriptor().clone();
    descriptor.num_field_elements = required;
    let mut bytes = Vec::new();
    descriptor.serialize_compressed(&mut bytes).unwrap();
    for coefficient in &setup.expanded.shared_matrix().as_field_slice()[..required] {
        coefficient.serialize_compressed(&mut bytes).unwrap();
    }
    commitment.serialize_compressed(&mut bytes).unwrap();
    if let Some(ordinary) = ordinary {
        ordinary.serialize_compressed(&mut bytes).unwrap();
    }
    bytes.extend_from_slice(&(proof.len() as u64).to_le_bytes());
    bytes.extend_from_slice(proof);
    bytes.extend_from_slice(continuation);
    let path = std::path::PathBuf::from(directory).join(format!(
        "binary-mixed-{mixed}-width-{width}-dev-{}-keccak-{}.golden",
        akita_params::DEV_PROTOCOL,
        cfg!(feature = "transcript-keccak"),
    ));
    if std::env::var("AKITA_CAPTURE_BINARY_BYTE_GOLDENS").as_deref() == Ok("1") {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    } else {
        let golden = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), golden.len(), "{} length", path.display());
        assert!(bytes == golden, "{} bytes changed", path.display());
        if let Some(directory) = std::env::var_os("AKITA_BINARY_BYTE_GOLDEN_RECEIPTS") {
            let receipt = std::path::PathBuf::from(directory).join(path.file_name().unwrap());
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(receipt)
                .unwrap()
                .write_all(b"matched\n")
                .unwrap();
        }
    }
}

/// Capture mode on the candidate must always consume independent dev bytes.
pub(super) fn compare_baseline(proof: &[u8], name: &str) {
    let directory = std::env::var_os("AKITA_BINARY_BASELINE_DIR");
    assert!(
        directory.is_some() || std::env::var_os("AKITA_BINARY_CAPTURE_DIR").is_none(),
        "AKITA_BINARY_BASELINE_DIR is required in capture mode",
    );
    if let Some(directory) = directory {
        let path = std::path::PathBuf::from(directory).join(name);
        let baseline = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("missing baseline {}: {error}", path.display()));
        assert_eq!(proof, baseline, "pinned dev binary root bytes changed");
    }
}
