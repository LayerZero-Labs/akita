//! Compare candidate captures with independently captured baseline bytes.

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
