//! Compare candidate captures with independently captured baseline bytes.

/// Capture mode on the candidate must always consume independent dev bytes.
pub(super) fn compare_baseline(proof: &[u8], name: &str) {
    if name.starts_with("p64-") {
        let pin = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts")
                .join(if akita_params::DEV_PROTOCOL {
                    "schedules-binary-source-p64-dev"
                } else {
                    "schedules-binary-source-p64"
                })
            .join(format!("{name}.pin"));
        let digest = akita_params::descriptor_bytes::digest_descriptor_bytes(proof);
        let actual = format!("{} {}\n", proof.len(),
            digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>());
        eprintln!("p64 reference {name}: {actual}");
        let expected = std::fs::read_to_string(&pin).unwrap_or_else(|error| {
            panic!("Linux reference pin pending at {}: {error}; observed {actual}", pin.display())
        });
        assert_eq!(actual, expected, "p64 singleton reference bytes changed");
        return;
    }
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
