//! Instantiate the shared measurement driver with the research field pair.
type F = jolt_field::Prime64Offset59;
type E = jolt_field::Ext2<F>;
const PROFILE: akita_params::binary_setup::BinaryFieldProfile =
    akita_params::binary_setup::BinaryFieldProfile::P64;
#[path = "run.rs"]
pub(super) mod run;

#[cfg(test)]
#[path = "research_tests.rs"]
mod tests;

#[cfg(test)]
fn fixture_directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(if akita_params::DEV_PROTOCOL {
            "binary-source-p64-dev"
        } else {
            "binary-source-p64"
        })
}

#[cfg(test)]
fn compare_baseline(proof: &[u8], shape: &str) {
    let pin = fixture_directory().join(format!("p64-{shape}.pin"));
    let digest = akita_params::descriptor_bytes::digest_descriptor_bytes(proof);
    let actual = format!("{} {}", proof.len(), crate::metrics::hex(&digest));
    let expected = std::fs::read_to_string(&pin).unwrap_or_else(|error| {
        panic!(
            "Linux reference pin pending at {}: {error}; observed {actual}",
            pin.display()
        )
    });
    assert_eq!(actual, expected.trim(), "p64 reference bytes changed");
}
