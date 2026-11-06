//! Research replay fixtures and the P64 adversarial backend.
#[cfg(test)]
type F = jolt_field::Prime64Offset59;
#[cfg(test)]
type E = jolt_field::Ext2<F>;
#[cfg(test)]
type BinaryTestSuffix = crate::run::config::P64;
// The shared backend also contains frontend helpers used by the P128 suite.
#[cfg(test)]
#[allow(dead_code)]
#[path = "../../tests/binary_source_end_to_end/fault_backend.rs"]
pub(crate) mod fault_backend;
#[cfg(test)]
std::thread_local! {
    pub(super) static FAULT: std::cell::Cell<Option<fault_backend::Fault>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
#[path = "research_tests.rs"]
mod tests;

#[cfg(test)]
pub(super) fn fixture_directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(if akita_params::DEV_PROTOCOL {
            "binary-source-p64-dev"
        } else {
            "binary-source-p64"
        })
}

#[cfg(test)]
fn compare_baseline(proof: &[u8], log: usize, shape: &str) {
    let pin = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/binary-source-full-size-digests.tsv");
    let key = format!("p64/{log}");
    let algorithm = match jolt_transcript::DIGEST_HASH {
        jolt_transcript::DigestHash::Sha3_256 => "sha3-256",
        jolt_transcript::DigestHash::Keccak256 => "keccak256",
    };
    let digest = akita_params::descriptor_bytes::digest_descriptor_bytes(proof);
    let length = proof.len().to_string();
    let digest = crate::metrics::hex(&digest);
    let actual = [key.as_str(), shape, &length, &digest, algorithm];
    let pins = std::fs::read_to_string(&pin).unwrap_or_else(|error| {
        panic!(
            "missing Linux reference pins at {}: {error}; observed {actual:?}",
            pin.display()
        )
    });
    let mut entries = pins.lines().map(str::split_whitespace).filter(|fields| {
        let mut fields = fields.clone();
        fields.next() == Some(key.as_str()) && fields.next() == Some(shape)
    });
    let expected = entries
        .next()
        .unwrap_or_else(|| {
            panic!(
                "missing reference pin for {key} {shape} in {}",
                pin.display()
            )
        })
        .collect::<Vec<_>>();
    assert!(
        entries.next().is_none(),
        "duplicate reference pin for {key} {shape}"
    );
    assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "p64 reference bytes or digest algorithm changed (selected {algorithm})"
    );
}
