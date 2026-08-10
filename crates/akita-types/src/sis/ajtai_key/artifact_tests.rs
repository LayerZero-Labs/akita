use super::*;
use sha3::{Digest, Sha3_256};

fn generated_artifact_digest() -> [u8; 32] {
    const DOMAIN: &[u8] = b"akita-sis-table-digest-adps16-quantum-128bit\0";
    const FILES: &[(&str, &[u8])] = &[
        ("q32.rs", include_bytes!("../generated_sis_table/q32.rs")),
        ("q64.rs", include_bytes!("../generated_sis_table/q64.rs")),
        ("q128.rs", include_bytes!("../generated_sis_table/q128.rs")),
        (
            "policy_audit.csv",
            include_bytes!("../generated_sis_table/policy_audit.csv"),
        ),
        (
            "policy_review.txt",
            include_bytes!("../generated_sis_table/policy_review.txt"),
        ),
    ];
    let mut hasher = Sha3_256::new();
    hasher.update(DOMAIN);
    for &(filename, contents) in FILES {
        hasher.update(
            u64::try_from(contents.len())
                .expect("artifact length fits u64")
                .to_le_bytes(),
        );
        hasher.update(filename.as_bytes());
        hasher.update([0]);
        hasher.update(contents);
    }
    hasher.finalize().into()
}

#[test]
fn generated_artifacts_match_runtime_digest() {
    let base_digest = generated_artifact_digest();
    assert_eq!(base_digest, SIS_TABLE_DIGEST);
}
