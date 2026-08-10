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
fn generated_artifacts_match_both_runtime_digests() {
    let base_digest = generated_artifact_digest();
    assert_eq!(base_digest, SIS_TABLE_DIGEST);

    let mut hasher = Sha3_256::new();
    hasher.update(b"akita-sis-table-q128-inner-d512-direct-v1\0");
    hasher.update(base_digest);
    hasher.update(SisModulusProfileId::Q128OffsetA7F7.modulus().to_le_bytes());
    hasher.update(512u32.to_le_bytes());
    hasher.update(super::super::SIS_REQUIRED_MAX_WIDTH.to_le_bytes());
    hasher.update(super::super::SIS_MAX_MODULE_RANK.to_le_bytes());
    hasher.update(
        u64::try_from(COEFF_LINF_BUCKETS.len())
            .expect("bucket count fits u64")
            .to_le_bytes(),
    );
    for &bound in COEFF_LINF_BUCKETS {
        hasher.update(bound.to_le_bytes());
    }
    for &bound in COEFF_LINF_BUCKETS {
        let widths = generated_sis_max_widths(
            DEFAULT_SIS_SECURITY_POLICY,
            SisModulusProfileId::Q128OffsetA7F7,
            512,
            bound,
        )
        .expect("generated q128 Inner/512 row");
        assert_eq!(widths.len(), super::super::SIS_MAX_MODULE_RANK as usize);
        for &width in widths {
            hasher.update(width.to_le_bytes());
        }
    }
    let extension_digest: [u8; 32] = hasher.finalize().into();
    assert_eq!(extension_digest, Q128_INNER_D512_DIGEST);
}
