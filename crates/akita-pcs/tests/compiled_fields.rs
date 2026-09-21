//! Catalog admission checks executable field capabilities before setup/proving.

fn schedule_artifact<Cfg: akita_config::CommitmentConfig>() -> Vec<u8> {
    let path = akita_config::test_support::workspace_schedule_artifact_path::<Cfg>();
    std::fs::read(path).expect("tracked schedule artifact")
}

#[cfg(all(
    feature = "field-fp64",
    not(feature = "field-fp32"),
    not(feature = "field-fp128")
))]
#[test]
fn fp64_only_catalog_admission_accepts_fp64_and_rejects_other_tiers() {
    use akita_config::proof_optimized::{fp128, fp32, fp64};
    use akita_pcs::AkitaCommitmentScheme;

    let fp64_artifact = schedule_artifact::<fp64::Dense>();
    AkitaCommitmentScheme::<fp64::Dense>::from_schedule_artifact(&fp64_artifact)
        .expect("Fp64 catalog is supported by an Fp64-only build");

    let fp32_artifact = schedule_artifact::<fp32::Dense>();
    let fp32_error =
        AkitaCommitmentScheme::<fp32::Dense>::from_schedule_artifact(&fp32_artifact).unwrap_err();
    assert!(fp32_error.to_string().contains("field-fp32"));

    let fp128_artifact = schedule_artifact::<fp128::Dense>();
    let fp128_error =
        AkitaCommitmentScheme::<fp128::Dense>::from_schedule_artifact(&fp128_artifact).unwrap_err();
    assert!(fp128_error.to_string().contains("field-fp128"));
}
