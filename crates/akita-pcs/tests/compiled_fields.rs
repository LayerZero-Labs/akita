//! Catalog admission checks executable field capabilities before setup/proving.

#[cfg(all(
    feature = "field-fp64",
    not(feature = "field-fp32"),
    not(feature = "field-fp128")
))]
#[test]
fn fp64_only_catalog_admission_accepts_fp64_and_rejects_other_tiers() {
    use akita_config::proof_optimized::{fp128, fp32, fp64};
    use akita_pcs::AkitaCommitmentScheme;

    let fp64_artifact = include_bytes!("../../../artifacts/schedules/fp64_dense.aks");
    AkitaCommitmentScheme::<fp64::Dense>::from_schedule_artifact(fp64_artifact)
        .expect("Fp64 catalog is supported by an Fp64-only build");

    let fp32_artifact = include_bytes!("../../../artifacts/schedules/fp32_dense.aks");
    let fp32_error =
        AkitaCommitmentScheme::<fp32::Dense>::from_schedule_artifact(fp32_artifact).unwrap_err();
    assert!(fp32_error.to_string().contains("field-fp32"));

    let fp128_artifact = include_bytes!("../../../artifacts/schedules/fp128_dense.aks");
    let fp128_error =
        AkitaCommitmentScheme::<fp128::Dense>::from_schedule_artifact(fp128_artifact).unwrap_err();
    assert!(fp128_error.to_string().contains("field-fp128"));
}
