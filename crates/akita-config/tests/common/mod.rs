use akita_config::{CommitmentConfig, TrustedScheduleCatalog};

pub(crate) fn checked_in_catalog<Cfg: CommitmentConfig>() -> TrustedScheduleCatalog<Cfg> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/schedules")
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(&bytes).expect("checked-in trusted artifact")
}
