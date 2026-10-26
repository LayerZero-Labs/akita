use akita_config::{CommitmentConfig, TrustedScheduleCatalog};

pub(super) fn checked_in_catalog<Cfg: CommitmentConfig>(
    chunking: akita_params::ChunkedWitnessCfg,
) -> TrustedScheduleCatalog<Cfg> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts")
        .join(akita_params::SCHEDULE_ARTIFACT_SET)
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(&bytes, chunking)
        .expect("checked-in trusted artifact")
}

pub(super) fn checked_in_artifact_bytes<Cfg: CommitmentConfig>() -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts")
        .join(akita_params::SCHEDULE_ARTIFACT_SET)
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    std::fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}
