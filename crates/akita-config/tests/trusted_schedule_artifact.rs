use akita_config::{
    proof_optimized::fp128, trusted_schedule_catalog_from_bytes,
    trusted_schedule_catalog_from_embedded, CommitmentConfig,
};
use akita_types::{OpeningScheduleSelection, ScheduleRowDigest};

#[test]
fn trusted_artifact_replaces_generated_rows_without_changing_selection() {
    let embedded = trusted_schedule_catalog_from_embedded::<fp128::Dense>()
        .expect("embedded migration catalog");
    let artifact = embedded.to_artifact_bytes().expect("encode artifact");
    let loaded = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&artifact)
        .expect("load trusted artifact");

    let key =
        akita_types::AkitaScheduleLookupKey::single(akita_types::PolynomialGroupLayout::new(14, 1));
    let embedded_row = embedded.resolve_key(&key).expect("embedded row");
    let loaded_row = loaded.resolve_key(&key).expect("artifact row");
    assert_eq!(loaded.catalog_digest(), embedded.catalog_digest());
    assert_eq!(loaded_row.selection(), embedded_row.selection());
    assert_eq!(loaded_row.profiles(), embedded_row.profiles());
    assert_eq!(loaded_row.schedule(), embedded_row.schedule());
}

#[test]
fn proof_selection_cannot_supply_schedule_content() {
    let catalog =
        trusted_schedule_catalog_from_embedded::<fp128::Dense>().expect("trusted schedule catalog");
    let unknown = OpeningScheduleSelection {
        row_digest: ScheduleRowDigest::from_bytes([0x5a; 32]),
    };
    let error = catalog
        .resolve_selection(unknown)
        .expect_err("an unknown proof selection must reject");
    assert!(format!("{error}").contains("trusted catalog"));
}

#[test]
fn artifact_for_one_config_is_rejected_by_another() {
    let dense =
        trusted_schedule_catalog_from_embedded::<fp128::Dense>().expect("dense migration catalog");
    let artifact = dense.to_artifact_bytes().expect("encode dense artifact");
    let error = trusted_schedule_catalog_from_bytes::<fp128::OneHot>(&artifact)
        .expect_err("a dense artifact must not install as a one-hot catalog");
    assert!(format!("{error}").contains("family"));
    assert_ne!(
        fp128::Dense::committed_source_class(),
        fp128::OneHot::committed_source_class()
    );
}
