use super::support::{checked_in_artifact_bytes, checked_in_catalog};
use akita_config::{
    policy_of, proof_optimized::fp128, CommitmentConfig, TrustedScheduleCatalog,
    ValidatedScheduleCatalog,
};
use akita_params::{
    DecompositionParams, OpeningScheduleSelection, ScheduleRowDigest, SisModulusProfileId,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug)]
struct CountingDense;

static RING_CHALLENGE_AUDITS: AtomicUsize = AtomicUsize::new(0);

impl CommitmentConfig for CountingDense {
    type Field = <fp128::Dense as CommitmentConfig>::Field;
    type ExtField = <fp128::Dense as CommitmentConfig>::ExtField;

    fn schedule_family_name() -> &'static str {
        fp128::Dense::schedule_family_name()
    }

    const RING_DIMENSION_SCHEDULE: akita_config::RingDimensionSchedule =
        fp128::Dense::RING_DIMENSION_SCHEDULE;
    const EXT_DEGREE: usize = fp128::Dense::EXT_DEGREE;

    fn decomposition() -> DecompositionParams {
        fp128::Dense::decomposition()
    }

    fn ring_challenge_config(
        dimension: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, akita_error::AkitaError> {
        RING_CHALLENGE_AUDITS.fetch_add(1, Ordering::SeqCst);
        fp128::Dense::ring_challenge_config(dimension)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        fp128::Dense::sis_modulus_profile()
    }

    fn opening_basis_range() -> (u32, u32) {
        fp128::Dense::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        fp128::Dense::inner_basis_range()
    }

    fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
        fp128::Dense::committed_source_class()
    }

    fn recursive_setup_planning() -> bool {
        fp128::Dense::recursive_setup_planning()
    }
}

#[test]
fn trusted_artifact_round_trip_preserves_rows_and_selection() {
    let checked_in =
        checked_in_catalog::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let artifact = checked_in.to_artifact_bytes().expect("encode artifact");
    let loaded = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &artifact,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect("load trusted artifact");

    let key =
        akita_params::ScheduleLookupKey::single(akita_params::PolynomialGroupLayout::new(14, 1));
    let checked_in_row = checked_in.resolve_key(&key).expect("checked-in row");
    let loaded_row = loaded.resolve_key(&key).expect("artifact row");
    assert_eq!(loaded.catalog_digest(), checked_in.catalog_digest());
    assert_eq!(loaded_row.selection(), checked_in_row.selection());
    assert_eq!(loaded_row.profiles(), checked_in_row.profiles());
    assert!(std::ptr::eq(
        loaded_row,
        loaded.resolve_selection(loaded_row.selection()).unwrap()
    ));
    assert!(std::ptr::eq(
        loaded_row,
        loaded.resolve_profiles(loaded_row.profiles()).unwrap()
    ));
    assert_eq!(loaded_row.schedule(), checked_in_row.schedule());
    assert!(loaded_row
        .schedule()
        .recursive_folds
        .iter()
        .any(|fold| fold.params.ring_relation_mode.is_reduced_evaluation()));
}

#[test]
fn proof_selection_cannot_supply_schedule_content() {
    let catalog =
        checked_in_catalog::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
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
        checked_in_catalog::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let artifact = dense.to_artifact_bytes().expect("encode dense artifact");
    let error = TrustedScheduleCatalog::<fp128::OneHot>::from_artifact_bytes(
        &artifact,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("a dense artifact must not install as a one-hot catalog");
    assert!(format!("{error}").contains("family"));
    assert_ne!(
        fp128::Dense::committed_source_class(),
        fp128::OneHot::committed_source_class()
    );
}

#[test]
fn config_binding_rejects_a_resolvable_catalog_with_the_wrong_family() {
    let dense =
        checked_in_catalog::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let row = dense.rows().next().expect("dense catalog row");
    let selection = row.selection();
    let raw = ValidatedScheduleCatalog::try_new(
        fp128::OneHot::schedule_family_name(),
        dense
            .rows()
            .map(|row| (row.profiles().clone(), row.schedule().clone())),
        &policy_of::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
        fp128::Dense::ring_challenge_config,
    )
    .expect("config-free catalog remains semantically valid");
    raw.resolve_selection(selection)
        .expect("the copied row remains resolvable before config binding");

    let error = TrustedScheduleCatalog::<fp128::Dense>::new(
        raw,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("a verifier-facing catalog must bind to the exact config family");
    assert!(error.to_string().contains("family"));
}

#[test]
fn direct_artifact_binding_does_not_repeat_the_row_audit() {
    let bytes = checked_in_artifact_bytes::<fp128::Dense>();

    RING_CHALLENGE_AUDITS.store(0, Ordering::SeqCst);
    ValidatedScheduleCatalog::from_artifact_bytes(
        &bytes,
        CountingDense::schedule_family_name(),
        &policy_of::<CountingDense>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
        CountingDense::ring_challenge_config,
    )
    .expect("raw artifact audit");
    let one_audit = RING_CHALLENGE_AUDITS.load(Ordering::SeqCst);
    assert!(one_audit > 0, "the fixture must exercise challenge audits");

    RING_CHALLENGE_AUDITS.store(0, Ordering::SeqCst);
    TrustedScheduleCatalog::<CountingDense>::from_artifact_bytes(
        &bytes,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect("direct config-bound artifact load");
    assert_eq!(
        RING_CHALLENGE_AUDITS.load(Ordering::SeqCst),
        one_audit,
        "binding must reuse the artifact audit instead of traversing every row again",
    );
}

#[test]
fn binding_revalidates_the_concrete_config_challenge_hook() {
    let catalog =
        checked_in_catalog::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let error = catalog
        .validate_binding(
            fp128::Dense::schedule_family_name(),
            &policy_of::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
            |_| {
                Err(akita_error::AkitaError::InvalidSetup(
                    "deliberately mismatched challenge hook".to_string(),
                ))
            },
        )
        .expect_err("binding must revalidate challenge hooks");
    assert!(format!("{error}").contains("deliberately mismatched challenge hook"));
}
