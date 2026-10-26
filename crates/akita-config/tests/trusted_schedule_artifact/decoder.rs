use super::json_mutation::*;
use super::support::{checked_in_artifact_bytes, checked_in_catalog};
use akita_config::{
    policy_of, proof_optimized::fp128, CommitmentConfig, TrustedScheduleCatalog,
    ValidatedScheduleCatalog, MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
    MAX_TRUSTED_SCHEDULE_ARTIFACT_ROW_BYTES,
};

#[test]
fn decoder_rejects_empty_oversized_and_noncanonical_bytes() {
    let empty = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &[],
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("an empty artifact must reject");
    assert!(format!("{empty}").contains("byte length"));

    let oversized = vec![b' '; MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES + 1];
    let oversized_error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &oversized,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("an oversized artifact must reject before decoding");
    assert!(format!("{oversized_error}").contains("byte length"));

    let mut noncanonical = checked_in_artifact_bytes::<fp128::Dense>();
    noncanonical.push(b'\n');
    let noncanonical_error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &noncanonical,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("trailing whitespace must reject");
    let noncanonical_error = noncanonical_error.to_string();
    assert!(
        noncanonical_error.contains("canonical JSON"),
        "unexpected noncanonical error: {noncanonical_error}"
    );
}

#[test]
fn decoder_rejects_family_and_row_limits_during_envelope_preflight() {
    let bytes = checked_in_artifact_bytes::<fp128::Dense>();

    let oversized_family = replace_family_name(&bytes, &"x".repeat(129));
    let family_error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &oversized_family,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("oversized family must reject during envelope decoding");
    let family_error = family_error.to_string();
    assert!(
        family_error.contains("family name length"),
        "unexpected family error: {family_error}"
    );

    let escaped_family = replace_family_name(&bytes, r"\u0066p128_dense");
    let family_error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &escaped_family,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("an escaped family spelling must reject before decoded allocation");
    assert!(family_error.to_string().contains("family token"));

    let oversized_rows = replace_rows_with_empty_objects(&bytes, 16_385);
    let row_error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &oversized_rows,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("the sentinel artifact row must reject before row decoding");
    assert!(format!("{row_error}").contains("row count exceeds 16384"));

    let oversized_row = pad_first_row(&bytes, MAX_TRUSTED_SCHEDULE_ARTIFACT_ROW_BYTES);
    let row_error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &oversized_row,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("an oversized raw row must reject before typed row decoding");
    assert!(format!("{row_error}").contains("row 0 byte length"));
}

#[test]
fn decoder_rejects_empty_root_and_recursive_groups_without_panicking() {
    let bytes = checked_in_artifact_bytes::<fp128::Dense>();
    for (label, occurrence) in [("root", 0), ("recursive", 1)] {
        let malformed = empty_nth_fold_group_list(&bytes, occurrence);
        let result = std::panic::catch_unwind(|| {
            TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
                &malformed,
                akita_params::ChunkedWitnessCfg::default_non_chunked(),
            )
        });
        let decoded = result.unwrap_or_else(|_| panic!("{label} empty groups panicked"));
        let error = decoded.expect_err("empty fold groups must reject");
        assert!(
            error.to_string().contains("own group"),
            "unexpected {label} empty-groups error: {error}"
        );
    }
}

#[test]
fn decoder_rejects_format_policy_and_duplicate_row_tampering() {
    let bytes = checked_in_artifact_bytes::<fp128::Dense>();

    let mut wrong_magic = bytes.clone();
    mutate_first_json_digit(&mut wrong_magic, b"\"magic\"");
    let error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &wrong_magic,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("wrong magic must reject");
    assert!(format!("{error}").contains("format"));

    let mut wrong_version = bytes.clone();
    mutate_first_json_digit(&mut wrong_version, b"\"version\"");
    let error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &wrong_version,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("unsupported version must reject");
    assert!(format!("{error}").contains("format"));

    let mut wrong_epoch = bytes.clone();
    mutate_first_json_digit(&mut wrong_epoch, b"\"protocol_epoch\"");
    let error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &wrong_epoch,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("wrong protocol epoch must reject");
    assert!(format!("{error}").contains("protocol epoch"));

    let mut wrong_policy = bytes.clone();
    mutate_first_json_number_without_changing_width(&mut wrong_policy, b"\"policy_digest\"");
    let error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &wrong_policy,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("wrong policy digest must reject");
    assert!(format!("{error}").contains("policy"));

    let duplicate = duplicate_first_json_row(&bytes);
    let error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &duplicate,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("duplicate semantic rows must reject");
    assert!(format!("{error}").contains("duplicate"));

    let reordered = swap_first_two_json_rows(&bytes);
    let error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        &reordered,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("noncanonical row order must reject");
    assert!(format!("{error}").contains("canonical digest order"));
}

#[test]
fn catalog_constructor_enforces_family_and_row_count_bounds() {
    let catalog =
        checked_in_catalog::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let row = catalog.rows().next().expect("nonempty catalog");
    let owned_row = (row.profiles().clone(), row.schedule().clone());
    let policy = policy_of::<fp128::Dense>(akita_params::ChunkedWitnessCfg::default_non_chunked());

    let empty = ValidatedScheduleCatalog::try_new(
        fp128::Dense::schedule_family_name(),
        std::iter::empty(),
        &policy,
        fp128::Dense::ring_challenge_config,
    )
    .expect_err("empty catalogs must reject");
    assert!(format!("{empty}").contains("row count"));

    struct PanicAfterSentinel<T> {
        row: T,
        polls: usize,
    }

    impl<T: Clone> Iterator for PanicAfterSentinel<T> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            assert!(
                self.polls <= 16_384,
                "catalog constructor polled beyond the sentinel row"
            );
            self.polls += 1;
            Some(self.row.clone())
        }
    }

    let too_many = ValidatedScheduleCatalog::try_new(
        fp128::Dense::schedule_family_name(),
        PanicAfterSentinel {
            row: owned_row,
            polls: 0,
        },
        &policy,
        fp128::Dense::ring_challenge_config,
    )
    .expect_err("catalog row count must be bounded");
    assert!(format!("{too_many}").contains("row count"));

    let long_family = ValidatedScheduleCatalog::try_new(
        "x".repeat(129),
        catalog
            .rows()
            .map(|row| (row.profiles().clone(), row.schedule().clone())),
        &policy,
        fp128::Dense::ring_challenge_config,
    )
    .expect_err("family names must be bounded");
    assert!(format!("{long_family}").contains("family name length"));
}

#[test]
fn artifact_rows_reject_a_second_profile_authority() {
    let bytes = checked_in_artifact_bytes::<fp128::Dense>();
    let text = String::from_utf8(bytes).unwrap();
    assert!(!text.contains("\"profiles\""));
    let duplicated = text.replacen("\"schedule\":", "\"profiles\":{},\"schedule\":", 1);
    let error = TrustedScheduleCatalog::<fp128::Dense>::from_artifact_bytes(
        duplicated.as_bytes(),
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect_err("serialized profiles must not supply a second authority");
    assert!(error.to_string().contains("unknown field `profiles`"));
}
