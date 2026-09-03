use akita_config::{
    policy_of, proof_optimized::fp128, trusted_schedule_catalog_from_bytes, CommitmentConfig,
    TrustedScheduleCatalog,
};
use akita_types::{OpeningScheduleSelection, ScheduleRowDigest};

fn checked_in_catalog<Cfg: CommitmentConfig>() -> TrustedScheduleCatalog {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/schedules")
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    trusted_schedule_catalog_from_bytes::<Cfg>(&bytes).expect("checked-in trusted artifact")
}

fn checked_in_artifact_bytes<Cfg: CommitmentConfig>() -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/schedules")
        .join(format!("{}.aks", Cfg::schedule_family_name()));
    std::fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn replace_once(bytes: &mut [u8], needle: &[u8], replacement: &[u8]) {
    assert_eq!(needle.len(), replacement.len());
    let offset = bytes
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("artifact marker");
    bytes[offset..offset + needle.len()].copy_from_slice(replacement);
}

fn duplicate_first_json_row(bytes: &[u8]) -> Vec<u8> {
    let rows_marker = b"\"rows\":[";
    let rows_start = bytes
        .windows(rows_marker.len())
        .position(|window| window == rows_marker)
        .map(|offset| offset + rows_marker.len())
        .expect("artifact rows marker");
    assert_eq!(bytes.get(rows_start), Some(&b'{'));

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut row_end = None;
    for (relative, &byte) in bytes[rows_start..].iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    row_end = Some(rows_start + relative + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let row_end = row_end.expect("complete first artifact row");
    let insert_at = bytes.len() - 2;
    assert_eq!(&bytes[insert_at..], b"]}");
    let mut duplicated = Vec::with_capacity(bytes.len() + row_end - rows_start + 1);
    duplicated.extend_from_slice(&bytes[..insert_at]);
    duplicated.push(b',');
    duplicated.extend_from_slice(&bytes[rows_start..row_end]);
    duplicated.extend_from_slice(&bytes[insert_at..]);
    duplicated
}

#[test]
fn trusted_artifact_round_trip_preserves_rows_and_selection() {
    let checked_in = checked_in_catalog::<fp128::Dense>();
    let artifact = checked_in.to_artifact_bytes().expect("encode artifact");
    let loaded = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&artifact)
        .expect("load trusted artifact");

    let key =
        akita_types::AkitaScheduleLookupKey::single(akita_types::PolynomialGroupLayout::new(14, 1));
    let checked_in_row = checked_in.resolve_key(&key).expect("checked-in row");
    let loaded_row = loaded.resolve_key(&key).expect("artifact row");
    assert_eq!(loaded.catalog_digest(), checked_in.catalog_digest());
    assert_eq!(loaded_row.selection(), checked_in_row.selection());
    assert_eq!(loaded_row.profiles(), checked_in_row.profiles());
    assert_eq!(loaded_row.schedule(), checked_in_row.schedule());
}

#[test]
fn proof_selection_cannot_supply_schedule_content() {
    let catalog = checked_in_catalog::<fp128::Dense>();
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
    let dense = checked_in_catalog::<fp128::Dense>();
    let artifact = dense.to_artifact_bytes().expect("encode dense artifact");
    let error = trusted_schedule_catalog_from_bytes::<fp128::OneHot>(&artifact)
        .expect_err("a dense artifact must not install as a one-hot catalog");
    assert!(format!("{error}").contains("family"));
    assert_ne!(
        fp128::Dense::committed_source_class(),
        fp128::OneHot::committed_source_class()
    );
}

#[test]
fn decoder_rejects_empty_oversized_and_noncanonical_bytes() {
    let empty = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&[])
        .expect_err("an empty artifact must reject");
    assert!(format!("{empty}").contains("byte length"));

    let oversized = vec![b' '; 64 * 1024 * 1024 + 1];
    let oversized_error = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&oversized)
        .expect_err("an oversized artifact must reject before decoding");
    assert!(format!("{oversized_error}").contains("byte length"));

    let mut noncanonical = checked_in_artifact_bytes::<fp128::Dense>();
    noncanonical.push(b'\n');
    let noncanonical_error = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&noncanonical)
        .expect_err("trailing whitespace must reject");
    assert!(format!("{noncanonical_error}").contains("canonical JSON"));
}

#[test]
fn decoder_rejects_format_policy_and_duplicate_row_tampering() {
    let bytes = checked_in_artifact_bytes::<fp128::Dense>();

    let mut wrong_magic = bytes.clone();
    replace_once(&mut wrong_magic, b"\"magic\":[65,", b"\"magic\":[66,");
    let error = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&wrong_magic)
        .expect_err("wrong magic must reject");
    assert!(format!("{error}").contains("format"));

    let mut wrong_version = bytes.clone();
    replace_once(&mut wrong_version, b"\"version\":1", b"\"version\":2");
    let error = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&wrong_version)
        .expect_err("unsupported version must reject");
    assert!(format!("{error}").contains("format"));

    let mut wrong_epoch = bytes.clone();
    let marker = b"\"protocol_epoch\":";
    let epoch_start = wrong_epoch
        .windows(marker.len())
        .position(|window| window == marker)
        .map(|offset| offset + marker.len())
        .expect("protocol epoch marker");
    let epoch_end = wrong_epoch[epoch_start..]
        .iter()
        .position(|byte| !byte.is_ascii_digit())
        .map(|relative| epoch_start + relative)
        .expect("protocol epoch terminator");
    let last_digit = wrong_epoch
        .get_mut(epoch_end - 1)
        .expect("protocol epoch digit");
    *last_digit = if *last_digit == b'9' {
        b'8'
    } else {
        *last_digit + 1
    };
    let error = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&wrong_epoch)
        .expect_err("wrong protocol epoch must reject");
    assert!(format!("{error}").contains("protocol epoch"));

    let mut wrong_policy = bytes.clone();
    let marker = b"\"policy_digest\":[";
    let first_digest_byte = wrong_policy
        .windows(marker.len())
        .position(|window| window == marker)
        .map(|offset| offset + marker.len())
        .expect("policy digest marker");
    let digit = wrong_policy[first_digest_byte];
    assert!(digit.is_ascii_digit());
    wrong_policy[first_digest_byte] = if digit == b'9' { b'8' } else { digit + 1 };
    let error = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&wrong_policy)
        .expect_err("wrong policy digest must reject");
    assert!(format!("{error}").contains("policy"));

    let duplicate = duplicate_first_json_row(&bytes);
    let error = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&duplicate)
        .expect_err("duplicate semantic rows must reject");
    assert!(format!("{error}").contains("duplicate"));
}

#[test]
fn catalog_constructor_enforces_family_and_row_count_bounds() {
    let catalog = checked_in_catalog::<fp128::Dense>();
    let row = catalog.rows().next().expect("nonempty catalog");
    let owned_row = (row.profiles().clone(), row.schedule().clone());
    let policy = policy_of::<fp128::Dense>();

    let empty = TrustedScheduleCatalog::try_new(
        fp128::Dense::schedule_family_name(),
        std::iter::empty(),
        &policy,
        fp128::Dense::ring_challenge_config,
    )
    .expect_err("empty catalogs must reject");
    assert!(format!("{empty}").contains("row count"));

    let too_many = TrustedScheduleCatalog::try_new(
        fp128::Dense::schedule_family_name(),
        std::iter::repeat_n(owned_row, 16_385),
        &policy,
        fp128::Dense::ring_challenge_config,
    )
    .expect_err("catalog row count must be bounded");
    assert!(format!("{too_many}").contains("row count"));

    let long_family = TrustedScheduleCatalog::try_new(
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
