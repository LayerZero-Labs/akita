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

fn json_row_ranges(bytes: &[u8]) -> Vec<std::ops::Range<usize>> {
    let rows_marker = b"\"rows\":[";
    let rows_start = bytes
        .windows(rows_marker.len())
        .position(|window| window == rows_marker)
        .map(|offset| offset + rows_marker.len())
        .expect("artifact rows marker");
    assert_eq!(bytes.get(rows_start), Some(&b'{'));

    let mut ranges = Vec::new();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut row_start = None;
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
            b'{' => {
                if depth == 0 {
                    row_start = Some(rows_start + relative);
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    ranges.push(
                        row_start.take().expect("artifact row start")..rows_start + relative + 1,
                    );
                }
            }
            b']' if depth == 0 => break,
            _ => {}
        }
    }
    ranges
}

fn duplicate_first_json_row(bytes: &[u8]) -> Vec<u8> {
    let first = json_row_ranges(bytes)
        .into_iter()
        .next()
        .expect("complete first artifact row");
    let insert_at = bytes.len() - 2;
    assert_eq!(&bytes[insert_at..], b"]}");
    let mut duplicated = Vec::with_capacity(bytes.len() + first.len() + 1);
    duplicated.extend_from_slice(&bytes[..insert_at]);
    duplicated.push(b',');
    duplicated.extend_from_slice(&bytes[first]);
    duplicated.extend_from_slice(&bytes[insert_at..]);
    duplicated
}

fn swap_first_two_json_rows(bytes: &[u8]) -> Vec<u8> {
    let ranges = json_row_ranges(bytes);
    let first = ranges.first().expect("first artifact row");
    let second = ranges.get(1).expect("second artifact row");
    assert_eq!(&bytes[first.end..second.start], b",");
    let mut swapped = Vec::with_capacity(bytes.len());
    swapped.extend_from_slice(&bytes[..first.start]);
    swapped.extend_from_slice(&bytes[second.clone()]);
    swapped.push(b',');
    swapped.extend_from_slice(&bytes[first.clone()]);
    swapped.extend_from_slice(&bytes[second.end..]);
    swapped
}

fn empty_nth_fold_group_list(bytes: &[u8], occurrence: usize) -> Vec<u8> {
    let marker = b"\"groups\":{\"entries\":[";
    let array_start = bytes
        .windows(marker.len())
        .enumerate()
        .filter_map(|(offset, window)| (window == marker).then_some(offset + marker.len() - 1))
        .nth(occurrence)
        .expect("fold group list marker");

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let array_end = bytes[array_start..]
        .iter()
        .enumerate()
        .find_map(|(relative, &byte)| {
            if in_string {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_string = false;
                }
                return None;
            }
            match byte {
                b'"' => in_string = true,
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(array_start + relative);
                    }
                }
                _ => {}
            }
            None
        })
        .expect("complete fold group list");

    let mut malformed = Vec::with_capacity(bytes.len() - (array_end - array_start - 1));
    malformed.extend_from_slice(&bytes[..array_start]);
    malformed.extend_from_slice(b"[]");
    malformed.extend_from_slice(&bytes[array_end + 1..]);
    malformed
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
    assert!(loaded_row
        .schedule()
        .recursive_folds
        .iter()
        .any(|fold| fold.params.ring_relation_mode.is_reduced_evaluation()));
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
fn decoder_rejects_empty_root_and_recursive_groups_without_panicking() {
    let bytes = checked_in_artifact_bytes::<fp128::Dense>();
    for (label, occurrence) in [("root", 0), ("recursive", 1)] {
        let malformed = empty_nth_fold_group_list(&bytes, occurrence);
        let result = std::panic::catch_unwind(|| {
            trusted_schedule_catalog_from_bytes::<fp128::Dense>(&malformed)
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

    let reordered = swap_first_two_json_rows(&bytes);
    let error = trusted_schedule_catalog_from_bytes::<fp128::Dense>(&reordered)
        .expect_err("noncanonical row order must reject");
    assert!(format!("{error}").contains("canonical digest order"));
}

#[test]
fn binding_revalidates_the_concrete_config_challenge_hook() {
    let catalog = checked_in_catalog::<fp128::Dense>();
    let error = catalog
        .validate_binding(
            fp128::Dense::schedule_family_name(),
            &policy_of::<fp128::Dense>(),
            |_| {
                Err(akita_error::AkitaError::InvalidSetup(
                    "deliberately mismatched challenge hook".to_string(),
                ))
            },
        )
        .expect_err("binding must revalidate challenge hooks");
    assert!(format!("{error}").contains("deliberately mismatched challenge hook"));
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
