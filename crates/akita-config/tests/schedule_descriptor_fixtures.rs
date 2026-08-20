//! Golden descriptor-byte fixtures for every generated schedule catalog.
//!
//! This is step 1 of `specs/parameter-struct-consolidation.md`. It exists to be
//! built **before** any type change, so that the byte-neutral steps (2 through 4)
//! can be proven byte-neutral rather than asserted, and so that the deliberate
//! break in step 5 is visible as an exact, reviewed diff instead of a surprise.
//!
//! # What is covered
//!
//! Every entry of every catalog whose Cargo feature is active, at each level of
//! the parameter surface:
//!
//! - the catalog identity and its `key_digest`;
//! - the lookup key;
//! - the whole-schedule descriptor;
//! - each fold's `CommittedGroupParams` descriptor;
//! - each group's frozen `GroupCommitPhaseParams` and its `GroupOpeningPlan`;
//! - the incoming setup prefix, where a recursive fold consumes one;
//! - the terminal descriptor;
//! - the `ScheduleRowDigest` for the row.
//!
//! Coverage is metadata-driven through
//! [`akita_planner::generated_families::ALL_GENERATED_FAMILIES`], so a new
//! family is picked up without editing this file. Under `--features
//! all-schedules` that is all 13 catalogs, which is what the spec requires:
//! single- and multi-group roots, chunked and unchunked, recursive folds,
//! B-sliced groups, subring-packing folds, terminal L2 routes, and bounded
//! committed dense sources.
//!
//! # Why digests rather than raw byte dumps
//!
//! A length plus a digest per level detects any change, which is the property
//! the spec needs, while keeping the committed file small enough to review in a
//! diff. On mismatch the failure prints the full hex of the differing record, so
//! debugging does not need the raw bytes checked in.
//!
//! # Regenerating
//!
//! ```text
//! AKITA_BLESS_SCHEDULE_FIXTURES=1 \
//!   cargo test -p akita-config --features all-schedules \
//!   --test schedule_descriptor_fixtures
//! ```
//!
//! Only bless a diff you have read. Steps 2 through 4 must produce an empty one.

#![allow(missing_docs)]

use akita_planner::generated_families::ALL_GENERATED_FAMILIES;
use akita_schedules::schedule_from_entry;
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupBatchProfile, FoldSchedule, ScheduleRowDigest,
};

const FIXTURE_PATH: &str = "tests/fixtures/schedule_descriptor_bytes.txt";
/// Byte-frozen subset: commit-phase profiles and catalog identity.
///
/// These must not move even when everything above them breaks. Keeping them in
/// a separate file means re-blessing the main fixture — which step 5 legitimately
/// requires — cannot silently take them along. Blessing this one is a deliberate,
/// separate act.
const FROZEN_PATH: &str = "tests/fixtures/schedule_frozen_profile_bytes.txt";
const BLESS_FROZEN_ENV: &str = "AKITA_BLESS_FROZEN_PROFILE_FIXTURES";
const BLESS_ENV: &str = "AKITA_BLESS_SCHEDULE_FIXTURES";

/// FNV-1a 64. Implemented here on purpose: the fixture must not drift because a
/// hashing dependency changed its output.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One captured descriptor record.
struct Record {
    label: String,
    bytes: Vec<u8>,
}

impl Record {
    fn line(&self) -> String {
        format!(
            "{:<58} len={:<6} fnv1a={:016x}",
            self.label,
            self.bytes.len(),
            fnv1a(&self.bytes)
        )
    }
}

fn push(records: &mut Vec<Record>, label: impl Into<String>, bytes: Vec<u8>) {
    records.push(Record {
        label: label.into(),
        bytes,
    });
}

/// Capture every descriptor level of one expanded row.
fn capture_row(
    records: &mut Vec<Record>,
    prefix: &str,
    key: &AkitaScheduleLookupKey,
    schedule: &FoldSchedule,
) {
    push(
        records,
        format!("{prefix} lookup_key"),
        key.canonical_descriptor_bytes(),
    );
    push(
        records,
        format!("{prefix} schedule"),
        schedule.canonical_descriptor_bytes(),
    );

    // Root fold: the fold record, its final group, and each precommitted group.
    let root = &schedule.root.params;
    push(
        records,
        format!("{prefix} root.fold"),
        root.canonical_descriptor_bytes(),
    );
    push(
        records,
        format!("{prefix} root.final_group.profile"),
        akita_types::GroupCommitPhaseParams::try_from_params(key.final_group, root)
            .map(|profile| profile.canonical_descriptor_bytes())
            .unwrap_or_default(),
    );
    for (index, group) in root.precommitted_groups.iter().enumerate() {
        push(
            records,
            format!("{prefix} root.precommitted[{index}].profile"),
            group.profile.canonical_descriptor_bytes(),
        );
        push(
            records,
            format!("{prefix} root.precommitted[{index}].opening"),
            group.opening.canonical_descriptor_bytes(),
        );
        push(
            records,
            format!("{prefix} root.precommitted[{index}].group"),
            group.canonical_descriptor_bytes(),
        );
    }

    // Recursive folds, including any consumed setup prefix.
    for (level, step) in schedule.recursive_folds.iter().enumerate() {
        push(
            records,
            format!("{prefix} recursive[{level}].fold"),
            step.params.canonical_descriptor_bytes(),
        );
        if let Some(prefix_group) = step.params.setup_prefix.as_ref() {
            push(
                records,
                format!("{prefix} recursive[{level}].setup_prefix.profile"),
                prefix_group.profile.canonical_descriptor_bytes(),
            );
            push(
                records,
                format!("{prefix} recursive[{level}].setup_prefix.opening"),
                prefix_group.opening.canonical_descriptor_bytes(),
            );
            push(
                records,
                format!("{prefix} recursive[{level}].setup_prefix.natural_len"),
                prefix_group
                    .setup_natural_len
                    .expect("setup prefix group")
                    .to_le_bytes()
                    .to_vec(),
            );
        }
    }

    push(
        records,
        format!("{prefix} terminal"),
        schedule.terminal.canonical_descriptor_bytes(),
    );

    // The row digest is what schedule selection compares, so pin it too.
    let profiles = CommittedGroupBatchProfile {
        final_group: akita_types::GroupCommitPhaseParams::try_from_params(key.final_group, root)
            .expect("final group profile"),
        precommitteds: key.precommitteds.clone(),
    };
    let digest: ScheduleRowDigest =
        akita_types::schedule_row_digest(&profiles, schedule).expect("schedule row digest");
    push(
        records,
        format!("{prefix} row_digest"),
        format!("{digest:?}").into_bytes(),
    );
}

fn collect_records() -> Vec<Record> {
    let mut records = Vec::new();
    let mut families: Vec<_> = ALL_GENERATED_FAMILIES.iter().collect();
    families.sort_by_key(|family| family.module_name);

    for family in families {
        let Some(table) = (family.schedule_catalog)() else {
            // Feature inactive in this build. The fixture records only what is
            // linked, and the header states the feature set it was blessed under.
            continue;
        };
        let name = family.module_name;
        let identity = table.identity;
        push(
            &mut records,
            format!("{name} catalog.identity"),
            format!("{identity:?}").into_bytes(),
        );
        push(
            &mut records,
            format!("{name} catalog.key_digest"),
            identity.key_digest.to_le_bytes().to_vec(),
        );
        push(
            &mut records,
            format!("{name} catalog.key_count"),
            identity.key_count.to_le_bytes().to_vec(),
        );

        let policy = (family.policy)();
        for (index, entry) in table.entries.iter().enumerate() {
            let key = AkitaScheduleLookupKey {
                final_group: entry.final_group,
                precommitteds: entry
                    .root
                    .precommitted_groups
                    .iter()
                    .map(|group| group.profile)
                    .collect(),
            };
            let schedule = schedule_from_entry(entry, &key, &policy, family.ring_challenge_config)
                .unwrap_or_else(|error| panic!("{name} entry {index} failed to expand: {error}"));
            capture_row(
                &mut records,
                &format!("{name} [{index:03}]"),
                &key,
                &schedule,
            );
        }
    }
    records
}

fn render(records: &[Record]) -> String {
    let mut out = String::new();
    out.push_str("# Golden descriptor-byte fixtures for the generated schedule catalogs.\n");
    out.push_str("#\n");
    out.push_str("# Owned by specs/parameter-struct-consolidation.md (step 1).\n");
    out.push_str("# Steps 2 through 4 of that plan produced an EMPTY diff here.\n");
    out.push_str("# Step 5d broke everything above the commit-phase profile on purpose:\n");
    out.push_str("#   - the terminal descriptor is now role-atomic (geometry, A role, fold),\n");
    out.push_str("#   - protocol_epoch 1 -> 2, schedule-row domain v2 -> v3,\n");
    out.push_str("#   - the FoldSchedule descriptor tag 1 -> 2.\n");
    out.push_str("# Old proof bytes no longer verify. The commit-phase profile bytes and the\n");
    out.push_str("# catalog sort key did NOT move; that subset has its own frozen fixture in\n");
    out.push_str("# schedule_frozen_profile_bytes.txt, which must be blessed separately.\n");
    out.push_str("#\n");
    out.push_str("# Regenerate (only after reading the diff):\n");
    out.push_str("#   AKITA_BLESS_SCHEDULE_FIXTURES=1 cargo test -p akita-config \\\n");
    out.push_str("#     --features all-schedules --test schedule_descriptor_fixtures\n");
    out.push_str("#\n");
    out.push_str("# Format: <family> [<entry>] <level>  len=<bytes> fnv1a=<hex>\n");
    out.push_str("# Digest is FNV-1a 64 over the canonical descriptor bytes.\n");
    out.push('\n');
    for record in records {
        out.push_str(&record.line());
        out.push('\n');
    }
    out
}

/// A fixture that cannot fail is worse than no fixture, because it reads as
/// coverage. Prove the mechanism detects a single flipped bit at every level
/// this harness captures, and that the digest is not accidentally
/// length-only.
#[test]
fn the_fixture_mechanism_detects_change() {
    let records = collect_records();
    assert!(records.len() > 100, "expected a populated capture");
    let baseline = render(&records);

    for index in [0usize, records.len() / 2, records.len() - 1] {
        let mut mutated: Vec<Record> = records
            .iter()
            .map(|record| Record {
                label: record.label.clone(),
                bytes: record.bytes.clone(),
            })
            .collect();
        let target = &mut mutated[index];
        if target.bytes.is_empty() {
            target.bytes.push(1);
        } else {
            // Flip one bit, preserving length, so a length-only comparison
            // would miss it.
            target.bytes[0] ^= 0x01;
        }
        let before = records[index].bytes.len();
        let after = mutated[index].bytes.len();
        assert_ne!(
            render(&mutated),
            baseline,
            "flipping a bit in record {index} ({}) did not change the fixture",
            records[index].label
        );
        if before == after {
            assert_ne!(
                fnv1a(&records[index].bytes),
                fnv1a(&mutated[index].bytes),
                "digest is length-only for record {index}"
            );
        }
    }

    // A reordering must also be visible: the labels carry position.
    let mut swapped: Vec<Record> = records
        .iter()
        .map(|record| Record {
            label: record.label.clone(),
            bytes: record.bytes.clone(),
        })
        .collect();
    let last = swapped.len() - 1;
    swapped.swap(0, last);
    assert_ne!(render(&swapped), baseline, "record order is not pinned");
}

/// Family name a record line belongs to (the first whitespace-delimited token).
fn family_of(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}

/// Is this record part of the byte-frozen commit-phase surface?
///
/// Deliberately excludes `catalog.identity`: it embeds `protocol_epoch`, so it
/// *must* move when the epoch is bumped, and freezing it would assert the
/// opposite of what §10.2 says. What is frozen is the commit-phase profile
/// bytes, the catalog sort key and its digest, and the lookup keys — the things
/// that must hold still while everything above them breaks.
fn is_frozen(label: &str) -> bool {
    label.contains(".profile")
        || label.contains("lookup_key")
        || label.contains("catalog.key_digest")
        || label.contains("catalog.key_count")
}

/// The commit-phase profile bytes and catalog identity are frozen.
///
/// `specs/parameter-struct-consolidation.md` §10.2 keeps these fixed while the
/// levels above them break once, because profile bytes feed the catalog
/// `key_digest` and are the catalog sort key. A change here means entry ordering
/// shifted or a committed digest moved, which is a different and much larger
/// event than re-encoding a fold.
#[test]
fn commit_phase_profile_and_catalog_bytes_are_frozen() {
    let records = collect_records();
    let frozen: Vec<&Record> = records.iter().filter(|r| is_frozen(&r.label)).collect();
    assert!(
        frozen.len() > 50,
        "expected a populated frozen subset, got {}",
        frozen.len()
    );
    let rendered: String = frozen.iter().map(|r| format!("{}\n", r.line())).collect();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FROZEN_PATH);

    if std::env::var_os(BLESS_FROZEN_ENV).is_some() {
        std::fs::create_dir_all(path.parent().expect("fixture parent"))
            .expect("create fixture dir");
        std::fs::write(&path, &rendered).expect("write frozen fixture");
        eprintln!("blessed {} frozen records", frozen.len());
        return;
    }

    let expected = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "missing frozen fixture {}: {error}\n\
             create it with {BLESS_FROZEN_ENV}=1 and the all-schedules feature",
            path.display()
        )
    });
    let linked: std::collections::BTreeSet<&str> =
        frozen.iter().map(|r| family_of(&r.label)).collect();
    let expected: String = expected
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter(|line| linked.contains(family_of(line)))
        .map(|line| format!("{line}\n"))
        .collect();
    assert_eq!(
        expected, rendered,
        "commit-phase profile or catalog bytes moved. This is NOT ordinary \
         step-5 churn: profile bytes are the catalog sort key and feed key_digest. \
         Do not re-bless without establishing why entry ordering or a committed \
         digest changed."
    );
}

#[test]
fn generated_schedule_descriptor_bytes_are_stable() {
    let records = collect_records();
    assert!(
        !records.is_empty(),
        "no generated catalog was linked; run with --features all-schedules"
    );

    let rendered = render(&records);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_PATH);

    if std::env::var_os(BLESS_ENV).is_some() {
        // Blessing must never record a partial catalog set, or a later
        // all-schedules run would look like a regression.
        let linked: std::collections::BTreeSet<&str> =
            records.iter().map(|r| family_of(&r.label)).collect();
        assert_eq!(
            linked.len(),
            ALL_GENERATED_FAMILIES.len(),
            "refusing to bless a partial fixture: {} of {} families linked. \
             Re-run with --features all-schedules.",
            linked.len(),
            ALL_GENERATED_FAMILIES.len()
        );
        std::fs::create_dir_all(path.parent().expect("fixture parent"))
            .expect("create fixture dir");
        std::fs::write(&path, &rendered).expect("write fixture");
        eprintln!("blessed {} records into {}", records.len(), path.display());
        return;
    }

    let full = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "missing descriptor fixture {}: {error}\n\
             create it with {BLESS_ENV}=1 and the all-schedules feature",
            path.display()
        )
    });

    // The fixture is blessed under `all-schedules`, but this test also runs in
    // builds that link fewer catalogs. Compare only the families actually
    // linked, so the check is meaningful in every feature set instead of
    // passing in exactly one of them. Under `all-schedules` this is the whole
    // file.
    let linked: std::collections::BTreeSet<&str> =
        records.iter().map(|r| family_of(&r.label)).collect();
    let expected: String = full
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter(|line| linked.contains(family_of(line)))
        .map(|line| format!("{line}\n"))
        .collect();
    let rendered_body: String = rendered
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| format!("{line}\n"))
        .collect();
    eprintln!(
        "checked {} records across {}/{} linked families",
        records.len(),
        linked.len(),
        ALL_GENERATED_FAMILIES.len()
    );
    assert!(
        !expected.is_empty(),
        "no fixture lines matched the linked families {linked:?}; \
         the fixture may predate a family rename"
    );

    let rendered = rendered_body;
    if expected == rendered {
        return;
    }

    // Report the first few differing records with full hex, so a reviewer does
    // not need the raw bytes committed to diagnose a break.
    let by_label: std::collections::HashMap<&str, &Record> = records
        .iter()
        .map(|record| (record.label.as_str(), record))
        .collect();
    let mut detail = String::new();
    let mut shown = 0usize;
    let mut differing = 0usize;
    for (expected_line, actual_line) in expected.lines().zip(rendered.lines()) {
        if expected_line == actual_line {
            continue;
        }
        differing += 1;
        if shown >= 8 {
            continue;
        }
        shown += 1;
        detail.push_str(&format!(
            "\n  expected: {expected_line}\n  actual:   {actual_line}\n"
        ));
        let label = actual_line.split("  ").next().unwrap_or("").trim();
        if let Some(record) = by_label.get(label) {
            detail.push_str(&format!("  actual bytes: {}\n", hex(&record.bytes)));
        }
    }

    panic!(
        "generated schedule descriptor bytes changed ({differing} differing records, \
         {} expected lines, {} actual lines).\n\
         If this change is intended, read the diff and re-bless with \
         {BLESS_ENV}=1.\n{detail}",
        expected.lines().count(),
        rendered.lines().count(),
    );
}
