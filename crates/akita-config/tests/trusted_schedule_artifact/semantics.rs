use super::support::{checked_in_artifact_bytes, checked_in_catalog};
use akita_config::{
    proof_optimized::fp128, CommitmentConfig, RecursiveCommitmentConfig, TrustedScheduleCatalog,
};

/// Edit a checked-in artifact structurally. Admission audits rows before its
/// canonical-bytes check, so the edited artifact reaches the audit. The edited
/// bytes are not in canonical form, so every caller must pin the audit's own
/// diagnostic: any edit at all is an `InvalidSetup` at the canonical-bytes check.
fn edited_artifact<Cfg: CommitmentConfig>(edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let mut value: serde_json::Value =
        serde_json::from_slice(&checked_in_artifact_bytes::<Cfg>()).expect("artifact JSON");
    edit(&mut value);
    serde_json::to_vec_pretty(&value).expect("artifact JSON")
}

/// The first recursive fold's own ("final") group. It is the last entry: a
/// setup prefix and precommitted groups precede it in canonical order.
fn first_recursive_group(value: &mut serde_json::Value) -> &mut serde_json::Value {
    value["rows"]
        .as_array_mut()
        .expect("rows")
        .iter_mut()
        .find_map(|row| {
            row["schedule"]["recursive_folds"]
                .as_array_mut()
                .and_then(|folds| folds.first_mut())
        })
        .and_then(|fold| fold["params"]["groups"]["entries"].as_array_mut())
        .and_then(|entries| entries.last_mut())
        .expect("a row with a recursive fold group")
}

fn rejection_message<Cfg: CommitmentConfig>(bytes: &[u8]) -> String {
    match TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(
        bytes,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    ) {
        Err(akita_error::AkitaError::InvalidSetup(message)) => message,
        other => panic!("edited artifact must be rejected as InvalidSetup, got {other:?}"),
    }
}

/// Assert the digit-range pass rejected `role` of `group`, naming `edited`
/// (the out-of-range half of the pair, e.g. `"log_basis 17 "`).
fn assert_digit_range_rejection<Cfg: CommitmentConfig>(
    bytes: &[u8],
    group: &str,
    role: &str,
    edited: &str,
) {
    let message = rejection_message::<Cfg>(bytes);
    assert!(
        message.starts_with(&format!("{group}: {role} log_basis "))
            && message.contains(edited)
            && message.ends_with(" is outside the field decomposition"),
        "{group} {role} must fail the digit-range pass on {edited:?}, got {message:?}"
    );
}

#[test]
fn unsupported_terminal_inner_log_basis_is_rejected_not_panicking() {
    for log_basis in [0, 17, 128] {
        let bytes = edited_artifact::<fp128::Dense>(|value| {
            value["rows"][0]["schedule"]["terminal"]["inner"]["digits"]["log_basis"] =
                log_basis.into();
        });
        assert_digit_range_rejection::<fp128::Dense>(
            &bytes,
            "terminal fold",
            "A",
            &format!("log_basis {log_basis} "),
        );
    }
}

#[test]
fn unsupported_fold_group_log_basis_is_rejected_not_panicking() {
    type Cfg = RecursiveCommitmentConfig<akita_config::proof_optimized::fp32::Dense>;
    const GROUP: &str = "recursive fold 0 final group";
    for log_basis in [0, 17, 128] {
        let edited = format!("log_basis {log_basis} ");
        let outer = edited_artifact::<Cfg>(|value| {
            first_recursive_group(value)["profile"]["outer"]["digits"]["log_basis"] =
                log_basis.into();
        });
        if log_basis == 0 {
            // The structural check owns a zero commitment basis.
            assert_eq!(
                rejection_message::<Cfg>(&outer),
                "commitment group layout requires nonzero outer basis and digit depth"
            );
        } else {
            assert_digit_range_rejection::<Cfg>(&outer, GROUP, "B", &edited);
        }
        let opening = edited_artifact::<Cfg>(|value| {
            first_recursive_group(value)["opening"]["log_basis_open"] = log_basis.into();
        });
        assert_digit_range_rejection::<Cfg>(&opening, GROUP, "opening", &edited);
    }
}

/// A depth beyond the field decomposition is rejected before any digit formula
/// reads it. `u64::MAX` once made admission iterate the balanced-digit series
/// for as many rounds as the artifact asked.
#[test]
fn out_of_range_digit_depths_are_rejected_before_digit_formulas() {
    let edited = format!("digit depth {} ", u64::MAX);
    let root_fold = edited_artifact::<fp128::Dense>(|value| {
        value["rows"][0]["schedule"]["root"]["params"]["groups"]["entries"][0]["opening"]
            ["num_digits_fold"] = u64::MAX.into();
    });
    assert_digit_range_rejection::<fp128::Dense>(&root_fold, "root final group", "fold", &edited);
    let terminal_fold = edited_artifact::<fp128::Dense>(|value| {
        value["rows"][0]["schedule"]["terminal"]["fold"]["num_digits"] = u64::MAX.into();
    });
    assert_digit_range_rejection::<fp128::Dense>(&terminal_fold, "terminal fold", "fold", &edited);

    type Cfg = RecursiveCommitmentConfig<akita_config::proof_optimized::fp32::Dense>;
    for (key, role) in [("num_digits_open", "opening"), ("num_digits_fold", "fold")] {
        let bytes = edited_artifact::<Cfg>(|value| {
            first_recursive_group(value)["opening"][key] = u64::MAX.into();
        });
        assert_digit_range_rejection::<Cfg>(&bytes, "recursive fold 0 final group", role, &edited);
    }
}

#[test]
fn canonical_chunk_ownership_rejects_coverage_preserving_mutations() {
    let catalog = checked_in_catalog::<fp128::OneHotMultiChunk>(
        akita_params::MultiChunkProfileId::W8R2.cfg(),
    );
    let key =
        akita_params::ScheduleLookupKey::single(akita_params::PolynomialGroupLayout::singleton(16));
    let row = catalog.resolve_key(&key).unwrap();
    let policy = catalog.policy();
    let admit = |schedule| {
        akita_config::ResolvedScheduleRow::try_new(row.profiles().clone(), schedule, policy)
    };
    admit(row.schedule().clone()).unwrap();
    let mut mutations = Vec::new();
    let mut changed = row.schedule().clone();
    let consumer = &mut changed.recursive_folds[0].params;
    let old_end = consumer.witness_chunk_ends[0];
    assert!(old_end + 1 < consumer.witness_chunk_ends[1]);
    consumer.witness_chunk_ends[0] += 1;
    let ranges = consumer
        .witness_block_ranges(0, consumer.witness_chunk.num_chunks)
        .unwrap();
    assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
    assert_eq!(ranges.first().unwrap().start, 0);
    assert_eq!(ranges.last().unwrap().end, consumer.blocks().live_blocks);
    mutations.push(changed);
    let mut changed = row.schedule().clone();
    changed.root.params.successor_block_len = changed
        .root
        .params
        .successor_block_len
        .map(|width| width * 2);
    mutations.push(changed);
    let mut changed = row.schedule().clone();
    changed.recursive_folds[0].params.witness_chunk_ends.pop();
    mutations.push(changed);
    let mut changed = row.schedule().clone();
    changed.root.output_witness_len += 64;
    changed.recursive_folds[0].input_witness_len += 64;
    mutations.push(changed);
    // The pre-cutover geometry used proportional recursive ownership and no
    // successor padding. Each half of that old/new combination must reject.
    let mut changed = row.schedule().clone();
    changed.recursive_folds[0].params.witness_chunk_ends.clear();
    mutations.push(changed);
    let mut changed = row.schedule().clone();
    changed.root.params.successor_block_len = None;
    mutations.push(changed);
    for changed in mutations {
        assert!(matches!(
            admit(changed),
            Err(akita_error::AkitaError::InvalidSetup(_))
        ));
    }
}
