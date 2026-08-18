use super::*;
use std::path::Path;

#[test]
fn same_revision_drift_is_reported_before_baseline_comparison() {
    assert!(should_emit_catalog_drift_report(false, 0));
    assert!(should_emit_catalog_drift_report(false, 3));
    assert!(!should_emit_catalog_drift_report(true, 0));
    assert!(should_emit_catalog_drift_report(true, 3));
}

#[test]
fn positional_family_filters_are_checked_and_ordered() {
    let one = parse_args_from(vec!["generated".into(), "fp32_dense".into()]).expect("known family");
    assert_eq!(
        one.family_filter.as_deref(),
        Some(&["fp32_dense".into()][..])
    );
    assert_eq!(
        selected_families(one.family_filter.as_deref())
            .iter()
            .map(|family| family.module_name)
            .collect::<Vec<_>>(),
        vec!["fp32_dense"],
    );

    let multiple = parse_args_from(vec![
        "generated".into(),
        "fp64_dense".into(),
        "fp32_dense".into(),
    ])
    .expect("known families");
    let selected = selected_families(multiple.family_filter.as_deref())
        .iter()
        .map(|family| family.module_name)
        .collect::<Vec<_>>();
    assert_eq!(selected, vec!["fp64_dense", "fp32_dense"]);

    let all = parse_args_from(vec!["generated".into()]).expect("all families");
    assert!(all.family_filter.is_none());
    assert_eq!(selected_families(None).len(), ALL_GENERATED_FAMILIES.len());

    let progress = parse_args_from(vec![
        "generated".into(),
        "--row-progress".into(),
        "fp32_dense".into(),
    ])
    .expect("row progress");
    assert!(progress.row_progress);

    let report = parse_args_from(vec![
        "generated".into(),
        "--check-catalog".into(),
        "--catalog-report".into(),
        "report.tsv".into(),
    ]);
    if cfg!(feature = "catalog-check") {
        assert_eq!(
            report.expect("catalog report").catalog_report,
            Some(PathBuf::from("report.tsv"))
        );
    } else {
        assert!(report
            .err()
            .expect("catalog check feature")
            .contains("catalog-check"));
    }
    assert!(parse_args_from(vec![
        "generated".into(),
        "--catalog-report".into(),
        "report.tsv".into(),
    ])
    .err()
    .expect("report requires comparison")
    .contains("requires --check-catalog or --catalog-baseline"));

    let baseline = parse_args_from(vec![
        "generated".into(),
        "--catalog-baseline".into(),
        "base.tsv".into(),
        "--catalog-report".into(),
        "report.tsv".into(),
    ])
    .expect("revision comparison");
    assert_eq!(baseline.catalog_baseline, Some(PathBuf::from("base.tsv")));
    assert_eq!(baseline.catalog_report, Some(PathBuf::from("report.tsv")));
    assert!(parse_args_from(vec![
        "generated".into(),
        "--catalog-baseline".into(),
        "base.tsv".into(),
        "fp32_dense".into(),
    ])
    .err()
    .expect("partial revision comparison must reject")
    .contains("complete generated family set"));

    let unknown = parse_args_from(vec!["generated".into(), "not_a_family".into()])
        .err()
        .expect("unknown family must reject");
    assert!(unknown.contains("unknown schedule family"));
}

#[test]
fn explicit_scalar_sweep_replaces_default_catalog_work() {
    let family = family_by_name("fp128_onehot").expect("known family");
    let explicit_rows = ExplicitRows {
        final_group: Some(parse_explicit_group("fp128_onehot:14:1").expect("explicit group")),
        precommitted_groups: Vec::new(),
    };

    let spec = emit_spec_with_overrides(
        family,
        &GenerationPreplans::default(),
        PathBuf::from("generated"),
        &explicit_rows,
        "generator command",
    )
    .expect("explicit emit spec");

    assert_eq!(spec.keys, vec![PolynomialGroupLayout::new(14, 1)]);
    assert!(spec.group_batch_keys.is_empty());
    assert_eq!(spec.generator_command, "generator command");
}

#[test]
fn explicit_group_rejects_source_metadata() {
    assert!(parse_explicit_group("fp128_onehot:14:1:256").is_err());
}

#[cfg(feature = "catalog-check")]
#[test]
fn catalog_comparison_reports_complete_key_union() {
    let family = family_by_name("fp32_dense").expect("known family");
    let table = (family.schedule_catalog)().expect("compiled fp32 dense table");
    let spec = wiring_emit_spec(family, PathBuf::from("generated"));
    let entries = table
        .entries
        .iter()
        .copied()
        .map(|entry| {
            let key = entry.to_runtime_lookup_key();
            let schedule = akita_schedules::schedule_from_entry(
                &entry,
                &key,
                &spec.policy,
                spec.ring_challenge_config,
            )
            .expect("expand compiled row");
            assert_eq!(
                akita_schedules::expanded_schedule_proof_payload_bytes(
                    &key,
                    &schedule,
                    &spec.policy,
                )
                .expect("expanded proof payload"),
                akita_schedules::estimate_proof_bytes(
                    &entry,
                    &key,
                    &spec.policy,
                    spec.ring_challenge_config,
                )
                .expect("generated proof payload"),
            );
            (key, schedule)
        })
        .collect::<Vec<_>>();

    let equal = compare_materialized_catalog(&spec, table, &entries).expect("equal report");
    assert_eq!(equal.changed_rows, 0);
    assert_eq!(equal.report.matches("\tequal\t").count(), entries.len());
    assert!(CATALOG_DRIFT_REPORT_HEADER.ends_with("compiled_policy\tregenerated_policy\n"));
    assert!(equal
        .report
        .lines()
        .all(|line| line.split('\t').count() == 13));

    let removed = compare_materialized_catalog(&spec, table, &entries[..entries.len() - 1])
        .expect("removed report");
    assert_eq!(removed.changed_rows, 1);
    assert!(removed.report.contains("\tremoved\t"));

    let empty_table = akita_schedules::GeneratedScheduleTable {
        entries: &[],
        identity: table.identity,
    };
    let added =
        compare_materialized_catalog(&spec, empty_table, &entries[..1]).expect("added report");
    assert_eq!(added.changed_rows, 1);
    assert!(added.report.contains("\tadded\t"));

    let mut changed_entries = entries.clone();
    changed_entries[0].1.root.input_witness_len += 1;
    let changed =
        compare_materialized_catalog(&spec, table, &changed_entries).expect("changed report");
    assert_eq!(changed.changed_rows, 1);
    assert!(changed.report.contains("\tchanged\t"));

    let snapshot = catalog_snapshot::write_snapshot(
        materialized_snapshot_rows(&spec, &entries).expect("snapshot rows"),
    )
    .expect("write snapshot");
    let parsed = catalog_snapshot::parse_snapshot(&snapshot).expect("parse snapshot");
    assert_eq!(parsed.len(), entries.len());
    assert!(parsed.iter().all(|row| {
        row.logical_key.starts_with("final=")
            && row.policy.contains("partial=")
            && row.policy.contains("quotient=")
    }));
}

#[cfg(feature = "catalog-check")]
#[test]
fn generated_w8r2_row_preserves_the_two_level_packing_boundary() {
    use akita_types::OpeningMethod;

    let family =
        family_by_name("fp128_onehot_recursive_multi_chunk_w8r2").expect("known W8R2 family");
    let table = (family.schedule_catalog)().expect("compiled W8R2 table");
    assert_eq!(table.entries.len(), 1);
    let entry = table.entries[0];
    let key = entry.to_runtime_lookup_key();
    let spec = wiring_emit_spec(family, PathBuf::from("generated"));
    let expand = || {
        akita_schedules::schedule_from_entry(&entry, &key, &spec.policy, spec.ring_challenge_config)
            .expect("expand W8R2 row")
    };
    let schedule = expand();
    assert_eq!(schedule, expand(), "generated replay must be deterministic");
    schedule.validate_structure().expect("valid W8R2 schedule");

    assert_eq!(
        schedule.root.params.final_group.commitment.opening_method,
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        },
    );
    assert_eq!(schedule.root.params.precommitted_groups.len(), 2);
    let shared_opening_dimension = schedule
        .root
        .params
        .final_group
        .commitment
        .role_dims()
        .d_d();
    let mut expected_precommit_signatures = Vec::new();
    for (index, group) in schedule.root.params.precommitted_groups.iter().enumerate() {
        let OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } = group.commitment.opening.opening_method
        else {
            panic!("packing root must use packing for every precommitted group");
        };
        let d_a = group.commitment.layout.inner_commit_matrix.ring_dimension();
        let geometry = akita_types::SubringCoefficientPackingGeometry::try_new(
            spec.policy.claim_ext_degree,
            d_a,
            challenge_subring_dimension,
        )
        .expect("admissible group-local packing geometry");
        assert!(geometry
            .partial_base_field_width()
            .is_multiple_of(shared_opening_dimension));
        group
            .commitment
            .d_segment_width(spec.policy.claim_ext_degree, shared_opening_dimension)
            .expect("group-local packing width at the shared D dimension");
        expected_precommit_signatures.push(format!(
            "pre{index}=PACK,s={},h={},partial={},quotient={},src=canonical,dA={d_a},sec=Linf",
            geometry.challenge_subring_dimension(),
            geometry.packing_factor(),
            geometry.partial_base_field_width(),
            geometry.partial_base_field_width(),
        ));
    }

    let first_recursive = schedule
        .recursive_folds
        .first()
        .expect("W8R2 row has a recursive packing fold");
    assert_eq!(
        first_recursive.params.witness.opening_method,
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        },
    );
    assert_eq!(
        first_recursive
            .params
            .incoming_setup_prefix
            .as_ref()
            .expect("first recursive fold consumes the setup prefix")
            .commitment_params
            .opening
            .opening_method,
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        },
    );
    assert!(schedule.recursive_folds[1..]
        .iter()
        .all(|fold| fold.params.witness.opening_method == OpeningMethod::EvaluationTrace));

    let policy_signature =
        catalog_policy_signature(&spec, &schedule).expect("W8R2 policy signature");
    assert!(policy_signature.contains("L0[chunks=8@2,eor=0,in="));
    assert!(policy_signature
        .contains("witness=PACK,s=64,h=4,partial=64,quotient=64,src=canonical,dA=256,sec=Linf"));
    for expected in expected_precommit_signatures {
        assert!(policy_signature.contains(&expected));
    }
    assert!(policy_signature.contains("L1[chunks=8@2,eor=0,in="));
    assert!(policy_signature.contains("prefix=PACK,s=64"));
    assert!(policy_signature.contains("L2[chunks=1@0,eor=0,in="));
    assert!(policy_signature.contains("witness=ET,s=-,h=-,partial=-,quotient=-"));
    assert!(policy_signature.contains("/T[method=ET,src=canonical,eor=0,input="));
    assert!(!policy_signature.contains(['\t', '\n']));

    let terminal_eor = akita_types::extension_opening_reduction_level_bytes(
        spec.policy
            .challenge_field_bits()
            .expect("challenge field bits"),
        spec.policy.claim_ext_degree,
        akita_types::PolynomialGroupLayout::singleton(
            akita_types::padded_boolean_opening_vars(schedule.terminal.input_witness_len)
                .expect("terminal opening vars"),
        ),
    )
    .expect("terminal EOR price");
    assert_eq!(
        terminal_eor, 0,
        "the fp128 base-field terminal follows the ET/EOR pricing path, whose width-one reduction is empty",
    );
    assert_eq!(
        akita_schedules::expanded_schedule_proof_payload_bytes(&key, &schedule, &spec.policy,)
            .expect("expanded proof payload"),
        akita_schedules::estimate_proof_bytes(
            &entry,
            &key,
            &spec.policy,
            spec.ring_challenge_config,
        )
        .expect("generated proof payload"),
    );

    assert_eq!(
        source_encoding_signature(
            akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
                extension_degree: 2,
            }
        ),
        "tensor-k2",
    );
    assert_ne!(
        source_encoding_signature(
            akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
                extension_degree: 2,
            }
        ),
        source_encoding_signature(
            akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
                extension_degree: 4,
            }
        ),
    );

    let mut activation_changed = schedule.clone();
    activation_changed
        .root
        .params
        .final_group
        .commitment
        .witness_chunk
        .num_activated_levels = 1;
    assert_ne!(
        policy_signature,
        catalog_policy_signature(&spec, &activation_changed).expect("activation policy signature"),
    );

    let mut input_changed = schedule.clone();
    input_changed.root.input_witness_len += 1;
    assert_ne!(
        policy_signature,
        catalog_policy_signature(&spec, &input_changed).expect("input-length policy signature"),
    );
}

#[test]
fn explicit_sweeps_reject_the_checked_in_generated_tree() {
    let explicit_rows = ExplicitRows {
        final_group: Some(parse_explicit_group("fp128_onehot:14:1").expect("explicit group")),
        precommitted_groups: Vec::new(),
    };
    let checked_in_generated_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../akita-schedules/src/generated");

    let error = validate_explicit_output_isolation(
        &checked_in_generated_dir.join("diagnostic"),
        &explicit_rows,
    )
    .expect_err("checked-in generated tree must be protected");
    assert!(error.contains("isolated output directory"));

    let isolated = env::temp_dir().join(format!(
        "akita-explicit-schedule-test-{}",
        std::process::id()
    ));
    validate_explicit_output_isolation(&isolated, &explicit_rows)
        .expect("isolated explicit output");
    validate_explicit_output_isolation(&checked_in_generated_dir, &ExplicitRows::default())
        .expect("ordinary full regeneration may target the checked-in catalog");
}

#[cfg(unix)]
#[test]
fn output_resolution_applies_parent_after_resolving_symlink() {
    use std::os::unix::fs::symlink;

    let root = env::temp_dir().join(format!(
        "akita-schedule-path-test-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let target = root.join("real/deep");
    fs::create_dir_all(&target).expect("create symlink target");
    symlink(&target, root.join("link")).expect("create test symlink");

    let resolved = resolved_output_path(&root.join("link/../isolated"))
        .expect("resolve output through symlink");
    let canonical_root = fs::canonicalize(&root).expect("canonical test root");
    assert_eq!(resolved, canonical_root.join("real/isolated"));

    fs::remove_dir_all(&root).expect("remove test directory");
}
