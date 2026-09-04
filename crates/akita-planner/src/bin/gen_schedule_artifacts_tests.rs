use super::*;

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
        selected_families(one.family_filter.as_deref())
            .iter()
            .map(|family| family.family_name())
            .collect::<Vec<_>>(),
        vec!["fp32_dense"],
    );

    let multiple = parse_args_from(vec![
        "generated".into(),
        "fp64_dense".into(),
        "fp32_dense".into(),
    ])
    .expect("known families");
    assert_eq!(
        selected_families(multiple.family_filter.as_deref())
            .iter()
            .map(|family| family.family_name())
            .collect::<Vec<_>>(),
        vec!["fp64_dense", "fp32_dense"],
    );

    let all = parse_args_from(vec!["generated".into()]).expect("all families");
    assert!(all.family_filter.is_none());
    assert_eq!(selected_families(None).len(), ALL_GENERATED_FAMILIES.len());

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
    )
    .expect("explicit emit spec");

    assert_eq!(spec.keys, vec![PolynomialGroupLayout::new(14, 1)]);
    assert!(spec.grouped_requests.is_empty());
}

#[test]
fn explicit_rows_accept_adaptive_families() {
    let args = parse_args_from(vec![
        "/tmp/akita-adaptive-explicit-test".into(),
        "--final-group".into(),
        "fp64_dense:8:1".into(),
    ])
    .expect("adaptive explicit family");

    assert_eq!(
        args.family_filter.as_deref(),
        Some(&["fp64_dense".into()][..])
    );
}

#[test]
fn explicit_group_rejects_source_metadata() {
    assert!(parse_explicit_group("fp128_onehot:14:1:256").is_err());
}

#[test]
fn strict_catalog_baseline_requires_a_baseline_path() {
    let error = parse_args_from(vec![
        "out".into(),
        "--require-catalog-baseline-match".into(),
    ])
    .err()
    .expect("strict baseline matching without a baseline must reject");
    assert!(error.contains("requires --catalog-baseline"));
}

#[test]
fn every_grouped_artifact_precommit_has_a_shipped_scalar_producer() {
    let artifact_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/schedules");
    let catalogs = ALL_GENERATED_FAMILIES
        .iter()
        .map(|family| {
            let path = artifact_dir.join(format!("{}.aks", family.family_name()));
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            let catalog = akita_schedules::TrustedScheduleCatalog::from_artifact_bytes(
                &bytes,
                family.family_name(),
                &(family.policy)(),
                family.ring_challenge_config,
            )
            .unwrap_or_else(|error| panic!("failed to load {}: {error}", family.family_name()));
            (family.family_name(), catalog)
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    let scalar_producers = catalogs
        .iter()
        .flat_map(|(family, catalog)| {
            catalog
                .rows()
                .filter(|row| row.profiles().precommitteds.is_empty())
                .map(move |row| (*family, row.profiles().final_group))
        })
        .collect::<Vec<_>>();

    for (family, catalog) in &catalogs {
        for row in catalog.rows() {
            for precommitted in &row.profiles().precommitteds {
                assert!(
                    scalar_producers
                        .iter()
                        .any(|(_, producer)| producer == precommitted),
                    "family {family} contains a grouped precommit without an exact shipped scalar producer: {:?}",
                    precommitted.group,
                );
            }
        }
    }

    for (recursive_family, base_family) in [
        ("fp128_onehot_recursive", "fp128_onehot"),
        (
            "fp128_onehot_recursive_multi_chunk_w8r2",
            "fp128_onehot_multi_chunk",
        ),
    ] {
        let recursive = catalogs
            .get(recursive_family)
            .unwrap_or_else(|| panic!("missing recursive artifact {recursive_family}"));
        let base_producers = scalar_producers
            .iter()
            .filter(|(family, _)| *family == base_family)
            .map(|(_, producer)| producer)
            .collect::<Vec<_>>();
        for row in recursive.rows() {
            for precommitted in &row.profiles().precommitteds {
                assert!(
                    base_producers.contains(&precommitted),
                    "recursive artifact {recursive_family} precommit is not produced by base family {base_family}: {:?}",
                    precommitted.group,
                );
            }
        }
    }
}
