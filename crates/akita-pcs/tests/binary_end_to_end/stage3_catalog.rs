//! A selected root-offloading catalog for the binary fixtures.

use super::{BinaryTestSuffix, CommitmentConfig, TrustedScheduleCatalog};
use akita_config::policy_of;
use akita_params::{
    ChunkedWitnessCfg, PolynomialGroupLayout, ScheduleLookupKey, SetupContributionMode,
};
use akita_schedules::{RecursiveSetupSearchPolicy, SelectionPolicyId, ValidatedScheduleCatalog};
use std::sync::LazyLock;

fn artifact_path() -> std::path::PathBuf {
    let directory = if akita_params::DEV_PROTOCOL {
        "schedules-binary-stage3-dev"
    } else {
        "schedules-binary-stage3"
    };
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts")
        .join(directory)
        .join(format!("{}.aks", BinaryTestSuffix::schedule_family_name()))
}

pub(super) fn catalog() -> TrustedScheduleCatalog<BinaryTestSuffix> {
    static CATALOG: LazyLock<TrustedScheduleCatalog<BinaryTestSuffix>> = LazyLock::new(load);
    CATALOG.clone()
}

fn load() -> TrustedScheduleCatalog<BinaryTestSuffix> {
    let path = artifact_path();
    if std::env::var("AKITA_REGENERATE_STAGE3_CATALOG").as_deref() == Ok("1") {
        let catalog = plan::<BinaryTestSuffix>();
        std::fs::create_dir_all(path.parent().expect("artifact directory"))
            .expect("create fixture artifact directory");
        std::fs::write(
            path,
            catalog
                .catalog()
                .to_artifact_bytes()
                .expect("encode fixture"),
        )
        .expect("write regenerated fixture artifact");
        return catalog;
    }
    TrustedScheduleCatalog::from_artifact_bytes(
        &std::fs::read(path).expect("root-offloaded fixture artifact"),
        ChunkedWitnessCfg::default_non_chunked(),
    )
    .expect("root-offloaded fixture catalog")
}

/// Plan the binary fixture rows for one ordinary suffix configuration: an
/// 18-variable root, a 23-variable carrier, and the mixed root over both.
pub(super) fn plan<Cfg: CommitmentConfig>() -> TrustedScheduleCatalog<Cfg> {
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let policy = policy_of::<Cfg>(chunking);
    assert!(policy.recursive_setup_planning);
    let mut direct_policy = policy;
    direct_policy.recursive_setup_planning = false;
    direct_policy.selection_policy = SelectionPolicyId::for_policy(false);
    direct_policy.recursive_setup_search_policy = RecursiveSetupSearchPolicy::Exhaustive;
    let contract = Cfg::committed_source_contract().expect("fixture contract");
    let singleton = |vars| ScheduleLookupKey::single(PolynomialGroupLayout::singleton(vars));
    let direct_rows: Vec<_> = [23, 18]
        .into_iter()
        .map(|vars| {
            let planned = akita_planner::find_schedule(
                &singleton(vars),
                contract,
                &[],
                &direct_policy,
                Cfg::ring_challenge_config,
            )
            .expect("direct fixture schedule");
            (
                akita_params::CommittedGroupBatchProfile {
                    final_group: planned.schedule.root.params.own_group().profile,
                    precommitteds: Vec::new(),
                },
                planned.schedule,
            )
        })
        .collect();
    let direct = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        direct_rows,
        &direct_policy,
        Cfg::ring_challenge_config,
    )
    .expect("direct fixture catalog");
    let carrier = direct.resolve_key(&singleton(23)).expect("carrier row");
    let mixed_key = ScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(18),
        precommitteds: vec![carrier.profiles().final_group],
        root_accumulated_setup_prefix_len: 0,
    };
    // The larger singleton is only a commitment producer for the mixed root.
    let mut rows = vec![(carrier.profiles().clone(), carrier.schedule().clone())];
    for key in [singleton(18), mixed_key] {
        let main = direct
            .resolve_key(&ScheduleLookupKey::single(key.final_group))
            .expect("direct fixture main row");
        let baseline = if key.precommitteds.is_empty() {
            main.schedule().clone()
        } else {
            let producers = key
                .precommitteds
                .iter()
                .map(|&profile| {
                    akita_planner::emit::PrecommittedProducer::try_new(profile, contract)
                        .expect("fixture producer contract")
                })
                .collect();
            akita_planner::find_adapted_schedule(
                main,
                &akita_planner::emit::GroupedGenerationRequest::new(key.final_group, producers),
                contract,
                &direct_policy,
                Cfg::ring_challenge_config,
            )
            .expect("direct grouped fixture schedule")
            .schedule
        };
        let planned = akita_planner::find_schedule_with_root_setup_mode(
            &key,
            contract,
            &vec![contract; key.precommitteds.len()],
            &policy,
            Cfg::ring_challenge_config,
            SetupContributionMode::Recursive,
            (!key.precommitteds.is_empty()).then_some(&baseline.root.params),
        )
        .expect("root-offloaded fixture schedule");
        let prefix = planned.schedule.recursive_folds[0]
            .params
            .setup_prefix()
            .expect("selected root prefix");
        assert!(prefix.n_prefix().expect("prefix domain") >= 4096);
        eprintln!(
            "stage3 catalog groups={} root_output={} prefix={} folds={}",
            1 + key.precommitteds.len(),
            planned.schedule.root.output_witness_len,
            prefix.n_prefix().expect("prefix domain"),
            planned.schedule.recursive_folds.len() + 2,
        );
        let root_slot = prefix.slot_id().expect("selected root slot");
        assert!(planned.schedule.recursive_folds.iter().skip(1).all(|fold| {
            fold.params
                .setup_prefix()
                .and_then(|prefix| prefix.slot_id())
                .is_none_or(|slot| slot != root_slot)
        }));
        let profiles = akita_params::CommittedGroupBatchProfile {
            final_group: planned.schedule.root.params.own_group().profile,
            precommitteds: key.precommitteds.clone(),
        };
        rows.push((profiles, planned.schedule));
    }
    let catalog = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        rows,
        &policy,
        Cfg::ring_challenge_config,
    )
    .expect("root-offloaded fixture catalog");
    TrustedScheduleCatalog::new(catalog, chunking).expect("root-offloaded fixture binding")
}

#[test]
fn stage3_catalog_consumes_each_selected_root_prefix() {
    let catalog = catalog();
    let mut offloaded_roots = 0;
    for row in catalog.catalog().rows() {
        if row.schedule().recursive_folds[0]
            .params
            .setup_prefix()
            .is_some()
        {
            offloaded_roots += 1;
        }
        akita_config::required_setup_prefix_slot_ids_for_schedule(
            row.schedule(),
            &row.profiles().opening_layout().unwrap(),
        )
        .expect("all scheduled prefixes have matching producer footprints");
    }
    assert_eq!(offloaded_roots, 2);
}

#[test]
fn stage3_p64_typed_prefix_identity_differs_from_p128() {
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_types::binary::{binary_word_manifest_fixture, BinaryTypedPrefix};
    let row = catalog()
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(18),
        ))
        .unwrap()
        .clone();
    let slot = row.schedule().recursive_folds[0]
        .params
        .setup_prefix()
        .unwrap()
        .slot_id()
        .unwrap();
    let legacy = binary_word_manifest_fixture(BinaryFieldProfile::P128, 1);
    let p64 = binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
    let old = BinaryTypedPrefix::new(&legacy, 0, 64).unwrap();
    let new = BinaryTypedPrefix::new(&p64, 0, 64).unwrap();
    let old_id = old.identity(&legacy, &slot).unwrap();
    let new_id = new.identity(&p64, &slot).unwrap();
    assert!(old_id.starts_with(b"akita/binary-source/typed-prefix/a-then-digit-pieces/v3"));
    assert!(new_id.starts_with(b"akita/binary-source/typed-prefix/a-then-digit-pieces/p64-e64/v4"));
    assert_ne!(old_id, new_id);
    assert!(new.identity(&legacy, &slot).is_err());
    assert!(old.identity(&p64, &slot).is_err());
}
