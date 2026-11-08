//! Catalog accounting controls for a staged four-group p64 session.

use super::{find_native_root_schedule, NativeRootTransition};
use crate::prime_planes::price_grouped_root;
use akita_config::proof_optimized::fp64::{Dense, Field};
use akita_config::{
    required_native_setup_prefix_slot_ids_for_row, CommitmentConfig, NativeSetupRequirements,
    RecursiveCommitmentConfig, SetupRequirements, TrustedScheduleCatalog,
};
use akita_params::{ChunkedWitnessCfg, OpeningScheduleSelection, ScheduleRowDigest};
use akita_schedules::ValidatedScheduleCatalog;
use akita_types::binary_source::{
    grouped_suffix_sources_fixture, GroupedNativeRootPlan, GroupedPrefixCapacity, GroupedTypedPrefix,
};
use akita_types::AkitaSetupDescriptor;
use std::sync::OnceLock;

type Config = RecursiveCommitmentConfig<Dense>;

struct Fixture {
    catalog: TrustedScheduleCatalog<Config>,
    bindings: Vec<(OpeningScheduleSelection, GroupedPrefixCapacity)>,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let chunking = ChunkedWitnessCfg::default_non_chunked();
        let ordinary =
            akita_config::test_support::workspace_schedule_catalog::<Config>(chunking).unwrap();
        let seed = ordinary
            .rows()
            .find(|row| row.profiles().final_group.group.num_vars() == 25)
            .unwrap();
        let second = ordinary
            .rows()
            .find(|row| row.profiles().final_group.group.num_vars() == 29)
            .unwrap();
        assert_eq!(
            seed.schedule().root.params.role_dims(),
            second.schedule().root.params.role_dims()
        );
        let (binary, signed) = grouped_suffix_sources_fixture();
        let plan = GroupedNativeRootPlan::new(&binary, &signed, 128).unwrap();
        let price = price_grouped_root(&plan).unwrap();
        let transition = NativeRootTransition::new(
            price.outgoing_physical_cells,
            3,
            price.typed_prefix_natural_len,
        )
        .unwrap();
        let schedule = find_native_root_schedule(
            seed,
            transition,
            ordinary.policy(),
            Config::ring_challenge_config,
        )
        .unwrap();
        let first = &schedule.recursive_folds[0].params;
        let plan = GroupedNativeRootPlan::new(&binary, &signed, first.d_a()).unwrap();
        let capacity = GroupedPrefixCapacity::new(
            &plan,
            first.own_group().profile,
            first.setup_prefix().unwrap().slot_id().unwrap(),
        )
        .unwrap();
        let mut second_schedule = schedule.clone();
        second_schedule.root = second.schedule().root.clone();
        second_schedule.root.output_witness_len = price.outgoing_physical_cells;
        // Retain ordinary rows too: their independent precommits and prefix slots
        // must remain covered, even though the selected root uses a typed stream.
        let catalog = TrustedScheduleCatalog::new(
            ValidatedScheduleCatalog::try_new_native_root(
                Config::schedule_family_name(),
                ordinary
                    .rows()
                    .filter(|row| {
                        row.selection() != seed.selection() && row.selection() != second.selection()
                    })
                    .map(|row| (row.profiles().clone(), row.schedule().clone())),
                [
                    (seed.profiles().clone(), schedule, transition),
                    (second.profiles().clone(), second_schedule, transition),
                ],
                ordinary.policy(),
                Config::ring_challenge_config,
            )
            .unwrap(),
            chunking,
        )
        .unwrap();
        let bindings = catalog
            .rows()
            .filter(|row| row.is_native_root())
            .map(|row| (row.selection(), capacity.clone()))
            .collect();
        Fixture { catalog, bindings }
    })
}

fn requirements(fixture: &Fixture) -> NativeSetupRequirements<Field> {
    NativeSetupRequirements::from_catalog(&fixture.catalog, 32, 16, &fixture.bindings).unwrap()
}

fn descriptor(requirements: &NativeSetupRequirements<Field>) -> AkitaSetupDescriptor {
    let requirements = requirements.requirements();
    AkitaSetupDescriptor {
        max_num_vars: requirements.max_num_vars(),
        max_num_batched_polys: requirements.max_num_batched_polys(),
        num_field_elements: requirements.matrix_capacity().num_field_elements,
        setup_seed: [41; 32].into(),
    }
}

#[test]
fn native_setup_accepts_staged_four_group_p64() {
    let fixture = fixture();
    let requirements = requirements(fixture);
    let slots = requirements.requirements().prefix_slot_ids();
    let setup = descriptor(&requirements);
    let bound = requirements.bind_setup(setup.clone(), slots).unwrap();
    assert_eq!(bound.len(), 2);
    assert_eq!(bound[0].0, fixture.bindings[0].0);
    let capacity = &fixture.bindings[0].1;
    let direct = GroupedTypedPrefix::new(
        capacity.plan(),
        *capacity.successor(),
        capacity.first().clone(),
        setup,
    )
    .unwrap();
    assert_eq!(bound[0].1.identity(), direct.identity());
    bound[0].1.validate_plan(capacity.plan()).unwrap();
    let mut expected = std::collections::BTreeSet::new();
    for row in fixture.catalog.rows_within_setup_capacity(32, 16).unwrap() {
        let ids = if row.is_native_root() {
            required_native_setup_prefix_slot_ids_for_row(row, capacity).unwrap()
        } else {
            akita_config::required_setup_prefix_slot_ids_for_schedule(
                row.schedule(),
                &row.profiles().opening_layout().unwrap(),
            )
            .unwrap()
        };
        let scheduled: std::collections::BTreeSet<_> = row
            .schedule()
            .recursive_folds
            .iter()
            .filter_map(|fold| {
                fold.params
                    .setup_prefix()
                    .and_then(|prefix| prefix.slot_id())
            })
            .collect();
        assert_eq!(ids, scheduled.iter().cloned().collect::<Vec<_>>());
        expected.extend(scheduled);
        assert!(
            requirements
                .requirements()
                .matrix_capacity()
                .num_field_elements
                >= akita_params::setup_matrix_capacity_for_schedule(row.schedule())
                    .unwrap()
                    .num_field_elements
        );
    }
    assert_eq!(slots, expected.into_iter().collect::<Vec<_>>());
    assert!(
        requirements
            .requirements()
            .matrix_capacity()
            .num_field_elements
            >= capacity.matrix_capacity().num_field_elements
    );
    assert!(SetupRequirements::from_catalog(&fixture.catalog, 32, 16).is_err());
}

#[test]
fn native_setup_rejects_foreign_row() {
    let fixture = fixture();
    let mut bindings = fixture.bindings.clone();
    bindings[0].0.row_digest = ScheduleRowDigest::from_bytes([0; 32]);
    assert!(NativeSetupRequirements::from_catalog(&fixture.catalog, 32, 16, &bindings).is_err());
    bindings[0].0 = fixture
        .catalog
        .rows()
        .find(|row| !row.is_native_root())
        .unwrap()
        .selection();
    assert!(NativeSetupRequirements::from_catalog(&fixture.catalog, 32, 16, &bindings).is_err());
}

#[test]
fn native_setup_rejects_missing_or_extra_bindings() {
    let fixture = fixture();
    assert!(NativeSetupRequirements::from_catalog(&fixture.catalog, 32, 16, &[]).is_err());
    assert!(NativeSetupRequirements::from_catalog(
        &fixture.catalog,
        32,
        16,
        &fixture.bindings[..1]
    )
    .is_err());
    let mut bindings = fixture.bindings.clone();
    bindings.push(bindings[0].clone());
    assert!(NativeSetupRequirements::from_catalog(&fixture.catalog, 32, 16, &bindings).is_err());
}

#[test]
fn native_setup_rejects_reordered_or_omitted_slots() {
    let fixture = fixture();
    let requirements = requirements(fixture);
    let setup = descriptor(&requirements);
    let slots = requirements.requirements().prefix_slot_ids();
    assert!(slots.len() >= 2);
    let mut reordered = slots.to_vec();
    reordered.swap(0, 1);
    assert!(requirements.bind_setup(setup.clone(), &reordered).is_err());
    for omitted in 0..slots.len() {
        let mut short = slots.to_vec();
        short.remove(omitted);
        assert!(requirements.bind_setup(setup.clone(), &short).is_err());
    }
    let mut extra = slots.to_vec();
    extra.push(slots[0].clone());
    assert!(requirements.bind_setup(setup, &extra).is_err());
}

#[test]
fn native_setup_rejects_first_prefix_mismatch() {
    let fixture = fixture();
    let row = fixture
        .catalog
        .resolve_selection(fixture.bindings[0].0)
        .unwrap();
    let capacity = &fixture.bindings[0].1;
    let mut schedule = row.schedule().clone();
    let first = &mut schedule.recursive_folds[0].params;
    let mut prefix = *first.setup_prefix().unwrap();
    prefix.setup_natural_len = Some(capacity.first().natural_len - 1);
    first.set_setup_prefix(Some(prefix)).unwrap();
    let transition = NativeRootTransition::new(
        schedule.root.output_witness_len,
        3,
        capacity.first().natural_len - 1,
    )
    .unwrap();
    let foreign = transition
        .resolve(row.profiles().clone(), schedule, fixture.catalog.policy())
        .unwrap();
    assert!(required_native_setup_prefix_slot_ids_for_row(&foreign, capacity).is_err());
}

#[test]
fn native_setup_rejects_short_suffix_capacity() {
    let fixture = fixture();
    let requirements = requirements(fixture);
    let mut setup = descriptor(&requirements);
    let local = fixture.bindings[0].1.matrix_capacity().num_field_elements;
    let row = fixture
        .catalog
        .resolve_selection(fixture.bindings[0].0)
        .unwrap();
    let mut suffix = 1;
    for fold in &row.schedule().recursive_folds {
        akita_params::accumulate_matrix_field_elements_for_level(&fold.params, &mut suffix)
            .unwrap();
    }
    akita_params::accumulate_terminal_matrix_field_elements(&row.schedule().terminal, &mut suffix)
        .unwrap();
    assert!(
        suffix > local,
        "fixture must exercise capacity beyond the grouped root"
    );
    setup.num_field_elements = suffix - 1;
    GroupedTypedPrefix::bind_capacity(fixture.bindings[0].1.clone(), setup.clone()).unwrap();
    assert!(requirements
        .bind_setup(setup, requirements.requirements().prefix_slot_ids())
        .is_err());
}

#[test]
fn native_setup_refuses_ordinary_serialization() {
    assert!(fixture().catalog.to_artifact_bytes().is_err());
}

#[test]
fn native_setup_rejects_reordered_bindings() {
    let fixture = fixture();
    let mut bindings = fixture.bindings.clone();
    bindings.reverse();
    assert!(NativeSetupRequirements::from_catalog(&fixture.catalog, 32, 16, &bindings).is_err());
}

#[test]
fn native_setup_rejects_conflicting_prefix_streams() {
    let fixture = fixture();
    let mut bindings = fixture.bindings.clone();
    let (mut binary, signed) = grouped_suffix_sources_fixture();
    binary.reverse();
    let capacity = &bindings[1].1;
    let plan = GroupedNativeRootPlan::new(
        &binary,
        &signed,
        capacity.plan().repacking().destination_ring_dimension(),
    )
    .unwrap();
    bindings[1].1 =
        GroupedPrefixCapacity::new(&plan, *capacity.successor(), capacity.first().clone()).unwrap();
    assert!(NativeSetupRequirements::from_catalog(&fixture.catalog, 32, 16, &bindings).is_err());
}

#[test]
fn native_setup_rejects_successor_profile_mismatch() {
    let fixture = fixture();
    let mut bindings = fixture.bindings.clone();
    let capacity = &bindings[0].1;
    let mut successor = *capacity.successor();
    let matrix = successor.outer.matrix;
    let key = matrix.sis_table_key();
    successor.outer.matrix = akita_params::OuterCommitMatrixParams::try_new(
        key.policy,
        key.table_digest,
        key.modulus_profile,
        matrix.output_rank() + 1,
        matrix.input_width(),
        key.coeff_linf_bound,
        usize::try_from(key.ring_dimension).unwrap(),
    )
    .unwrap();
    let changed =
        GroupedPrefixCapacity::new(capacity.plan(), successor, capacity.first().clone()).unwrap();
    // Check the row directly so a registry collision cannot mask the comparison.
    let row = fixture.catalog.resolve_selection(bindings[0].0).unwrap();
    assert!(required_native_setup_prefix_slot_ids_for_row(row, &changed).is_err());
    bindings[0].1 = changed;
    assert!(NativeSetupRequirements::from_catalog(&fixture.catalog, 32, 16, &bindings).is_err());
}
