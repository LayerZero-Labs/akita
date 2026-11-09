//! Catalog accounting controls for a staged four-group p64 session.

use super::{find_native_root_schedule, NativeRootTransition};
use crate::prime_planes::price_grouped_root;
#[path = "native_test_fixture.rs"]
mod resources;
use resources::*;

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
    // The native suffix shares the grouped root's padded typed prefix. Use the
    // retained ordinary row's larger suffix to exercise the catalog envelope.
    let row = fixture
        .catalog
        .rows()
        .find(|row| !row.is_native_root() && row.profiles().final_group.group.num_vars() == 27)
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


#[test]
fn native_setup_accepts_actual_eight_profiles() {
    let fixture = actual_profiles::actual_fixture();
    let row = fixture.catalog.resolve_selection(fixture.bindings[0].0).unwrap();
    assert_eq!(row.profiles().precommitteds.len(), 7);
    assert_eq!(row.profiles().final_group.inner.digits.num_digits, 12);
    assert_eq!(fixture.catalog.policy().decomposition.log_commit_bound, 64);
    assert_eq!(fixture.catalog.resolve_profiles(row.profiles(), 0).unwrap().selection(),
        row.selection());
    row.validate_opening_layout(&row.profiles().opening_layout().unwrap()).unwrap();
    let old = super::setup_tests::fixture();
    let retained = old.catalog.resolve_selection(old.bindings[0].0).unwrap().schedule();
    assert_eq!(row.schedule().recursive_folds, retained.recursive_folds);
    assert_eq!(row.schedule().terminal, retained.terminal);
    let requirements = requirements(fixture);
    let slots = required_native_setup_prefix_slot_ids_for_row(row, &fixture.bindings[0].1).unwrap();
    assert!(slots.iter().all(|slot| requirements.requirements().prefix_slot_ids().contains(slot)));
    let bound = requirements.bind_setup(descriptor(&requirements),
        requirements.requirements().prefix_slot_ids()).unwrap();
    bound[0].1.validate_plan(fixture.bindings[0].1.plan()).unwrap();
    assert!(akita_schedules::ResolvedScheduleRow::try_new(
        row.profiles().clone(), row.schedule().clone(), fixture.catalog.policy(),
    ).is_err());
}

#[test]
fn native_setup_rejects_carrier_only_profiles() {
    let actual = actual_profiles::actual_fixture();
    let old = fixture();
    let carrier = old.catalog.resolve_selection(old.bindings[0].0).unwrap().profiles();
    assert!(actual.catalog.resolve_profiles(carrier, 0).is_err());
    let row = actual.catalog.resolve_selection(actual.bindings[0].0).unwrap();
    assert!(row.validate_opening_layout(&carrier.opening_layout().unwrap()).is_err());
}

#[test]
fn native_setup_rejects_swapped_source_profiles() {
    let fixture = actual_profiles::actual_fixture();
    let row = fixture.catalog.resolve_selection(fixture.bindings[0].0).unwrap();
    let mut profiles = row.profiles().clone();
    profiles.precommitteds.swap(4, 5);
    assert_eq!(profiles.opening_layout().unwrap(), row.profiles().opening_layout().unwrap());
    assert!(fixture.catalog.resolve_profiles(&profiles, 0).is_err());
    let (mut binary, signed) = actual_profiles::sources();
    binary.swap(0, 1);
    let capacity = &fixture.bindings[0].1;
    let changed = GroupedNativeRootPlan::new(&binary, &signed,
        capacity.plan().repacking().destination_ring_dimension()).unwrap();
    let changed = GroupedPrefixCapacity::new(&changed, *capacity.successor(),
        capacity.first().clone()).unwrap();
    assert!(required_native_setup_prefix_slot_ids_for_row(row, &changed).is_err());
}

#[test]
fn native_setup_rejects_changed_signed_width() {
    let fixture = actual_profiles::actual_fixture();
    let (binary, mut signed) = actual_profiles::sources();
    let original = signed[3].0;
    signed[3].1 = akita_params::ordinary_source::OrdinarySourceWidth::new(34).unwrap();
    signed[3].1.validate_profile(&original).unwrap();
    let capacity = &fixture.bindings[0].1;
    let changed = GroupedNativeRootPlan::new(&binary, &signed,
        capacity.plan().repacking().destination_ring_dimension()).unwrap();
    assert_eq!(*changed.ordinary().unwrap().plans()[3].profile(), original);
    let requirements = requirements(fixture);
    let binding = requirements.bind_setup(
        descriptor(&requirements), requirements.requirements().prefix_slot_ids(),
    ).unwrap().remove(0).1;
    assert!(binding.validate_plan(&changed).is_err());
}

#[test]
fn native_setup_rejects_p128_source_profile() {
    let fixture = actual_profiles::actual_fixture();
    let row = fixture.catalog.resolve_selection(fixture.bindings[0].0).unwrap();
    let mut profiles = row.profiles().clone();
    profiles.final_group = akita_types::binary_source::binary_ordinary_profile_fixture(
        akita_params::binary_setup::BinaryFieldProfile::P128, Some((33, 1)),
    );
    profiles.final_group.validate_frozen_precommit(128).unwrap();
    assert!(fixture.catalog.resolve_profiles(&profiles, 0).is_err());
    let sources = row.grouped_root().unwrap();
    assert!(akita_schedules::native_root::NativeRootSources::new(
        profiles, *sources.signed_widths(), sources.identity().to_vec(),
    ).is_err());
}

#[path = "native_open_controls.rs"]
mod native_open_controls;

#[test]
fn native_open_accepts_staged_four_group_p64() {
    let opening = native_open_fixture::opening();
    assert!(!opening.proof.is_empty());
    opening.verify(&opening.claims, &opening.setup, &opening.proof).unwrap();
    let mut trailing = opening.proof.clone();
    trailing.push(0);
    assert!(opening.verify(&opening.claims, &opening.setup, &trailing).is_err());
}

#[test]
fn native_open_rejects_foreign_prefix_context() {
    native_open_controls::reject_foreign_context(false);
}

#[test]
fn native_open_rejects_substituted_plan() {
    native_open_controls::reject_substituted_plan();
}

#[test]
fn native_open_rejects_source_order() {
    native_open_controls::reject_source_order();
}

#[test]
fn native_open_rejects_wrong_field() {
    native_open_controls::reject_wrong_field();
}

#[test]
fn native_open_rejects_short_setup_capacity() {
    native_open_controls::reject_short_capacity();
}

#[test]
fn native_open_rejects_tampered_value_point_0() {
    native_open_controls::reject_values(0);
}

#[test]
fn native_open_rejects_tampered_value_point_1() {
    native_open_controls::reject_values(1);
}

#[test]
fn native_open_rejects_tampered_typed_prefix_identity() {
    native_open_controls::reject_foreign_context(true);
}

#[path = "native_weight_controls.rs"]
mod native_weight_controls;

#[test]
fn native_open_translated_weights_match_assembled_oracle() {
    native_weight_controls::compare(false);
}

#[test]
fn native_open_assembled_prime_readings_match_coefficient_oracle() {
    native_weight_controls::compare(true);
}
