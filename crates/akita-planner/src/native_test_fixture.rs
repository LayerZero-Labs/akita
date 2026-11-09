//! Shared setup resources for grouped opening controls.
#![allow(unused_imports)]

use super::{find_native_root_schedule, price_grouped_root, NativeRootTransition};
pub(crate) use akita_config::proof_optimized::fp64::{Dense, Field};
pub(crate) use akita_config::{
    required_native_setup_prefix_slot_ids_for_row, CommitmentConfig, NativeSetupRequirements,
    RecursiveCommitmentConfig, SetupRequirements, TrustedScheduleCatalog,
};
pub(crate) use akita_params::{ChunkedWitnessCfg, OpeningScheduleSelection, ScheduleRowDigest};
pub(crate) use akita_schedules::ValidatedScheduleCatalog;
pub(crate) use akita_types::binary_source::{
    grouped_suffix_sources_fixture, GroupedNativeRootPlan, GroupedPrefixCapacity,
    GroupedTypedPrefix,
};
pub(crate) use akita_types::AkitaSetupDescriptor;
pub(crate) use std::sync::OnceLock;

pub(crate) type Config = RecursiveCommitmentConfig<Dense>;

pub(crate) struct Fixture {
    pub(crate) catalog: TrustedScheduleCatalog<Config>,
    pub(crate) bindings: Vec<(OpeningScheduleSelection, GroupedPrefixCapacity)>,
}

pub(crate) fn fixture() -> &'static Fixture {
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
            transition.clone(),
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
                    (seed.profiles().clone(), schedule, transition.clone()),
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

pub(crate) fn requirements(fixture: &Fixture) -> NativeSetupRequirements<Field> {
    NativeSetupRequirements::from_catalog(&fixture.catalog, 32, 16, &fixture.bindings).unwrap()
}

pub(crate) fn descriptor(requirements: &NativeSetupRequirements<Field>) -> AkitaSetupDescriptor {
    let requirements = requirements.requirements();
    AkitaSetupDescriptor {
        max_num_vars: requirements.max_num_vars(),
        max_num_batched_polys: requirements.max_num_batched_polys(),
        num_field_elements: requirements.matrix_capacity().num_field_elements,
        setup_seed: [41; 32].into(),
    }
}

#[path = "native_actual_profiles.rs"]
pub(crate) mod actual_profiles;

#[path = "native_open_fixture.rs"]
pub(crate) mod native_open_fixture;

#[path = "native_source_data.rs"]
pub(crate) mod native_source_data;
