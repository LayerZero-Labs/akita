//! Trusted local fixture admission of the exact native-root transition.

use super::{catalog, config::FullSize};
use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog};
use akita_params::{
    ChunkedWitnessCfg, CommittedGroupBatchProfile, FoldSchedule, PolynomialGroupLayout,
    ScheduleLookupKey, SetupContributionMode,
};
use akita_planner::native_root::{find_native_root_schedule, NativeRootTransition};
use akita_schedules::{
    RecursiveSetupSearchPolicy, ResolvedScheduleRow, SelectionPolicyId, ValidatedScheduleCatalog,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Row = (CommittedGroupBatchProfile, FoldSchedule);

/// Private process handoff, never a supported schedule artifact or proof input.
#[derive(Serialize, Deserialize)]
struct FixtureCatalog {
    ordinary: Vec<Row>,
    native: Row,
    physical_cells: usize,
    prefix_cells: usize,
}

impl FixtureCatalog {
    fn admit(self) -> Result<TrustedScheduleCatalog<FullSize>> {
        let chunking = ChunkedWitnessCfg::default_non_chunked();
        let policy = policy_of::<FullSize>(chunking);
        let transition = NativeRootTransition::new(self.physical_cells, 3, self.prefix_cells)?;
        Ok(TrustedScheduleCatalog::new(
            ValidatedScheduleCatalog::try_new_native_root(
                FullSize::schedule_family_name(),
                self.ordinary,
                [(self.native.0, self.native.1, transition)],
                &policy,
                FullSize::ring_challenge_config,
            )?,
            chunking,
        )?)
    }
}

pub(super) fn build(
    log: usize,
    shape: &str,
    directory: &Path,
) -> Result<(TrustedScheduleCatalog<FullSize>, ScheduleLookupKey)> {
    let policy = policy_of::<FullSize>(ChunkedWitnessCfg::default_non_chunked());
    let mut direct_policy = policy;
    direct_policy.recursive_setup_planning = false;
    direct_policy.selection_policy = SelectionPolicyId::for_policy(false);
    direct_policy.recursive_setup_search_policy = RecursiveSetupSearchPolicy::Exhaustive;
    // The optional ordinary native relation is degree bounded by the binary
    // root's 648-coordinate contract. Preserve the ordinary search otherwise.
    direct_policy.ring_dimension_schedule.potential_a_dimensions = &[64, 128, 256, 512];
    let singleton = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(log));
    let contract = FullSize::committed_source_contract()?;
    let mut ordinary = Vec::new();
    let producer = if shape == "word-prime-ordinary" {
        Some(akita_planner::find_schedule(
            &singleton,
            contract,
            &[],
            &direct_policy,
            FullSize::ring_challenge_config,
        )?)
    } else {
        None
    };
    let ordinary_profile = producer
        .as_ref()
        .map(|row| row.schedule.root.params.own_group().profile);
    let key = if let Some(profile) = ordinary_profile {
        ordinary.push((
            CommittedGroupBatchProfile {
                final_group: profile,
                precommitteds: vec![],
            },
            producer
                .as_ref()
                .ok_or("missing ordinary producer")?
                .schedule
                .clone(),
        ));
        ScheduleLookupKey {
            final_group: profile.group,
            precommitteds: vec![profile],
            root_accumulated_setup_prefix_len: 0,
        }
    } else {
        singleton
    };
    let seed = akita_planner::find_schedule_with_root_setup_mode(
        &key,
        contract,
        &vec![contract; key.precommitteds.len()],
        &policy,
        FullSize::ring_challenge_config,
        SetupContributionMode::Recursive,
        producer.as_ref().map(|row| &row.schedule.root.params),
    )?;
    let profiles = CommittedGroupBatchProfile {
        final_group: seed.schedule.root.params.own_group().profile,
        precommitteds: key.precommitteds.clone(),
    };
    let carrier = profiles
        .precommitteds
        .first()
        .unwrap_or(&profiles.final_group);
    let price = catalog::plan(log, shape, carrier, ordinary_profile, 64)?;
    let prefix_cells = price.typed_prefix.natural_len;
    let transition = NativeRootTransition::new(price.outgoing_physical_cells, 3, prefix_cells)?;
    let seed = ResolvedScheduleRow::try_new(profiles.clone(), seed.schedule, &policy)?;
    let schedule =
        find_native_root_schedule(&seed, transition, &policy, FullSize::ring_challenge_config)?;
    let dimension = schedule
        .recursive_folds
        .first()
        .ok_or("missing native successor")?
        .params
        .d_a();
    let actual_price = catalog::plan(log, shape, carrier, ordinary_profile, dimension)?;
    if actual_price.outgoing_physical_cells != price.outgoing_physical_cells {
        return Err("native physical witness changed with the destination dimension".into());
    }
    println!("native_transition physical_cells={} prefix_cells={prefix_cells} canonical_plan_matches=true", price.outgoing_physical_cells);
    let fixture = FixtureCatalog {
        ordinary,
        native: (profiles, schedule),
        physical_cells: price.outgoing_physical_cells,
        prefix_cells,
    };
    std::fs::write(
        directory.join("native-catalog.json"),
        serde_json::to_vec(&fixture)?,
    )?;
    Ok((fixture.admit()?, key))
}

pub(super) fn load(directory: &Path) -> Result<TrustedScheduleCatalog<FullSize>> {
    let fixture: FixtureCatalog =
        serde_json::from_slice(&std::fs::read(directory.join("native-catalog.json"))?)?;
    fixture.admit()
}
