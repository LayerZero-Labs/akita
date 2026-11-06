//! Trusted local fixture admission of the exact native-root transition.

use super::{catalog, config::MeasurementConfig, metrics::hex};
use akita_config::{policy_of, TrustedScheduleCatalog};
use akita_params::{
    ChunkedWitnessCfg, CommittedGroupBatchProfile, FoldSchedule, PolynomialGroupLayout,
    ScheduleLookupKey, SetupContributionMode,
};
use akita_planner::native_root::{find_native_root_schedule, NativeRootTransition};
use akita_schedules::{
    policy_digest, PlannerPolicy, RecursiveSetupSearchPolicy, ResolvedScheduleRow,
    SelectionPolicyId, ValidatedScheduleCatalog,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Row = (CommittedGroupBatchProfile, FoldSchedule);

/// Request and planner identity a fixture catalog was planned for.
#[derive(Serialize, Deserialize, PartialEq)]
struct CatalogIdentity {
    dev_protocol: bool,
    log_carriers: usize,
    shape: String,
    family: String,
    policy_digest: [u8; 32],
    direct_policy_digest: [u8; 32],
}

impl CatalogIdentity {
    fn new<C: MeasurementConfig>(log_carriers: usize, shape: &str) -> Self {
        let (policy, direct_policy) = policies::<C>();
        Self {
            dev_protocol: akita_params::DEV_PROTOCOL,
            log_carriers,
            shape: shape.to_owned(),
            family: C::schedule_family_name().to_owned(),
            policy_digest: policy_digest(&policy),
            direct_policy_digest: policy_digest(&direct_policy),
        }
    }

    fn describe(&self) -> String {
        format!(
            "dev_protocol={} log_carriers={} shape={} family={} policy_digest={} direct_policy_digest={}",
            self.dev_protocol,
            self.log_carriers,
            self.shape,
            self.family,
            hex(&self.policy_digest),
            hex(&self.direct_policy_digest)
        )
    }

    fn check(&self, requested: &Self) -> Result<()> {
        if self != requested {
            return Err(format!(
                "native catalog was planned for {}, not the requested {}",
                self.describe(),
                requested.describe()
            )
            .into());
        }
        Ok(())
    }
}

/// Trusted research handoff and replay fixture, never a supported schedule artifact.
/// Checked-in copies belong under `crates/akita-pcs/tests/fixtures/`, outside the
/// production artifact catalogs. Every load revalidates identity and native rows.
#[derive(Serialize, Deserialize)]
struct FixtureCatalog {
    identity: CatalogIdentity,
    ordinary: Vec<Row>,
    native: Row,
    physical_cells: usize,
    prefix_cells: usize,
}

impl FixtureCatalog {
    /// Re-price the binary root from the catalog's own rows. The witness
    /// size depends on the shape and size, so rows planned for another
    /// geometry disagree with the stored transition even under a matching
    /// identity.
    fn check_rows<C: MeasurementConfig>(&self, log: usize, shape: &str) -> Result<()> {
        let profiles = &self.native.0;
        let carrier = profiles
            .precommitteds
            .first()
            .unwrap_or(&profiles.final_group);
        let ordinary = self
            .ordinary
            .first()
            .map(|(profiles, _)| profiles.final_group);
        if ordinary.is_some() != (shape == "binary-and-prime-claims-plus-ordinary") {
            return Err(format!("native catalog ordinary rows do not match shape {shape}").into());
        }
        let price = catalog::plan::<C>(log, shape, carrier, ordinary, 64)?;
        if price.outgoing_physical_cells != self.physical_cells
            || price.typed_prefix.natural_len != self.prefix_cells
        {
            return Err(format!(
                "native catalog stores physical_cells={} prefix_cells={}, but log_carriers={log} shape={shape} needs {} and {}",
                self.physical_cells,
                self.prefix_cells,
                price.outgoing_physical_cells,
                price.typed_prefix.natural_len
            )
            .into());
        }
        Ok(())
    }

    fn admit<C: MeasurementConfig>(
        self,
        log: usize,
        shape: &str,
    ) -> Result<TrustedScheduleCatalog<C>> {
        self.identity.check(&CatalogIdentity::new::<C>(log, shape))?;
        self.check_rows::<C>(log, shape)?;
        let chunking = ChunkedWitnessCfg::default_non_chunked();
        let policy = policy_of::<C>(chunking);
        let transition = NativeRootTransition::new(self.physical_cells, 3, self.prefix_cells)?;
        Ok(TrustedScheduleCatalog::new(
            ValidatedScheduleCatalog::try_new_native_root(
                C::schedule_family_name(),
                self.ordinary,
                [(self.native.0, self.native.1, transition)],
                &policy,
                C::ring_challenge_config,
            )?,
            chunking,
        )?)
    }
}

/// Reuse the fixture directory's `native-catalog.json` first, then the optional
/// `BINARY_SOURCE_CATALOG_CACHE`, and plan only when neither exists.
///
/// This precedence applies to P128 too, including repeated invocations in a used
/// directory. Use a fresh directory and unset the cache to measure fresh planning.
/// Cache filenames include the field-specific family, so legacy P128 cache names
/// are not reused. Identity and row validation establish admissibility, not that
/// a cached row equals the current planner's preferred output. A mismatch is an
/// error; a valid alternate row can select different setup and proof bytes.
pub(super) fn build<C: MeasurementConfig>(
    log: usize,
    shape: &str,
    directory: &Path,
) -> Result<(TrustedScheduleCatalog<C>, ScheduleLookupKey)> {
    let cache = std::env::var_os("BINARY_SOURCE_CATALOG_CACHE").map(|cache| {
        Path::new(&cache).join(format!(
            "native-catalog-{}-{}-{log}-{shape}.json",
            C::schedule_family_name(),
            if akita_params::DEV_PROTOCOL {
                "dev"
            } else {
                "current"
            }
        ))
    });
    let local = directory.join("native-catalog.json");
    let cached = if local.is_file() {
        Some((local.clone(), std::fs::read(&local)?))
    } else {
        cache
            .as_ref()
            .and_then(|path| std::fs::read(path).ok().map(|bytes| (path.clone(), bytes)))
    };
    if let Some((path, bytes)) = cached {
        let rejected = |error: Box<dyn std::error::Error>| {
            format!(
                "catalog cache entry {} rejected: {error}; delete it to re-plan",
                path.display()
            )
        };
        std::fs::write(directory.join("native-catalog.json"), &bytes)?;
        let fixture: FixtureCatalog =
            serde_json::from_slice(&bytes).map_err(|e| rejected(e.into()))?;
        let key = match fixture.ordinary.first() {
            Some((profiles, _)) => ScheduleLookupKey {
                final_group: profiles.final_group.group,
                precommitteds: vec![profiles.final_group],
                root_accumulated_setup_prefix_len: 0,
            },
            None => ScheduleLookupKey::single(PolynomialGroupLayout::singleton(log.max(18))),
        };
        println!(
            "native_transition physical_cells={} prefix_cells={} catalog_cache=hit",
            fixture.physical_cells, fixture.prefix_cells
        );
        return Ok((fixture.admit::<C>(log, shape).map_err(rejected)?, key));
    }
    let (catalog, key) = plan::<C>(log, shape, directory)?;
    if let Some(path) = cache {
        std::fs::copy(directory.join("native-catalog.json"), path)?;
    }
    Ok((catalog, key))
}

/// The fixture policy and the direct policy that plans the ordinary producer.
fn policies<C: MeasurementConfig>() -> (PlannerPolicy, PlannerPolicy) {
    let policy = policy_of::<C>(ChunkedWitnessCfg::default_non_chunked());
    let mut direct_policy = policy;
    direct_policy.recursive_setup_planning = false;
    direct_policy.selection_policy = SelectionPolicyId::for_policy(false);
    direct_policy.recursive_setup_search_policy = RecursiveSetupSearchPolicy::Exhaustive;
    // The optional ordinary native relation is degree bounded by the binary
    // root's 648-coordinate contract. Preserve the ordinary search otherwise.
    direct_policy.ring_dimension_schedule.potential_a_dimensions = &[64, 128, 256, 512];
    (policy, direct_policy)
}

fn plan<C: MeasurementConfig>(
    log: usize,
    shape: &str,
    directory: &Path,
) -> Result<(TrustedScheduleCatalog<C>, ScheduleLookupKey)> {
    let (policy, direct_policy) = policies::<C>();
    let singleton = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(log.max(18)));
    let contract = C::committed_source_contract()?;
    let mut ordinary = Vec::new();
    let producer = if shape == "binary-and-prime-claims-plus-ordinary" {
        Some(akita_planner::find_schedule(
            &singleton,
            contract,
            &[],
            &direct_policy,
            C::ring_challenge_config,
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
        C::ring_challenge_config,
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
    let price = catalog::plan::<C>(log, shape, carrier, ordinary_profile, 64)?;
    let prefix_cells = price.typed_prefix.natural_len;
    let transition = NativeRootTransition::new(price.outgoing_physical_cells, 3, prefix_cells)?;
    let seed = ResolvedScheduleRow::try_new(profiles.clone(), seed.schedule, &policy)?;
    let schedule =
        find_native_root_schedule(&seed, transition, &policy, C::ring_challenge_config)?;
    let dimension = schedule
        .recursive_folds
        .first()
        .ok_or("missing native successor")?
        .params
        .d_a();
    let actual_price = catalog::plan::<C>(log, shape, carrier, ordinary_profile, dimension)?;
    if actual_price.outgoing_physical_cells != price.outgoing_physical_cells {
        return Err("native physical witness changed with the destination dimension".into());
    }
    println!("native_transition physical_cells={} prefix_cells={prefix_cells} canonical_plan_matches=true", price.outgoing_physical_cells);
    let fixture = FixtureCatalog {
        identity: CatalogIdentity::new::<C>(log, shape),
        ordinary,
        native: (profiles, schedule),
        physical_cells: price.outgoing_physical_cells,
        prefix_cells,
    };
    std::fs::write(
        directory.join("native-catalog.json"),
        serde_json::to_vec(&fixture)?,
    )?;
    Ok((fixture.admit::<C>(log, shape)?, key))
}

pub(super) fn load<C: MeasurementConfig>(
    log: usize,
    shape: &str,
    directory: &Path,
) -> Result<TrustedScheduleCatalog<C>> {
    let fixture: FixtureCatalog =
        serde_json::from_slice(&std::fs::read(directory.join("native-catalog.json"))?)?;
    fixture.admit::<C>(log, shape)
}

#[cfg(test)]
mod tests {
    use super::CatalogIdentity;
    use crate::run::config::{P128, P64};

    #[test]
    fn identity_rejects_another_request() {
        let word = CatalogIdentity::new::<P64>(22, "binary-claim");
        assert!(word
            .check(&CatalogIdentity::new::<P128>(22, "binary-claim"))
            .is_err());
        assert!(word
            .check(&CatalogIdentity::new::<P64>(22, "binary-claim"))
            .is_ok());
        assert!(word
            .check(&CatalogIdentity::new::<P64>(22, "binary-and-prime-claims"))
            .is_err());
        assert!(word
            .check(&CatalogIdentity::new::<P64>(23, "binary-claim"))
            .is_err());
    }
}
