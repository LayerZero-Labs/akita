//! Trusted local fixture admission of the exact native-root transition.
//!
//! The measured sizes plan through this builder. The 2^10 smoke uses the
//! oversized ordinary carrier in `catalog.rs`: no ordinary suffix admits its
//! exact native transition.

use super::{catalog, claims, metrics::hex, FullSize, NAME};
use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog};
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
type Key = ScheduleLookupKey;

/// Source size of the smoke, which plans an oversized ordinary carrier.
pub(super) const SMOKE: usize = 10;

fn singleton(log: usize) -> Key {
    ScheduleLookupKey::single(PolynomialGroupLayout::singleton(producer_vars(log)))
}

/// Variables of the ordinary producer and binary carrier profile. The smoke
/// keeps the 18-variable ordinary producer: a 10-variable mixed search has no
/// admitted schedule.
pub(super) fn producer_vars(log: usize) -> usize {
    log.max(18)
}

/// Request and planner identity a fixture catalog was planned for.
#[derive(Serialize, Deserialize, PartialEq)]
struct CatalogIdentity {
    dev_protocol: bool,
    field: String,
    log_carriers: usize,
    shape: String,
    family: String,
    policy_digest: [u8; 32],
    direct_policy_digest: [u8; 32],
}

impl CatalogIdentity {
    fn new(log_carriers: usize, shape: &str) -> Self {
        let (policy, direct_policy) = policies();
        Self {
            dev_protocol: akita_params::DEV_PROTOCOL,
            field: NAME.to_owned(),
            log_carriers,
            shape: shape.to_owned(),
            family: FullSize::schedule_family_name().to_owned(),
            policy_digest: policy_digest(&policy),
            direct_policy_digest: policy_digest(&direct_policy),
        }
    }

    fn describe(&self) -> String {
        format!(
            "dev_protocol={} field={} log_carriers={} shape={} family={} policy_digest={} direct_policy_digest={}",
            self.dev_protocol,
            self.field,
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

/// Private process handoff, never a supported schedule artifact or proof input.
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
    fn check_rows(&self, log: usize, shape: &str) -> Result<()> {
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
        let price = catalog::plan(log, shape, carrier, ordinary, 64)?;
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

    /// The one admission boundary for planned, cached and loaded catalogs:
    /// the request identity, the repriced rows, then the native transition.
    fn admit(self, log: usize, shape: &str) -> Result<(TrustedScheduleCatalog<FullSize>, Key)> {
        self.identity.check(&CatalogIdentity::new(log, shape))?;
        self.check_rows(log, shape)?;
        let key = match self.ordinary.first() {
            Some((profiles, _)) => ScheduleLookupKey {
                final_group: profiles.final_group.group,
                precommitteds: vec![profiles.final_group],
                root_accumulated_setup_prefix_len: 0,
            },
            None => singleton(log),
        };
        let chunking = ChunkedWitnessCfg::default_non_chunked();
        let policy = policy_of::<FullSize>(chunking);
        let transition = NativeRootTransition::new(self.physical_cells, 3, self.prefix_cells)?;
        let catalog = TrustedScheduleCatalog::new(
            ValidatedScheduleCatalog::try_new_native_root(
                FullSize::schedule_family_name(),
                self.ordinary,
                [(self.native.0, self.native.1, transition)],
                &policy,
                FullSize::ring_challenge_config,
            )?,
            chunking,
        )?;
        Ok((catalog, key))
    }
}

/// Reuse a planned fixture catalog from `BINARY_SOURCE_CATALOG_CACHE`.
///
/// Planning is deterministic, so a cached catalog planned for the same
/// protocol, field, size, shape and planner policies is the catalog that
/// planning would produce. Repeated prover measurements use it to skip
/// planning; it is never a supported artifact. An entry is admitted by the
/// identity and rows it contains, never by its filename, and a mismatch is an
/// error.
pub(super) fn build(
    log: usize,
    shape: &str,
    directory: &Path,
) -> Result<(TrustedScheduleCatalog<FullSize>, Key)> {
    if log == SMOKE {
        let (catalog, key) = catalog::build(log, shape)?;
        std::fs::write(
            directory.join("catalog.aks"),
            catalog.catalog().to_artifact_bytes()?,
        )?;
        return Ok((catalog, key));
    }
    let cache = std::env::var_os("BINARY_SOURCE_CATALOG_CACHE").map(|cache| {
        Path::new(&cache).join(format!(
            "native-catalog-{}-{NAME}-{log}-{shape}.json",
            if akita_params::DEV_PROTOCOL {
                "dev"
            } else {
                "current"
            }
        ))
    });
    if let Some((path, bytes)) = cache
        .as_ref()
        .and_then(|path| Some((path, std::fs::read(path).ok()?)))
    {
        let rejected = |error: Box<dyn std::error::Error>| {
            format!(
                "catalog cache entry {} rejected: {error}; delete it to re-plan",
                path.display()
            )
        };
        std::fs::write(directory.join("native-catalog.json"), &bytes)?;
        let fixture: FixtureCatalog =
            serde_json::from_slice(&bytes).map_err(|e| rejected(e.into()))?;
        println!(
            "native_transition physical_cells={} prefix_cells={} catalog_cache=hit",
            fixture.physical_cells, fixture.prefix_cells
        );
        return fixture
            .admit(log, shape)
            .map_err(|error| rejected(error).into());
    }
    let (catalog, key) = plan(log, shape, directory)?;
    if let Some(path) = cache {
        std::fs::copy(directory.join("native-catalog.json"), path)?;
    }
    Ok((catalog, key))
}

/// The fixture policy and the direct policy that plans the ordinary producer.
fn policies() -> (PlannerPolicy, PlannerPolicy) {
    let policy = policy_of::<FullSize>(ChunkedWitnessCfg::default_non_chunked());
    let mut direct_policy = policy;
    direct_policy.recursive_setup_planning = false;
    direct_policy.selection_policy = SelectionPolicyId::for_policy(false);
    direct_policy.recursive_setup_search_policy = RecursiveSetupSearchPolicy::Exhaustive;
    // The optional ordinary native relation is degree bounded by the binary
    // root's 648-coordinate contract. Preserve the ordinary search otherwise.
    direct_policy.ring_dimension_schedule.potential_a_dimensions = &[64, 128, 256, 512];
    (policy, direct_policy)
}

fn plan(
    log: usize,
    shape: &str,
    directory: &Path,
) -> Result<(TrustedScheduleCatalog<FullSize>, Key)> {
    let (policy, direct_policy) = policies();
    let singleton = singleton(log);
    let contract = FullSize::committed_source_contract()?;
    let mut ordinary = Vec::new();
    let producer = if shape == "binary-and-prime-claims-plus-ordinary" {
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
        identity: CatalogIdentity::new(log, shape),
        ordinary,
        native: (profiles, schedule),
        physical_cells: price.outgoing_physical_cells,
        prefix_cells,
    };
    std::fs::write(
        directory.join("native-catalog.json"),
        serde_json::to_vec(&fixture)?,
    )?;
    let (catalog, admitted) = fixture.admit(log, shape)?;
    if admitted != key {
        return Err("native catalog key differs from the planned key".into());
    }
    Ok((catalog, key))
}

pub(super) fn load(
    log: usize,
    shape: &str,
    directory: &Path,
) -> Result<TrustedScheduleCatalog<FullSize>> {
    if log == SMOKE {
        return Ok(TrustedScheduleCatalog::from_artifact_bytes(
            &std::fs::read(directory.join("catalog.aks"))?,
            ChunkedWitnessCfg::default_non_chunked(),
        )?);
    }
    let fixture: FixtureCatalog =
        serde_json::from_slice(&std::fs::read(directory.join("native-catalog.json"))?)?;
    Ok(fixture.admit(log, shape)?.0)
}

/// Plan the fixture catalog and print the root grinding plan it selects,
/// without committing or proving.
pub(super) fn plan_only(log: usize, shape: &str, directory: &Path) -> Result<()> {
    let (catalog, key) = build(log, shape, directory)?;
    let selected = catalog.resolve_key(&key)?;
    let schedule = selected.schedule();
    let ordinary = akita_config::derive_transcript_grinding_plan::<FullSize>(
        schedule,
        &selected.profiles().opening_layout()?,
        None,
    )?;
    let next = schedule
        .recursive_folds
        .first()
        .ok_or("missing ordinary successor")?;
    let domain_len = akita_params::witness_commitment_domain_len(
        schedule.root.output_witness_len,
        next.params.d_a(),
    )?;
    let descriptor = claims::descriptor(log)?;
    let coordinates = descriptor.index_map().columns() * descriptor.polynomials().len();
    let plan = akita_types::binary::binary_root_grinding_plan(
        &ordinary,
        super::FIELD,
        &[coordinates],
        domain_len,
        648,
    )?;
    println!("root_grinding field={NAME} domain_len={domain_len} column_coordinates={coordinates}");
    for run in plan.runs().iter().filter(|run| run.site().level() == 0) {
        println!(
            "root_grinding_site site={:?} kind={:?} loss={} grind_bits={}",
            run.site(),
            run.kind(),
            run.loss_factor(),
            run.grind_bits()
        );
    }
    println!("plan_only=true proof_measured=false");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::CatalogIdentity;

    #[test]
    fn identity_rejects_another_request() {
        let word = CatalogIdentity::new(22, "binary-claim");
        assert!(word
            .check(&CatalogIdentity::new(22, "binary-claim"))
            .is_ok());
        assert!(word
            .check(&CatalogIdentity::new(22, "binary-and-prime-claims"))
            .is_err());
        assert!(word
            .check(&CatalogIdentity::new(23, "binary-claim"))
            .is_err());
    }
}
