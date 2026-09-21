//! Reproduce full-search timings on frozen Aerie mixed-opening producer profiles.
//!
//! With `catalog-gen` enabled, pass `joint OUTPUT_DESCRIPTOR ARTIFACT INDEX`.
//! The artifact is Aerie's `composed_fp64_d128_bound6.aks` at revision
//! `688eeae2c96f3a2c54f02963e3c55dc3d5fba4b0`. Indices 0 and 3 select the
//! two benchmark keys after sorting by summed producer arities. The source
//! contracts below follow that artifact's CT/Falcon/W1 producer order.
//! Compilation, artifact loading, and descriptor serialization are excluded
//! from the printed elapsed time. Compare descriptors as well as timings.

use akita_config::proof_optimized::fp64::{Dense, ExtensionField, Field};
use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig, RingDimensionScheduleMode};
use akita_types::{
    AkitaScheduleLookupKey, DecompositionParams, PolynomialGroupLayout, SisModulusProfileId,
};
#[derive(Clone)]
struct Probe<const B: u32>;
impl<const B: u32> CommitmentConfig for Probe<B> {
    type Field = Field;
    type ExtField = ExtensionField;
    const RING_DIMENSION_SCHEDULE_MODE: RingDimensionScheduleMode =
        RingDimensionScheduleMode::UniformDimension {
            ring_dimension: 128,
        };
    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: B,
            log_open_bound: Some(64),
        }
    }
    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, akita_error::AkitaError> {
        Dense::ring_challenge_config(d)
    }
    fn sis_modulus_profile() -> SisModulusProfileId {
        SisModulusProfileId::Q64Offset59
    }
    fn opening_basis_range() -> (u32, u32) {
        (3, 6)
    }
    fn inner_basis_range() -> (u32, u32) {
        (3, 11)
    }
    fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
        akita_types::sis::CommittedSourceClass::BalancedSignedDigit
    }
    fn schedule_family_name() -> &'static str {
        "composed_fp64_d128_bound6"
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = &args[1];
    let key = if mode == "joint" {
        let bytes = std::fs::read(&args[3]).unwrap();
        let cat =
            akita_config::TrustedScheduleCatalog::<Probe<6>>::from_artifact_bytes(&bytes).unwrap();
        let mut rows = cat
            .rows()
            .filter(|r| r.profiles().precommitteds.len() == 10)
            .collect::<Vec<_>>();
        rows.sort_by_key(|r| {
            r.profiles()
                .precommitteds
                .iter()
                .map(|p| p.group.num_vars())
                .sum::<usize>()
        });
        let index = args
            .get(4)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let r = rows[index];
        AkitaScheduleLookupKey {
            final_group: r.profiles().final_group.group,
            precommitteds: r.profiles().precommitteds.clone(),
        }
    } else {
        AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
            14,
            if mode == "scalar" { 1 } else { 4 },
        ))
    };
    let policy = policy_of::<Probe<6>>();
    let honest = honest_fold_policy_of::<Probe<6>>();
    let policies = (0..key.precommitteds.len())
        .map(|i| {
            if (1..=2).contains(&i) {
                Probe::<18>::committed_source_contract().unwrap()
            } else {
                Probe::<6>::committed_source_contract().unwrap()
            }
        })
        .collect::<Vec<_>>();
    let start = std::time::Instant::now();
    let result = akita_planner::find_schedule(
        &key,
        honest,
        &policies,
        &policy,
        Probe::<6>::ring_challenge_config,
    )
    .unwrap();
    let elapsed = start.elapsed();
    result.schedule.validate_structure().unwrap();
    std::fs::write(&args[2], result.schedule.canonical_descriptor_bytes()).unwrap();
    println!(
        "{mode}: elapsed={elapsed:?} groups={} polys={} estimate={:?}",
        key.precommitteds.len() + 1,
        key.final_group.num_polynomials(),
        result.estimate
    );
}
