//! Per-source producer planning for explicitly selected signed native sources.

use akita_challenges::FoldChallengeFamily;
use akita_error::AkitaError;
use akita_params::ordinary_source::OrdinarySourceWidth;
use akita_params::{
    PlannedFoldSchedule, PolynomialGroupLayout, ScheduleLookupKey, SisModulusProfileId,
};

use crate::{find_schedule, PlannerPolicy};

/// Plan a standalone P-native producer with its own exact width and slot count.
///
/// This uses the ordinary planner and its existing SIS tables without changing
/// any catalog. The result is a producer schedule, not admission of a mixed
/// root: that requires the complete source-indexed extraction/work ledger.
pub fn plan_signed_ordinary_source(
    group: PolynomialGroupLayout,
    width: OrdinarySourceWidth,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<FoldChallengeFamily, AkitaError>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    // Validate the caller's complete domain before selecting a sorted prefix.
    akita_schedules::planner_support::validate_policy(policy)?;
    if policy.sis_modulus_profile != SisModulusProfileId::Q128Offset275
        || policy.claim_ext_degree != 1
        || policy.chal_ext_degree != 1
        || !(policy.inner_basis_range.0..=policy.inner_basis_range.1).contains(&3)
        || !(policy.opening_basis_range.0..=policy.opening_basis_range.1).contains(&3)
        || !policy
            .ring_dimension_schedule
            .potential_b_dimensions
            .contains(&64)
        || !policy
            .ring_dimension_schedule
            .suffix_dimensions
            .contains(&64)
    {
        return Err(AkitaError::InvalidSetup(
            "signed native producer requires the prime field and base-eight policy".into(),
        ));
    }
    let mut selected = *policy;
    selected.decomposition = width.decomposition();
    selected.inner_basis_range = (3, 3);
    selected.opening_basis_range = (3, 3);
    // Restrict the caller's dimension domain to the native compiler's current
    // A<=512/B64 geometry; never introduce a dimension absent from its policy.
    let a_dimensions = policy.ring_dimension_schedule.potential_a_dimensions;
    let a_count = a_dimensions.partition_point(|&dimension| dimension <= 512);
    selected.ring_dimension_schedule.potential_a_dimensions = a_dimensions
        .get(..a_count)
        .ok_or(AkitaError::InvalidProof)?;
    selected.ring_dimension_schedule.potential_b_dimensions = &[64];
    selected.ring_dimension_schedule.suffix_dimensions = &[64];
    let planned = find_schedule(
        &ScheduleLookupKey::single(group),
        width.producer_contract()?,
        &[],
        &selected,
        ring_challenge_config,
    )?;
    width.validate_profile(&planned.schedule.root.params.own_group().profile)?;
    Ok(planned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::{policy_of, proof_optimized::fp128::DenseBounded, CommitmentConfig};
    use akita_params::ChunkedWitnessCfg;

    #[test]
    fn unsorted_dimension_domain_rejects_before_prefix_selection() {
        let mut policy = policy_of::<DenseBounded>(ChunkedWitnessCfg::default_non_chunked());
        policy.sis_modulus_profile = SisModulusProfileId::Q128Offset275;
        policy.inner_basis_range = (3, 3);
        policy.opening_basis_range = (3, 3);
        policy.ring_dimension_schedule.potential_a_dimensions = &[64, 256, 1024];
        policy.ring_dimension_schedule.potential_b_dimensions = &[64];
        policy.ring_dimension_schedule.potential_d_dimensions = &[64];
        policy.ring_dimension_schedule.suffix_dimensions = &[64];
        // Validate the same dimensions in sorted order so no unsupported
        // dimension or unrelated policy error can satisfy this regression.
        akita_schedules::planner_support::validate_policy(&policy).unwrap();
        // An unchecked partition can discard the trailing admissible dimension
        // together with 1024 and hide the invalid caller policy from the planner.
        policy.ring_dimension_schedule.potential_a_dimensions = &[64, 1024, 256];
        let error = plan_signed_ordinary_source(
            PolynomialGroupLayout::singleton(18),
            OrdinarySourceWidth::new(9).unwrap(),
            &policy,
            DenseBounded::ring_challenge_config,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            AkitaError::InvalidSetup(message)
                if message == "adaptive A domain must be strictly sorted and duplicate-free"
        ));
    }
}
