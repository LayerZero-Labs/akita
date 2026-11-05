//! Per-source producer planning for explicitly selected signed native sources.

use akita_params::FoldChallengeDistribution;
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
    ring_challenge_config: impl Fn(usize) -> Result<FoldChallengeDistribution, AkitaError> + Sync,
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
    use akita_params::sis::{CommittedSourceClass, CommittedSourceContract};
    use akita_params::{ChunkedWitnessCfg, DecompositionParams};

    #[test]
    fn signed_producer_planning_preserves_exact_contract_and_dimension_domain() {
        // Use the existing DenseBounded planning policy and sorted domain from
        // the rejection regression, retaining its security and cost policies.
        let mut policy = policy_of::<DenseBounded>(ChunkedWitnessCfg::default_non_chunked());
        policy.sis_modulus_profile = SisModulusProfileId::Q128Offset275;
        policy.ring_dimension_schedule.potential_a_dimensions = &[64, 256, 1024];
        policy.ring_dimension_schedule.potential_b_dimensions = &[64];
        policy.ring_dimension_schedule.potential_d_dimensions = &[64];
        policy.ring_dimension_schedule.suffix_dimensions = &[64];
        akita_schedules::planner_support::validate_policy(&policy).unwrap();
        let group = PolynomialGroupLayout::new(18, 3);
        for (bits, digits) in [(8, 3), (9, 4)] {
            let width = OrdinarySourceWidth::new(bits).unwrap();
            let decomposition = DecompositionParams {
                log_basis: 3,
                log_commit_bound: bits,
                log_open_bound: Some(128),
            };
            let contract = CommittedSourceContract::try_new(
                CommittedSourceClass::BalancedSignedDigit,
                decomposition,
            )
            .unwrap();
            let offset = 1u128 << (bits - 1);
            assert_eq!(contract.declared_bounds(), (Some(offset), Some(offset - 1)));
            assert_eq!(width.producer_contract().unwrap(), contract);
            let planned = plan_signed_ordinary_source(
                group,
                width,
                &policy,
                DenseBounded::ring_challenge_config,
            )
            .unwrap();
            let schedule = &planned.schedule;
            schedule.validate_structure().unwrap();
            let profile = &schedule.root.params.own_group().profile;
            assert_eq!(profile.group, group);
            assert_eq!(profile.inner.digits.log_basis, 3);
            assert_eq!(profile.inner.digits.num_digits, digits);
            assert_eq!(profile.outer.digits.log_basis, 3);
            assert_eq!(profile.outer.digits.num_digits, 43);
            width.validate_profile(profile).unwrap();
            for fold in std::iter::once(&schedule.root).chain(&schedule.recursive_folds) {
                let dimensions = fold.params.role_dims();
                let domain = policy.ring_dimension_schedule;
                assert!(domain.potential_a_dimensions.contains(&dimensions.d_a()));
                assert!(domain.potential_b_dimensions.contains(&dimensions.d_b()));
                assert!(domain.potential_d_dimensions.contains(&dimensions.d_d()));
                assert!(dimensions.d_a() <= 512);
                assert_eq!(dimensions.d_b(), 64);
                assert_eq!(dimensions.d_d(), 64);
            }
            assert_eq!(schedule.terminal.d_a(), 64);
            assert!(policy
                .ring_dimension_schedule
                .suffix_dimensions
                .contains(&schedule.terminal.d_a()));

            // The result stores geometry, not its source contract. Compare it
            // with ordinary planning under the independently declared bounds.
            let mut expected_policy = policy;
            expected_policy.decomposition = decomposition;
            expected_policy.inner_basis_range = (3, 3);
            expected_policy.opening_basis_range = (3, 3);
            expected_policy
                .ring_dimension_schedule
                .potential_a_dimensions = &[64, 256];
            let expected = find_schedule(
                &ScheduleLookupKey::single(group),
                contract,
                &[],
                &expected_policy,
                DenseBounded::ring_challenge_config,
            )
            .unwrap();
            assert_eq!(planned.schedule, expected.schedule);
        }
    }

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
