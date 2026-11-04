//! Explicit root-edge selection through the ordinary schedule search.

use akita_error::AkitaError;
use akita_params::SetupContributionMode;
use akita_params::{sis::CommittedSourceContract, PlannedFoldSchedule, ScheduleLookupKey};

/// Extension seam: select the root setup edge while reusing ordinary planning.
/// Tracking issue: <https://github.com/LayerZero-Labs/akita/issues/285>.
///
/// All commitment, response, suffix and terminal geometries are rebuilt by the
/// canonical search. Recursive edges require a policy that admits setup
/// offloading. Existing unrestricted entry points retain their search domain.
pub fn find_schedule_with_root_setup_mode(
    key: &ScheduleLookupKey,
    final_source_contract: CommittedSourceContract,
    precommitted_source_contracts: &[CommittedSourceContract],
    policy: &crate::PlannerPolicy,
    ring_challenge_config: impl Fn(
        usize,
    )
        -> Result<akita_challenges::FoldChallengeDistribution, AkitaError>,
    root_setup_mode: SetupContributionMode,
    root_main_constraint: Option<&akita_params::CommittedGroupParams>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    if root_setup_mode == SetupContributionMode::Recursive && !policy.recursive_setup_planning {
        return Err(AkitaError::InvalidInput(
            "recursive root setup edge requires recursive setup planning".into(),
        ));
    }
    crate::planner::find_schedule_in_relation_order(
        key,
        final_source_contract,
        precommitted_source_contracts,
        policy,
        ring_challenge_config,
        crate::planner::ScheduleSearchOptions {
            root_setup_mode: Some(root_setup_mode),
            root_main_constraint,
            ..crate::planner::ScheduleSearchOptions::canonical()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::{proof_optimized::fp128::DenseBounded, CommitmentConfig};

    #[test]
    fn recursive_root_setup_requires_an_admitting_policy() {
        let policy = akita_config::policy_of::<DenseBounded>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        );
        let result = find_schedule_with_root_setup_mode(
            &ScheduleLookupKey::single(akita_params::PolynomialGroupLayout::singleton(18)),
            DenseBounded::committed_source_contract().unwrap(),
            &[],
            &policy,
            DenseBounded::ring_challenge_config,
            SetupContributionMode::Recursive,
            None,
        );
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }
}
