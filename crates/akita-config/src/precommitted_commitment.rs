//! Generated standalone commitment profiles.

use crate::{policy_of, CommitmentConfig};
use akita_field::AkitaError;
use akita_types::{CommittedGroupProfile, PolynomialGroupLayout};

/// Resolve the generated standalone precommit profile for one group.
pub fn committed_group_profile<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<CommittedGroupProfile, AkitaError> {
    Cfg::validate_sis_modulus_profile()?;
    akita_schedules::resolve_generated_precommitted_group_profile(
        key,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::schedule_catalog(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;

    #[test]
    fn same_layout_can_resolve_config_specific_profiles() {
        let key = PolynomialGroupLayout::new(16, 1);
        let dense = committed_group_profile::<fp128::Dense>(&key).expect("dense profile");
        let one_hot = committed_group_profile::<fp128::OneHot>(&key).expect("one-hot profile");
        assert_ne!(
            dense, one_hot,
            "commitment config must affect standalone commitment parameters"
        );
    }

    #[test]
    fn dense_precommit_profile_uses_independent_basis() {
        let key = PolynomialGroupLayout::new(15, 2);
        let profile = committed_group_profile::<fp128::Dense>(&key).expect("dense profile");
        assert_eq!(profile.inner_commit_matrix.ring_dimension(), 64);
        assert_eq!(profile.outer_commit_matrix.ring_dimension(), 64);
        assert!(profile.log_basis_inner > profile.log_basis_outer);
    }
}
