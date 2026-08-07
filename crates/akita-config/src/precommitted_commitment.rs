//! Exact precommitment config adapter.
//!
//! This adapter is for staggered workflows that need ordinary commit calls to
//! freeze the A/source and B/outer commitment layout before the final multi-group
//! root is known. The root basis is deterministic from the base config's runtime
//! catalog policy, so precommitments use the exact root layout rather than a
//! worst-case envelope over every supported basis.

use crate::{policy_of, CommitmentConfig};
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::{
    accumulate_matrix_field_elements_for_level, CommitmentRingDims, CommittedGroupParams,
    CommittedGroupProfile, DecompositionParams, FoldSchedule, OpenCommitMatrixParams,
    OpeningClaimsLayout, PolynomialGroupLayout, SetupMatrixCapacity, SisModulusProfileId,
};
use std::marker::PhantomData;

/// Config adapter that routes ordinary commit selection through the exact
/// precommit layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrecommittedCommitmentConfig<Cfg>(PhantomData<Cfg>);

impl<Cfg: CommitmentConfig> CommitmentConfig for PrecommittedCommitmentConfig<Cfg> {
    type Field = Cfg::Field;
    type ExtField = Cfg::ExtField;

    const D: usize = Cfg::D;
    const RING_DIMENSION_CANDIDATES: &'static [CommitmentRingDims] = Cfg::RING_DIMENSION_CANDIDATES;
    const SELECTIVE_L2_FOLD_CAPS: &'static [akita_schedules::SelectiveL2FoldCap] =
        Cfg::SELECTIVE_L2_FOLD_CAPS;

    fn decomposition() -> DecompositionParams {
        Cfg::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Cfg::ring_challenge_config(d)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Cfg::sis_modulus_profile()
    }

    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        Cfg::selection_policy()
    }

    fn setup_matrix_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixCapacity, AkitaError> {
        if max_num_batched_polys == 0 {
            return Err(AkitaError::InvalidSetup(
                "max_num_batched_polys must be at least 1".to_string(),
            ));
        }
        let mut max_field_elements = 1usize;
        for num_polys in 1..=max_num_batched_polys {
            let opening_batch = OpeningClaimsLayout::new(max_num_vars, num_polys)?;
            let params = Self::get_params_for_batched_commitment(&opening_batch)?;
            accumulate_matrix_field_elements_for_level(&params, &mut max_field_elements)?;
        }
        Ok(SetupMatrixCapacity {
            num_field_elements: max_field_elements,
        })
    }

    fn basis_range() -> (u32, u32) {
        Cfg::basis_range()
    }

    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
        Cfg::root_honest_fold_policy()
    }

    fn schedule_catalog() -> Option<akita_schedules::GeneratedScheduleTable> {
        Cfg::schedule_catalog()
    }

    fn supports_multi_group_final_commit() -> bool {
        false
    }

    fn get_params_for_prove(
        _opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        Err(AkitaError::InvalidSetup(
            "PrecommittedCommitmentConfig is only for precommit layouts; proving must use the regular config"
                .to_string(),
        ))
    }

    fn get_params_for_batched_commitment(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<CommittedGroupParams, AkitaError> {
        opening_batch.check()?;
        if opening_batch.num_groups() != 1 {
            return Err(AkitaError::InvalidSetup(
                "PrecommittedCommitmentConfig only commits standalone precommitted groups"
                    .to_string(),
            ));
        }
        let key = opening_batch.root_final_group_layout()?;
        committed_group_params::<Cfg>(&key)
    }
}

/// Resolve standalone A/B commitment parameters for one group using the base
/// config's generated precommit profile.
///
/// This legacy adapter exists for callers still shaped around
/// [`CommittedGroupParams`]. The generated source of truth is the
/// descriptor-only [`committed_group_profile`]; final grouped-root expansion
/// derives opening metadata later from the selected schedule row.
pub fn committed_group_params<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<CommittedGroupParams, AkitaError> {
    let profile = committed_group_profile::<Cfg>(key)?;
    precommit_profile_as_commit_params::<Cfg>(profile)
}

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

fn precommit_profile_as_commit_params<Cfg: CommitmentConfig>(
    profile: CommittedGroupProfile,
) -> Result<CommittedGroupParams, AkitaError> {
    let policy = policy_of::<Cfg>();
    let d_open = profile.outer_commit_matrix.ring_dimension();
    let open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        1,
        1,
        profile.outer_commit_matrix.coeff_linf_bound(),
        d_open,
    );
    Ok(CommittedGroupParams {
        payload_mode: akita_types::CommitmentPayloadMode::Compressed,
        log_basis_inner: profile.log_basis_inner,
        log_basis_outer: profile.log_basis_outer,
        log_basis_open: profile.log_basis_outer,
        inner_commit_matrix: profile.inner_commit_matrix,
        outer_commit_matrix: profile.outer_commit_matrix,
        open_commit_matrix,
        num_live_ring_elements_per_claim: profile.num_live_ring_elements_per_claim,
        num_positions_per_block: profile.num_positions_per_block,
        num_live_blocks: profile.num_live_blocks,
        fold_challenge_config: Cfg::ring_challenge_config(
            profile.inner_commit_matrix.ring_dimension(),
        )?,
        num_digits_inner: profile.num_digits_inner,
        num_digits_outer: profile.num_digits_outer,
        num_digits_open: profile.num_digits_outer,
        num_digits_fold: 1,
        witness_chunk: policy.witness_chunk_for_level(0),
        precommitted_groups: Vec::new(),
        setup_prefix: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_types::CommittedGroupProfile;

    #[test]
    fn same_layout_can_resolve_config_specific_profiles() {
        let key = PolynomialGroupLayout::new(16, 1);
        let dense = committed_group_params::<fp128::D64Dense>(&key).expect("dense params");
        let one_hot = committed_group_params::<fp128::D64OneHot>(&key).expect("one-hot params");
        assert_ne!(
            CommittedGroupProfile::from_params(key, &dense),
            CommittedGroupProfile::from_params(key, &one_hot),
            "commitment config must affect standalone commitment parameters"
        );
    }

    #[test]
    fn dense_precommit_profile_uses_dense_config() {
        let key = PolynomialGroupLayout::new(15, 2);
        let params = committed_group_params::<fp128::D64Dense>(&key).expect("dense params");
        assert_eq!(params.log_basis_inner, 3);
        assert_eq!(params.log_basis_outer, 3);
        assert_eq!(params.num_digits_inner, 43);
    }
}
