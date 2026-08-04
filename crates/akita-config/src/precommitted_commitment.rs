//! Exact precommitment config adapter.
//!
//! This adapter is for staggered workflows that need ordinary commit calls to
//! freeze the A/source and B/outer commitment layout before the final multi-group
//! root is known. The root basis is deterministic from the base config's runtime
//! catalog policy, so precommitments use the exact root layout rather than a
//! worst-case envelope over every supported basis.

use crate::{honest_fold_policy_of, policy_of, CommitmentConfig};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_schedules::planner_support::{
    planned_next_witness_len, scalar_root_fold_level_params_candidate,
};
use akita_types::{
    accumulate_matrix_field_elements_for_level, AkitaScheduleInputs, CommitmentRingDims,
    CommittedGroupParams, DecompositionParams, FoldSchedule, OpeningClaimsLayout,
    PolynomialGroupLayout, SetupMatrixEnvelope, SisModulusProfileId,
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

    fn decomposition() -> DecompositionParams {
        Cfg::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Cfg::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Cfg::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Cfg::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Cfg::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
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
        Ok(SetupMatrixEnvelope {
            max_setup_len: max_field_elements.div_ceil(Cfg::D),
        })
    }

    fn basis_range() -> (u32, u32) {
        Cfg::basis_range()
    }

    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
        Cfg::root_honest_fold_policy()
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
/// config's native root source policy.
pub fn committed_group_params<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<CommittedGroupParams, AkitaError> {
    key.validate()?;
    standalone_precommit_params::<Cfg>(key)
}

fn standalone_precommit_params<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<CommittedGroupParams, AkitaError> {
    let honest_fold_policy = honest_fold_policy_of::<Cfg>();
    let mut policy = policy_of::<Cfg>().direct_only();
    policy.basis_range = (policy.basis_range.0, policy.basis_range.0);

    let witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("precommit witness too large".into()))?;
    let fold_challenge_shape = Cfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: 0,
        input_witness_len: witness_len,
    });
    let field_bits = policy.decomposition.field_bits();
    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(0);
    let mut best: Option<(usize, CommittedGroupParams)> = None;

    for candidate_log_basis in min_log_basis..=max_log_basis {
        for dimensions in Cfg::RING_DIMENSION_CANDIDATES.iter().copied() {
            let Ok(ring_challenge_cfg) = Cfg::ring_challenge_config(dimensions.d_a()) else {
                continue;
            };
            let alpha = (dimensions.d_a() as u32).trailing_zeros() as usize;
            let reduced_vars = key.num_vars().saturating_sub(alpha);
            if reduced_vars == 0 {
                continue;
            }
            let min_block_index_bits = if reduced_vars >= 3 { 1 } else { 0 };
            let max_block_index_bits = (reduced_vars - 1).min(usize::BITS as usize - 1);
            for block_index_bits in (min_block_index_bits..=max_block_index_bits).rev() {
                let Some(candidate_params) = scalar_root_fold_level_params_candidate(
                    &policy,
                    &ring_challenge_cfg,
                    dimensions,
                    key.num_vars(),
                    key.num_polynomials(),
                    candidate_log_basis,
                    block_index_bits,
                    fold_challenge_shape,
                    honest_fold_policy,
                )?
                else {
                    continue;
                };
                let next_witness_len = planned_next_witness_len(
                    field_bits,
                    &candidate_params,
                    key.num_polynomials(),
                    policy.chunks_at_level(0),
                )?;
                match &best {
                    Some((best_len, _)) if *best_len <= next_witness_len => {}
                    _ => best = Some((next_witness_len, candidate_params)),
                }
            }
        }
    }

    best.map(|(_, params)| params).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "no standalone precommit parameters found for layout {key:?} under this commitment config"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_types::CommittedGroupProfile;

    #[test]
    fn same_layout_can_resolve_config_specific_profiles() {
        let key = PolynomialGroupLayout::new(15, 2);
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
