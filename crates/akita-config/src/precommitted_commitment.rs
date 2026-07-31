//! Exact precommitment config adapter.
//!
//! This adapter is for staggered workflows that need ordinary commit calls to
//! freeze the A/source and B/outer commitment layout before the final multi-group
//! root is known. The root basis is deterministic from the base config's runtime
//! catalog policy, so precommitments use the exact root layout rather than a
//! worst-case envelope over every supported basis.

use crate::{policy_of, CommitmentConfig};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    accumulate_matrix_field_elements_for_level, AkitaScheduleInputs, AkitaScheduleLookupKey,
    CommitmentRingDims, CommittedGroupParams, CommittedGroupProfile, DecompositionParams,
    FoldSchedule, OpeningClaimsLayout, PolynomialGroupLayout, SetupMatrixCapacity,
    SisModulusProfileId,
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

/// Resolve the exact standalone A/B commitment parameters for one group.
///
/// A generated grouped row may carry the frozen precommit descriptor even when
/// the catalog intentionally omits a scalar proof row for that source. This
/// function extracts those exact A/B facts; otherwise it resolves the exact
/// scalar row. It never runs the planner at runtime.
pub fn committed_group_params<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<CommittedGroupParams, AkitaError> {
    key.validate()?;
    if let Some(catalog) = Cfg::schedule_catalog() {
        let policy = policy_of::<Cfg>();
        akita_schedules::validate_catalog_identity(
            &catalog,
            &policy,
            Cfg::ring_challenge_config,
            Cfg::fold_challenge_shape_at_level,
        )?;

        let mut resolved = None;
        for entry in catalog.entries {
            for (group_idx, generated_group) in entry.root.precommitted_groups.iter().enumerate() {
                if generated_group.descriptor.group != *key {
                    continue;
                }
                let runtime_key = entry.to_runtime_lookup_key();
                let schedule = akita_schedules::schedule_from_entry(
                    entry,
                    &runtime_key,
                    &policy,
                    Cfg::ring_challenge_config,
                    Cfg::fold_challenge_shape_at_level,
                )?;
                let precommitted = schedule
                    .root
                    .params
                    .precommitted_groups
                    .get(group_idx)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "generated precommit row did not expand the expected group".to_string(),
                        )
                    })?;
                let group = &precommitted.commitment;
                let mut params = schedule.root.params.final_group.commitment.clone();
                params.log_basis_inner = group.layout.log_basis_inner;
                params.log_basis_outer = group.layout.log_basis_outer;
                params.log_basis_open = group.log_basis_open;
                params.inner_commit_matrix = group.layout.inner_commit_matrix;
                params.outer_commit_matrix = group.layout.outer_commit_matrix;
                params.num_live_ring_elements_per_claim =
                    group.layout.num_live_ring_elements_per_claim;
                params.num_positions_per_block = group.layout.num_positions_per_block;
                params.num_live_blocks = group.layout.num_live_blocks;
                params.num_digits_inner = group.layout.num_digits_inner;
                params.num_digits_outer = group.layout.num_digits_outer;
                params.num_digits_open = group.num_digits_open;
                params.num_digits_fold = group.num_digits_fold;
                params.precommitted_groups.clear();
                record_unique_precommit_profile(key, &mut resolved, params)?;
            }
        }
        if let Some(params) = resolved {
            return Ok(params);
        }
    }

    Ok(Cfg::runtime_schedule(AkitaScheduleLookupKey::single(*key))?
        .root
        .params
        .final_group
        .commitment)
}

fn record_unique_precommit_profile(
    key: &PolynomialGroupLayout,
    resolved: &mut Option<CommittedGroupParams>,
    candidate: CommittedGroupParams,
) -> Result<(), AkitaError> {
    if let Some(existing) = resolved {
        let existing_profile = CommittedGroupProfile::from_params(*key, existing);
        let candidate_profile = CommittedGroupProfile::from_params(*key, &candidate);
        if existing_profile != candidate_profile {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule catalog assigns multiple commitment profiles to standalone layout {key:?}"
            )));
        }
    } else {
        *resolved = Some(candidate);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;

    #[test]
    fn duplicate_layout_with_distinct_profiles_is_rejected() {
        let key = PolynomialGroupLayout::singleton(14);
        let dense = fp128::D64Dense::runtime_schedule(AkitaScheduleLookupKey::single(key))
            .expect("dense row")
            .root
            .params
            .final_group
            .commitment;
        let one_hot = fp128::D64OneHot::runtime_schedule(AkitaScheduleLookupKey::single(key))
            .expect("one-hot row")
            .root
            .params
            .final_group
            .commitment;
        assert_ne!(
            CommittedGroupProfile::from_params(key, &dense),
            CommittedGroupProfile::from_params(key, &one_hot),
            "test rows must carry distinct valid commitment profiles"
        );

        let mut resolved = None;
        record_unique_precommit_profile(&key, &mut resolved, dense).expect("first row");
        let error = record_unique_precommit_profile(&key, &mut resolved, one_hot)
            .expect_err("ambiguous layout must reject");
        assert!(error.to_string().contains("multiple commitment profiles"));
    }
}
