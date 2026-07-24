//! Exact one-hot precommitment config adapter.
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
    AkitaScheduleInputs, AkitaScheduleLookupKey, CommittedGroupParams, DecompositionParams,
    FoldSchedule, OpeningClaimsLayout, PolynomialGroupLayout, PrecommittedGroupDescriptor,
    SetupMatrixEnvelope, SisModulusProfileId,
};
use std::marker::PhantomData;

/// Config adapter that routes ordinary commit selection through the exact
/// one-hot precommit layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrecommittedCommitmentConfig<Cfg>(PhantomData<Cfg>);

impl<Cfg: CommitmentConfig> CommitmentConfig for PrecommittedCommitmentConfig<Cfg> {
    type Field = Cfg::Field;
    type ExtField = Cfg::ExtField;

    const D: usize = Cfg::D;

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
        let mut envelope = SetupMatrixEnvelope::minimum();
        for num_polys in 1..=max_num_batched_polys {
            let opening_batch = OpeningClaimsLayout::new(max_num_vars, num_polys)?;
            let params = Self::get_params_for_batched_commitment(&opening_batch)?;
            akita_types::accumulate_matrix_envelope_for_level(
                &params,
                &mut envelope.max_setup_len,
            )?;
        }
        Ok(envelope)
    }

    fn basis_range() -> (u32, u32) {
        Cfg::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Cfg::onehot_chunk_size()
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
        precommitted_commit_params::<Cfg>(&key)
    }
}

pub(crate) fn precommitted_group_params<Cfg: CommitmentConfig>(
    group: PolynomialGroupLayout,
) -> Result<PrecommittedGroupDescriptor, AkitaError> {
    group.validate()?;
    let singleton = OpeningClaimsLayout::new(group.num_vars(), group.num_polynomials())?;
    let params =
        <PrecommittedCommitmentConfig<Cfg> as CommitmentConfig>::get_params_for_batched_commitment(
            &singleton,
        )?;
    Ok(PrecommittedGroupDescriptor::from_params(group, &params))
}

pub(crate) fn precommitted_commit_params<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<CommittedGroupParams, AkitaError> {
    if let Some(params) = catalog_precommitted_commit_params::<Cfg>(key)? {
        return Ok(params);
    }
    let schedule = precommitted_commit_schedule::<Cfg>(key)?;
    Ok(schedule.root.params.final_group.commitment.clone())
}

fn catalog_precommitted_commit_params<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<Option<CommittedGroupParams>, AkitaError> {
    let Some(catalog) = Cfg::schedule_catalog() else {
        return Ok(None);
    };
    let policy = policy_of::<Cfg>();
    akita_schedules::validate_catalog_identity(
        &catalog,
        &policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?;

    for entry in catalog.entries {
        let Some((group_idx, _)) = entry
            .root
            .precommitted_groups
            .iter()
            .enumerate()
            .find(|(_, group)| group.descriptor.group == *key)
        else {
            continue;
        };
        let runtime_key = AkitaScheduleLookupKey {
            final_group: entry.root.final_group.layout,
            precommitteds: entry
                .root
                .precommitted_groups
                .iter()
                .map(|group| group.descriptor)
                .collect(),
        };
        let schedule = akita_schedules::schedule_from_entry(
            entry,
            &runtime_key,
            &policy,
            Cfg::ring_challenge_config,
            Cfg::fold_challenge_shape_at_level,
        )?;
        let Some(precommitted) = schedule.root.params.precommitted_groups.get(group_idx) else {
            return Err(AkitaError::InvalidSetup(
                "generated precommit row did not expand the expected group".to_string(),
            ));
        };
        if precommitted.descriptor.group != *key {
            return Err(AkitaError::InvalidSetup(
                "generated precommit row expanded a different group".to_string(),
            ));
        }

        let mut params = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(*key))?
            .root
            .params
            .final_group
            .commitment;
        let group = &precommitted.commitment;
        params.log_basis_inner = group.layout.log_basis_inner;
        params.log_basis_outer = group.layout.log_basis_outer;
        params.log_basis_open = group.log_basis_open;
        params.inner_commit_matrix = group.inner_commit_matrix.clone();
        params.outer_commit_matrix = group.outer_commit_matrix.clone();
        params.num_live_ring_elements_per_claim = group.layout.num_live_ring_elements_per_claim;
        params.num_positions_per_block = group.layout.num_positions_per_block;
        params.num_live_blocks = group.layout.num_live_blocks;
        params.num_digits_inner = group.num_digits_inner;
        params.num_digits_outer = group.num_digits_outer;
        params.num_digits_open = group.num_digits_open;
        params.num_digits_fold_one = group.num_digits_fold_one;
        return Ok(Some(params));
    }

    Ok(None)
}

pub(crate) fn precommitted_commit_schedule<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<FoldSchedule, AkitaError> {
    if Cfg::decomposition().log_commit_bound != 1 {
        return Err(AkitaError::InvalidSetup(
            "precommitments require a one-hot config".to_string(),
        ));
    }
    key.validate()?;

    // Runtime config must remain planner-free. The generated catalog identity
    // already fixes the root basis to the configured minimum, so resolving the
    // singleton runtime key yields the exact frozen root params without the
    // former conservative widening pass.
    Cfg::runtime_schedule(AkitaScheduleLookupKey::single(*key))
}
