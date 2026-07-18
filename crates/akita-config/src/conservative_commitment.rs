//! Conservative one-hot commitment config adapter.
//!
//! This adapter is for staggered workflows that need ordinary commit calls to
//! use a B rank conservative for a later multi-group root whose final basis is not
//! known at precommit time.

use crate::{policy_of, CommitmentConfig};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    min_secure_rank, rounded_up_collision_inf_norm, SisSecurityPolicyId, SisTableKey,
};
use akita_types::{
    AjtaiKeyParams, AkitaScheduleInputs, AkitaScheduleLookupKey, DecompositionParams, LevelParams,
    OpeningClaimsLayout, PolynomialGroupLayout, PrecommittedGroupParams, Schedule,
    SetupMatrixEnvelope, SisModulusProfileId,
};
use std::marker::PhantomData;

/// Config adapter that routes ordinary commit/prove schedule selection through
/// the conservative one-hot precommit layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConservativeCommitmentConfig<Cfg>(PhantomData<Cfg>);

impl<Cfg: CommitmentConfig> CommitmentConfig for ConservativeCommitmentConfig<Cfg> {
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
        let mut envelope = SetupMatrixEnvelope { max_setup_len: 1 };
        for num_polys in 1..=max_num_batched_polys {
            let opening_batch = OpeningClaimsLayout::new(max_num_vars, num_polys)?;
            let params = Self::get_params_for_batched_commitment(&opening_batch)?;
            crate::matrix_envelope::accumulate_matrix_envelope_for_level(
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

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Cfg::schedule_catalog()
    }

    fn supports_multi_group_final_commit() -> bool {
        false
    }

    fn get_params_for_prove(_opening_batch: &OpeningClaimsLayout) -> Result<Schedule, AkitaError> {
        Err(AkitaError::InvalidSetup(
            "ConservativeCommitmentConfig is only for precommit layouts; proving must use the regular config"
                .to_string(),
        ))
    }

    fn get_params_for_batched_commitment(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<LevelParams, AkitaError> {
        opening_batch.check()?;
        if opening_batch.num_groups() != 1 {
            return Err(AkitaError::InvalidSetup(
                "ConservativeCommitmentConfig only commits standalone precommitted groups"
                    .to_string(),
            ));
        }
        let key = opening_batch.root_final_group_layout()?;
        conservative_commit_params::<Cfg>(&key)
    }
}

pub(crate) fn conservative_precommitted_group_params<Cfg: CommitmentConfig>(
    group: PolynomialGroupLayout,
) -> Result<PrecommittedGroupParams, AkitaError> {
    group.validate()?;
    let singleton = OpeningClaimsLayout::new(group.num_vars(), group.num_polynomials())?;
    let params =
        <ConservativeCommitmentConfig<Cfg> as CommitmentConfig>::get_params_for_batched_commitment(
            &singleton,
        )?;
    Ok(PrecommittedGroupParams::from_params(group, &params))
}

pub(crate) fn conservative_commit_params<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<LevelParams, AkitaError> {
    let schedule = conservative_commit_schedule::<Cfg>(key)?;
    Ok(schedule.root_fold()?.params.clone())
}

pub(crate) fn conservative_commit_schedule<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<Schedule, AkitaError> {
    if Cfg::decomposition().log_commit_bound != 1 {
        return Err(AkitaError::InvalidSetup(
            "conservative commitments require a one-hot config".to_string(),
        ));
    }
    key.validate()?;

    let (min_basis, _) = Cfg::basis_range();
    let mut policy = policy_of::<Cfg>();
    policy.basis_range = (min_basis, min_basis);
    policy.decomposition.log_basis = min_basis;
    let mut schedule = akita_planner::find_group_batch_schedule(
        &AkitaScheduleLookupKey::single(*key),
        &policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?;
    let params = &mut schedule.root_fold_mut()?.params;
    widen_conservative_commit_params::<Cfg>(params, policy.sis_security_policy)?;
    Ok(schedule)
}

fn widen_conservative_commit_params<Cfg: CommitmentConfig>(
    params: &mut LevelParams,
    sis_security_policy: SisSecurityPolicyId,
) -> Result<(), AkitaError> {
    let (min_basis, max_basis) = Cfg::basis_range();
    if params.log_basis != min_basis {
        return Err(AkitaError::InvalidSetup(
            "conservative commit planner did not use the minimum configured log_basis".to_string(),
        ));
    }

    let conservative_norm = rounded_up_collision_inf_norm(
        sis_security_policy,
        Cfg::sis_modulus_profile(),
        akita_types::SisMatrixRole::B,
        Cfg::D,
        max_basis,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup(
            "no conservative B-role norm for conservative commitment".to_string(),
        )
    })?;
    let conservative_n_b = min_secure_rank(
        SisTableKey {
            policy: sis_security_policy,
            table_digest: akita_types::sis::SisTableDigest::CURRENT,
            modulus_profile: Cfg::sis_modulus_profile(),
            role: akita_types::SisMatrixRole::B,
            ring_dimension: Cfg::D as u32,
            coeff_linf_bound: conservative_norm,
        },
        params.b_key.col_len() as u64,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup(
            "no conservative B-role rank for conservative commitment".to_string(),
        )
    })?;
    params.b_key = AjtaiKeyParams::try_new(
        sis_security_policy,
        akita_types::sis::SisTableDigest::CURRENT,
        Cfg::sis_modulus_profile(),
        akita_types::SisMatrixRole::B,
        conservative_n_b,
        params.b_key.col_len(),
        conservative_norm,
        Cfg::D,
    )?;
    Ok(())
}
