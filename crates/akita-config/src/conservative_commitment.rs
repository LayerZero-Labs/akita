//! Conservative one-hot commitment config adapter.
//!
//! This adapter is for staggered workflows that need ordinary commit calls to
//! use a B rank conservative for a later multi-group root whose final basis is not
//! known at precommit time.

use crate::matrix_envelope::accumulate_matrix_envelope_for_level;
use crate::proof_optimized::setup_envelope_poly_counts;
use crate::{policy_of, CommitmentConfig};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    min_secure_rank, rounded_up_collision_inf_norm, SisTableKey, DEFAULT_SIS_SECURITY_BITS,
};
use akita_types::{
    AjtaiKeyParams, AkitaScheduleInputs, DecompositionParams, LevelParams, OpeningClaimsLayout,
    PolynomialGroupLayout, Schedule, SetupMatrixEnvelope, SisModulusFamily, Step,
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

    fn sis_modulus_family() -> SisModulusFamily {
        Cfg::sis_modulus_family()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Cfg::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys)
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

    fn get_params_for_prove(opening_batch: &OpeningClaimsLayout) -> Result<Schedule, AkitaError> {
        let key = opening_batch.root_final_group_layout()?;
        conservative_commit_schedule::<Cfg>(&key)
    }

    fn get_params_for_batched_commitment(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<LevelParams, AkitaError> {
        let schedule = Self::get_params_for_prove(opening_batch)?;
        Ok(root_commit_params(&schedule, "conservative commit schedule")?.clone())
    }
}

pub(crate) fn conservative_commit_params<Cfg: CommitmentConfig>(
    key: &PolynomialGroupLayout,
) -> Result<LevelParams, AkitaError> {
    let schedule = conservative_commit_schedule::<Cfg>(key)?;
    Ok(root_commit_params(&schedule, "conservative commit schedule")?.clone())
}

pub(crate) fn inflate_setup_envelope_for_conservative_commitments<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    envelope: &mut SetupMatrixEnvelope,
) -> Result<(), AkitaError> {
    let poly_counts = setup_envelope_poly_counts(max_num_batched_polys);
    for num_vars in 1..=max_num_vars {
        for &num_polys in &poly_counts {
            let key = PolynomialGroupLayout::new(num_vars, num_polys);
            if let Ok(params) = conservative_commit_params::<Cfg>(&key) {
                accumulate_matrix_envelope_for_level(&params, &mut envelope.max_setup_len)?;
            }
        }
    }
    Ok(())
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
    let mut schedule = akita_planner::find_schedule::<Cfg::Field>(
        *key,
        &policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?;
    let params = root_commit_params_mut(&mut schedule, "conservative commit schedule")?;
    widen_conservative_commit_params::<Cfg>(params)?;
    Ok(schedule)
}

fn widen_conservative_commit_params<Cfg: CommitmentConfig>(
    params: &mut LevelParams,
) -> Result<(), AkitaError> {
    let (min_basis, max_basis) = Cfg::basis_range();
    if params.log_basis != min_basis {
        return Err(AkitaError::InvalidSetup(
            "conservative commit planner did not use the minimum configured log_basis".to_string(),
        ));
    }

    let conservative_norm = rounded_up_collision_inf_norm(
        DEFAULT_SIS_SECURITY_BITS,
        Cfg::sis_modulus_family(),
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
            min_security_bits: DEFAULT_SIS_SECURITY_BITS,
            family: Cfg::sis_modulus_family(),
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
        DEFAULT_SIS_SECURITY_BITS,
        Cfg::sis_modulus_family(),
        conservative_n_b,
        params.b_key.col_len(),
        conservative_norm,
        Cfg::D,
    )?;
    Ok(())
}

fn root_commit_params<'a>(
    schedule: &'a Schedule,
    context: &str,
) -> Result<&'a LevelParams, AkitaError> {
    match schedule.steps.first() {
        Some(Step::Fold(root_step)) => Ok(&root_step.params),
        Some(Step::Direct(direct)) => direct.params.as_ref().ok_or_else(|| {
            AkitaError::InvalidSetup(format!("root-direct {context} is missing commit params"))
        }),
        None => Err(AkitaError::InvalidSetup(format!("{context} has no steps"))),
    }
}

fn root_commit_params_mut<'a>(
    schedule: &'a mut Schedule,
    context: &str,
) -> Result<&'a mut LevelParams, AkitaError> {
    match schedule.steps.first_mut() {
        Some(Step::Fold(root_step)) => Ok(&mut root_step.params),
        Some(Step::Direct(direct)) => direct.params.as_mut().ok_or_else(|| {
            AkitaError::InvalidSetup(format!("root-direct {context} is missing commit params"))
        }),
        None => Err(AkitaError::InvalidSetup(format!("{context} has no steps"))),
    }
}
