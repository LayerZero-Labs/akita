//! Multi-group batching config adapter.
//!
//! This adapter is for workflows that precommit groups independently and later
//! build a grouped root schedule. Conservative precommit sizing is included in
//! setup capacity, while final grouped-root schedule lookup is resolved as this
//! config's canonical grouped runtime schedule.

use crate::conservative_commitment::conservative_commit_params;
use crate::{policy_of, CommitmentConfig};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleInputs, DecompositionParams, GroupBatchAkitaScheduleLookupKey, LevelParams,
    Schedule, SetupMatrixEnvelope, SisModulusFamily, Step,
};
use std::marker::PhantomData;

/// Config adapter for multi-group batching.
#[derive(Clone, Copy, Debug, Default)]
pub struct MultiGroupBatchConfig<Cfg>(PhantomData<Cfg>);

impl<Cfg: CommitmentConfig> MultiGroupBatchConfig<Cfg> {
    /// Resolve the canonical grouped runtime schedule for a final grouped root.
    ///
    /// Unlike scalar [`CommitmentConfig::runtime_schedule`], this consumes the
    /// full group-batch key so the grouped root cannot alias a scalar table key.
    pub fn runtime_schedule(
        key: &GroupBatchAkitaScheduleLookupKey,
    ) -> Result<Schedule, AkitaError> {
        akita_planner::resolve_group_batch_schedule(
            key,
            &policy_of::<Cfg>(),
            Cfg::ring_challenge_config,
            Cfg::fold_challenge_shape_at_level,
            Cfg::schedule_catalog(),
        )
    }

    /// Commit params for the final/main group in a grouped root plan.
    ///
    /// These are derived from the grouped runtime schedule and are intentionally
    /// not the conservative precommit params.
    pub fn get_params_for_batched_commitment(
        key: &GroupBatchAkitaScheduleLookupKey,
    ) -> Result<LevelParams, AkitaError> {
        let schedule = Self::runtime_schedule(key)?;
        Ok(root_commit_params(&schedule, "grouped runtime schedule")?.clone())
    }
}

impl<Cfg: CommitmentConfig> CommitmentConfig for MultiGroupBatchConfig<Cfg> {
    type Field = Cfg::Field;
    type ExtField = Cfg::ExtField;

    const D: usize = Cfg::D;
    const TIERED_COMMITMENT: bool = Cfg::TIERED_COMMITMENT;

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
        // Start from the wrapped config's ordinary setup envelope, then add the
        // extra conservative B-rank capacity needed for precommitted groups.
        let mut envelope = Cfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys)?;
        inflate_for_conservative_precommit::<Cfg>(
            max_num_vars,
            max_num_batched_polys,
            &mut envelope.max_setup_len,
        )?;
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
}

fn inflate_for_conservative_precommit<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    for num_vars in 1..=max_num_vars {
        for num_polys in setup_envelope_poly_counts(max_num_batched_polys) {
            let key = akita_types::AkitaScheduleLookupKey::new(num_vars, num_polys);
            if let Ok(params) = conservative_commit_params::<Cfg>(&key) {
                accumulate_matrix_envelope_for_level(&params, max_setup_len)?;
            }
        }
    }
    Ok(())
}

fn setup_envelope_poly_counts(max_num_batched_polys: usize) -> [usize; 2] {
    if max_num_batched_polys <= 1 {
        [1, 1]
    } else {
        [1, max_num_batched_polys]
    }
}

fn accumulate_matrix_envelope_for_level(
    lp: &LevelParams,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let a_len = lp
        .a_key
        .row_len()
        .checked_mul(lp.inner_width())
        .ok_or_else(|| AkitaError::InvalidSetup("A setup envelope overflow".to_string()))?;
    let b_len = lp
        .b_key
        .row_len()
        .checked_mul(lp.outer_width())
        .ok_or_else(|| AkitaError::InvalidSetup("B setup envelope overflow".to_string()))?;
    let d_len = lp
        .d_key
        .row_len()
        .checked_mul(lp.d_matrix_width())
        .ok_or_else(|| AkitaError::InvalidSetup("D setup envelope overflow".to_string()))?;
    let f_len = match lp.f_key.as_ref() {
        Some(fk) => fk
            .row_len()
            .checked_mul(fk.col_len())
            .ok_or_else(|| AkitaError::InvalidSetup("F setup envelope overflow".to_string()))?,
        None => 0,
    };
    *max_setup_len = (*max_setup_len).max(a_len).max(b_len).max(d_len).max(f_len);
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
