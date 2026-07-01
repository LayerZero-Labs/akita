//! Multi-group batching config adapter.
//!
//! This adapter is for workflows that precommit groups independently and later
//! build a grouped root schedule. Conservative precommit sizing is included in
//! setup capacity, while final grouped-root schedule lookup is resolved as this
//! config's canonical grouped runtime schedule.

use crate::conservative_commitment::inflate_setup_envelope_for_conservative_commitments;
use crate::{policy_of, CommitmentConfig};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, DecompositionParams, LevelParams, Schedule,
    SetupMatrixEnvelope, SisModulusFamily, Step,
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
    pub fn runtime_schedule(key: &AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
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
        key: &AkitaScheduleLookupKey,
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
        inflate_setup_envelope_for_conservative_commitments::<Cfg>(
            max_num_vars,
            max_num_batched_polys,
            &mut envelope,
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
