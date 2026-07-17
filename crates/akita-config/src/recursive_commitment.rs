//! Recursive setup-offloading config adapter.

use crate::CommitmentConfig;
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, ChunkedWitnessCfg, DecompositionParams,
    OpeningClaimsLayout, Schedule, SetupMatrixEnvelope, SisModulusProfileId, SETUP_OFFLOAD_D_SETUP,
};
#[cfg(any(
    feature = "schedules-fp128-d64-onehot-recursive",
    feature = "schedules-fp128-d64-onehot-recursive-multi-chunk-w8r2"
))]
use std::any::TypeId;
use std::marker::PhantomData;

/// Config adapter that enables recursion-aware setup offloading schedules.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecursiveCommitmentConfig<Cfg>(PhantomData<Cfg>);

impl<Cfg: CommitmentConfig> CommitmentConfig for RecursiveCommitmentConfig<Cfg> {
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
        crate::proof_optimized::proof_optimized_max_setup_matrix_size::<Self>(
            max_num_vars,
            max_num_batched_polys,
        )
    }

    fn basis_range() -> (u32, u32) {
        Cfg::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Cfg::onehot_chunk_size()
    }

    fn chunked_witness_cfg() -> ChunkedWitnessCfg {
        Cfg::chunked_witness_cfg()
    }

    fn recursive_setup_planning() -> bool {
        true
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        #[cfg(feature = "schedules-fp128-d64-onehot-recursive")]
        {
            if TypeId::of::<Cfg>() == TypeId::of::<crate::proof_optimized::fp128::D64OneHot>() {
                return Some(akita_schedules::fp128_d64_onehot_recursive_table());
            }
        }
        #[cfg(feature = "schedules-fp128-d64-onehot-recursive-multi-chunk-w8r2")]
        {
            if TypeId::of::<Cfg>()
                == TypeId::of::<crate::proof_optimized::fp128::D64OneHotMultiChunk>()
            {
                return Some(akita_schedules::fp128_d64_onehot_recursive_multi_chunk_w8r2_table());
            }
        }
        None
    }

    fn runtime_schedule(
        key: akita_types::AkitaScheduleLookupKey,
    ) -> Result<akita_types::Schedule, AkitaError> {
        if Cfg::D != SETUP_OFFLOAD_D_SETUP {
            return Err(AkitaError::InvalidSetup(
                "recursive setup planning requires D64".to_string(),
            ));
        }
        if key.precommitteds.is_empty() {
            return Cfg::runtime_schedule(key);
        }
        akita_planner::resolve_group_batch_schedule(
            &key,
            &crate::policy_of::<Self>(),
            Self::ring_challenge_config,
            Self::fold_challenge_shape_at_level,
            Self::schedule_catalog(),
        )
    }

    fn get_params_for_prove(layout: &OpeningClaimsLayout) -> Result<Schedule, AkitaError> {
        Self::runtime_schedule(recursive_schedule_key::<Cfg>(layout)?)
    }
}

fn recursive_schedule_key<Cfg: CommitmentConfig>(
    layout: &OpeningClaimsLayout,
) -> Result<AkitaScheduleLookupKey, AkitaError> {
    layout.check()?;
    let final_group = layout.root_final_group_layout()?;
    if layout.num_groups() == 1 {
        return Ok(AkitaScheduleLookupKey::single(final_group));
    }
    let precommitteds = layout
        .root_precommitted_group_layouts()?
        .iter()
        .copied()
        .map(crate::conservative_commitment::conservative_precommitted_group_params::<Cfg>)
        .collect::<Result<Vec<_>, _>>()?;
    let key = AkitaScheduleLookupKey {
        final_group,
        precommitteds,
    };
    key.validate()?;
    Ok(key)
}
