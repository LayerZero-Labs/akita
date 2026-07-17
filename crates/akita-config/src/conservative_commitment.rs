//! Conservative one-hot commitment config adapter.
//!
//! This adapter is for staggered workflows that need ordinary commit calls to
//! freeze the A/source and B/outer commitment layout before the final multi-group
//! root is known.

use crate::{policy_of, CommitmentConfig};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_t_ring_count, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm,
    AjtaiKeyParams, SisMatrixRole, SisTableKey,
};
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, DecompositionParams, LevelParams,
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
    // A precommitted group is a pre-existing, independently formed commitment;
    // the distributed multi-chunk layout only refines the *fold* witness, not how
    // an earlier commitment was formed. Freeze precommits single-chunk so a
    // multi-chunk base config (e.g. the W8R2 preset used for recursive
    // setup-offloading) produces the same frozen params as its single-chunk
    // sibling. Otherwise the frozen precommit diverges from the shipped
    // (single-chunk-frozen) recursive catalog key and the multi-group planner
    // can fall back to an invalid grouped root-direct.
    policy.witness_chunk = akita_types::ChunkedWitnessCfg::default();
    let mut schedule = akita_planner::find_group_batch_schedule(
        &AkitaScheduleLookupKey::single(*key),
        &policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?;
    let widened = widen_conservative_commit_params::<Cfg>(schedule.root_fold()?.params.clone())?;
    schedule.root_fold_mut()?.params = widened;
    Ok(schedule)
}

fn widen_conservative_commit_params<Cfg: CommitmentConfig>(
    mut params: LevelParams,
) -> Result<LevelParams, AkitaError> {
    let policy = policy_of::<Cfg>();
    let (min_basis, _) = Cfg::basis_range();
    if params.log_basis_open != min_basis {
        return Err(AkitaError::InvalidSetup(
            "conservative commit planner did not use the minimum configured log_basis_open"
                .to_string(),
        ));
    }

    let witness_decomposition = DecompositionParams {
        log_basis: params.log_basis_inner,
        ..policy.decomposition
    };
    let inner_width = params.a_key.col_len();
    let mut conservative_a_rows = 0usize;
    let mut conservative_a_bound = 0u128;
    let mut conservative_b_bound = 0u128;

    for log_basis_open in policy.basis_range.0..=policy.basis_range.1 {
        let a_bound = rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            policy.ring_dimension,
            witness_decomposition,
            log_basis_open,
            &params.fold_challenge_config,
            params.fold_challenge_shape,
            true,
            policy.onehot_chunk_size,
            policy.ring_subfield_norm_bound,
            params.num_live_blocks,
            1,
            inner_width as u64,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("no conservative A-role norm".to_string()))?;
        let a_key = AjtaiKeyParams::try_new_with_min_rank(
            SisTableKey {
                policy: policy.sis_security_policy,
                table_digest: policy.sis_table_digest,
                modulus_profile: policy.sis_modulus_profile,
                role: SisMatrixRole::A,
                ring_dimension: policy.ring_dimension as u32,
                coeff_linf_bound: a_bound,
            },
            inner_width,
        )?;
        conservative_a_rows = conservative_a_rows.max(a_key.row_len());
        conservative_a_bound = conservative_a_bound.max(a_bound);

        let b_bound = rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            SisMatrixRole::B,
            policy.ring_dimension,
            log_basis_open,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("no conservative B-role norm".to_string()))?;
        conservative_b_bound = conservative_b_bound.max(b_bound);
    }

    params.a_key = AjtaiKeyParams::try_new(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        SisMatrixRole::A,
        conservative_a_rows,
        inner_width,
        conservative_a_bound,
        policy.ring_dimension,
    )?;
    let outer_width = decomposed_t_ring_count(
        conservative_a_rows,
        params.num_digits_outer,
        params.num_live_blocks,
        1,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("conservative B width overflow".to_string()))?;
    params.b_key = AjtaiKeyParams::try_new_with_min_rank(
        SisTableKey {
            policy: policy.sis_security_policy,
            table_digest: policy.sis_table_digest,
            modulus_profile: policy.sis_modulus_profile,
            role: SisMatrixRole::B,
            ring_dimension: policy.ring_dimension as u32,
            coeff_linf_bound: conservative_b_bound,
        },
        outer_width,
    )?;
    params.stamp_role_dims_from_keys();
    Ok(params)
}
