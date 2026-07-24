//! Exact one-hot precommitment config adapter.
//!
//! This adapter is for staggered workflows that need ordinary commit calls to
//! freeze the A/source and B/outer commitment layout before the final multi-group
//! root is known. The root basis is deterministic from the base config's existing
//! planner policy, so precommitments use the exact root layout rather than a
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
    let schedule = precommitted_commit_schedule::<Cfg>(key)?;
    Ok(schedule.root.params.final_group.commitment.clone())
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

    // Freeze the whole planning probe to the known root basis. Allowing a
    // hypothetical standalone suffix to choose larger bases feeds back into the
    // root geometry and can collapse a small precommit below the live-block
    // count required by a later multi-chunk root. Only the probe is
    // single-basis: the returned value is its root commitment params. Unlike the
    // former conservative adapter, those exact ranks and bounds are kept rather
    // than widened across alternative opening bases.
    let mut policy = policy_of::<Cfg>();
    let root_basis = policy.basis_range.0;
    policy.basis_range = (root_basis, root_basis);
    // A precommitted group is a pre-existing, independently formed commitment;
    // the distributed multi-chunk layout only refines the *fold* witness, not how
    // an earlier commitment was formed. Freeze precommits single-chunk so a
    // multi-chunk base config (e.g. the W8R2 preset used for recursive
    // setup-offloading) produces the same frozen params as its single-chunk
    // sibling. Otherwise the frozen precommit diverges from the shipped
    // (single-chunk-frozen) recursive catalog key and the multi-group planner
    // can fall back to an invalid grouped root-direct.
    policy.witness_chunk = akita_types::ChunkedWitnessCfg::default();
    let planned = akita_planner::find_group_batch_schedule(
        &AkitaScheduleLookupKey::single(*key),
        &policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?;
    Ok(planned.schedule)
}
