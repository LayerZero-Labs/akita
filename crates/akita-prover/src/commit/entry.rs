//! Config-driven commit entry points: param resolution + root tensor-projection
//! decision, then the shared `commit_with_validated_params` kernel.

use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, RandomSampling};
use akita_types::{
    root_tensor_projection_enabled, schedule_root_fold_step, AkitaCommitmentHint,
    AkitaExpandedSetup, FpExtEncoding, OpeningBatch, RingCommitment,
};

use akita_config::CommitmentConfig;

use crate::commit::ajtai::backend::CommitBackend;
use crate::commit::pipeline::{
    commit_with_validated_params, prepare_batched_commit_inputs, prepare_commit_inputs,
    validate_batched_onehot_chunk_size_for_params, validate_commit_level_params,
    validate_onehot_chunk_size_for_params,
};
use crate::{AkitaPolyOps, RootTensorProjectionPoly};

/// Decide whether a root commitment must be tensor-projected before commit.
///
/// # Errors
///
/// Propagates [`CommitmentConfig::get_params_for_prove`].
fn should_transform_root_commitment<Cfg, const D: usize>(
    opening_batch: &OpeningBatch,
) -> Result<bool, AkitaError>
where
    Cfg: CommitmentConfig,
{
    if !root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField, Cfg::ExtField, D>(
        opening_batch.num_vars(),
    ) {
        return Ok(false);
    }
    let schedule = Cfg::get_params_for_prove(opening_batch)?;
    Ok(schedule_root_fold_step(&schedule).is_some())
}

/// Commit a group of polynomials under config `Cfg`.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or commitment
/// execution fails.
#[allow(clippy::type_complexity)]
pub fn commit<Cfg, const D: usize, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
) -> Result<
    (
        RingCommitment<Cfg::Field, D>,
        AkitaCommitmentHint<Cfg::Field, D>,
    ),
    AkitaError,
>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide,
    <Cfg::Field as HasWide>::Wide:
        akita_field::AdditiveGroup + From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: AkitaPolyOps<Cfg::Field, D> + crate::commit::AjtaiOpeningView<Cfg::Field, D>,
    B: CommitBackend<Cfg::Field>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    let opening_batch = prepare_commit_inputs::<Cfg::Field, D, P>(polys, expanded)?;
    let params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
    validate_onehot_chunk_size_for_params::<Cfg::Field, D, P>(polys, &params)?;
    if should_transform_root_commitment::<Cfg, D>(&opening_batch)? {
        let transformed = polys
            .iter()
            .map(|poly| poly.tensor_packed_extension_root_poly::<Cfg::ExtField>())
            .collect::<Result<Vec<RootTensorProjectionPoly<Cfg::Field, D>>, _>>()?;
        validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
        return commit_with_validated_params::<
            Cfg::Field,
            D,
            RootTensorProjectionPoly<Cfg::Field, D>,
            B,
        >(&transformed, backend, prepared, &params);
    }
    validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
    commit_with_validated_params::<Cfg::Field, D, P, B>(polys, backend, prepared, &params)
}

/// Commit one polynomial bundle under config `Cfg`.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or commitment
/// execution fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit<Cfg, const D: usize, P, B>(
    polys_per_commitment_group: &[&[P]],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
) -> Result<
    Vec<(
        RingCommitment<Cfg::Field, D>,
        AkitaCommitmentHint<Cfg::Field, D>,
    )>,
    AkitaError,
>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide,
    <Cfg::Field as HasWide>::Wide:
        akita_field::AdditiveGroup + From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: AkitaPolyOps<Cfg::Field, D> + crate::commit::AjtaiOpeningView<Cfg::Field, D>,
    B: CommitBackend<Cfg::Field>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    let opening_batch =
        prepare_batched_commit_inputs::<Cfg::Field, D, P>(polys_per_commitment_group, expanded)?;
    let params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
    for group in polys_per_commitment_group {
        validate_batched_onehot_chunk_size_for_params::<Cfg::Field, D, P>(group, &params)?;
    }
    if should_transform_root_commitment::<Cfg, D>(&opening_batch)? {
        let transformed: Vec<Vec<RootTensorProjectionPoly<Cfg::Field, D>>> =
            polys_per_commitment_group
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .map(|poly| poly.tensor_packed_extension_root_poly::<Cfg::ExtField>())
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<_, _>>()?;
        validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
        return transformed
            .iter()
            .map(|group| {
                commit_with_validated_params::<
                    Cfg::Field,
                    D,
                    RootTensorProjectionPoly<Cfg::Field, D>,
                    B,
                >(group, backend, prepared, &params)
            })
            .collect();
    }
    validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
    polys_per_commitment_group
        .iter()
        .map(|group| {
            commit_with_validated_params::<Cfg::Field, D, P, B>(group, backend, prepared, &params)
        })
        .collect()
}
