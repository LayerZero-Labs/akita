//! Prover-owned commitment kernels.

use crate::compute::compression::{execute_compression_chains, CompressionExecutionInput};
use crate::compute::{
    tensor_root_projection, CommitInnerPlan, OperationCtx, RootCommitKernel, RootCommitSource,
    RootPolyMeta, RuntimeCommitBackendFor, RuntimeRootCommitBackend, RuntimeRootCommitPoly,
    UniformProverStack,
};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::validation::validate_i8_setup_log_basis;
use crate::{CommitInnerWitness, RootTensorProjectionPoly};
use akita_config::{ensure_prover_schedule_fits_setup, CommitmentConfig};
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, HalvingField, RandomSampling,
};
use akita_types::{
    dispatch_for_field, root_tensor_projection_enabled, validate_role_dims,
    validate_role_dims_for_field, AkitaCommitmentHint, AkitaExpandedSetup, AkitaScheduleLookupKey,
    Commitment, CommittedGroup, CommittedGroupParams, CommittedGroupProfile, CompressionChainPlan,
    DigitBlocks, FpExtEncoding, OpeningClaimsLayout, OpeningScheduleSelection,
    PolynomialGroupLayout, RingVec,
};

/// Commitment output plus prover-side hint for one committed polynomial bundle.
///
/// D-free protocol storage: a flat [`Commitment`] plus the semantic A-native
/// inner rows needed when the commitment is opened.
pub type CommitmentWithHint<F> = (Commitment<F>, AkitaCommitmentHint<F>);

/// Frozen layout, commitment rows, and prover hint for one standalone group.
pub type CommittedGroupWithHint<F> = (CommittedGroup<F>, AkitaCommitmentHint<F>);

/// Final committed group, prover hint, and the exact generated row selected for
/// the complete ordered commitment batch.
pub type FinalCommittedGroupWithHint<F> = (
    CommittedGroup<F>,
    AkitaCommitmentHint<F>,
    OpeningScheduleSelection,
);

#[tracing::instrument(skip_all, name = "validate_commit_inner_shape")]
pub(crate) fn validate_commit_inner_shape<F, const D: usize>(
    inner: &CommitInnerWitness<F>,
    num_live_blocks: usize,
    n_a: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    inner.ensure_ring_dim::<D>()?;

    let expected_rows = num_live_blocks
        .checked_mul(n_a)
        .ok_or_else(|| AkitaError::InvalidSetup("inner commitment row count overflow".into()))?;
    let actual_rows = inner.inner_rows.count();
    if actual_rows != expected_rows {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {actual_rows} inner commitment rows, expected {expected_rows}"
        )));
    }
    for block_idx in 0..num_live_blocks {
        let block_rows = inner.block_rows::<D>(block_idx, n_a)?;
        if block_rows.len() != n_a {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {} A rows for inner commitment block {}, expected {}",
                block_rows.len(),
                block_idx,
                n_a
            )));
        }
    }

    Ok(())
}

pub(crate) fn validate_commit_level_params<F>(
    params: &CommittedGroupParams,
    setup: &AkitaExpandedSetup<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if params.num_live_blocks == 0 || params.num_positions_per_block == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero num_live_blocks and num_positions_per_block".to_string(),
        ));
    }
    if params.num_digits_inner == 0 || params.num_digits_outer == 0 || params.num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero digit depths".to_string(),
        ));
    }
    validate_i8_setup_log_basis(
        params.log_basis_inner,
        "for i8 witness commitment decomposition",
    )?;
    validate_i8_setup_log_basis(
        params.log_basis_outer,
        "for i8 outer commitment decomposition",
    )?;
    validate_i8_setup_log_basis(params.log_basis_open, "for i8 opening decomposition")?;
    let dims = params.role_dims();
    validate_role_dims(dims)?;
    validate_role_dims_for_field::<F>(dims)?;
    let expected_a_width = params
        .num_positions_per_block
        .checked_mul(params.num_digits_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("A commit width overflow".to_string()))?;
    if params.inner_commit_matrix.input_width() != expected_a_width {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params A width {} does not match num_positions_per_block * num_digits_inner = {expected_a_width}",
            params.inner_commit_matrix.input_width()
        )));
    }
    if params.outer_commit_matrix.input_width() == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params require nonzero B width, got B={}",
            params.outer_commit_matrix.input_width()
        )));
    }
    if params.open_commit_matrix.input_width() == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params require nonzero D width, got D={}",
            params.open_commit_matrix.input_width()
        )));
    }
    let a_required = params
        .inner_commit_matrix
        .output_rank()
        .checked_mul(params.inner_commit_matrix.input_width())
        .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))?;
    let a_available = setup.shared_matrix.num_field_elements() / dims.d_a();
    if a_required > a_available {
        return Err(AkitaError::InvalidSetup(format!(
            "A-role commit params require {a_required} setup ring elements at d={}, but setup has {a_available}",
            dims.d_a()
        )));
    }
    let b_required = params
        .outer_commit_matrix
        .output_rank()
        .checked_mul(params.outer_commit_matrix.input_width())
        .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".to_string()))?;
    let b_available = setup.shared_matrix.num_field_elements() / dims.d_b();
    if b_required > b_available {
        return Err(AkitaError::InvalidSetup(format!(
            "B-role commit params require {b_required} setup ring elements at d={}, but setup has {b_available}",
            dims.d_b()
        )));
    }
    // Commitment materialization uses only A and B. In particular, a
    // standalone group extracted from an approved multi-group row may retain
    // that row's shared D geometry, which is consumed only if the group later
    // participates in the selected opening schedule. Charging D here would
    // reject a setup that exactly fits the standalone commitment profile.
    Ok(())
}

pub(crate) fn validate_commit_outer_input_nonempty(active_len: usize) -> Result<(), AkitaError> {
    if active_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit B input must be nonempty".to_string(),
        ));
    }
    Ok(())
}

/// Validate a singleton commitment request against prover setup capacity.
///
/// # Errors
///
/// Returns an error if the request is empty, mixes polynomial dimensions, or
/// exceeds the prover setup capacity.
pub fn prepare_commit_inputs<F, P>(
    polys: &[P],
    setup: &AkitaExpandedSetup<F>,
) -> Result<OpeningClaimsLayout, AkitaError>
where
    F: FieldCore,
    P: RootPolyMeta<F>,
{
    if polys.is_empty() {
        return Err(AkitaError::InvalidInput(
            "commit requires at least one polynomial".to_string(),
        ));
    }
    let num_vars = polys[0].num_vars();
    if polys.iter().any(|p| p.num_vars() != num_vars) {
        return Err(AkitaError::InvalidInput(
            "all polynomials in a batched commit must have the same num_vars".to_string(),
        ));
    }
    if polys.len() > setup.seed.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
            "commit received {} polynomials but setup supports at most {}",
            polys.len(),
            setup.seed.max_num_batched_polys
        )));
    }
    if num_vars > setup.seed.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "commit received a polynomial with {} variables but setup supports at most {}",
            num_vars, setup.seed.max_num_vars
        )));
    }

    OpeningClaimsLayout::new(num_vars, polys.len())
}

#[cfg(test)]
fn checked_commit_b_input_len(total_polys: usize, per_poly: usize) -> Result<usize, AkitaError> {
    total_polys.checked_mul(per_poly).ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "commit B digit input length overflow for {total_polys} polynomials with {per_poly} digits each"
        ))
    })
}

/// A-role root tensor projection at `transform_ring_d` when the schedule calls for it.
fn tensor_project_roots<F, P, E, B>(
    transform_ring_d: usize,
    tensor_ctx: &OperationCtx<'_, F, B>,
    polys: &[P],
) -> Result<Vec<RootTensorProjectionPoly<F>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: FpExtEncoding<F>,
    P: RuntimeRootCommitPoly<F>,
    B: RuntimeRootCommitBackend<F, P, E>,
{
    let backend = tensor_ctx.backend();
    let prepared = tensor_ctx.prepared();
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        transform_ring_d,
        |D| {
            polys
                .iter()
                .map(|poly| tensor_root_projection::<F, P, E, B, D>(backend, Some(prepared), poly))
                .collect()
        }
    )
}

fn commit_with_validated_params<F, P, B>(
    polys: &[P],
    ctx: &OperationCtx<'_, F, B>,
    params: &CommittedGroupParams,
) -> Result<CommitmentWithHint<F>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootCommitSource<F, 32>
        + RootCommitSource<F, 64>
        + RootCommitSource<F, 128>
        + RootCommitSource<F, 256>
        + RootCommitSource<F, 512>
        + RootPolyMeta<F>,
    B: RuntimeCommitBackendFor<F, P>,
{
    let backend = ctx.backend();
    let prepared = ctx.prepared();
    // Per-role ring dimensions for this level: the inner commit digits are
    // A-role data, the outer `B·t̂` rows are B-role data. The mixed-row spec
    // feeds diverging dims here (uniform today).
    let dims = params.role_dims();
    let plan = CommitInnerPlan::from_level(params);
    let num_live_blocks = params.num_live_blocks;
    let n_a = params.inner_commit_matrix.output_rank();
    let num_digits_open = params.num_digits_outer;
    let log_basis = params.log_basis_outer;
    let n_b = params.outer_commit_matrix.output_rank();
    let (commitment, inner_rows, compression_witness, compression_quotients) = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        dims.d_a(),
        |D_A| {
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                F,
                dims.d_b(),
                |D_B| {
                    let prepared_polynomials = cfg_iter!(polys)
                        .map(|poly| -> Result<(RingVec<F>, DigitBlocks), AkitaError> {
                            let view = RootCommitSource::<F, D_A>::commit_view(poly)?;
                            let inner = RootCommitKernel::<_, F, D_A>::commit_inner(
                                backend, prepared, view, plan,
                            )?;
                            validate_commit_inner_shape::<F, D_A>(&inner, num_live_blocks, n_a)?;
                            let blocks = (0..num_live_blocks)
                                .map(|block| inner.block_rows::<D_A>(block, n_a))
                                .collect::<Result<Vec<_>, _>>()?;
                            let digits = decompose_commit_blocks_into::<F, D_A, D_B>(
                                &blocks,
                                num_digits_open,
                                log_basis,
                            )?;
                            Ok((inner.into_inner_rows(), digits))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let total_planes =
                        prepared_polynomials
                            .iter()
                            .try_fold(0usize, |total, (_, digits)| {
                                total.checked_add(digits.total_planes()).ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "commit B input plane count overflow".to_string(),
                                    )
                                })
                            })?;
                    validate_commit_outer_input_nonempty(total_planes)?;
                    let mut b_input_digits = Vec::with_capacity(total_planes);
                    for (_, digits) in &prepared_polynomials {
                        b_input_digits.extend_from_slice(digits.typed_planes::<D_B>()?);
                    }
                    let u = backend.digit_rows::<D_B>(prepared, n_b, &b_input_digits, log_basis)?;
                    if u.len() != n_b {
                        return Err(AkitaError::InvalidSetup(format!(
                            "backend returned {} B commitment rows, expected {n_b}",
                            u.len(),
                        )));
                    }
                    let source = RingVec::from_ring_elems(&u);
                    let plan = CompressionChainPlan::for_complete_source(
                        params.outer_commit_matrix.sis_table_key().modulus_profile,
                        source.coeff_len(),
                    )?;
                    let (mut outputs, _) = execute_compression_chains(
                        ctx,
                        vec![CompressionExecutionInput {
                            id: (),
                            plan,
                            coefficients: source.into_coeffs(),
                        }],
                    )?;
                    let output = outputs.pop().ok_or(AkitaError::InvalidProof)?;
                    let terminal_ring_dim = output
                        .witness
                        .plan()
                        .maps()
                        .last()
                        .ok_or(AkitaError::InvalidProof)?
                        .ring_dimension();
                    let payload = RingVec::from_coeffs_with_ring_dim(
                        output.terminal.coefficients().to_vec(),
                        terminal_ring_dim,
                    )?;
                    Ok::<_, AkitaError>((
                        Commitment::new(payload),
                        prepared_polynomials
                            .into_iter()
                            .map(|(rows, _)| rows)
                            .collect::<Vec<_>>(),
                        output.witness,
                        output.quotients,
                    ))
                }
            )
        }
    )?;
    let hint = AkitaCommitmentHint::new_with_outer_compression(
        dims.d_a(),
        inner_rows,
        &compression_witness,
        &compression_quotients,
    )?;
    Ok((commitment, hint))
}

/// Commit a group of polynomials using already-selected level parameters.
///
/// Config/schedule policy chooses `params`; this function owns only the
/// prover-side matrix work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if input validation, inner witness commitment, or hint
/// allocation fails.
pub fn commit_with_params<F, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<F>,
    ctx: &OperationCtx<'_, F, B>,
    params: &CommittedGroupParams,
) -> Result<CommitmentWithHint<F>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootCommitSource<F, 32>
        + RootCommitSource<F, 64>
        + RootCommitSource<F, 128>
        + RootCommitSource<F, 256>
        + RootCommitSource<F, 512>
        + RootPolyMeta<F>,
    B: RuntimeCommitBackendFor<F, P>,
{
    prepare_commit_inputs::<F, P>(polys, expanded)?;
    validate_commit_level_params::<F>(params, expanded)?;
    commit_with_validated_params::<F, P, B>(polys, ctx, params)
}

/// Decide whether a root commitment must be tensor-projected before commit.
///
/// Root tensor projection only applies when the field tower admits it and the
/// config-selected schedule starts with a fold. The ring dimension is the
/// prove schedule's root fold A-role dimension — the same schedule-derived
/// value `prepare_root` uses when it makes the matching prove-side decision.
///
/// # Errors
///
/// Propagates [`CommitmentConfig::get_params_for_prove`].
///
/// Returns `Some(ring_d)` — the dimension the projection operation must run
/// at — when the transform applies, `None` otherwise.
fn root_transform_ring_dim<Cfg>(
    opening_batch: &OpeningClaimsLayout,
) -> Result<Option<usize>, AkitaError>
where
    Cfg: CommitmentConfig,
{
    if Cfg::EXT_DEGREE == 1 {
        return Ok(None);
    }
    let schedule = Cfg::get_params_for_prove(opening_batch)?;
    let root_fold = schedule.root_fold();
    let ring_d = root_fold.params.final_group.commitment.role_dims().d_a();
    Ok(root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField>(
        ring_d,
        opening_batch.max_num_vars(),
    )
    .then_some(ring_d))
}

/// `ring_d` is the group-commit layout's schedule-derived ring dimension.
fn should_transform_group_commitment<Cfg>(key: &PolynomialGroupLayout, ring_d: usize) -> bool
where
    Cfg: CommitmentConfig,
{
    root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField>(ring_d, key.num_vars())
}

/// Commit a group of polynomials under config `Cfg`.
///
/// The prover crate owns input validation, the root tensor-projection
/// transform decision, config-driven layout selection, and commitment
/// execution.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or commitment
/// execution fails.
#[allow(clippy::type_complexity)]
pub fn commit<Cfg, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    stack: &UniformProverStack<'_, Cfg::Field, B>,
) -> Result<CommittedGroupWithHint<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: RuntimeRootCommitPoly<Cfg::Field>,
    B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
{
    let commit_ctx = stack.commit();
    let tensor_ctx = stack.tensor();
    let opening_batch = prepare_commit_inputs::<Cfg::Field, P>(polys, expanded)?;
    let params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
    let (commitment, hint) =
        if let Some(transform_ring_d) = root_transform_ring_dim::<Cfg>(&opening_batch)? {
            // A-role tensor-projection operation at the prove schedule's root fold
            // ring dimension.
            let transformed = tensor_project_roots::<Cfg::Field, P, Cfg::ExtField, B>(
                transform_ring_d,
                tensor_ctx,
                polys,
            )?;
            validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
            commit_with_validated_params::<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, B>(
                &transformed,
                commit_ctx,
                &params,
            )?
        } else {
            validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
            commit_with_validated_params::<Cfg::Field, P, B>(polys, commit_ctx, &params)?
        };
    let group = opening_batch.root_final_group_layout()?;
    let descriptor = CommittedGroupProfile::from_params(group, &params);
    Ok((CommittedGroup::new(descriptor, commitment), hint))
}

/// Validate a batched commitment request and derive its `OpeningClaimsLayout`.
///
/// The input slice is one commitment group. Its natural polynomial arity
/// selects that group's root layout.
///
/// # Errors
///
/// Returns an error if the bundle is empty, exceeds the prover setup capacity,
/// or has a variable count exceeding the prover setup capacity.
pub fn prepare_batched_commit_inputs<F, P>(
    polys: &[P],
    setup: &AkitaExpandedSetup<F>,
) -> Result<OpeningClaimsLayout, AkitaError>
where
    F: FieldCore,
    P: RootPolyMeta<F>,
{
    if polys.is_empty() {
        return Err(AkitaError::InvalidInput(
            "batched_commit commitment group must be nonempty".to_string(),
        ));
    }
    let padded_num_vars = polys
        .iter()
        .map(RootPolyMeta::num_vars)
        .max()
        .ok_or_else(|| {
            AkitaError::InvalidInput("batched_commit bundles must be nonempty".to_string())
        })?;
    if padded_num_vars > setup.seed.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received a polynomial with {} variables but setup supports at most {}",
            padded_num_vars, setup.seed.max_num_vars
        )));
    }

    if polys.len() > setup.seed.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received {} polynomials but setup supports at most {}",
            polys.len(),
            setup.seed.max_num_batched_polys
        )));
    }

    OpeningClaimsLayout::new(padded_num_vars, polys.len())
}

fn validate_group_commit_inputs<F, P>(
    polys: &[P],
    setup: &AkitaExpandedSetup<F>,
) -> Result<PolynomialGroupLayout, AkitaError>
where
    F: FieldCore,
    P: RootPolyMeta<F>,
{
    let opening_batch = prepare_commit_inputs::<F, P>(polys, setup)?;
    Ok(PolynomialGroupLayout::new(
        opening_batch.max_num_vars(),
        opening_batch.num_total_polynomials(),
    ))
}

/// Commit one standalone group with the exact fixed-root layout.
///
/// Grouped proving is still guarded until the opening phase lands; this API only
/// produces the precommit metadata and commitment object required by that later
/// finalization path.
///
/// # Errors
///
/// Returns an error if the group is unsupported by the setup or no exact
/// generated row supports it.
pub fn commit_group<Cfg, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    stack: &UniformProverStack<'_, Cfg::Field, B>,
) -> Result<CommittedGroupWithHint<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: RuntimeRootCommitPoly<Cfg::Field>,
    B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
{
    let commit_ctx = stack.commit();
    let tensor_ctx = stack.tensor();
    let key = validate_group_commit_inputs::<Cfg::Field, P>(polys, expanded)?;
    let params = akita_config::committed_group_params::<Cfg>(&key)?;
    validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
    let (commitment, hint) =
        if should_transform_group_commitment::<Cfg>(&key, params.role_dims().d_a()) {
            // A-role tensor-projection operation at the group layout's ring
            // dimension.
            let transform_d = params.role_dims().d_a();
            let transformed = tensor_project_roots::<Cfg::Field, P, Cfg::ExtField, B>(
                transform_d,
                tensor_ctx,
                polys,
            )?;
            commit_with_validated_params::<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, B>(
                &transformed,
                commit_ctx,
                &params,
            )?
        } else {
            commit_with_validated_params::<Cfg::Field, P, B>(polys, commit_ctx, &params)?
        };
    let descriptor = CommittedGroupProfile::from_params(key, &params);
    Ok((CommittedGroup::new(descriptor, commitment), hint))
}

fn final_group_key_from_polys<Cfg, P>(
    polys: &[P],
    setup: &AkitaExpandedSetup<Cfg::Field>,
    precommitteds: Vec<CommittedGroupProfile>,
) -> Result<AkitaScheduleLookupKey, AkitaError>
where
    Cfg: CommitmentConfig,
    P: RootPolyMeta<Cfg::Field>,
{
    let opening_batch = prepare_batched_commit_inputs::<Cfg::Field, P>(polys, setup)?;
    if precommitteds.is_empty() {
        return Err(AkitaError::InvalidInput(
            "commit_final_group requires at least one precommitted group".to_string(),
        ));
    }
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
        ),
        precommitteds,
    };
    key.validate(Cfg::decomposition().field_bits())?;
    Ok(key)
}

fn should_transform_final_group_commitment<Cfg>(
    key: &AkitaScheduleLookupKey,
    ring_d: usize,
) -> Result<bool, AkitaError>
where
    Cfg: CommitmentConfig,
{
    if !root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField>(
        ring_d,
        key.final_group.num_vars(),
    ) {
        return Ok(false);
    }
    Cfg::runtime_schedule(key.clone())?;
    Ok(true)
}

/// Commit the final polynomial bundle for a multi-group root commitment.
///
/// The final group shape is derived from `polys`; `precommitteds` supplies the
/// schedule keys for prior groups in transcript order. Each precommitted key is
/// resolved through the exact precommitment config to freeze its layout
/// before selecting the final group's multi-group root commitment layout.
///
/// # Errors
///
/// Returns an error if input validation, multi-group parameter selection, or
/// commitment execution fails.
pub fn commit_final_group<Cfg, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    stack: &UniformProverStack<'_, Cfg::Field, B>,
    precommitteds: Vec<CommittedGroupProfile>,
) -> Result<FinalCommittedGroupWithHint<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: RuntimeRootCommitPoly<Cfg::Field>,
    B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
{
    let commit_ctx = stack.commit();
    let tensor_ctx = stack.tensor();
    let schedule_key =
        final_group_key_from_polys::<Cfg, P>(polys, expanded, precommitteds.clone())?;
    let schedule = Cfg::runtime_schedule(schedule_key.clone())?;
    let opening_layout = schedule_key.opening_layout()?;
    ensure_prover_schedule_fits_setup::<Cfg>(expanded, &schedule, &opening_layout)?;
    let params = schedule.root_fold().params.final_group.commitment.clone();
    validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
    let (commitment, hint) =
        if should_transform_final_group_commitment::<Cfg>(&schedule_key, params.role_dims().d_a())?
        {
            let transform_d = params.role_dims().d_a();
            let transformed = tensor_project_roots::<Cfg::Field, P, Cfg::ExtField, B>(
                transform_d,
                tensor_ctx,
                polys,
            )?;
            commit_with_validated_params::<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, B>(
                &transformed,
                commit_ctx,
                &params,
            )
        } else {
            commit_with_validated_params::<Cfg::Field, P, B>(polys, commit_ctx, &params)
        }?;
    let descriptor = CommittedGroupProfile::from_params(schedule_key.final_group, &params);
    let batch_profile = akita_types::CommittedGroupBatchProfile {
        final_group: descriptor,
        precommitteds,
    };
    let selection = Cfg::select_schedule_for_profiles(&batch_profile)?.selection();
    Ok((CommittedGroup::new(descriptor, commitment), hint, selection))
}

/// Commit one polynomial bundle under config `Cfg`.
///
/// The config-selected schedule supplies the resolved root commitment layout.
/// The root tensor-projection transform is applied internally when the field
/// tower and schedule call for it.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or commitment
/// execution fails.
pub fn batched_commit<Cfg, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    stack: &UniformProverStack<'_, Cfg::Field, B>,
) -> Result<CommittedGroupWithHint<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: RuntimeRootCommitPoly<Cfg::Field>,
    B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
{
    let commit_ctx = stack.commit();
    let tensor_ctx = stack.tensor();
    let opening_batch = prepare_batched_commit_inputs::<Cfg::Field, P>(polys, expanded)?;
    let params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
    let (commitment, hint) =
        if let Some(transform_ring_d) = root_transform_ring_dim::<Cfg>(&opening_batch)? {
            // A-role tensor-projection operation at the prove schedule's root fold
            // ring dimension.
            let transformed = tensor_project_roots::<Cfg::Field, P, Cfg::ExtField, B>(
                transform_ring_d,
                tensor_ctx,
                polys,
            )?;
            validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
            commit_with_validated_params::<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, B>(
                &transformed,
                commit_ctx,
                &params,
            )?
        } else {
            validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
            commit_with_validated_params::<Cfg::Field, P, B>(polys, commit_ctx, &params)?
        };
    let group = opening_batch.root_final_group_layout()?;
    let descriptor = CommittedGroupProfile::from_params(group, &params);
    Ok((CommittedGroup::new(descriptor, commitment), hint))
}

/// Commit one polynomial bundle using already-selected level parameters.
///
/// The caller has already resolved the root commitment layout (e.g.
/// via [`batched_commit`]); this function owns only the prover-side matrix
/// work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if batched input validation fails or commitment execution
/// fails.
pub fn batched_commit_with_params<F, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<F>,
    ctx: &OperationCtx<'_, F, B>,
    params: &CommittedGroupParams,
) -> Result<CommitmentWithHint<F>, AkitaError>
where
    F: FieldCore
        + CanonicalField
        + RandomSampling
        + FromPrimitiveInt
        + HalvingField
        + HasWide
        + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootCommitSource<F, 32>
        + RootCommitSource<F, 64>
        + RootCommitSource<F, 128>
        + RootCommitSource<F, 256>
        + RootCommitSource<F, 512>
        + RootPolyMeta<F>,
    B: RuntimeCommitBackendFor<F, P>,
{
    prepare_batched_commit_inputs::<F, P>(polys, expanded)?;
    validate_commit_level_params::<F>(params, expanded)?;
    commit_with_validated_params::<F, P, B>(polys, ctx, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AkitaProverSetup;
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp64;
    use akita_types::{OpenCommitMatrixParams, SetupMatrixCapacity, SisModulusProfileId};

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn inner_witness(recomposed_blocks: usize, rows_per_block: usize) -> CommitInnerWitness<F> {
        CommitInnerWitness::from_rows(vec![
            vec![CyclotomicRing::<F, D>::zero(); rows_per_block];
            recomposed_blocks
        ])
    }

    #[test]
    fn commit_inner_shape_accepts_expected_layout() {
        let inner = inner_witness(2, 3);
        validate_commit_inner_shape::<F, D>(&inner, 2, 3).expect("shape should match");
    }

    #[test]
    fn commit_inner_shape_rejects_bad_block_count() {
        let inner = inner_witness(1, 3);
        assert!(validate_commit_inner_shape::<F, D>(&inner, 2, 3).is_err());
    }

    #[test]
    fn commit_inner_shape_rejects_bad_row_count() {
        let inner = inner_witness(2, 2);
        assert!(validate_commit_inner_shape::<F, D>(&inner, 2, 3).is_err());
    }

    #[test]
    fn commit_inner_shape_accepts_many_all_zero_blocks() {
        let num_live_blocks = 1024;
        let inner = inner_witness(num_live_blocks, 3);
        validate_commit_inner_shape::<F, D>(&inner, num_live_blocks, 3).expect("all-zero blocks");
    }

    #[test]
    fn commit_level_params_reject_log_basis_above_i8_range() {
        let expanded = AkitaProverSetup::<F>::generate_with_capacity(
            5,
            1,
            SetupMatrixCapacity {
                num_field_elements: D,
            },
        )
        .unwrap()
        .expanded;
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q32Offset99,
            D,
            9,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(2, 4, 2, 2, 2)
        .unwrap();

        assert!(matches!(
            validate_commit_level_params::<F>(&params, &expanded),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn commit_level_params_do_not_charge_unused_shared_d_footprint() {
        let expanded = AkitaProverSetup::<F>::generate_with_capacity(
            5,
            1,
            SetupMatrixCapacity {
                num_field_elements: D,
            },
        )
        .unwrap()
        .expanded;
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q32Offset99,
            D,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(1, 1, 1, 1, 1)
        .unwrap();
        let d_key = params.open_commit_matrix.sis_table_key();
        params.open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
            d_key.policy,
            d_key.table_digest,
            d_key.modulus_profile,
            8,
            8,
            d_key.coeff_linf_bound,
            D,
        );

        validate_commit_level_params::<F>(&params, &expanded)
            .expect("standalone commitment only materializes A and B");
    }

    #[test]
    fn commit_b_input_len_rejects_overflow() {
        assert_eq!(checked_commit_b_input_len(3, 5).expect("fits"), 15);
        assert!(matches!(
            checked_commit_b_input_len(usize::MAX, 2),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn commit_outer_input_validation_allows_logical_input_longer_than_setup_stride() {
        validate_commit_outer_input_nonempty(9).expect("logical B input may exceed row stride");
        assert!(matches!(
            validate_commit_outer_input_nonempty(0),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
}
