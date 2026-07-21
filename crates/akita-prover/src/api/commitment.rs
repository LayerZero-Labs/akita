//! Prover-owned commitment kernels.

use crate::compute::{
    tensor_root_projection, CommitInnerPlan, OperationCtx, RootCommitKernel, RootCommitSource,
    RootPolyMeta, RuntimeCommitBackendFor, RuntimeRootCommitBackend, RuntimeRootCommitPoly,
    UniformProverStack,
};
use crate::validation::validate_i8_setup_log_basis;
use crate::{CommitInnerWitness, RootTensorProjectionPoly};
use akita_config::{ensure_schedule_fits_setup, CommitmentConfig, ConservativeCommitmentConfig};
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, RandomSampling};
use akita_types::{
    dispatch_for_field, root_tensor_projection_enabled, validate_role_dims,
    validate_role_dims_for_field, AkitaCommitmentHint, AkitaExpandedSetup, AkitaScheduleLookupKey,
    Commitment, CommittedGroupParams, DigitBlocks, FpExtEncoding, OpeningClaimsLayout,
    PolynomialGroupLayout, PrecommittedGroupDescriptor, MULTI_GROUP_ROOT_DENSE_UNSUPPORTED,
};

/// Commitment output plus prover-side hint for one committed polynomial bundle.
///
/// D-free protocol storage: a flat [`Commitment`] plus the D-free
/// [`AkitaCommitmentHint`] (decomposed digit stream only; recomposed inner rows
/// are recomputed on demand, see [`crate::compute::recompose_hint_inner_rows`]).
pub type CommitmentWithHint<F> = (Commitment<F>, AkitaCommitmentHint<F>);

/// Frozen layout, commitment rows, and prover hint for one standalone group.
pub type CommittedGroupWithHint<F> = (
    PrecommittedGroupDescriptor,
    Commitment<F>,
    AkitaCommitmentHint<F>,
);

pub(crate) fn commit_inner_block_digit_count(
    n_a: usize,
    num_digits_open: usize,
) -> Result<usize, AkitaError> {
    if num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "num_digits_open must be nonzero for inner commitment digits".to_string(),
        ));
    }
    n_a.checked_mul(num_digits_open).ok_or_else(|| {
        AkitaError::InvalidSetup(
            "commit inner witness block digit count overflowed usize".to_string(),
        )
    })
}

pub(crate) fn commit_inner_flat_digit_count(
    num_live_blocks: usize,
    n_a: usize,
    num_digits_open: usize,
) -> Result<usize, AkitaError> {
    num_live_blocks
        .checked_mul(commit_inner_block_digit_count(n_a, num_digits_open)?)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "commit inner witness flat digit count overflowed usize".to_string(),
            )
        })
}

#[tracing::instrument(skip_all, name = "validate_commit_inner_shape")]
pub(crate) fn validate_commit_inner_shape<F, const D: usize>(
    inner: &CommitInnerWitness<F>,
    num_live_blocks: usize,
    n_a: usize,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let expected_block_digits = commit_inner_block_digit_count(n_a, num_digits_open)?;
    let expected_flat_digits =
        commit_inner_flat_digit_count(num_live_blocks, n_a, num_digits_open)?;
    validate_i8_setup_log_basis(log_basis, "when recomposing i8 inner commitment digits")?;

    inner.ensure_ring_dim::<D>()?;

    if inner.block_count() != num_live_blocks {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} inner commitment blocks, expected {}",
            inner.block_count(),
            num_live_blocks
        )));
    }
    for block_idx in 0..num_live_blocks {
        let block_rows = inner.recomposed_block_trusted::<D>(block_idx)?;
        if block_rows.len() != n_a {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {} A rows for inner commitment block {}, expected {}",
                block_rows.len(),
                block_idx,
                n_a
            )));
        }
    }

    if inner.decomposed_inner_rows.block_count() != num_live_blocks {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} decomposed inner commitment blocks, expected {}",
            inner.decomposed_inner_rows.block_count(),
            num_live_blocks
        )));
    }
    for (block_idx, &block_digits) in inner.decomposed_inner_rows.block_sizes().iter().enumerate() {
        if block_digits != expected_block_digits {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {} decomposed digits for inner commitment block {}, expected {}",
                block_digits, block_idx, expected_block_digits
            )));
        }
    }
    if inner.decomposed_inner_rows.total_planes() != expected_flat_digits {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} total decomposed inner commitment digits, expected {}",
            inner.decomposed_inner_rows.total_planes(),
            expected_flat_digits
        )));
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
    let setup_len = setup
        .shared_matrix
        .total_ring_elements_at_dyn(params.d_a())?;
    let a_required = params
        .inner_commit_matrix
        .output_rank()
        .checked_mul(params.inner_commit_matrix.input_width())
        .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))?;
    let b_required = params
        .outer_commit_matrix
        .output_rank()
        .checked_mul(params.outer_commit_matrix.input_width())
        .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".to_string()))?;
    let d_required = params
        .open_commit_matrix
        .output_rank()
        .checked_mul(params.open_commit_matrix.input_width())
        .ok_or_else(|| AkitaError::InvalidSetup("D setup footprint overflow".to_string()))?;
    let required = a_required.max(b_required).max(d_required);
    if required > setup_len {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params require {required} setup ring elements but setup has {setup_len}",
        )));
    }
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

pub(crate) fn validate_onehot_chunk_size_for_params<F, P>(
    polys: &[P],
    params: &CommittedGroupParams,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    P: RootPolyMeta<F>,
{
    let expected = params.onehot_chunk_size;
    if expected <= 1 {
        return Ok(());
    }
    for (poly_idx, poly) in polys.iter().enumerate() {
        if let Some(actual) = poly.onehot_chunk_size() {
            if actual != expected {
                return Err(AkitaError::InvalidInput(format!(
                    "one-hot polynomial {poly_idx} uses onehot_k={actual}, but this \
                     config/layout requires onehot_k={expected}"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_batched_onehot_chunk_size_for_params<F, P>(
    polys: &[P],
    params: &CommittedGroupParams,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    P: RootPolyMeta<F>,
{
    let expected = params.onehot_chunk_size;
    if expected <= 1 {
        return Ok(());
    }
    for (poly_idx, poly) in polys.iter().enumerate() {
        match poly.onehot_chunk_size() {
            Some(actual) if actual == expected => {}
            Some(actual) => {
                return Err(AkitaError::InvalidInput(format!(
                    "one-hot polynomial {poly_idx} uses onehot_k={actual}, but this \
                     config/layout requires onehot_k={expected}"
                )));
            }
            None => {
                return Err(AkitaError::InvalidInput(format!(
                    "polynomial {poly_idx} is dense, but this config/layout requires \
                     one-hot polynomials with onehot_k={expected}"
                )));
            }
        }
    }
    Ok(())
}

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
    F: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootCommitSource<F, 32>
        + RootCommitSource<F, 64>
        + RootCommitSource<F, 128>
        + RootCommitSource<F, 256>
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
    let b_input_len_per_poly = commit_inner_flat_digit_count(
        params.num_live_blocks,
        params.inner_commit_matrix.output_rank(),
        params.num_digits_outer,
    )?;
    let total_b_input_len = checked_commit_b_input_len(polys.len(), b_input_len_per_poly)?;
    let num_live_blocks = params.num_live_blocks;
    let n_a = params.inner_commit_matrix.output_rank();
    let num_digits_open = params.num_digits_outer;
    let log_basis = params.log_basis_outer;
    // A-role operation: per-poly inner commit + digit decomposition. The digit
    // planes leave the arm as one FLAT `Vec<i8>` carrier (the per-matrix seam
    // between the inner A-role and outer B-role commitment halves) plus the
    // D-free `DigitBlocks` hint payload; recomposed inner rows are recomputed
    // on demand from the digit stream (S5 re-home), not cached here.
    let (b_input_flat, decomposed_digit_blocks) = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        dims.d_a(),
        |D_A| {
            let flat_len = total_b_input_len.checked_mul(D_A).ok_or_else(|| {
                AkitaError::InvalidSetup("commit inner digit carrier length overflow".to_string())
            })?;
            let per_poly_flat_len = b_input_len_per_poly.checked_mul(D_A).ok_or_else(|| {
                AkitaError::InvalidSetup("commit inner digit carrier length overflow".to_string())
            })?;
            let mut b_input_flat = vec![0i8; flat_len];
            let mut decomposed_digit_blocks: Vec<DigitBlocks> =
                (0..polys.len()).map(|_| DigitBlocks::empty(D_A)).collect();
            cfg_chunks_mut!(b_input_flat, per_poly_flat_len)
                .zip(cfg_iter!(polys))
                .zip(cfg_iter_mut!(decomposed_digit_blocks))
                .try_for_each(|((dst, poly), decomposed)| -> Result<(), AkitaError> {
                    let view = RootCommitSource::<F, D_A>::commit_view(poly)?;
                    let inner =
                        RootCommitKernel::<_, F, D_A>::commit_inner(backend, prepared, view, plan)?;
                    validate_commit_inner_shape::<F, D_A>(
                        &inner,
                        num_live_blocks,
                        n_a,
                        num_digits_open,
                        log_basis,
                    )?;
                    let typed_digits = inner.decomposed_inner_rows_trusted::<D_A>()?;
                    dst.copy_from_slice(typed_digits.typed_planes::<D_A>()?.as_flattened());
                    *decomposed = typed_digits.clone();
                    Ok(())
                })?;
            Ok::<_, AkitaError>((b_input_flat, decomposed_digit_blocks))
        }
    )?;
    validate_commit_outer_input_nonempty(b_input_flat.len())?;
    let n_b = params.outer_commit_matrix.output_rank();
    // B-role operation: the sent commitment rows `u = B·t̂`.
    let commitment = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Outer),
        F,
        dims.d_b(),
        |D_B| {
            let (b_input_digits, remainder) = b_input_flat.as_chunks::<D_B>();
            if !remainder.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "commit digit carrier is not aligned to the outer ring dimension".to_string(),
                ));
            }
            let u = backend.digit_rows::<D_B>(prepared, n_b, b_input_digits, log_basis)?;
            if u.len() != n_b {
                return Err(AkitaError::InvalidSetup(format!(
                    "backend returned {} B commitment rows, expected {n_b}",
                    u.len(),
                )));
            }
            Ok::<_, AkitaError>(Commitment::from_ring_elems(&u))
        }
    )?;
    let hint = AkitaCommitmentHint::new(decomposed_digit_blocks);
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
    F: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootCommitSource<F, 32>
        + RootCommitSource<F, 64>
        + RootCommitSource<F, 128>
        + RootCommitSource<F, 256>
        + RootPolyMeta<F>,
    B: RuntimeCommitBackendFor<F, P>,
{
    prepare_commit_inputs::<F, P>(polys, expanded)?;
    validate_commit_level_params::<F>(params, expanded)?;
    validate_onehot_chunk_size_for_params::<F, P>(polys, params)?;
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
fn should_transform_group_commitment<Cfg>(
    key: &PolynomialGroupLayout,
    ring_d: usize,
) -> Result<bool, AkitaError>
where
    Cfg: CommitmentConfig,
{
    if !root_tensor_projection_enabled::<Cfg::Field, Cfg::ExtField>(ring_d, key.num_vars()) {
        return Ok(false);
    }
    Cfg::runtime_schedule(AkitaScheduleLookupKey::single(*key))?;
    Ok(true)
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
) -> Result<CommitmentWithHint<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: RuntimeRootCommitPoly<Cfg::Field>,
    B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
{
    let commit_ctx = stack.commit();
    let tensor_ctx = stack.tensor();
    let opening_batch = prepare_commit_inputs::<Cfg::Field, P>(polys, expanded)?;
    let params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
    validate_onehot_chunk_size_for_params::<Cfg::Field, P>(polys, &params)?;
    if let Some(transform_ring_d) = root_transform_ring_dim::<Cfg>(&opening_batch)? {
        // A-role tensor-projection operation at the prove schedule's root fold
        // ring dimension.
        let transformed = tensor_project_roots::<Cfg::Field, P, Cfg::ExtField, B>(
            transform_ring_d,
            tensor_ctx,
            polys,
        )?;
        validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
        return commit_with_validated_params::<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, B>(
            &transformed,
            commit_ctx,
            &params,
        );
    }
    validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
    commit_with_validated_params::<Cfg::Field, P, B>(polys, commit_ctx, &params)
}

/// Validate a batched commitment request and derive its `OpeningClaimsLayout`.
///
/// The input slice is one commitment group at the shared opening point.
/// Polynomials may have smaller natural arity than the shared padded batch
/// domain; the largest arity selects the root layout.
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
    if polys.iter().any(|poly| poly.onehot_chunk_size().is_none()) {
        return Err(AkitaError::InvalidInput(
            MULTI_GROUP_ROOT_DENSE_UNSUPPORTED.to_string(),
        ));
    }
    Ok(PolynomialGroupLayout::new(
        opening_batch.max_num_vars(),
        opening_batch.num_total_polynomials(),
    ))
}

/// Commit one standalone one-hot commitment group with conservative B rank.
///
/// Grouped proving is still guarded until the opening phase lands; this API only
/// produces the precommit metadata and commitment object required by that later
/// finalization path.
///
/// # Errors
///
/// Returns an error if the group is empty, dense, unsupported by the setup, or
/// cannot be planned under the conservative-rank policy.
pub fn commit_group<Cfg, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    stack: &UniformProverStack<'_, Cfg::Field, B>,
) -> Result<CommittedGroupWithHint<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: RuntimeRootCommitPoly<Cfg::Field>,
    B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
{
    let commit_ctx = stack.commit();
    let tensor_ctx = stack.tensor();
    let key = validate_group_commit_inputs::<Cfg::Field, P>(polys, expanded)?;
    let opening_batch = OpeningClaimsLayout::new(key.num_vars(), key.num_polynomials())?;
    let params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
    validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
    validate_onehot_chunk_size_for_params::<Cfg::Field, P>(polys, &params)?;
    let (commitment, hint) =
        if should_transform_group_commitment::<Cfg>(&key, params.role_dims().d_a())? {
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
    Ok((
        PrecommittedGroupDescriptor::from_params(key, &params),
        commitment,
        hint,
    ))
}

fn precommitted_layouts_from_keys<Cfg>(
    precommitteds: Vec<PolynomialGroupLayout>,
) -> Result<Vec<PrecommittedGroupDescriptor>, AkitaError>
where
    Cfg: CommitmentConfig,
{
    if precommitteds.is_empty() {
        return Err(AkitaError::InvalidInput(
            "commit_final_group requires at least one precommitted group".to_string(),
        ));
    }
    precommitteds
        .into_iter()
        .map(|key| {
            key.validate()?;
            let singleton = OpeningClaimsLayout::new(key.num_vars(), key.num_polynomials())?;
            let params = <ConservativeCommitmentConfig<Cfg> as CommitmentConfig>::get_params_for_batched_commitment(
                &singleton,
            )?;
            Ok(PrecommittedGroupDescriptor::from_params(key, &params))
        })
        .collect()
}

fn final_group_key_from_polys<Cfg, P>(
    polys: &[P],
    setup: &AkitaExpandedSetup<Cfg::Field>,
    precommitteds: Vec<PolynomialGroupLayout>,
) -> Result<AkitaScheduleLookupKey, AkitaError>
where
    Cfg: CommitmentConfig,
    P: RootPolyMeta<Cfg::Field>,
{
    let opening_batch = prepare_batched_commit_inputs::<Cfg::Field, P>(polys, setup)?;
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
        ),
        precommitteds: precommitted_layouts_from_keys::<Cfg>(precommitteds)?,
    };
    key.validate()?;
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
/// resolved through the conservative commitment config to freeze its layout
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
    precommitteds: Vec<PolynomialGroupLayout>,
) -> Result<CommitmentWithHint<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: RuntimeRootCommitPoly<Cfg::Field>,
    B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
{
    let commit_ctx = stack.commit();
    let tensor_ctx = stack.tensor();
    let schedule_key = final_group_key_from_polys::<Cfg, P>(polys, expanded, precommitteds)?;
    let schedule = Cfg::runtime_schedule(schedule_key.clone())?;
    let opening_layout = schedule_key.opening_layout()?;
    ensure_schedule_fits_setup::<Cfg>(expanded, &schedule, &opening_layout)?;
    let params = schedule.root_fold().params.final_group.commitment.clone();
    validate_batched_onehot_chunk_size_for_params::<Cfg::Field, P>(polys, &params)?;
    validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
    if should_transform_final_group_commitment::<Cfg>(&schedule_key, params.role_dims().d_a())? {
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
    }
}

/// Commit one polynomial bundle under config `Cfg`.
///
/// The config-selected schedule supplies the shared root commitment layout.
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
) -> Result<CommitmentWithHint<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide + 'static,
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>,
    P: RuntimeRootCommitPoly<Cfg::Field>,
    B: RuntimeRootCommitBackend<Cfg::Field, P, Cfg::ExtField>,
{
    let commit_ctx = stack.commit();
    let tensor_ctx = stack.tensor();
    let opening_batch = prepare_batched_commit_inputs::<Cfg::Field, P>(polys, expanded)?;
    let params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
    validate_batched_onehot_chunk_size_for_params::<Cfg::Field, P>(polys, &params)?;
    if let Some(transform_ring_d) = root_transform_ring_dim::<Cfg>(&opening_batch)? {
        // A-role tensor-projection operation at the prove schedule's root fold
        // ring dimension.
        let transformed = tensor_project_roots::<Cfg::Field, P, Cfg::ExtField, B>(
            transform_ring_d,
            tensor_ctx,
            polys,
        )?;
        validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
        return commit_with_validated_params::<Cfg::Field, RootTensorProjectionPoly<Cfg::Field>, B>(
            &transformed,
            commit_ctx,
            &params,
        );
    }
    validate_commit_level_params::<Cfg::Field>(&params, expanded)?;
    commit_with_validated_params::<Cfg::Field, P, B>(polys, commit_ctx, &params)
}

/// Commit one polynomial bundle using already-selected level parameters.
///
/// The caller has already resolved the shared root commitment layout (e.g.
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
    F: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    P: RootCommitSource<F, 32>
        + RootCommitSource<F, 64>
        + RootCommitSource<F, 128>
        + RootCommitSource<F, 256>
        + RootPolyMeta<F>,
    B: RuntimeCommitBackendFor<F, P>,
{
    prepare_batched_commit_inputs::<F, P>(polys, expanded)?;
    validate_commit_level_params::<F>(params, expanded)?;
    validate_batched_onehot_chunk_size_for_params::<F, P>(polys, params)?;
    commit_with_validated_params::<F, P, B>(polys, ctx, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linear::check_decomposed_rows_i8_match;
    use crate::{AkitaProverSetup, MultilinearPolynomial, OneHotPoly};
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp64;
    use akita_types::DigitBlocks;
    use akita_types::{SetupMatrixEnvelope, SisModulusProfileId};

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn inner_witness(
        recomposed_blocks: usize,
        rows_per_block: usize,
        block_sizes: Vec<usize>,
    ) -> CommitInnerWitness<F> {
        CommitInnerWitness::from_parts(
            vec![vec![CyclotomicRing::<F, D>::zero(); rows_per_block]; recomposed_blocks],
            DigitBlocks::zeroed(block_sizes, D).expect("valid digit blocks"),
        )
        .expect("valid inner witness")
    }

    #[test]
    fn commit_inner_shape_accepts_expected_layout() {
        let inner = inner_witness(2, 3, vec![6, 6]);
        validate_commit_inner_shape::<F, D>(&inner, 2, 3, 2, 4).expect("shape should match");
    }

    #[test]
    fn commit_inner_shape_rejects_bad_block_count() {
        let inner = inner_witness(1, 3, vec![6, 6]);
        assert!(validate_commit_inner_shape::<F, D>(&inner, 2, 3, 2, 4).is_err());
    }

    #[test]
    fn commit_inner_shape_rejects_bad_digit_block_size() {
        let inner = inner_witness(2, 3, vec![6, 5]);
        assert!(validate_commit_inner_shape::<F, D>(&inner, 2, 3, 2, 4).is_err());
    }

    #[test]
    fn commit_inner_shape_rejects_recomposition_mismatch() {
        let mut inner = inner_witness(1, 1, vec![2]);
        inner.decomposed_inner_rows.digits_mut()[0] = 1;
        assert!(check_decomposed_rows_i8_match::<F, D>(&inner, 1, 2, 4).is_err());
    }

    #[test]
    fn commit_inner_shape_rejects_nonzero_digits_on_zero_row() {
        let mut inner = inner_witness(1, 3, vec![6]);
        inner.decomposed_inner_rows.digits_mut()[2 * D] = 1;
        assert!(check_decomposed_rows_i8_match::<F, D>(&inner, 3, 2, 4).is_err());
    }

    #[test]
    fn commit_inner_shape_accepts_many_all_zero_blocks() {
        let num_live_blocks = 1024;
        let inner = inner_witness(num_live_blocks, 3, vec![6; num_live_blocks]);
        validate_commit_inner_shape::<F, D>(&inner, num_live_blocks, 3, 2, 4)
            .expect("all-zero blocks");
        check_decomposed_rows_i8_match::<F, D>(&inner, 3, 2, 4).expect("digit consistency");
    }

    #[test]
    fn commit_inner_shape_rejects_log_basis_above_i8_range() {
        let inner = inner_witness(1, 1, vec![2]);
        assert!(matches!(
            validate_commit_inner_shape::<F, D>(&inner, 1, 1, 2, 7),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn commit_level_params_reject_log_basis_above_i8_range() {
        let expanded = AkitaProverSetup::<F>::generate_with_capacity(
            5,
            1,
            D,
            SetupMatrixEnvelope { max_setup_len: 8 },
        )
        .unwrap()
        .expanded;
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q32Offset99,
            D,
            7,
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
    fn commit_b_input_len_rejects_overflow() {
        assert_eq!(checked_commit_b_input_len(3, 5).expect("fits"), 15);
        assert!(matches!(
            checked_commit_b_input_len(usize::MAX, 2),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn onehot_chunk_size_validator_rejects_mismatched_k() {
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q32Offset99,
            D,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_onehot_chunk_size(256);
        let wrong = OneHotPoly::<F, u16>::new(64, D, vec![Some(1), None]).unwrap();
        let ok = OneHotPoly::<F, u16>::new(256, D, vec![Some(1), None]).unwrap();

        assert!(matches!(
            validate_onehot_chunk_size_for_params::<F, _>(&[wrong], &params),
            Err(AkitaError::InvalidInput(_))
        ));
        validate_onehot_chunk_size_for_params::<F, _>(&[ok], &params)
            .expect("matching onehot_k should be accepted");
    }

    #[test]
    fn validate_onehot_chunk_size_rejects_wrapped_onehot_mismatch() {
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q32Offset99,
            D,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_onehot_chunk_size(256);
        let wrong_wrapped = MultilinearPolynomial::<F, u16>::onehot(
            OneHotPoly::<F, u16>::new(64, D, vec![Some(1), None]).unwrap(),
        );
        let ok_wrapped = MultilinearPolynomial::<F, u16>::onehot(
            OneHotPoly::<F, u16>::new(256, D, vec![Some(1), None]).unwrap(),
        );

        assert!(matches!(
            validate_onehot_chunk_size_for_params::<F, _>(&[wrong_wrapped], &params),
            Err(AkitaError::InvalidInput(_))
        ));
        validate_onehot_chunk_size_for_params::<F, _>(&[ok_wrapped], &params)
            .expect("matching wrapped onehot_k should be accepted");
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
