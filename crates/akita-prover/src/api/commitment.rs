//! Prover-owned commitment kernels.

use crate::backend::MultilinearPolynomialView;
use crate::compute::{
    CommitInnerPlan, CommitmentComputeBackend, OperationCtx, RootCommitBackend, RootCommitKernel,
    RootCommitPoly, RootCommitPolys, RootCommitSource, RootPolyShape, RootTensorSource,
    TensorProjectionKernel,
};
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use crate::validation::validate_i8_setup_log_basis;
use crate::DigitRowsComputeBackend;
use crate::{
    CommitInnerWitness, CpuBackend, MultilinearPolynomial, OneHotIndex, RootTensorProjectionPoly,
};
use akita_algebra::CyclotomicRing;
use akita_config::CommitmentConfig;
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, RandomSampling};
use akita_types::{
    root_tensor_projection_enabled, schedule_root_fold_step, AkitaCommitmentHint,
    AkitaExpandedSetup, ClaimIncidenceSummary, FlatDigitBlocks, LevelParams, RingCommitment,
    RingSubfieldEncoding,
};

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
    num_blocks: usize,
    n_a: usize,
    num_digits_open: usize,
) -> Result<usize, AkitaError> {
    num_blocks
        .checked_mul(commit_inner_block_digit_count(n_a, num_digits_open)?)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "commit inner witness flat digit count overflowed usize".to_string(),
            )
        })
}

pub(crate) fn validate_commit_inner_witness_shape<F, const D: usize>(
    inner: &CommitInnerWitness<F, D>,
    num_blocks: usize,
    n_a: usize,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let expected_block_digits = commit_inner_block_digit_count(n_a, num_digits_open)?;
    let expected_flat_digits = commit_inner_flat_digit_count(num_blocks, n_a, num_digits_open)?;
    validate_i8_setup_log_basis(log_basis, "when recomposing i8 inner commitment digits")?;

    if inner.recomposed_inner_rows.len() != num_blocks {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} inner commitment blocks, expected {}",
            inner.recomposed_inner_rows.len(),
            num_blocks
        )));
    }
    for (block_idx, block_rows) in inner.recomposed_inner_rows.iter().enumerate() {
        if block_rows.len() != n_a {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {} A rows for inner commitment block {}, expected {}",
                block_rows.len(),
                block_idx,
                n_a
            )));
        }
    }

    if inner.decomposed_inner_rows.block_count() != num_blocks {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} decomposed inner commitment blocks, expected {}",
            inner.decomposed_inner_rows.block_count(),
            num_blocks
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
    if inner.decomposed_inner_rows.flat_digits().len() != expected_flat_digits {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} total decomposed inner commitment digits, expected {}",
            inner.decomposed_inner_rows.flat_digits().len(),
            expected_flat_digits
        )));
    }
    for (block_idx, block_digits) in inner.decomposed_inner_rows.iter_blocks().enumerate() {
        let recomposed_block = &inner.recomposed_inner_rows[block_idx];
        for (row_idx, row_digits) in block_digits.chunks(num_digits_open).enumerate() {
            let recomposed = CyclotomicRing::gadget_recompose_pow2_i8(row_digits, log_basis);
            if recomposed_block[row_idx] != recomposed {
                return Err(AkitaError::InvalidSetup(format!(
                    "backend returned recomposed row {row_idx} for inner commitment block {block_idx} that does not match its decomposed digits"
                )));
            }
        }
    }

    Ok(())
}

pub(crate) fn validate_commit_level_params<F, const D: usize>(
    params: &LevelParams,
    setup: &AkitaExpandedSetup<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore,
{
    if params.ring_dimension != D {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params ring dimension {} does not match static D={D}",
            params.ring_dimension
        )));
    }
    if params.num_blocks == 0 || params.block_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero num_blocks and block_len".to_string(),
        ));
    }
    if params.num_digits_commit == 0 || params.num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "commit params require nonzero digit depths".to_string(),
        ));
    }
    validate_i8_setup_log_basis(params.log_basis, "for i8 commitment decomposition")?;
    let expected_a_width = params
        .block_len
        .checked_mul(params.num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("A commit width overflow".to_string()))?;
    if params.a_key.col_len() != expected_a_width {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params A width {} does not match block_len * num_digits_commit = {expected_a_width}",
            params.a_key.col_len()
        )));
    }
    if params.b_key.col_len() == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params require nonzero B width, got B={}",
            params.b_key.col_len()
        )));
    }
    // TODO: re-enable this D-side nonzero check (or scope it to non-root-direct
    // schedules) once root-direct commit params no longer carry a
    // zero-width D-key placeholder. Root-direct schedules don't run
    // the relation fold (which is what consumes D), so the planner
    // deliberately emits `d_key.col_len = 0`. This check should
    // eventually be gated on schedule shape (root-direct vs. fold-root)
    // rather than disabled outright.
    // if params.d_key.col_len() == 0 {
    //     return Err(AkitaError::InvalidSetup(format!(
    //         "commit params require nonzero D width, got D={}",
    //         params.d_key.col_len()
    //     )));
    // }
    let setup_len = setup.shared_matrix.total_ring_elements_at::<D>()?;
    let a_required = params
        .a_key
        .row_len()
        .checked_mul(params.a_key.col_len())
        .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))?;
    let b_required = params
        .b_key
        .row_len()
        .checked_mul(params.b_key.col_len())
        .ok_or_else(|| AkitaError::InvalidSetup("B setup footprint overflow".to_string()))?;
    let d_required = params
        .d_key
        .row_len()
        .checked_mul(params.d_key.col_len())
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

fn commit_inner_plan(params: &LevelParams) -> CommitInnerPlan {
    CommitInnerPlan {
        n_a: params.a_key.row_len(),
        block_len: params.block_len,
        num_digits_commit: params.num_digits_commit,
        num_digits_open: params.num_digits_open,
        log_basis: params.log_basis,
    }
}

/// Validate a singleton commitment request against prover setup capacity.
///
/// # Errors
///
/// Returns an error if the request is empty, mixes polynomial dimensions, or
/// exceeds the prover setup capacity.
pub fn prepare_commit_inputs<F, const D: usize, P>(
    polys: &[P],
    setup: &AkitaExpandedSetup<F>,
) -> Result<ClaimIncidenceSummary, AkitaError>
where
    F: FieldCore,
    P: RootPolyShape<F, D>,
{
    if polys.is_empty() {
        return Err(AkitaError::InvalidInput(
            "commit requires at least one polynomial".to_string(),
        ));
    }
    let num_vars = RootPolyShape::num_vars(&polys[0]);
    if polys.iter().any(|p| RootPolyShape::num_vars(p) != num_vars) {
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

    ClaimIncidenceSummary::same_point(num_vars, polys.len())
}

fn checked_commit_b_input_len(total_polys: usize, per_poly: usize) -> Result<usize, AkitaError> {
    total_polys.checked_mul(per_poly).ok_or_else(|| {
        AkitaError::InvalidInput(format!(
            "commit B digit input length overflow for {total_polys} polynomials with {per_poly} digits each"
        ))
    })
}

fn commit_poly_inner_witness<F, const D: usize, P, B>(
    poly: &P,
    ctx: &OperationCtx<'_, F, B, D>,
    plan: CommitInnerPlan,
) -> Result<CommitInnerWitness<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
    P: RootCommitSource<F, D>,
    B: CommitmentComputeBackend<F>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
    let view = poly.commit_view()?;
    RootCommitKernel::commit_inner_witness(ctx.backend(), ctx.prepared(), view, plan)
}

fn commit_multilinear_poly_inner_witness<'p, F, const D: usize, I>(
    poly: &MultilinearPolynomial<'p, F, D, I>,
    ctx: &OperationCtx<'_, F, CpuBackend, D>,
    plan: CommitInnerPlan,
) -> Result<CommitInnerWitness<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    let view = poly.commit_view()?;
    <CpuBackend as RootCommitKernel<MultilinearPolynomialView<'_, 'p, F, D, I>, F, D>>::commit_inner_witness(
        ctx.backend(),
        ctx.prepared(),
        view,
        plan,
    )
}

fn commit_multilinear_with_validated_params<'p, F, const D: usize, I>(
    polys: &'p [MultilinearPolynomial<'p, F, D, I>],
    ctx: &OperationCtx<'_, F, CpuBackend, D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide,
    I: OneHotIndex,
{
    let plan = commit_inner_plan(params);
    let b_input_len_per_poly = commit_inner_flat_digit_count(
        params.num_blocks,
        params.a_key.row_len(),
        params.num_digits_open,
    )?;
    let total_b_input_len = checked_commit_b_input_len(polys.len(), b_input_len_per_poly)?;
    let mut b_input_digits = vec![[0i8; D]; total_b_input_len];
    let mut decomposed_inner_rows: Vec<FlatDigitBlocks<D>> = (0..polys.len())
        .map(|_| FlatDigitBlocks::new(Vec::new(), Vec::new()))
        .collect::<Result<_, _>>()?;
    let mut recomposed_inner_rows: Vec<Vec<Vec<CyclotomicRing<F, D>>>> =
        vec![Vec::new(); polys.len()];
    cfg_chunks_mut!(b_input_digits, b_input_len_per_poly)
        .zip(cfg_iter!(polys))
        .zip(cfg_iter_mut!(decomposed_inner_rows))
        .zip(cfg_iter_mut!(recomposed_inner_rows))
        .try_for_each(
            |(((dst, poly), decomposed), recomposed)| -> Result<(), AkitaError> {
                let inner = commit_multilinear_poly_inner_witness(poly, ctx, plan)?;
                validate_commit_inner_witness_shape(
                    &inner,
                    params.num_blocks,
                    params.a_key.row_len(),
                    params.num_digits_open,
                    params.log_basis,
                )?;
                dst.copy_from_slice(inner.decomposed_inner_rows.flat_digits());
                *decomposed = inner.decomposed_inner_rows;
                *recomposed = inner.recomposed_inner_rows;
                Ok(())
            },
        )?;
    #[cfg(feature = "zk")]
    let b_blinding_digits =
        sample_blinding_digits::<F, D>(params.b_key.row_len(), params.log_basis)?;
    validate_commit_outer_input_nonempty(b_input_digits.len())?;
    #[cfg(feature = "zk")]
    let mut u: Vec<CyclotomicRing<F, D>> = ctx.backend().digit_rows::<D>(
        ctx.prepared(),
        params.b_key.row_len(),
        &b_input_digits,
        params.log_basis,
    )?;
    #[cfg(not(feature = "zk"))]
    let u: Vec<CyclotomicRing<F, D>> = ctx.backend().digit_rows::<D>(
        ctx.prepared(),
        params.b_key.row_len(),
        &b_input_digits,
        params.log_basis,
    )?;
    #[cfg(feature = "zk")]
    {
        let blinding_rows = ctx.backend().zk_b_digit_rows::<D>(
            ctx.prepared(),
            params.b_key.row_len(),
            b_blinding_digits.flat_digits().len(),
            b_blinding_digits.flat_digits(),
        )?;
        for (row, blinding) in u.iter_mut().zip(blinding_rows) {
            *row += blinding;
        }
    }
    if u.len() != params.b_key.row_len() {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} B commitment rows, expected {}",
            u.len(),
            params.b_key.row_len()
        )));
    }
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
        decomposed_inner_rows,
        recomposed_inner_rows,
        #[cfg(feature = "zk")]
        vec![b_blinding_digits],
    );
    Ok((RingCommitment { u }, hint))
}

/// Commit a borrowed [`MultilinearPolynomial`] batch on [`CpuBackend`].
///
/// Generic [`commit`] / [`CommitmentProver::commit`] require
/// `for<'a> RootCommitKernel<<P as RootCommitSource>::CommitView<'a>>` with `P` still a type
/// parameter. For [`MultilinearPolynomial`], `CommitView<'a>` carries a second lifetime tied
/// to the wrapped dense/one-hot borrow, and the current trait solver then requires those borrows
/// to be `'static`. This entry point keeps the wrapper lifetime explicit via UFCS on
/// [`MultilinearPolynomialView`].
///
/// Root tensor projection is not supported here: configs whose schedule requires projection
/// before commit return [`AkitaError::InvalidInput`]. Use generic [`commit`] with a
/// homogeneous root type when projection is required.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, root tensor projection is
/// required, or commitment execution fails.
#[allow(clippy::type_complexity)]
pub fn commit_multilinear_polynomials<Cfg, const D: usize, I>(
    polys: &[MultilinearPolynomial<'_, Cfg::Field, D, I>],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    backend: &CpuBackend,
    prepared: &<CpuBackend as crate::compute::ComputeBackendSetup<Cfg::Field>>::PreparedSetup<D>,
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
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>,
    I: OneHotIndex,
{
    let ctx = OperationCtx::new(backend, prepared, expanded)?;
    let incidence =
        prepare_commit_inputs::<Cfg::Field, D, MultilinearPolynomial<'_, Cfg::Field, D, I>>(
            polys, expanded,
        )?;
    if should_transform_root_commitment::<Cfg, D>(&incidence)? {
        return Err(AkitaError::InvalidInput(
            "commit_multilinear_polynomials does not support root tensor projection; use `commit` with owned roots or a homogeneous backend type".to_string(),
        ));
    }
    let params = Cfg::get_params_for_batched_commitment(&incidence)?;
    validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
    commit_multilinear_with_validated_params(polys, &ctx, &params)
}

fn tensor_project_root<F, E, const D: usize, P, B>(
    poly: &P,
    ctx: &OperationCtx<'_, F, B, D>,
) -> Result<RootTensorProjectionPoly<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: akita_field::ExtField<F> + RingSubfieldEncoding<F>,
    P: RootTensorSource<F, D>,
    B: CommitmentComputeBackend<F>
        + for<'a> TensorProjectionKernel<<P as RootTensorSource<F, D>>::TensorView<'a>, F, E, D>,
{
    let view = poly.tensor_view()?;
    TensorProjectionKernel::root_projection(ctx.backend(), Some(ctx.prepared()), view)
}

fn commit_with_validated_params<F, const D: usize, P, B>(
    polys: &[P],
    ctx: &OperationCtx<'_, F, B, D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: RootCommitSource<F, D>,
    B: CommitmentComputeBackend<F>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
    let plan = commit_inner_plan(params);
    let b_input_len_per_poly = commit_inner_flat_digit_count(
        params.num_blocks,
        params.a_key.row_len(),
        params.num_digits_open,
    )?;
    let total_b_input_len = checked_commit_b_input_len(polys.len(), b_input_len_per_poly)?;
    let mut b_input_digits = vec![[0i8; D]; total_b_input_len];
    let mut decomposed_inner_rows: Vec<FlatDigitBlocks<D>> = (0..polys.len())
        .map(|_| FlatDigitBlocks::new(Vec::new(), Vec::new()))
        .collect::<Result<_, _>>()?;
    let mut recomposed_inner_rows: Vec<Vec<Vec<CyclotomicRing<F, D>>>> =
        vec![Vec::new(); polys.len()];
    cfg_chunks_mut!(b_input_digits, b_input_len_per_poly)
        .zip(cfg_iter!(polys))
        .zip(cfg_iter_mut!(decomposed_inner_rows))
        .zip(cfg_iter_mut!(recomposed_inner_rows))
        .try_for_each(
            |(((dst, poly), decomposed), recomposed)| -> Result<(), AkitaError> {
                let inner = commit_poly_inner_witness(poly, ctx, plan)?;
                validate_commit_inner_witness_shape(
                    &inner,
                    params.num_blocks,
                    params.a_key.row_len(),
                    params.num_digits_open,
                    params.log_basis,
                )?;
                dst.copy_from_slice(inner.decomposed_inner_rows.flat_digits());
                *decomposed = inner.decomposed_inner_rows;
                *recomposed = inner.recomposed_inner_rows;
                Ok(())
            },
        )?;
    #[cfg(feature = "zk")]
    let b_blinding_digits =
        sample_blinding_digits::<F, D>(params.b_key.row_len(), params.log_basis)?;
    validate_commit_outer_input_nonempty(b_input_digits.len())?;
    #[cfg(feature = "zk")]
    let mut u: Vec<CyclotomicRing<F, D>> = ctx.backend().digit_rows::<D>(
        ctx.prepared(),
        params.b_key.row_len(),
        &b_input_digits,
        params.log_basis,
    )?;
    #[cfg(not(feature = "zk"))]
    let u: Vec<CyclotomicRing<F, D>> = ctx.backend().digit_rows::<D>(
        ctx.prepared(),
        params.b_key.row_len(),
        &b_input_digits,
        params.log_basis,
    )?;
    #[cfg(feature = "zk")]
    {
        let blinding_rows = ctx.backend().zk_b_digit_rows::<D>(
            ctx.prepared(),
            params.b_key.row_len(),
            b_blinding_digits.flat_digits().len(),
            b_blinding_digits.flat_digits(),
        )?;
        for (row, blinding) in u.iter_mut().zip(blinding_rows) {
            *row += blinding;
        }
    }
    if u.len() != params.b_key.row_len() {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} B commitment rows, expected {}",
            u.len(),
            params.b_key.row_len()
        )));
    }
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
        decomposed_inner_rows,
        recomposed_inner_rows,
        #[cfg(feature = "zk")]
        vec![b_blinding_digits],
    );
    Ok((RingCommitment { u }, hint))
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
pub fn commit_with_params<F, const D: usize, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: RootCommitSource<F, D>,
    B: CommitmentComputeBackend<F>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
    let ctx = OperationCtx::new(backend, prepared, expanded)?;
    prepare_commit_inputs::<F, D, P>(polys, expanded)?;
    validate_commit_level_params::<F, D>(params, expanded)?;
    commit_with_validated_params::<F, D, P, B>(polys, &ctx, params)
}

#[allow(clippy::type_complexity)]
fn commit_tensor_projected_root<Cfg, const D: usize, P, B>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    ctx: &OperationCtx<'_, Cfg::Field, B, D>,
    incidence: &ClaimIncidenceSummary,
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
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>,
    P: RootCommitPoly<Cfg::Field, D>,
    B: RootCommitBackend<Cfg::Field, P, Cfg::ChallengeField, D>,
{
    let transformed = polys
        .iter()
        .map(|poly| tensor_project_root::<Cfg::Field, Cfg::ChallengeField, D, P, B>(poly, ctx))
        .collect::<Result<Vec<RootTensorProjectionPoly<Cfg::Field, D>>, _>>()?;
    let params = Cfg::get_params_for_batched_commitment(incidence)?;
    validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
    commit_with_validated_params::<Cfg::Field, D, RootTensorProjectionPoly<Cfg::Field, D>, B>(
        &transformed,
        ctx,
        &params,
    )
}

#[allow(clippy::type_complexity)]
fn batched_commit_tensor_projected_root<Cfg, const D: usize, P, B>(
    polys_per_point: &[&[P]],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    ctx: &OperationCtx<'_, Cfg::Field, B, D>,
    incidence: &ClaimIncidenceSummary,
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
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>,
    P: RootCommitPoly<Cfg::Field, D>,
    B: RootCommitBackend<Cfg::Field, P, Cfg::ChallengeField, D>,
{
    let transformed: Vec<Vec<RootTensorProjectionPoly<Cfg::Field, D>>> = polys_per_point
        .iter()
        .map(|polys| {
            polys
                .iter()
                .map(|poly| {
                    tensor_project_root::<Cfg::Field, Cfg::ChallengeField, D, P, B>(poly, ctx)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<_, _>>()?;
    let transformed_refs: Vec<&[RootTensorProjectionPoly<Cfg::Field, D>]> =
        transformed.iter().map(Vec::as_slice).collect();
    let params = Cfg::get_params_for_batched_commitment(incidence)?;
    validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
    batched_commit_with_validated_params::<Cfg::Field, D, RootTensorProjectionPoly<Cfg::Field, D>, B>(
        &transformed_refs,
        ctx,
        &params,
    )
}

/// Decide whether a root commitment must be tensor-projected before commit.
///
/// Root tensor projection only applies when the field tower admits it and the
/// config-selected schedule starts with a fold. This is the prover-owned
/// analogue of the former scheme-local `should_transform_root_commitment`.
///
/// # Errors
///
/// Propagates [`CommitmentConfig::get_params_for_prove`].
fn should_transform_root_commitment<Cfg, const D: usize>(
    incidence: &ClaimIncidenceSummary,
) -> Result<bool, AkitaError>
where
    Cfg: CommitmentConfig,
{
    if !root_tensor_projection_enabled::<Cfg::Field, Cfg::ClaimField, Cfg::ChallengeField, D>(
        incidence.num_vars(),
    ) {
        return Ok(false);
    }
    let schedule = Cfg::get_params_for_prove(incidence)?;
    Ok(schedule_root_fold_step(&schedule).is_some())
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
pub fn commit<Cfg, const D: usize, P, B>(
    bundle: RootCommitPolys<'_, P>,
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
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>,
    P: RootCommitPoly<Cfg::Field, D>,
    B: RootCommitBackend<Cfg::Field, P, Cfg::ChallengeField, D>,
{
    let polys = bundle.as_slice();
    let ctx = OperationCtx::new(backend, prepared, expanded)?;
    let incidence = prepare_commit_inputs::<Cfg::Field, D, P>(polys, expanded)?;
    if should_transform_root_commitment::<Cfg, D>(&incidence)? {
        return commit_tensor_projected_root::<Cfg, D, P, B>(polys, expanded, &ctx, &incidence);
    }
    let params = Cfg::get_params_for_batched_commitment(&incidence)?;
    validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
    commit_with_validated_params::<Cfg::Field, D, P, B>(polys, &ctx, &params)
}

impl<'a, P> RootCommitPolys<'a, P> {
    /// Commit this bundle with `P` fixed from `self`, for rustc inference at call sites.
    #[allow(clippy::type_complexity)]
    pub fn commit_with<Cfg, const D: usize, B>(
        self,
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
        <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
        Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field>,
        Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>,
        P: RootCommitPoly<Cfg::Field, D>,
        B: RootCommitBackend<Cfg::Field, P, Cfg::ChallengeField, D>,
    {
        commit::<Cfg, D, P, B>(self, expanded, backend, prepared)
    }
}

/// Validate a multipoint commitment request and derive its
/// `ClaimIncidenceSummary`.
///
/// `polys_per_point[i]` is the polynomial bundle committed at opening point
/// `i`. Bundles may differ in length; every bundle must be nonempty and every
/// polynomial across every bundle must share the same `num_vars`.
///
/// # Errors
///
/// Returns an error if `polys_per_point` is empty, any bundle is empty, any
/// polynomial dimension mismatches, the total polynomial count overflows or
/// exceeds the prover setup capacity, the point count exceeds the prover
/// setup capacity, or the variable count exceeds the prover setup capacity.
pub fn prepare_batched_commit_inputs<F, const D: usize, P>(
    polys_per_point: &[&[P]],
    setup: &AkitaExpandedSetup<F>,
) -> Result<ClaimIncidenceSummary, AkitaError>
where
    F: FieldCore,
    P: RootPolyShape<F, D>,
{
    if polys_per_point.is_empty() {
        return Err(AkitaError::InvalidInput(
            "batched_commit requires at least one opening point".to_string(),
        ));
    }
    if polys_per_point.len() > setup.seed.max_num_points {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received {} opening points but setup supports at most {}",
            polys_per_point.len(),
            setup.seed.max_num_points
        )));
    }
    let first_bundle = polys_per_point.first().ok_or_else(|| {
        AkitaError::InvalidInput("batched_commit requires at least one opening point".to_string())
    })?;
    let first_poly = first_bundle.first().ok_or_else(|| {
        AkitaError::InvalidInput("batched_commit bundles must be nonempty".to_string())
    })?;
    let num_vars = RootPolyShape::num_vars(first_poly);
    if num_vars > setup.seed.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received a polynomial with {} variables but setup supports at most {}",
            num_vars, setup.seed.max_num_vars
        )));
    }

    let mut num_polys_per_point = Vec::with_capacity(polys_per_point.len());
    let mut total_polys = 0usize;
    for (point_idx, bundle) in polys_per_point.iter().enumerate() {
        if bundle.is_empty() {
            return Err(AkitaError::InvalidInput(format!(
                "batched_commit bundle at point {point_idx} is empty"
            )));
        }
        if bundle
            .iter()
            .any(|p| RootPolyShape::num_vars(p) != num_vars)
        {
            return Err(AkitaError::InvalidInput(
                "batched_commit requires every polynomial to share num_vars".to_string(),
            ));
        }
        num_polys_per_point.push(bundle.len());
        total_polys = total_polys.checked_add(bundle.len()).ok_or_else(|| {
            AkitaError::InvalidInput("batched_commit total polynomial count overflow".to_string())
        })?;
    }
    if total_polys > setup.seed.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received {total_polys} polynomials but setup supports at most {}",
            setup.seed.max_num_batched_polys
        )));
    }

    ClaimIncidenceSummary::from_point_polys(num_vars, num_polys_per_point)
}

/// Commit one polynomial bundle per opening point under config `Cfg`.
///
/// The config-selected schedule supplies the shared root commitment layout.
/// Every per-point bundle is committed with that one layout, guaranteeing the
/// produced commitments are compatible with the layout `batched_prove` will
/// select for the same incidence. The root tensor-projection transform is
/// applied internally when the field tower and schedule call for it.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or any per-
/// point commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit<Cfg, const D: usize, P, B>(
    polys_per_point: &[&[P]],
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
    <Cfg::Field as HasWide>::Wide: From<Cfg::Field> + ReduceTo<Cfg::Field>,
    Cfg::ClaimField: RingSubfieldEncoding<Cfg::Field>,
    Cfg::ChallengeField: RingSubfieldEncoding<Cfg::Field>,
    P: RootCommitPoly<Cfg::Field, D>,
    B: RootCommitBackend<Cfg::Field, P, Cfg::ChallengeField, D>,
{
    let ctx = OperationCtx::new(backend, prepared, expanded)?;
    let incidence = prepare_batched_commit_inputs::<Cfg::Field, D, P>(polys_per_point, expanded)?;
    if should_transform_root_commitment::<Cfg, D>(&incidence)? {
        return batched_commit_tensor_projected_root::<Cfg, D, P, B>(
            polys_per_point,
            expanded,
            &ctx,
            &incidence,
        );
    }
    let params = Cfg::get_params_for_batched_commitment(&incidence)?;
    validate_commit_level_params::<Cfg::Field, D>(&params, expanded)?;
    batched_commit_with_validated_params::<Cfg::Field, D, P, B>(polys_per_point, &ctx, &params)
}

#[allow(clippy::type_complexity)]
fn batched_commit_with_validated_params<F, const D: usize, P, B>(
    polys_per_point: &[&[P]],
    ctx: &OperationCtx<'_, F, B, D>,
    params: &LevelParams,
) -> Result<Vec<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>)>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: RootCommitSource<F, D>,
    B: CommitmentComputeBackend<F>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
    let mut out = Vec::with_capacity(polys_per_point.len());
    for polys in polys_per_point {
        out.push(commit_with_validated_params::<F, D, P, B>(
            polys, ctx, params,
        )?);
    }
    Ok(out)
}

/// Commit one polynomial bundle per opening point using already-selected
/// level parameters.
///
/// The caller has already resolved the shared root commitment layout (e.g.
/// via [`batched_commit`]); this function owns only the prover-
/// side matrix work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if batched input validation fails or any per-point
/// commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_params<F, const D: usize, P, B>(
    polys_per_point: &[&[P]],
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<Vec<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>)>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: RootCommitSource<F, D>,
    B: CommitmentComputeBackend<F>
        + for<'a> RootCommitKernel<<P as RootCommitSource<F, D>>::CommitView<'a>, F, D>,
{
    let ctx = OperationCtx::new(backend, prepared, expanded)?;
    prepare_batched_commit_inputs::<F, D, P>(polys_per_point, expanded)?;
    validate_commit_level_params::<F, D>(params, expanded)?;
    batched_commit_with_validated_params::<F, D, P, B>(polys_per_point, &ctx, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AkitaProverSetup;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp64;
    use akita_types::{SetupMatrixEnvelope, SisModulusFamily};

    type F = Fp64<4294967197>;
    const D: usize = 32;

    fn inner_witness(
        recomposed_blocks: usize,
        rows_per_block: usize,
        block_sizes: Vec<usize>,
    ) -> CommitInnerWitness<F, D> {
        let total_digits = block_sizes.iter().sum();
        CommitInnerWitness {
            recomposed_inner_rows: vec![
                vec![CyclotomicRing::<F, D>::zero(); rows_per_block];
                recomposed_blocks
            ],
            decomposed_inner_rows: FlatDigitBlocks::new(vec![[0i8; D]; total_digits], block_sizes)
                .expect("valid flat digit blocks"),
        }
    }

    #[test]
    fn commit_inner_witness_shape_accepts_expected_layout() {
        let inner = inner_witness(2, 3, vec![6, 6]);
        validate_commit_inner_witness_shape(&inner, 2, 3, 2, 4).expect("shape should match");
    }

    #[test]
    fn commit_inner_witness_shape_rejects_bad_block_count() {
        let inner = inner_witness(1, 3, vec![6, 6]);
        assert!(validate_commit_inner_witness_shape(&inner, 2, 3, 2, 4).is_err());
    }

    #[test]
    fn commit_inner_witness_shape_rejects_bad_digit_block_size() {
        let inner = inner_witness(2, 3, vec![6, 5]);
        assert!(validate_commit_inner_witness_shape(&inner, 2, 3, 2, 4).is_err());
    }

    #[test]
    fn commit_inner_witness_shape_rejects_recomposition_mismatch() {
        let mut inner = inner_witness(1, 1, vec![2]);
        inner.decomposed_inner_rows.flat_digits_mut()[0][0] = 1;
        assert!(validate_commit_inner_witness_shape(&inner, 1, 1, 2, 4).is_err());
    }

    #[test]
    fn commit_inner_witness_shape_rejects_log_basis_above_i8_range() {
        let inner = inner_witness(1, 1, vec![2]);
        assert!(matches!(
            validate_commit_inner_witness_shape(&inner, 1, 1, 2, 7),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn commit_level_params_reject_log_basis_above_i8_range() {
        let expanded = AkitaProverSetup::<F, D>::generate_with_capacity(
            5,
            1,
            1,
            SetupMatrixEnvelope {
                max_setup_len: 8,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
            },
        )
        .unwrap()
        .expanded;
        let params = LevelParams::params_only(
            SisModulusFamily::Q32,
            D,
            7,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(1, 1, 2, 2, 0)
        .unwrap();

        assert!(matches!(
            validate_commit_level_params::<F, D>(&params, &expanded),
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
    fn commit_outer_input_validation_allows_logical_input_longer_than_setup_stride() {
        validate_commit_outer_input_nonempty(9).expect("logical B input may exceed row stride");
        assert!(matches!(
            validate_commit_outer_input_nonempty(0),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn commit_matches_commit_with_params_on_dense_poly() {
        use crate::compute::{ComputeBackendSetup, RootCommitPolys};
        use crate::DensePoly;
        use akita_config::proof_optimized::fp64;
        use akita_config::CommitmentConfig;
        use akita_field::FromPrimitiveInt;

        type Cfg = fp64::D32Full;
        type PolyF = <Cfg as CommitmentConfig>::Field;
        const POLY_D: usize = Cfg::D;
        const NUM_VARS: usize = 10;

        let len = 1usize << NUM_VARS;
        let evals: Vec<PolyF> = (0..len)
            .map(|idx| PolyF::from_u64((idx as u64) + 1))
            .collect();
        let poly = DensePoly::<PolyF, POLY_D>::from_field_evals(NUM_VARS, &evals).unwrap();
        let polys = [poly];

        let setup = AkitaProverSetup::<PolyF, POLY_D>::generate_with_capacity(
            NUM_VARS,
            1,
            1,
            SetupMatrixEnvelope {
                max_setup_len: 8192,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
            },
        )
        .unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let expanded = &setup.expanded;
        let incidence =
            prepare_commit_inputs::<PolyF, POLY_D, DensePoly<PolyF, POLY_D>>(&polys, expanded)
                .unwrap();
        let params = Cfg::get_params_for_batched_commitment(&incidence).unwrap();

        let via_params = commit_with_params::<PolyF, POLY_D, DensePoly<PolyF, POLY_D>, CpuBackend>(
            &polys,
            expanded,
            &CpuBackend,
            &prepared,
            &params,
        )
        .expect("commit_with_params");
        let via_commit = commit::<Cfg, POLY_D, DensePoly<PolyF, POLY_D>, CpuBackend>(
            RootCommitPolys::new(&polys),
            expanded,
            &CpuBackend,
            &prepared,
        )
        .expect("commit");

        assert_eq!(via_params.0.u, via_commit.0.u);
        assert_eq!(via_params.1, via_commit.1);
    }
}
