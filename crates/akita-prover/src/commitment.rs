//! Prover-owned commitment kernels.

use crate::crt_ntt::NttSlotCache;
use crate::linear::mat_vec_mul_ntt_single_i8;
use crate::{HachiPolyOps, HachiProverSetup};
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore, HachiError};
use akita_types::{
    checked_total_groups, FlatDigitBlocks, HachiCommitmentHint, HachiRootBatchSummary, LevelParams,
    RingCommitment, Schedule, Step,
};

/// Config-free summary of a validated singleton commitment request.
pub struct PreparedCommitInputs {
    /// Number of variables in every committed polynomial.
    pub num_vars: usize,
    /// Number of polynomials committed together.
    pub num_polys: usize,
}

/// Config-free summary of a validated grouped batched commitment request.
pub struct PreparedBatchedCommitInputs {
    /// Number of variables in every committed polynomial.
    pub num_vars: usize,
    /// Number of polynomials across all commitment groups.
    pub total_claims: usize,
    /// Polynomial count for each commitment group.
    pub claim_group_sizes: Vec<usize>,
    /// Number of distinct opening points represented by the grouped shape.
    pub point_count: usize,
}

/// Validate a singleton commitment request against prover setup capacity.
///
/// # Errors
///
/// Returns an error if the request is empty, mixes polynomial dimensions, or
/// exceeds the prover setup capacity.
pub fn prepare_commit_inputs<F, const D: usize, P>(
    polys: &[P],
    setup: &HachiProverSetup<F, D>,
) -> Result<PreparedCommitInputs, HachiError>
where
    F: FieldCore,
    P: HachiPolyOps<F, D>,
{
    if polys.is_empty() {
        return Err(HachiError::InvalidInput(
            "commit requires at least one polynomial".to_string(),
        ));
    }
    let num_vars = polys[0].num_vars();
    if polys.iter().any(|p| p.num_vars() != num_vars) {
        return Err(HachiError::InvalidInput(
            "all polynomials in a batched commit must have the same num_vars".to_string(),
        ));
    }
    if polys.len() > setup.expanded.seed.max_num_batched_polys {
        return Err(HachiError::InvalidInput(format!(
            "commit received {} polynomials but setup supports at most {}",
            polys.len(),
            setup.expanded.seed.max_num_batched_polys
        )));
    }
    if num_vars > setup.expanded.seed.max_num_vars {
        return Err(HachiError::InvalidInput(format!(
            "commit received a polynomial with {} variables but setup supports at most {}",
            num_vars, setup.expanded.seed.max_num_vars
        )));
    }

    Ok(PreparedCommitInputs {
        num_vars,
        num_polys: polys.len(),
    })
}

/// Validate and summarize grouped batched commitment inputs.
///
/// # Errors
///
/// Returns an error if the grouped shape is malformed, mixes polynomial
/// dimensions, overflows, or exceeds the prover setup capacity.
pub fn prepare_batched_commit_inputs<F, const D: usize, P>(
    poly_groups: &[&[P]],
    point_group_sizes: &[usize],
    setup: &HachiProverSetup<F, D>,
) -> Result<PreparedBatchedCommitInputs, HachiError>
where
    F: FieldCore,
    P: HachiPolyOps<F, D>,
{
    if poly_groups.is_empty() {
        return Err(HachiError::InvalidInput(
            "batched_commit requires at least one commitment group".to_string(),
        ));
    }
    let total_groups = checked_total_groups(point_group_sizes, "batched_commit")?;
    if total_groups != poly_groups.len() {
        return Err(HachiError::InvalidInput(
            "batched_commit point group sizes do not match commitment groups".to_string(),
        ));
    }
    let num_vars = poly_groups[0]
        .first()
        .ok_or_else(|| {
            HachiError::InvalidInput(
                "batched_commit requires nonempty commitment groups".to_string(),
            )
        })?
        .num_vars();
    if num_vars > setup.expanded.seed.max_num_vars {
        return Err(HachiError::InvalidInput(format!(
            "batched_commit received polynomials with {} variables but setup supports at most {}",
            num_vars, setup.expanded.seed.max_num_vars
        )));
    }
    if point_group_sizes.len() > setup.expanded.seed.max_num_points {
        return Err(HachiError::InvalidInput(format!(
            "batched_commit received {} opening points but setup supports at most {}",
            point_group_sizes.len(),
            setup.expanded.seed.max_num_points
        )));
    }

    let mut claim_group_sizes = Vec::with_capacity(poly_groups.len());
    let mut total_claims = 0usize;
    for group in poly_groups {
        if group.is_empty() {
            return Err(HachiError::InvalidInput(
                "batched_commit requires nonempty commitment groups".to_string(),
            ));
        }
        if group.iter().any(|poly| poly.num_vars() != num_vars) {
            return Err(HachiError::InvalidInput(
                "batched_commit requires all polynomials to have the same num_vars".to_string(),
            ));
        }
        let group_claims = group.len();
        claim_group_sizes.push(group_claims);
        total_claims = total_claims.checked_add(group_claims).ok_or_else(|| {
            HachiError::InvalidInput("batched_commit total claim count overflow".to_string())
        })?;
    }
    if total_claims > setup.expanded.seed.max_num_batched_polys {
        return Err(HachiError::InvalidInput(format!(
            "batched_commit received {total_claims} polynomials but setup supports at most {}",
            setup.expanded.seed.max_num_batched_polys
        )));
    }

    Ok(PreparedBatchedCommitInputs {
        num_vars,
        total_claims,
        claim_group_sizes,
        point_count: point_group_sizes.len(),
    })
}

/// Commit a group of polynomials using already-selected level parameters.
///
/// Root config/schedule policy chooses `params`; this function owns only the
/// prover-side matrix work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if an inner witness commitment or hint allocation fails.
pub fn commit_with_params<F, const D: usize, P>(
    polys: &[P],
    setup: &HachiProverSetup<F, D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, HachiCommitmentHint<F, D>), HachiError>
where
    F: FieldCore + CanonicalField,
    P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let t_hat_flat_len_per_poly =
        params.num_blocks * params.a_key.row_len() * params.num_digits_open;
    let mut t_hat_flat = vec![[0i8; D]; polys.len() * t_hat_flat_len_per_poly];
    let mut t_hat_vec: Vec<FlatDigitBlocks<D>> = (0..polys.len())
        .map(|_| FlatDigitBlocks::new(Vec::new(), Vec::new()))
        .collect::<Result<_, _>>()?;
    let mut t_vec: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = vec![Vec::new(); polys.len()];
    cfg_chunks_mut!(t_hat_flat, t_hat_flat_len_per_poly)
        .zip(cfg_iter!(polys))
        .zip(cfg_iter_mut!(t_hat_vec))
        .zip(cfg_iter_mut!(t_vec))
        .try_for_each(|(((dst, poly), t_hat), t)| -> Result<(), HachiError> {
            let inner = poly.commit_inner_witness(
                &setup.expanded.shared_matrix,
                &setup.ntt_shared,
                params.a_key.row_len(),
                params.block_len,
                params.num_digits_commit,
                params.num_digits_open,
                params.log_basis,
                setup.expanded.seed.max_stride,
            )?;
            dst.copy_from_slice(inner.t_hat.flat_digits());
            *t_hat = inner.t_hat;
            *t = inner.t;
            Ok(())
        })?;
    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
        &setup.ntt_shared,
        params.b_key.row_len(),
        setup.expanded.seed.max_stride,
        &t_hat_flat,
    );
    Ok((
        RingCommitment { u },
        HachiCommitmentHint::with_t(t_hat_vec, t_vec),
    ))
}

/// Commit a group of polynomials using caller-supplied config policy.
///
/// The prover crate owns config-free input validation and commitment execution;
/// the caller supplies only the layout-selection policy.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or commitment
/// execution fails.
pub fn commit_with_policy<F, const D: usize, P, SelectParams>(
    polys: &[P],
    setup: &HachiProverSetup<F, D>,
    select_params: SelectParams,
) -> Result<(RingCommitment<F, D>, HachiCommitmentHint<F, D>), HachiError>
where
    F: FieldCore + CanonicalField,
    P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    SelectParams: FnOnce(usize, usize) -> Result<LevelParams, HachiError>,
{
    let prepared = prepare_commit_inputs::<F, D, P>(polys, setup)?;
    let params = select_params(prepared.num_vars, prepared.num_polys)?;
    commit_with_params::<F, D, P>(polys, setup, &params)
}

/// Commit multiple polynomial groups with one already-selected root layout.
///
/// Root config/schedule policy chooses `params`; this function owns the
/// repeated prover-side commitment work for each supplied group.
///
/// # Errors
///
/// Returns an error if any group commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_params<F, const D: usize, P>(
    poly_groups: &[&[P]],
    setup: &HachiProverSetup<F, D>,
    params: &LevelParams,
) -> Result<(Vec<RingCommitment<F, D>>, Vec<HachiCommitmentHint<F, D>>), HachiError>
where
    F: FieldCore + CanonicalField,
    P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let mut commitments = Vec::with_capacity(poly_groups.len());
    let mut hints = Vec::with_capacity(poly_groups.len());
    for group in poly_groups {
        let (commitment, hint) = commit_with_params::<F, D, P>(group, setup, params)?;
        commitments.push(commitment);
        hints.push(hint);
    }
    Ok((commitments, hints))
}

/// Commit multiple polynomial groups using caller-supplied schedule policy.
///
/// The prover crate owns grouped input validation, schedule-shape interpretation
/// for commitment layout selection, and repeated commitment execution. The
/// caller supplies concrete schedule/direct layout policy.
///
/// # Errors
///
/// Returns an error if input validation, schedule/direct parameter selection,
/// or any group commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_policy<F, const D: usize, P, SelectSchedule, DirectParams>(
    poly_groups: &[&[P]],
    point_group_sizes: &[usize],
    setup: &HachiProverSetup<F, D>,
    select_schedule: SelectSchedule,
    direct_params: DirectParams,
) -> Result<(Vec<RingCommitment<F, D>>, Vec<HachiCommitmentHint<F, D>>), HachiError>
where
    F: FieldCore + CanonicalField,
    P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    SelectSchedule:
        FnOnce(usize, usize, usize, HachiRootBatchSummary) -> Result<Schedule, HachiError>,
    DirectParams: FnOnce(usize, usize) -> Result<LevelParams, HachiError>,
{
    let prepared = prepare_batched_commit_inputs::<F, D, P>(poly_groups, point_group_sizes, setup)?;
    let batch_summary = HachiRootBatchSummary::from_claim_group_sizes(
        &prepared.claim_group_sizes,
        prepared.point_count,
    )?;
    let schedule = select_schedule(
        setup.expanded.seed.max_num_vars,
        prepared.num_vars,
        prepared.total_claims,
        batch_summary,
    )?;
    let params = match schedule.steps.first() {
        Some(Step::Fold(root_step)) => root_step.params.clone(),
        Some(Step::Direct(_)) => direct_params(prepared.num_vars, prepared.total_claims)?,
        None => {
            return Err(HachiError::InvalidSetup(
                "batched_commit schedule is empty".to_string(),
            ));
        }
    };

    batched_commit_with_params::<F, D, P>(poly_groups, setup, &params)
}
