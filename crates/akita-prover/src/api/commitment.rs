//! Prover-owned commitment kernels.

use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::mat_vec_mul_ntt_single_i8;
use crate::{AkitaPolyOps, AkitaProverSetup, DensePoly};
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::{
    checked_total_claims, checked_total_groups, AkitaCommitmentHint, AkitaRootBatchSummary,
    AkitaVerifierSetup, DirectWitnessProof, FlatDigitBlocks, LevelParams, MultiPointBatchShape,
    RingCommitment,
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
    setup: &AkitaProverSetup<F, D>,
) -> Result<PreparedCommitInputs, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
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
    if polys.len() > setup.expanded.seed.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
            "commit received {} polynomials but setup supports at most {}",
            polys.len(),
            setup.expanded.seed.max_num_batched_polys
        )));
    }
    if num_vars > setup.expanded.seed.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
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
    setup: &AkitaProverSetup<F, D>,
) -> Result<PreparedBatchedCommitInputs, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    if poly_groups.is_empty() {
        return Err(AkitaError::InvalidInput(
            "batched_commit requires at least one commitment group".to_string(),
        ));
    }
    let total_groups = checked_total_groups(point_group_sizes, "batched_commit")?;
    if total_groups != poly_groups.len() {
        return Err(AkitaError::InvalidInput(
            "batched_commit point group sizes do not match commitment groups".to_string(),
        ));
    }
    let num_vars = poly_groups[0]
        .first()
        .ok_or_else(|| {
            AkitaError::InvalidInput(
                "batched_commit requires nonempty commitment groups".to_string(),
            )
        })?
        .num_vars();
    if num_vars > setup.expanded.seed.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received polynomials with {} variables but setup supports at most {}",
            num_vars, setup.expanded.seed.max_num_vars
        )));
    }
    if point_group_sizes.len() > setup.expanded.seed.max_num_points {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received {} opening points but setup supports at most {}",
            point_group_sizes.len(),
            setup.expanded.seed.max_num_points
        )));
    }

    let mut claim_group_sizes = Vec::with_capacity(poly_groups.len());
    let mut total_claims = 0usize;
    for group in poly_groups {
        if group.is_empty() {
            return Err(AkitaError::InvalidInput(
                "batched_commit requires nonempty commitment groups".to_string(),
            ));
        }
        if group.iter().any(|poly| poly.num_vars() != num_vars) {
            return Err(AkitaError::InvalidInput(
                "batched_commit requires all polynomials to have the same num_vars".to_string(),
            ));
        }
        let group_claims = group.len();
        claim_group_sizes.push(group_claims);
        total_claims = total_claims.checked_add(group_claims).ok_or_else(|| {
            AkitaError::InvalidInput("batched_commit total claim count overflow".to_string())
        })?;
    }
    if total_claims > setup.expanded.seed.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
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
/// Config/schedule policy chooses `params`; this function owns only the
/// prover-side matrix work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if an inner witness commitment or hint allocation fails.
pub fn commit_with_params<F, const D: usize, P>(
    polys: &[P],
    setup: &AkitaProverSetup<F, D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
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
        .try_for_each(|(((dst, poly), t_hat), t)| -> Result<(), AkitaError> {
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
        AkitaCommitmentHint::with_t(t_hat_vec, t_vec),
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
    setup: &AkitaProverSetup<F, D>,
    select_params: SelectParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    SelectParams: FnOnce(usize, usize) -> Result<LevelParams, AkitaError>,
{
    let prepared = prepare_commit_inputs::<F, D, P>(polys, setup)?;
    let params = select_params(prepared.num_vars, prepared.num_polys)?;
    commit_with_params::<F, D, P>(polys, setup, &params)
}

/// Commit multiple polynomial groups with one already-selected root layout.
///
/// Config/schedule policy chooses `params`; this function owns the
/// repeated prover-side commitment work for each supplied group.
///
/// When `params.groups.is_some()`, each commitment group uses its own
/// per-group `(m, r, B, digit_count)` from `params.groups[i]` while
/// sharing `params`'s outer `D, A`, ring dimension, log_basis, and
/// challenge config. This is the book §5.3 "split commitment" shape
/// (multi-group batched Hachi). When `params.groups.is_none()`, every
/// group inherits the outer LP's `(m, r, B, digit_count)` — bit-equivalent
/// to today's single-LP shape.
///
/// # Errors
///
/// Returns an error if any group commitment fails, or if
/// `params.groups.is_some()` and the per-group spec count disagrees with
/// `poly_groups.len()`.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_params<F, const D: usize, P>(
    poly_groups: &[&[P]],
    setup: &AkitaProverSetup<F, D>,
    params: &LevelParams,
) -> Result<(Vec<RingCommitment<F, D>>, Vec<AkitaCommitmentHint<F, D>>), AkitaError>
where
    F: FieldCore + CanonicalField,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let specs = params.group_specs(poly_groups.len())?;
    let mut commitments = Vec::with_capacity(poly_groups.len());
    let mut hints = Vec::with_capacity(poly_groups.len());
    for (group, spec) in poly_groups.iter().zip(specs.iter()) {
        let per_group_lp = spec.lower_into_outer(params);
        let (commitment, hint) = commit_with_params::<F, D, P>(group, setup, &per_group_lp)?;
        commitments.push(commitment);
        hints.push(hint);
    }
    Ok((commitments, hints))
}

/// Commit multiple polynomial groups using caller-supplied commitment policy.
///
/// The prover crate owns grouped input validation and repeated commitment
/// execution. The caller supplies only commitment layout policy; proving
/// schedule selection stays out of the commitment path.
///
/// # Errors
///
/// Returns an error if input validation, commitment parameter selection, or
/// any group commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_policy<F, const D: usize, P, SelectParams>(
    poly_groups: &[&[P]],
    point_group_sizes: &[usize],
    setup: &AkitaProverSetup<F, D>,
    select_params: SelectParams,
) -> Result<(Vec<RingCommitment<F, D>>, Vec<AkitaCommitmentHint<F, D>>), AkitaError>
where
    F: FieldCore + CanonicalField,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    SelectParams: FnOnce(usize, usize, AkitaRootBatchSummary) -> Result<LevelParams, AkitaError>,
{
    let prepared = prepare_batched_commit_inputs::<F, D, P>(poly_groups, point_group_sizes, setup)?;
    let batch_summary = AkitaRootBatchSummary::from_claim_group_sizes(
        &prepared.claim_group_sizes,
        prepared.point_count,
    )?;
    let params = select_params(
        setup.expanded.seed.max_num_vars,
        prepared.num_vars,
        batch_summary,
    )?;

    batched_commit_with_params::<F, D, P>(poly_groups, setup, &params)
}

/// Recompute root-direct commitments from direct witnesses and compare them to
/// the proof commitments.
///
/// This is a preservation helper for the current root-direct verifier path. It
/// intentionally lives with prover commitment machinery until root-direct
/// verification is redesigned around a lighter verifier-side contract.
///
/// # Errors
///
/// Returns an error if the direct witness shape does not match the batch shape,
/// if witness reconstruction fails, or if any recomputed commitment differs
/// from the proof commitment.
pub fn verify_root_direct_commitments_with_params<F, const D: usize>(
    witnesses: &[DirectWitnessProof<F>],
    setup: &AkitaVerifierSetup<F>,
    flat_commitments: &[RingCommitment<F, D>],
    batch_shape: &MultiPointBatchShape,
    params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if flat_commitments.len() != batch_shape.claim_group_sizes.len() {
        return Err(AkitaError::InvalidProof);
    }
    let total_groups = checked_total_groups(
        &batch_shape.point_group_sizes,
        "root_direct_commitment_check",
    )?;
    if total_groups != batch_shape.claim_group_sizes.len() {
        return Err(AkitaError::InvalidProof);
    }
    let total_claims = checked_total_claims(
        &batch_shape.claim_group_sizes,
        "root_direct_commitment_check",
    )?;
    if total_claims != witnesses.len() {
        return Err(AkitaError::InvalidProof);
    }

    let total = setup.expanded.shared_matrix.total_ring_elements_at::<D>();
    let verifier_ntt = crate::kernels::crt_ntt::build_ntt_slot(
        setup.expanded.shared_matrix.ring_view::<D>(1, total),
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    let temp_setup = AkitaProverSetup {
        expanded: setup.expanded.clone(),
        ntt_shared: verifier_ntt,
        tiered_s_cache: std::sync::Arc::new(std::sync::OnceLock::new()),
    };

    let mut claim_offset = 0usize;
    let mut poly_groups = Vec::with_capacity(batch_shape.claim_group_sizes.len());
    for &group_size in &batch_shape.claim_group_sizes {
        let group_witnesses = &witnesses[claim_offset..claim_offset + group_size];
        let group_polys = group_witnesses
            .iter()
            .map(|witness| {
                let field_witness = witness
                    .as_field_elements()
                    .ok_or(AkitaError::InvalidProof)?
                    .coeffs();
                let coeff_len = field_witness.len();
                if !coeff_len.is_power_of_two() {
                    return Err(AkitaError::InvalidProof);
                }
                let num_vars = coeff_len.trailing_zeros() as usize;
                DensePoly::<F, D>::from_field_evals(num_vars, field_witness)
                    .map_err(|_| AkitaError::InvalidProof)
            })
            .collect::<Result<Vec<_>, _>>()?;

        poly_groups.push(group_polys);
        claim_offset += group_size;
    }
    let poly_group_refs = poly_groups
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<&[DensePoly<F, D>]>>();

    let mut expected_commitments = Vec::with_capacity(poly_group_refs.len());
    for group in poly_group_refs {
        let (commitment, _) =
            commit_with_params::<F, D, DensePoly<F, D>>(group, &temp_setup, params)
                .map_err(|_| AkitaError::InvalidProof)?;
        expected_commitments.push(commitment);
    }

    if expected_commitments != flat_commitments {
        return Err(AkitaError::InvalidProof);
    }

    Ok(())
}
