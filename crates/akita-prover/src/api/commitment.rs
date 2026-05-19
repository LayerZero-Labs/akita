//! Prover-owned commitment kernels.

use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::mat_vec_mul_ntt_single_i8;
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use crate::{AkitaPolyOps, AkitaProverSetup};
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_types::{
    checked_total_groups, AkitaCommitmentHint, ClaimIncidenceSummary, FlatDigitBlocks, LevelParams,
    RingCommitment,
};

/// Validate a singleton commitment request against prover setup capacity.
///
/// # Errors
///
/// Returns an error if the request is empty, mixes polynomial dimensions, or
/// exceeds the prover setup capacity.
pub fn prepare_commit_inputs<F, const D: usize, P>(
    polys: &[P],
    setup: &AkitaProverSetup<F, D>,
) -> Result<ClaimIncidenceSummary, AkitaError>
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

    ClaimIncidenceSummary::same_point(num_vars, polys.len())
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
) -> Result<ClaimIncidenceSummary, AkitaError>
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

    let mut group_poly_counts = Vec::with_capacity(poly_groups.len());
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
        group_poly_counts.push(group_claims);
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

    ClaimIncidenceSummary::from_point_group_counts(
        num_vars,
        group_poly_counts,
        point_group_sizes.to_vec(),
    )
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
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let b_input_len_per_poly = params.num_blocks * params.a_key.row_len() * params.num_digits_open;
    let mut b_input_digits = vec![[0i8; D]; polys.len() * b_input_len_per_poly];
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
                dst.copy_from_slice(inner.decomposed_inner_rows.flat_digits());
                *decomposed = inner.decomposed_inner_rows;
                *recomposed = inner.recomposed_inner_rows;
                Ok(())
            },
        )?;
    #[cfg(feature = "zk")]
    let b_blinding_digits = {
        let b_blinding_digits =
            sample_blinding_digits::<F, D>(params.b_key.row_len(), params.log_basis)?;
        b_input_digits.extend_from_slice(b_blinding_digits.flat_digits());
        b_blinding_digits
    };
    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
        &setup.ntt_shared,
        params.b_key.row_len(),
        setup.expanded.seed.max_stride,
        &b_input_digits,
    );
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
        decomposed_inner_rows,
        recomposed_inner_rows,
        #[cfg(feature = "zk")]
        vec![b_blinding_digits],
    );
    Ok((RingCommitment { u }, hint))
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
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    SelectParams: FnOnce(&ClaimIncidenceSummary) -> Result<LevelParams, AkitaError>,
{
    let incidence = prepare_commit_inputs::<F, D, P>(polys, setup)?;
    let params = select_params(&incidence)?;
    commit_with_params::<F, D, P>(polys, setup, &params)
}

/// Commit multiple polynomial groups with one already-selected root layout.
///
/// Config/schedule policy chooses `params`; this function owns the
/// repeated prover-side commitment work for each supplied group.
///
/// # Errors
///
/// Returns an error if any group commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_params<F, const D: usize, P>(
    poly_groups: &[&[P]],
    setup: &AkitaProverSetup<F, D>,
    params: &LevelParams,
) -> Result<(Vec<RingCommitment<F, D>>, Vec<AkitaCommitmentHint<F, D>>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
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
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    SelectParams: FnOnce(&ClaimIncidenceSummary) -> Result<LevelParams, AkitaError>,
{
    let incidence =
        prepare_batched_commit_inputs::<F, D, P>(poly_groups, point_group_sizes, setup)?;
    let params = select_params(&incidence)?;

    batched_commit_with_params::<F, D, P>(poly_groups, setup, &params)
}
