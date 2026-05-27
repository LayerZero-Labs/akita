//! Prover-owned commitment kernels.

use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::mat_vec_mul_ntt_single_i8;
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
use crate::{AkitaPolyOps, AkitaProverSetup};
use akita_algebra::CyclotomicRing;
use akita_config::CommitmentConfig;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, RandomSampling};
use akita_types::{
    AkitaCommitmentHint, ClaimIncidenceSummary, FlatDigitBlocks, LevelParams, RingCommitment,
    SetupRoleDimensions,
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

/// Commit a group of polynomials using already-selected level parameters.
///
/// Config/schedule policy chooses `params`; this function owns only the
/// prover-side matrix work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if an inner witness commitment or hint allocation fails.
fn commit_group_with_setup_widths<F, const D: usize, P>(
    #[cfg_attr(not(feature = "zk"), allow(unused_variables))] point_idx: usize,
    polys: &[P],
    setup: &AkitaProverSetup<F, D>,
    params: &LevelParams,
    a_setup_width: usize,
    b_setup_width: usize,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + RandomSampling,
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
                    a_setup_width,
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
        b_blinding_digits
    };
    #[cfg_attr(not(feature = "zk"), allow(unused_mut))]
    let mut u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
        &setup.ntt_shared,
        params.b_key.row_len(),
        b_setup_width,
        &b_input_digits,
    );
    #[cfg(feature = "zk")]
    {
        let blinding_rows = akita_types::zk::b_blinding_negacyclic_rows::<F, D>(
            &setup.expanded.seed.zk_blinding_seed,
            point_idx,
            params.b_key.row_len(),
            b_blinding_digits.flat_digits(),
        );
        for (row, blinding_row) in u.iter_mut().zip(blinding_rows) {
            *row += blinding_row;
        }
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
/// Returns an error if an inner witness commitment or hint allocation fails.
pub fn commit_with_params<F, const D: usize, P>(
    polys: &[P],
    setup: &AkitaProverSetup<F, D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    if polys.is_empty() {
        return Err(AkitaError::InvalidInput(
            "commit requires at least one polynomial".to_string(),
        ));
    }
    let role_dimensions =
        SetupRoleDimensions::for_batched_shape(params, &[polys.len()], polys.len())?;
    let b_setup_width = role_dimensions.b_setup_width;
    commit_group_with_setup_widths(
        0,
        polys,
        setup,
        params,
        role_dimensions.a_setup_width,
        b_setup_width,
    )
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
    setup: &AkitaProverSetup<F, D>,
) -> Result<ClaimIncidenceSummary, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    if polys_per_point.is_empty() {
        return Err(AkitaError::InvalidInput(
            "batched_commit requires at least one opening point".to_string(),
        ));
    }
    if polys_per_point.len() > setup.expanded.seed.max_num_points {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received {} opening points but setup supports at most {}",
            polys_per_point.len(),
            setup.expanded.seed.max_num_points
        )));
    }
    let first_bundle = polys_per_point.first().ok_or_else(|| {
        AkitaError::InvalidInput("batched_commit requires at least one opening point".to_string())
    })?;
    let first_poly = first_bundle.first().ok_or_else(|| {
        AkitaError::InvalidInput("batched_commit bundles must be nonempty".to_string())
    })?;
    let num_vars = first_poly.num_vars();
    if num_vars > setup.expanded.seed.max_num_vars {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received a polynomial with {} variables but setup supports at most {}",
            num_vars, setup.expanded.seed.max_num_vars
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
        if bundle.iter().any(|p| p.num_vars() != num_vars) {
            return Err(AkitaError::InvalidInput(
                "batched_commit requires every polynomial to share num_vars".to_string(),
            ));
        }
        num_polys_per_point.push(bundle.len());
        total_polys = total_polys.checked_add(bundle.len()).ok_or_else(|| {
            AkitaError::InvalidInput("batched_commit total polynomial count overflow".to_string())
        })?;
    }
    if total_polys > setup.expanded.seed.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received {total_polys} polynomials but setup supports at most {}",
            setup.expanded.seed.max_num_batched_polys
        )));
    }

    ClaimIncidenceSummary::from_point_polys(num_vars, num_polys_per_point)
}

/// Commit one polynomial bundle per opening point using already-selected
/// level parameters.
///
/// The caller has already resolved the shared root commitment layout (e.g.
/// via [`crate::batched_commit`]); this function owns only the prover-
/// side matrix work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if any per-point commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_params<F, const D: usize, P>(
    polys_per_point: &[&[P]],
    setup: &AkitaProverSetup<F, D>,
    params: &LevelParams,
) -> Result<Vec<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>)>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    if polys_per_point.is_empty() || polys_per_point.iter().any(|polys| polys.is_empty()) {
        return Err(AkitaError::InvalidInput(
            "batched commit requires at least one polynomial per point".to_string(),
        ));
    }
    let num_polys_per_point: Vec<usize> = polys_per_point.iter().map(|polys| polys.len()).collect();
    let num_claims = num_polys_per_point.iter().try_fold(0usize, |acc, &count| {
        acc.checked_add(count).ok_or_else(|| {
            AkitaError::InvalidInput("batched commit polynomial count overflow".to_string())
        })
    })?;
    let role_dimensions =
        SetupRoleDimensions::for_batched_shape(params, &num_polys_per_point, num_claims)?;
    let b_setup_width = role_dimensions.b_setup_width;
    let mut out = Vec::with_capacity(polys_per_point.len());
    for (point_idx, polys) in polys_per_point.iter().enumerate() {
        out.push(commit_group_with_setup_widths::<F, D, P>(
            point_idx,
            polys,
            setup,
            params,
            role_dimensions.a_setup_width,
            b_setup_width,
        )?);
    }
    Ok(out)
}

/// Commit a group of polynomials.
///
/// Routes through `Cfg::get_params_for_batched_commitment` so all per-config
/// layout decisions land in the trait body.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or commitment
/// execution fails.
pub fn commit<F, Cfg, P, const D: usize>(
    polys: &[P],
    setup: &AkitaProverSetup<F, D>,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + RandomSampling,
    Cfg: CommitmentConfig<Field = F>,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let incidence = prepare_commit_inputs::<F, D, P>(polys, setup)?;
    let params = Cfg::get_params_for_batched_commitment(&incidence)?;
    commit_with_params::<F, D, P>(polys, setup, &params)
}

/// Commit one polynomial bundle per opening point.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or any
/// per-point commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit<F, Cfg, P, const D: usize>(
    polys_per_point: &[&[P]],
    setup: &AkitaProverSetup<F, D>,
) -> Result<Vec<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>)>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + RandomSampling,
    Cfg: CommitmentConfig<Field = F>,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let incidence = prepare_batched_commit_inputs::<F, D, P>(polys_per_point, setup)?;
    let params = Cfg::get_params_for_batched_commitment(&incidence)?;
    batched_commit_with_params::<F, D, P>(polys_per_point, setup, &params)
}
