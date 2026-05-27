//! Prover-owned commitment kernels.

#[cfg(not(feature = "zk"))]
use crate::kernels::crt_ntt::build_ntt_slot;
#[cfg(not(feature = "zk"))]
use crate::kernels::linear::mat_vec_mul_ntt_single_i8;
#[cfg(not(feature = "zk"))]
use crate::kernels::matrix::derive_tier1_f_matrix_flat;
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
#[cfg(not(feature = "zk"))]
use crate::protocol::tiered_commit::{tiered_commit, TieredCommitParams};
use crate::{AkitaPolyOps, CommitInnerWitness, CommitmentComputeBackend};
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_types::{
    AkitaCommitmentHint, AkitaExpandedSetup, ClaimIncidenceSummary, FlatDigitBlocks, LevelParams,
    RingCommitment,
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
    if !(1..=128).contains(&log_basis) {
        return Err(AkitaError::InvalidSetup(
            "log_basis must be in 1..=128 when recomposing inner commitment digits".to_string(),
        ));
    }

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
    if setup.seed.max_stride == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup max_stride must be nonzero".to_string(),
        ));
    }
    if !(1..=128).contains(&params.log_basis) {
        return Err(AkitaError::InvalidSetup(
            "commit params log_basis must be in 1..=128".to_string(),
        ));
    }
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
    if params.b_key.col_len() == 0 || params.d_key.col_len() == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params require nonzero B and D widths, got B={} D={}",
            params.b_key.col_len(),
            params.d_key.col_len()
        )));
    }
    if params.a_key.col_len() > setup.seed.max_stride
        || params.b_key.col_len() > setup.seed.max_stride
        || params.d_key.col_len() > setup.seed.max_stride
    {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params column widths A={} B={} D={} exceed setup max_stride {}",
            params.a_key.col_len(),
            params.b_key.col_len(),
            params.d_key.col_len(),
            setup.seed.max_stride
        )));
    }
    let max_rows = setup.shared_matrix.total_ring_elements_at::<D>()? / setup.seed.max_stride;
    if params.a_key.row_len() > max_rows
        || params.b_key.row_len() > max_rows
        || params.d_key.row_len() > max_rows
    {
        return Err(AkitaError::InvalidSetup(format!(
            "commit params row counts A={} B={} D={} exceed setup max_rows {max_rows}",
            params.a_key.row_len(),
            params.b_key.row_len(),
            params.d_key.row_len()
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
pub fn prepare_commit_inputs<F, const D: usize, P>(
    polys: &[P],
    setup: &AkitaExpandedSetup<F>,
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

fn commit_with_validated_params<F, const D: usize, P, B>(
    polys: &[P],
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
{
    if params.is_tiered_root() {
        return commit_tiered_with_params::<F, D, P, B>(polys, backend, prepared, params);
    }
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
                let inner = poly.commit_inner_witness(
                    backend,
                    prepared,
                    params.a_key.row_len(),
                    params.block_len,
                    params.num_digits_commit,
                    params.num_digits_open,
                    params.log_basis,
                )?;
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
    let b_blinding_digits = {
        let b_blinding_digits =
            sample_blinding_digits::<F, D>(params.b_key.row_len(), params.log_basis)?;
        b_input_digits.extend_from_slice(b_blinding_digits.flat_digits());
        b_blinding_digits
    };
    validate_commit_outer_input_nonempty(b_input_digits.len())?;
    let u: Vec<CyclotomicRing<F, D>> =
        backend.digit_rows::<D>(prepared, params.b_key.row_len(), &b_input_digits)?;
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

/// Tiered (two-tier) commit per `specs/tiered_commit.md`.
///
/// Splits the flat `t̂` into `params.split_factor` equal contiguous
/// chunks against the same `B'`. For each chunk computes
/// `u_i = B' · t̂_i`, gadget-decomposes `u_i → û_i` with depth
/// `δ_outer` and basis `2^{outer_log_basis}`, and finally multiplies
/// the concatenated `û_concat = û_1 ‖ … ‖ û_f` against the F matrix
/// (derived deterministically from the setup seed under the
/// `b"tier1-f"` domain label) to produce `u_final = F · û_concat`.
///
/// The public commitment is `RingCommitment { u: u_final }`
/// (`u.len() == params.outer_commitment_rows() == n_F`). The hint
/// carries `û_concat` as a single per-point entry in `outer_digits`,
/// alongside the existing per-poly inner-witness data.
///
/// # Errors
///
/// Returns an error when `params.split_factor < 2`, when any other
/// tiered-commit shape validation fails, or when an inner commitment
/// fails.
pub fn commit_tiered_with_params<F, const D: usize, P, B>(
    polys: &[P],
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
{
    // Spec §7: tiering under `--features zk` is intentionally out of
    // scope for the first landing.
    #[cfg(feature = "zk")]
    {
        let _ = (polys, backend, prepared, params);
        Err(AkitaError::InvalidSetup(
            "tiered commit path is not enabled under `--features zk` in this revision; \
             see specs/tiered_commit.md §7"
                .to_string(),
        ))
    }
    #[cfg(not(feature = "zk"))]
    {
        commit_tiered_with_params_inner::<F, D, P, B>(polys, backend, prepared, params)
    }
}

#[cfg(not(feature = "zk"))]
fn commit_tiered_with_params_inner<F, const D: usize, P, B>(
    polys: &[P],
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
{
    if !params.is_tiered_root() {
        return Err(AkitaError::InvalidInput(
            "commit_tiered_with_params requires params.is_tiered_root() (split_factor > 1)"
                .to_string(),
        ));
    }
    let split_factor = params.split_factor;
    let n_b_prime = params.b_key.row_len();
    let n_f = params.f_key.row_len();
    let outer_log_basis = params.outer_log_basis;
    let num_digits_outer = params.num_digits_outer;
    let expected_f_width = n_b_prime
        .checked_mul(split_factor)
        .and_then(|w| w.checked_mul(num_digits_outer))
        .ok_or_else(|| AkitaError::InvalidSetup("tiered F width overflow".to_string()))?;
    if params.f_key.col_len() != expected_f_width {
        return Err(AkitaError::InvalidSetup(format!(
            "f_key.col_len() = {} does not match n_b' · split_factor · num_digits_outer = {}",
            params.f_key.col_len(),
            expected_f_width,
        )));
    }
    if n_f == 0 {
        return Err(AkitaError::InvalidSetup(
            "tiered commit requires f_key.row_len() ≥ 1".to_string(),
        ));
    }

    // Stage 1: build t̂ via the same per-poly inner-witness path the
    // legacy commit takes — through the compute backend.
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
                let inner = poly.commit_inner_witness(
                    backend,
                    prepared,
                    params.a_key.row_len(),
                    params.block_len,
                    params.num_digits_commit,
                    params.num_digits_open,
                    params.log_basis,
                )?;
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

    // Stage 2: feed t̂ into the tiered kernel. B' uses the backend's
    // `digit_rows`; F uses a locally-built NTT cache derived from the
    // public matrix seed.
    let expanded = backend.prepared_expanded_setup::<D>(prepared);
    let f_flat = derive_tier1_f_matrix_flat::<F, D>(
        n_f.checked_mul(expected_f_width).ok_or_else(|| {
            AkitaError::InvalidSetup("tiered F backing storage total overflow".to_string())
        })?,
        &expanded.seed.public_matrix_seed,
    );
    let f_ntt = build_ntt_slot(f_flat.ring_view::<D>(n_f, expected_f_width)?)?;
    let b_prime_multiply = |chunk: &[[i8; D]]| -> Vec<CyclotomicRing<F, D>> {
        backend
            .digit_rows::<D>(prepared, n_b_prime, chunk)
            .expect("tiered B' digit_rows failed")
    };
    let f_multiply = |uhat_concat: &[[i8; D]]| -> Vec<CyclotomicRing<F, D>> {
        mat_vec_mul_ntt_single_i8::<F, D>(&f_ntt, n_f, expected_f_width, uhat_concat)
    };

    let out = tiered_commit::<F, _, _, D>(
        TieredCommitParams {
            split_factor,
            n_b_prime,
            outer_log_basis,
            num_digits_outer,
        },
        &b_input_digits,
        b_prime_multiply,
        f_multiply,
    )?;

    // Stage 3: assemble the hint with `uhat_concat` as the single
    // per-opening-point entry.
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
        decomposed_inner_rows,
        recomposed_inner_rows,
    )
    .with_outer_digits(vec![out.uhat_concat]);

    Ok((RingCommitment { u: out.u_final }, hint))
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
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    prepare_commit_inputs::<F, D, P>(polys, expanded)?;
    validate_commit_level_params::<F, D>(params, expanded)?;
    commit_with_validated_params::<F, D, P, B>(polys, backend, prepared, params)
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
pub fn commit_with_policy<F, const D: usize, P, B, SelectParams>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    select_params: SelectParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
    SelectParams: FnOnce(&ClaimIncidenceSummary) -> Result<LevelParams, AkitaError>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    let incidence = prepare_commit_inputs::<F, D, P>(polys, expanded)?;
    let params = select_params(&incidence)?;
    validate_commit_level_params::<F, D>(&params, expanded)?;
    commit_with_validated_params::<F, D, P, B>(polys, backend, prepared, &params)
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
    P: AkitaPolyOps<F, D>,
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
    let num_vars = first_poly.num_vars();
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
    if total_polys > setup.seed.max_num_batched_polys {
        return Err(AkitaError::InvalidInput(format!(
            "batched_commit received {total_polys} polynomials but setup supports at most {}",
            setup.seed.max_num_batched_polys
        )));
    }

    ClaimIncidenceSummary::from_point_polys(num_vars, num_polys_per_point)
}

/// Commit one polynomial bundle per opening point using a caller-supplied
/// layout-selection policy.
///
/// The policy callback receives the full multipoint incidence and returns the
/// shared root commitment layout. Every per-point bundle is then committed
/// with that one layout via [`commit_with_params`], guaranteeing that the
/// produced commitments are compatible with the layout `batched_prove` will
/// select for the same incidence.
///
/// # Errors
///
/// Returns an error if input validation, parameter selection, or any per-
/// point commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_policy<F, const D: usize, P, B, SelectParams>(
    polys_per_point: &[&[P]],
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    select_params: SelectParams,
) -> Result<Vec<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>)>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
    SelectParams: FnOnce(&ClaimIncidenceSummary) -> Result<LevelParams, AkitaError>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    let incidence = prepare_batched_commit_inputs::<F, D, P>(polys_per_point, expanded)?;
    let params = select_params(&incidence)?;
    validate_commit_level_params::<F, D>(&params, expanded)?;
    batched_commit_with_validated_params::<F, D, P, B>(polys_per_point, backend, prepared, &params)
}

#[allow(clippy::type_complexity)]
fn batched_commit_with_validated_params<F, const D: usize, P, B>(
    polys_per_point: &[&[P]],
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<Vec<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>)>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
{
    let mut out = Vec::with_capacity(polys_per_point.len());
    for polys in polys_per_point {
        out.push(commit_with_validated_params::<F, D, P, B>(
            polys, backend, prepared, params,
        )?);
    }
    Ok(out)
}

/// Commit one polynomial bundle per opening point using already-selected
/// level parameters.
///
/// The caller has already resolved the shared root commitment layout (e.g.
/// via [`batched_commit_with_policy`]); this function owns only the prover-
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
    P: AkitaPolyOps<F, D>,
    B: CommitmentComputeBackend<F>,
{
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    prepare_batched_commit_inputs::<F, D, P>(polys_per_point, expanded)?;
    validate_commit_level_params::<F, D>(params, expanded)?;
    batched_commit_with_validated_params::<F, D, P, B>(polys_per_point, backend, prepared, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp64;

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

// Tier-3 commit-side tests removed during the origin/main merge: they
// targeted the pre-#105 `AkitaProverSetup`-based commit_with_params
// API. The production `fp128::D32OneHotFastVerify` preset retains
// end-to-end coverage through `crates/akita-pcs/examples/portable_bench.rs`.
