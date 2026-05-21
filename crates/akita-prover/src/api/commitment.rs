//! Prover-owned commitment kernels.

use crate::kernels::crt_ntt::{build_ntt_slot, NttSlotCache};
use crate::kernels::linear::mat_vec_mul_ntt_single_i8;
use crate::kernels::matrix::derive_tier1_f_matrix_flat;
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;
#[cfg(not(feature = "zk"))]
use crate::protocol::tiered_commit::{tiered_commit, TieredCommitParams};
use crate::{AkitaPolyOps, AkitaProverSetup};
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_types::{
    AkitaCommitmentHint, ClaimIncidenceSummary, FlatDigitBlocks, LevelParams, RingCommitment,
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
/// Dispatches on `params.is_tiered_root()`:
///   - **Tiered (`split_factor > 1`):** delegates to
///     [`commit_tiered_with_params`] with an F NTT cache derived
///     on-demand from the setup's `public_matrix_seed` via
///     [`derive_tier1_f_matrix_flat`]. Cache derivation is a one-shot
///     cost per call; future revisions may memoize per `(n_F, F_width)`
///     on the prover setup once that integration lands.
///   - **Legacy (`split_factor == 1`):** runs today's single-tier
///     `u = B · t̂` path verbatim.
///
/// # Errors
///
/// Returns an error if an inner witness commitment or hint allocation
/// fails, or — in the tiered branch — if the F NTT cache cannot be
/// built for the current field / ring-dimension pair.
pub fn commit_with_params<F, const D: usize, P>(
    polys: &[P],
    setup: &AkitaProverSetup<F, D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    if params.is_tiered_root() {
        // F lives in its own backing storage, derived deterministically
        // from the same public matrix seed with a distinct label.
        let f_width = params.f_key.col_len();
        let n_f = params.f_key.row_len();
        let total_f_ring_elements = n_f.checked_mul(f_width).ok_or_else(|| {
            AkitaError::InvalidSetup("tiered F backing storage total overflow".to_string())
        })?;
        let f_flat = derive_tier1_f_matrix_flat::<F, D>(
            total_f_ring_elements,
            &setup.expanded.seed.public_matrix_seed,
        );
        let f_ntt_cache = build_ntt_slot(f_flat.ring_view::<D>(n_f, f_width)?)?;
        return commit_tiered_with_params::<F, D, P>(polys, setup, params, &f_ntt_cache, f_width);
    }

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
    let hint = {
        #[cfg(feature = "zk")]
        {
            AkitaCommitmentHint::with_recomposed_inner_rows(
                decomposed_inner_rows,
                recomposed_inner_rows,
                vec![b_blinding_digits],
            )
        }
        #[cfg(not(feature = "zk"))]
        {
            AkitaCommitmentHint::with_recomposed_inner_rows(
                decomposed_inner_rows,
                recomposed_inner_rows,
            )
        }
    };
    Ok((RingCommitment { u }, hint))
}

/// Tiered (two-tier) commit per `specs/tiered_commit.md`.
///
/// Splits the flat `t̂` into `params.split_factor` equal contiguous
/// chunks against the **same** `B'` (a column-window view of the shared
/// matrix's B-row block; physically realised by passing the chunk-width
/// digit vector into the existing `mat_vec_mul_ntt_single_i8` kernel,
/// which honours a `vec_len < max_stride` input). For each chunk it
/// computes `u_i = B' · t̂_i`, gadget-decomposes `u_i → û_i` with depth
/// `δ_outer` and basis `2^{outer_log_basis}`, and finally multiplies the
/// concatenated `û_concat = û_1 ‖ … ‖ û_f` against the caller-supplied F
/// matrix to produce `u_final = F · û_concat`.
///
/// The public commitment is `RingCommitment { u: u_final }`
/// (`u.len() == lp.outer_commitment_rows() == n_F`). The hint carries
/// `û_concat` as a single per-point entry in `outer_digits`, alongside
/// the existing per-poly `decomposed_inner_rows` / `recomposed_inner_rows`.
///
/// `f_ntt_cache` is the precomputed NTT cache of the F matrix; its
/// physical row stride is `f_max_stride`. F's row count is
/// `params.f_key.row_len()` and its active column width is
/// `params.f_key.col_len() = n_b' · split_factor · num_digits_outer`.
/// (Once `AkitaProverSetup` is extended to embed F, the runtime call
/// site will pull both `f_ntt_cache` and `f_max_stride` from `setup`;
/// this signature lets unit tests inject a synthetic F.)
///
/// # Errors
///
/// Returns an error when `params.split_factor < 2`, when any other
/// tiered-commit shape validation fails (see
/// `crate::protocol::tiered_commit::TieredCommitParams::validate`),
/// or when an inner commitment fails.
pub fn commit_tiered_with_params<F, const D: usize, P>(
    polys: &[P],
    setup: &AkitaProverSetup<F, D>,
    params: &LevelParams,
    f_ntt_cache: &NttSlotCache<D>,
    f_max_stride: usize,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    // Spec §7: tiering under `--features zk` is intentionally out
    // of scope for the first landing — the planner must not emit
    // `split_factor > 1` candidates when `zk` is on until a follow-
    // up resolves the ûhat blinding question. We reject loudly at
    // the function boundary (rather than partway through the body)
    // so the rest of the function is unambiguously dead code under
    // `--features zk` and clippy can stop warning about
    // unreachable variable bindings.
    #[cfg(feature = "zk")]
    {
        let _ = (polys, setup, params, f_ntt_cache, f_max_stride);
        Err(AkitaError::InvalidSetup(
            "tiered commit path is not enabled under `--features zk` in this revision; \
             see specs/tiered_commit.md §7"
                .to_string(),
        ))
    }
    #[cfg(not(feature = "zk"))]
    {
        commit_tiered_with_params_inner(polys, setup, params, f_ntt_cache, f_max_stride)
    }
}

#[cfg(not(feature = "zk"))]
fn commit_tiered_with_params_inner<F, const D: usize, P>(
    polys: &[P],
    setup: &AkitaProverSetup<F, D>,
    params: &LevelParams,
    f_ntt_cache: &NttSlotCache<D>,
    f_max_stride: usize,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    if !params.is_tiered_root() {
        return Err(AkitaError::InvalidInput(
            "commit_tiered_with_params requires params.is_tiered_root() (split_factor > 1); \
             use commit_with_params for the legacy path"
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

    // Stage 1: build t̂ exactly as the legacy path. We deliberately
    // reuse the legacy preamble verbatim so any later refactor of the
    // inner-witness commit kernel benefits both paths.
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
    // Stage 2: feed t̂ into the closure-based tiered kernel. The B' arm
    // calls the existing NTT kernel on the shared matrix with a chunk-
    // length input; the F arm uses the caller-supplied F NTT cache.
    let n_b_prime_local = n_b_prime;
    let setup_ntt = &setup.ntt_shared;
    let setup_max_stride = setup.expanded.seed.max_stride;
    let b_prime_multiply = |chunk: &[[i8; D]]| -> Vec<CyclotomicRing<F, D>> {
        // The kernel reads each B row's first `chunk.len()` cells (i.e.
        // B'), since `vec_len = vec.len().min(inner_width)`. Callers
        // that pass a `chunk.len() > setup_max_stride` would silently
        // drop tail cells; tiered_commit's own length validation rejects
        // that shape upstream.
        mat_vec_mul_ntt_single_i8(setup_ntt, n_b_prime_local, setup_max_stride, chunk)
    };
    let f_multiply = |uhat_concat: &[[i8; D]]| -> Vec<CyclotomicRing<F, D>> {
        mat_vec_mul_ntt_single_i8(f_ntt_cache, n_f, f_max_stride, uhat_concat)
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

    // Stage 3: assemble the hint. `outer_digits` carries `uhat_concat`
    // as one entry (this is a per-opening-point hint; the batched commit
    // surface above this function flattens per-point entries).
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(
        decomposed_inner_rows,
        recomposed_inner_rows,
    )
    .with_outer_digits(vec![out.uhat_concat]);

    Ok((RingCommitment { u: out.u_final }, hint))
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
pub fn batched_commit_with_policy<F, const D: usize, P, SelectParams>(
    polys_per_point: &[&[P]],
    setup: &AkitaProverSetup<F, D>,
    select_params: SelectParams,
) -> Result<Vec<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>)>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    SelectParams: FnOnce(&ClaimIncidenceSummary) -> Result<LevelParams, AkitaError>,
{
    let incidence = prepare_batched_commit_inputs::<F, D, P>(polys_per_point, setup)?;
    let params = select_params(&incidence)?;
    batched_commit_with_params::<F, D, P>(polys_per_point, setup, &params)
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
/// Returns an error if any per-point commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_params<F, const D: usize, P>(
    polys_per_point: &[&[P]],
    setup: &AkitaProverSetup<F, D>,
    params: &LevelParams,
) -> Result<Vec<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>)>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let mut out = Vec::with_capacity(polys_per_point.len());
    for polys in polys_per_point {
        out.push(commit_with_params::<F, D, P>(polys, setup, params)?);
    }
    Ok(out)
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use crate::backend::DensePoly;
    use crate::kernels::crt_ntt::build_ntt_slot;
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp64;
    use akita_types::layout::sis_derivation::balanced_digit_delta_bound;
    use akita_types::{AjtaiKeyParams, FlatMatrix, LevelParams as LP, SisModulusFamily};

    type F = Fp64<4294967197>;
    const D: usize = 4;

    /// Build a small `LevelParams` configured for the tiered root commit
    /// path. The shape is deliberately tiny so the test stays fast while
    /// still exercising every code path in `commit_tiered_with_params`.
    #[allow(clippy::too_many_arguments)]
    fn tiered_level_params(
        n_a: usize,
        n_b_prime: usize,
        n_d: usize,
        n_f: usize,
        num_blocks: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        split_factor: usize,
        outer_log_basis: u32,
        num_digits_outer: usize,
    ) -> LP {
        let log_basis: u32 = 2;
        let stage1 = SparseChallengeConfig::Uniform {
            weight: 2,
            nonzero_coeffs: vec![-1, 1],
        };
        let chunk_width = n_a * num_digits_open * num_blocks / split_factor;
        let f_width = n_b_prime * split_factor * num_digits_outer;
        let inner_width = block_len * num_digits_commit;
        let d_matrix_width = num_digits_open * num_blocks;
        LP {
            ring_dimension: D,
            log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                SisModulusFamily::Q64,
                n_a,
                inner_width,
                balanced_digit_delta_bound(log_basis),
                D,
            ),
            // The B key carries B' shape under tiering.
            b_key: AjtaiKeyParams::new_unchecked(
                SisModulusFamily::Q64,
                n_b_prime,
                chunk_width,
                balanced_digit_delta_bound(log_basis),
                D,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                SisModulusFamily::Q64,
                n_d,
                d_matrix_width,
                balanced_digit_delta_bound(log_basis),
                D,
            ),
            num_blocks,
            block_len,
            m_vars: block_len.trailing_zeros() as usize,
            r_vars: num_blocks.trailing_zeros() as usize,
            stage1_config: stage1,
            num_digits_commit,
            num_digits_open,
            num_digits_fold: 1,
            split_factor,
            outer_log_basis,
            num_digits_outer,
            f_key: AjtaiKeyParams::new_unchecked(
                SisModulusFamily::Q64,
                n_f,
                f_width,
                balanced_digit_delta_bound(outer_log_basis),
                D,
            ),
        }
    }

    fn legacy_level_params_matching(tier: &LP) -> LP {
        let mut legacy = tier.clone();
        legacy.split_factor = 1;
        legacy.outer_log_basis = 0;
        legacy.num_digits_outer = 0;
        legacy.f_key = AjtaiKeyParams::default();
        // Legacy B key spans the full outer width.
        legacy.b_key = AjtaiKeyParams::new_unchecked(
            tier.b_key.sis_family(),
            tier.b_key.row_len(),
            tier.full_outer_width(),
            tier.b_key.collision_inf(),
            tier.ring_dimension,
        );
        legacy
    }

    /// Build an F NTT cache from a deterministic small `FlatMatrix`.
    fn build_synthetic_f_cache(n_f: usize, f_width: usize) -> (NttSlotCache<D>, usize) {
        // Use small distinct values per cell so any indexing mismatch
        // would change the resulting product.
        let total = n_f * f_width * D;
        let data: Vec<F> = (0..total)
            .map(|idx| F::from_canonical_u128_reduced(1 + (idx as u128 * 13) % 97))
            .collect();
        let flat = FlatMatrix::<F>::from_flat_data(data, D);
        let view = flat
            .ring_view::<D>(n_f, f_width)
            .expect("test F view shape");
        let cache = build_ntt_slot(view).expect("build F NTT cache");
        (cache, f_width)
    }

    // Tiered outer-digit parameters covering the full Fp64 modulus
    // (≈ 32 bits). `outer_log_basis = 6` (the max the i8 storage admits)
    // with `num_digits_outer = 6` covers ≤ 2^36 entries, well above the
    // field modulus, so the gadget identity `u_i = G · û_i` holds
    // exactly.
    const OUTER_LOG_BASIS: u32 = 6;
    const NUM_DIGITS_OUTER: usize = 6;

    #[test]
    fn commit_tiered_with_params_rejects_split_factor_one() {
        let mut tier =
            tiered_level_params(1, 2, 1, 2, 2, 2, 1, 2, 2, OUTER_LOG_BASIS, NUM_DIGITS_OUTER);
        tier.split_factor = 1;
        let f_width = tier.b_key.row_len() * 2 * NUM_DIGITS_OUTER;
        let (f_cache, f_stride) = build_synthetic_f_cache(2, f_width);
        let poly = DensePoly::<F, D>::from_field_evals(
            (4u32 + D.trailing_zeros()) as usize - D.trailing_zeros() as usize,
            &vec![F::zero(); 1 << ((D.trailing_zeros() as usize) + 2)],
        )
        .expect("dense");
        let setup = AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, 4, 16).unwrap();
        let err = commit_tiered_with_params::<F, D, _>(
            std::slice::from_ref(&poly),
            &setup,
            &tier,
            &f_cache,
            f_stride,
        )
        .expect_err("split_factor == 1 must take the legacy path");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("is_tiered_root"),
            "error names is_tiered_root: {msg}"
        );
    }

    #[test]
    fn commit_tiered_with_params_recomposes_to_b_prime_chunks() {
        // f = 2 chunks, n_b' = 2, depth NUM_DIGITS_OUTER (full-field for
        // Fp64), basis 2^OUTER_LOG_BASIS; n_F = 2.
        // n_a = 1, num_blocks = 2, block_len = 2, num_digits_commit = 1,
        // num_digits_open = 2 ⇒ outer_width = 4, chunk_width = 2,
        // f_width = 2 * 2 * NUM_DIGITS_OUTER.
        let tier =
            tiered_level_params(1, 2, 1, 2, 2, 2, 1, 2, 2, OUTER_LOG_BASIS, NUM_DIGITS_OUTER);
        let f_width = tier.f_key.col_len();
        let n_f = tier.f_key.row_len();
        let (f_cache, f_stride) = build_synthetic_f_cache(n_f, f_width);

        // Choose a polynomial with num_ring_elems = num_blocks * block_len
        // = 4; with D = 4 that gives 16 field evaluations = 2^4 evals.
        let num_ring = tier.num_blocks * tier.block_len;
        let num_vars = (num_ring * D).trailing_zeros() as usize;
        let evals: Vec<F> = (0..(1usize << num_vars))
            .map(|idx| F::from_canonical_u128_reduced(((idx as u128) * 17 + 3) % 991))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).expect("dense");

        let max_stride = tier.full_outer_width().max(tier.inner_width());
        let max_rows = tier
            .a_key
            .row_len()
            .max(tier.b_key.row_len())
            .max(tier.d_key.row_len())
            .max(tier.f_key.row_len());
        let setup = AkitaProverSetup::<F, D>::generate_with_capacity(
            num_vars + 4,
            1,
            1,
            max_rows,
            max_stride,
        )
        .expect("setup");

        let (commitment, hint) = commit_tiered_with_params::<F, D, _>(
            std::slice::from_ref(&poly),
            &setup,
            &tier,
            &f_cache,
            f_stride,
        )
        .expect("tiered commit");

        // Shape contract.
        assert_eq!(
            commitment.u.len(),
            tier.outer_commitment_rows(),
            "RingCommitment.u length matches outer_commitment_rows (= n_F)"
        );
        assert_eq!(commitment.u.len(), n_f);
        assert_eq!(
            hint.outer_digits().len(),
            1,
            "one outer_digits entry per opening-point commitment"
        );
        let uhat_concat = &hint.outer_digits()[0];
        assert_eq!(
            uhat_concat.flat_digits().len(),
            tier.b_key.row_len() * tier.split_factor * tier.num_digits_outer,
            "uhat_concat digit count = n_b' * split * δ_outer"
        );

        // Independently rebuild u_final from t_hat using the kernel via
        // the tiered_commit helper directly.
        let inner = <DensePoly<F, D> as AkitaPolyOps<F, D>>::commit_inner_witness(
            &poly,
            &setup.expanded.shared_matrix,
            &setup.ntt_shared,
            tier.a_key.row_len(),
            tier.block_len,
            tier.num_digits_commit,
            tier.num_digits_open,
            tier.log_basis,
            setup.expanded.seed.max_stride,
        )
        .expect("inner");

        let b_input_digits: Vec<[i8; D]> = inner.decomposed_inner_rows.flat_digits().to_vec();
        let expected = crate::protocol::tiered_commit::tiered_commit::<F, _, _, D>(
            crate::protocol::tiered_commit::TieredCommitParams {
                split_factor: tier.split_factor,
                n_b_prime: tier.b_key.row_len(),
                outer_log_basis: tier.outer_log_basis,
                num_digits_outer: tier.num_digits_outer,
            },
            &b_input_digits,
            |chunk| {
                mat_vec_mul_ntt_single_i8::<F, D>(
                    &setup.ntt_shared,
                    tier.b_key.row_len(),
                    setup.expanded.seed.max_stride,
                    chunk,
                )
            },
            |uhat_concat| mat_vec_mul_ntt_single_i8::<F, D>(&f_cache, n_f, f_stride, uhat_concat),
        )
        .expect("reference tiered commit");

        assert_eq!(
            commitment.u, expected.u_final,
            "commit_tiered_with_params output matches the reference tiered_commit kernel"
        );
        assert_eq!(
            uhat_concat.flat_digits(),
            expected.uhat_concat.flat_digits(),
            "uhat_concat flat digits match the reference reconstruction"
        );

        // Sanity: the gadget identity holds for every (chunk, b'_row).
        let depth = tier.num_digits_outer;
        let n_b = tier.b_key.row_len();
        let log_basis = tier.outer_log_basis;
        // Recompute u_i directly using the existing NTT kernel on each
        // chunk and confirm gadget recomposition.
        for chunk_idx in 0..tier.split_factor {
            let lo = chunk_idx * (b_input_digits.len() / tier.split_factor);
            let hi = lo + (b_input_digits.len() / tier.split_factor);
            let u_i = mat_vec_mul_ntt_single_i8::<F, D>(
                &setup.ntt_shared,
                n_b,
                setup.expanded.seed.max_stride,
                &b_input_digits[lo..hi],
            );
            for (row, expected_u) in u_i.iter().enumerate() {
                let plane_offset = chunk_idx * n_b * depth + row * depth;
                let digits = &uhat_concat.flat_digits()[plane_offset..plane_offset + depth];
                let recomposed =
                    CyclotomicRing::<F, D>::gadget_recompose_pow2_i8(digits, log_basis);
                assert_eq!(
                    recomposed, *expected_u,
                    "gadget identity holds for chunk = {chunk_idx}, row = {row}"
                );
            }
        }
    }

    #[test]
    fn commit_tiered_with_params_rejects_mismatched_f_key_width() {
        let mut tier =
            tiered_level_params(1, 2, 1, 2, 2, 2, 1, 2, 2, OUTER_LOG_BASIS, NUM_DIGITS_OUTER);
        // Corrupt f_key.col_len so the validation triggers.
        tier.f_key = AjtaiKeyParams::new_unchecked(
            SisModulusFamily::Q64,
            tier.f_key.row_len(),
            tier.f_key.col_len() + 1,
            tier.f_key.collision_inf(),
            D,
        );

        let n_f = tier.f_key.row_len();
        // Build an F cache whose width matches the *corrupted* col_len so
        // the validation error is what triggers, not an NTT-cache mismatch.
        let (f_cache, f_stride) = build_synthetic_f_cache(n_f, tier.f_key.col_len());

        let num_ring = tier.num_blocks * tier.block_len;
        let num_vars = (num_ring * D).trailing_zeros() as usize;
        let evals = vec![F::zero(); 1usize << num_vars];
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).expect("dense");

        let max_stride = tier.full_outer_width().max(tier.inner_width());
        let max_rows = tier
            .a_key
            .row_len()
            .max(tier.b_key.row_len())
            .max(tier.d_key.row_len())
            .max(tier.f_key.row_len());
        let setup = AkitaProverSetup::<F, D>::generate_with_capacity(
            num_vars + 4,
            1,
            1,
            max_rows,
            max_stride,
        )
        .unwrap();

        let err = commit_tiered_with_params::<F, D, _>(
            std::slice::from_ref(&poly),
            &setup,
            &tier,
            &f_cache,
            f_stride,
        )
        .expect_err("mismatched f_key width must be rejected");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("f_key.col_len"),
            "error names f_key.col_len: {msg}"
        );
    }

    #[test]
    fn commit_with_params_dispatches_to_tiered_when_is_tiered_root() {
        // Use the unified `commit_with_params` entry point and verify it
        // produces the same `u_final` as a direct `commit_tiered_with_params`
        // call with a hand-built F NTT cache derived from the same seed
        // and label. This guards the public dispatch contract.
        use crate::kernels::matrix::derive_tier1_f_matrix_flat;

        let tier =
            tiered_level_params(1, 2, 1, 2, 2, 2, 1, 2, 2, OUTER_LOG_BASIS, NUM_DIGITS_OUTER);
        let n_f = tier.f_key.row_len();
        let f_width = tier.f_key.col_len();

        let num_ring = tier.num_blocks * tier.block_len;
        let num_vars = (num_ring * D).trailing_zeros() as usize;
        let evals: Vec<F> = (0..(1usize << num_vars))
            .map(|idx| F::from_canonical_u128_reduced(((idx as u128) * 41 + 5) % 977))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).expect("dense");

        let max_stride = tier.full_outer_width().max(tier.inner_width());
        let max_rows = tier
            .a_key
            .row_len()
            .max(tier.b_key.row_len())
            .max(tier.d_key.row_len())
            .max(tier.f_key.row_len());
        let setup = AkitaProverSetup::<F, D>::generate_with_capacity(
            num_vars + 4,
            1,
            1,
            max_rows,
            max_stride,
        )
        .expect("setup");

        // Path A: unified entry point.
        let (commitment_a, hint_a) =
            commit_with_params::<F, D, _>(std::slice::from_ref(&poly), &setup, &tier)
                .expect("unified commit dispatches to tiered");

        // Path B: explicit tiered call with the same setup-derived F.
        let total_f_ring_elements = n_f * f_width;
        let f_flat = derive_tier1_f_matrix_flat::<F, D>(
            total_f_ring_elements,
            &setup.expanded.seed.public_matrix_seed,
        );
        let f_view = f_flat
            .ring_view::<D>(n_f, f_width)
            .expect("test F view shape");
        let f_cache = build_ntt_slot(f_view).expect("f cache");
        let (commitment_b, hint_b) = commit_tiered_with_params::<F, D, _>(
            std::slice::from_ref(&poly),
            &setup,
            &tier,
            &f_cache,
            f_width,
        )
        .expect("explicit tiered commit");

        assert_eq!(
            commitment_a.u, commitment_b.u,
            "unified dispatch must produce the same u_final as explicit tiered call"
        );
        assert_eq!(
            hint_a.outer_digits()[0].flat_digits(),
            hint_b.outer_digits()[0].flat_digits(),
            "unified dispatch must produce the same uhat_concat as explicit tiered call"
        );
    }

    #[test]
    fn legacy_commit_with_params_unchanged_for_split_factor_one() {
        // Sanity check: the legacy `commit_with_params` path is
        // untouched when `split_factor == 1`. We construct a tiered
        // LevelParams, derive a matching legacy LevelParams, and verify
        // commit_with_params produces an `n_b`-length commitment exactly
        // matching today's behaviour.
        let tier =
            tiered_level_params(1, 2, 1, 2, 2, 2, 1, 2, 2, OUTER_LOG_BASIS, NUM_DIGITS_OUTER);
        let legacy = legacy_level_params_matching(&tier);
        assert!(!legacy.is_tiered_root());
        assert_eq!(legacy.outer_commitment_rows(), legacy.b_key.row_len());

        let num_ring = legacy.num_blocks * legacy.block_len;
        let num_vars = (num_ring * D).trailing_zeros() as usize;
        let evals: Vec<F> = (0..(1usize << num_vars))
            .map(|idx| F::from_canonical_u128_reduced(((idx as u128) * 31 + 7) % 503))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).expect("dense");

        let max_stride = legacy.full_outer_width().max(legacy.inner_width());
        let max_rows = legacy
            .a_key
            .row_len()
            .max(legacy.b_key.row_len())
            .max(legacy.d_key.row_len());
        let setup = AkitaProverSetup::<F, D>::generate_with_capacity(
            num_vars + 4,
            1,
            1,
            max_rows,
            max_stride,
        )
        .unwrap();

        let (commitment, hint) =
            commit_with_params::<F, D, _>(std::slice::from_ref(&poly), &setup, &legacy)
                .expect("legacy commit");
        assert_eq!(commitment.u.len(), legacy.b_key.row_len());
        assert!(hint.outer_digits().is_empty());
    }
}
