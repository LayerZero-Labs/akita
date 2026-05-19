//! Root and recursive level verifier replay for Akita proofs.
//!
//! This module owns the transcript and algebra checks for an already selected
//! root or fold level. Schedule/config dispatch stays with the scheme crate
//! until the verifier-facing config boundary is extracted.

use crate::{
    derive_stage1_challenges, ring_switch_verifier, verify_stage2_with_setup_claim_reduction,
    AkitaStage1Verifier, AkitaStage2Verifier, Stage2MEvalSource,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::ring::trace;
use akita_algebra::ring::{mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_single_i8, NttSlotCache};
use akita_algebra::{CyclotomicRing, EqPolynomial};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, RandomSampling};
use akita_sumcheck::{verify_sumcheck, SumcheckInstanceVerifier};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD,
    ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_EVAL_BATCH, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_BATCH_REL, CHALLENGE_SUMCHECK_ROUND,
};

/// Mirror the prover's `prove_recursive_multi_fold_with_params`
/// per-claim-LP / per-tier merge to count the distinct y-ring slots
/// produced at this level. Consecutive claims with identical
/// `per_claim_lp` AND `Some(tier)` collapse into one group; otherwise
/// each claim is its own group. See book §5.4 line 752.
fn expected_num_groups_for_recursive<F: FieldCore>(
    claims: &[RecursiveOpeningClaim<F>],
    lp: &LevelParams,
) -> usize {
    if claims.is_empty() {
        return 0;
    }
    let mut groups = 0usize;
    let mut prev_lp: Option<&LevelParams> = None;
    let mut prev_tier: Option<akita_types::TieredSetupParams> = None;
    let dummy = lp.clone();
    for claim in claims {
        let lp_i = claim.per_claim_lp.as_ref().unwrap_or(&dummy);
        let cur_tier = claim.tier_marker;
        let mergeable = match (prev_lp, prev_tier, cur_tier) {
            (Some(prev), Some(pt), Some(ct)) => prev == lp_i && pt == ct,
            _ => false,
        };
        if !mergeable {
            groups += 1;
            prev_lp = Some(lp_i);
            prev_tier = cur_tier;
        }
    }
    groups
}
use akita_transcript::Transcript;
use akita_types::{
    append_batch_shape_to_transcript, append_batched_commitments_to_transcript,
    checked_total_claims, flatten_batched_commitment_rows, prepare_root_opening_point,
    reduce_inner_opening_to_ring_element, relation_claim_from_rows_with_layout,
    reorder_stage1_coords, ring_opening_point_from_field, schedule_num_fold_levels,
    tiered_setup_chunk_index_map, tiered_setup_chunk_opening_point, tiered_setup_group_lp,
    untiered_setup_group_lp, w_ring_element_count, w_ring_element_count_with_claim_groups,
    AkitaBatchedProof, AkitaLevelProof, AkitaProofStep, AkitaStage1Proof, AkitaStage2Proof,
    AkitaVerifierSetup, BasisMode, BlockOrder, DirectWitnessProof, FlatRingVec, GroupSpec,
    LevelParams, MultiPointBatchShape, PreparedRootOpeningPoint, RecursiveOpeningClaim,
    RingCommitment, RingOpeningPoint, Schedule, Step, TieredSetupCacheKey, TieredSetupCommitments,
    TieredSetupParams,
};
use std::sync::Arc;
use std::time::Instant;

/// Verifier state carried between recursive fold levels.
///
/// Each entry of `claims` is one polynomial opening that the next fold
/// level must discharge. The single-poly recursive path uses
/// `claims.len() == 1`; Phase D-full slice F adds an additional claim
/// for the shared setup polynomial `S` so the next level discharges
/// the deferred `S(r_setup) = y_setup` claim alongside the folded
/// witness via multi-claim batched Hachi.
pub struct RecursiveVerifierState<F: FieldCore> {
    /// Recursive opening claims to discharge at the next fold level.
    pub claims: Vec<RecursiveOpeningClaim<F>>,
}

/// Deferred setup-polynomial opening emitted by setup-claim reduction.
pub struct DeferredSetupOpening<F: FieldCore> {
    /// Raw setup-claim point in `row | col | coeff` order.
    pub r_setup: Vec<F>,
    /// Number of row variables in `r_setup`.
    pub row_bits: usize,
    /// Number of column variables in `r_setup`.
    pub col_bits: usize,
    /// Number of coefficient variables in `r_setup`.
    pub coeff_bits: usize,
    /// Claimed value `S(opening_point)`.
    pub opening: F,
    /// Live setup row count used by the originating M-table.
    pub row_count: usize,
    /// Live setup column count used by the originating M-table.
    pub col_count: usize,
}

/// Output of one verified fold level.
pub struct VerifyLevelOutput<F: FieldCore> {
    /// Stage-2 challenges used as the next recursive `w` opening point.
    pub challenges: Vec<F>,
    /// Optional setup opening to batch into the next recursive fold.
    pub setup_opening: Option<DeferredSetupOpening<F>>,
}

fn setup_opening_point_from_r_setup<F: FieldCore>(
    r_setup: &[F],
    row_bits: usize,
    col_bits: usize,
    coeff_bits: usize,
    setup_lp: &LevelParams,
) -> Result<Vec<F>, AkitaError> {
    let expected = row_bits + col_bits + coeff_bits;
    if r_setup.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: r_setup.len(),
        });
    }
    let rows = &r_setup[..row_bits];
    let cols = &r_setup[row_bits..row_bits + col_bits];
    let coeffs = &r_setup[row_bits + col_bits..];
    let mut flat_bits = Vec::with_capacity(row_bits + col_bits);
    flat_bits.extend_from_slice(cols);
    flat_bits.extend_from_slice(rows);
    flat_bits.resize(setup_lp.m_vars + setup_lp.r_vars, F::zero());
    let block_bits = &flat_bits[setup_lp.m_vars..setup_lp.m_vars + setup_lp.r_vars];
    let elem_bits = &flat_bits[..setup_lp.m_vars];
    let mut out = Vec::with_capacity(coeff_bits + setup_lp.r_vars + setup_lp.m_vars);
    out.extend_from_slice(coeffs);
    out.extend_from_slice(block_bits);
    out.extend_from_slice(elem_bits);
    Ok(out)
}

fn derive_setup_commitment_flat<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    row_count: usize,
    col_count: usize,
    s_lp: &LevelParams,
) -> Result<FlatRingVec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    let stride = setup.expanded.seed.max_stride.max(1);
    let s_view = setup
        .expanded
        .shared_matrix
        .ring_view::<D>(row_count, stride);
    let a_view = setup
        .expanded
        .shared_matrix
        .ring_view::<D>(s_lp.a_key.row_len(), stride);
    let b_view = setup
        .expanded
        .shared_matrix
        .ring_view::<D>(s_lp.b_key.row_len(), stride);
    let q = (-F::one()).to_canonical_u128() + 1;
    let commit_params =
        BalancedDecomposePow2I8Params::new(s_lp.num_digits_commit, s_lp.log_basis, q);
    let open_params = BalancedDecomposePow2I8Params::new(s_lp.num_digits_open, s_lp.log_basis, q);
    let mut t_rows =
        vec![vec![CyclotomicRing::<F, D>::zero(); s_lp.a_key.row_len()]; s_lp.num_blocks];
    for (block_idx, t_for_block) in t_rows.iter_mut().enumerate() {
        for elem_idx in 0..s_lp.block_len {
            let global = block_idx * s_lp.block_len + elem_idx;
            let row = global / col_count;
            let col_in_s = global % col_count;
            if row >= row_count {
                continue;
            }
            let mut digits = vec![[0i8; D]; s_lp.num_digits_commit];
            s_view.row(row)[col_in_s]
                .balanced_decompose_pow2_i8_into_with_params(&mut digits, &commit_params);
            for (digit_idx, digit) in digits.iter().enumerate() {
                let digit_ring = CyclotomicRing::from_coefficients(digit.map(F::from_i8));
                let matrix_col = elem_idx * s_lp.num_digits_commit + digit_idx;
                for (a_idx, acc) in t_for_block.iter_mut().enumerate() {
                    *acc += a_view.row(a_idx)[matrix_col] * digit_ring;
                }
            }
        }
    }
    let mut u = vec![CyclotomicRing::<F, D>::zero(); s_lp.b_key.row_len()];
    for (block_idx, t_for_block) in t_rows.iter().enumerate() {
        for (a_idx, t) in t_for_block.iter().enumerate() {
            let mut digits = vec![[0i8; D]; s_lp.num_digits_open];
            t.balanced_decompose_pow2_i8_into_with_params(&mut digits, &open_params);
            for (digit_idx, digit) in digits.iter().enumerate() {
                let digit_ring = CyclotomicRing::from_coefficients(digit.map(F::from_i8));
                let matrix_col = block_idx * s_lp.a_key.row_len() * s_lp.num_digits_open
                    + a_idx * s_lp.num_digits_open
                    + digit_idx;
                for (b_idx, acc) in u.iter_mut().enumerate() {
                    *acc += b_view.row(b_idx)[matrix_col] * digit_ring;
                }
            }
        }
    }
    Ok(FlatRingVec::from_ring_elems::<D>(&u))
}

/// Verifier-side commit on a sequence of ring elements under `s_lp`.
///
/// Mirrors the prover's `commit_dense_s_handle` in
/// `akita-prover/src/protocol/flow.rs`: digit-decompose the input ring
/// elements, recompose `t_rows` via the inner Ajtai matrix `A`, then
/// digit-decompose `t_rows` and form `u = B · t̂`, using the shared
/// CRT+NTT mat-vec kernels instead of naive ring multiplication.
///
/// Used by both [`derive_setup_commitment_flat`] (which builds its
/// input from a shared-matrix sub-view) and
/// [`derive_tiered_setup_material_for_verifier`] (which builds its
/// inputs from per-chunk slices of the shared matrix and the
/// concatenated chunk-commits for the meta tier). Both must use this
/// helper so the verifier's material is bit-identical to the prover's.
///
/// `ntt_shared` is the full setup-matrix CRT+NTT cache (mirrors the
/// prover's [`AkitaProverSetup::ntt_shared`]). The mat-vec dispatcher
/// reinterprets the flat slot as `num_rows × max_stride` per call, so
/// both the inner `A · digits` and outer `B · t̂` accesses share one
/// cache and we pay the NTT preprocessing cost once across the whole
/// verify.
fn derive_commitment_for_ring_slice<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    rings: &[CyclotomicRing<F, D>],
    s_lp: &LevelParams,
    ntt_shared: &NttSlotCache<D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    let stride = setup.expanded.seed.max_stride.max(1);

    let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..s_lp.num_blocks)
        .map(|block_idx| {
            let start = block_idx * s_lp.block_len;
            if start >= rings.len() {
                &[] as &[CyclotomicRing<F, D>]
            } else {
                &rings[start..(start + s_lp.block_len).min(rings.len())]
            }
        })
        .collect();
    let t_rows = mat_vec_mul_ntt_i8_dense(
        ntt_shared,
        s_lp.a_key.row_len(),
        stride,
        &block_slices,
        s_lp.num_digits_commit,
        s_lp.log_basis,
    );

    let q = (-F::one()).to_canonical_u128() + 1;
    let open_params = BalancedDecomposePow2I8Params::new(s_lp.num_digits_open, s_lp.log_basis, q);
    let mut flat_digits =
        Vec::with_capacity(s_lp.num_blocks * s_lp.a_key.row_len() * s_lp.num_digits_open);
    for t_for_block in &t_rows {
        for t in t_for_block {
            let mut digits = vec![[0i8; D]; s_lp.num_digits_open];
            t.balanced_decompose_pow2_i8_into_with_params(&mut digits, &open_params);
            flat_digits.extend_from_slice(&digits);
        }
    }
    Ok(mat_vec_mul_ntt_single_i8(
        ntt_shared,
        s_lp.b_key.row_len(),
        stride,
        &flat_digits,
    ))
}

/// Compute the meta-tier `LevelParams` from the per-chunk LP and tier
/// shape. Mirrors the prover-side `meta_lp_from_chunks` in
/// `akita-prover/src/protocol/flow.rs` so prover and verifier derive
/// the meta tier under the same shape.
fn meta_lp_from_chunks(
    next_level: &LevelParams,
    chunk_lp: &LevelParams,
    tier: TieredSetupParams,
) -> Result<LevelParams, AkitaError> {
    let meta_field_len = tier.num_chunks * chunk_lp.b_key.row_len() * next_level.ring_dimension;
    let next_pow2 = meta_field_len.next_power_of_two();
    untiered_setup_group_lp(next_level, next_pow2)
}

/// Compute the opening of a dense ring polynomial at the (padded)
/// opening point, mirroring the prover-side
/// `DensePoly::evaluate_and_fold` chain exactly so transcript-absorbed
/// per-claim openings match between prover and verifier.
///
/// The prover computes each routed chunk's per-claim opening as
/// `coefficients()[0](evaluate_and_fold(chunk_poly, opening_point)
/// · σ_{-1}(v))` (see
/// `prove_recursive_multi_fold_with_params` in
/// [crates/akita-prover/src/protocol/flow.rs] line 1574-1583).
/// The previous verifier wrote the routed `y_setup` into every chunk +
/// meta claim slot, which diverged the recursive transcript at the
/// openings absorption and caused the per-group trace check to reject
/// (book §5.4 routes share folding challenges).
///
/// Because the routed S material is public (chunks are slices of the
/// shared setup matrix; the meta input is the padded concat of the
/// chunk B-commitments, which the verifier itself derived), the
/// verifier can recompute the per-claim opening at no proof-byte cost
/// and full soundness — there is nothing for a malicious prover to
/// lie about.
///
/// `pub(crate)` for use by the tiered claim expansion path; exposed
/// to integration tests via the `__test_dense_ring_opening_at_point`
/// shim below so the prover/verifier handshake can be cross-checked
/// without spinning up an end-to-end run.
///
/// # Errors
///
/// Returns an error if the opening point cannot be reduced to a ring
/// element at `D` or if the outer point cannot be expanded into the
/// `(a, b)` fold/eval scalars at the claim LP's `(r_vars, m_vars)`.
pub(crate) fn dense_ring_opening_at_point<F, const D: usize>(
    coeffs: &[CyclotomicRing<F, D>],
    opening_point: &[F],
    claim_lp: &LevelParams,
    alpha_bits: usize,
) -> Result<F, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let target_num_vars = claim_lp.m_vars + claim_lp.r_vars + alpha_bits;
    let mut padded_point = opening_point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];

    let ring_opening_point = ring_opening_point_from_field::<F>(
        reduced_point,
        claim_lp.r_vars,
        claim_lp.m_vars,
        BasisMode::Lagrange,
        BlockOrder::ColumnMajor,
    )?;
    let inner_reduction =
        reduce_inner_opening_to_ring_element::<F, D>(inner_point, BasisMode::Lagrange)?;

    let block_len = claim_lp.block_len;
    let num_blocks = coeffs.len().div_ceil(block_len);
    let folded: Vec<CyclotomicRing<F, D>> = (0..num_blocks)
        .map(|i| {
            let start = i * block_len;
            let end = (start + block_len).min(coeffs.len());
            let block = &coeffs[start..end];
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (b_j, &a_j) in block.iter().zip(ring_opening_point.a.iter()) {
                acc += b_j.scale(&a_j);
            }
            acc
        })
        .collect();
    let eval = folded
        .iter()
        .zip(ring_opening_point.b.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f, s)| {
            acc + f.scale(s)
        });
    Ok((eval * inner_reduction.sigma_m1()).coefficients()[0])
}

/// Expand a routed tiered S claim into `k + 1` `RecursiveOpeningClaim`
/// entries (k chunks at `chunk_lp` + 1 meta at `meta_lp`).
///
/// Derives the verifier-side per-chunk and meta commitments via
/// [`derive_tiered_setup_material_for_verifier`] and the per-claim
/// openings via [`dense_ring_opening_at_point`] reading public material
/// directly.
/// All chunk + meta claims share the routed opening point because
/// book §5.4 line 949 dictates "the two polynomials share folding
/// challenges". The unused `_y_setup` parameter (the routed
/// setup-claim-reduction output) is intentionally NOT used as the
/// per-claim opening: it is the AGGREGATE opening at the routed
/// point, while each chunk owns its own MLE at the same point.
#[allow(clippy::too_many_arguments)]
fn expand_tiered_setup_claims<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    next_level_params: &LevelParams,
    row_count: usize,
    col_count: usize,
    tier: TieredSetupParams,
    r_setup: &[F],
    row_bits: usize,
    col_bits: usize,
    coeff_bits: usize,
    y_setup: F,
    claims: &mut Vec<RecursiveOpeningClaim<F>>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Send + Sync + 'static,
{
    let setup_field_len = row_count * col_count * D;
    let full_s_lp = untiered_setup_group_lp(next_level_params, setup_field_len)?;
    let opening_point =
        setup_opening_point_from_r_setup(r_setup, row_bits, col_bits, coeff_bits, &full_s_lp)?;
    let chunk_lp = tiered_setup_group_lp(next_level_params, setup_field_len, tier)?;
    let meta_lp = meta_lp_from_chunks(next_level_params, &chunk_lp, tier)?;
    let material = tiered_setup_material_for_verifier::<F, D>(
        setup, row_count, col_count, &chunk_lp, &meta_lp, tier,
    )?;
    let chunk_w_len = chunk_lp.num_blocks * chunk_lp.block_len * D;
    let meta_input_pow2 = (tier.num_chunks * chunk_lp.b_key.row_len()).next_power_of_two();
    let meta_w_len = meta_input_pow2 * D;
    let alpha_bits = D.trailing_zeros() as usize;

    // Re-read s_rings to slice per-chunk coefficients for opening MLEs.
    // The setup matrix is public, so the verifier and prover trivially
    // agree on these values. Cost is k chunks × chunk_n × D field ops.
    let live_n_s = row_count * col_count;
    let n_s = live_n_s.next_power_of_two();
    let log_shrink = tier.log2_shrink()? as usize;
    let chunk_indices = tiered_setup_chunk_index_map(
        chunk_lp.r_vars + log_shrink,
        chunk_lp.m_vars + log_shrink,
        tier,
    )?;
    let chunk_opening_point = tiered_setup_chunk_opening_point(
        &opening_point,
        alpha_bits,
        chunk_lp.r_vars + log_shrink,
        chunk_lp.m_vars + log_shrink,
        tier,
    )?;
    let r_high_start = alpha_bits + chunk_lp.r_vars;
    let m_high_start = alpha_bits + chunk_lp.r_vars + log_shrink + chunk_lp.m_vars;
    let eq_high_r = EqPolynomial::evals(&opening_point[r_high_start..r_high_start + log_shrink]);
    let eq_high_m = EqPolynomial::evals(&opening_point[m_high_start..m_high_start + log_shrink]);
    let stride = setup.expanded.seed.max_stride.max(1);
    let s_view = setup
        .expanded
        .shared_matrix
        .ring_view::<D>(row_count, stride);
    let mut s_rings = Vec::with_capacity(n_s);
    for row in 0..row_count {
        s_rings.extend_from_slice(&s_view.row(row)[..col_count]);
    }
    s_rings.resize(n_s, CyclotomicRing::<F, D>::zero());

    let mut recombined_setup_opening = F::zero();
    for (j, indices) in chunk_indices.iter().enumerate() {
        let commitment = FlatRingVec::from_ring_elems::<D>(&material.chunk_b_commitments[j]);
        let chunk_slice = indices.iter().map(|&idx| s_rings[idx]).collect::<Vec<_>>();
        let chunk_opening = dense_ring_opening_at_point::<F, D>(
            &chunk_slice,
            &chunk_opening_point,
            &chunk_lp,
            alpha_bits,
        )?;
        let high_m = j / tier.shrink_factor;
        let high_r = j % tier.shrink_factor;
        recombined_setup_opening += eq_high_m[high_m] * eq_high_r[high_r] * chunk_opening;
        claims.push(RecursiveOpeningClaim {
            opening_point: chunk_opening_point.clone(),
            opening: chunk_opening,
            commitment,
            basis: BasisMode::Lagrange,
            w_len: chunk_w_len,
            log_basis: next_level_params.log_basis,
            per_claim_lp: Some(chunk_lp.clone()),
            tier_marker: Some(tier),
        });
    }
    if recombined_setup_opening != y_setup {
        tracing::debug!("[expand_tiered_setup_claims] recombined setup opening mismatch");
        return Err(AkitaError::InvalidProof);
    }

    // Meta input poly: padded concat of chunk B-commitments (book line
    // 695 "binds the collection of per-chunk commitment vectors via a
    // standard Akita commitment"). Mirrors the prover's
    // `build_tiered_handle_material` construction.
    let meta_len: usize = material.chunk_b_commitments.iter().map(Vec::len).sum();
    let mut meta_input: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(meta_input_pow2);
    for chunk in &material.chunk_b_commitments {
        meta_input.extend_from_slice(chunk);
    }
    meta_input.resize(meta_input_pow2, CyclotomicRing::<F, D>::zero());
    debug_assert!(
        meta_len <= meta_input_pow2,
        "meta input concatenation overruns the pow2-padded buffer"
    );

    let meta_commitment = FlatRingVec::from_ring_elems::<D>(&material.meta_b_commitment);
    let meta_opening =
        dense_ring_opening_at_point::<F, D>(&meta_input, &opening_point, &meta_lp, alpha_bits)?;
    claims.push(RecursiveOpeningClaim {
        opening_point: opening_point.to_vec(),
        opening: meta_opening,
        commitment: meta_commitment,
        basis: BasisMode::Lagrange,
        w_len: meta_w_len,
        log_basis: next_level_params.log_basis,
        per_claim_lp: Some(meta_lp),
        tier_marker: None,
    });
    Ok(())
}

/// Derive verifier-side tiered routed-S material from the public setup
/// matrix.
///
/// Mirrors the prover's `build_tiered_handle_material` in
/// `akita-prover/src/protocol/flow.rs`: row-major linearization of the
/// shared matrix into `tier.num_chunks` equal chunks of `chunk_n =
/// n_s / k` ring elements each, commit each chunk under `chunk_lp`,
/// then commit the concatenated chunk-commits (padded to next pow2)
/// under `meta_lp`.
///
/// Derive tiered setup commitments, consulting [`AkitaVerifierSetup::tiered_s_cache`]
/// first. Soundness anchors on deterministic derivation from the public
/// shared matrix bound in `setup.expanded`.
///
/// # Errors
///
/// Returns an error if `tier.num_chunks == 0`, if `n_s` is not a
/// multiple of `k`, or if the chunk size is not a power of two. Also
/// returns an error if a previously-cached entry was populated under
/// a different ring dimension `D`.
pub fn derive_tiered_setup_material_for_verifier<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    row_count: usize,
    col_count: usize,
    chunk_lp: &LevelParams,
    meta_lp: &LevelParams,
    tier: TieredSetupParams,
) -> Result<TieredSetupCommitments<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + 'static,
{
    Ok((*tiered_setup_material_for_verifier::<F, D>(
        setup, row_count, col_count, chunk_lp, meta_lp, tier,
    )?)
    .clone())
}

fn tiered_setup_material_for_verifier<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    row_count: usize,
    col_count: usize,
    chunk_lp: &LevelParams,
    meta_lp: &LevelParams,
    tier: TieredSetupParams,
) -> Result<Arc<TieredSetupCommitments<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + 'static,
{
    let key = TieredSetupCacheKey::from_lp(tier, row_count, col_count, chunk_lp, meta_lp);
    setup.tiered_s_cache_get_or_init::<AkitaError, D>(key, || {
        derive_tiered_setup_material_for_verifier_uncached::<F, D>(
            setup, row_count, col_count, chunk_lp, meta_lp, tier,
        )
    })
}

fn derive_tiered_setup_material_for_verifier_uncached<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    row_count: usize,
    col_count: usize,
    chunk_lp: &LevelParams,
    meta_lp: &LevelParams,
    tier: TieredSetupParams,
) -> Result<TieredSetupCommitments<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + 'static,
{
    let live_n_s = row_count * col_count;
    let n_s = live_n_s.next_power_of_two();
    if tier.num_chunks == 0 {
        return Err(AkitaError::InvalidInput(
            "tiered routing requires num_chunks >= 1".to_string(),
        ));
    }
    if !n_s.is_multiple_of(tier.num_chunks) {
        return Err(AkitaError::InvalidInput(format!(
                "padded shared matrix has {n_s} ring elements but tier num_chunks = {} requires a multiple",
                tier.num_chunks
        )));
    }
    let chunk_n = n_s / tier.num_chunks;
    if chunk_n == 0 || !chunk_n.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "tiered chunk size {chunk_n} must be a non-zero power of two"
        )));
    }
    let log_shrink = tier.log2_shrink()? as usize;
    let chunk_indices = tiered_setup_chunk_index_map(
        chunk_lp.r_vars + log_shrink,
        chunk_lp.m_vars + log_shrink,
        tier,
    )?;
    let stride = setup.expanded.seed.max_stride.max(1);
    let view = setup
        .expanded
        .shared_matrix
        .ring_view::<D>(row_count, stride);
    let mut s_rings = Vec::with_capacity(n_s);
    for row in 0..row_count {
        s_rings.extend_from_slice(&view.row(row)[..col_count]);
    }
    s_rings.resize(n_s, CyclotomicRing::<F, D>::zero());

    let ntt_shared = setup.ntt_shared_get_or_init::<D>()?;
    let mut chunk_b_commitments: Vec<Vec<CyclotomicRing<F, D>>> =
        Vec::with_capacity(tier.num_chunks);
    for indices in &chunk_indices {
        let chunk_slice = indices.iter().map(|&idx| s_rings[idx]).collect::<Vec<_>>();
        let u =
            derive_commitment_for_ring_slice::<F, D>(setup, &chunk_slice, chunk_lp, &ntt_shared)?;
        chunk_b_commitments.push(u);
    }

    let meta_len = chunk_b_commitments.iter().map(Vec::len).sum::<usize>();
    if meta_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "tiered meta commitment input is empty".to_string(),
        ));
    }
    let next_pow2 = meta_len.next_power_of_two();
    let mut meta_input: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(next_pow2);
    for chunk in &chunk_b_commitments {
        meta_input.extend_from_slice(chunk);
    }
    meta_input.resize(next_pow2, CyclotomicRing::<F, D>::zero());
    let meta_b_commitment =
        derive_commitment_for_ring_slice::<F, D>(setup, &meta_input, meta_lp, &ntt_shared)?;

    let commitments = TieredSetupCommitments {
        chunk_b_commitments,
        meta_b_commitment,
        params: tier,
    };
    commitments.validate_shape()?;
    Ok(commitments)
}

/// Pre-populate [`AkitaVerifierSetup::tiered_s_cache`] for every tiered
/// fold step in `schedule` (book §5 / Figure 12 line 817 "preprocessed
/// shared-matrix commitment `C_S`").
///
/// The cascade state — `(lp_with_groups, claim_group_sizes, num_eval_rows,
/// num_points)` at each level — is fully determined by the schedule alone,
/// so we can drive the same `derive_tiered_setup_material_for_verifier`
/// path the first verify call would, paying the tiered-S derivation cost
/// at setup time and turning the first verify into a cache hit. The cache
/// key is reconstructed via [`LevelParams::setup_polynomial_padded_dims`],
/// which the runtime `PreparedMEval::setup_polynomial_padded_dims` mirrors;
/// a mismatch would manifest as the runtime path missing the cache and
/// re-deriving, never as an unsound key.
///
/// Soundness anchors on `derive_tiered_setup_material_for_verifier` reading
/// from `setup.expanded.shared_matrix`; this function adds no other code
/// path and never injects pre-computed commitments.
///
/// Returns silently on the first level whose cascade shape this helper
/// can't reconstruct cleanly (non-fold successor, tiered axis underflow,
/// etc.). Any uncached level just falls back to lazy derivation on the
/// first verify call, so partial pre-pop is always safe.
///
/// # Errors
///
/// Returns an error only if a tiered-S derivation fails outright; cache
/// misses caused by unsupported schedule shapes return `Ok(())` and leave
/// the cache empty for the affected level.
pub fn prepopulate_tiered_s_cache<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    schedule: &Schedule,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + Send + Sync + 'static,
{
    // Cascade state mirroring `verify_one_level`'s pre-stage1 construction:
    // - L=0 (root): singleton W, no `lp.groups`.
    // - L>0 with tiered cascade: 3 groups `[W, chunks, meta]`,
    //   `claim_group_sizes = [1, k, 1]`, 3 distinct opening points.
    // - L>0 with un-tiered S cascade: 2 groups `[W, S]`,
    //   `claim_group_sizes = [1, 1]`, 2 distinct opening points.
    // - L>0 without cascade: degenerate single-group `[W]`.
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Ok(());
    };
    let mut current_lp = root_step.params.clone();
    let mut current_claim_group_sizes: Vec<usize> = vec![1];
    let mut current_num_eval_rows = 1usize;
    let mut current_num_points = 1usize;

    for (level_idx, step) in schedule.steps.iter().enumerate() {
        let Step::Fold(fold_step) = step else { break };

        // Dims this level's M-table would observe at runtime.
        let (row_count, col_count_padded) = match current_lp.setup_polynomial_padded_dims(
            &current_claim_group_sizes,
            current_num_eval_rows,
            current_num_points,
        ) {
            Ok(dims) => dims,
            Err(_) => return Ok(()),
        };

        // If this level routes S recursively, the NEXT step is necessarily
        // a fold step that consumes that S claim; we need its `LevelParams`
        // to derive `chunk_lp` / `meta_lp` / `s_lp`.
        let next_lp = match schedule.steps.get(level_idx + 1) {
            Some(Step::Fold(next_fold)) => Some(next_fold.params.clone()),
            _ => None,
        };

        if fold_step.tier_setup_params.is_tiered() {
            let Some(next_lp) = next_lp.as_ref() else {
                return Ok(());
            };
            let setup_field_len = row_count.saturating_mul(col_count_padded).saturating_mul(D);
            let Ok(chunk_lp) =
                tiered_setup_group_lp(next_lp, setup_field_len, fold_step.tier_setup_params)
            else {
                return Ok(());
            };
            let Ok(meta_lp) = meta_lp_from_chunks(next_lp, &chunk_lp, fold_step.tier_setup_params)
            else {
                return Ok(());
            };
            tiered_setup_material_for_verifier::<F, D>(
                setup,
                row_count,
                col_count_padded,
                &chunk_lp,
                &meta_lp,
                fold_step.tier_setup_params,
            )?;
            current_lp = LevelParams {
                groups: Some(vec![
                    GroupSpec::from_outer(next_lp),
                    GroupSpec {
                        tier: Some(fold_step.tier_setup_params),
                        ..GroupSpec::from_outer(&chunk_lp)
                    },
                    GroupSpec::from_outer(&meta_lp),
                ]),
                ..next_lp.clone()
            };
            current_claim_group_sizes = vec![1, fold_step.tier_setup_params.num_chunks, 1];
            current_num_eval_rows = 3;
            current_num_points = 3;
        } else if fold_step.s_field_len_emitted > 0 {
            let Some(next_lp) = next_lp.as_ref() else {
                return Ok(());
            };
            let setup_field_len = row_count.saturating_mul(col_count_padded).saturating_mul(D);
            let Ok(s_lp) = untiered_setup_group_lp(next_lp, setup_field_len) else {
                return Ok(());
            };
            current_lp = LevelParams {
                groups: Some(vec![
                    GroupSpec::from_outer(next_lp),
                    GroupSpec::from_outer(&s_lp),
                ]),
                ..next_lp.clone()
            };
            current_claim_group_sizes = vec![1, 1];
            current_num_eval_rows = 2;
            current_num_points = 2;
        } else if let Some(next_lp) = next_lp.as_ref() {
            current_lp = next_lp.clone();
            current_claim_group_sizes = vec![1];
            current_num_eval_rows = 1;
            current_num_points = 1;
        }
    }
    Ok(())
}

/// Verify the root proof payload for singleton and multi-point batched proofs.
///
/// This replays the canonical root transcript layout: batch-shape header,
/// commitments, padded opening points, per-claim field openings, one gamma
/// challenge per claim, and gamma-combined per-point y-rings.
///
/// # Errors
///
/// Returns an error if the proof shape is inconsistent, any public trace check
/// fails, ring-switch replay fails, or either sumcheck verifier rejects.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn verify_root_level<F, T, const D: usize>(
    y_rings_flat: &FlatRingVec<F>,
    v_flat: &FlatRingVec<F>,
    stage1: &AkitaStage1Proof<F>,
    stage2: &AkitaStage2Proof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    openings: &[F],
    commitments: &[RingCommitment<F, D>],
    batch_shape: &MultiPointBatchShape,
    root_lp: &LevelParams,
    batched_lp: &LevelParams,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    routes_setup_recursively: bool,
) -> Result<VerifyLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
{
    let total_t = Instant::now();
    let y_rings = y_rings_flat.as_ring_slice::<D>()?;
    let v_typed = v_flat.as_ring_slice::<D>()?;
    let num_claims = checked_total_claims(&batch_shape.claim_group_sizes, "batched_verify")
        .map_err(|_| AkitaError::InvalidProof)?;
    let num_points = prepared_points.len();
    tracing::debug!(
        "[verify_root] start D={D} num_claims={num_claims} num_points={num_points} is_last={is_last} route_setup={routes_setup_recursively}"
    );
    if num_points == 0
        || y_rings.len() != num_points
        || openings.len() != num_claims
        || commitments.len() != batch_shape.claim_group_sizes.len()
        || batch_shape.claim_to_point.len() != num_claims
    {
        tracing::debug!("[verify_root] shape check failed");
        return Err(AkitaError::InvalidProof);
    }
    if commitments
        .iter()
        .any(|commitment| commitment.u.len() != root_lp.b_key.row_len())
    {
        tracing::debug!("[verify_root] commitment width check failed");
        return Err(AkitaError::InvalidProof);
    }
    // Mirror the prover's commitment-rows optimization: avoid a clone when
    // there is only a single commitment.
    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    append_batch_shape_to_transcript::<F, T>(
        &batch_shape.point_group_sizes,
        &batch_shape.claim_group_sizes,
        transcript,
    );
    append_batched_commitments_to_transcript(commitments, transcript);
    for prepared_point in prepared_points {
        for pt in &prepared_point.padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    for opening in openings {
        transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
    }
    let gamma: Vec<F> = (0..openings.len())
        .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
        .collect();
    for y_ring in y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    // Per-point trace check: for each opening point `j`, verify
    // `trace(y_j · σ_{-1}(v_j)) = d · Σ_{ι: point(ι)=j} γ_ι · opening_ι`.
    // Each opening point carries its own inner reduction `v_j`, which may
    // differ across the batch.
    let d_field = F::from_u64(root_lp.ring_dimension as u64);
    let mut batched_openings_per_point = vec![F::zero(); num_points];
    for (claim_idx, (&opening, &g)) in openings.iter().zip(gamma.iter()).enumerate() {
        let point_idx = batch_shape.claim_to_point[claim_idx];
        batched_openings_per_point[point_idx] += g * opening;
    }
    for (point_idx, (y_ring, &batched_opening)) in y_rings
        .iter()
        .zip(batched_openings_per_point.iter())
        .enumerate()
    {
        let v = &prepared_points[point_idx].inner_reduction;
        let trace_lhs = trace::<F, { D }>(&(*y_ring * v.sigma_m1()));
        let trace_rhs = d_field * batched_opening;
        if trace_lhs != trace_rhs {
            tracing::debug!(
                "[verify_root] trace check failed point_idx={point_idx} after {:?}",
                total_t.elapsed()
            );
            return Err(AkitaError::InvalidProof);
        }
    }
    tracing::debug!(
        "[verify_root] trace checks ok after {:?}",
        total_t.elapsed()
    );

    let stage1_challenges = derive_stage1_challenges::<F, T, D>(
        transcript,
        v_typed,
        root_lp.num_blocks,
        num_claims,
        batched_lp,
    )?;

    let w_len = if is_last {
        final_w.map_or(0, DirectWitnessProof::num_elems)
    } else {
        w_ring_element_count_with_claim_groups::<F>(
            batched_lp,
            &batch_shape.claim_group_sizes,
            num_points,
        ) * D
    };

    let ring_opening_points: Vec<RingOpeningPoint<F>> = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect();
    let t = Instant::now();
    tracing::debug!("[verify_root] ring_switch_verifier start w_len={w_len}");
    let rs = ring_switch_verifier::<F, T, { D }>(
        &ring_opening_points,
        &batch_shape.claim_to_point,
        &stage1_challenges,
        w_len,
        &stage2.next_w_commitment,
        transcript,
        batched_lp,
        &batch_shape.claim_group_sizes,
        &gamma,
        num_points,
    )?;
    tracing::debug!(
        "[verify_root] ring_switch_verifier done after {:?} col_bits={} ring_bits={}",
        t.elapsed(),
        rs.col_bits,
        rs.ring_bits
    );
    let relation_claim = relation_claim_from_rows_with_layout(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_rows,
        y_rings,
        &batched_lp.m_row_layout(batch_shape.claim_group_sizes.len(), num_points),
    );
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let t = Instant::now();
        tracing::debug!("[verify_root] stage1 verify start");
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let out = stage1_verifier.verify(stage1, transcript)?;
        tracing::debug!("[verify_root] stage1 verify done after {:?}", t.elapsed());
        out
    };
    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let gamma_range: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let gamma_rel: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH_REL);
    let stage2_input_claim = gamma_range * stage1.s_claim + gamma_rel * relation_claim;
    let m_eval_source = Stage2MEvalSource::new(rs.prepared_m_eval);
    let stage2_verifier = (if is_last {
        let fw = final_w.ok_or(AkitaError::InvalidProof)?;
        AkitaStage2Verifier::new_with_direct_witness(
            gamma_range,
            gamma_rel,
            stage1.s_claim,
            fw,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &rs.tau1,
            v_typed,
            commitment_rows,
            y_rings,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        AkitaStage2Verifier::new_with_claimed_w_eval(
            gamma_range,
            gamma_rel,
            stage1.s_claim,
            stage2.next_w_eval,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            &ring_opening_points,
            &rs.tau1,
            v_typed,
            commitment_rows,
            y_rings,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    })
    .with_relation_claim(relation_claim);
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        tracing::debug!(
            "[verify_root] stage2 input claim mismatch after {:?}",
            total_t.elapsed()
        );
        return Err(AkitaError::InvalidProof);
    }
    tracing::debug!(
        "[verify_root] stage2 input claim ok after {:?}",
        total_t.elapsed()
    );
    // S-1: schedule shape must equal proof shape at the stage-2 dispatch.
    // The prover emits `stage2.setup_claim_reduction` iff
    // `batched_lp.use_setup_claim_reduction == true` (independent of
    // whether the deferred claim is routed recursively or anchored by
    // the cleartext mle check inside `verify_setup_claim_reduction`).
    // Without this check the verifier dispatches purely on proof shape
    // and ignores the schedule.
    //
    // The schedule-says-route case (`routes_setup_recursively == true`,
    // i.e. planner's `s_field_len_emitted > 0`) implies
    // `use_setup_claim_reduction == true` by construction (see
    // `akita-planner/src/schedule_params.rs:200`), so this single check
    // also covers audit S-1 prong #2.
    let lp_says_emit = batched_lp.use_setup_claim_reduction;
    let proof_says_emit = stage2.setup_claim_reduction.is_some();
    if lp_says_emit != proof_says_emit {
        return Err(AkitaError::InvalidProof);
    }
    let mut deferred_setup_opening = None;
    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        if let Some(payload) = stage2.setup_claim_reduction.as_ref() {
            let t = Instant::now();
            tracing::debug!(
                "[verify_root] stage2/setup-claim verify start route_setup={routes_setup_recursively}"
            );
            // `routes_setup_recursively == true`: the deferred
            // `S(r_setup) = y_setup` claim is discharged by the next
            // fold level's joint multi-group open (book §5.3 lines
            // 627-660). Otherwise the cleartext mle check inside
            // `verify_setup_claim_reduction` must run as the anchor.
            // Soundness requires routing only when the next step is a
            // fold; the planner's `s_field_len_emitted` is the source
            // of this decision.
            let (stage2_challenges, r_setup, s_opening_value) =
                verify_stage2_with_setup_claim_reduction::<F, _, D>(
                    &stage2.sumcheck,
                    payload,
                    &stage2_verifier,
                    transcript,
                    routes_setup_recursively,
                )?;
            tracing::debug!(
                "[verify_root] stage2/setup-claim verify done after {:?}",
                t.elapsed()
            );
            if routes_setup_recursively {
                let (row_bits, col_bits, coeff_bits) = stage2_verifier
                    .prepared_m_eval()
                    .setup_polynomial_padded_dims(setup.expanded.seed.max_stride);
                deferred_setup_opening = Some(DeferredSetupOpening {
                    r_setup,
                    row_bits,
                    col_bits,
                    coeff_bits,
                    opening: s_opening_value,
                    row_count: stage2_verifier
                        .prepared_m_eval()
                        .setup_polynomial_row_count(),
                    col_count: 1usize << col_bits,
                });
            }
            stage2_challenges
        } else {
            let t = Instant::now();
            tracing::debug!("[verify_root] stage2 verify start");
            let out = verify_sumcheck::<F, _, F, _, _>(
                &stage2.sumcheck,
                &stage2_verifier,
                transcript,
                |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
            )?;
            tracing::debug!("[verify_root] stage2 verify done after {:?}", t.elapsed());
            out
        }
    };

    tracing::debug!("[verify_root] done after {:?}", total_t.elapsed());
    Ok(VerifyLevelOutput {
        challenges: sumcheck_challenges,
        setup_opening: deferred_setup_opening,
    })
}

/// Verify one recursive fold level.
///
/// Drives multi-claim verification: `current_state.claims` may carry one
/// or more recursive opening claims, and `level_proof.y_ring` is decoded
/// as a per-point flat ring vector aligned to those claims under the
/// inference rule "one claim per opening point, one commitment per
/// claim".
///
/// At the final level, `final_w` is provided and the verifier checks
/// `w_val` from it directly. At intermediate levels,
/// `level_proof.next_w_eval()` is used. The returned challenges become
/// the opening point for the next level.
///
/// # Errors
///
/// Returns an error if the level proof shape is inconsistent, the public trace
/// check fails, ring-switch replay fails, or either sumcheck verifier rejects.
///
/// # Panics
///
/// Panics if the Phase 5 grouping accumulator's `last_mut()` returns
/// `None` after a merge — this is internally unreachable because the
/// merge branch only fires when the previous claim already pushed a
/// non-empty entry into `claim_group_sizes`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[tracing::instrument(skip_all, name = "verify_one_level")]
pub fn verify_one_level<F, T, const D: usize>(
    level_proof: &AkitaLevelProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<F>,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
    routes_setup_recursively: bool,
) -> Result<VerifyLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
{
    let total_t = Instant::now();
    let claims = current_state.claims.as_slice();
    let num_claims = claims.len();
    tracing::debug!("[verify_one_level] start D={D} num_claims={num_claims} is_last={is_last}");
    if num_claims == 0 {
        return Err(AkitaError::InvalidProof);
    }

    let claim_lps: Vec<LevelParams> = claims
        .iter()
        .map(|claim| claim.per_claim_lp.clone().unwrap_or_else(|| lp.clone()))
        .collect();
    // Phase 5 grouping (book §5.4): mirrors the prover-side grouping in
    // `prove_recursive_multi_fold_with_params`. Consecutive claims with
    // identical per-claim LP AND the same `Some(tier)` marker collapse
    // into one `GroupSpec` with `claim_count = run_length` and tier
    // preserved. Other patterns produce one group per claim.
    let mut batch_groups: Vec<GroupSpec> = Vec::new();
    let mut claim_group_sizes: Vec<usize> = Vec::new();
    let mut prev_lp: Option<&LevelParams> = None;
    let mut prev_tier: Option<TieredSetupParams> = None;
    for (claim_idx, lp_i) in claim_lps.iter().enumerate() {
        let cur_tier = claims[claim_idx].tier_marker;
        let mergeable = match (prev_lp, prev_tier, cur_tier) {
            (Some(prev), Some(pt), Some(ct)) => prev == lp_i && pt == ct,
            _ => false,
        };
        if mergeable {
            *claim_group_sizes.last_mut().unwrap() += 1;
        } else {
            let mut spec = GroupSpec::from_outer(lp_i);
            spec.tier = cur_tier;
            batch_groups.push(spec);
            claim_group_sizes.push(1);
            prev_lp = Some(lp_i);
            prev_tier = cur_tier;
        }
    }
    // Map each claim to its enclosing GROUP index. For un-tiered runs
    // every group is a singleton so `claim_to_point[claim] == claim`.
    // For tier-marked merges (e.g. k chunks merged into one group) all
    // merged claims share the same point index. This mirrors the
    // prover's `prove_recursive_multi_fold_with_params` mapping at
    // [crates/akita-prover/src/protocol/flow.rs] line 1594.
    let num_eval_rows = claim_group_sizes.len();
    let mut claim_to_point: Vec<usize> = Vec::with_capacity(num_claims);
    for (group_idx, &group_size) in claim_group_sizes.iter().enumerate() {
        for _ in 0..group_size {
            claim_to_point.push(group_idx);
        }
    }
    let batch_lp = if num_claims == 1 && claim_lps[0] == *lp {
        lp.clone()
    } else {
        LevelParams {
            groups: Some(batch_groups.clone()),
            ..lp.clone()
        }
    };

    let y_rings = level_proof.y_ring.as_ring_slice::<D>()?;
    if y_rings.len() != num_eval_rows {
        tracing::debug!(
            "[verify_one_level] y_ring mismatch got={} expected={num_eval_rows}",
            y_rings.len()
        );
        return Err(AkitaError::InvalidProof);
    }
    let v_typed = level_proof.v.as_ring_slice::<D>()?;

    let alpha_bits = lp.ring_dimension.trailing_zeros() as usize;
    let mut padded_points: Vec<Vec<F>> = Vec::with_capacity(num_claims);
    let mut inner_reductions: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(num_claims);
    let mut ring_opening_points: Vec<akita_types::RingOpeningPoint<F>> =
        Vec::with_capacity(num_claims);
    for (claim_idx, claim) in claims.iter().enumerate() {
        let claim_lp = &claim_lps[claim_idx];
        let target_num_vars = claim_lp.m_vars + claim_lp.r_vars + alpha_bits;
        if claim.opening_point.len() < alpha_bits {
            return Err(AkitaError::InvalidSetup(
                "opening point length underflow".to_string(),
            ));
        }
        let mut padded_point = claim.opening_point.clone();
        padded_point.resize(target_num_vars, F::zero());
        let inner_point = &padded_point[..alpha_bits];
        let reduced_opening_point = &padded_point[alpha_bits..];

        let inner_reduction =
            reduce_inner_opening_to_ring_element::<F, { D }>(inner_point, claim.basis)?;
        let ring_opening_point = ring_opening_point_from_field::<F>(
            reduced_opening_point,
            claim_lp.r_vars,
            claim_lp.m_vars,
            claim.basis,
            block_order,
        )?;
        padded_points.push(padded_point);
        inner_reductions.push(inner_reduction);
        ring_opening_points.push(ring_opening_point);
    }

    // Transcript layout. For N == 1 we keep today's recursive wire shape
    // (one commitment + padded point + y-ring, no `gamma`). For N > 1 we
    // mirror the root multi-claim layout: append all commitments and
    // padded points, then openings, sample `gamma`, then append the per-
    // point `gamma`-combined y-rings.
    for claim in claims.iter() {
        claim
            .commitment
            .append_as_ring_slice::<T, D>(ABSORB_COMMITMENT, transcript)?;
    }
    for padded_point in &padded_points {
        for pt in padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    let gamma: Vec<F> = if num_claims > 1 {
        for claim in claims.iter() {
            transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, &claim.opening);
        }
        (0..num_claims)
            .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
            .collect()
    } else {
        vec![F::one()]
    };
    for y_ring in y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    // Per-point trace check: the wire `y_rings` carry per-GROUP gamma
    // combinations (book §5.4 tier-aware merge: chunks merged into one
    // group share the routed S opening point and their per-claim y_rings
    // sum into ONE group y_ring). For un-tiered runs every group is a
    // singleton, so this reduces to per-claim trace.
    //
    // The inner_reduction for a group is read from its FIRST claim:
    // merged claims (chunks) share the same opening point, so any
    // claim's inner_reduction is the group's inner_reduction. The merge
    // rule preserves this invariant by only merging claims with the
    // same per-claim LP AND the same tier marker (which implies the
    // shared opening point).
    let d_field = F::from_u64(lp.ring_dimension as u64);
    let mut batched_openings_per_point = vec![F::zero(); num_eval_rows];
    for (claim_idx, (claim, &g)) in claims.iter().zip(gamma.iter()).enumerate() {
        let point_idx = claim_to_point[claim_idx];
        batched_openings_per_point[point_idx] += g * claim.opening;
    }
    // Build per-group inner reductions: claim_to_point maps claim → group,
    // so the first claim in each group is the canonical representative.
    let mut group_inner_reductions: Vec<&CyclotomicRing<F, D>> = Vec::with_capacity(num_eval_rows);
    let mut cursor = 0usize;
    for &group_size in &claim_group_sizes {
        group_inner_reductions.push(&inner_reductions[cursor]);
        cursor += group_size;
    }
    for (point_idx, (y_ring, &batched_opening)) in y_rings
        .iter()
        .zip(batched_openings_per_point.iter())
        .enumerate()
    {
        let inner_reduction = group_inner_reductions[point_idx];
        let trace_lhs = trace::<F, { D }>(&(*y_ring * inner_reduction.sigma_m1()));
        let trace_rhs = d_field * batched_opening;
        if trace_lhs != trace_rhs {
            tracing::debug!(
                "[verify_one_level] trace check failed point_idx={point_idx} groups={claim_group_sizes:?} after {:?}",
                total_t.elapsed()
            );
            return Err(AkitaError::InvalidProof);
        }
    }
    tracing::debug!(
        "[verify_one_level] trace checks ok groups={claim_group_sizes:?} after {:?}",
        total_t.elapsed()
    );

    transcript.append_serde(
        akita_transcript::labels::ABSORB_PROVER_V,
        &akita_types::RingSliceSerializer(v_typed),
    );
    let total_stage1_blocks = batch_lp
        .group_layouts(&claim_group_sizes, num_eval_rows)?
        .last()
        .map(|layout| layout.block_start + layout.claim_count * layout.spec.num_blocks)
        .unwrap_or(0);
    // Mirror prover-side rounding for tensor stage-1 challenges
    // (see `crates/akita-prover/src/protocol/quadratic_equation.rs`).
    let (challenge_blocks, challenge_claims) = if batch_lp.groups_are_homogeneous() {
        (batch_lp.num_blocks, num_claims)
    } else {
        (total_stage1_blocks.next_power_of_two().max(1), 1)
    };
    let stage1_challenges = akita_challenges::sample_stage1_challenges::<F, T, D>(
        transcript,
        challenge_blocks,
        challenge_claims,
        &batch_lp.stage1_config,
        &batch_lp.stage1_challenge_shape,
    )?;

    let w_len = if is_last {
        final_w.map_or(0, DirectWitnessProof::num_elems)
    } else if num_claims == 1 {
        w_ring_element_count::<F>(&batch_lp) * D
    } else {
        w_ring_element_count_with_claim_groups::<F>(&batch_lp, &claim_group_sizes, num_eval_rows)
            * D
    };
    tracing::debug!(w_len, is_last, num_claims, "verify ring_switch");

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if num_claims == 1 {
        None
    } else {
        let mut rows = Vec::with_capacity(batch_lp.total_b_row_count(num_claims));
        for claim in claims.iter() {
            rows.extend_from_slice(claim.commitment.as_ring_slice::<D>()?);
        }
        Some(rows)
    };
    let commitment_u: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(rows) => rows.as_slice(),
        None => claims[0].commitment.as_ring_slice::<D>()?,
    };

    // Dedupe ring_opening_points per GROUP to mirror the prover's
    // `group_ring_opening_points` (book §5.4 tier-aware merge: grouped
    // claims share an opening point).
    let mut group_ring_opening_points: Vec<akita_types::RingOpeningPoint<F>> =
        Vec::with_capacity(num_eval_rows);
    let mut cursor = 0usize;
    for &group_size in &claim_group_sizes {
        group_ring_opening_points.push(ring_opening_points[cursor].clone());
        cursor += group_size;
    }
    let t = Instant::now();
    tracing::debug!("[verify_one_level] ring_switch_verifier start");
    let rs = ring_switch_verifier::<F, T, { D }>(
        &group_ring_opening_points,
        &claim_to_point,
        &stage1_challenges,
        w_len,
        level_proof.next_w_commitment(),
        transcript,
        &batch_lp,
        &claim_group_sizes,
        &gamma,
        num_eval_rows,
    )?;
    tracing::debug!(
        "[verify_one_level] ring_switch_verifier done after {:?} col_bits={} ring_bits={}",
        t.elapsed(),
        rs.col_bits,
        rs.ring_bits
    );
    let relation_claim = relation_claim_from_rows_with_layout(
        &rs.tau1,
        rs.alpha,
        v_typed,
        commitment_u,
        y_rings,
        &batch_lp.m_row_layout(claim_group_sizes.len(), num_eval_rows),
    );
    let stage1 = &level_proof.stage1;
    let stage2 = &level_proof.stage2;
    let tau0_reordered = reorder_stage1_coords(&rs.tau0, rs.col_bits, rs.ring_bits);
    let stage1_verifier = AkitaStage1Verifier::new(tau0_reordered, rs.b);
    let r_stage1 = {
        let t = Instant::now();
        tracing::debug!("[verify_one_level] stage1 verify start");
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let out = stage1_verifier.verify(stage1, transcript)?;
        tracing::debug!(
            "[verify_one_level] stage1 verify done after {:?}",
            t.elapsed()
        );
        out
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &stage1.s_claim);
    let gamma_range: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let gamma_rel: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH_REL);
    let stage2_input_claim = gamma_range * stage1.s_claim + gamma_rel * relation_claim;
    let m_eval_source = Stage2MEvalSource::new(rs.prepared_m_eval);

    let stage2_verifier = (if is_last {
        let fw = final_w.ok_or(AkitaError::InvalidProof)?;
        AkitaStage2Verifier::new_with_direct_witness(
            gamma_range,
            gamma_rel,
            stage1.s_claim,
            fw,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            &group_ring_opening_points,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_rings,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    } else {
        AkitaStage2Verifier::new_with_claimed_w_eval(
            gamma_range,
            gamma_rel,
            stage1.s_claim,
            stage2.next_w_eval,
            r_stage1.clone(),
            rs.alpha_evals_y,
            m_eval_source,
            &setup.expanded,
            &group_ring_opening_points,
            &rs.tau1,
            v_typed,
            commitment_u,
            y_rings,
            rs.alpha,
            rs.col_bits,
            rs.ring_bits,
        )
    })
    .with_relation_claim(relation_claim);
    if stage2_input_claim != SumcheckInstanceVerifier::input_claim(&stage2_verifier) {
        tracing::debug!(
            "[verify_one_level] stage2 input claim mismatch after {:?}",
            total_t.elapsed()
        );
        return Err(AkitaError::InvalidProof);
    }
    tracing::debug!(
        "[verify_one_level] stage2 input claim ok after {:?}",
        total_t.elapsed()
    );

    // S-1: schedule shape must equal proof shape at the stage-2 dispatch.
    // See `verify_root_level` for the contract; the rationale is
    // identical at every fold level. The level's `lp` is the source of
    // truth for whether the prover emits `setup_claim_reduction`.
    let lp_says_emit = lp.use_setup_claim_reduction;
    let proof_says_emit = stage2.setup_claim_reduction.is_some();
    if lp_says_emit != proof_says_emit {
        return Err(AkitaError::InvalidProof);
    }
    let mut deferred_setup_opening = None;
    let challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        if let Some(payload) = stage2.setup_claim_reduction.as_ref() {
            let t = Instant::now();
            tracing::debug!(
                "[verify_one_level] stage2/setup-claim verify start route_setup={routes_setup_recursively}"
            );
            // `routes_setup_recursively`: see the equivalent comment in
            // `verify_root_level`. Sourced from the planner's
            // `s_field_len_emitted` at this level.
            let (stage2_challenges, r_setup, s_opening_value) =
                verify_stage2_with_setup_claim_reduction::<F, _, D>(
                    &stage2.sumcheck,
                    payload,
                    &stage2_verifier,
                    transcript,
                    routes_setup_recursively,
                )?;
            tracing::debug!(
                "[verify_one_level] stage2/setup-claim verify done after {:?}",
                t.elapsed()
            );
            if routes_setup_recursively {
                let (row_bits, col_bits, coeff_bits) = stage2_verifier
                    .prepared_m_eval()
                    .setup_polynomial_padded_dims(setup.expanded.seed.max_stride);
                deferred_setup_opening = Some(DeferredSetupOpening {
                    r_setup,
                    row_bits,
                    col_bits,
                    coeff_bits,
                    opening: s_opening_value,
                    row_count: stage2_verifier
                        .prepared_m_eval()
                        .setup_polynomial_row_count(),
                    col_count: 1usize << col_bits,
                });
            }
            stage2_challenges
        } else {
            let t = Instant::now();
            tracing::debug!("[verify_one_level] stage2 verify start");
            let stage2_challenges = verify_sumcheck::<F, _, F, _, _>(
                &stage2.sumcheck,
                &stage2_verifier,
                transcript,
                |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
            )?;
            tracing::debug!(
                "[verify_one_level] stage2 verify done after {:?}",
                t.elapsed()
            );
            stage2_challenges
        }
    };

    tracing::debug!("[verify_one_level] done after {:?}", total_t.elapsed());
    Ok(VerifyLevelOutput {
        challenges,
        setup_opening: deferred_setup_opening,
    })
}

fn scheduled_recursive_verify_level<F: FieldCore>(
    schedule: &Schedule,
    level: usize,
    current_state: &RecursiveVerifierState<F>,
) -> Result<(LevelParams, usize, Option<LevelParams>), AkitaError> {
    let Some(Step::Fold(step)) = schedule.steps.get(level) else {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule is missing fold step at level {level}"
        )));
    };
    let claim = &current_state.claims[0];
    if step.current_w_len != claim.w_len || step.params.log_basis != claim.log_basis {
        return Err(AkitaError::InvalidSetup(
            "scheduled recursive level did not match runtime state".to_string(),
        ));
    }
    let next_level_params = match schedule.steps.get(level + 1) {
        Some(Step::Fold(next_step)) => Some(next_step.params.clone()),
        Some(Step::Direct(_)) => None,
        None => {
            return Err(AkitaError::InvalidSetup(
                "schedule is missing successor step".to_string(),
            ))
        }
    };
    Ok((step.params.clone(), step.next_w_len, next_level_params))
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn dispatch_verify_level<F, T>(
    level_d: usize,
    level_proof: &AkitaLevelProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    current_state: &RecursiveVerifierState<F>,
    is_last: bool,
    final_w: Option<&DirectWitnessProof<F>>,
    lp: &LevelParams,
    block_order: BlockOrder,
    routes_setup_recursively: bool,
) -> Result<VerifyLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    T: Transcript<F>,
{
    match level_d {
        32 => verify_one_level::<F, T, 32>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
            routes_setup_recursively,
        ),
        64 => verify_one_level::<F, T, 64>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
            routes_setup_recursively,
        ),
        128 => verify_one_level::<F, T, 128>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
            routes_setup_recursively,
        ),
        256 => verify_one_level::<F, T, 256>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
            routes_setup_recursively,
        ),
        512 => verify_one_level::<F, T, 512>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
            routes_setup_recursively,
        ),
        1024 => verify_one_level::<F, T, 1024>(
            level_proof,
            setup,
            transcript,
            current_state,
            is_last,
            final_w,
            lp,
            block_order,
            routes_setup_recursively,
        ),
        _ => Err(AkitaError::InvalidProof),
    }
}

/// Verify all recursive fold levels after the root proof.
///
/// The supplied `schedule` is the already-selected public schedule for this
/// proof shape. This function checks that each proof level matches that
/// schedule, dispatches to the corresponding ring dimension, and threads the
/// verifier state to the next recursive commitment.
///
/// # Errors
///
/// Returns an error if the schedule is malformed for the supplied proof,
/// decoded proof dimensions do not match, any fold-level verifier rejects, or
/// the recursive witness handoff has the wrong shape.
///
/// # Panics
///
/// Panics if the Phase 5 grouping accumulator's `last_mut()` returns
/// `None` after a merge — this is internally unreachable because the
/// merge branch only fires when the previous claim already pushed a
/// non-empty entry into `next_claim_group_sizes`.
pub fn verify_batched_recursive_suffix<F, T, const D: usize>(
    proof: &AkitaBatchedProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    schedule: &Schedule,
    mut current_state: RecursiveVerifierState<F>,
    final_w: Option<&DirectWitnessProof<F>>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + Send + Sync + 'static,
    T: Transcript<F>,
{
    let num_levels = proof.num_fold_levels();
    for (offset, level_proof) in proof.fold_levels().enumerate() {
        let level_index = offset + 1;
        let is_last = offset == num_levels - 1;
        let (current_lp, next_w_len, scheduled_next_params) =
            scheduled_recursive_verify_level(schedule, level_index, &current_state)?;
        let level_d = current_lp.ring_dimension;
        // Multi-ring shape check: the level proof's y_ring carries one
        // ring element per GROUP at this level (book §5.4 tier-aware
        // merge: chunks merged into one group share a y_ring). For
        // un-tiered runs every claim is its own group so this equals
        // `current_state.claims.len()`; for tiered runs the count
        // collapses to `num_groups < num_claims`. The number is computed
        // here speculatively from the same merge rule `verify_one_level`
        // applies, so the shape check rejects loudly if the proof's
        // y_ring count disagrees with the per-claim merge.
        let expected_num_y_rings =
            expected_num_groups_for_recursive(&current_state.claims, &current_lp);
        if !current_state
            .claims
            .iter()
            .all(|claim| claim.commitment.can_decode_vec(level_d))
            || !level_proof
                .y_ring
                .can_decode_count(level_d, expected_num_y_rings)
            || !level_proof.v.can_decode_vec(level_d)
        {
            return Err(AkitaError::InvalidProof);
        }

        let level_step = match schedule.steps.get(level_index) {
            Some(Step::Fold(step)) => step,
            _ => return Err(AkitaError::InvalidProof),
        };
        let routes_setup_recursively = level_step.s_field_len_emitted > 0;
        let verified = if level_d == D {
            verify_one_level::<F, T, D>(
                level_proof,
                setup,
                transcript,
                &current_state,
                is_last,
                if is_last { final_w } else { None },
                &current_lp,
                BlockOrder::ColumnMajor,
                routes_setup_recursively,
            )?
        } else {
            dispatch_verify_level::<F, T>(
                level_d,
                level_proof,
                setup,
                transcript,
                &current_state,
                is_last,
                if is_last { final_w } else { None },
                &current_lp,
                BlockOrder::ColumnMajor,
                routes_setup_recursively,
            )?
        };

        if !is_last {
            let scheduled_next_params = scheduled_next_params.ok_or(AkitaError::InvalidProof)?;
            let next_level_d = scheduled_next_params.ring_dimension;
            if next_level_d == 0 || !level_proof.next_w_commitment().can_decode_vec(next_level_d) {
                return Err(AkitaError::InvalidProof);
            }
            // Account for multi-claim w_ring sizing on cascade levels.
            // The runtime witness produced by this level has size
            // `w_ring(current_lp, num_claims_at_level) * level_d`. We
            // already know `num_claims_at_level == current_state.claims.len()`.
            let claim_lps: Vec<LevelParams> = current_state
                .claims
                .iter()
                .map(|claim| {
                    claim
                        .per_claim_lp
                        .clone()
                        .unwrap_or_else(|| current_lp.clone())
                })
                .collect();
            // Phase 5 grouping: same merge rule as `verify_one_level`.
            let mut batch_groups: Vec<GroupSpec> = Vec::new();
            let mut next_claim_group_sizes: Vec<usize> = Vec::new();
            let mut prev_lp: Option<&LevelParams> = None;
            let mut prev_tier: Option<TieredSetupParams> = None;
            for (claim_idx, lp_i) in claim_lps.iter().enumerate() {
                let cur_tier = current_state.claims[claim_idx].tier_marker;
                let mergeable = match (prev_lp, prev_tier, cur_tier) {
                    (Some(prev), Some(pt), Some(ct)) => prev == lp_i && pt == ct,
                    _ => false,
                };
                if mergeable {
                    *next_claim_group_sizes.last_mut().unwrap() += 1;
                } else {
                    let mut spec = GroupSpec::from_outer(lp_i);
                    spec.tier = cur_tier;
                    batch_groups.push(spec);
                    next_claim_group_sizes.push(1);
                    prev_lp = Some(lp_i);
                    prev_tier = cur_tier;
                }
            }
            let batch_lp = if expected_num_y_rings == 1 && claim_lps[0] == current_lp {
                current_lp.clone()
            } else {
                LevelParams {
                    groups: Some(batch_groups.clone()),
                    ..current_lp.clone()
                }
            };
            let computed_next_w_len = if expected_num_y_rings == 1 {
                w_ring_element_count::<F>(&batch_lp) * level_d
            } else {
                w_ring_element_count_with_claim_groups::<F>(
                    &batch_lp,
                    &next_claim_group_sizes,
                    expected_num_y_rings,
                ) * level_d
            };
            if computed_next_w_len != next_w_len {
                return Err(AkitaError::InvalidProof);
            }
            let mut claims = vec![RecursiveOpeningClaim {
                opening_point: verified.challenges,
                opening: level_proof.next_w_eval(),
                commitment: level_proof.next_w_commitment().clone(),
                basis: BasisMode::Lagrange,
                w_len: next_w_len,
                log_basis: scheduled_next_params.log_basis,
                per_claim_lp: None,
                tier_marker: None,
            }];
            if let Some(setup_opening) = verified.setup_opening {
                let setup_field_len = setup_opening.row_count * setup_opening.col_count * level_d;
                if level_step.tier_setup_params.is_tiered() {
                    expand_tiered_setup_claims::<F, D>(
                        setup,
                        &scheduled_next_params,
                        setup_opening.row_count,
                        setup_opening.col_count,
                        level_step.tier_setup_params,
                        &setup_opening.r_setup,
                        setup_opening.row_bits,
                        setup_opening.col_bits,
                        setup_opening.coeff_bits,
                        setup_opening.opening,
                        &mut claims,
                    )?;
                } else {
                    let s_lp = untiered_setup_group_lp(&scheduled_next_params, setup_field_len)?;
                    let opening_point = setup_opening_point_from_r_setup(
                        &setup_opening.r_setup,
                        setup_opening.row_bits,
                        setup_opening.col_bits,
                        setup_opening.coeff_bits,
                        &s_lp,
                    )?;
                    let commitment = derive_setup_commitment_flat::<F, D>(
                        setup,
                        setup_opening.row_count,
                        setup_opening.col_count,
                        &s_lp,
                    )?;
                    claims.push(RecursiveOpeningClaim {
                        opening_point,
                        opening: setup_opening.opening,
                        commitment,
                        basis: BasisMode::Lagrange,
                        w_len: setup_field_len,
                        log_basis: scheduled_next_params.log_basis,
                        per_claim_lp: Some(s_lp),
                        tier_marker: None,
                    });
                }
            }
            // S-5: when the schedule routes S recursively, the next-level
            // claim batch must contain MORE than the W claim alone (at
            // least one untiered S-claim entry, or tiered chunk + meta
            // entries). S-1 already binds the proof's
            // `setup_claim_reduction` presence to `routes_setup_recursively`,
            // so the only way to reach `claims.len() == 1` here is a bug
            // in the expansion logic — reject loudly.
            if routes_setup_recursively && claims.len() == 1 {
                return Err(AkitaError::InvalidProof);
            }
            current_state = RecursiveVerifierState { claims };
        }
    }

    Ok(())
}

/// Verify the folded-root branch of a batched opening proof.
///
/// The caller owns config-backed schedule selection and passes the derived
/// root verifier layout plus the first recursive-level params. This function
/// owns the fold-root proof-shape checks, root opening preparation, root
/// transcript replay, and recursive suffix handoff.
///
/// # Errors
///
/// Returns an error if the proof is not a folded-root proof, the schedule does
/// not match the proof shape, the root proof rejects, or a recursive suffix
/// level rejects.
#[allow(clippy::too_many_arguments)]
pub fn verify_fold_batched_proof<F, T, const D: usize>(
    proof: &AkitaBatchedProof<F>,
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    opening_points: &[&[F]],
    openings: &[F],
    commitments: &[RingCommitment<F, D>],
    batch_shape: &MultiPointBatchShape,
    basis: BasisMode,
    schedule: &Schedule,
    root_lp: &LevelParams,
    next_level_params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + FromPrimitiveInt + Send + Sync + 'static,
    T: Transcript<F>,
{
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidProof);
    };
    let fold_root = proof.root.as_fold().ok_or(AkitaError::InvalidProof)?;
    let expected_recursive_levels = schedule_num_fold_levels(schedule)
        .checked_sub(1)
        .ok_or(AkitaError::InvalidProof)?;
    if proof.num_fold_levels() != expected_recursive_levels {
        return Err(AkitaError::InvalidProof);
    }

    let y_coeff_len = fold_root.y_rings.coeff_len();
    if !y_coeff_len.is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    // One public y-ring per distinct opening point.
    if y_coeff_len / D != opening_points.len() {
        return Err(AkitaError::InvalidProof);
    }

    let final_w = proof
        .steps
        .last()
        .and_then(AkitaProofStep::as_direct)
        .ok_or(AkitaError::InvalidProof)?;
    let final_w = Some(final_w);
    let alpha_bits = root_lp.ring_dimension.trailing_zeros() as usize;
    let prepared_points = opening_points
        .iter()
        .map(|opening_point| {
            prepare_root_opening_point::<F, D>(opening_point, basis, root_lp, alpha_bits)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| AkitaError::InvalidProof)?;

    let has_recursive_levels = proof.num_fold_levels() > 0;
    let root_routes_setup_recursively = root_step.s_field_len_emitted > 0;
    let root_verified = verify_root_level::<F, T, D>(
        &fold_root.y_rings,
        &fold_root.v,
        &fold_root.stage1,
        &fold_root.stage2,
        setup,
        transcript,
        &prepared_points,
        openings,
        commitments,
        batch_shape,
        root_lp,
        &root_step.params,
        !has_recursive_levels,
        if has_recursive_levels { None } else { final_w },
        root_routes_setup_recursively,
    )?;

    if has_recursive_levels {
        let first_level_d = next_level_params.ring_dimension;
        if !fold_root
            .stage2
            .next_w_commitment
            .can_decode_vec(first_level_d)
        {
            return Err(AkitaError::InvalidProof);
        }

        let mut claims = vec![RecursiveOpeningClaim {
            opening_point: root_verified.challenges,
            opening: fold_root.stage2.next_w_eval,
            commitment: fold_root.stage2.next_w_commitment.clone(),
            basis: BasisMode::Lagrange,
            w_len: root_step.next_w_len,
            log_basis: next_level_params.log_basis,
            per_claim_lp: None,
            tier_marker: None,
        }];
        if let Some(setup_opening) = root_verified.setup_opening {
            let setup_field_len = setup_opening.row_count * setup_opening.col_count * D;
            if root_step.tier_setup_params.is_tiered() {
                expand_tiered_setup_claims::<F, D>(
                    setup,
                    next_level_params,
                    setup_opening.row_count,
                    setup_opening.col_count,
                    root_step.tier_setup_params,
                    &setup_opening.r_setup,
                    setup_opening.row_bits,
                    setup_opening.col_bits,
                    setup_opening.coeff_bits,
                    setup_opening.opening,
                    &mut claims,
                )?;
            } else {
                let s_lp = untiered_setup_group_lp(next_level_params, setup_field_len)?;
                let opening_point = setup_opening_point_from_r_setup(
                    &setup_opening.r_setup,
                    setup_opening.row_bits,
                    setup_opening.col_bits,
                    setup_opening.coeff_bits,
                    &s_lp,
                )?;
                let commitment = derive_setup_commitment_flat::<F, D>(
                    setup,
                    setup_opening.row_count,
                    setup_opening.col_count,
                    &s_lp,
                )?;
                claims.push(RecursiveOpeningClaim {
                    opening_point,
                    opening: setup_opening.opening,
                    commitment,
                    basis: BasisMode::Lagrange,
                    w_len: setup_field_len,
                    log_basis: next_level_params.log_basis,
                    per_claim_lp: Some(s_lp),
                    tier_marker: None,
                });
            }
        }
        // S-5: see the equivalent comment in `verify_batched_recursive_suffix`.
        // When the schedule routes S recursively the first recursive level
        // must receive at least one S-claim alongside the W-claim.
        if root_routes_setup_recursively && claims.len() == 1 {
            return Err(AkitaError::InvalidProof);
        }
        let current_state = RecursiveVerifierState { claims };
        verify_batched_recursive_suffix::<F, T, D>(
            proof,
            setup,
            transcript,
            schedule,
            current_state,
            final_w,
        )?;
    }

    Ok(())
}
