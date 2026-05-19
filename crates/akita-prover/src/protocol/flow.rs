//! Prover flow state shared by root orchestration during crate extraction.

use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::{mat_vec_mul_ntt_i8_dense, mat_vec_mul_ntt_single_i8};
use crate::protocol::ring_switch::{
    ring_switch_build_w, ring_switch_finalize, ring_switch_finalize_with_claim_groups,
    RingSwitchOutput,
};
use crate::protocol::setup_claim_reduction::prove_setup_claim_reduction;
use crate::protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover};
use crate::{
    AkitaPolyOps, DensePoly, MultiDNttCaches, ProverClaims, QuadraticEquation,
    RecursiveCommitmentHintCache, RecursiveWitnessFlat, RecursiveWitnessView,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_field::fields::wide::HasWide;
use akita_field::fields::HasUnreducedOps;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField, RandomSampling};
use akita_sumcheck::{multilinear_eval, prove_sumcheck, SumcheckProof};
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD,
    ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_EVAL_BATCH, CHALLENGE_SUMCHECK_BATCH,
    CHALLENGE_SUMCHECK_BATCH_REL, CHALLENGE_SUMCHECK_ROUND, CHALLENGE_TIERED_CHUNK_AGGREGATION,
};
use akita_transcript::Transcript;
use akita_types::{
    append_batch_shape_to_transcript, append_batched_commitments_to_transcript,
    checked_total_claims, flatten_batched_commitment_rows, prepare_root_opening_point,
    reduce_inner_opening_to_ring_element, relation_claim_from_rows_with_layout,
    reorder_stage1_coords, ring_opening_point_from_field, schedule_is_root_direct,
    schedule_num_fold_levels, tiered_setup_chunk_index_map, tiered_setup_chunk_opening_point,
    tiered_setup_group_lp, untiered_setup_group_lp, validate_batched_inputs, AkitaBatchedProof,
    AkitaBatchedRootProof, AkitaCommitmentHint, AkitaExpandedSetup, AkitaLevelProof,
    AkitaProofStep, AkitaRootBatchSummary, AkitaScheduleInputs, AkitaScheduleLookupKey,
    AkitaStage1Proof, BasisMode, BlockOrder, DirectWitnessProof, FlatDigitBlocks, FlatRingVec,
    GroupSpec, LevelParams, MultiPointBatchShape, PackedDigits, PreparedRootOpeningPoint,
    RingCommitment, Schedule, SetupClaimReductionPayload, Step, TieredSetupParams,
};
use akita_verifier::{materialize_setup_claim_polynomial, prepare_m_eval};
use std::time::Instant;

/// Prover-side handle for one polynomial whose recursive opening the
/// next fold level must serve.
///
/// Mirrors the verifier's `RecursiveOpeningClaim`: `w`/`commitment`/
/// `hint` together materialize the next-level proof of the opening at
/// `opening_point`. `opening_point` is the stage-2 sumcheck challenge
/// vector produced at the level that emitted this handle.
///
/// `per_handle_lp` is the optional per-handle [`LevelParams`] override
/// that the multi-group batched Hachi commit at the next level
/// consumes (book §5.3 lines 643–660). `None` inherits the level's
/// shared LP; `Some(lp)` carries this handle's per-commitment-group
/// `(m, r, B, digit_count)`. Heterogeneous per-handle LPs are collapsed
/// into [`LevelParams::groups`] by
/// [`prove_recursive_multi_fold_with_params`] and dispatched through
/// the multi-group commit kernel.
///
/// `tiered` carries the book §5.4 routed tiered S material. When
/// `Some(_)`, this handle replaces a dense `S` handle: the next fold
/// expands the tiered material into `k + 1` claims (k chunks under
/// shared `chunk_lp` + 1 meta under `meta_lp`) before invoking the
/// multi-claim path. Mutually exclusive with `dense_poly`: a routed
/// setup handle is either the un-tiered (`f = 1`) dense form or the
/// tiered (`f > 1`) chunked + meta form.
pub struct RecursivePolyHandle<F: FieldCore> {
    /// Recursive witness whose opening will be proved at the next level.
    pub w: RecursiveWitnessFlat,
    /// Field-polynomial payload for setup-side recursive openings.
    ///
    /// `None` is the ordinary recursive `w` handle. `Some(poly)` is the
    /// un-tiered setup polynomial `S` entering the next fold unfolded, as in
    /// book §5.3 lines 627-660.
    pub dense_poly: Option<FlatRingVec<F>>,
    /// Commitment to the recursive witness.
    pub commitment: FlatRingVec<F>,
    /// D-erased recursive commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
    /// Digit basis for `w`, as `log2(b)`.
    pub log_basis: u32,
    /// Opening point at which the next level evaluates this commitment.
    pub opening_point: Vec<F>,
    /// Optional per-handle [`LevelParams`] override (see struct docs).
    pub per_handle_lp: Option<LevelParams>,
    /// Tiered routed-S material (book §5.4). `None` for ordinary handles
    /// and the un-tiered (`f = 1`) S handle path; `Some(_)` activates the
    /// chunked + meta multi-claim expansion at the next level.
    pub tiered: Option<TieredHandleMaterial<F>>,
}

/// Routed tiered-S material for the next fold level (book §5.4).
///
/// Carries everything the next level needs to expand the logical
/// `S(r_i, r_x, r_k) = y_setup` claim into `k + 1` commitment groups under
/// shared per-chunk parameters plus the tier-3 meta tier. `chunk_polys`
/// and `meta_input_poly` carry the polynomial coefficient material;
/// `chunk_commitments` and `meta_commitment` are the precomputable
/// B-side outputs (book line 951: "the matrix commitment
/// `B_S t̂_S = u_S` can be precomputed during setup"); `chunk_hints`
/// and `meta_hint` carry the prover-only digit decompositions and
/// recomposed t-rows that the next level's `prove_recursive_multi_fold`
/// step consumes per `commit_inner_witness`. `chunk_lp` and `meta_lp`
/// are the per-claim LPs the multi-group commit binds them under;
/// `tier` records `(f, k)`; `opening_point` is the projected setup
/// opening point that all chunks share.
pub struct TieredHandleMaterial<F: FieldCore> {
    /// Per-chunk polynomials, length `tier.num_chunks`.
    pub chunk_polys: Vec<FlatRingVec<F>>,
    /// Per-chunk B-side commitments `u_{S,j}`.
    pub chunk_commitments: Vec<FlatRingVec<F>>,
    /// Per-chunk recursive hint caches.
    pub chunk_hints: Vec<RecursiveCommitmentHintCache<F>>,
    /// Meta-tier input polynomial built from concatenated chunk commits.
    pub meta_input_poly: FlatRingVec<F>,
    /// Meta-tier B-side commitment `u_meta`.
    pub meta_commitment: FlatRingVec<F>,
    /// Meta-tier recursive hint cache.
    pub meta_hint: RecursiveCommitmentHintCache<F>,
    /// Per-chunk LP under which each chunk commits.
    pub chunk_lp: LevelParams,
    /// Meta-tier LP under which the meta polynomial commits.
    pub meta_lp: LevelParams,
    /// Tiered shape `(f, k)`.
    pub tier: TieredSetupParams,
    /// Opening point shared across the chunk + meta claims.
    pub opening_point: Vec<F>,
}

/// Runtime state carried between recursive prove levels.
///
/// Each entry of `handles` is one polynomial whose opening must be
/// proved at the next fold level. The single-poly recursive path uses
/// `handles.len() == 1`; Phase D-full slice F adds an additional handle
/// for the shared setup polynomial `S` so the next level discharges
/// the deferred `S(r_i, r_x, r_k) = y_setup` claim alongside the folded
/// witness via multi-claim batched Hachi.
pub struct RecursiveProverState<F: FieldCore> {
    /// Per-polynomial handles to discharge at the next fold level.
    pub handles: Vec<RecursivePolyHandle<F>>,
}

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: FieldCore> {
    /// Fold proof produced at this level.
    pub level_proof: AkitaLevelProof<F>,
    /// Recursive prover state for the next level.
    pub next_state: RecursiveProverState<F>,
}

/// Typed polynomial payload consumed by a recursive multi-claim fold.
pub enum RecursiveHandlePoly<'a, F: FieldCore, const D: usize> {
    /// Existing recursive digit witness.
    Witness(RecursiveWitnessView<'a, F, D>),
    /// Fresh field-element polynomial, used by the un-tiered S group.
    Dense(DensePoly<F, D>),
}

impl<'a, F, const D: usize> RecursiveHandlePoly<'a, F, D>
where
    F: FieldCore + CanonicalField,
{
    pub(crate) fn num_ring_elems(&self) -> usize {
        match self {
            Self::Witness(w) => w.num_ring_elems(),
            Self::Dense(poly) => poly.num_ring_elems(),
        }
    }

    pub(crate) fn evaluate_and_fold(
        &self,
        eval_outer_scalars: &[F],
        fold_scalars: &[F],
        block_len: usize,
        num_blocks: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        match self {
            Self::Witness(w) => {
                w.evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len, num_blocks)
            }
            Self::Dense(poly) => {
                poly.evaluate_and_fold(eval_outer_scalars, fold_scalars, block_len)
            }
        }
    }

    pub(crate) fn decompose_fold_integer(
        &self,
        challenges: &[akita_challenges::IntegerChallenge],
        block_len: usize,
        num_blocks: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<crate::DecomposeFoldWitness<F, D>, AkitaError> {
        match self {
            Self::Witness(w) => Ok(
                w.decompose_fold_integer(challenges, block_len, num_blocks, num_digits, log_basis)
            ),
            Self::Dense(poly) => {
                poly.decompose_fold_integer(challenges, block_len, num_digits, log_basis)
            }
        }
    }
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
        return Err(AkitaError::InvalidSetup(format!(
            "setup route point length mismatch: expected {expected}, actual {}, row_bits={row_bits}, col_bits={col_bits}, coeff_bits={coeff_bits}",
            r_setup.len()
        )));
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
    let mut point = Vec::with_capacity(coeff_bits + setup_lp.r_vars + setup_lp.m_vars);
    point.extend_from_slice(coeffs);
    point.extend_from_slice(block_bits);
    point.extend_from_slice(elem_bits);
    Ok(point)
}

fn setup_claim_opening_point_from_r_setup<F: FieldCore>(
    r_setup: &[F],
    row_bits: usize,
    coeff_bits: usize,
    setup_lp: &LevelParams,
) -> Result<Vec<F>, AkitaError> {
    setup_opening_point_from_r_setup(r_setup, row_bits, 0, coeff_bits, setup_lp)
}

fn commit_dense_s_handle_direct<F, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    coeffs: &[CyclotomicRing<F, D>],
    s_lp: &LevelParams,
) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let stride = expanded.seed.max_stride.max(1);
    let a_view = expanded
        .shared_matrix
        .ring_view::<D>(s_lp.a_key.row_len(), stride);
    let b_view = expanded
        .shared_matrix
        .ring_view::<D>(s_lp.b_key.row_len(), stride);
    let q = (-F::one()).to_canonical_u128() + 1;
    let commit_params =
        BalancedDecomposePow2I8Params::new(s_lp.num_digits_commit, s_lp.log_basis, q);
    let open_params = BalancedDecomposePow2I8Params::new(s_lp.num_digits_open, s_lp.log_basis, q);
    let t_rows: Vec<Vec<CyclotomicRing<F, D>>> = cfg_into_iter!(0..s_lp.num_blocks)
        .map(|block_idx| {
            let mut t_for_block = vec![CyclotomicRing::<F, D>::zero(); s_lp.a_key.row_len()];
            for elem_idx in 0..s_lp.block_len {
                let global = block_idx * s_lp.block_len + elem_idx;
                let Some(value) = coeffs.get(global) else {
                    continue;
                };
                let mut digits = vec![[0i8; D]; s_lp.num_digits_commit];
                value.balanced_decompose_pow2_i8_into_with_params(&mut digits, &commit_params);
                for (digit_idx, digit) in digits.iter().enumerate() {
                    let digit_ring = CyclotomicRing::from_coefficients(digit.map(F::from_i8));
                    let matrix_col = elem_idx * s_lp.num_digits_commit + digit_idx;
                    for (a_idx, acc) in t_for_block.iter_mut().enumerate() {
                        *acc += a_view.row(a_idx)[matrix_col] * digit_ring;
                    }
                }
            }
            t_for_block
        })
        .collect();

    let mut t_hat_blocks = Vec::with_capacity(s_lp.num_blocks);
    let mut u = vec![CyclotomicRing::<F, D>::zero(); s_lp.b_key.row_len()];
    for (block_idx, t_for_block) in t_rows.iter().enumerate() {
        let mut block_digits = Vec::with_capacity(s_lp.a_key.row_len() * s_lp.num_digits_open);
        for (a_idx, t) in t_for_block.iter().enumerate() {
            let mut digits = vec![[0i8; D]; s_lp.num_digits_open];
            t.balanced_decompose_pow2_i8_into_with_params(&mut digits, &open_params);
            for (digit_idx, digit) in digits.iter().enumerate() {
                block_digits.push(*digit);
                let digit_ring = CyclotomicRing::from_coefficients(digit.map(F::from_i8));
                let matrix_col = block_idx * s_lp.a_key.row_len() * s_lp.num_digits_open
                    + a_idx * s_lp.num_digits_open
                    + digit_idx;
                for (b_idx, acc) in u.iter_mut().enumerate() {
                    *acc += b_view.row(b_idx)[matrix_col] * digit_ring;
                }
            }
        }
        t_hat_blocks.push(block_digits);
    }
    let t_hat = FlatDigitBlocks::from_blocks(t_hat_blocks);
    let hint = AkitaCommitmentHint::singleton_with_t(t_hat, t_rows);
    Ok((
        FlatRingVec::from_ring_elems::<D>(&u),
        RecursiveCommitmentHintCache::from_typed(hint)?,
    ))
}

/// Build the `TieredHandleMaterial` for the routed S claim of a level
/// whose `tier.is_tiered()` is true.
///
/// Reads `setup_field_len / D` ring elements of the shared matrix in
/// row-major order (matching the un-tiered S handle's `(row_count,
/// col_count)` view) and partitions them into `tier.num_chunks` equal
/// chunks of `chunk_n = setup_field_len / D / k` ring elements each.
/// Commits each chunk under `chunk_lp` and the meta-tier polynomial
/// (concatenated chunk B-side commitments padded to a power of two)
/// under `meta_lp`.
///
/// The verifier mirror in `derive_tiered_setup_material_for_verifier`
/// must use the same partition rule.
///
/// `opening_point` is the routed setup-claim opening point, shared
/// across the chunk + meta sub-claims at the next fold level.
#[allow(clippy::too_many_arguments)]
/// In-flight tiered routed-S material for the recursive prover state.
///
/// Shares its chunk-index map and meta-concat recipe with
/// [`crate::api::tiered_setup::derive_tiered_setup_handle_bundle`].
/// The two callers consume different IO shapes (this one reads a
/// multi-row `ring_view(row_count, max_stride)` and emits in-flight
/// `TieredHandleMaterial` with `(chunk_lp, meta_lp, tier,
/// opening_point)` fields; the setup-time variant reads
/// `ring_view(1, n_s)` and emits the verifier-derivable
/// [`akita_types::TieredSetupCommitments`]) and commit primitives
/// (`commit_dense_s_handle_direct` here vs the high-level
/// `commit_with_params` there). Per `.cursor/rules/code_changes.mdc`
/// "smallest coherent change", the shared abstraction would add a
/// commit closure + output builder layer larger than the duplication
/// it saves; keep both. Any future change to the chunk partition rule,
/// meta packing, or per-chunk LP shape must update both call sites in
/// lockstep (book §5.4 lines 686-754).
fn build_tiered_handle_material<F, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    row_count: usize,
    col_count: usize,
    chunk_lp: &LevelParams,
    meta_lp: &LevelParams,
    tier: TieredSetupParams,
    opening_point: Vec<F>,
    setup_rings_override: Option<&[CyclotomicRing<F, D>]>,
) -> Result<TieredHandleMaterial<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let total_t = Instant::now();
    tracing::debug!(
        "[tiered_material] start D={D} row_count={row_count} col_count={col_count} chunks={} chunk_blocks={} chunk_block_len={} meta_blocks={} meta_block_len={}",
        tier.num_chunks,
        chunk_lp.num_blocks,
        chunk_lp.block_len,
        meta_lp.num_blocks,
        meta_lp.block_len
    );
    tracing::debug!(
        "[tiered_material] chunk digits commit={} open={} fold={} a_width={} b_width={} setup_stride={}",
        chunk_lp.num_digits_commit,
        chunk_lp.num_digits_open,
        chunk_lp.num_digits_fold,
        chunk_lp.a_key.col_len(),
        chunk_lp.b_key.col_len(),
        expanded.seed.max_stride
    );
    if tier.num_chunks == 0 {
        return Err(AkitaError::InvalidInput(
            "tiered routing requires num_chunks >= 1".to_string(),
        ));
    }
    let live_s_rings = row_count
        .checked_mul(col_count)
        .ok_or_else(|| AkitaError::InvalidSetup("tiered S size overflow".to_string()))?;
    let s_total_rings = live_s_rings.next_power_of_two();
    if !s_total_rings.is_multiple_of(tier.num_chunks) {
        return Err(AkitaError::InvalidInput(format!(
            "padded routed S has {s_total_rings} ring elements but tier num_chunks = {} requires a multiple",
            tier.num_chunks
        )));
    }
    let chunk_n = s_total_rings / tier.num_chunks;
    if chunk_n == 0 || !chunk_n.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "tiered chunk size {chunk_n} must be a non-zero power of two"
        )));
    }
    let log_shrink = tier.log2_shrink()? as usize;
    let full_r_vars = chunk_lp.r_vars + log_shrink;
    let full_m_vars = chunk_lp.m_vars + log_shrink;
    let chunk_indices = tiered_setup_chunk_index_map(full_r_vars, full_m_vars, tier)?;

    // Row-major linearization of the routed setup polynomial. The
    // setup-claim reducer routes the `r_x`-fixed derived polynomial
    // (`setup_rings_override`); older/raw setup utilities use the
    // shared-matrix view directly.
    let stride = expanded.seed.max_stride.max(1);
    let mut s_rings: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(s_total_rings);
    if let Some(rings) = setup_rings_override {
        if rings.len() != live_s_rings {
            return Err(AkitaError::InvalidSize {
                expected: live_s_rings,
                actual: rings.len(),
            });
        }
        s_rings.extend_from_slice(rings);
    } else {
        let view = expanded.shared_matrix.ring_view::<D>(row_count, stride);
        for row in 0..row_count {
            s_rings.extend_from_slice(&view.row(row)[..col_count]);
        }
    }
    s_rings.resize(s_total_rings, CyclotomicRing::<F, D>::zero());

    // Materialize the k chunk inputs once, then commit them under the
    // SHARED `chunk_lp` via a single batched A-step over
    // `k * num_blocks_chunk` blocks (book §5.4 lines 686-700: chunks
    // share the per-chunk `(D, A, B)` matrices). The per-chunk
    // sequential `commit_dense_s_handle_direct` loop would otherwise
    // pay k× the NTT-cache load and column-tile reduce overhead.
    let chunk_ring_inputs: Vec<Vec<CyclotomicRing<F, D>>> = chunk_indices
        .iter()
        .map(|indices| indices.iter().map(|&idx| s_rings[idx]).collect::<Vec<_>>())
        .collect();
    let chunk_polys: Vec<FlatRingVec<F>> = chunk_ring_inputs
        .iter()
        .map(|chunk| FlatRingVec::from_ring_elems::<D>(chunk))
        .collect();

    let n_a = chunk_lp.a_key.row_len();
    let n_b = chunk_lp.b_key.row_len();
    let mut block_slices: Vec<&[CyclotomicRing<F, D>]> =
        Vec::with_capacity(tier.num_chunks * chunk_lp.num_blocks);
    for chunk_input in &chunk_ring_inputs {
        for block_idx in 0..chunk_lp.num_blocks {
            let start = block_idx * chunk_lp.block_len;
            let slice: &[CyclotomicRing<F, D>] = if start >= chunk_input.len() {
                &[]
            } else {
                let end = (start + chunk_lp.block_len).min(chunk_input.len());
                &chunk_input[start..end]
            };
            block_slices.push(slice);
        }
    }
    let a_step_t = Instant::now();
    let t_rows_all = mat_vec_mul_ntt_i8_dense(
        ntt_shared,
        n_a,
        stride,
        &block_slices,
        chunk_lp.num_digits_commit,
        chunk_lp.log_basis,
    );
    tracing::debug!(
        "[tiered_material] batched A-step over {} blocks done after {:?}",
        block_slices.len(),
        a_step_t.elapsed()
    );

    let q = (-F::one()).to_canonical_u128() + 1;
    let open_params =
        BalancedDecomposePow2I8Params::new(chunk_lp.num_digits_open, chunk_lp.log_basis, q);
    let b_step_t = Instant::now();
    let chunk_outputs: Vec<(Vec<CyclotomicRing<F, D>>, RecursiveCommitmentHintCache<F>)> =
        cfg_into_iter!(0..tier.num_chunks)
            .map(|chunk_idx| -> Result<_, AkitaError> {
                let t_start = chunk_idx * chunk_lp.num_blocks;
                let t_for_chunk = &t_rows_all[t_start..t_start + chunk_lp.num_blocks];
                let mut t_hat_blocks: Vec<Vec<[i8; D]>> = Vec::with_capacity(chunk_lp.num_blocks);
                let mut flat_digits =
                    Vec::with_capacity(chunk_lp.num_blocks * n_a * chunk_lp.num_digits_open);
                for t_for_block in t_for_chunk {
                    let mut block_digits = vec![[0i8; D]; n_a * chunk_lp.num_digits_open];
                    for (a_idx, t) in t_for_block.iter().enumerate() {
                        let dst = &mut block_digits[a_idx * chunk_lp.num_digits_open
                            ..(a_idx + 1) * chunk_lp.num_digits_open];
                        t.balanced_decompose_pow2_i8_into_with_params(dst, &open_params);
                    }
                    flat_digits.extend_from_slice(&block_digits);
                    t_hat_blocks.push(block_digits);
                }
                let u = mat_vec_mul_ntt_single_i8(ntt_shared, n_b, stride, &flat_digits);
                let t_hat = FlatDigitBlocks::from_blocks(t_hat_blocks);
                let hint = AkitaCommitmentHint::singleton_with_t(t_hat, t_for_chunk.to_vec());
                let cache = RecursiveCommitmentHintCache::from_typed(hint)?;
                Ok((u, cache))
            })
            .collect::<Result<Vec<_>, _>>()?;
    tracing::debug!(
        "[tiered_material] per-chunk B-step (×{}) done after {:?}",
        tier.num_chunks,
        b_step_t.elapsed()
    );

    let mut chunk_commitments: Vec<FlatRingVec<F>> = Vec::with_capacity(tier.num_chunks);
    let mut chunk_commitments_typed: Vec<Vec<CyclotomicRing<F, D>>> =
        Vec::with_capacity(tier.num_chunks);
    let mut chunk_hints: Vec<RecursiveCommitmentHintCache<F>> = Vec::with_capacity(tier.num_chunks);
    for (u, hint) in chunk_outputs {
        chunk_commitments.push(FlatRingVec::from_ring_elems::<D>(&u));
        chunk_commitments_typed.push(u);
        chunk_hints.push(hint);
    }

    let meta_len = chunk_commitments_typed.iter().map(Vec::len).sum::<usize>();
    if meta_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "tiered meta commitment input is empty".to_string(),
        ));
    }
    let next_pow2 = meta_len.next_power_of_two();
    let mut meta_input: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(next_pow2);
    for chunk in &chunk_commitments_typed {
        meta_input.extend_from_slice(chunk);
    }
    meta_input.resize(next_pow2, CyclotomicRing::<F, D>::zero());

    let t = Instant::now();
    tracing::debug!(
        "[tiered_material] meta commit start (meta_len={} padded={})",
        meta_len,
        next_pow2
    );
    let meta_input_poly = FlatRingVec::from_ring_elems::<D>(&meta_input);
    let (meta_commitment, meta_hint) =
        commit_dense_s_handle_direct::<F, D>(expanded, &meta_input, meta_lp)?;
    tracing::debug!(
        "[tiered_material] meta commit done after {:?}; total {:?}",
        t.elapsed(),
        total_t.elapsed()
    );
    Ok(TieredHandleMaterial {
        chunk_polys,
        chunk_commitments,
        chunk_hints,
        meta_input_poly,
        meta_commitment,
        meta_hint,
        chunk_lp: chunk_lp.clone(),
        meta_lp: meta_lp.clone(),
        tier,
        opening_point,
    })
}

/// Pick the meta-tier `LevelParams` from the per-chunk `LevelParams` and
/// the tier shape. The meta-tier polynomial is the concatenation of the
/// `k` per-chunk B-side commitments, padded to a power of two; its
/// length in ring elements is `k * n_B_chunk`. The meta-tier LP carries
/// shape suitable for committing that many ring elements under the
/// outer level's shared `(D, A)`.
fn meta_lp_from_chunks(
    next_level: &LevelParams,
    chunk_lp: &LevelParams,
    tier: TieredSetupParams,
) -> Result<LevelParams, AkitaError> {
    let meta_field_len = tier.num_chunks * chunk_lp.b_key.row_len() * next_level.ring_dimension;
    let next_pow2 = meta_field_len.next_power_of_two();
    let inherited = untiered_setup_group_lp(next_level, next_pow2)?;
    akita_types::derive_chunk_sis_ranks_from_widths(inherited)
}

/// Raw pieces produced by the unified root-level prover.
///
/// Callers assemble either a singleton or batched root proof from these
/// components while sharing the same inner prover flow.
pub struct RootLevelRawOutput<F: FieldCore, const D: usize> {
    /// Gamma-combined public y-rings, one per opening point.
    pub y_rings: Vec<CyclotomicRing<F, D>>,
    /// Public v rows for the root relation.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Stage-1 sumcheck proof.
    pub stage1: AkitaStage1Proof<F>,
    /// Stage-2 sumcheck proof.
    pub stage2_sumcheck: SumcheckProof<F>,
    /// Optional setup-side claim-reduction payload appended after stage 2.
    pub stage2_setup_claim_reduction: Option<SetupClaimReductionPayload<F>>,
    /// Recursive witness commitment carried in the proof.
    pub w_commitment_proof: FlatRingVec<F>,
    /// Claimed terminal evaluation of the recursive witness at this level.
    pub w_eval: F,
    /// Recursive prover state for the first suffix level.
    pub next_state: RecursiveProverState<F>,
}

/// Outcome of the recursive fold suffix after the root level.
pub struct RecursiveSuffixOutcome<F: FieldCore> {
    /// Per-level fold proofs, in order. Does not include the root proof.
    pub levels: Vec<AkitaLevelProof<F>>,
    /// Total fold-level count reached, including the root level.
    pub num_levels: usize,
    /// Prover state at the terminal direct step.
    pub final_state: RecursiveProverState<F>,
    /// `log_basis` for the terminal packed-digit witness.
    pub final_log_basis: u32,
}

/// Config-free flattened view of batched prover claims.
pub struct PreparedBatchedProveInputs<'a, F: FieldCore, P, const D: usize> {
    /// Distinct opening points in caller order.
    pub opening_points: Vec<&'a [F]>,
    /// Commitments flattened in point/group order.
    pub commitments_by_point: Vec<RingCommitment<F, D>>,
    /// Multipoint batch shape derived from the claims.
    pub batch_shape: MultiPointBatchShape,
    /// Total claim count used by schedule/layout lookup.
    pub layout_num_claims: usize,
    /// Number of variables in every opened polynomial.
    pub num_vars: usize,
    /// Polynomials flattened in claim order.
    pub flat_polys: Vec<&'a P>,
    /// Commitment hints flattened in claim-group order.
    pub flat_hints: Vec<AkitaCommitmentHint<F, D>>,
}

/// Pick the `log_basis` for the terminal packed-digit witness.
///
/// The planner's final direct step is authoritative and must match the
/// runtime recursive state.
///
/// # Errors
///
/// Returns an error if the schedule does not terminate in a direct step or if
/// the terminal direct step does not match the runtime witness length/basis.
pub fn resolve_final_log_basis<F>(
    schedule: &Schedule,
    current_state: &RecursiveProverState<F>,
) -> Result<u32, AkitaError>
where
    F: FieldCore,
{
    let Some(Step::Direct(direct_step)) = schedule.steps.last() else {
        return Err(AkitaError::InvalidSetup(
            "schedule must terminate in a direct step".to_string(),
        ));
    };
    let handle = &current_state.handles[0];
    if direct_step.current_w_len != handle.w.len() || direct_step.bits_per_elem != handle.log_basis
    {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled direct step did not match final runtime state: direct.current_w_len={}, handle.w.len()={}, direct.bits={}, handle.log_basis={}",
            direct_step.current_w_len,
            handle.w.len(),
            direct_step.bits_per_elem,
            handle.log_basis
        )));
    }
    Ok(direct_step.bits_per_elem)
}

/// Assemble fold-level proofs followed by the terminal packed-digit witness.
pub fn build_final_proof_steps<F>(
    levels: Vec<AkitaLevelProof<F>>,
    final_state: &RecursiveProverState<F>,
    final_log_basis: u32,
) -> Vec<AkitaProofStep<F>>
where
    F: FieldCore,
{
    let final_handle = &final_state.handles[0];
    let final_w =
        PackedDigits::from_i8_digits_with_min_bits(final_handle.w.as_i8_digits(), final_log_basis);
    let mut steps = levels
        .into_iter()
        .map(AkitaProofStep::Fold)
        .collect::<Vec<_>>();
    steps.push(AkitaProofStep::Direct(DirectWitnessProof::PackedDigits(
        final_w,
    )));
    steps
}

/// Validate and flatten batched prover claims into the root proof shape.
///
/// # Errors
///
/// Returns an error if the claim shape exceeds setup capacity, mixes
/// incompatible dimensions, or has malformed batch counts.
pub fn prepare_batched_prove_inputs<'a, F, P, const D: usize>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, F, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
) -> Result<PreparedBatchedProveInputs<'a, F, P, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    validate_batched_inputs(expanded, &claims, |group| group.polynomials.len(), true)?;

    let opening_points: Vec<&'a [F]> = claims.iter().map(|(point, _)| *point).collect();
    let commitments_by_point: Vec<RingCommitment<F, D>> = claims
        .iter()
        .flat_map(|(_, groups)| groups.iter().map(|group| group.commitment.clone()))
        .collect();
    let num_vars = opening_points[0].len();
    let batch_shape = MultiPointBatchShape {
        point_group_sizes: claims.iter().map(|(_, groups)| groups.len()).collect(),
        claim_group_sizes: claims
            .iter()
            .flat_map(|(_, groups)| groups.iter().map(|group| group.polynomials.len()))
            .collect(),
        claim_to_point: claims
            .iter()
            .enumerate()
            .flat_map(|(point_idx, (_, groups))| {
                groups
                    .iter()
                    .flat_map(move |group| std::iter::repeat_n(point_idx, group.polynomials.len()))
            })
            .collect(),
    };
    let layout_num_claims = checked_total_claims(&batch_shape.claim_group_sizes, "batched_prove")?;

    let flat_polys = claims
        .iter()
        .flat_map(|(_, groups)| groups.iter().flat_map(|group| group.polynomials.iter()))
        .collect();
    let flat_hints = claims
        .into_iter()
        .flat_map(|(_, groups)| groups.into_iter().map(|group| group.hint))
        .collect();

    Ok(PreparedBatchedProveInputs {
        opening_points,
        commitments_by_point,
        batch_shape,
        layout_num_claims,
        num_vars,
        flat_polys,
        flat_hints,
    })
}

/// Build a root-direct batched proof from already-validated prover claims.
///
/// Root schedule policy decides when the direct shortcut applies. This helper
/// owns only the config-free proof payload assembly from polynomial direct
/// witnesses.
///
/// # Errors
///
/// Returns an error if any polynomial cannot produce a direct root witness.
pub fn prove_root_direct_from_claims<F, const D: usize, P, C, H>(
    claims: &ProverClaims<'_, F, P, C, H>,
) -> Result<AkitaBatchedProof<F>, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let flat_polys = claims
        .iter()
        .flat_map(|(_, groups)| groups.iter().flat_map(|group| group.polynomials.iter()))
        .collect::<Vec<_>>();
    prove_root_direct_from_polys::<F, D, P>(&flat_polys)
}

/// Build a root-direct batched proof from flattened polynomial references.
///
/// # Errors
///
/// Returns an error if any polynomial cannot produce a direct root witness.
pub fn prove_root_direct_from_polys<F, const D: usize, P>(
    polys: &[&P],
) -> Result<AkitaBatchedProof<F>, AkitaError>
where
    F: FieldCore,
    P: AkitaPolyOps<F, D>,
{
    let witnesses = polys
        .iter()
        .map(|poly| poly.direct_root_witness())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AkitaBatchedProof {
        root: AkitaBatchedRootProof::new_direct(witnesses),
        steps: Vec::new(),
    })
}

/// Drive batched proving up to the config-selected folded-root policy.
///
/// This owns the config-free top-level prover work: validate/flatten public
/// prover claims, derive the schedule lookup key, select the schedule through
/// the supplied policy callback, apply the root-direct shortcut when the
/// selected schedule says no fold is needed, and derive the first recursive
/// schedule inputs for folded roots. Folded-root proving still runs in the
/// caller-supplied closure while config-selected recursive commitment layouts
/// remain outside this crate.
///
/// # Errors
///
/// Returns an error if claim preparation, schedule selection, root-direct
/// witness construction, root-next parameter selection, or folded-root proving
/// fails.
#[allow(clippy::too_many_arguments)]
pub fn prove_batched_with_policy<
    'a,
    F,
    T,
    P,
    const D: usize,
    SelectSchedule,
    SelectRootNext,
    ProveFolded,
>(
    expanded: &AkitaExpandedSetup<F>,
    claims: ProverClaims<'a, F, P, RingCommitment<F, D>, AkitaCommitmentHint<F, D>>,
    transcript: &mut T,
    basis: BasisMode,
    select_schedule: SelectSchedule,
    select_root_next_params: SelectRootNext,
    prove_folded: ProveFolded,
) -> Result<AkitaBatchedProof<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D>,
    SelectSchedule:
        FnOnce(usize, usize, usize, AkitaRootBatchSummary) -> Result<Schedule, AkitaError>,
    SelectRootNext: FnOnce(&Schedule, AkitaScheduleInputs) -> Result<LevelParams, AkitaError>,
    ProveFolded: FnOnce(
        PreparedBatchedProveInputs<'a, F, P, D>,
        Schedule,
        LevelParams,
        &mut T,
        BasisMode,
    ) -> Result<AkitaBatchedProof<F>, AkitaError>,
{
    let prepared_claims = prepare_batched_prove_inputs::<F, P, D>(expanded, claims)?;
    let batch_summary = AkitaRootBatchSummary::from_claim_group_sizes(
        &prepared_claims.batch_shape.claim_group_sizes,
        prepared_claims.opening_points.len(),
    )?;
    let max_num_vars = expanded.seed.max_num_vars;
    let root_key = AkitaScheduleLookupKey::with_batch(
        max_num_vars,
        prepared_claims.num_vars,
        prepared_claims.layout_num_claims,
        batch_summary,
    );
    let schedule = select_schedule(
        max_num_vars,
        prepared_claims.num_vars,
        prepared_claims.layout_num_claims,
        batch_summary,
    )?;

    if schedule_is_root_direct(&schedule) {
        return prove_root_direct_from_polys::<F, D, P>(&prepared_claims.flat_polys);
    }

    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    };
    let next_inputs = AkitaScheduleInputs {
        max_num_vars: root_key.max_num_vars,
        level: 1,
        current_w_len: root_step.next_w_len,
    };
    let root_next_params = select_root_next_params(&schedule, next_inputs)?;

    prove_folded(
        prepared_claims,
        schedule,
        root_next_params,
        transcript,
        basis,
    )
}

/// Build the recursive suffix from a root handoff, then assemble the final
/// folded batched proof.
///
/// The caller owns suffix schedule/config policy inside `build_suffix`; this
/// helper owns the config-free handoff from root raw output into suffix
/// construction and final proof assembly.
///
/// # Errors
///
/// Returns an error if suffix construction fails.
pub fn build_folded_batched_proof_with_suffix<F, const D: usize, BuildSuffix>(
    raw: RootLevelRawOutput<F, D>,
    build_suffix: BuildSuffix,
) -> Result<(AkitaBatchedProof<F>, usize), AkitaError>
where
    F: FieldCore,
    BuildSuffix: FnOnce(RecursiveProverState<F>) -> Result<RecursiveSuffixOutcome<F>, AkitaError>,
{
    let RootLevelRawOutput {
        y_rings,
        v,
        stage1,
        stage2_sumcheck,
        stage2_setup_claim_reduction,
        w_commitment_proof,
        w_eval,
        next_state,
    } = raw;
    let suffix = build_suffix(next_state)?;
    let RecursiveSuffixOutcome {
        levels,
        num_levels,
        final_state,
        final_log_basis,
    } = suffix;
    let root = AkitaBatchedRootProof::new_two_stage_with_setup_claim_reduction::<D>(
        y_rings,
        v,
        stage1,
        stage2_sumcheck,
        stage2_setup_claim_reduction,
        w_commitment_proof,
        w_eval,
    );
    let steps = build_final_proof_steps(levels, &final_state, final_log_basis);
    Ok((AkitaBatchedProof { root, steps }, num_levels))
}

/// Prove a folded batched root and assemble the recursive suffix.
///
/// The prover crate owns config-free folded-root preparation: root schedule
/// shape checks, opening-point reduction, commitment row shape validation,
/// root fold proving, recursive suffix handoff, and final proof assembly. The
/// caller supplies the already-selected first recursive commitment params plus
/// policy callbacks for committing root's next `w` and proving the suffix.
///
/// # Errors
///
/// Returns an error if the schedule is not folded, root inputs are malformed,
/// root proving fails, or suffix construction fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_folded_batched_with_policy<'a, F, T, P, const D: usize, CommitRootNext, BuildSuffix>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    prepared_claims: PreparedBatchedProveInputs<'a, F, P, D>,
    schedule: &Schedule,
    basis: BasisMode,
    root_next_params: &LevelParams,
    commit_root_next: CommitRootNext,
    build_suffix: BuildSuffix,
) -> Result<(AkitaBatchedProof<F>, usize), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    CommitRootNext: FnOnce(
        &mut MultiDNttCaches,
        &RecursiveWitnessFlat,
    )
        -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
    BuildSuffix: FnOnce(
        &mut MultiDNttCaches,
        &mut MultiDNttCaches,
        RecursiveProverState<F>,
        &Schedule,
        &mut T,
    ) -> Result<RecursiveSuffixOutcome<F>, AkitaError>,
{
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        return Err(AkitaError::InvalidSetup(
            "root schedule does not start with a fold".to_string(),
        ));
    };

    let mut ntt_cache = MultiDNttCaches::new();
    let mut commit_ntt_cache = MultiDNttCaches::new();
    let alpha_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
    let prepared_points = prepared_claims
        .opening_points
        .iter()
        .map(|opening_point| {
            prepare_root_opening_point::<F, D>(opening_point, basis, &root_step.params, alpha_bits)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if prepared_claims
        .commitments_by_point
        .iter()
        .any(|commitment| commitment.u.len() != root_step.params.b_key.row_len())
    {
        return Err(AkitaError::InvalidInput(
            "batched_prove received a commitment with the wrong length".to_string(),
        ));
    }

    let raw = prove_root_fold_with_params::<F, T, D, P, _>(
        expanded,
        ntt_shared,
        transcript,
        &prepared_claims.flat_polys,
        &prepared_claims.batch_shape,
        &prepared_points,
        &prepared_claims.commitments_by_point,
        prepared_claims.flat_hints,
        &root_step.params,
        root_next_params,
        root_step.s_field_len_emitted > 0,
        root_step.tier_setup_params,
        |w| commit_root_next(&mut commit_ntt_cache, w),
    )?;

    build_folded_batched_proof_with_suffix::<F, D, _>(raw, |next_state| {
        build_suffix(
            &mut ntt_cache,
            &mut commit_ntt_cache,
            next_state,
            schedule,
            transcript,
        )
    })
}

/// Drive recursive fold suffix levels using caller-supplied schedule and
/// ring-dimension policies.
///
/// Root config policy selects the current/next level parameters through
/// `select_fold_execution`, and dynamic ring dispatch lives inside
/// `prove_level`. This helper owns the config-free suffix loop, state
/// threading, and terminal direct-basis resolution.
///
/// # Errors
///
/// Returns an error if schedule selection, level proving, or terminal direct
/// basis resolution fails.
pub fn prove_recursive_suffix_with_policy<F, SelectFold, ProveLevel>(
    max_num_vars: usize,
    initial_state: RecursiveProverState<F>,
    schedule: &Schedule,
    mut select_fold_execution: SelectFold,
    mut prove_level: ProveLevel,
) -> Result<RecursiveSuffixOutcome<F>, AkitaError>
where
    F: FieldCore,
    SelectFold:
        FnMut(usize, AkitaScheduleInputs, u32) -> Result<(LevelParams, LevelParams), AkitaError>,
    ProveLevel: FnMut(
        usize,
        &RecursiveProverState<F>,
        &LevelParams,
        LevelParams,
        bool,
        TieredSetupParams,
    ) -> Result<ProveLevelOutput<F>, AkitaError>,
{
    let mut levels = Vec::new();
    let mut current_state = initial_state;
    let mut level = 1usize;
    let planned_num_levels = schedule_num_fold_levels(schedule);

    loop {
        let handle = &current_state.handles[0];
        let current_w_len = handle.w.len();
        if level >= planned_num_levels {
            break;
        }

        let inputs = AkitaScheduleInputs {
            max_num_vars,
            level,
            current_w_len,
        };
        let (level_params, next_params) = select_fold_execution(level, inputs, handle.log_basis)?;
        // Phase D-full cascade: only route `S` to the next fold when
        // the planner's `s_field_len_emitted` at this level is nonzero.
        // Otherwise the deferred setup claim is discharged by the
        // verifier's cleartext mle check inside
        // `verify_setup_claim_reduction`.
        // Tiered (book §5.4) cascade: if the schedule emitted a tiered
        // FoldStep, dispatch the tiered routing path; else un-tiered.
        let (route_setup_recursively, tier_setup_params) = match schedule.steps.get(level) {
            Some(Step::Fold(step)) => (step.s_field_len_emitted > 0, step.tier_setup_params),
            _ => (false, TieredSetupParams::un_tiered()),
        };
        let out = prove_level(
            level,
            &current_state,
            &level_params,
            next_params,
            route_setup_recursively,
            tier_setup_params,
        )?;

        levels.push(out.level_proof);
        current_state = out.next_state;
        level += 1;
    }

    let final_log_basis = resolve_final_log_basis(schedule, &current_state)?;

    Ok(RecursiveSuffixOutcome {
        levels,
        num_levels: level,
        final_state: current_state,
        final_log_basis,
    })
}

/// Prove one recursive fold level after the caller has built its quadratic
/// equation and selected the commitment policy for the next `w`.
///
/// The caller owns config/schedule decisions through `commit_w_for_next`; this
/// function owns the config-free prover mechanics: build `w`, commit it using
/// that closure, finish ring switching, run stage-1/stage-2 sumchecks, and
/// produce the next recursive state.
///
/// When the level's planner emits `s_field_len_emitted > 0` (book §5.3
/// lines 627-660: setup polynomial routed recursively), the deferred
/// setup-claim-reduction `(r_setup, s_opening_value)` is pushed into
/// the next level's recursive state as a second handle alongside the
/// folded witness. Otherwise the per-level cleartext `mle` check in
/// `verify_setup_claim_reduction` discharges the deferred claim.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_fold_level_from_quadratic<F, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    level: usize,
    lp: &LevelParams,
    next_params: &LevelParams,
    route_setup_recursively: bool,
    tier_setup_params: TieredSetupParams,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let total_t = Instant::now();
    tracing::debug!(
        "[fold_from_qe] level={level} start D={D} route_setup={route_setup_recursively} tier={:?}",
        tier_setup_params
    );
    let t = Instant::now();
    let w = ring_switch_build_w::<F, { D }>(&mut quad_eq, expanded, ntt_shared, lp)?;
    tracing::debug!(
        "[fold_from_qe] level={level} ring_switch_build_w done after {:?}",
        t.elapsed()
    );
    let (w_commitment_flat, w_hint_cache) = {
        let t = Instant::now();
        tracing::debug!("[fold_from_qe] level={level} commit_w_for_next start");
        let _span = tracing::info_span!("commit_w_level", level).entered();
        let out = commit_w_for_next(&w)?;
        tracing::debug!(
            "[fold_from_qe] level={level} commit_w_for_next done after {:?}",
            t.elapsed()
        );
        out
    };
    let w_commitment_proof = w_commitment_flat.clone();

    let t = Instant::now();
    tracing::debug!("[fold_from_qe] level={level} ring_switch_finalize start");
    let rs = ring_switch_finalize::<F, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        w,
        w_commitment_flat,
        &w_commitment_proof,
        w_hint_cache,
        lp,
    )?;
    tracing::debug!(
        "[fold_from_qe] level={level} ring_switch_finalize done after {:?}",
        t.elapsed()
    );

    let claim_to_point = quad_eq.claim_to_point().to_vec();
    let claim_group_sizes = quad_eq.claim_group_sizes().to_vec();
    let gamma_for_prepare = quad_eq.gamma().to_vec();
    let num_eval_rows_for_prepare = quad_eq.num_eval_rows();
    let opening_points_len = quad_eq.opening_points().len();
    let stage1_challenges_for_prepare = quad_eq.challenges.clone();
    let relation_claim = relation_claim_from_rows_with_layout::<F, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        commitment_rows,
        &y_rings,
        &lp.m_row_layout(claim_group_sizes.len(), num_eval_rows_for_prepare),
    );
    let RingSwitchOutput {
        w,
        w_commitment,
        w_hint,
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b,
        alpha,
    } = rs;
    let w_commitment = w_commitment.ok_or_else(|| {
        AkitaError::InvalidSetup("prover ring switch dropped w commitment".to_string())
    })?;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    let (stage1_proof, r_stage1, s_claim) = {
        let t = Instant::now();
        tracing::debug!("[fold_from_qe] level={level} stage1 start");
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = AkitaStage1Prover::new(
            &w_evals_compact,
            &tau0_reordered,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        )?;
        let (stage1_proof, r_stage1) = stage1_prover.prove(transcript)?;
        let s_claim = stage1_proof.s_claim;
        tracing::debug!(
            "[fold_from_qe] level={level} stage1 done after {:?}",
            t.elapsed()
        );
        (stage1_proof, r_stage1, s_claim)
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &s_claim);
    let gamma_range: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let gamma_rel: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH_REL);
    let (stage2_sumcheck, sumcheck_challenges, _stage2_final_claim, w_eval) = {
        let t = Instant::now();
        tracing::debug!(
            "[fold_from_qe] level={level} stage2 start live_x_cols={live_x_cols} col_bits={col_bits} ring_bits={ring_bits} b={b}"
        );
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            gamma_range,
            gamma_rel,
            w_evals_compact,
            &r_stage1,
            s_claim,
            b,
            alpha_evals_y.clone(),
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        );
        let (stage2_sumcheck, sumcheck_challenges, stage2_final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut stage2_prover, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level).entered();
            stage2_prover.final_w_eval()
        };
        tracing::debug!(
            "[fold_from_qe] level={level} stage2 done after {:?}",
            t.elapsed()
        );
        (
            stage2_sumcheck,
            sumcheck_challenges,
            stage2_final_claim,
            w_eval,
        )
    };

    let (setup_claim_reduction, r_setup) = if lp.use_setup_claim_reduction {
        let t = Instant::now();
        tracing::debug!("[fold_from_qe] level={level} setup_claim_reduction start");
        let _span = tracing::info_span!("setup_claim_reduction", level).entered();
        let prepared = prepare_m_eval::<F, D>(
            &stage1_challenges_for_prepare,
            alpha,
            lp,
            &tau1,
            &claim_group_sizes,
            &gamma_for_prepare,
            num_eval_rows_for_prepare,
            opening_points_len,
            &claim_to_point,
        )?;
        let x_challenges = &sumcheck_challenges[ring_bits..];
        let claim_scale = {
            let y_challenges = &sumcheck_challenges[..ring_bits];
            w_eval * multilinear_eval(&alpha_evals_y, y_challenges)?
        };
        let out = prove_setup_claim_reduction::<F, _, D>(
            &prepared,
            expanded,
            x_challenges,
            alpha,
            claim_scale,
            transcript,
        )?;
        let r_setup = out.challenges.clone();
        let payload = SetupClaimReductionPayload {
            m_setup_eval: out.input_claim,
            s_opening_value: out.s_opening_value,
            sumcheck: out.proof,
        };
        tracing::debug!(
            "[fold_from_qe] level={level} setup_claim_reduction done after {:?}",
            t.elapsed()
        );
        (Some(payload), Some(r_setup))
    } else {
        (None, None)
    };

    let s_opening_value_for_route = setup_claim_reduction
        .as_ref()
        .map(|payload| payload.s_opening_value);
    let (level_proof, sumcheck_challenges) = (
        AkitaLevelProof::new_two_stage_with_setup_claim_reduction::<D>(
            y_rings,
            quad_eq.v,
            stage1_proof,
            stage2_sumcheck,
            setup_claim_reduction,
            w_commitment_proof,
            w_eval,
        ),
        sumcheck_challenges,
    );

    let mut handles = vec![RecursivePolyHandle {
        w,
        dense_poly: None,
        commitment: w_commitment,
        hint: w_hint.ok_or_else(|| {
            AkitaError::InvalidSetup("prover ring switch dropped recursive hint cache".to_string())
        })?,
        log_basis: next_params.log_basis,
        opening_point: sumcheck_challenges.clone(),
        per_handle_lp: None,
        tiered: None,
    }];
    let _ = s_opening_value_for_route;
    if route_setup_recursively {
        if let Some(r_setup) = r_setup {
            let prepared = prepare_m_eval::<F, D>(
                &stage1_challenges_for_prepare,
                alpha,
                lp,
                &tau1,
                &claim_group_sizes,
                &gamma_for_prepare,
                num_eval_rows_for_prepare,
                opening_points_len,
                &claim_to_point,
            )?;
            let x_challenges = &sumcheck_challenges[ring_bits..];
            let (row_bits, coeff_bits) = prepared.setup_claim_reduction_dims();
            let row_count = 1usize << row_bits;
            let col_count = 1usize;
            let setup_field_len = row_count * D;
            let mut setup_rings =
                materialize_setup_claim_polynomial::<F, D>(&prepared, x_challenges, expanded)?;
            setup_rings.resize(row_count, CyclotomicRing::<F, D>::zero());
            if tier_setup_params.is_tiered() {
                let full_s_lp = untiered_setup_group_lp(next_params, setup_field_len)?;
                let setup_opening_point = setup_claim_opening_point_from_r_setup(
                    &r_setup, row_bits, coeff_bits, &full_s_lp,
                )?;
                let chunk_lp =
                    tiered_setup_group_lp(next_params, setup_field_len, tier_setup_params)?;
                let meta_lp = meta_lp_from_chunks(next_params, &chunk_lp, tier_setup_params)?;
                let tiered_material = build_tiered_handle_material::<F, D>(
                    expanded,
                    ntt_shared,
                    row_count,
                    col_count,
                    &chunk_lp,
                    &meta_lp,
                    tier_setup_params,
                    setup_opening_point.clone(),
                    Some(&setup_rings),
                )?;
                handles.push(RecursivePolyHandle {
                    w: RecursiveWitnessFlat::default(),
                    dense_poly: None,
                    commitment: FlatRingVec::default(),
                    hint: RecursiveCommitmentHintCache::default(),
                    log_basis: next_params.log_basis,
                    opening_point: setup_opening_point,
                    per_handle_lp: None,
                    tiered: Some(tiered_material),
                });
            } else {
                let s_lp = untiered_setup_group_lp(next_params, setup_field_len)?;
                let setup_opening_point =
                    setup_claim_opening_point_from_r_setup(&r_setup, row_bits, coeff_bits, &s_lp)?;
                let (commitment, hint) =
                    commit_dense_s_handle_direct(expanded, &setup_rings, &s_lp)?;
                let dense_poly = FlatRingVec::from_ring_elems::<D>(&setup_rings);
                handles.push(RecursivePolyHandle {
                    w: RecursiveWitnessFlat::default(),
                    dense_poly: Some(dense_poly),
                    commitment,
                    hint,
                    log_basis: next_params.log_basis,
                    opening_point: setup_opening_point,
                    per_handle_lp: Some(s_lp),
                    tiered: None,
                });
            }
        }
    }

    tracing::debug!(
        "[fold_from_qe] level={level} done after {:?}",
        total_t.elapsed()
    );
    Ok(ProveLevelOutput {
        level_proof,
        next_state: RecursiveProverState { handles },
    })
}

/// Prove one recursive fold level using already-selected current and next
/// level parameters.
///
/// Thin single-claim wrapper over [`prove_recursive_multi_fold_with_params`].
/// Construction sites with one polynomial pass through this helper to
/// preserve the legacy single-claim recursive wire shape.
///
/// # Errors
///
/// Returns whatever [`prove_recursive_multi_fold_with_params`] returns.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_fold_with_params<F, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    witness: &RecursiveWitnessView<'_, F, D>,
    opening_point: &[F],
    hint: AkitaCommitmentHint<F, D>,
    commitment: &FlatRingVec<F>,
    level: usize,
    level_params: &LevelParams,
    next_params: &LevelParams,
    route_setup_recursively: bool,
    tier_setup_params: TieredSetupParams,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let witness_poly = [RecursiveHandlePoly::Witness(*witness)];
    prove_recursive_multi_fold_with_params::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        &witness_poly,
        &[opening_point],
        vec![hint],
        &[commitment],
        &[None],
        &[None],
        level,
        level_params,
        next_params,
        route_setup_recursively,
        tier_setup_params,
        commit_w_for_next,
    )
}

/// Prove one recursive fold level with N polynomial claims jointly.
///
/// All `witnesses`, `opening_points`, `hints`, `commitments`, and
/// `per_claim_lps` slices must have the same length `N`. Each claim's
/// opening point may have a different length (each is padded to the
/// level's `m_vars + r_vars + alpha_bits` independently).
///
/// `per_claim_lps[i]` is the optional per-claim [`LevelParams`]
/// override for claim `i`: `None` means claim `i` inherits the level's
/// shared `level_params`; `Some(lp)` carries claim `i`'s per-
/// commitment-group `(m, r, B, digit_count)` for the multi-group
/// batched Hachi commit at the next level. Heterogeneous per-claim
/// LPs are grouped via [`LevelParams::groups`] and dispatched through
/// the multi-group commit kernel; see the verifier mirror in
/// `verify_one_level`.
///
/// The wire shape for `N == 1` exactly matches today's legacy
/// single-claim recursive wire: one commitment + one padded point + one
/// y-ring, no openings absorbed, no `gamma` sampled. For `N > 1` the
/// transcript layout mirrors [`verify_one_level`]'s multi-claim path:
/// commitments × N, padded points × N, openings × N, sample `gamma` × N,
/// y-rings × N. (For now this assumes a 1-claim-per-point layout, so
/// `num_eval_rows == N` and each y-ring carries a single claim's
/// contribution.)
///
/// Phase D-full slice F discharges the deferred setup-claim
/// `(r_setup, s_opening_value)` here as `claims[1]` (the `S` opening),
/// alongside the folded witness as `claims[0]`, and lifts the
/// per-claim-LP restriction so the `w`-claim and `S`-claim can carry
/// distinct `(m, r, B, digit_count)` under shared outer `(D, A)`.
///
/// # Errors
///
/// Returns an error if slice lengths disagree, any opening-point
/// length underflows the level's alpha, witness folding fails, the
/// recursive quadratic equation rejects, or the folded prover fails.
///
/// # Panics
///
/// Panics if the Phase 5 grouping accumulator's `last_mut()` returns
/// `None` after a merge — this is internally unreachable because the
/// merge branch only fires when the previous claim already pushed a
/// non-empty entry into `batch_claim_group_sizes`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_multi_fold_with_params<F, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    witnesses: &[RecursiveHandlePoly<'_, F, D>],
    opening_points: &[&[F]],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    commitments: &[&FlatRingVec<F>],
    per_claim_lps: &[Option<LevelParams>],
    tiered_per_claim: &[Option<TieredSetupParams>],
    level: usize,
    level_params: &LevelParams,
    next_params: &LevelParams,
    route_setup_recursively: bool,
    tier_setup_params: TieredSetupParams,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let total_t = Instant::now();
    tracing::debug!(
        "[recursive_multi] level={level} start D={D} num_claims={}",
        witnesses.len()
    );
    let num_claims = witnesses.len();
    if num_claims == 0
        || opening_points.len() != num_claims
        || hints.len() != num_claims
        || commitments.len() != num_claims
        || per_claim_lps.len() != num_claims
        || tiered_per_claim.len() != num_claims
    {
        return Err(AkitaError::InvalidInput(
            "prove_recursive_multi_fold_with_params: slice length mismatch".to_string(),
        ));
    }
    let claim_lps: Vec<LevelParams> = per_claim_lps
        .iter()
        .map(|lp| lp.clone().unwrap_or_else(|| level_params.clone()))
        .collect();
    // Phase 5 grouping (book §5.4): consecutive claims that share the
    // same per-claim LP AND carry the same `Some(tier)` marker collapse
    // into ONE `GroupSpec` with `claim_count = run_length` and the
    // tier mark preserved. This is the routed-tiered chunks pattern:
    // a tiered handle expands to `k` claims with identical `chunk_lp`
    // (and `tier_per_claim[i] = Some(t)`), which here become one
    // commitment group with `claim_count = k` carrying `tier = Some(t)`.
    // All other patterns produce one group per claim, matching the
    // existing slice F shape.
    let mut batch_groups: Vec<GroupSpec> = Vec::new();
    let mut batch_claim_group_sizes: Vec<usize> = Vec::new();
    let mut prev_lp: Option<&LevelParams> = None;
    let mut prev_tier: Option<TieredSetupParams> = None;
    for (claim_idx, lp) in claim_lps.iter().enumerate() {
        let cur_tier = tiered_per_claim[claim_idx];
        let mergeable = match (prev_lp, prev_tier, cur_tier) {
            (Some(prev), Some(pt), Some(ct)) => prev == lp && pt == ct,
            _ => false,
        };
        if mergeable {
            *batch_claim_group_sizes.last_mut().unwrap() += 1;
        } else {
            let mut spec = GroupSpec::from_outer(lp);
            spec.tier = cur_tier;
            batch_groups.push(spec);
            batch_claim_group_sizes.push(1);
            prev_lp = Some(lp);
            prev_tier = cur_tier;
        }
    }
    let batch_level_params = if num_claims == 1 && claim_lps[0] == *level_params {
        level_params.clone()
    } else {
        LevelParams {
            groups: Some(batch_groups.clone()),
            ..level_params.clone()
        }
    };
    tracing::debug!(
        "[recursive_multi] level={level} grouped claims={batch_claim_group_sizes:?} after {:?}",
        total_t.elapsed()
    );

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level,
            num_claims,
            "prove_recursive_multi_fold_with_params"
        );
    }

    let alpha = level_params.ring_dimension.trailing_zeros() as usize;

    // Per-claim padded points, ring opening points, inner reductions,
    // evaluate_and_fold outputs.
    let mut padded_points: Vec<Vec<F>> = Vec::with_capacity(num_claims);
    let mut ring_opening_points: Vec<akita_types::RingOpeningPoint<F>> =
        Vec::with_capacity(num_claims);
    let mut inner_reductions: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(num_claims);
    let mut per_claim_y_rings: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(num_claims);
    let mut per_claim_w_folded: Vec<Vec<CyclotomicRing<F, D>>> = Vec::with_capacity(num_claims);
    let t = Instant::now();
    for (claim_idx, (witness, opening_point)) in
        witnesses.iter().zip(opening_points.iter()).enumerate()
    {
        let claim_lp = &claim_lps[claim_idx];
        let target_num_vars = claim_lp.m_vars + claim_lp.r_vars + alpha;
        if opening_point.len() < alpha {
            return Err(AkitaError::InvalidPointDimension {
                expected: alpha,
                actual: opening_point.len(),
            });
        }
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(target_num_vars, F::zero());
        let inner_point = &padded_point[..alpha];
        let outer_point = &padded_point[alpha..];

        let inner_reduction =
            reduce_inner_opening_to_ring_element::<F, { D }>(inner_point, BasisMode::Lagrange)?;
        let ring_opening_point = {
            let _span =
                tracing::info_span!("ring_opening_point", level, claim_idx = claim_idx).entered();
            ring_opening_point_from_field::<F>(
                outer_point,
                claim_lp.r_vars,
                claim_lp.m_vars,
                BasisMode::Lagrange,
                BlockOrder::ColumnMajor,
            )?
        };

        let fold_scalars = &ring_opening_point.a;
        let eval_outer_scalars = &ring_opening_point.b;
        let (y_ring, w_folded) = {
            let _span = tracing::info_span!(
                "evaluate_and_fold",
                level,
                claim_idx = claim_idx,
                num_ring_elems = witness.num_ring_elems()
            )
            .entered();
            witness.evaluate_and_fold(
                eval_outer_scalars,
                fold_scalars,
                claim_lp.block_len,
                claim_lp.num_blocks,
            )
        };

        padded_points.push(padded_point);
        ring_opening_points.push(ring_opening_point);
        inner_reductions.push(inner_reduction);
        per_claim_y_rings.push(y_ring);
        per_claim_w_folded.push(w_folded);
        tracing::debug!(
            "[recursive_multi] level={level} evaluate_and_fold claim {}/{} done after {:?}",
            claim_idx + 1,
            num_claims,
            t.elapsed()
        );
    }
    tracing::debug!(
        "[recursive_multi] level={level} all evaluate_and_fold done after {:?}",
        t.elapsed()
    );

    // Multi-claim transcript layout mirroring `verify_one_level`:
    //   commitments × N, padded points × N, [openings × N, sample γ × N if N>1],
    //   y-rings × N_points.
    for commitment in commitments {
        commitment.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;
    }
    for padded_point in &padded_points {
        for pt in padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
    }
    let gamma: Vec<F> = if num_claims > 1 {
        // Each claim's opening is the first coefficient of
        // `y_ring * σ_{-1}(v)`. This matches the verifier's per-point
        // trace check: `trace(y_ring * σ_{-1}(v)) = d · γ · opening`.
        let openings: Vec<F> = inner_reductions
            .iter()
            .zip(per_claim_y_rings.iter())
            .map(|(inner_reduction, y_ring)| {
                (*y_ring * inner_reduction.sigma_m1()).coefficients()[0]
            })
            .collect();
        for opening in &openings {
            transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
        }
        (0..num_claims)
            .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
            .collect()
    } else {
        vec![F::one()]
    };
    // Phase 5: y_rings are one-per-GROUP (not one-per-claim) because
    // grouped claims (e.g. tiered chunks) share an opening point and
    // their per-claim y_rings sum (gamma-batched) into the group's
    // y_ring. This matches the verifier-side y-ring slot count.
    let mut claim_to_point: Vec<usize> = Vec::with_capacity(num_claims);
    for (point_idx, &group_size) in batch_claim_group_sizes.iter().enumerate() {
        for _ in 0..group_size {
            claim_to_point.push(point_idx);
        }
    }
    let num_groups_for_y = batch_claim_group_sizes.len();
    let mut group_y_rings: Vec<CyclotomicRing<F, D>> =
        vec![CyclotomicRing::<F, D>::zero(); num_groups_for_y];
    for (claim_idx, y_ring) in per_claim_y_rings.iter().enumerate() {
        let p = claim_to_point[claim_idx];
        group_y_rings[p] += y_ring.scale(&gamma[claim_idx]);
    }
    for y_ring in &group_y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }
    tracing::debug!(
        "[recursive_multi] level={level} transcript/openings done after {:?}",
        total_t.elapsed()
    );

    // Build commitment-row references for the recursive QE.
    let claim_us: Vec<&[CyclotomicRing<F, D>]> = commitments
        .iter()
        .map(|c| c.as_ring_slice::<D>())
        .collect::<Result<Vec<_>, _>>()?;
    let claim_group_sizes: Vec<usize> = batch_claim_group_sizes.clone();
    let num_eval_rows = batch_claim_group_sizes.len();

    // Phase 5: aggregate per-claim hints and per-claim commitment u-vectors
    // into per-GROUP entries (one entry per `claim_group_sizes` slot).
    // For chunks-as-1-group with `claim_count = k`, this concatenates
    // the `k` chunk hints into one hint with `inner_opening_digits.len() == k`
    // and concatenates the `k` chunk u-vectors into one group u-vector.
    // For un-tiered (claim_group_sizes == [1; N]), this is a no-op
    // restructuring that produces one entry per claim.
    let mut group_hints: Vec<AkitaCommitmentHint<F, D>> = Vec::new();
    let mut group_us: Vec<Vec<CyclotomicRing<F, D>>> = Vec::new();
    let mut hint_iter = hints.into_iter();
    let mut u_idx = 0usize;
    for &group_size in &claim_group_sizes {
        let mut group_inner: Vec<akita_types::FlatDigitBlocks<D>> = Vec::with_capacity(group_size);
        let mut group_t: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = Vec::with_capacity(group_size);
        let mut have_t = true;
        for _ in 0..group_size {
            let hint = hint_iter.next().ok_or_else(|| {
                AkitaError::InvalidInput(
                    "prove_recursive_multi_fold: hint count mismatch with claim_group_sizes"
                        .to_string(),
                )
            })?;
            let (inner, t_opt) = hint.into_parts();
            for digits in inner {
                group_inner.push(digits);
            }
            match t_opt {
                Some(t_rows) => group_t.extend(t_rows),
                None => have_t = false,
            }
        }
        let group_hint = if have_t {
            AkitaCommitmentHint::with_t(group_inner, group_t)
        } else {
            AkitaCommitmentHint::new(group_inner)
        };
        group_hints.push(group_hint);
        let mut group_u: Vec<CyclotomicRing<F, D>> = Vec::new();
        for _ in 0..group_size {
            group_u.extend_from_slice(claim_us[u_idx]);
            u_idx += 1;
        }
        group_us.push(group_u);
    }
    let group_us_refs: Vec<&[CyclotomicRing<F, D>]> = group_us.iter().map(Vec::as_slice).collect();
    tracing::debug!(
        "[recursive_multi] level={level} grouped hints/u done after {:?}",
        total_t.elapsed()
    );

    // Phase 5: dedupe ring_opening_points per group (per opening point)
    // since `claim_to_point` maps claims to point indices and grouped
    // claims share their opening point.
    let mut group_ring_opening_points: Vec<akita_types::RingOpeningPoint<F>> =
        Vec::with_capacity(num_eval_rows);
    let mut taken = vec![false; num_eval_rows];
    for (claim_idx, point_idx) in claim_to_point.iter().copied().enumerate() {
        if !taken[point_idx] {
            // The first claim mapping to this point owns the canonical
            // opening point representation.
            if group_ring_opening_points.len() <= point_idx {
                group_ring_opening_points
                    .resize_with(point_idx + 1, || ring_opening_points[claim_idx].clone());
            }
            group_ring_opening_points[point_idx] = ring_opening_points[claim_idx].clone();
            taken[point_idx] = true;
        }
    }

    let t = Instant::now();
    tracing::debug!(
        "[recursive_multi] level={level} QuadraticEquation::new_recursive_prover start"
    );
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_recursive_prover(
        ntt_shared,
        group_ring_opening_points,
        claim_to_point,
        witnesses,
        per_claim_w_folded,
        &claim_group_sizes,
        batch_level_params.clone(),
        group_hints,
        transcript,
        &group_us_refs,
        &group_y_rings,
        gamma,
        num_eval_rows,
        expanded.seed.max_stride,
    )?);
    tracing::debug!(
        "[recursive_multi] level={level} QuadraticEquation built after {:?}",
        t.elapsed()
    );

    // Commitment-rows slice for `prove_fold_level_from_quadratic`. For
    // N == 1 this is just the single commitment's u; for N > 1 we
    // concatenate all per-group u-rows.
    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if num_claims == 1 {
        None
    } else {
        let mut rows = Vec::with_capacity(batch_level_params.total_b_row_count(num_eval_rows));
        for u in &group_us {
            rows.extend_from_slice(u);
        }
        Some(rows)
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(rows) => rows.as_slice(),
        None => group_us_refs[0],
    };

    tracing::debug!(
        "[recursive_multi] level={level} prove_fold_level_from_quadratic start after {:?}",
        total_t.elapsed()
    );
    let out = prove_fold_level_from_quadratic::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        commitment_rows,
        level,
        &batch_level_params,
        next_params,
        route_setup_recursively,
        tier_setup_params,
        quad_eq,
        group_y_rings,
        commit_w_for_next,
    )?;
    tracing::debug!(
        "[recursive_multi] level={level} done after {:?}",
        total_t.elapsed()
    );
    Ok(out)
}

/// Prove one recursive fold level from D-erased recursive state using
/// caller-supplied config policy.
///
/// The prover crate owns the state unpacking, typed recursive witness view,
/// typed hint conversion, opening-point handoff, and fold proof mechanics.
/// The caller supplies only the current-witness layout policy and the
/// next-level recursive commitment policy.
///
/// # Errors
///
/// Returns an error if the current witness cannot be viewed at `D`, the hint
/// cannot be typed at `D`, layout selection fails, or recursive proving fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_recursive_level_with_policy<F, T, const D: usize, CurrentLayout, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    current_state: &RecursiveProverState<F>,
    level: usize,
    level_params: &LevelParams,
    next_params: &LevelParams,
    route_setup_recursively: bool,
    tier_setup_params: TieredSetupParams,
    current_layout: CurrentLayout,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    CurrentLayout: FnOnce(&LevelParams, usize) -> Result<LevelParams, AkitaError>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let total_t = Instant::now();
    tracing::debug!(
        "[recursive_level] level={level} start D={D} handles={}",
        current_state.handles.len()
    );
    let _setup_span = tracing::info_span!("inter_level_setup", level).entered();

    if current_state.handles.is_empty() {
        return Err(AkitaError::InvalidInput(
            "prove_recursive_level_with_policy: empty recursive state".to_string(),
        ));
    }
    let current_w_len = current_state.handles[0].w.len();
    let w_lp = current_layout(level_params, current_w_len)?;
    tracing::debug!(
        "[recursive_level] level={level} current_layout done after {:?}",
        total_t.elapsed()
    );
    // Gather views only for handles that carry a recursive witness
    // (W-style). Tiered and dense-S handles do not have a witness view.
    let views: Vec<RecursiveWitnessView<'_, F, D>> = current_state
        .handles
        .iter()
        .filter(|h| h.dense_poly.is_none() && h.tiered.is_none())
        .map(|h| h.w.view::<F, D>())
        .collect::<Result<_, _>>()?;
    let mut view_iter = views.iter();

    // Expand handles into per-claim sub-claims for the L+1 multi-fold.
    // Non-tiered handles produce 1 entry each. Tiered handles produce
    // ONE aggregated chunks claim (book §5.4 lines 686-754, §5.5 line
    // 752 "Growth ≈ 1.0-3.0×") + 1 meta claim.
    //
    // γ-folding: after absorbing the k chunk u_j + meta u_meta into
    // the transcript, sample γ ∈ F^k via CHALLENGE_TIERED_CHUNK_AGGREGATION.
    // Aggregate chunk_poly_agg = Σ_j γ_j · chunk_polys[j] then re-commit
    // via the standard chunk_lp chain → (u_agg_fresh, hint_agg). The L+1
    // M-relation verifies `u_agg_fresh = M_chunks · chunk_poly_agg` by
    // standard linearity; the verifier mirror in expand_tiered_setup_claims
    // computes the same u_agg_fresh from public S + γ + chunk_indices.
    let mut handle_polys: Vec<RecursiveHandlePoly<'_, F, D>> = Vec::new();
    let mut opening_points: Vec<Vec<F>> = Vec::new();
    let mut typed_hints: Vec<AkitaCommitmentHint<F, D>> = Vec::new();
    let mut commitment_owned: Vec<FlatRingVec<F>> = Vec::new();
    let mut per_claim_lps: Vec<Option<LevelParams>> = Vec::new();
    let mut tiered_per_claim: Vec<Option<TieredSetupParams>> = Vec::new();
    for h in &current_state.handles {
        match (h.dense_poly.as_ref(), h.tiered.as_ref()) {
            (Some(_), Some(_)) => {
                return Err(AkitaError::InvalidInput(
                    "RecursivePolyHandle carries both dense_poly and tiered material".to_string(),
                ));
            }
            (Some(poly), None) => {
                let coeffs = poly.try_to_vec::<D>()?;
                handle_polys.push(RecursiveHandlePoly::Dense(DensePoly::from_ring_coeffs(
                    coeffs,
                )));
                opening_points.push(h.opening_point.clone());
                typed_hints.push(h.hint.to_typed::<D>()?);
                commitment_owned.push(h.commitment.clone());
                per_claim_lps.push(h.per_handle_lp.clone());
                tiered_per_claim.push(None);
            }
            (None, Some(tiered)) => {
                tracing::debug!(
                    "[recursive_level] level={level} aggregating tiered handle chunks={} after {:?}",
                    tiered.tier.num_chunks,
                    total_t.elapsed()
                );
                let log_shrink = tiered.tier.log2_shrink()? as usize;
                let chunk_opening_point = tiered_setup_chunk_opening_point(
                    &tiered.opening_point,
                    D.trailing_zeros() as usize,
                    tiered.chunk_lp.r_vars + log_shrink,
                    tiered.chunk_lp.m_vars + log_shrink,
                    tiered.tier,
                )?;

                // γ-folding step: bind γ to the k chunk u_j and meta u
                // before computing the aggregation. Mirrors the verifier's
                // expand_tiered_setup_claims absorption + γ sample.
                for u in &tiered.chunk_commitments {
                    u.append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;
                }
                tiered
                    .meta_commitment
                    .append_as_ring_commitment::<T, D>(ABSORB_COMMITMENT, transcript)?;
                let gamma_chunk: Vec<F> = (0..tiered.tier.num_chunks)
                    .map(|_| transcript.challenge_scalar(CHALLENGE_TIERED_CHUNK_AGGREGATION))
                    .collect();

                // chunk_poly_agg = Σ_j γ_chunk[j] · chunk_polys[j]
                let chunk_n = tiered.chunk_polys[0].as_ring_slice::<D>()?.len();
                let mut chunk_poly_agg: Vec<CyclotomicRing<F, D>> =
                    vec![CyclotomicRing::<F, D>::zero(); chunk_n];
                for (j, (chunk_poly, gamma_j)) in tiered
                    .chunk_polys
                    .iter()
                    .zip(gamma_chunk.iter())
                    .enumerate()
                {
                    let rings = chunk_poly.as_ring_slice::<D>()?;
                    if rings.len() != chunk_n {
                        return Err(AkitaError::InvalidSetup(format!(
                            "tiered chunk_polys[{j}] has {} rings; expected {chunk_n}",
                            rings.len()
                        )));
                    }
                    for (acc, ring) in chunk_poly_agg.iter_mut().zip(rings.iter()) {
                        *acc += ring.scale(gamma_j);
                    }
                }

                // Re-commit chunk_poly_agg under chunk_lp via the standard
                // dense chain. Produces u_agg_fresh (1 chunks_agg commitment)
                // and hint_agg (proper i8 t̂_agg digits) without overflow.
                let (u_agg_flat, hint_agg) = commit_dense_s_handle_direct::<F, D>(
                    expanded,
                    &chunk_poly_agg,
                    &tiered.chunk_lp,
                )?;

                handle_polys.push(RecursiveHandlePoly::Dense(DensePoly::from_ring_coeffs(
                    chunk_poly_agg,
                )));
                opening_points.push(chunk_opening_point);
                typed_hints.push(hint_agg.to_typed::<D>()?);
                commitment_owned.push(u_agg_flat);
                per_claim_lps.push(Some(tiered.chunk_lp.clone()));
                tiered_per_claim.push(Some(tiered.tier));

                let meta_coeffs = tiered.meta_input_poly.try_to_vec::<D>()?;
                handle_polys.push(RecursiveHandlePoly::Dense(DensePoly::from_ring_coeffs(
                    meta_coeffs,
                )));
                opening_points.push(tiered.opening_point.clone());
                typed_hints.push(tiered.meta_hint.to_typed::<D>()?);
                commitment_owned.push(tiered.meta_commitment.clone());
                per_claim_lps.push(Some(tiered.meta_lp.clone()));
                tiered_per_claim.push(None);
                tracing::debug!(
                    "[recursive_level] level={level} aggregated tiered handle after {:?}",
                    total_t.elapsed()
                );
            }
            (None, None) => {
                let view = *view_iter.next().ok_or_else(|| {
                    AkitaError::InvalidInput("missing recursive witness view".to_string())
                })?;
                handle_polys.push(RecursiveHandlePoly::Witness(view));
                opening_points.push(h.opening_point.clone());
                typed_hints.push(h.hint.to_typed::<D>()?);
                commitment_owned.push(h.commitment.clone());
                per_claim_lps.push(h.per_handle_lp.clone());
                tiered_per_claim.push(None);
            }
        }
    }
    let commitment_refs: Vec<&FlatRingVec<F>> = commitment_owned.iter().collect();
    let opening_point_refs: Vec<&[F]> = opening_points.iter().map(Vec::as_slice).collect();
    drop(_setup_span);

    tracing::debug!(
        "[recursive_level] level={level} calling recursive_multi claims={} after {:?}",
        handle_polys.len(),
        total_t.elapsed()
    );
    prove_recursive_multi_fold_with_params::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        &handle_polys,
        &opening_point_refs,
        typed_hints,
        &commitment_refs,
        &per_claim_lps,
        &tiered_per_claim,
        level,
        &w_lp,
        next_params,
        route_setup_recursively,
        tier_setup_params,
        commit_w_for_next,
    )
}

/// Prove the folded root level using already-selected root and next-level
/// parameters.
///
/// The caller owns schedule/config selection and passes the expected next
/// recursive witness length, next digit basis, and commitment policy for that
/// witness. This function owns root polynomial folding, public root transcript
/// setup, root quadratic-equation construction, and the folded-root prover
/// mechanics.
///
/// # Errors
///
/// Returns an error if root inputs are malformed, polynomial folding or
/// quadratic-equation construction fails, or the folded-root prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root_fold_with_params<F, T, const D: usize, P, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    polys: &[&P],
    batch_shape: &MultiPointBatchShape,
    prepared_points: &[PreparedRootOpeningPoint<F, D>],
    commitments: &[RingCommitment<F, D>],
    hints: Vec<AkitaCommitmentHint<F, D>>,
    root_params: &LevelParams,
    next_params: &LevelParams,
    route_setup_recursively: bool,
    tier_setup_params: TieredSetupParams,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    P: AkitaPolyOps<F, D, CommitCache = NttSlotCache<D>>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let claim_to_point = &batch_shape.claim_to_point;
    let claim_group_sizes = &batch_shape.claim_group_sizes;
    let point_group_sizes = &batch_shape.point_group_sizes;

    if prepared_points.is_empty() || claim_to_point.len() != polys.len() {
        return Err(AkitaError::InvalidInput(
            "invalid root-level inputs".to_string(),
        ));
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= prepared_points.len())
    {
        return Err(AkitaError::InvalidInput(
            "root-level claim-to-point index out of range".to_string(),
        ));
    }

    {
        let x: u8 = 0;
        tracing::trace!(
            stack_ptr = format_args!("{:#x}", &x as *const u8 as usize),
            level = 0usize,
            num_claims = claim_to_point.len(),
            num_points = prepared_points.len(),
            "prove_root_fold_with_params"
        );
    }

    let (per_claim_y_rings, w_folded_by_poly) = {
        let _span = tracing::info_span!(
            "evaluate_and_fold",
            level = 0usize,
            num_polys = polys.len(),
            num_points = prepared_points.len()
        )
        .entered();
        let mut per_claim_y_rings = Vec::with_capacity(polys.len());
        let mut w_folded_by_poly = Vec::with_capacity(polys.len());
        for (poly, &point_idx) in polys.iter().zip(claim_to_point.iter()) {
            let prepared_point = &prepared_points[point_idx];
            let (y_ring, w_folded) = poly.evaluate_and_fold(
                &prepared_point.ring_opening_point.b,
                &prepared_point.ring_opening_point.a,
                root_params.block_len,
            );
            per_claim_y_rings.push(y_ring);
            w_folded_by_poly.push(w_folded);
        }
        (per_claim_y_rings, w_folded_by_poly)
    };

    append_batch_shape_to_transcript::<F, T>(point_group_sizes, claim_group_sizes, transcript);
    append_batched_commitments_to_transcript(commitments, transcript);
    for prepared_point in prepared_points {
        for pt in &prepared_point.padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
    }

    let openings: Vec<F> = per_claim_y_rings
        .iter()
        .zip(claim_to_point.iter())
        .map(|(y_ring, &point_idx)| {
            let v = &prepared_points[point_idx].inner_reduction;
            (*y_ring * v.sigma_m1()).coefficients()[0]
        })
        .collect();
    for opening in &openings {
        transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
    }
    let gamma: Vec<F> = (0..polys.len())
        .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
        .collect();

    let num_points = prepared_points.len();
    let mut y_rings = vec![CyclotomicRing::<F, D>::zero(); num_points];
    for (claim_idx, y_ring) in per_claim_y_rings.iter().enumerate() {
        let point_idx = claim_to_point[claim_idx];
        y_rings[point_idx] += y_ring.scale(&gamma[claim_idx]);
    }
    for y_ring in &y_rings {
        transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
    }

    let ring_opening_points = prepared_points
        .iter()
        .map(|prepared_point| prepared_point.ring_opening_point.clone())
        .collect();
    let quad_eq = Box::new(QuadraticEquation::<F, { D }>::new_prover(
        ntt_shared,
        ring_opening_points,
        claim_to_point.clone(),
        polys,
        w_folded_by_poly,
        claim_group_sizes,
        root_params.clone(),
        hints,
        transcript,
        commitments,
        &y_rings,
        gamma,
        expanded.seed.max_stride,
    )?);

    let commitment_rows_owned: Option<Vec<CyclotomicRing<F, D>>> = if commitments.len() == 1 {
        None
    } else {
        Some(flatten_batched_commitment_rows(commitments))
    };
    let commitment_rows: &[CyclotomicRing<F, D>] = match &commitment_rows_owned {
        Some(v) => v.as_slice(),
        None => commitments[0].u.as_slice(),
    };

    prove_root_fold_from_quadratic::<F, T, D, _>(
        expanded,
        ntt_shared,
        transcript,
        commitment_rows,
        root_params,
        next_params,
        route_setup_recursively,
        tier_setup_params,
        quad_eq,
        y_rings,
        commit_w_for_next,
    )
}

/// Prove the folded root level after root orchestration has built its
/// quadratic equation and selected the next recursive commitment policy.
///
/// The root caller owns transcript setup for public openings and gamma
/// batching, schedule selection, and the commitment-row view used by the root
/// relation. This function owns the config-free prover mechanics from `w`
/// construction through the stage proofs and next recursive state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_root_fold_from_quadratic<F, T, const D: usize, CommitW>(
    expanded: &AkitaExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    lp: &akita_types::LevelParams,
    next_params: &LevelParams,
    route_setup_recursively: bool,
    tier_setup_params: TieredSetupParams,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasUnreducedOps + HasWide + HalvingField,
    T: Transcript<F>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError>,
{
    let w = ring_switch_build_w::<F, { D }>(&mut quad_eq, expanded, ntt_shared, lp)?;
    let (w_commitment_flat, w_hint_cache) = {
        let _span = tracing::info_span!("commit_w_level", level = 0usize).entered();
        commit_w_for_next(&w)?
    };
    let w_commitment_proof = w_commitment_flat.clone();

    let rs = ring_switch_finalize_with_claim_groups::<F, T, { D }>(
        &quad_eq,
        expanded,
        transcript,
        w,
        w_commitment_flat,
        &w_commitment_proof,
        w_hint_cache,
        lp,
    )?;

    let relation_claim = relation_claim_from_rows_with_layout::<F, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        commitment_rows,
        &y_rings,
        &lp.m_row_layout(quad_eq.claim_group_sizes().len(), quad_eq.num_eval_rows()),
    );

    let RingSwitchOutput {
        w,
        w_commitment,
        w_hint,
        w_evals_compact,
        live_x_cols,
        m_evals_x,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b,
        alpha,
    } = rs;
    let w_commitment = w_commitment.ok_or_else(|| {
        AkitaError::InvalidSetup("prover ring switch dropped w commitment".to_string())
    })?;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    let (stage1_proof, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = AkitaStage1Prover::new(
            &w_evals_compact,
            &tau0_reordered,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        )?;
        let (stage1_proof, r_stage1) = stage1_prover.prove(transcript)?;
        let s_claim = stage1_proof.s_claim;
        (stage1_proof, r_stage1, s_claim)
    };

    transcript.append_serde(ABSORB_SUMCHECK_S_CLAIM, &s_claim);
    let gamma_range: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let gamma_rel: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH_REL);
    let claim_to_point = quad_eq.claim_to_point().to_vec();
    let claim_group_sizes = quad_eq.claim_group_sizes().to_vec();
    let gamma_for_prepare = quad_eq.gamma().to_vec();
    let num_eval_rows_for_prepare = quad_eq.num_eval_rows();
    let opening_points_len = quad_eq.opening_points().len();
    let stage1_challenges_for_prepare = quad_eq.challenges.clone();
    let (stage2_sumcheck, sumcheck_challenges, _stage2_final_claim, w_eval) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let mut stage2_prover = AkitaStage2Prover::new(
            gamma_range,
            gamma_rel,
            w_evals_compact,
            &r_stage1,
            s_claim,
            b,
            alpha_evals_y.clone(),
            m_evals_x,
            live_x_cols,
            col_bits,
            ring_bits,
            relation_claim,
        );
        let (stage2_sumcheck, sumcheck_challenges, stage2_final_claim) =
            prove_sumcheck::<F, _, F, _, _>(&mut stage2_prover, transcript, |tr| {
                tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND)
            })?;

        let w_eval = {
            let _span = tracing::info_span!("multilinear_eval", level = 0usize).entered();
            stage2_prover.final_w_eval()
        };
        (
            stage2_sumcheck,
            sumcheck_challenges,
            stage2_final_claim,
            w_eval,
        )
    };

    let (stage2_setup_claim_reduction, r_setup) = if lp.use_setup_claim_reduction {
        let _span = tracing::info_span!("setup_claim_reduction", level = 0usize).entered();
        let prepared = prepare_m_eval::<F, D>(
            &stage1_challenges_for_prepare,
            alpha,
            lp,
            &tau1,
            &claim_group_sizes,
            &gamma_for_prepare,
            num_eval_rows_for_prepare,
            opening_points_len,
            &claim_to_point,
        )?;
        let x_challenges = &sumcheck_challenges[ring_bits..];
        let claim_scale = {
            let y_challenges = &sumcheck_challenges[..ring_bits];
            w_eval * multilinear_eval(&alpha_evals_y, y_challenges)?
        };
        let out = prove_setup_claim_reduction::<F, _, D>(
            &prepared,
            expanded,
            x_challenges,
            alpha,
            claim_scale,
            transcript,
        )?;
        let r_setup = out.challenges.clone();
        let payload = SetupClaimReductionPayload {
            m_setup_eval: out.input_claim,
            s_opening_value: out.s_opening_value,
            sumcheck: out.proof,
        };
        (Some(payload), Some(r_setup))
    } else {
        (None, None)
    };

    let mut handles = vec![RecursivePolyHandle {
        w,
        dense_poly: None,
        commitment: w_commitment,
        hint: w_hint.ok_or_else(|| {
            AkitaError::InvalidSetup("prover ring switch dropped recursive hint cache".to_string())
        })?,
        log_basis: next_params.log_basis,
        opening_point: sumcheck_challenges.clone(),
        per_handle_lp: None,
        tiered: None,
    }];
    if route_setup_recursively {
        if let (Some(r_setup), Some(payload)) = (r_setup, stage2_setup_claim_reduction.as_ref()) {
            let prepared = prepare_m_eval::<F, D>(
                &stage1_challenges_for_prepare,
                alpha,
                lp,
                &tau1,
                &claim_group_sizes,
                &gamma_for_prepare,
                num_eval_rows_for_prepare,
                opening_points_len,
                &claim_to_point,
            )?;
            let x_challenges = &sumcheck_challenges[ring_bits..];
            let (row_bits, coeff_bits) = prepared.setup_claim_reduction_dims();
            let row_count = 1usize << row_bits;
            let col_count = 1usize;
            let setup_field_len = row_count * D;
            let mut setup_rings =
                materialize_setup_claim_polynomial::<F, D>(&prepared, x_challenges, expanded)?;
            setup_rings.resize(row_count, CyclotomicRing::<F, D>::zero());
            if tier_setup_params.is_tiered() {
                let full_s_lp = untiered_setup_group_lp(next_params, setup_field_len)?;
                let setup_opening_point = setup_claim_opening_point_from_r_setup(
                    &r_setup, row_bits, coeff_bits, &full_s_lp,
                )?;
                let chunk_lp =
                    tiered_setup_group_lp(next_params, setup_field_len, tier_setup_params)?;
                let meta_lp = meta_lp_from_chunks(next_params, &chunk_lp, tier_setup_params)?;
                let tiered_material = build_tiered_handle_material::<F, D>(
                    expanded,
                    ntt_shared,
                    row_count,
                    col_count,
                    &chunk_lp,
                    &meta_lp,
                    tier_setup_params,
                    setup_opening_point.clone(),
                    Some(&setup_rings),
                )?;
                handles.push(RecursivePolyHandle {
                    w: RecursiveWitnessFlat::default(),
                    dense_poly: None,
                    commitment: FlatRingVec::default(),
                    hint: RecursiveCommitmentHintCache::default(),
                    log_basis: next_params.log_basis,
                    opening_point: setup_opening_point,
                    per_handle_lp: None,
                    tiered: Some(tiered_material),
                });
            } else {
                let s_lp = untiered_setup_group_lp(next_params, setup_field_len)?;
                let setup_opening_point =
                    setup_claim_opening_point_from_r_setup(&r_setup, row_bits, coeff_bits, &s_lp)?;
                let (commitment, hint) =
                    commit_dense_s_handle_direct(expanded, &setup_rings, &s_lp)?;
                let dense_poly = FlatRingVec::from_ring_elems::<D>(&setup_rings);
                handles.push(RecursivePolyHandle {
                    w: RecursiveWitnessFlat::default(),
                    dense_poly: Some(dense_poly),
                    commitment,
                    hint,
                    log_basis: next_params.log_basis,
                    opening_point: setup_opening_point,
                    per_handle_lp: Some(s_lp),
                    tiered: None,
                });
            }
            let _ = payload;
        }
    }

    Ok(RootLevelRawOutput {
        y_rings,
        v: quad_eq.v,
        stage1: stage1_proof,
        stage2_sumcheck,
        stage2_setup_claim_reduction,
        w_commitment_proof,
        w_eval,
        next_state: RecursiveProverState { handles },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AkitaProverSetup;
    use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
    use akita_field::Prime128OffsetA7F7;
    use akita_types::AjtaiKeyParams;
    use akita_verifier::derive_tiered_setup_material_for_verifier;

    type TestField = Prime128OffsetA7F7;
    const D_TEST: usize = 32;

    fn test_level_params() -> LevelParams {
        LevelParams {
            ring_dimension: D_TEST,
            log_basis: 1,
            a_key: AjtaiKeyParams::new_unchecked(1, 4, 0, D_TEST),
            b_key: AjtaiKeyParams::new_unchecked(1, 4, 0, D_TEST),
            d_key: AjtaiKeyParams::new_unchecked(1, 4, 0, D_TEST),
            num_blocks: 4,
            block_len: 4,
            m_vars: 2,
            r_vars: 2,
            stage1_config: SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
            stage1_challenge_shape: Stage1ChallengeShape::Flat,
            use_setup_claim_reduction: true,
            num_digits_commit: 1,
            num_digits_open: 1,
            num_digits_fold: 1,
            groups: None,
        }
    }

    #[test]
    fn tiered_handle_material_matches_verifier_derivation() {
        let setup = AkitaProverSetup::<TestField, D_TEST>::generate_with_capacity(8, 4, 2, 4, 1024)
            .expect("setup");
        let next_lp = test_level_params();
        let tier = TieredSetupParams::new(2).expect("tier");
        let row_count = 3usize;
        let col_count = 4usize;
        let setup_field_len = row_count * col_count * D_TEST;
        let full_s_lp = untiered_setup_group_lp(&next_lp, setup_field_len).expect("full s lp");
        let chunk_lp = tiered_setup_group_lp(&next_lp, setup_field_len, tier).expect("chunk lp");
        let meta_lp = meta_lp_from_chunks(&next_lp, &chunk_lp, tier).expect("meta lp");
        let opening_len = D_TEST.trailing_zeros() as usize + full_s_lp.r_vars + full_s_lp.m_vars;
        let opening_point: Vec<TestField> = (0..opening_len)
            .map(|i| TestField::from_u64(17 + i as u64 * 11))
            .collect();

        let prover_material = build_tiered_handle_material::<TestField, D_TEST>(
            &setup.expanded,
            &setup.ntt_shared,
            row_count,
            col_count,
            &chunk_lp,
            &meta_lp,
            tier,
            opening_point,
            None,
        )
        .expect("prover material");
        let verifier_setup = setup.verifier_setup();
        let verifier_material = derive_tiered_setup_material_for_verifier::<TestField, D_TEST>(
            &verifier_setup,
            row_count,
            col_count,
            &chunk_lp,
            &meta_lp,
            tier,
        )
        .expect("verifier material");

        let prover_chunk_commitments = prover_material
            .chunk_commitments
            .iter()
            .map(|u| u.as_ring_slice::<D_TEST>().map(|s| s.to_vec()))
            .collect::<Result<Vec<_>, _>>()
            .expect("chunk commitments");
        assert_eq!(
            prover_chunk_commitments,
            verifier_material.chunk_b_commitments
        );
        assert_eq!(
            prover_material
                .meta_commitment
                .as_ring_slice::<D_TEST>()
                .expect("meta commitment"),
            verifier_material.meta_b_commitment.as_slice()
        );
    }
}
