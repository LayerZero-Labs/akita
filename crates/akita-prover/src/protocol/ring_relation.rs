//! Ring-relation prover for the Akita PCS (§4.2).
//!
//! Builds the stage-1 relation instance and witness (`M`, `y`, `z`, `v`) via
//! [`RingRelationProver`].
use crate::compute::{
    BatchDecomposeFoldOutcome, DecomposeFoldBatchPlan, DecomposeFoldPlan, FlatDigitBlocks,
    OpeningBatchKernel, OpeningFoldKernel, OperationCtx, RootOpeningSource, RootPolyMeta,
    RuntimeOpeningProveBackendFor,
};
use crate::validation::validate_i8_setup_log_basis;
use crate::{
    CyclicRowsComputeBackend, DecomposeFoldWitness, DigitRowsComputeBackend, ProverOpeningData,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, SparseChallenge};
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt, HalvingField};
use akita_transcript::labels::{ABSORB_PROVER_V, ABSORB_TERMINAL_E_HAT};
use akita_transcript::Transcript;
use akita_types::{
    assemble_relation_y, dispatch_ring_dim_result, CommitmentRingDims, RelationYLayout, RingVec,
    RingView,
};
use akita_types::{gadget_row_scalars, AkitaCommitmentHint, MRowLayout};
use akita_types::{LevelParams, RingRelationInstance};
use akita_types::{RingMultiplierOpeningPoint, RingOpeningPoint};

use super::fold_grind::{self, ProverTranscriptGrind};
use super::ring_relation_witness::RingRelationWitness;
use std::time::Instant;

mod relation_quotient;
mod repeated_b;

pub use relation_quotient::{compute_relation_quotient, RelationQuotientShape};

fn absorb_terminal_e_folded_fields<F, T>(
    transcript: &mut T,
    e_folded: &RingVec<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + akita_serialization::AkitaSerialize,
    T: Transcript<F>,
{
    let bytes = akita_types::e_folded_segment_bytes::<F>(e_folded)?;
    if bytes.is_empty() {
        return Err(AkitaError::InvalidInput(
            "terminal e_folded absorb cannot be empty".to_string(),
        ));
    }
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_E_HAT, &bytes);
    Ok(())
}

fn decompose_e_hat<F: FieldCore + CanonicalField, const D: usize>(
    pre_folded_e: &[&[CyclotomicRing<F, D>]],
    depth_open: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError> {
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(depth_open, log_basis, q);
    let total_rows: usize = pre_folded_e.iter().map(|rows| rows.len()).sum();
    let mut e_hat = FlatDigitBlocks::zeroed(vec![depth_open; total_rows])?;
    let mut offset = 0usize;
    for folded_rows in pre_folded_e {
        for w_i in *folded_rows {
            w_i.balanced_decompose_pow2_i8_into_with_params(
                &mut e_hat.flat_digits_mut()[offset..offset + depth_open],
                &decompose_params,
            );
            offset += depth_open;
        }
    }
    Ok(e_hat)
}

/// Concatenate per-group D-free commitment hints into one batched hint covering
/// all claims in claim order.
///
/// Recomposed inner rows are no longer cached on the hint (S4/S5): they are
/// recomputed on demand from the decomposed digit stream
/// ([`recompose_hint_inner_rows`] / [`crate::compute::recompose_flat_hint_inner_rows`]).
fn flatten_commitment_hints_for_ring_relation<F>(
    hints: Vec<AkitaCommitmentHint<F>>,
    group_sizes: &[usize],
) -> Result<AkitaCommitmentHint<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if hints.len() != group_sizes.len() {
        return Err(AkitaError::InvalidInput(
            "prover hint group count does not match commitment groups".to_string(),
        ));
    }

    let mut decomposed_inner_rows = Vec::new();
    for (hint, &group_size) in hints.into_iter().zip(group_sizes.iter()) {
        let digits_by_poly = hint.into_parts();
        if digits_by_poly.len() != group_size {
            return Err(AkitaError::InvalidInput(
                "prover hint group sizes do not match polynomial groups".to_string(),
            ));
        }
        decomposed_inner_rows.extend(digits_by_poly);
    }

    Ok(AkitaCommitmentHint::new(decomposed_inner_rows))
}

pub(super) fn aggregate_decompose_fold_witnesses<F: FieldCore, const D: usize>(
    witnesses: Vec<DecomposeFoldWitness<F>>,
) -> Result<DecomposeFoldWitness<F>, AkitaError> {
    let Some((first, rest)) = witnesses.split_first() else {
        return Err(AkitaError::InvalidInput(
            "batched decompose_fold requires at least one witness".to_string(),
        ));
    };
    first.ensure_ring_dim::<D>()?;
    let row_count = first.row_count();
    let mut z_folded_rings = first.z_folded_rings_trusted::<D>().to_vec();
    let mut centered_coeffs = first.centered_coeffs_owned::<D>();

    for witness in rest {
        witness.ensure_ring_dim::<D>()?;
        if witness.row_count() != row_count {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold witness length mismatch".to_string(),
            ));
        }
        for (dst, src) in z_folded_rings
            .iter_mut()
            .zip(witness.z_folded_rings_trusted::<D>())
        {
            *dst += *src;
        }
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs_trusted::<D>())
        {
            for k in 0..D {
                dst[k] = dst[k].checked_add(src[k]).ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "batched decompose_fold centered coefficient overflow".to_string(),
                    )
                })?;
            }
        }
    }

    let centered_inf_norm = centered_coeffs
        .iter()
        .flat_map(|coeffs| coeffs.iter())
        .map(|coeff| coeff.unsigned_abs())
        .max()
        .unwrap_or(0);

    Ok(DecomposeFoldWitness::from_parts(
        z_folded_rings,
        centered_coeffs,
        centered_inf_norm,
    ))
}

/// Extracted level numbers consumed by [`build_point_decompose_fold_witness`].
///
/// A-role fold shape: `block_len` / `num_digits_commit` / `log_basis` are the
/// only schedule quantities the point-fold kernel reads. Callers extract them
/// from the schedule (`from_level`); the kernel must not read schedule types.
#[derive(Debug, Clone, Copy)]
pub(super) struct PointFoldShape {
    /// Ring elements per fold block (A-role row length).
    pub block_len: usize,
    /// Digit planes in the commit decomposition.
    pub num_digits_commit: usize,
    /// Log2 of the decomposition basis.
    pub log_basis: u32,
}

impl PointFoldShape {
    /// Extract the point-fold shape from one schedule level.
    pub(super) fn from_level(lp: &LevelParams) -> Self {
        Self {
            block_len: lp.block_len,
            num_digits_commit: lp.num_digits_commit,
            log_basis: lp.log_basis,
        }
    }
}

pub(super) fn build_point_decompose_fold_witness<F, P, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    challenges: &Challenges,
    point_polys: &[&P],
    point_indices: &[usize],
    shape: PointFoldShape,
) -> Result<DecomposeFoldWitness<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
    P: RootOpeningSource<F, D>,
    B: crate::compute::ComputeBackendSetup<F>
        + for<'a> OpeningBatchKernel<P::OpeningBatchView<'a>, F, D>
        + for<'a> OpeningFoldKernel<P::OpeningView<'a>, F, D>,
{
    match challenges {
        Challenges::Sparse {
            challenges: sparse,
            num_blocks_per_claim,
            ..
        } => {
            let mut point_challenges =
                Vec::with_capacity(point_indices.len() * *num_blocks_per_claim);
            for &claim_idx in point_indices {
                let start = claim_idx
                    .checked_mul(*num_blocks_per_claim)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("batched challenge offset overflow".to_string())
                    })?;
                let end = start.checked_add(*num_blocks_per_claim).ok_or_else(|| {
                    AkitaError::InvalidSetup("batched challenge offset overflow".to_string())
                })?;
                point_challenges.extend_from_slice(sparse.get(start..end).ok_or(
                    AkitaError::InvalidSize {
                        expected: end,
                        actual: sparse.len(),
                    },
                )?);
            }
            let batch_view = P::opening_batch(point_polys)?;
            match OpeningBatchKernel::decompose_fold_batch(
                backend,
                prepared,
                batch_view,
                DecomposeFoldBatchPlan::Sparse {
                    challenges: &point_challenges,
                    block_len: shape.block_len,
                    num_digits: shape.num_digits_commit,
                    log_basis: shape.log_basis,
                },
            )? {
                BatchDecomposeFoldOutcome::Fused(z_point) => Ok(z_point),
                BatchDecomposeFoldOutcome::FallbackPerPoly => {
                    let witnesses: Vec<DecomposeFoldWitness<F>> = point_polys
                        .iter()
                        .zip(point_challenges.chunks(*num_blocks_per_claim))
                        .map(|(poly, poly_challenges)| -> Result<_, AkitaError> {
                            OpeningFoldKernel::decompose_fold(
                                backend,
                                prepared,
                                poly.opening_view()?,
                                DecomposeFoldPlan {
                                    challenges: poly_challenges,
                                    block_len: shape.block_len,
                                    num_digits: shape.num_digits_commit,
                                    log_basis: shape.log_basis,
                                },
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    aggregate_decompose_fold_witnesses::<F, D>(witnesses)
                }
                BatchDecomposeFoldOutcome::Unsupported => Err(AkitaError::InvalidSetup(
                    "sparse batched fold is unsupported for this polynomial backend".to_string(),
                )),
            }
        }
        Challenges::Tensor { factored: _ } => {
            let selected = challenges.select_claims::<D>(point_indices)?;
            let point_factored = match selected {
                Challenges::Tensor { factored } => factored,
                Challenges::Sparse { .. } => {
                    return Err(AkitaError::InvalidSetup(
                        "tensor claim selection returned sparse challenges".to_string(),
                    ))
                }
            };
            let batch_view = P::opening_batch(point_polys)?;
            match OpeningBatchKernel::decompose_fold_batch(
                backend,
                prepared,
                batch_view,
                DecomposeFoldBatchPlan::Tensor {
                    tensor: &point_factored,
                    block_len: shape.block_len,
                    num_digits: shape.num_digits_commit,
                    log_basis: shape.log_basis,
                },
            )? {
                BatchDecomposeFoldOutcome::Fused(witness) => Ok(witness),
                BatchDecomposeFoldOutcome::FallbackPerPoly
                | BatchDecomposeFoldOutcome::Unsupported => Err(AkitaError::InvalidSetup(
                    "polynomial backend has no tensor-shaped fold kernel".to_string(),
                )),
            }
        }
    }
}

fn compute_v_rows<F, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup,
    row_len: usize,
    e_hat: &FlatDigitBlocks<D>,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    let rows = backend.digit_rows::<D>(prepared, row_len, e_hat.flat_digits(), log_basis)?;
    if rows.len() != row_len {
        return Err(AkitaError::InvalidProof);
    }
    Ok(rows)
}

/// Compute and absorb the D-block rows `v = D * e_hat`.
///
/// D-role kernel: `d_row_len` is the D-matrix row count and `e_hat` carries
/// the opening digits at the D-role ring dimension. Callers extract both from
/// the schedule; this function must not read schedule types.
fn compute_v_rows_for_layout<F, T, RB, const D: usize>(
    ring_switch_ctx: &OperationCtx<'_, F, RB>,
    transcript: &mut T,
    d_row_len: usize,
    log_basis: u32,
    e_hat: &FlatDigitBlocks<D>,
    m_row_layout: MRowLayout,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    RB: DigitRowsComputeBackend<F>,
{
    let backend = ring_switch_ctx.backend();
    let prepared = ring_switch_ctx.prepared();
    match m_row_layout {
        MRowLayout::WithDBlock => {
            let _span = tracing::info_span!(
                "compute_relation_v",
                e_hat_planes = e_hat.flat_digits().len()
            )
            .entered();
            let v = compute_v_rows(backend, prepared, d_row_len, e_hat, log_basis)?;
            // Absorb `v` via the canonical D-free flat encoder (byte-identical to
            // the former `RingSliceSerializer` typed path; S2 byte-identity test).
            akita_types::RingVec::from_ring_elems(&v).append_flat_to_transcript(
                ABSORB_PROVER_V,
                D,
                transcript,
            )?;
            Ok(v)
        }
        MRowLayout::WithoutDBlock => Ok(Vec::new()),
    }
}

/// Validate the chunked-witness configuration at the prover boundary (no-panic
/// contract), before any witness math. Mirrors the planner entry guard and the
/// verifier layout resolution.
pub(crate) fn validate_chunked_witness_cfg(lp: &LevelParams) -> Result<(), AkitaError> {
    lp.witness_chunk.validate()?;
    let w = lp.witness_chunk.num_chunks;
    if w > 1 {
        if !lp.num_blocks.is_multiple_of(w) {
            return Err(AkitaError::InvalidSetup(
                "witness chunk count must divide num_blocks".to_string(),
            ));
        }
        if !(lp.num_blocks / w).is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "witness chunk block window must be a power of two".to_string(),
            ));
        }
        if lp.tier_split > 1 {
            return Err(AkitaError::InvalidSetup(
                "multi-chunk witness layout for tiered commitments is not supported".to_string(),
            ));
        }
    }
    Ok(())
}

/// Restrict sparse fold challenges to one chunk's global block window
/// `[chunk·blocks_per_chunk, (chunk+1)·blocks_per_chunk)`, zeroing all other
/// blocks. Folding under these yields the partial response `z_i = Σ_{j∈I_i}
/// c_j s_j`.
pub(super) fn window_sparse_challenges(
    challenges: &Challenges,
    chunk: usize,
    blocks_per_chunk: usize,
) -> Result<Challenges, AkitaError> {
    match challenges {
        Challenges::Sparse {
            challenges: sparse,
            num_blocks_per_claim,
            num_claims,
        } => {
            let lo = chunk * blocks_per_chunk;
            let hi = lo + blocks_per_chunk;
            let windowed: Vec<SparseChallenge> = sparse
                .iter()
                .enumerate()
                .map(|(idx, ch)| {
                    let block = idx % num_blocks_per_claim;
                    if (lo..hi).contains(&block) {
                        ch.clone()
                    } else {
                        SparseChallenge {
                            positions: Vec::new(),
                            coeffs: Vec::new(),
                        }
                    }
                })
                .collect();
            Challenges::from_sparse(windowed, *num_blocks_per_claim, *num_claims)
        }
        Challenges::Tensor { .. } => Err(AkitaError::InvalidSetup(
            "chunked fold response requires sparse fold challenges".to_string(),
        )),
    }
}

/// Prover-side builder for the ring relation $M(x) \cdot z = y(x) + (X^D + 1) \cdot r(x)$.
pub struct RingRelationProver;

impl RingRelationProver {
    /// Root-level constructor for one shared opening point with one or more
    /// polynomial slots.
    ///
    /// `opening_point` is the single ring-level opening point used by the
    /// batch.
    /// For the trivial single-claim case use `polys = &[poly]` and
    /// `gamma = vec![F::one()]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the batched hints, folded witnesses, or decomposed
    /// aggregate witness are malformed.
    ///
    /// # Panics
    ///
    /// Panics if the batched `e_hat` decomposition or flattened batched hints
    /// produced by the prover do not preserve the expected block sizes.  These
    /// invariants hold by construction for well-formed inputs accepted by the
    /// error checks above and are therefore treated as internal programming
    /// errors rather than recoverable failures.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    #[tracing::instrument(skip_all, name = "RingRelationProver::new")]
    #[inline(never)]
    pub fn new<'a, F, PointF, T, P, OB, RB>(
        opening_ctx: &OperationCtx<'_, F, OB>,
        ring_switch_ctx: &OperationCtx<'_, F, RB>,
        opening_point: RingOpeningPoint<F>,
        ring_multiplier_point: RingMultiplierOpeningPoint<F>,
        fold_claims: ProverOpeningData<'a, PointF, P, F>,
        pre_folded_e_by_poly: Vec<RingVec<F>>,
        lp: LevelParams,
        transcript: &mut T,
        row_coefficient_rings: RingVec<F>,
        m_row_layout: MRowLayout,
        terminal_tail_t_vectors: Option<usize>,
    ) -> Result<(RingRelationInstance<F>, RingRelationWitness<F>), AkitaError>
    where
        F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        PointF: Clone,
        T: Transcript<F> + ProverTranscriptGrind<F>,
        P: RootOpeningSource<F, 32>
            + RootOpeningSource<F, 64>
            + RootOpeningSource<F, 128>
            + RootOpeningSource<F, 256>
            + RootPolyMeta<F>,
        OB: DigitRowsComputeBackend<F> + RuntimeOpeningProveBackendFor<F, P>,
        RB: DigitRowsComputeBackend<F>,
    {
        validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
        validate_chunked_witness_cfg(&lp)?;
        // Per-role ring dimensions for this level; the mixed-row spec feeds
        // diverging role dims here (uniform today).
        let dims = CommitmentRingDims::uniform(lp.ring_dimension);
        let opening_batch = fold_claims.opening_claims().layout();
        let polys = fold_claims.flat_polys();
        let group_sizes = opening_batch.group_sizes();
        let num_groups = fold_claims.opening_claims().num_groups();
        let mut hints = Vec::with_capacity(num_groups);
        // Sent-commitment rows are B-role data; keep them as flat coefficients
        // and validate the ring count under `d_b` (no-panic length gate).
        let mut commitment_row_coeffs: Vec<F> = Vec::new();
        for group_index in 0..num_groups {
            let group_commitment = fold_claims.opening_claims().group_commitment(group_index)?;
            let group_hint = fold_claims.group_hint(group_index)?;
            let group_rows =
                RingView::new(group_commitment.rows().coeffs(), dims.d_b())?.num_rings();
            if group_rows != lp.effective_commit_rows() {
                return Err(AkitaError::InvalidInput(
                    "batched prover received a commitment with the wrong length".to_string(),
                ));
            }
            commitment_row_coeffs.extend_from_slice(group_commitment.rows().coeffs());
            hints.push(group_hint.clone());
        }
        let commitment_rows = RingVec::from_coeffs(commitment_row_coeffs);
        if opening_point.a.len() < lp.block_len || opening_point.b.len() != lp.num_blocks {
            return Err(AkitaError::InvalidInput(
                "batched prover opening-point layout mismatch".to_string(),
            ));
        }
        if ring_multiplier_point.a_len() < lp.block_len
            || ring_multiplier_point.b_len() != lp.num_blocks
        {
            return Err(AkitaError::InvalidInput(
                "batched prover ring-multiplier opening-point layout mismatch".to_string(),
            ));
        }
        let num_claims = opening_batch.num_total_polynomials();
        if polys.is_empty() {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one polynomial".to_string(),
            ));
        }
        if polys.len() != pre_folded_e_by_poly.len() || polys.len() != num_claims {
            return Err(AkitaError::InvalidInput(
                "batched prover input lengths do not match".to_string(),
            ));
        }
        // Row-coefficient rings are A-role data (fold coefficients).
        if !row_coefficient_rings.can_decode_vec(dims.d_a())
            || row_coefficient_rings.coeff_len() / dims.d_a() != num_claims
        {
            return Err(AkitaError::InvalidInput(
                "batched prover row coefficient length does not match claim count".to_string(),
            ));
        }
        let gamma = row_coefficient_rings
            .coeffs()
            .iter()
            .copied()
            .step_by(dims.d_a())
            .collect::<Vec<_>>();

        // Extracted level numbers for the D-role and fused-y operations below;
        // the kernels inside the dispatch arms must not read schedule types.
        let num_digits_open = lp.num_digits_open;
        let log_basis = lp.log_basis;
        let d_row_len = lp.d_key.row_len();
        let effective_commit_rows = lp.effective_commit_rows();
        let b_inner_rows_per_group = lp.b_inner_rows_per_group();
        let n_a = lp.a_key.row_len();

        // D-role operations: decompose the folded opening rows into `e_hat`
        // digits and (non-terminal layouts) compute + absorb the D-block rows
        // `v = D * e_hat`. Both consume the same digits at `d_d`, so they share
        // one kernel-entry dispatch; the flat `FlatDigitBlocks` / `RingVec` come
        // back out as D-free carriers.
        //
        // Terminal layout drops the D-block from the M-matrix entirely:
        // `v = D · e_hat` never travels on the wire, the verifier never
        // reconstructs it, and downstream prover paths (`ring_switch_build_w`,
        // `relation_claim_from_rows_extension`) consume an empty `v` slice.
        // Skip the D-NTT under Terminal.
        let (e_hat, v) = dispatch_ring_dim_result!(dims.d_d(), |D_D| {
            let pre_folded_typed = pre_folded_e_by_poly
                .iter()
                .map(RingVec::as_ring_slice::<D_D>)
                .collect::<Result<Vec<_>, _>>()?;
            let e_hat_typed = {
                let _span = tracing::info_span!("decompose_batched_e_hat").entered();
                decompose_e_hat::<F, D_D>(&pre_folded_typed, num_digits_open, log_basis)?
            };
            let v_typed = compute_v_rows_for_layout::<F, T, RB, D_D>(
                ring_switch_ctx,
                transcript,
                d_row_len,
                log_basis,
                &e_hat_typed,
                m_row_layout,
            )?;
            Ok::<_, AkitaError>((
                e_hat_typed.into_digit_blocks(),
                RingVec::from_ring_elems(&v_typed),
            ))
        })?;
        let flattened_hint = flatten_commitment_hints_for_ring_relation::<F>(hints, &group_sizes)?;
        let opening_backend = opening_ctx.backend();

        // Concatenated folded `e` rows in claim order (flat A-role storage).
        let e_folded = RingVec::from_coeffs(
            pre_folded_e_by_poly
                .iter()
                .flat_map(|block| block.coeffs().iter().copied())
                .collect(),
        );
        if matches!(m_row_layout, MRowLayout::WithoutDBlock) {
            absorb_terminal_e_folded_fields::<F, T>(transcript, &e_folded)?;
        }
        // Distributed-prover chunked layout: the grind emits one folded response
        // per block window (`z_i`), and the global response is their sum
        // (`Σ_i z_i = z`, exact coefficient-wise i32 accumulation).
        let (z_folded_rings, z_folded_centered_per_chunk, challenges, fold_grind_nonce) =
            fold_grind::sample_fold_decompose_witness::<F, _, OB, T>(
                opening_backend,
                Some(opening_ctx.prepared()),
                transcript,
                &polys,
                &lp,
                num_claims,
                terminal_tail_t_vectors,
            )?;

        // `y` spans roles (D rows | COMMIT rows | B_inner zeros | A zeros).
        // Terminal levels drop the D-block from M entirely, so `n_d` is zero
        // and `v` stays empty.
        let n_d_active = match m_row_layout {
            MRowLayout::WithDBlock => d_row_len,
            MRowLayout::WithoutDBlock => 0,
        };
        let y = assemble_relation_y::<F>(
            dims,
            RelationYLayout {
                n_d: n_d_active,
                commit_rows_per_group: effective_commit_rows,
                b_inner_rows_per_group,
                n_a,
            },
            &v,
            &commitment_rows,
        )?;

        let instance = RingRelationInstance::new(
            m_row_layout,
            challenges,
            opening_point,
            ring_multiplier_point,
            opening_batch,
            gamma,
            row_coefficient_rings,
            y,
            v,
            dims,
        )?;
        instance.check_v_shape_for_level(&lp)?;
        let witness = RingRelationWitness::from_flat_parts(
            z_folded_rings,
            z_folded_centered_per_chunk,
            fold_grind_nonce,
            e_hat,
            e_folded,
            flattened_hint,
            dims.uniform_dim()?,
        );
        Ok((instance, witness))
    }
}
