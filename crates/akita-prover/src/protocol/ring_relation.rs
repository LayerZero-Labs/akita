//! Ring-relation prover for the Akita PCS (§4.2).
//!
//! Builds the stage-1 relation instance and witness (`M`, `y`, `z`, `v`) via
//! [`RingRelationProver`].
use crate::compute::{
    BatchDecomposeFoldOutcome, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OperationCtx, RootOpeningSource,
};
use crate::validation::validate_i8_setup_log_basis;
use crate::{
    CyclicRowsComputeBackend, DecomposeFoldWitness, DigitRowsComputeBackend, ProverOpeningBatch,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, IntegerChallenge, SparseChallenge};
use akita_field::parallel::*;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt, HalvingField};
use akita_transcript::labels::{ABSORB_PROVER_V, ABSORB_TERMINAL_E_HAT};
use akita_transcript::Transcript;
use akita_types::{
    gadget_row_scalars, AkitaCommitmentHint, ErasedCommitmentHint, FlatDigitBlocks, FlatRingVec,
    MRowLayout, OpeningBatchShape, RingBuf, RingSliceSerializer,
};
use akita_types::{LevelParams, RingRelationInstance};
use akita_types::{RingMultiplierOpeningPoint, RingOpeningPoint};

use super::fold_grind::{self, ProverTranscriptGrind};
use super::ring_relation_witness::RingRelationWitness;
use std::time::Instant;

mod relation_quotient;
mod repeated_b;

pub use akita_types::generate_y;
pub use relation_quotient::compute_relation_quotient;

type RingRelationProveOutput<F> = (
    RingRelationInstance<F>,
    RingRelationWitness<F>,
    FlatRingVec<F>,
);

fn absorb_terminal_e_folded_fields<F, T, const D: usize>(
    transcript: &mut T,
    e_folded: &[CyclotomicRing<F, D>],
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + akita_serialization::AkitaSerialize,
    T: Transcript<F>,
{
    let bytes = akita_types::e_folded_segment_bytes::<F, D>(e_folded)?;
    if bytes.is_empty() {
        return Err(AkitaError::InvalidInput(
            "terminal e_folded absorb cannot be empty".to_string(),
        ));
    }
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_E_HAT, &bytes);
    Ok(())
}

fn decompose_e_hat<F: FieldCore + CanonicalField, const D: usize>(
    pre_folded_e: &[Vec<CyclotomicRing<F, D>>],
    depth_open: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks, AkitaError> {
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(depth_open, log_basis, q);
    let total_rows: usize = pre_folded_e.iter().map(Vec::len).sum();
    let mut e_hat = FlatDigitBlocks::zeroed::<D>(vec![depth_open; total_rows])?;
    let mut offset = 0usize;
    for folded_rows in pre_folded_e {
        for w_i in folded_rows {
            w_i.balanced_decompose_pow2_i8_into_with_params(
                &mut e_hat.flat_digits_trusted_mut::<D>()[offset..offset + depth_open],
                &decompose_params,
            );
            offset += depth_open;
        }
    }
    Ok(e_hat)
}

fn flatten_commitment_hints_for_ring_relation<F, const D: usize>(
    hints: Vec<AkitaCommitmentHint<F, D>>,
    group_sizes: &[usize],
    num_digits_open: usize,
    log_basis: u32,
) -> Result<AkitaCommitmentHint<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if hints.len() != group_sizes.len() {
        return Err(AkitaError::InvalidInput(
            "prover hint group count does not match commitment groups".to_string(),
        ));
    }

    let mut decomposed_inner_rows = Vec::new();
    let mut t_rows_by_poly = Vec::new();
    for (mut hint, &group_size) in hints.into_iter().zip(group_sizes.iter()) {
        if hint.decomposed_inner_rows.len() != group_size {
            return Err(AkitaError::InvalidInput(
                "prover hint group sizes do not match polynomial groups".to_string(),
            ));
        }
        hint.ensure_recomposed_inner_rows(num_digits_open, log_basis)?;
        let (digits_by_poly, rows_by_poly) = hint.into_parts();
        decomposed_inner_rows.extend(digits_by_poly);
        let rows_by_poly = rows_by_poly.ok_or_else(|| {
            AkitaError::InvalidInput("missing recomposed inner rows in prover hint".to_string())
        })?;
        t_rows_by_poly.extend(rows_by_poly);
    }

    Ok(AkitaCommitmentHint::with_recomposed_inner_rows(
        decomposed_inner_rows,
        t_rows_by_poly,
    ))
}

fn aggregate_decompose_fold_witnesses<F: FieldCore, const D: usize>(
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

pub(super) fn build_point_decompose_fold_witness<F, P, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    challenges: &Challenges,
    point_polys: &[&P],
    point_indices: &[usize],
    lp: &LevelParams,
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
                    block_len: lp.block_len,
                    num_digits: lp.num_digits_commit,
                    log_basis: lp.log_basis,
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
                                    block_len: lp.block_len,
                                    num_digits: lp.num_digits_commit,
                                    log_basis: lp.log_basis,
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
                    block_len: lp.block_len,
                    num_digits: lp.num_digits_commit,
                    log_basis: lp.log_basis,
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
    e_hat: &FlatDigitBlocks,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    let rows = backend.digit_rows::<D>(
        prepared,
        row_len,
        e_hat.flat_digits_trusted::<D>(),
        log_basis,
    )?;
    if rows.len() != row_len {
        return Err(AkitaError::InvalidProof);
    }
    Ok(rows)
}

fn compute_v_rows_for_layout<F, T, RB, const D: usize>(
    ring_switch_ctx: &OperationCtx<'_, F, RB>,
    transcript: &mut T,
    lp: &LevelParams,
    e_hat: &FlatDigitBlocks,
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
            let _span =
                tracing::info_span!("compute_relation_v", e_hat_planes = e_hat.plane_count())
                    .entered();
            let v = compute_v_rows(backend, prepared, lp.d_key.row_len(), e_hat, lp.log_basis)?;
            transcript.append_serde(ABSORB_PROVER_V, &RingSliceSerializer(&v));
            Ok(v)
        }
        MRowLayout::WithoutDBlock => Ok(Vec::new()),
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
    pub fn new<'a, F, PointF, const D: usize, T, P, OB, RB>(
        opening_ctx: &OperationCtx<'_, F, OB>,
        ring_switch_ctx: &OperationCtx<'_, F, RB>,
        opening_point: RingOpeningPoint<F>,
        ring_multiplier_point: RingMultiplierOpeningPoint<F>,
        fold_claims: ProverOpeningBatch<'a, PointF, P, F, D>,
        pre_folded_e_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
        lp: LevelParams,
        transcript: &mut T,
        row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
        m_row_layout: MRowLayout,
        terminal_tail_t_vectors: Option<usize>,
    ) -> Result<RingRelationProveOutput<F>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        PointF: Clone,
        T: Transcript<F> + ProverTranscriptGrind<F>,
        P: RootOpeningSource<F, D>,
        OB: DigitRowsComputeBackend<F>
            + for<'b> OpeningBatchKernel<P::OpeningBatchView<'b>, F, D>
            + for<'b> OpeningFoldKernel<P::OpeningView<'b>, F, D>,
        RB: DigitRowsComputeBackend<F>,
    {
        validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
        let opening_batch = fold_claims.to_opening_shape::<F>()?;
        let polys = fold_claims.flat_polys();
        let group_sizes = opening_batch.num_polys_per_commitment_group();
        let commitment_flat = fold_claims.single_fold_commitment()?;
        let commitment_rows = commitment_flat.as_ring_slice_trusted::<D>();
        if commitment_rows.len() != lp.effective_commit_rows() {
            return Err(AkitaError::InvalidInput(
                "batched prover received a commitment with the wrong length".to_string(),
            ));
        }
        let mut hints = Vec::with_capacity(fold_claims.groups.len());
        for group in &fold_claims.groups {
            hints.push(group.commitment.1.clone());
        }
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
        let num_claims = opening_batch.num_polynomials();
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
        if row_coefficient_rings.len() != num_claims {
            return Err(AkitaError::InvalidInput(
                "batched prover row coefficient length does not match claim count".to_string(),
            ));
        }
        let gamma = row_coefficient_rings
            .iter()
            .map(|ring| ring.coefficients()[0])
            .collect::<Vec<_>>();

        let e_hat = {
            let _span = tracing::info_span!("decompose_batched_e_hat").entered();
            decompose_e_hat::<F, D>(&pre_folded_e_by_poly, lp.num_digits_open, lp.log_basis)?
        };
        let flattened_hint = flatten_commitment_hints_for_ring_relation::<F, D>(
            hints,
            &group_sizes,
            lp.num_digits_open,
            lp.log_basis,
        )?;

        // Terminal layout drops the D-block from the M-matrix entirely:
        // `v = D · e_hat` never travels on the wire, the verifier never
        // reconstructs it, and downstream prover paths (`ring_switch_build_w`,
        // `relation_claim_from_rows_extension`) consume an empty `v` slice.
        // Skip the D-NTT under Terminal.
        let opening_backend = opening_ctx.backend();
        let v = compute_v_rows_for_layout::<F, T, RB, D>(
            ring_switch_ctx,
            transcript,
            &lp,
            &e_hat,
            m_row_layout,
        )?;

        if matches!(m_row_layout, MRowLayout::WithoutDBlock) {
            let e_folded_flat: Vec<CyclotomicRing<F, D>> = pre_folded_e_by_poly
                .iter()
                .flat_map(|block| block.iter().cloned())
                .collect();
            absorb_terminal_e_folded_fields::<F, T, D>(transcript, &e_folded_flat)?;
        }
        let (z_folded_rings, challenges, fold_grind_nonce) =
            fold_grind::sample_fold_decompose_witness::<F, _, OB, T, D>(
                opening_backend,
                Some(opening_ctx.prepared()),
                transcript,
                &polys,
                &lp,
                num_claims,
                terminal_tail_t_vectors,
            )?;

        // Terminal levels drop the D-block from M entirely, so `y` must
        // also drop the D-rows (the `v = D · ŵ` segment). Pass an empty
        // `v` slice with `n_d_active = 0` so `generate_y` emits
        // `[consistency | commitment_rows | A-zeros]` (no D-block).
        let (y_v_slice, n_d_active) = match m_row_layout {
            MRowLayout::WithDBlock => (v.as_slice(), lp.d_key.row_len()),
            MRowLayout::WithoutDBlock => (&[][..], 0usize),
        };
        let y = generate_y::<F, D>(
            y_v_slice,
            commitment_rows,
            n_d_active,
            lp.effective_commit_rows(),
            lp.b_inner_rows_per_group(),
            lp.a_key.row_len(),
        )?;
        let e_folded = pre_folded_e_by_poly.into_iter().flatten().collect();

        let instance = RingRelationInstance::new_from_rings::<D>(
            m_row_layout,
            challenges,
            opening_point,
            ring_multiplier_point,
            opening_batch,
            gamma,
            row_coefficient_rings,
            y,
            v,
        )?;
        instance.check_v_shape_for_level::<D>(&lp)?;
        let witness = RingRelationWitness::from_typed(
            z_folded_rings,
            fold_grind_nonce,
            e_hat,
            e_folded,
            flattened_hint,
        );
        Ok((instance, witness, commitment_flat))
    }

    /// Build the ring-relation instance and witness for one suffix fold level.
    ///
    /// Commitment rows are read from `commitment` only; no [`ProverOpeningBatch`]
    /// shell is required.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "RingRelationProver::new_suffix")]
    #[inline(never)]
    pub fn new_suffix<'a, F, T, P, OB, RB, const D: usize>(
        opening_ctx: &OperationCtx<'_, F, OB>,
        ring_switch_ctx: &OperationCtx<'_, F, RB>,
        opening_point: RingOpeningPoint<F>,
        ring_multiplier_point: RingMultiplierOpeningPoint<F>,
        commitment: FlatRingVec<F>,
        hint: ErasedCommitmentHint<F>,
        polys: &[&'a P],
        pre_folded_e_by_poly: Vec<Vec<CyclotomicRing<F, D>>>,
        lp: LevelParams,
        transcript: &mut T,
        row_coefficient_rings: Vec<CyclotomicRing<F, D>>,
        opening_batch: OpeningBatchShape,
        m_row_layout: MRowLayout,
        terminal_tail_t_vectors: Option<usize>,
    ) -> Result<RingRelationProveOutput<F>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F> + ProverTranscriptGrind<F>,
        P: RootOpeningSource<F, D>,
        OB: DigitRowsComputeBackend<F>
            + for<'b> OpeningBatchKernel<P::OpeningBatchView<'b>, F, D>
            + for<'b> OpeningFoldKernel<P::OpeningView<'b>, F, D>,
        RB: DigitRowsComputeBackend<F>,
    {
        validate_i8_setup_log_basis(lp.log_basis, "for i8 prover decomposition")?;
        let num_claims = polys.len();
        if num_claims != 1 {
            return Err(AkitaError::InvalidInput(
                "suffix fold claims require exactly one polynomial".to_string(),
            ));
        }
        let commitment_rows = commitment.as_ring_slice_trusted::<D>();
        if commitment_rows.len() != lp.effective_commit_rows() {
            return Err(AkitaError::InvalidInput(
                "suffix fold commitment has the wrong length".to_string(),
            ));
        }
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
        if polys.is_empty() {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one polynomial".to_string(),
            ));
        }
        if polys.len() != pre_folded_e_by_poly.len() {
            return Err(AkitaError::InvalidInput(
                "batched prover input lengths do not match".to_string(),
            ));
        }
        if row_coefficient_rings.len() != num_claims {
            return Err(AkitaError::InvalidInput(
                "batched prover row coefficient length does not match claim count".to_string(),
            ));
        }
        let gamma = row_coefficient_rings
            .iter()
            .map(|ring| ring.coefficients()[0])
            .collect::<Vec<_>>();

        let e_hat = {
            let _span = tracing::info_span!("decompose_batched_e_hat").entered();
            decompose_e_hat::<F, D>(&pre_folded_e_by_poly, lp.num_digits_open, lp.log_basis)?
        };
        let mut hint = hint;
        hint.ensure_ring_dim::<D>()?;
        hint.ensure_recomposed_inner_rows::<D>(lp.num_digits_open, lp.log_basis)?;

        let opening_backend = opening_ctx.backend();
        let v = compute_v_rows_for_layout::<F, T, RB, D>(
            ring_switch_ctx,
            transcript,
            &lp,
            &e_hat,
            m_row_layout,
        )?;

        if matches!(m_row_layout, MRowLayout::WithoutDBlock) {
            let e_folded_flat: Vec<CyclotomicRing<F, D>> = pre_folded_e_by_poly
                .iter()
                .flat_map(|block| block.iter().cloned())
                .collect();
            absorb_terminal_e_folded_fields::<F, T, D>(transcript, &e_folded_flat)?;
        }
        let (z_folded_rings, challenges, fold_grind_nonce) =
            fold_grind::sample_fold_decompose_witness::<F, _, OB, T, D>(
                opening_backend,
                Some(opening_ctx.prepared()),
                transcript,
                polys,
                &lp,
                num_claims,
                terminal_tail_t_vectors,
            )?;

        let (y_v_slice, n_d_active) = match m_row_layout {
            MRowLayout::WithDBlock => (v.as_slice(), lp.d_key.row_len()),
            MRowLayout::WithoutDBlock => (&[][..], 0usize),
        };
        let y = generate_y::<F, D>(
            y_v_slice,
            commitment_rows,
            n_d_active,
            lp.effective_commit_rows(),
            lp.b_inner_rows_per_group(),
            lp.a_key.row_len(),
        )?;
        let e_folded: Vec<CyclotomicRing<F, D>> =
            pre_folded_e_by_poly.into_iter().flatten().collect();

        let instance = RingRelationInstance::new_from_rings::<D>(
            m_row_layout,
            challenges,
            opening_point,
            ring_multiplier_point,
            opening_batch,
            gamma,
            row_coefficient_rings,
            y,
            v,
        )?;
        instance.check_v_shape_for_level::<D>(&lp)?;
        let witness = RingRelationWitness::new(
            z_folded_rings,
            fold_grind_nonce,
            e_hat,
            RingBuf::from_ring_elems(&e_folded),
            hint,
        );
        Ok((instance, witness, commitment))
    }
}
