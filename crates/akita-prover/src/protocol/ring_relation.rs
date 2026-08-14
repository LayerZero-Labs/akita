//! Ring-relation prover for the Akita PCS (§4.2).
//!
//! Builds the stage-1 relation instance and witness (`M`, `y`, `z`, `v`) via
//! [`RingRelationProver`].
use crate::compute::{
    BatchDecomposeFoldOutcome, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OperationCtx, RingSwitchProveBackend, RingSwitchRelationKernel,
    RingSwitchRelationPlan, RootOpeningSource, RuntimeRingSwitchProveBackend,
};
use crate::validation::validate_i8_setup_log_basis;
use crate::{DecomposeFoldWitness, DigitRowsComputeBackend, ProverOpeningData};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, SparseChallenge};
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt, HalvingField};
use akita_transcript::labels::ABSORB_OPENING_PAYLOAD;
use akita_transcript::Transcript;
use akita_types::dispatch_for_field;
use akita_types::RingMultiplierOpeningPoint;
use akita_types::{assemble_compressed_relation_rhs, assemble_relation_rhs, RingVec};
use akita_types::{gadget_row_scalars, DigitBlocks};
use akita_types::{CommittedGroupParams, RingRelationInstance};

use super::fold_grind::{self, ProverTranscriptGrind};
use super::ring_relation_witness::{RingRelationGroupWitness, RingRelationWitness};
use crate::backend::RingSwitchRelationView;

mod compression_witness;
mod relation_quotient;

pub(crate) use compression_witness::{
    materialize_compression_witness, CompressionSourceId, CompressionSourceWitness,
    CompressionWitnessMaterialization,
};
pub(crate) use relation_quotient::{compute_multi_group_relation_quotient, RelationQuotientOutput};

struct GroupOpeningMaterial<F: FieldCore> {
    e_hat: DigitBlocks,
    e_folded: RingVec<F>,
}

fn decompose_e_hat<F: FieldCore + CanonicalField, const D: usize>(
    pre_folded_e: &[&[CyclotomicRing<F, D>]],
    role_subcolumns: usize,
    depth_open: usize,
    log_basis: u32,
) -> Result<DigitBlocks, AkitaError> {
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2Params::new(depth_open, log_basis, q);
    let total_rows: usize = pre_folded_e.iter().map(|rows| rows.len()).sum();
    if role_subcolumns == 0 || !total_rows.is_multiple_of(role_subcolumns) {
        return Err(AkitaError::InvalidSetup(
            "E rows do not form complete native-role subcolumn groups".into(),
        ));
    }
    let planes_per_semantic = role_subcolumns
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("E digit block width overflow".into()))?;
    let mut e_hat =
        DigitBlocks::zeroed(vec![planes_per_semantic; total_rows / role_subcolumns], D)?;
    let mut offset = 0usize;
    for folded_rows in pre_folded_e {
        for w_i in *folded_rows {
            w_i.balanced_decompose_pow2_i8_into_with_params(
                &mut e_hat.typed_planes_mut::<D>()?[offset..offset + depth_open],
                &decompose_params,
            );
            offset += depth_open;
        }
    }
    Ok(e_hat)
}

fn concat_digit_blocks<'a>(
    blocks: impl IntoIterator<Item = &'a DigitBlocks>,
) -> Result<DigitBlocks, AkitaError> {
    let mut blocks = blocks.into_iter();
    let Some(first) = blocks.next() else {
        return Err(AkitaError::InvalidInput(
            "multi-group digit concatenation requires at least one group".to_string(),
        ));
    };
    let stride = first.digit_stride();
    let mut digits = Vec::new();
    let mut block_sizes = Vec::new();
    for block in std::iter::once(first).chain(blocks) {
        if block.digit_stride() != stride {
            return Err(AkitaError::InvalidInput(
                "multi-group digit blocks have mixed ring dimensions".to_string(),
            ));
        }
        digits.extend_from_slice(block.digits());
        block_sizes.extend_from_slice(block.block_sizes());
    }
    DigitBlocks::new(digits, block_sizes, stride)
}

pub(super) fn aggregate_decompose_fold_witnesses<F: FieldCore, const D: usize>(
    witnesses: Vec<DecomposeFoldWitness<F>>,
) -> Result<DecomposeFoldWitness<F>, AkitaError> {
    let mut witnesses = witnesses.into_iter();
    let Some(first) = witnesses.next() else {
        return Err(AkitaError::InvalidInput(
            "batched decompose_fold requires at least one witness".to_string(),
        ));
    };
    first.ensure_ring_dim::<D>()?;
    let row_count = first.row_count();
    let (z_folded_rings, mut centered_coeffs) = first.into_owned_flat_parts();
    let mut z_folded_coeffs = z_folded_rings.into_coeffs();

    for witness in witnesses {
        witness.ensure_ring_dim::<D>()?;
        if witness.row_count() != row_count {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold witness length mismatch".to_string(),
            ));
        }
        for (dst, src) in z_folded_coeffs
            .iter_mut()
            .zip(witness.z_folded_rings.coeffs())
        {
            *dst += *src;
        }
        for (dst, src) in centered_coeffs
            .iter_mut()
            .zip(witness.centered_coeffs_flat())
        {
            *dst = dst.checked_add(*src).ok_or_else(|| {
                AkitaError::InvalidInput(
                    "batched decompose_fold centered coefficient overflow".to_string(),
                )
            })?;
        }
    }

    DecomposeFoldWitness::from_owned_flat_parts::<D>(
        akita_types::RingVec::from_coeffs_with_ring_dim(z_folded_coeffs, D)?,
        centered_coeffs,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_point_decompose_fold_witness<F, P, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    challenges: &Challenges,
    point_polys: &[&P],
    point_indices: &[usize],
    num_positions_per_block: usize,
    num_digits_inner: usize,
    log_basis_inner: u32,
) -> Result<DecomposeFoldWitness<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
    P: RootOpeningSource<F, D>,
    B: crate::compute::ComputeBackendSetup<F>
        + for<'a> OpeningBatchKernel<P::OpeningBatchView<'a>, F, D>
        + for<'a> OpeningFoldKernel<P::OpeningView<'a>, F, D>,
{
    let point_challenges = challenges.select_claims(point_indices)?;
    let batch_view = P::opening_batch(point_polys)?;
    match OpeningBatchKernel::decompose_fold_batch(
        backend,
        prepared,
        batch_view,
        DecomposeFoldBatchPlan::Sparse {
            challenges: point_challenges.as_slice(),
            num_positions_per_block,
            num_digits: num_digits_inner,
            log_basis: log_basis_inner,
        },
    )? {
        BatchDecomposeFoldOutcome::Fused(z_point) => Ok(z_point),
        BatchDecomposeFoldOutcome::FallbackPerPoly => {
            let witnesses: Vec<DecomposeFoldWitness<F>> = point_polys
                .iter()
                .zip(
                    point_challenges
                        .as_slice()
                        .chunks(point_challenges.num_live_blocks_per_claim()),
                )
                .map(|(poly, poly_challenges)| -> Result<_, AkitaError> {
                    OpeningFoldKernel::decompose_fold(
                        backend,
                        prepared,
                        poly.opening_view()?,
                        DecomposeFoldPlan {
                            challenges: poly_challenges,
                            num_positions_per_block,
                            num_digits: num_digits_inner,
                            log_basis: log_basis_inner,
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

/// Convert scalar or multi-group multiplier-point carriers into the multi-group internal form.
pub trait IntoRingMultiplierOpeningPointVec<F: FieldCore> {
    fn into_vec(self) -> Vec<RingMultiplierOpeningPoint<F>>;
}

impl<F: FieldCore> IntoRingMultiplierOpeningPointVec<F> for RingMultiplierOpeningPoint<F> {
    fn into_vec(self) -> Vec<RingMultiplierOpeningPoint<F>> {
        vec![self]
    }
}

impl<F: FieldCore> IntoRingMultiplierOpeningPointVec<F> for Vec<RingMultiplierOpeningPoint<F>> {
    fn into_vec(self) -> Vec<RingMultiplierOpeningPoint<F>> {
        self
    }
}

struct RelationDRows<F: FieldCore, const D: usize> {
    reduced: Vec<CyclotomicRing<F, D>>,
    quotients: Vec<CyclotomicRing<F, D>>,
}

/// Compute the private D-block rows `v = D * e_hat` and their relation quotients.
///
/// D-role kernel: `d_row_len` is the D-matrix row count and `e_hat` carries
/// the opening digits at the D-role ring dimension. Callers extract both from
/// the schedule; this function must not read schedule types.
fn compute_relation_d_rows<F, RB, const D: usize>(
    ring_switch_ctx: &OperationCtx<'_, F, RB>,
    d_row_len: usize,
    log_basis: u32,
    e_hat: &DigitBlocks,
) -> Result<RelationDRows<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    RB: RingSwitchProveBackend<F, D>,
{
    let backend = ring_switch_ctx.backend();
    let prepared = ring_switch_ctx.prepared();
    let _span = tracing::info_span!(
        "compute_relation_v",
        e_hat_planes = e_hat.typed_planes::<D>()?.len()
    )
    .entered();
    let rows = RingSwitchRelationKernel::relation_rows(
        backend,
        prepared,
        RingSwitchRelationView {
            e_hat: e_hat.typed_planes::<D>()?,
            t_hat: &[],
            z_segment: &[],
            z_folded_centered_inf_norm: 0,
        },
        RingSwitchRelationPlan {
            n_d: d_row_len,
            n_b: 0,
            n_a: 0,
            log_basis_open: log_basis,
            log_basis_outer: log_basis,
        },
    )?;
    if rows.d_negacyclic.len() != d_row_len
        || rows.d_cyclic.len() != d_row_len
        || !rows.b_cyclic.is_empty()
        || !rows.a_quotients.is_empty()
    {
        return Err(AkitaError::InvalidProof);
    }
    let quotients = rows
        .d_cyclic
        .iter()
        .zip(&rows.d_negacyclic)
        .map(|(cyclic, reduced)| {
            relation_quotient::quotient_from_cyclic_and_reduced(cyclic, reduced)
        })
        .collect();
    Ok(RelationDRows {
        reduced: rows.d_negacyclic,
        quotients,
    })
}

/// Validate the chunked-witness configuration at the prover boundary (no-panic
/// contract), before any witness math. Mirrors the planner entry guard and the
/// verifier layout resolution.
pub(crate) fn validate_chunked_witness_cfg(lp: &CommittedGroupParams) -> Result<(), AkitaError> {
    lp.witness_chunk.validate()
}

/// Restrict sparse fold challenges to one chunk's exact global block range,
/// zeroing all other blocks. Folding under these yields the partial response
/// `z_i = Σ_{j∈I_i} c_j s_j`.
pub(super) fn window_sparse_challenges(
    challenges: &Challenges,
    fold_range: std::ops::Range<usize>,
) -> Result<Challenges, AkitaError> {
    let windowed: Vec<SparseChallenge> = challenges
        .as_slice()
        .iter()
        .enumerate()
        .map(|(index, challenge)| {
            let block = index % challenges.num_live_blocks_per_claim();
            if fold_range.contains(&block) {
                challenge.clone()
            } else {
                SparseChallenge {
                    positions: Vec::new().into(),
                    coeffs: Vec::new().into(),
                }
            }
        })
        .collect();
    Challenges::from_sparse(
        windowed,
        challenges.num_live_blocks_per_claim(),
        challenges.num_claims(),
    )
}

/// Prover-side builder for the ring relation $M(x) \cdot z = y(x) + (X^D + 1) \cdot r(x)$.
pub struct RingRelationProver;

impl RingRelationProver {
    /// Root-level constructor for one or more group-local opening points and
    /// polynomial slots.
    ///
    /// `group_ring_multiplier_points` contains one prepared entry per ordered claim group.
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
    /// Panics if the batched `e_hat` decomposition does not preserve the
    /// expected block sizes. This
    /// invariants hold by construction for well-formed inputs accepted by the
    /// error checks above and are therefore treated as internal programming
    /// errors rather than recoverable failures.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    #[allow(private_bounds)]
    #[tracing::instrument(skip_all, name = "RingRelationProver::new")]
    #[inline(never)]
    pub fn new<'a, F, PointF, T, P, OB, RB>(
        opening_ctx: &OperationCtx<'_, F, OB>,
        ring_switch_ctx: &OperationCtx<'_, F, RB>,
        group_ring_multiplier_points: impl IntoRingMultiplierOpeningPointVec<F>,
        block_claims: ProverOpeningData<'a, PointF, P, F>,
        pre_folded_e_by_poly: Vec<RingVec<F>>,
        lp: CommittedGroupParams,
        transcript: &mut T,
        row_coefficient_rings: RingVec<F>,
    ) -> Result<(RingRelationInstance<F>, RingRelationWitness<F>), AkitaError>
    where
        F: FieldCore + CanonicalField + FromPrimitiveInt + HalvingField + HasWide + 'static,
        <F as HasWide>::Wide: From<F> + ReduceTo<F>,
        PointF: Clone,
        T: Transcript<F> + ProverTranscriptGrind<F>,
        PointF: akita_types::FpExtEncoding<F>
            + akita_field::ExtField<F>
            + akita_serialization::AkitaSerialize,
        P: crate::protocol::core::RootProverGroupOpening<F, PointF, OB>,
        OB: DigitRowsComputeBackend<F>,
        RB: DigitRowsComputeBackend<F> + RuntimeRingSwitchProveBackend<F>,
    {
        let prepare_span = tracing::info_span!("ring_relation_prepare_inputs").entered();
        validate_i8_setup_log_basis(lp.log_basis_open, "for i8 prover opening decomposition")?;
        validate_chunked_witness_cfg(&lp)?;
        let dims = lp.role_dims();
        let opening_batch = block_claims.opening_layout()?;
        let num_groups = block_claims.opening_claims().num_groups();
        let group_ring_multiplier_points = group_ring_multiplier_points.into_vec();
        if group_ring_multiplier_points.len() != num_groups {
            return Err(AkitaError::InvalidInput(
                "ring relation prover group point count mismatch".to_string(),
            ));
        }
        let mut hints = Vec::with_capacity(num_groups);
        for group_index in 0..num_groups {
            hints.push(block_claims.group_hint(group_index)?.clone());
        }
        let relation_geometry =
            akita_types::RelationWitnessGeometry::for_evaluation_trace_execution(
                &lp,
                &opening_batch,
            )?;
        let relation_rhs_layout = relation_geometry.rhs_layout();
        // Compressed commitments contain terminal F payloads, whose complete
        // B images are recovered from the retained first-map digits. Raw
        // suffix commitments already contain those B images directly.
        let mut commitment_row_coeffs: Vec<F> = Vec::new();
        let mut group_payloads = Vec::with_capacity(num_groups);
        let commit_group_order = if lp.has_precommitted_groups() {
            opening_batch.root_group_order()?
        } else {
            (0..num_groups).collect()
        };
        for (relation_group_index, &group_index) in commit_group_order.iter().enumerate() {
            let group_commitment = block_claims
                .opening_claims()
                .group_commitment(group_index)?;
            if lp.payload_mode.is_compressed() {
                let (planned_group_index, plan) =
                    relation_rhs_layout.group_compression_plan(relation_group_index)?;
                if planned_group_index != group_index
                    || group_commitment.rows().coeff_len() != plan.terminal_coefficients()
                {
                    return Err(AkitaError::InvalidInput(
                        "batched prover received a malformed compressed commitment".to_string(),
                    ));
                }
                let retained = hints[group_index].outer_compression_witness(plan)?;
                let source = retained
                    .stages()
                    .first()
                    .ok_or(AkitaError::InvalidProof)?
                    .recompose::<F>()?;
                commitment_row_coeffs.extend(source);
                group_payloads.push(group_commitment.rows().coeffs().to_vec());
            } else {
                let group_dims = lp.group_role_dims(&opening_batch, group_index)?;
                if !group_commitment.rows().can_decode_vec(group_dims.d_b())
                    || group_commitment.rows().coeff_len() / group_dims.d_b()
                        != lp.group_commitment_rows(&opening_batch, group_index)?
                {
                    return Err(AkitaError::InvalidInput(
                        "batched prover received a malformed raw commitment".to_string(),
                    ));
                }
                commitment_row_coeffs.extend_from_slice(group_commitment.rows().coeffs());
            }
        }
        let commitment_rows = RingVec::from_coeffs(commitment_row_coeffs);
        for (group_index, ring_multiplier_point) in group_ring_multiplier_points.iter().enumerate()
        {
            let group_lp = lp.group_params(&opening_batch, group_index)?;
            if ring_multiplier_point.position_len() != group_lp.num_positions_per_block()
                || ring_multiplier_point.fold_len() != group_lp.num_live_blocks()
            {
                return Err(AkitaError::InvalidInput(
                    "batched prover ring-multiplier opening-point layout mismatch".to_string(),
                ));
            }
        }
        let num_claims = opening_batch.num_total_polynomials();
        if num_claims == 0 {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one polynomial".to_string(),
            ));
        }
        if pre_folded_e_by_poly.len() != num_claims {
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
        let d_log_basis = lp.shared_d_digit_log_basis();
        let d_row_len = lp.open_commit_matrix.output_rank();
        drop(prepare_span);

        // D-role operations: decompose the folded opening rows into `e_hat`
        // digits and (non-terminal layouts) compute + absorb the D-block rows
        // `v = D * e_hat`. Both consume the same digits at `d_d`, so they share
        // one kernel-entry dispatch; the flat `DigitBlocks` / `RingVec` come
        // back out as D-free carriers.
        //
        // Terminal layout drops the D-block from the M-matrix entirely:
        // `v = D · e_hat` never travels on the wire, the verifier never
        // reconstructs it, and downstream prover paths (`ring_switch_build_w`,
        // `relation_claim_from_rows_extension`) consume an empty `v` slice.
        // Skip the D-NTT under Terminal.
        let opening_rows_span = tracing::info_span!(
            "ring_relation_opening_rows",
            groups = num_groups,
            claims = num_claims,
            d_d = dims.d_d(),
        )
        .entered();
        let mut group_openings = Vec::with_capacity(num_groups);
        let mut offset = 0usize;
        for group_index in 0..num_groups {
            let k_g = opening_batch.group_layout(group_index)?.num_polynomials();
            let end = offset.checked_add(k_g).ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group e-folded offset overflow".to_string())
            })?;
            let group_lp = lp.group_params(&opening_batch, group_index)?;
            let group_dims = lp.group_role_dims(&opening_batch, group_index)?;
            let (e_hat_g, e_folded_g) = dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Opening),
                F,
                group_dims.d_d(),
                |D_D| {
                    let pre_folded_typed = pre_folded_e_by_poly[offset..end]
                        .iter()
                        .map(RingVec::as_ring_slice::<D_D>)
                        .collect::<Result<Vec<_>, _>>()?;
                    let e_hat_typed = {
                        let _span =
                            tracing::info_span!("decompose_group_e_hat", group_index, claims = k_g)
                                .entered();
                        decompose_e_hat::<F, D_D>(
                            &pre_folded_typed,
                            group_dims.d_a() / group_dims.d_d(),
                            group_lp.num_digits_open(),
                            group_lp.log_basis_open(),
                        )?
                    };
                    Ok::<_, AkitaError>((
                        e_hat_typed,
                        RingVec::from_coeffs(
                            pre_folded_e_by_poly[offset..end]
                                .iter()
                                .flat_map(|block| block.coeffs().iter().copied())
                                .collect(),
                        ),
                    ))
                }
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!("D-role opening decomposition failed: {err:?}"))
            })?;
            group_openings.push(GroupOpeningMaterial {
                e_hat: e_hat_g,
                e_folded: e_folded_g,
            });
            offset = end;
        }
        let e_hat_concat = if lp.has_precommitted_groups() {
            Some(concat_digit_blocks(
                opening_batch
                    .root_group_order()?
                    .into_iter()
                    .map(|group_index| &group_openings[group_index].e_hat),
            )?)
        } else {
            None
        };
        let e_hat = e_hat_concat
            .as_ref()
            .or_else(|| group_openings.first().map(|group| &group.e_hat))
            .ok_or(AkitaError::InvalidProof)?;
        let (v, d_quotients) = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            dims.d_d(),
            |D_D| {
                if d_row_len == 0 {
                    Ok::<_, AkitaError>((
                        RingVec::from_coeffs(Vec::new()),
                        RingVec::from_coeffs(Vec::new()),
                    ))
                } else {
                    let d_rows = compute_relation_d_rows::<F, RB, D_D>(
                        ring_switch_ctx,
                        d_row_len,
                        d_log_basis,
                        e_hat,
                    )?;
                    Ok::<_, AkitaError>((
                        RingVec::from_ring_elems(&d_rows.reduced),
                        RingVec::from_ring_elems(&d_rows.quotients),
                    ))
                }
            }
        )
        .map_err(|err| AkitaError::InvalidInput(format!("D-role v failed: {err:?}")))?;
        let compression = if lp.payload_mode.is_compressed() {
            let retained_outer_sources = commit_group_order
                .iter()
                .enumerate()
                .map(|(relation_group_index, &group_index)| {
                    let (planned_group_index, plan) =
                        relation_rhs_layout.group_compression_plan(relation_group_index)?;
                    if planned_group_index != group_index {
                        return Err(AkitaError::InvalidSetup(
                            "compression group order disagrees with the relation layout".into(),
                        ));
                    }
                    CompressionSourceWitness::from_outer_hint(
                        group_index,
                        plan,
                        &hints[group_index],
                        group_payloads[relation_group_index].clone(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (compression, compression_report) = materialize_compression_witness(
                ring_switch_ctx,
                &relation_rhs_layout,
                retained_outer_sources,
                &v,
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!(
                    "compression witness materialization failed: {err:?}"
                ))
            })?;
            let opening_source = compression.source(CompressionSourceId::Opening)?;
            let opening_terminal_ring_dim = opening_source
                .witness
                .plan()
                .maps()
                .last()
                .ok_or(AkitaError::InvalidProof)?
                .ring_dimension();
            RingVec::from_coeffs_with_ring_dim(
                opening_source.terminal.coefficients().to_vec(),
                opening_terminal_ring_dim,
            )?
            .append_flat_to_transcript(
                ABSORB_OPENING_PAYLOAD,
                opening_terminal_ring_dim,
                transcript,
            )?;
            tracing::info!(
                sources = compression_report.sources,
                maps = compression_report.maps,
                batches = compression_report.batches.len(),
                source_bytes = compression_report.source_bytes,
                terminal_bytes = compression_report.terminal_bytes,
                retained_bytes = compression_report.retained_packed_witness_bytes,
                peak_scratch_bytes = compression_report.executor_peak_scratch_bytes,
                "materialized compression witness"
            );
            Some(compression)
        } else {
            v.append_flat_to_transcript(ABSORB_OPENING_PAYLOAD, dims.d_d(), transcript)?;
            None
        };
        drop(opening_rows_span);

        // Distributed-prover chunked layout: the grind emits one folded response
        // per block window (`z_i`), and the global response is their sum
        // (`Σ_i z_i = z`, exact coefficient-wise i32 accumulation).
        let fold_grind_span = tracing::info_span!(
            "ring_relation_fold_grind",
            groups = num_groups,
            claims = num_claims,
        )
        .entered();
        let grind_groups = (0..num_groups)
            .map(|group_index| {
                Ok(fold_grind::FoldGrindGroup {
                    group_index,
                    group: block_claims.group(group_index)?,
                    params: lp.group_params(&opening_batch, group_index)?,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let _grind_span = tracing::info_span!("fold_grind_sample").entered();
        let (grind_outputs, fold_grind_nonce) =
            fold_grind::sample_multi_group_fold_decompose_witnesses::<F, PointF, _, OB, T>(
                opening_ctx,
                transcript,
                &lp,
                &opening_batch,
                &grind_groups,
                None,
            )
            .map_err(|err| AkitaError::InvalidInput(format!("fold grind failed: {err:?}")))?;
        drop(_grind_span);
        let mut group_challenges = Vec::with_capacity(num_groups);
        let mut group_z = Vec::with_capacity(num_groups);
        for output in grind_outputs {
            group_challenges.push(output.challenges);
            group_z.push((output.witness, output.coefficients));
        }
        drop(fold_grind_span);

        // Relation rhs spans roles (consistency | [A | B]* | D), with each
        // B group expanded in slice-major then physical-row order.
        // Terminal levels drop the D-block from M entirely, so `n_d` is zero
        // and `v` stays empty.
        let instance_span = tracing::info_span!("ring_relation_build_instance").entered();
        let relation_rhs = if let Some(compression) = &compression {
            let group_terminal_payloads = (0..relation_rhs_layout.groups.len())
                .map(|relation_group_index| {
                    let (group_index, _) =
                        relation_rhs_layout.group_compression_plan(relation_group_index)?;
                    Ok(compression
                        .source(CompressionSourceId::Outer { group_index })?
                        .terminal
                        .coefficients())
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let opening_terminal_payload = compression
                .source(CompressionSourceId::Opening)?
                .terminal
                .coefficients();
            assemble_compressed_relation_rhs::<F>(
                &relation_rhs_layout,
                &group_terminal_payloads,
                opening_terminal_payload,
            )
        } else {
            assemble_relation_rhs::<F>(&relation_rhs_layout, &v, &commitment_rows)
        }
        .map_err(|err| AkitaError::InvalidInput(format!("relation rhs failed: {err:?}")))?;

        let instance = RingRelationInstance::new(
            group_challenges,
            group_ring_multiplier_points,
            opening_batch.clone(),
            gamma,
            row_coefficient_rings,
            relation_rhs,
            v,
            dims,
        )
        .map_err(|err| AkitaError::InvalidInput(format!("relation instance failed: {err:?}")))?;
        instance
            .check_v_shape_for_level(&lp)
            .map_err(|err| AkitaError::InvalidInput(format!("v shape failed: {err:?}")))?;
        drop(instance_span);

        let witness_span = tracing::info_span!("ring_relation_build_witness").entered();
        let witness = if lp.has_precommitted_groups() {
            let mut groups = Vec::with_capacity(num_groups);
            if hints.len() != num_groups {
                return Err(AkitaError::InvalidProof);
            }
            for (group_index, ((z_material, hint), opening)) in group_z
                .into_iter()
                .zip(hints)
                .zip(group_openings)
                .enumerate()
            {
                let (z_folded_rings, z_folded_coefficients) = z_material;
                let GroupOpeningMaterial { e_hat, e_folded } = opening;
                let k_g = opening_batch.group_layout(group_index)?.num_polynomials();
                let group_dims = lp.group_role_dims(&opening_batch, group_index)?;
                if hint.ring_dim() != group_dims.d_a() || hint.inner_rows().len() != k_g {
                    return Err(AkitaError::InvalidInput(
                        "prover hint shape does not match its commitment group".into(),
                    ));
                }
                groups.push(RingRelationGroupWitness::from_parts(
                    z_folded_rings,
                    z_folded_coefficients,
                    e_hat,
                    e_folded,
                    hint,
                    group_dims,
                ));
            }
            RingRelationWitness::from_groups(fold_grind_nonce, groups, d_quotients, compression)
        } else {
            if hints.len() != 1 {
                return Err(AkitaError::InvalidProof);
            }
            let hint = hints.into_iter().next().ok_or(AkitaError::InvalidProof)?;
            let (z_folded_rings, z_folded_coefficients) =
                group_z.into_iter().next().ok_or(AkitaError::InvalidProof)?;
            let GroupOpeningMaterial { e_hat, e_folded } = group_openings
                .into_iter()
                .next()
                .ok_or(AkitaError::InvalidProof)?;
            RingRelationWitness::from_flat_parts(
                z_folded_rings,
                z_folded_coefficients,
                fold_grind_nonce,
                e_hat,
                e_folded,
                hint,
                dims,
                d_quotients,
                compression,
            )
        };
        drop(witness_span);
        Ok((instance, witness))
    }
}
