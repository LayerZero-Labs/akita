//! Shared per-fold verifier replay (EOR, stage-1/2/3, ring switch).

use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, MulBaseUnreduced, RandomSampling,
};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::SumcheckInstanceVerifierExt;
use akita_transcript::labels::{
    ABSORB_EVALUATION_CLAIMS, ABSORB_NEXT_LEVEL_WITNESS_BINDING, ABSORB_RANGE_IMAGE_EVALUATION,
    ABSORB_STAGE2_NEXT_W_EVAL, ABSORB_STAGE3_NEXT_W_EVAL, ABSORB_SUMCHECK_CLAIM,
    ABSORB_TERMINAL_W_REMAINDER, CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::{
    assemble_relation_rhs, build_multi_group_root_stage2_trace_table,
    build_trace_claim_multi_group_root, build_trace_claim_root, build_trace_table_scaled,
    derive_tensor_extension_opening_claim_from_partials, dispatch_for_field,
    ensure_trace_stage2_supported, prepare_opening_point,
    proof::relation::evaluation_trace_row_weight, relation_claim_from_layout_extension,
    relation_rhs_layout_for, ring_subfield_packed_extension_opening_point,
    tensor_equality_factor_eval_at_point, tensor_opening_split, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, trace_public_weights_recursive,
    trace_public_weights_root_terms, trace_terms_recursive, trace_weight_layout_from_segment,
    AkitaStage1Proof, AkitaStage2Proof, AkitaVerifierSetup, BasisMode, DigitRangeEqualityPoint,
    DigitRangePlan, ExtensionOpeningReductionProof, FoldLinfProtocolBinding, FpExtEncoding,
    LevelParams, OpeningClaimsLayout, PreparedOpeningPoint, RelationMatrixRowLayout,
    RingMultiplierOpeningPoint, RingOpeningPoint, RingRelationInstance, RingVec,
    SegmentTypedWitness, SetupSumcheckProof, TerminalWitnessTranscriptParts, TraceClaim,
    EXTENSION_OPENING_REDUCTION_DEGREE,
};

use crate::protocol::ring_switch::{
    ring_switch_verifier, RingSwitchReplay, RingSwitchVerifyOutput,
};
use crate::stages::stage1::{derive_multi_group_stage1_challenges, AkitaStage1Verifier};
use crate::stages::stage2::AkitaStage2Verifier;
use crate::stages::SetupSumcheckVerifier;

use super::FoldVerifyOutput;

pub(in crate::protocol::core) struct FoldEorReplay<F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) prepared_points: Vec<PreparedOpeningPoint<F, E>>,
    pub(in crate::protocol::core) reduction_challenges: Option<Vec<E>>,
    pub(in crate::protocol::core) final_relation: Option<(E, Vec<E>)>,
}

#[derive(Clone, Copy)]
struct EorReductionShape {
    split_bits: usize,
    width: usize,
    num_rounds: usize,
}

fn eor_reduction_shape<F, E>(
    opening_num_vars: usize,
    partials_len: usize,
    num_claims: usize,
) -> Result<EorReductionShape, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let (split_bits, width) =
        tensor_opening_split::<F, E>().map_err(|_| AkitaError::InvalidProof)?;
    let num_rounds = opening_num_vars
        .checked_sub(split_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let expected_partials = width
        .checked_mul(num_claims)
        .ok_or(AkitaError::InvalidProof)?;
    if width == 1 || partials_len != expected_partials {
        return Err(AkitaError::InvalidProof);
    }
    Ok(EorReductionShape {
        split_bits,
        width,
        num_rounds,
    })
}

fn eor_input_claim_from_partials<F, E>(
    partials: &[E],
    shape: EorReductionShape,
    eta: &[E],
    row_coefficients: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if shape.width == 0
        || !partials.len().is_multiple_of(shape.width)
        || row_coefficients.len() != partials.len() / shape.width
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut input_claim = E::zero();
    for (&row_coefficient, partials) in row_coefficients
        .iter()
        .zip(partials.chunks_exact(shape.width))
    {
        let row_partials = tensor_row_partials_from_columns::<F, E>(partials)?;
        let claim = tensor_reduction_claim_from_rows::<F, E>(&row_partials, eta)?;
        input_claim += row_coefficient * claim;
    }
    Ok(input_claim)
}

#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_fold_eor<F, E, T>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    challenge_point: &[E],
    openings: &[E],
    row_coefficients: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    lp: &LevelParams,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let d_a = lp.role_dims().d_a();
    dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
        verify_fold_eor_kernel::<F, E, T, D>(
            extension_opening_reduction,
            challenge_point,
            openings,
            row_coefficients,
            opening_batch,
            basis,
            lp.num_positions_per_block,
            lp.num_live_blocks,
            d_a.trailing_zeros() as usize,
            requires_reduction,
            transcript,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_fold_eor_kernel<F, E, T, const D: usize>(
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<E>>,
    challenge_point: &[E],
    openings: &[E],
    row_coefficients: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    alpha_bits: usize,
    requires_reduction: bool,
    transcript: &mut T,
) -> Result<FoldEorReplay<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let num_claims = opening_batch.num_total_polynomials();
    if openings.len() != num_claims || row_coefficients.len() != num_claims {
        return Err(AkitaError::InvalidProof);
    }

    let mut eor_trace_final: Option<(E, Vec<E>)> = None;
    let reduction_check = if let Some(reduction) = extension_opening_reduction {
        if <E as ExtField<F>>::EXT_DEGREE == 1 {
            return Err(AkitaError::InvalidProof);
        }
        let shape = eor_reduction_shape::<F, E>(
            opening_batch.max_num_vars(),
            reduction.partials.len(),
            num_claims,
        )?;
        if challenge_point.len() > opening_batch.max_num_vars() {
            return Err(AkitaError::InvalidProof);
        }
        let mut eor_point = challenge_point.to_vec();
        eor_point.resize(opening_batch.max_num_vars(), E::zero());
        for (claim_idx, opening) in openings.iter().copied().enumerate().take(num_claims) {
            let partial_start = claim_idx * shape.width;
            let partial_end = partial_start + shape.width;
            let partials = &reduction.partials[partial_start..partial_end];
            let expected =
                derive_tensor_extension_opening_claim_from_partials::<F, E>(&eor_point, partials)?;
            if expected != opening {
                return Err(AkitaError::InvalidProof);
            }
            for partial in partials {
                append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, partial);
            }
        }
        let eta = (0..shape.split_bits)
            .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH))
            .collect::<Vec<_>>();
        let input_claim = eor_input_claim_from_partials::<F, E>(
            &reduction.partials,
            shape,
            &eta,
            row_coefficients,
        )?;
        // Non-zk EOR sumcheck: bind the input claim, then replay the public
        // sumcheck rounds. The final claim is enforced downstream through the
        // fused stage-2 `trace_eval_target` and per-claim scales, so there is no
        // standalone on-wire opening handle to read here.
        transcript.append_serde(ABSORB_SUMCHECK_CLAIM, &input_claim);
        let (final_claim, rho) = reduction.sumcheck.verify::<F, T, _>(
            input_claim,
            shape.num_rounds,
            EXTENSION_OPENING_REDUCTION_DEGREE,
            transcript,
            |tr| sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND),
        )?;
        let final_factor = tensor_equality_factor_eval_at_point::<F, E>(
            &eor_point[shape.split_bits..],
            &eta,
            &rho,
        )?;
        eor_trace_final = Some((final_claim, vec![final_factor]));
        Some(rho)
    } else if requires_reduction && <E as ExtField<F>>::EXT_DEGREE != 1 {
        return Err(AkitaError::InvalidProof);
    } else {
        None
    };

    let prepared_points = if let Some(rho) = &reduction_check {
        let protocol_point =
            ring_subfield_packed_extension_opening_point::<F, E, D>(rho.len(), rho)?;
        let prepared = prepare_opening_point::<F, E, D>(
            &protocol_point,
            basis,
            num_positions_per_block,
            num_live_blocks,
            alpha_bits,
        )?;
        vec![prepared]
    } else {
        Vec::new()
    };
    Ok(FoldEorReplay {
        prepared_points,
        reduction_challenges: reduction_check,
        final_relation: eor_trace_final,
    })
}

pub(in crate::protocol::core) struct PreparedFoldReplay<'a, F: FieldCore, E: FieldCore> {
    pub(in crate::protocol::core) lp: &'a LevelParams,
    pub(in crate::protocol::core) relation_matrix_row_layout: RelationMatrixRowLayout,
    pub(in crate::protocol::core) fold_grind_nonce: u32,
    pub(in crate::protocol::core) v: RingVec<F>,
    /// Normalized opening geometry (one group for scalar/suffix folds, `G`
    /// groups for multi-group roots).
    pub(in crate::protocol::core) opening_shape: OpeningClaimsLayout,
    /// Sent commitment rows concatenated in M-row (final-first
    /// `root_group_order`) order — the single group's rows for scalar/suffix
    /// folds, `concat_g u_g` for multi-group roots. Matches the prover's
    /// `RingRelationProver` commitment-row concatenation and
    /// `relation_rhs_layout_for` block order.
    pub(in crate::protocol::core) commitment_rows: RingVec<F>,
    pub(in crate::protocol::core) row_coefficients: Vec<E>,
    /// Per-group ring opening points in `OpeningClaims` order.
    pub(in crate::protocol::core) group_ring_opening_points: Vec<RingOpeningPoint<F>>,
    /// Per-group ring multiplier points in `OpeningClaims` order.
    pub(in crate::protocol::core) group_ring_multiplier_points: Vec<RingMultiplierOpeningPoint<F>>,
    pub(in crate::protocol::core) w_len: usize,
    pub(in crate::protocol::core) payload: PreparedFoldPayload<'a, F, E>,
    pub(in crate::protocol::core) trace: TracePreparation<F, E>,
}

/// Fused evaluation-trace inputs for one fold, grouped out of
/// [`PreparedFoldReplay`] because every build site produces them together and
/// [`build_trace_wire`] (and the terminal-direct path) consume them together:
/// the per-group prepared opening points reused for the trace term, the
/// optional root trace-block opening, the trace evaluation target/scale,
/// optional per-claim scales, and the opening basis.
pub(in crate::protocol::core) struct TracePreparation<F: FieldCore, E: FieldCore> {
    /// Per-group prepared opening points in `OpeningClaims` order (one element
    /// for scalar/suffix folds). Reused for the fused trace term.
    pub(in crate::protocol::core) prepared_points: Option<Vec<PreparedOpeningPoint<F, E>>>,
    pub(in crate::protocol::core) block_opening: Option<Vec<E>>,
    pub(in crate::protocol::core) eval_target: E,
    pub(in crate::protocol::core) eval_scale: E,
    pub(in crate::protocol::core) claim_scales: Option<Vec<E>>,
    pub(in crate::protocol::core) basis: BasisMode,
}

#[derive(Clone, Copy)]
pub(in crate::protocol::core) enum PreparedNextWitness<'a, F: FieldCore> {
    Commitment {
        commitment: &'a RingVec<F>,
        ring_dim: usize,
    },
    TerminalT(&'a [u8]),
}

pub(in crate::protocol::core) enum PreparedFoldPayload<'a, F: FieldCore, E: FieldCore> {
    Terminal {
        final_witness: &'a SegmentTypedWitness<F>,
        transcript: TerminalWitnessTranscriptParts,
    },
    Recursive {
        stage1: &'a AkitaStage1Proof<E>,
        stage2: &'a AkitaStage2Proof<F, E>,
        next_witness: PreparedNextWitness<'a, F>,
        next_witness_ring_dim: usize,
        next_opening_source_len: usize,
        stage3: Option<(&'a SetupSumcheckProof<E>, &'a LevelParams)>,
    },
}

struct Stage1Replay<E: FieldCore> {
    batching_coeff: E,
    range_image_evaluation: E,
    stage1_point: Vec<E>,
}

fn verify_stage1<F, E, T>(
    proof: &AkitaStage1Proof<E>,
    rs: &RingSwitchVerifyOutput<E>,
    transcript: &mut T,
) -> Result<Stage1Replay<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let num_rounds = rs
        .col_bits
        .checked_add(rs.ring_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("stage-1 variable count overflow".to_string()))?;
    if rs.tau0.len() != num_rounds {
        return Err(AkitaError::InvalidSize {
            expected: num_rounds,
            actual: rs.tau0.len(),
        });
    }
    let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &rs.tau0,
        rs.col_bits,
        rs.ring_bits,
    )?;
    let plan = DigitRangePlan::new(rs.b)?;
    let stage1_verifier = AkitaStage1Verifier::new(equality_point, plan);
    let stage1_point = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        stage1_verifier.verify::<F, T>(proof, transcript)?
    };
    transcript.append_serde(ABSORB_RANGE_IMAGE_EVALUATION, &proof.range_image_evaluation);
    let batching_coeff: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
    Ok(Stage1Replay {
        batching_coeff,
        range_image_evaluation: proof.range_image_evaluation,
        stage1_point,
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_stage2<F, E, T>(
    transcript: &mut T,
    setup: &AkitaVerifierSetup<F>,
    relation_instance: &RingRelationInstance<F>,
    stage2: &AkitaStage2Proof<F, E>,
    physical_w_len: usize,
    stage1: Stage1Replay<E>,
    rs: &RingSwitchVerifyOutput<E>,
    relation_claim: E,
    lp: &LevelParams,
    setup_claim: Option<E>,
    destination_source_len: usize,
    destination_ring_dim: usize,
    trace: Option<TraceWireAtRoleA<'_, F, E>>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let witness_eval = stage2.next_w_eval();
    let d_a = lp.role_dims().d_a();
    dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
        let trace_claim = trace
            .map(|wire| {
                wire.into_claim::<D>(destination_source_len, destination_ring_dim, physical_w_len)
            })
            .transpose()
            .map_err(|_| AkitaError::InvalidProof)?;
        verify_stage2_kernel::<F, E, T, D>(
            transcript,
            setup,
            relation_instance,
            stage2,
            stage1,
            rs,
            relation_claim,
            witness_eval,
            setup_claim,
            trace_claim,
        )
    })
}

/// Recursive (non-root) fold trace wire: a single prepared opening point folded
/// with a scalar eval scale.
struct RecursiveTraceWire<'a, F: FieldCore, E: FieldCore> {
    lp: &'a LevelParams,
    layout: akita_types::TraceWeightLayout,
    trace_coeff: E,
    trace_opening_claim: E,
    prepared_point: PreparedOpeningPoint<F, E>,
    trace_basis: BasisMode,
    trace_eval_scale: E,
}

/// Scalar (single-group) root fold trace wire.
struct RootTraceWire<'a, F: FieldCore, E: FieldCore> {
    lp: &'a LevelParams,
    layout: akita_types::TraceWeightLayout,
    prepared_point: PreparedOpeningPoint<F, E>,
    trace_block_opening: Vec<E>,
    trace_basis: BasisMode,
    row_coefficients: Vec<E>,
    trace_coeff: E,
    trace_eval_target: E,
    trace_claim_scales: Option<Vec<E>>,
    opening_batch: OpeningClaimsLayout,
}

/// Grouped (`G > 1`) root fold trace wire: one prepared opening point per group.
struct MultiGroupRootTraceWire<'a, F: FieldCore, E: FieldCore> {
    lp: &'a LevelParams,
    layout: akita_types::TraceWeightLayout,
    prepared_points: Vec<PreparedOpeningPoint<F, E>>,
    row_coefficients: Vec<E>,
    trace_coeff: E,
    trace_eval_target: E,
    trace_claim_scales: Option<Vec<E>>,
    opening_batch: OpeningClaimsLayout,
    live_x_cols: usize,
    trace_basis: BasisMode,
}

/// Fused evaluation-trace claim shaped for stage 2, tagged by fold topology.
/// Each variant carries only the inputs its topology needs and knows how to
/// build its own [`TraceClaim`]; the shared structured/dense remap is applied
/// once in [`TraceWireAtRoleA::into_claim`].
enum TraceWireAtRoleA<'a, F: FieldCore, E: FieldCore> {
    Recursive(RecursiveTraceWire<'a, F, E>),
    Root(RootTraceWire<'a, F, E>),
    MultiGroupRoot(MultiGroupRootTraceWire<'a, F, E>),
}

impl<F, E> RecursiveTraceWire<'_, F, E>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    fn into_trace_claim<const D: usize>(
        self,
        destination_ring_dim: usize,
    ) -> Result<TraceClaim<F, E, D>, AkitaError> {
        let trace_terms = trace_terms_recursive(
            &self.prepared_point,
            self.lp,
            self.trace_basis,
            self.trace_eval_scale,
        )?;
        let dense_evals = if destination_ring_dim == D {
            None
        } else {
            let public = trace_public_weights_recursive::<F, E, D>(
                &self.prepared_point,
                self.trace_eval_scale,
            )?;
            let live_x_cols = akita_types::opening_domain_len(self.layout.opening_source_len)?;
            Some(
                build_trace_table_scaled(&self.layout, &public, live_x_cols, E::one())?
                    .materialize_dense(live_x_cols, D),
            )
        };
        Ok(TraceClaim {
            layout: self.layout,
            trace_coeff: self.trace_coeff,
            trace_opening_claim: self.trace_opening_claim,
            trace_terms,
            trace_term_batches: Vec::new(),
            dense_evals,
        })
    }
}

impl<F, E> RootTraceWire<'_, F, E>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    fn into_trace_claim<const D: usize>(
        self,
        destination_ring_dim: usize,
    ) -> Result<TraceClaim<F, E, D>, AkitaError> {
        let mut claim = build_trace_claim_root::<F, E, D>(
            self.layout,
            self.lp,
            &self.opening_batch,
            &self.prepared_point,
            &self.trace_block_opening,
            self.trace_basis,
            &self.row_coefficients,
            self.trace_coeff,
            self.trace_eval_target,
            self.trace_claim_scales.as_deref(),
        )?;
        if destination_ring_dim != D {
            let public = trace_public_weights_root_terms::<F, E, D>(
                self.lp.num_live_blocks,
                &self.opening_batch,
                &self.prepared_point,
                &self.row_coefficients,
                self.trace_claim_scales.as_deref(),
            )?;
            let live_x_cols = akita_types::opening_domain_len(claim.layout.opening_source_len)?;
            claim.dense_evals = Some(
                build_trace_table_scaled(&claim.layout, &public, live_x_cols, E::one())?
                    .materialize_dense(live_x_cols, D),
            );
        }
        Ok(claim)
    }
}

impl<F, E> MultiGroupRootTraceWire<'_, F, E>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    fn into_trace_claim<const D: usize>(
        self,
        destination_ring_dim: usize,
    ) -> Result<TraceClaim<F, E, D>, AkitaError> {
        let mut claim = build_trace_claim_multi_group_root::<F, E, D>(
            self.layout,
            self.lp,
            &self.opening_batch,
            &self.prepared_points,
            &self.row_coefficients,
            self.trace_claim_scales.as_deref(),
            self.trace_basis,
            self.trace_coeff,
            self.trace_eval_target,
            self.live_x_cols,
        )?;
        if destination_ring_dim != D {
            claim.dense_evals = Some(
                build_multi_group_root_stage2_trace_table::<F, E>(
                    D,
                    &claim.layout.witness_layout,
                    claim.layout.opening_source_len,
                    self.lp,
                    &self.opening_batch,
                    &self.prepared_points,
                    &self.row_coefficients,
                    self.trace_claim_scales.as_deref(),
                    E::one(),
                    self.live_x_cols,
                )?
                .into_ring_dense()?,
            );
        }
        Ok(claim)
    }
}

impl<F, E> TraceWireAtRoleA<'_, F, E>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FieldCore + FromPrimitiveInt,
{
    fn into_claim<const D: usize>(
        self,
        destination_source_len: usize,
        destination_ring_dim: usize,
        physical_field_len: usize,
    ) -> Result<TraceClaim<F, E, D>, AkitaError> {
        let mut claim = match self {
            Self::Recursive(wire) => wire.into_trace_claim::<D>(destination_ring_dim),
            Self::Root(wire) => wire.into_trace_claim::<D>(destination_ring_dim),
            Self::MultiGroupRoot(wire) => wire.into_trace_claim::<D>(destination_ring_dim),
        }?;
        if destination_ring_dim == D {
            remap_trace_claim_structured(&mut claim, destination_source_len, physical_field_len)?;
        } else {
            remap_trace_claim_dense(
                &mut claim,
                destination_source_len,
                destination_ring_dim,
                physical_field_len,
            )?;
        }
        Ok(claim)
    }
}

fn remap_trace_claim_structured<F, E, const D: usize>(
    claim: &mut TraceClaim<F, E, D>,
    destination_source_len: usize,
    physical_field_len: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    let destination_capacity = destination_source_len
        .checked_mul(D)
        .ok_or(AkitaError::InvalidProof)?;
    if physical_field_len > destination_capacity || claim.dense_evals.is_some() {
        return Err(AkitaError::InvalidProof);
    }
    let col_bits =
        akita_types::opening_domain_len(destination_source_len)?.trailing_zeros() as usize;
    claim.layout.opening_source_len = destination_source_len;
    claim.layout.col_bits = col_bits;
    for batch in &mut claim.trace_term_batches {
        batch.layout.opening_source_len = destination_source_len;
        batch.layout.col_bits = col_bits;
    }
    Ok(())
}

fn remap_trace_claim_dense<F, E, const D: usize>(
    claim: &mut TraceClaim<F, E, D>,
    destination_source_len: usize,
    destination_ring_dim: usize,
    physical_field_len: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    let source_len = claim.layout.opening_source_len;
    let source = claim.dense_evals.take().ok_or(AkitaError::InvalidProof)?;
    let source_capacity = source_len.checked_mul(D).ok_or(AkitaError::InvalidProof)?;
    let destination_capacity = destination_source_len
        .checked_mul(destination_ring_dim)
        .ok_or(AkitaError::InvalidProof)?;
    if destination_ring_dim == 0
        || physical_field_len > source_capacity
        || physical_field_len > destination_capacity
    {
        return Err(AkitaError::InvalidProof);
    }
    let destination_len = akita_types::opening_domain_len(destination_source_len)?
        .checked_mul(destination_ring_dim)
        .ok_or(AkitaError::InvalidProof)?;
    let mut destination = vec![E::zero(); destination_len];
    for physical in 0..physical_field_len {
        let source_col = akita_types::checked_opening_source_index(source_len, physical / D)?;
        let source_index = source_col * D + physical % D;
        let destination_col = akita_types::checked_opening_source_index(
            destination_source_len,
            physical / destination_ring_dim,
        )?;
        let destination_index =
            destination_col * destination_ring_dim + physical % destination_ring_dim;
        destination[destination_index] =
            *source.get(source_index).ok_or(AkitaError::InvalidProof)?;
    }
    claim.dense_evals = Some(destination);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_stage2_kernel<F, E, T, const D: usize>(
    transcript: &mut T,
    setup: &AkitaVerifierSetup<F>,
    relation_instance: &RingRelationInstance<F>,
    stage2: &AkitaStage2Proof<F, E>,
    stage1: Stage1Replay<E>,
    rs: &RingSwitchVerifyOutput<E>,
    relation_claim: E,
    witness_eval: E,
    setup_claim: Option<E>,
    trace: Option<TraceClaim<F, E, D>>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let stage2_verifier = AkitaStage2Verifier::new(
        stage1.batching_coeff,
        stage1.range_image_evaluation,
        witness_eval,
        stage1.stage1_point,
        &rs.relation_matrix_evaluator,
        &setup.expanded,
        relation_instance,
        rs.alpha,
        setup_claim,
        relation_claim,
        rs.col_bits,
        rs.ring_bits,
        trace,
    )?;

    let sumcheck_challenges = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        stage2_verifier.verify::<F, T, _>(&stage2.sumcheck_proof, transcript, |tr| {
            sample_ext_challenge::<F, E, T>(tr, CHALLENGE_SUMCHECK_ROUND)
        })?
    };
    transcript.absorb_and_record_serde(ABSORB_STAGE2_NEXT_W_EVAL, &stage2.next_w_eval());
    Ok(sumcheck_challenges)
}

fn verify_stage3<F, E, T>(
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    rs: &RingSwitchVerifyOutput<E>,
    sumcheck_challenges: &[E],
    stage2_next_w_eval: E,
    stage3: Option<(&SetupSumcheckProof<E>, &LevelParams)>,
) -> Result<Option<FoldVerifyOutput<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    if let Some((proof, next_fold_level_params)) = stage3 {
        let witness_rounds = rs.col_bits.checked_add(rs.ring_bits).ok_or_else(|| {
            AkitaError::InvalidSetup("stage-3 witness round count overflow".to_string())
        })?;
        if sumcheck_challenges.len() != witness_rounds {
            return Err(AkitaError::InvalidSize {
                expected: witness_rounds,
                actual: sumcheck_challenges.len(),
            });
        }
        let setup_coefficient_bits = rs
            .relation_matrix_evaluator
            .role_dims
            .d_a()
            .trailing_zeros() as usize;
        let setup_x_challenges = sumcheck_challenges
            .get(setup_coefficient_bits..)
            .ok_or(AkitaError::InvalidProof)?;
        let eta = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_SUMCHECK_BATCH);
        let verifier = SetupSumcheckVerifier::new::<F>(
            &rs.relation_matrix_evaluator,
            setup_x_challenges,
            &rs.tau1,
            rs.alpha,
        )?;
        let (rho_w, rho_setup) = verifier.verify_batched_stage3::<F, T>(
            setup,
            next_fold_level_params,
            proof,
            stage2_next_w_eval,
            sumcheck_challenges,
            witness_rounds,
            eta,
            transcript,
        )?;
        transcript.absorb_and_record_serde(ABSORB_STAGE3_NEXT_W_EVAL, &proof.next_w_eval);
        let setup_prefix_opening = next_fold_level_params
            .setup_prefix
            .as_ref()
            .map(|_| (rho_setup, proof.setup_prefix_eval));
        return Ok(Some((rho_w, setup_prefix_opening)));
    }
    Ok(None)
}

/// Transcript-free shape/consistency gate run before any replay: validates the
/// concatenated commitment rows and the opening vector decode at their ring
/// dimensions, checks the fold grind nonce against the batch contract, and
/// confirms the per-group opening/multiplier point counts. Because it never
/// touches the transcript it cannot perturb replay bytes; a failure here is a
/// malformed-proof rejection.
fn validate_fold_inputs<F, E>(prepared: &PreparedFoldReplay<'_, F, E>) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore,
{
    let _span = tracing::info_span!("fold_validate_inputs").entered();
    let role_dims = prepared.lp.role_dims();
    let num_groups = prepared.opening_shape.num_groups();
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Outer),
        F,
        role_dims.d_b(),
        |D| prepared.commitment_rows.as_ring_slice::<D>().map(|_| ())
    )?;
    prepared
        .lp
        .fold_witness_grind_batch_contract(
            &prepared.opening_shape,
            FoldLinfProtocolBinding::CURRENT.max_grind_attempts,
        )?
        .validate_nonce(prepared.fold_grind_nonce)?;
    if !prepared.v.coeffs().is_empty() {
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            role_dims.d_d(),
            |D| prepared.v.as_ring_slice::<D>().map(|_| ())
        )?;
    }
    if prepared.group_ring_opening_points.len() != num_groups
        || prepared.group_ring_multiplier_points.len() != num_groups
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

/// Shape the fused evaluation-trace claim wire that `verify_stage2` folds in,
/// or `None` when the fold carries no trace opening. Selects the recursive /
/// grouped-root / scalar-root variant from the level params and the prepared
/// trace inputs. Transcript-free: it only reshapes already-absorbed data, so it
/// cannot perturb replay bytes.
///
/// `EvaluationTrace` is the last padded relation row, so openings are weighted
/// by `eq(tau1, EvaluationTrace_row_index)`.
fn build_trace_wire<'a, F, E>(
    lp: &'a LevelParams,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    row_coefficients: &[E],
    trace: &TracePreparation<F, E>,
    relation_instance: &RingRelationInstance<F>,
    rs: &RingSwitchVerifyOutput<E>,
    d_a: usize,
) -> Result<Option<TraceWireAtRoleA<'a, F, E>>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
{
    let opening_batch = relation_instance.opening_batch();
    let evaluation_trace_row =
        lp.evaluation_trace_row_index_for_layout(relation_matrix_row_layout, opening_batch)?;
    let evaluation_trace_weight = evaluation_trace_row_weight(evaluation_trace_row, &rs.tau1)?;
    ensure_trace_stage2_supported(<E as ExtField<F>>::EXT_DEGREE)?;
    let trace_witness_layout = relation_instance.segment_layout(lp, None)?;
    let trace_opening_source_len = trace_witness_layout.total_len();
    let trace_x_cols = akita_types::opening_domain_len(trace_opening_source_len)?;
    let trace_col_bits = trace_x_cols.trailing_zeros() as usize;
    let trace_ring_bits = d_a.trailing_zeros() as usize;
    let trace_wire = if trace.prepared_points.is_none() {
        None
    } else if trace.block_opening.is_none() && !lp.has_precommitted_groups() {
        let layout = trace_weight_layout_from_segment(
            lp,
            &trace_witness_layout,
            trace_opening_source_len,
            trace_col_bits,
            trace_ring_bits,
            lp.num_live_blocks,
        )?;
        let prepared_point = trace
            .prepared_points
            .as_ref()
            .and_then(|points| points.first())
            .ok_or(AkitaError::InvalidProof)?;
        Some(TraceWireAtRoleA::Recursive(RecursiveTraceWire {
            lp,
            layout,
            trace_coeff: evaluation_trace_weight,
            trace_opening_claim: evaluation_trace_weight * trace.eval_target,
            prepared_point: prepared_point.clone(),
            trace_basis: trace.basis,
            trace_eval_scale: trace.eval_scale,
        }))
    } else if lp.has_precommitted_groups() {
        // Grouped root: dense trace-weight table (per-group `num_live_blocks`,
        // `num_digits_open`, and group-major e-hat offset). The layout is inert
        // for dense evaluation, but remains the canonical checked description
        // of one opening-digit segment. Use its first relation group rather
        // than the aggregate batch width: groups may have unequal claim counts.
        let group_id = trace_witness_layout.first_group_index()?;
        let group_lp = lp.group_params(opening_batch, group_id)?;
        let num_trace_blocks = opening_batch
            .group_layout(group_id)?
            .num_polynomials()
            .checked_mul(group_lp.num_live_blocks())
            .ok_or_else(|| AkitaError::InvalidSetup("trace block count overflow".to_string()))?;
        let layout = trace_weight_layout_from_segment(
            lp,
            &trace_witness_layout,
            trace_opening_source_len,
            trace_col_bits,
            trace_ring_bits,
            num_trace_blocks,
        )?;
        // Stage 2 is evaluated over the Boolean capacity of the exact physical
        // witness source; the unused suffix is zero.
        let live_x_cols = trace_x_cols;
        let prepared_points = trace
            .prepared_points
            .as_ref()
            .ok_or(AkitaError::InvalidProof)?
            .clone();
        Some(TraceWireAtRoleA::MultiGroupRoot(MultiGroupRootTraceWire {
            lp,
            layout,
            prepared_points,
            row_coefficients: row_coefficients.to_vec(),
            trace_coeff: evaluation_trace_weight,
            trace_eval_target: trace.eval_target,
            trace_claim_scales: trace.claim_scales.clone(),
            opening_batch: opening_batch.clone(),
            live_x_cols,
            trace_basis: trace.basis,
        }))
    } else {
        let num_trace_blocks = opening_batch
            .num_total_polynomials()
            .checked_mul(lp.num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("trace block count overflow".to_string()))?;
        let layout = trace_weight_layout_from_segment(
            lp,
            &trace_witness_layout,
            trace_opening_source_len,
            trace_col_bits,
            trace_ring_bits,
            num_trace_blocks,
        )?;
        Some(TraceWireAtRoleA::Root(RootTraceWire {
            lp,
            layout,
            prepared_point: trace
                .prepared_points
                .as_ref()
                .and_then(|points| points.first())
                .ok_or(AkitaError::InvalidProof)?
                .clone(),
            trace_block_opening: trace
                .block_opening
                .as_ref()
                .ok_or(AkitaError::InvalidProof)?
                .clone(),
            trace_basis: trace.basis,
            row_coefficients: row_coefficients.to_vec(),
            trace_coeff: evaluation_trace_weight,
            trace_eval_target: trace.eval_target,
            trace_claim_scales: trace.claim_scales.clone(),
            opening_batch: opening_batch.clone(),
        }))
    };
    Ok(trace_wire)
}

/// Terminal (quotient-free) fold: replay the witness-remainder absorb, then
/// verify the direct ring relations and the fused trace opening against the
/// final witness. A terminal fold produces no next-level challenges, so it
/// returns an empty challenge vector and no carried setup-prefix opening.
#[allow(clippy::too_many_arguments)]
fn verify_terminal_fold<F, E, T>(
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    relation_instance: &RingRelationInstance<F>,
    lp: &LevelParams,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    trace: &TracePreparation<F, E>,
    row_coefficients: &[E],
    final_witness: &SegmentTypedWitness<F>,
    terminal_replay: TerminalWitnessTranscriptParts,
) -> Result<FoldVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let role_dims = lp.role_dims();
    let _terminal_span = tracing::info_span!(
        "verify_terminal_direct_fold",
        d_a = role_dims.d_a(),
        d_b = role_dims.d_b(),
        groups = relation_instance.opening_batch().num_groups()
    )
    .entered();
    if relation_matrix_row_layout != RelationMatrixRowLayout::WithoutCommitmentBlocks {
        return Err(AkitaError::InvalidProof);
    }
    {
        let _span = tracing::info_span!("terminal_transcript_absorb").entered();
        transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_replay.response);
    }
    super::terminal_direct::verify_terminal_ring_relations(
        setup,
        relation_instance,
        lp,
        final_witness,
    )?;
    super::terminal_direct::verify_terminal_trace(
        relation_instance,
        lp,
        final_witness,
        trace
            .prepared_points
            .as_deref()
            .ok_or(AkitaError::InvalidProof)?,
        row_coefficients,
        trace.claim_scales.as_deref(),
        trace.eval_scale,
        trace.eval_target,
    )?;
    Ok((Vec::new(), None))
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(in crate::protocol::core) fn verify_fold<F, E, T>(
    setup: &AkitaVerifierSetup<F>,
    transcript: &mut T,
    prepared: PreparedFoldReplay<'_, F, E>,
) -> Result<FoldVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField + FromPrimitiveInt,
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt + AkitaSerialize + MulBaseUnreduced<F>,
    T: Transcript<F>,
{
    let opening_shape = prepared.opening_shape.clone();
    let num_groups = opening_shape.num_groups();
    let commitment_rows = &prepared.commitment_rows;
    let role_dims = prepared.lp.role_dims();
    let _fold_span = tracing::info_span!(
        "verify_fold",
        d_a = role_dims.d_a(),
        d_b = role_dims.d_b(),
        d_d = role_dims.d_d(),
        groups = num_groups,
        terminal = matches!(&prepared.payload, PreparedFoldPayload::Terminal { .. })
    )
    .entered();
    validate_fold_inputs::<F, E>(&prepared)?;
    let group_challenges = {
        let _span = tracing::info_span!("fold_derive_stage1_challenges").entered();
        derive_multi_group_stage1_challenges::<F, T>(
            transcript,
            prepared.v.coeffs(),
            role_dims.d_d(),
            role_dims.d_a(),
            &opening_shape,
            prepared.lp,
            prepared.relation_matrix_row_layout,
            prepared.fold_grind_nonce,
        )?
    };
    let (relation_rhs_layout, relation_instance) = {
        let _span = tracing::info_span!("fold_prepare_relation").entered();
        let (gamma, row_coefficient_rings) = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            role_dims.d_a(),
            |D| {
                RingRelationInstance::<F>::gamma_and_row_rings_from_coefficients::<D, E>(
                    &prepared.row_coefficients,
                )
            }
        )?;
        let relation_rhs_layout = relation_rhs_layout_for(
            prepared.lp,
            &opening_shape,
            prepared.relation_matrix_row_layout,
        )?;
        let relation_rhs = assemble_relation_rhs::<F>(
            role_dims,
            &relation_rhs_layout,
            &prepared.v,
            commitment_rows,
        )?;
        let relation_instance = RingRelationInstance::new(
            prepared.relation_matrix_row_layout,
            group_challenges,
            prepared.group_ring_opening_points,
            prepared.group_ring_multiplier_points,
            opening_shape.clone(),
            gamma,
            row_coefficient_rings,
            relation_rhs,
            prepared.v,
            role_dims,
        )?;
        relation_instance.check_v_shape_for_level(prepared.lp)?;
        (relation_rhs_layout, relation_instance)
    };
    let (stage1, stage2, next_witness, next_witness_ring_dim, next_opening_source_len, stage3) =
        match prepared.payload {
            PreparedFoldPayload::Terminal {
                final_witness,
                transcript: terminal_replay,
            } => {
                return verify_terminal_fold(
                    setup,
                    transcript,
                    &relation_instance,
                    prepared.lp,
                    prepared.relation_matrix_row_layout,
                    &prepared.trace,
                    &prepared.row_coefficients,
                    final_witness,
                    terminal_replay,
                );
            }
            PreparedFoldPayload::Recursive {
                stage1,
                stage2,
                next_witness,
                next_witness_ring_dim,
                next_opening_source_len,
                stage3,
            } => (
                stage1,
                stage2,
                next_witness,
                next_witness_ring_dim,
                next_opening_source_len,
                stage3,
            ),
        };
    let ring_switch_replay = RingSwitchReplay {
        setup: &setup.expanded,
        relation: &relation_instance,
        row_coefficients: &prepared.row_coefficients,
        lp: prepared.lp,
        opening_source_len: next_opening_source_len,
        opening_ring_dim: next_witness_ring_dim,
    };
    let d_a = role_dims.d_a();
    {
        let _span = tracing::info_span!("fold_bind_next_witness").entered();
        match next_witness {
            PreparedNextWitness::Commitment {
                commitment,
                ring_dim,
            } => {
                if ring_dim == 0 || !commitment.can_decode_vec(ring_dim) {
                    return Err(AkitaError::InvalidProof);
                }
                transcript.absorb_and_record_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, commitment);
            }
            PreparedNextWitness::TerminalT(t_state) if !t_state.is_empty() => {
                transcript.absorb_and_record_bytes(ABSORB_NEXT_LEVEL_WITNESS_BINDING, t_state);
            }
            PreparedNextWitness::TerminalT(_) => return Err(AkitaError::InvalidProof),
        }
    }
    let rs = dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
        ring_switch_verifier::<F, E, T, D>(
            &ring_switch_replay,
            prepared.w_len,
            transcript,
            RelationMatrixRowLayout::WithDBlock,
        )
    })?;
    let relation_claim = relation_claim_from_layout_extension::<F, E>(
        relation_instance.role_dims(),
        &relation_rhs_layout,
        &rs.tau1,
        rs.alpha,
        relation_instance.v(),
        commitment_rows,
    )?;
    let stage1_replay = verify_stage1::<F, E, T>(stage1, &rs, transcript)?;
    let trace_wire = build_trace_wire::<F, E>(
        prepared.lp,
        prepared.relation_matrix_row_layout,
        &prepared.row_coefficients,
        &prepared.trace,
        &relation_instance,
        &rs,
        d_a,
    )?;
    let setup_claim = stage3.as_ref().map(|(proof, _)| proof.claim);
    let sumcheck_challenges = verify_stage2::<F, E, T>(
        transcript,
        setup,
        &relation_instance,
        stage2,
        prepared.w_len,
        stage1_replay,
        &rs,
        relation_claim,
        prepared.lp,
        setup_claim,
        next_opening_source_len,
        next_witness_ring_dim,
        trace_wire,
    )?;
    let stage2_next_w_eval = if stage3.is_some() {
        stage2.next_w_eval()
    } else {
        E::zero()
    };
    let stage3_output = verify_stage3::<F, E, T>(
        setup,
        transcript,
        &rs,
        &sumcheck_challenges,
        stage2_next_w_eval,
        stage3,
    )?;
    Ok(stage3_output.unwrap_or((sumcheck_challenges, None)))
}
