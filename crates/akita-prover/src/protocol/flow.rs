//! Prover flow state shared by root orchestration during crate extraction.

use crate::dispatch_ring_dim_result;
use crate::protocol::extension_opening_reduction::{
    ExtensionOpeningReductionProver, ExtensionOpeningReductionTerm,
    SPARSE_TENSOR_FACTOR_MAX_LAZY_ROUNDS,
};
#[cfg(not(feature = "zk"))]
use crate::protocol::ring_switch::ring_switch_build_terminal_direct_w;
use crate::protocol::ring_switch::{
    ring_switch_build_w, ring_switch_finalize, ring_switch_finalize_terminal,
    ring_switch_finalize_terminal_with_gamma, ring_switch_finalize_with_gamma,
    NextWitnessCommitment, RingSwitchOutput,
};
use crate::protocol::sumcheck::{AkitaStage1Prover, AkitaStage2Prover, SetupSumcheckProver};
#[cfg(feature = "zk")]
use crate::protocol::zk_hiding_commit::commit_zk_hiding_witness;
use crate::protocol::RingRelationProver;
use crate::{
    AkitaPolyOps, CommittedPolynomials, ProverClaims, ProverComputeBackend,
    RecursiveCommitmentHintCache, RecursiveWitnessFlat, RecursiveWitnessView, RingRelationInstance,
    RingRelationWitness, RootTensorProjectionPoly,
};
use akita_algebra::CyclotomicRing;
use akita_config::{bind_transcript_instance_descriptor, CommitmentConfig};
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide};
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, Invertible, MulBaseUnreduced, PseudoMersenneField, RandomSampling,
};
use akita_serialization::AkitaSerialize;
#[cfg(feature = "zk")]
use akita_sumcheck::{
    CompressedUniPoly, EqFactoredUniPoly, SumcheckProofMasked, ZkSumcheckInstanceProverExt,
};
#[cfg(not(feature = "zk"))]
use akita_sumcheck::{SumcheckInstanceProverExt, SumcheckProof};
#[cfg(not(feature = "zk"))]
use akita_transcript::labels::ABSORB_TERMINAL_W_REMAINDER;
#[cfg(feature = "zk")]
use akita_transcript::labels::ABSORB_ZK_HIDING_COMMITMENT;
use akita_transcript::labels::{
    ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS, ABSORB_STAGE2_NEXT_W_EVAL,
    ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::{append_ext_field, sample_ext_challenge, Transcript};
use akita_types::{
    append_batched_commitments_to_transcript, append_claim_incidence_shape_to_transcript,
    append_claim_points_to_transcript, append_claim_values_to_transcript, basis_weights,
    check_extension_opening_reduction_output, check_tensor_extension_opening_claim,
    embed_ring_subfield_scalar, embed_ring_subfield_vector, flatten_batched_commitment_rows,
    folded_root_supports_opening_shape, prepare_recursive_opening_point_ext,
    prepare_root_opening_point_ext, recover_ring_subfield_inner_product,
    relation_claim_from_rows_extension, reorder_stage1_coords,
    ring_subfield_packed_extension_opening_point, root_direct_schedule,
    root_extension_opening_partials, root_tensor_projection_enabled,
    sample_public_row_coefficients, schedule_is_root_direct, schedule_num_fold_levels,
    schedule_root_fold_step, scheduled_fold_execution, scheduled_next_level_params,
    tensor_equality_factor_eval_at_point, tensor_equality_factor_evals,
    tensor_logical_claim_from_partials, tensor_opening_split, tensor_packed_witness_evals,
    tensor_partials_from_base_evals, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, terminal_witness_segment_layout, validate_batched_inputs,
    AkitaBatchedProof, AkitaBatchedRootProof, AkitaCommitmentHint, AkitaExpandedSetup,
    AkitaLevelProof, AkitaProofStep, AkitaScheduleInputs, AkitaStage1Proof, BasisMode, BlockOrder,
    ClaimIncidence, ClaimIncidenceLimits, ClaimIncidenceSummary, CleartextWitnessProof,
    CleartextWitnessShape, ExtensionOpeningReductionProof, FlatRingVec, IncidenceClaim,
    LevelParams, MRowLayout, PackedDigits, PreparedRootOpeningPoint, RingCommitment,
    RingMultiplierOpeningPoint, RingSubfieldEncoding, Schedule, SetupContributionMode,
    SetupSumcheckProof, Step, TerminalLevelProof, TerminalProofMode,
};
#[cfg(feature = "zk")]
use akita_types::{stage1_tree_stage_shapes, sumcheck_rounds, ZkHidingProof};
#[cfg(feature = "zk")]
use rand_core::OsRng;
#[cfg(feature = "zk")]
use std::array::from_fn;
use std::sync::Arc;

mod inputs;
mod recursive;
mod root_extension;
mod root_fold_eval;
mod root_fold;
#[cfg(test)]
mod tests;

pub use inputs::{
    build_folded_batched_proof_with_suffix, build_terminal_root_batched_proof,
    prepare_batched_prove_inputs, prove_batched, prove_folded_batched, prove_root_direct,
};
pub use recursive::{
    prove_fold_level_from_ring_relation, prove_recursive_fold_with_params, prove_recursive_level,
    prove_recursive_suffix, prove_terminal_fold_level_from_ring_relation,
    prove_terminal_recursive_fold_with_params, prove_terminal_recursive_level,
};
#[cfg(test)]
pub(in crate::protocol::flow) use recursive::{
    prove_recursive_extension_opening_reduction, recursive_witness_base_evals,
};
pub(in crate::protocol::flow) use root_extension::*;
pub(in crate::protocol::flow) use root_fold::evaluate_recursive_witness_at_multiplier_point;
pub use root_fold::{
    prove_root_fold_from_ring_relation, prove_root_fold_with_params,
    prove_terminal_root_fold_from_ring_relation, prove_terminal_root_fold_with_params,
};

/// Runtime state carried between recursive prove levels.
pub struct RecursiveProverState<F: FieldCore, L: FieldCore> {
    /// Current committed recursive witness representation.
    pub w: RecursiveWitnessFlat,
    /// Logical recursive witness when it differs from the committed representation.
    pub logical_w: Option<RecursiveWitnessFlat>,
    /// Current recursive witness commitment.
    pub commitment: FlatRingVec<F>,
    /// D-erased recursive commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
    /// Sumcheck challenges that become the next recursive opening point.
    pub sumcheck_challenges: Vec<L>,
    /// Claimed logical opening of `logical_w` at `sumcheck_challenges`.
    pub opening: L,
    /// Proof-level ZK hiding material fixed at batched-prove startup.
    #[cfg(feature = "zk")]
    pub zk_hiding: ZkHidingProverState<F>,
}

impl<F: FieldCore, L: FieldCore> RecursiveProverState<F, L> {
    /// Logical witness represented by the carried opening claim.
    #[inline]
    pub fn logical_w(&self) -> &RecursiveWitnessFlat {
        self.logical_w.as_ref().unwrap_or(&self.w)
    }
}

#[cfg(not(feature = "zk"))]
fn pack_terminal_direct_final_witness<F, T>(
    transcript: &mut T,
    logical_w: &RecursiveWitnessFlat,
    terminal_layout: akita_types::TerminalWitnessSegmentLayout,
    final_log_basis: u32,
) -> Result<CleartextWitnessProof<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let final_witness = CleartextWitnessProof::PackedDigits(
        PackedDigits::from_i8_digits_with_min_bits(logical_w.as_i8_digits(), final_log_basis),
    );
    let parts = final_witness.terminal_transcript_parts(terminal_layout)?;
    if final_witness.packed_i8_digits()?.as_slice() != logical_w.as_i8_digits() {
        return Err(AkitaError::InvalidInput(
            "terminal final witness does not match direct terminal witness".to_string(),
        ));
    }
    transcript.append_bytes(ABSORB_TERMINAL_W_REMAINDER, &parts.remainder);
    Ok(final_witness)
}

/// Cursor into the proof-level hiding witness allocated at batched-prove start.
#[cfg(feature = "zk")]
#[derive(Debug, PartialEq, Eq)]
pub struct ZkHidingProverState<F: FieldCore> {
    hiding_witness: Vec<F>,
    cursor: usize,
}

/// Top-level hiding commitment pieces fixed before transcript replay starts.
#[cfg(feature = "zk")]
#[derive(Debug, PartialEq, Eq)]
pub struct ZkHidingCommitment<F: FieldCore> {
    /// Wire-visible commitment to the proof-level hiding witness.
    pub u_blind: Vec<F>,
    /// Dedicated short Ajtai blinding digits used for `u_blind`.
    pub b_blinding_digits: Vec<i8>,
}

#[cfg(feature = "zk")]
impl<F: FieldCore> ZkHidingProverState<F> {
    fn new(hiding_witness: Vec<F>) -> Self {
        Self {
            hiding_witness,
            cursor: 0,
        }
    }

    fn take_values(&mut self, len: usize) -> Result<&[F], AkitaError> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or(AkitaError::InvalidProof)?;
        let values = self
            .hiding_witness
            .get(self.cursor..end)
            .ok_or(AkitaError::InvalidProof)?;
        self.cursor = end;
        Ok(values)
    }

    fn take_ext_scalar<L>(&mut self) -> Result<L, AkitaError>
    where
        L: ExtField<F>,
    {
        Ok(L::from_base_slice(self.take_values(L::EXT_DEGREE)?))
    }

    fn take_ring<const D: usize>(&mut self) -> Result<(usize, CyclotomicRing<F, D>), AkitaError> {
        let start = self.cursor;
        let coeffs = self.take_values(D)?;
        let ring = CyclotomicRing::from_coefficients(from_fn(|idx| coeffs[idx]));
        Ok((start, ring))
    }

    fn into_proof(self, commitment: ZkHidingCommitment<F>) -> Result<ZkHidingProof<F>, AkitaError> {
        if self.cursor != self.hiding_witness.len() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(ZkHidingProof {
            u_blind: commitment.u_blind,
            hiding_witness: self.hiding_witness,
            b_blinding_digits: commitment.b_blinding_digits,
        })
    }

    fn take_next_w_eval_mask<L>(&mut self) -> Result<L, AkitaError>
    where
        L: ExtField<F>,
    {
        self.take_ext_scalar()
    }

    fn take_eq_factored_rounds<L>(
        &mut self,
        rounds: usize,
        degree: usize,
    ) -> Result<Vec<EqFactoredUniPoly<L>>, AkitaError>
    where
        L: ExtField<F>,
    {
        let stored_coeffs = EqFactoredUniPoly::<L>::stored_coeff_count_for_degree(degree);
        (0..rounds)
            .map(|_| {
                let coeffs = (0..stored_coeffs)
                    .map(|_| self.take_ext_scalar())
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(EqFactoredUniPoly {
                    coeffs_except_linear_term: coeffs,
                })
            })
            .collect()
    }

    fn take_compressed_rounds<L>(
        &mut self,
        rounds: usize,
        degree: usize,
    ) -> Result<Vec<CompressedUniPoly<L>>, AkitaError>
    where
        L: ExtField<F>,
    {
        (0..rounds)
            .map(|_| {
                let coeffs = (0..degree)
                    .map(|_| self.take_ext_scalar())
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(CompressedUniPoly {
                    coeffs_except_linear_term: coeffs,
                })
            })
            .collect()
    }

    fn take_current_level_pads<L>(
        &mut self,
        rounds: usize,
        b: usize,
    ) -> Result<ZkLevelRoundPads<L>, AkitaError>
    where
        L: ExtField<F>,
    {
        let mut stage1_round_pads = Vec::new();
        let mut stage1_child_claim_masks = Vec::new();
        for shape in stage1_tree_stage_shapes(rounds, b) {
            stage1_round_pads.push(
                self.take_eq_factored_rounds(shape.sumcheck_proof.0, shape.sumcheck_proof.1)?,
            );
            if shape.child_claims != 0 {
                stage1_child_claim_masks.push(
                    (0..shape.child_claims)
                        .map(|_| self.take_ext_scalar())
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
        let stage2_round_pads = self.take_compressed_rounds(rounds, 3)?;
        Ok((
            stage1_round_pads,
            stage1_child_claim_masks,
            stage2_round_pads,
        ))
    }

    fn take_extension_opening_reduction_pads<L>(
        &mut self,
        partials: usize,
        rounds: usize,
    ) -> Result<(Vec<L>, Vec<CompressedUniPoly<L>>), AkitaError>
    where
        L: ExtField<F>,
    {
        let partial_masks = (0..partials)
            .map(|_| self.take_ext_scalar())
            .collect::<Result<Vec<_>, _>>()?;
        let round_pads =
            self.take_compressed_rounds(rounds, akita_types::EXTENSION_OPENING_REDUCTION_DEGREE)?;
        Ok((partial_masks, round_pads))
    }
}

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: FieldCore, L: FieldCore> {
    /// Fold proof produced at this level.
    pub level_proof: AkitaLevelProof<F, L>,
    /// Recursive prover state for the next level.
    pub next_state: RecursiveProverState<F, L>,
}

/// Raw pieces produced by the unified root-level prover.
///
/// Callers assemble either a singleton or batched root proof from these
/// components while sharing the same inner prover flow.
pub struct RootLevelRawOutput<F: FieldCore, L: FieldCore, const D: usize> {
    /// Proof-level ZK hiding commitment fixed before root challenges.
    #[cfg(feature = "zk")]
    pub zk_hiding_commitment: ZkHidingCommitment<F>,
    /// Gamma-combined public y-rings, one per opening point.
    pub y_rings: Vec<CyclotomicRing<F, D>>,
    /// Optional extension-opening reduction payload for folded root openings.
    /// `None` when the root proof uses ordinary degree-one openings.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    /// Public v rows for the root relation.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Stage-1 sumcheck proof.
    pub stage1: AkitaStage1Proof<L>,
    /// Stage-2 sumcheck proof.
    #[cfg(not(feature = "zk"))]
    pub stage2_sumcheck_proof: SumcheckProof<L>,
    /// ZK plain-opening round masks for the stage-2 sumcheck.
    #[cfg(feature = "zk")]
    pub stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
    /// Stage-3 setup product-sumcheck proof for recursive setup-contribution replay.
    pub stage3_sumcheck_proof: Option<SetupSumcheckProof<L>>,
    /// Recursive witness commitment carried in the proof.
    pub w_commitment_proof: FlatRingVec<F>,
    /// Claimed terminal evaluation of the recursive witness at this level.
    pub w_eval: L,
    /// Recursive prover state for the first suffix level.
    pub next_state: RecursiveProverState<F, L>,
}

/// Outcome of the recursive fold suffix after the root level.
pub struct RecursiveSuffixOutcome<F: FieldCore, L: FieldCore> {
    /// Per-level intermediate fold proofs, in order. Does not include the
    /// root proof or the terminal-level proof.
    pub intermediate_levels: Vec<AkitaLevelProof<F, L>>,
    /// Terminal fold proof shipping `final_witness` in cleartext.
    pub terminal: TerminalLevelProof<F, L>,
    /// Proof-level ZK hiding witness state after all suffix masks are consumed.
    #[cfg(feature = "zk")]
    pub zk_hiding: ZkHidingProverState<F>,
    /// Total fold-level count reached, including the root level and the
    /// terminal level.
    pub num_levels: usize,
}

fn root_claim_opening_from_y_ring<F, E, const D: usize>(
    y_ring: &CyclotomicRing<F, D>,
    prepared_point: &PreparedRootOpeningPoint<F, D>,
    inner_opening_point: &[E],
    basis: BasisMode,
) -> Result<E, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: RingSubfieldEncoding<F>,
{
    if <E as ExtField<F>>::EXT_DEGREE == 1 {
        return (*y_ring * prepared_point.inner_reduction.sigma_m1())
            .coefficients()
            .first()
            .copied()
            .map(E::lift_base)
            .ok_or_else(|| AkitaError::InvalidInput("empty root y-ring".to_string()));
    }
    if !D.is_multiple_of(<E as ExtField<F>>::EXT_DEGREE)
        || !(D / <E as ExtField<F>>::EXT_DEGREE).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "claim-field degree must divide the ring dimension into power-of-two slots".to_string(),
        ));
    }
    let packed_slots = D / <E as ExtField<F>>::EXT_DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if inner_opening_point.len() > packed_inner_bits
        && inner_opening_point[packed_inner_bits..]
            .iter()
            .any(|coord| !coord.is_zero())
    {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: inner_opening_point.len(),
        });
    }
    let mut point =
        inner_opening_point[..inner_opening_point.len().min(packed_inner_bits)].to_vec();
    point.resize(packed_inner_bits, E::zero());
    let weights = basis_weights(&point, basis)?;
    let inner_reduction = embed_ring_subfield_vector::<F, E, D>(
        &weights,
        AkitaError::InvalidInput(
            "root opening point does not encode in the ring-subfield basis".to_string(),
        ),
    )?;
    recover_ring_subfield_inner_product::<F, E, D>(y_ring, &inner_reduction)
}

fn row_coefficient_rings<F, L, const D: usize>(
    coefficients: &[L],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    L: RingSubfieldEncoding<F>,
{
    coefficients
        .iter()
        .copied()
        .map(|coefficient| {
            embed_ring_subfield_scalar::<F, L, D>(
                coefficient,
                AkitaError::InvalidInput(
                    "public-row coefficient does not encode in the ring-subfield basis".to_string(),
                ),
            )
        })
        .collect()
}

fn combine_root_y_rings<F, const D: usize>(
    per_claim_y_rings: &[CyclotomicRing<F, D>],
    incidence: &ClaimIncidenceSummary,
    row_coefficient_rings: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if per_claim_y_rings.len() != incidence.num_claims()
        || row_coefficient_rings.len() != incidence.num_claims()
        || incidence.claim_to_point().len() != incidence.num_claims()
    {
        return Err(AkitaError::InvalidInput(
            "root y-ring batching input lengths do not match".to_string(),
        ));
    }

    let mut y_rings = vec![CyclotomicRing::<F, D>::zero(); incidence.num_public_rows()];
    for (row_idx, row) in incidence.public_rows().iter().enumerate() {
        if row.claim_indices().is_empty() || row.point_idx() >= incidence.num_points() {
            return Err(AkitaError::InvalidInput(
                "root y-ring public-row incidence is invalid".to_string(),
            ));
        }
        for &claim_idx in row.claim_indices() {
            if claim_idx >= per_claim_y_rings.len()
                || incidence.claim_to_point()[claim_idx] != row.point_idx()
            {
                return Err(AkitaError::InvalidInput(
                    "root y-ring public-row term is inconsistent".to_string(),
                ));
            }
            y_rings[row_idx] += row_coefficient_rings[claim_idx] * per_claim_y_rings[claim_idx];
        }
    }
    Ok(y_rings)
}

/// Config-free flattened view of batched prover claims.
pub struct PreparedBatchedProveInputs<'a, F: FieldCore, E: FieldCore, P, const D: usize> {
    /// Distinct opening points in caller order.
    pub opening_points: Vec<&'a [E]>,
    /// Commitments flattened in point/group order.
    pub commitments_by_point: Vec<RingCommitment<F, D>>,
    /// Normalized incidence summary that owns canonical root claim routing.
    pub incidence_summary: ClaimIncidenceSummary,
    /// Polynomials flattened in claim order.
    pub flat_polys: Vec<&'a P>,
    /// Polynomials flattened in committed-group order.
    pub group_polys: Vec<&'a P>,
    /// Commitment hints flattened in claim-group order.
    pub flat_hints: Vec<AkitaCommitmentHint<F, D>>,
}

/// Assemble intermediate fold-level proofs followed by the terminal-level
/// proof.
///
/// The terminal proof already carries the cleartext `final_witness` (in
/// place of the prior `next_w_commitment`), so the recursive suffix is
/// `Intermediate(...) × N + Terminal(...)`.
pub fn build_final_proof_steps<F, L>(
    intermediate_levels: Vec<AkitaLevelProof<F, L>>,
    terminal: TerminalLevelProof<F, L>,
) -> Vec<AkitaProofStep<F, L>>
where
    F: FieldCore,
    L: ExtField<F>,
{
    let mut steps = intermediate_levels
        .into_iter()
        .map(AkitaProofStep::Intermediate)
        .collect::<Vec<_>>();
    steps.push(AkitaProofStep::Terminal(terminal));
    steps
}

#[cfg(feature = "zk")]
type ZkLevelRoundPads<L> = (
    Vec<Vec<akita_sumcheck::EqFactoredUniPoly<L>>>,
    Vec<Vec<L>>,
    Vec<akita_sumcheck::CompressedUniPoly<L>>,
);

#[cfg(feature = "zk")]
fn push_random_ext_scalar_slots<F, L>(out: &mut Vec<F>, rng: &mut OsRng)
where
    F: FieldCore + RandomSampling,
    L: ExtField<F>,
{
    out.extend((0..L::EXT_DEGREE).map(|_| F::random(&mut *rng)));
}

#[cfg(feature = "zk")]
fn append_zk_stage2_pad_slots<F, L>(rounds: usize, out: &mut Vec<F>, rng: &mut OsRng)
where
    F: FieldCore + RandomSampling,
    L: ExtField<F>,
{
    for _ in 0..rounds * 3 {
        push_random_ext_scalar_slots::<F, L>(out, rng);
    }
}

#[cfg(feature = "zk")]
fn append_zk_level_pad_slots<F, L>(
    params: &LevelParams,
    next_w_len: usize,
    include_stage1: bool,
    out: &mut Vec<F>,
    rng: &mut OsRng,
) -> Result<(), AkitaError>
where
    F: FieldCore + RandomSampling,
    L: ExtField<F>,
{
    let rounds = sumcheck_rounds(params.ring_dimension, next_w_len);
    if !include_stage1 {
        append_zk_stage2_pad_slots::<F, L>(rounds, out, rng);
        return Ok(());
    }
    let b = 1usize << params.log_basis;
    for shape in stage1_tree_stage_shapes(rounds, b) {
        let stored_coeffs =
            EqFactoredUniPoly::<L>::stored_coeff_count_for_degree(shape.sumcheck_proof.1);
        for _ in 0..shape.sumcheck_proof.0 * stored_coeffs {
            push_random_ext_scalar_slots::<F, L>(out, rng);
        }
        for _ in 0..shape.child_claims {
            push_random_ext_scalar_slots::<F, L>(out, rng);
        }
    }
    append_zk_stage2_pad_slots::<F, L>(rounds, out, rng);
    Ok(())
}

#[cfg(feature = "zk")]
fn append_zk_extension_reduction_slots<F, L>(
    partials: usize,
    rounds: usize,
    out: &mut Vec<F>,
    rng: &mut OsRng,
) where
    F: FieldCore + RandomSampling,
    L: ExtField<F>,
{
    let round_coeffs = akita_types::EXTENSION_OPENING_REDUCTION_DEGREE;
    for _ in 0..(partials + rounds * round_coeffs) {
        push_random_ext_scalar_slots::<F, L>(out, rng);
    }
}

#[cfg(feature = "zk")]
fn build_zk_hiding_context<F, E, L, B, const D: usize>(
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    schedule: &Schedule,
    root_commit_params: &LevelParams,
    num_vars: usize,
    num_claims: usize,
    num_root_points: usize,
) -> Result<(ZkHidingCommitment<F>, ZkHidingProverState<F>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F>,
    L: RingSubfieldEncoding<F> + ExtField<E> + ExtField<F>,
    B: ProverComputeBackend<F>,
{
    let mut rng = OsRng;
    let fold_steps = schedule
        .steps
        .iter()
        .filter_map(|step| match step {
            Step::Fold(fold) => Some(fold),
            Step::Direct(_) => None,
        })
        .collect::<Vec<_>>();
    let mut hiding_witness = Vec::new();

    if root_tensor_projection_enabled::<F, E, L, D>(num_vars) {
        let split_bits = <L as ExtField<F>>::EXT_DEGREE.trailing_zeros() as usize;
        append_zk_extension_reduction_slots::<F, L>(
            num_claims * <L as ExtField<F>>::EXT_DEGREE,
            num_vars - split_bits,
            &mut hiding_witness,
            &mut rng,
        );
    }
    // Root-level ring masks: one D-coefficient ring per requested opening point.
    // Later added to `y_rings` before the root ring-switch / sumcheck flow.
    hiding_witness.extend((0..num_root_points * D).map(|_| F::random(&mut rng)));
    if let Some(root_step) = fold_steps.first() {
        // Terminal folds skip Stage 1 and consume only Stage 2 pads.
        let root_has_stage1 = fold_steps.len() > 1;
        append_zk_level_pad_slots::<F, L>(
            &root_step.params,
            root_step.next_w_len,
            root_has_stage1,
            &mut hiding_witness,
            &mut rng,
        )?;
        if fold_steps.len() > 1 {
            // Root fold scalar: added to the root level's final next-witness
            // evaluation claim (`w_eval`) after Stage 2. Terminal roots have
            // no next witness and therefore consume no next-w eval mask.
            push_random_ext_scalar_slots::<F, L>(&mut hiding_witness, &mut rng);
        }
        let mut current_opening_vars =
            sumcheck_rounds(root_step.params.ring_dimension, root_step.next_w_len);
        for (step_idx, step) in fold_steps.iter().enumerate().skip(1) {
            if <L as ExtField<F>>::EXT_DEGREE > 1 {
                let split_bits = <L as ExtField<F>>::EXT_DEGREE.trailing_zeros() as usize;
                append_zk_extension_reduction_slots::<F, L>(
                    <L as ExtField<F>>::EXT_DEGREE,
                    current_opening_vars - split_bits,
                    &mut hiding_witness,
                    &mut rng,
                );
            }
            // Recursive-level ring mask: added to that level's `y_ring` before
            // ring-switching so the current ring-relation value is hidden.
            hiding_witness.extend((0..D).map(|_| F::random(&mut rng)));
            // Terminal recursive folds skip Stage 1 and consume only Stage 2 pads.
            let include_stage1 = step_idx + 1 < fold_steps.len();
            append_zk_level_pad_slots::<F, L>(
                &step.params,
                step.next_w_len,
                include_stage1,
                &mut hiding_witness,
                &mut rng,
            )?;
            if include_stage1 {
                // Recursive fold scalar: added to non-terminal levels' final
                // next-witness evaluation claim (`w_eval`) after Stage 2.
                push_random_ext_scalar_slots::<F, L>(&mut hiding_witness, &mut rng);
            }
            current_opening_vars = sumcheck_rounds(step.params.ring_dimension, step.next_w_len);
        }
    }
    let (u_blind, b_blinding_digits) = commit_zk_hiding_witness::<F, B, D>(
        backend,
        prepared,
        root_commit_params,
        &hiding_witness,
    )?;
    Ok((
        ZkHidingCommitment {
            u_blind,
            b_blinding_digits,
        },
        ZkHidingProverState::new(hiding_witness),
    ))
}
