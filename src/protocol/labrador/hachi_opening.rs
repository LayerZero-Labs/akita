//! Native Hachi -> Labrador opening frontend.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
use crate::primitives::serialization::{Compress, Valid};
use crate::protocol::commitment::transcript_append::AppendToTranscript;
use crate::protocol::commitment::{CommitmentConfig, HachiExpandedSetup, RingCommitment};
use crate::protocol::hachi_poly_ops::{BalancedDigitPoly, HachiPolyOps};
use crate::protocol::labrador::config::{
    estimate_handoff_recursive_proof, LabradorRecursiveSizeEstimate,
};
use crate::protocol::labrador::hachi_statement::{
    build_hachi_opening_constraints, build_hachi_opening_witness,
};
use crate::protocol::labrador::types::{LabradorStatement, LabradorWitness};
use crate::protocol::labrador::{prove_with_plan, verify as verify_labrador};
use crate::protocol::opening_point::{ring_opening_point_from_field, BasisMode, RingOpeningPoint};
use crate::protocol::proof::{
    FlatCommitmentHint, FlatLabradorProof, FlatLabradorWitness, FlatRingVec, HachiCommitmentHint,
    LabradorHandoffKind, LabradorTail, PackedDigits,
};
use crate::protocol::ring_switch::WCommitmentConfig;
use crate::protocol::transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt, HachiSerialize};
use std::time::Instant;

/// Canonical public handoff claim at the end of the Hachi folding loop.
#[derive(Debug, Clone)]
pub(crate) struct HachiOpeningPublic<F: FieldCore> {
    /// D-erased commitment to the recursive witness `current_w`.
    pub commitment: FlatRingVec<F>,
    /// Canonical recursive opening point.
    pub point: Vec<F>,
    /// Claimed scalar evaluation of `current_w` at `point`.
    pub claimed_eval: F,
    /// Digit basis used if the handoff falls back to a direct packed tail.
    pub witness_log_basis: u32,
}

/// Prover-side witness material available at the handoff boundary.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HachiOpeningWitnessRef<'a> {
    pub w_digits: &'a [i8],
    pub hint: &'a FlatCommitmentHint,
}

/// Typed public handoff claim after runtime D dispatch.
pub(crate) struct HachiOpeningClaimRef<'a, F: FieldCore, const D: usize> {
    pub commitment: &'a RingCommitment<F, D>,
    pub point: &'a [F],
    pub claimed_eval: &'a F,
    pub witness_log_basis: u32,
}

/// Typed prover witness after runtime D dispatch.
pub(crate) struct HachiOpeningProverWitnessRef<'a, F: FieldCore, const D: usize> {
    pub w_digits: &'a [i8],
    pub hint: &'a HachiCommitmentHint<F, D>,
}

#[derive(Debug, Clone)]
pub(crate) struct HachiOpeningEstimate {
    pub estimated_tail_bytes: usize,
    pub proof_bytes: usize,
    pub final_witness_bytes: usize,
    pub serialized_witness_bytes: usize,
}

struct PreparedHachiOpening<F: FieldCore, const D: usize> {
    padded_point: Vec<F>,
    statement: LabradorStatement<F, D>,
    witness: LabradorWitness<F, D>,
    y_ring: CyclotomicRing<F, D>,
    witness_norm_bound_sq: u128,
    estimate: LabradorRecursiveSizeEstimate,
}

fn direct_tail_from_claim<F: FieldCore>(
    claim: &HachiOpeningPublic<F>,
    witness: HachiOpeningWitnessRef<'_>,
) -> PackedDigits {
    PackedDigits::from_i8_digits(witness.w_digits, claim.witness_log_basis)
}

fn initial_row_lengths<F: FieldCore>(proof: &FlatLabradorProof<F>) -> Vec<usize> {
    if let Some(level0) = proof.levels.first() {
        level0.input_row_lengths.clone()
    } else {
        proof
            .final_opening_witness
            .rows
            .iter()
            .map(|row| row.count())
            .collect()
    }
}

fn resolve_handoff_point<F, const D: usize, Cfg>(
    claim: &HachiOpeningClaimRef<'_, F, D>,
) -> Result<
    (
        crate::protocol::commitment::HachiCommitmentLayout,
        Vec<F>,
        RingOpeningPoint<F>,
    ),
    HachiError,
>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    let alpha = D.trailing_zeros() as usize;
    if claim.point.len() < alpha {
        return Err(HachiError::InvalidPointDimension {
            expected: alpha,
            actual: claim.point.len(),
        });
    }

    let layout = <WCommitmentConfig<D, Cfg>>::commitment_layout(claim.point.len())?;
    let target_num_vars = layout.m_vars + layout.r_vars + alpha;
    let mut padded_point = claim.point.to_vec();
    padded_point.resize(target_num_vars, F::zero());
    let outer_point = &padded_point[alpha..];
    let ring_opening_point = ring_opening_point_from_field::<F>(
        outer_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
    )?;
    Ok((layout, padded_point, ring_opening_point))
}

fn matches_opening_claim<F: FieldCore + CanonicalField, const D: usize>(
    y_ring: &CyclotomicRing<F, D>,
    opening_point: &[F],
    opening_value: &F,
) -> bool {
    let alpha = D.trailing_zeros() as usize;
    let coeff_point = &opening_point[..alpha];
    let mut coeff_basis = vec![F::zero(); D];
    multilinear_lagrange_basis(&mut coeff_basis, coeff_point);
    let inner_ring = CyclotomicRing::from_slice(&coeff_basis);
    let d = F::from_u64(D as u64);
    let trace_lhs = (*y_ring * inner_ring.sigma_m1()).coefficients()[0] * d;
    trace_lhs == d * *opening_value
}

fn prepare_hachi_opening<F, const D: usize, Cfg>(
    claim: &HachiOpeningClaimRef<'_, F, D>,
    witness_ref: &HachiOpeningProverWitnessRef<'_, F, D>,
    expanded_setup: &HachiExpandedSetup<F>,
) -> Result<PreparedHachiOpening<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + HachiSerialize,
    Cfg: CommitmentConfig,
{
    let (layout, padded_point, ring_opening_point) = resolve_handoff_point::<F, D, Cfg>(claim)?;
    let w_poly = BalancedDigitPoly::<F, D>::from_i8_digits(witness_ref.w_digits)?;
    let (y_ring, _w_folded) = w_poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    if !matches_opening_claim::<F, D>(&y_ring, claim.point, claim.claimed_eval) {
        return Err(HachiError::InvalidInput(
            "native handoff boundary does not match the claimed opening".to_string(),
        ));
    }

    let witness = build_hachi_opening_witness::<F, D>(witness_ref.w_digits, witness_ref.hint)?;
    let current_w_ring_len = witness.rows().first().map_or(0, Vec::len);
    let constraints = build_hachi_opening_constraints::<F, D, WCommitmentConfig<D, Cfg>>(
        &expanded_setup.A,
        &expanded_setup.B,
        &ring_opening_point,
        claim.commitment,
        &y_ring,
        layout,
        current_w_ring_len,
    )?;
    let witness_norm_bound_sq = witness.norm();
    let estimate = estimate_handoff_recursive_proof::<F, D>(&witness, layout.log_basis as usize)?;
    let statement = LabradorStatement {
        inner_opening_payload: Vec::new(),
        linear_garbage_payload: Vec::new(),
        challenges: Vec::new(),
        constraints,
        reduced_constraints: None,
        witness_norm_bound_sq,
    };
    Ok(PreparedHachiOpening {
        padded_point,
        statement,
        witness,
        y_ring,
        witness_norm_bound_sq,
        estimate,
    })
}

pub(crate) fn estimate_hachi_opening_handoff<F, const D: usize, Cfg>(
    claim: &HachiOpeningClaimRef<'_, F, D>,
    witness_ref: &HachiOpeningProverWitnessRef<'_, F, D>,
    expanded_setup: &HachiExpandedSetup<F>,
) -> Result<HachiOpeningEstimate, HachiError>
where
    F: FieldCore + CanonicalField + HachiSerialize,
    Cfg: CommitmentConfig,
{
    let prepared = prepare_hachi_opening::<F, D, Cfg>(claim, witness_ref, expanded_setup)?;
    let serialized_witness_bytes =
        FlatLabradorWitness::from_typed(&prepared.witness).serialized_size(Compress::No);
    let y_ring_bytes = FlatRingVec::from_single(&prepared.y_ring).serialized_size(Compress::No);
    Ok(HachiOpeningEstimate {
        estimated_tail_bytes: prepared.estimate.proof_bytes
            + y_ring_bytes
            + prepared.witness_norm_bound_sq.serialized_size(Compress::No),
        proof_bytes: prepared.estimate.proof_bytes,
        final_witness_bytes: prepared.estimate.final_witness_bytes,
        serialized_witness_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hachi_opening_prove<F, T, const D: usize, Cfg>(
    claim: &HachiOpeningClaimRef<'_, F, D>,
    witness_ref: &HachiOpeningProverWitnessRef<'_, F, D>,
    expanded_setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
) -> Result<Option<LabradorTail<F>>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid + HachiSerialize,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    let t0 = Instant::now();
    let direct_tail = PackedDigits::from_i8_digits(witness_ref.w_digits, claim.witness_log_basis);
    let direct_tail_bytes = direct_tail.serialized_size(Compress::No);
    let prepared = prepare_hachi_opening::<F, D, Cfg>(claim, witness_ref, expanded_setup)?;
    let y_ring_bytes = FlatRingVec::from_single(&prepared.y_ring).serialized_size(Compress::No);
    let estimated_tail_bytes = prepared.estimate.proof_bytes
        + y_ring_bytes
        + prepared.witness_norm_bound_sq.serialized_size(Compress::No);
    tracing::info!(
        packed_direct_bytes = direct_tail_bytes,
        estimated_hachi_opening_tail_bytes = estimated_tail_bytes,
        estimated_labrador_proof_bytes = prepared.estimate.proof_bytes,
        estimated_labrador_final_witness_bytes = prepared.estimate.final_witness_bytes,
        selected_tail = if estimated_tail_bytes < direct_tail_bytes {
            "labrador_opening"
        } else {
            "direct"
        },
        "hachi_opening estimated tail comparison"
    );
    if estimated_tail_bytes >= direct_tail_bytes {
        return Ok(None);
    }

    let mut handoff_transcript = transcript.clone();
    claim
        .commitment
        .append_to_transcript(ABSORB_COMMITMENT, &mut handoff_transcript);
    for pt in &prepared.padded_point {
        handoff_transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    handoff_transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &prepared.y_ring);

    let comkey_seed = expanded_setup.labrador_comkey_seed();
    let plan = prepared.estimate.initial_plan.clone();
    let labrador_proof = prove_with_plan::<F, T, D>(
        prepared.witness.clone(),
        &prepared.statement,
        &plan,
        &comkey_seed,
        &mut handoff_transcript,
    )?;
    #[cfg(debug_assertions)]
    {
        let roundtrip = FlatLabradorProof::from_typed(&labrador_proof).to_typed::<D>();
        assert!(
            roundtrip == labrador_proof,
            "native Labrador proof roundtrip must preserve the proof"
        );

        let mut self_verify_transcript = handoff_transcript.clone();
        verify_labrador::<F, T, D>(
            &prepared.statement,
            &labrador_proof,
            &comkey_seed,
            &mut self_verify_transcript,
        )
        .expect("freshly generated native Labrador proof must verify");
    }
    let tail = LabradorTail {
        handoff_kind: LabradorHandoffKind::Opening,
        labrador_proof: FlatLabradorProof::from_typed(&labrador_proof),
        v: FlatRingVec::empty_with_ring_dim(D),
        y_ring: FlatRingVec::from_single(&prepared.y_ring),
        witness_norm_bound_sq: prepared.witness_norm_bound_sq,
    };
    let actual_tail_bytes = tail.serialized_size(Compress::No);
    tracing::info!(
        packed_direct_bytes = direct_tail_bytes,
        actual_hachi_opening_tail_bytes = actual_tail_bytes,
        levels = labrador_proof.levels.len(),
        "native hachi opening actual tail comparison"
    );
    if actual_tail_bytes >= direct_tail_bytes {
        return Ok(None);
    }
    *transcript = handoff_transcript;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        levels = labrador_proof.levels.len(),
        "native hachi opening prove complete"
    );
    Ok(Some(tail))
}

pub(crate) fn hachi_opening_verify<F, T, const D: usize, Cfg>(
    tail: &LabradorTail<F>,
    claim: &HachiOpeningClaimRef<'_, F, D>,
    expanded_setup: &HachiExpandedSetup<F>,
    transcript: &mut T,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + Valid + HachiSerialize,
    T: Transcript<F>,
    Cfg: CommitmentConfig,
{
    if tail.handoff_kind != LabradorHandoffKind::Opening {
        return Err(HachiError::InvalidProof);
    }

    let y_ring = tail.y_ring.try_to_single::<D>()?;
    if !matches_opening_claim::<F, D>(&y_ring, claim.point, claim.claimed_eval) {
        return Err(HachiError::InvalidProof);
    }

    let initial_row_lengths = initial_row_lengths(&tail.labrador_proof);
    if initial_row_lengths.len() != 2 {
        return Err(HachiError::InvalidProof);
    }

    let (layout, padded_point, ring_opening_point) = resolve_handoff_point::<F, D, Cfg>(claim)?;
    let constraints = build_hachi_opening_constraints::<F, D, WCommitmentConfig<D, Cfg>>(
        &expanded_setup.A,
        &expanded_setup.B,
        &ring_opening_point,
        claim.commitment,
        &y_ring,
        layout,
        initial_row_lengths[0],
    )?;
    let statement = LabradorStatement {
        inner_opening_payload: Vec::new(),
        linear_garbage_payload: Vec::new(),
        challenges: Vec::new(),
        constraints,
        reduced_constraints: None,
        witness_norm_bound_sq: tail.witness_norm_bound_sq,
    };

    claim
        .commitment
        .append_to_transcript(ABSORB_COMMITMENT, transcript);
    for pt in &padded_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

    let labrador_proof = tail.labrador_proof.to_typed::<D>();
    let comkey_seed = expanded_setup.labrador_comkey_seed();
    verify_labrador::<F, T, D>(&statement, &labrador_proof, &comkey_seed, transcript)?;
    Ok(())
}

pub(crate) fn direct_tail_from_boundary<F: FieldCore>(
    claim: &HachiOpeningPublic<F>,
    witness: HachiOpeningWitnessRef<'_>,
) -> PackedDigits {
    direct_tail_from_claim(claim, witness)
}
