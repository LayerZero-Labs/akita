//! Prover flow state shared by root orchestration during crate extraction.

use crate::crt_ntt::NttSlotCache;
use crate::ring_switch::{
    ring_switch_build_w, ring_switch_finalize, ring_switch_finalize_with_claim_groups,
    RingSwitchOutput,
};
use crate::{
    HachiStage1Prover, HachiStage2Prover, QuadraticEquation, RecursiveCommitmentHintCache,
    RecursiveWitnessFlat,
};
use akita_algebra::fields::wide::HasWide;
use akita_algebra::fields::HasUnreducedOps;
use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, FieldCore, FieldSampling, HachiError};
use akita_sumcheck::{prove_sumcheck, SumcheckProof};
use akita_transcript::labels::{
    ABSORB_SUMCHECK_S_CLAIM, CHALLENGE_SUMCHECK_BATCH, CHALLENGE_SUMCHECK_ROUND,
};
use akita_transcript::Transcript;
use akita_types::{
    relation_claim_from_rows, reorder_stage1_coords, DirectWitnessProof, FlatRingVec,
    HachiExpandedSetup, HachiLevelProof, HachiProofStep, HachiStage1Proof, PackedDigits, Schedule,
    Step,
};

/// Runtime state carried between recursive prove levels.
pub struct RecursiveProverState<F: FieldCore> {
    /// Current recursive witness.
    pub w: RecursiveWitnessFlat,
    /// Current recursive witness commitment.
    pub commitment: FlatRingVec<F>,
    /// D-erased recursive commitment hint cache.
    pub hint: RecursiveCommitmentHintCache<F>,
    /// Current digit basis, as `log2(b)`.
    pub log_basis: u32,
    /// Sumcheck challenges that become the next recursive opening point.
    pub sumcheck_challenges: Vec<F>,
}

/// Output from a single prove level, used to extend proof wire data and state.
pub struct ProveLevelOutput<F: FieldCore> {
    /// Fold proof produced at this level.
    pub level_proof: HachiLevelProof<F>,
    /// Recursive prover state for the next level.
    pub next_state: RecursiveProverState<F>,
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
    pub stage1: HachiStage1Proof<F>,
    /// Stage-2 sumcheck proof.
    pub stage2_sumcheck: SumcheckProof<F>,
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
    pub levels: Vec<HachiLevelProof<F>>,
    /// Total fold-level count reached, including the root level.
    pub num_levels: usize,
    /// Prover state at the terminal direct step.
    pub final_state: RecursiveProverState<F>,
    /// `log_basis` for the terminal packed-digit witness.
    pub final_log_basis: u32,
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
) -> Result<u32, HachiError>
where
    F: FieldCore,
{
    let Some(Step::Direct(direct_step)) = schedule.steps.last() else {
        return Err(HachiError::InvalidSetup(
            "schedule must terminate in a direct step".to_string(),
        ));
    };
    if direct_step.current_w_len != current_state.w.len()
        || direct_step.bits_per_elem != current_state.log_basis
    {
        return Err(HachiError::InvalidSetup(
            "scheduled direct step did not match final runtime state".to_string(),
        ));
    }
    Ok(direct_step.bits_per_elem)
}

/// Assemble fold-level proofs followed by the terminal packed-digit witness.
pub fn build_final_proof_steps<F>(
    levels: Vec<HachiLevelProof<F>>,
    final_state: &RecursiveProverState<F>,
    final_log_basis: u32,
) -> Vec<HachiProofStep<F>>
where
    F: FieldCore,
{
    let final_w =
        PackedDigits::from_i8_digits_with_min_bits(final_state.w.as_i8_digits(), final_log_basis);
    let mut steps = levels
        .into_iter()
        .map(HachiProofStep::Fold)
        .collect::<Vec<_>>();
    steps.push(HachiProofStep::Direct(DirectWitnessProof::PackedDigits(
        final_w,
    )));
    steps
}

/// Prove one recursive fold level after the caller has built its quadratic
/// equation and selected the commitment policy for the next `w`.
///
/// The caller owns config/schedule decisions through `commit_w_for_next`; this
/// function owns the config-free prover mechanics: build `w`, commit it using
/// that closure, finish ring switching, run stage-1/stage-2 sumchecks, and
/// produce the next recursive state.
///
/// # Errors
///
/// Returns an error if ring switching, recursive commitment, or either
/// sumcheck prover fails.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn prove_fold_level_from_quadratic<F, T, const D: usize, CommitW>(
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    commitment_u: &[CyclotomicRing<F, D>],
    level: usize,
    lp: &akita_types::LevelParams,
    next_log_basis: u32,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_ring: CyclotomicRing<F, D>,
    commit_w_for_next: CommitW,
) -> Result<ProveLevelOutput<F>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError>,
{
    let w = ring_switch_build_w::<F, { D }>(&mut quad_eq, expanded, ntt_shared, lp)?;
    let (w_commitment_flat, w_hint_cache) = {
        let _span = tracing::info_span!("commit_w_level", level).entered();
        commit_w_for_next(&w)?
    };
    let w_commitment_proof = w_commitment_flat.clone();

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

    let relation_claim = relation_claim_from_rows::<F, D>(
        &rs.tau1,
        rs.alpha,
        &quad_eq.v,
        commitment_u,
        std::slice::from_ref(&y_ring),
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
        tau1: _,
        b,
        alpha: _,
    } = rs;
    let w_commitment = w_commitment.ok_or_else(|| {
        HachiError::InvalidSetup("prover ring switch dropped w commitment".to_string())
    })?;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    let (stage1_proof, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = HachiStage1Prover::new(
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
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let (stage2_sumcheck, sumcheck_challenges, _stage2_final_claim, w_eval) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let mut stage2_prover = HachiStage2Prover::new(
            batching_coeff,
            w_evals_compact,
            &r_stage1,
            s_claim,
            b,
            alpha_evals_y,
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
        (
            stage2_sumcheck,
            sumcheck_challenges,
            stage2_final_claim,
            w_eval,
        )
    };

    let (level_proof, sumcheck_challenges) = (
        HachiLevelProof::new_two_stage::<D>(
            y_ring,
            quad_eq.v,
            stage1_proof,
            stage2_sumcheck,
            w_commitment_proof,
            w_eval,
        ),
        sumcheck_challenges,
    );

    Ok(ProveLevelOutput {
        level_proof,
        next_state: RecursiveProverState {
            w,
            commitment: w_commitment,
            hint: w_hint.ok_or_else(|| {
                HachiError::InvalidSetup(
                    "prover ring switch dropped recursive hint cache".to_string(),
                )
            })?,
            log_basis: next_log_basis,
            sumcheck_challenges,
        },
    })
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
    expanded: &HachiExpandedSetup<F>,
    ntt_shared: &NttSlotCache<D>,
    transcript: &mut T,
    commitment_rows: &[CyclotomicRing<F, D>],
    lp: &akita_types::LevelParams,
    expected_w_len: usize,
    next_log_basis: u32,
    mut quad_eq: Box<QuadraticEquation<F, { D }>>,
    y_rings: Vec<CyclotomicRing<F, D>>,
    commit_w_for_next: CommitW,
) -> Result<RootLevelRawOutput<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + HasUnreducedOps + HasWide,
    T: Transcript<F>,
    CommitW: FnOnce(
        &RecursiveWitnessFlat,
    ) -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), HachiError>,
{
    let w = ring_switch_build_w::<F, { D }>(&mut quad_eq, expanded, ntt_shared, lp)?;
    if w.len() != expected_w_len {
        return Err(HachiError::InvalidSetup(
            "scheduled root next-w length did not match runtime witness".to_string(),
        ));
    }
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

    let relation_claim =
        relation_claim_from_rows::<F, D>(&rs.tau1, rs.alpha, &quad_eq.v, commitment_rows, &y_rings);

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
        tau1: _,
        b,
        alpha: _,
    } = rs;
    let w_commitment = w_commitment.ok_or_else(|| {
        HachiError::InvalidSetup("prover ring switch dropped w commitment".to_string())
    })?;
    let tau0_reordered = reorder_stage1_coords(&tau0, col_bits, ring_bits);
    let (stage1_proof, r_stage1, s_claim) = {
        let _sumcheck_span = tracing::info_span!("stage1_sumcheck").entered();
        let stage1_prover = HachiStage1Prover::new(
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
    let batching_coeff: F = transcript.challenge_scalar(CHALLENGE_SUMCHECK_BATCH);
    let (stage2_sumcheck, sumcheck_challenges, _stage2_final_claim, w_eval) = {
        let _sumcheck_span = tracing::info_span!("stage2_sumcheck").entered();
        let mut stage2_prover = HachiStage2Prover::new(
            batching_coeff,
            w_evals_compact,
            &r_stage1,
            s_claim,
            b,
            alpha_evals_y,
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

    Ok(RootLevelRawOutput {
        y_rings,
        v: quad_eq.v,
        stage1: stage1_proof,
        stage2_sumcheck,
        w_commitment_proof,
        w_eval,
        next_state: RecursiveProverState {
            w,
            commitment: w_commitment,
            hint: w_hint.ok_or_else(|| {
                HachiError::InvalidSetup(
                    "prover ring switch dropped recursive hint cache".to_string(),
                )
            })?,
            log_basis: next_log_basis,
            sumcheck_challenges,
        },
    })
}
