//! Prover flow state shared by root orchestration during crate extraction.

use crate::{RecursiveCommitmentHintCache, RecursiveWitnessFlat};
use akita_algebra::CyclotomicRing;
use akita_field::{FieldCore, HachiError};
use akita_sumcheck::SumcheckProof;
use akita_types::{
    DirectWitnessProof, FlatRingVec, HachiLevelProof, HachiProofStep, HachiStage1Proof,
    PackedDigits, Schedule, Step,
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
