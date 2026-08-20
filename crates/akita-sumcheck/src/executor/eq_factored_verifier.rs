//! Verifier replay for master-lifted eq-factored executor batches.

use super::plan::CheckedEqFactoredBatch;
use super::shared::{absorb_claims_and_sample_coefficients, invalid, validate_claim_count};
use crate::{advance_eq_factored_claim, EqFactoredSumcheckProof};
use akita_error::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::{labels, Transcript};

/// One local terminal value needed to finish a verified eq-factored batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EqFactoredTerminalObligation<E: FieldCore> {
    /// Logical member index whose local terminal value is required.
    pub member_index: usize,
    /// Offset of this member's local point in the shared master point.
    pub suffix_offset: usize,
    /// Number of coordinates in this member's local point.
    pub local_rounds: usize,
    /// Front-loaded batching coefficient for this member.
    pub batching_coefficient: E,
    /// Equality value accumulated over this member's virtual master prefix.
    pub prefix_equality_scalar: E,
}

impl<E: FieldCore> EqFactoredTerminalObligation<E> {
    /// Borrow this obligation's local challenge point from the master point.
    pub fn point<'a>(&self, master_point: &'a [E]) -> Result<&'a [E], AkitaError> {
        let expected = self
            .suffix_offset
            .checked_add(self.local_rounds)
            .ok_or_else(|| invalid("eq-factored terminal point dimension overflows"))?;
        if master_point.len() != expected {
            return Err(AkitaError::InvalidPointDimension {
                expected,
                actual: master_point.len(),
            });
        }
        master_point
            .get(self.suffix_offset..)
            .ok_or_else(|| invalid("eq-factored terminal suffix is invalid"))
    }
}

/// Transcript round replay result and typed local terminal obligations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EqFactoredBatchRoundResult<E: FieldCore> {
    /// Shared master challenge point sampled while replaying the proof.
    pub master_point: Vec<E>,
    /// Front-loaded coefficient for each logical member.
    pub batching_coefficients: Vec<E>,
    /// Local terminal obligations in logical member order.
    pub terminal_obligations: Vec<EqFactoredTerminalObligation<E>>,
    final_scaled_claim: E,
    final_claim_scale: E,
}

impl<E: FieldCore> EqFactoredBatchRoundResult<E> {
    /// Check local terminal values in logical member order.
    ///
    /// Each local value is multiplied by its known master-prefix equality
    /// scalar before the denominator-free final recurrence is checked.
    pub fn check_terminal_claims(&self, terminal_claims: &[E]) -> Result<(), AkitaError> {
        if terminal_claims.len() != self.terminal_obligations.len() {
            return Err(AkitaError::InvalidSize {
                expected: self.terminal_obligations.len(),
                actual: terminal_claims.len(),
            });
        }
        let expected = self.terminal_obligations.iter().zip(terminal_claims).fold(
            E::zero(),
            |sum, (obligation, terminal)| {
                sum + obligation.batching_coefficient
                    * obligation.prefix_equality_scalar
                    * *terminal
            },
        );
        if self.final_scaled_claim != self.final_claim_scale * expected {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}

/// Replay a master-lifted eq-factored executor batch proof.
///
/// The checked plan lifts every shorter suffix relation by its missing equality
/// prefix. This verifier therefore advances one master-factor recurrence for
/// the existing compact proof and returns local terminal obligations that undo
/// no scalars and require no inversions.
pub fn verify_eq_factored_executor_batch_rounds<F, T, E, S>(
    plan: &CheckedEqFactoredBatch<E>,
    input_claims: &[E],
    proof: &EqFactoredSumcheckProof<E>,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<EqFactoredBatchRoundResult<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
{
    validate_claim_count(&plan.geometry, input_claims)?;
    plan.validate_proof(proof)?;

    let mut master_point = Vec::new();
    master_point
        .try_reserve_exact(plan.geometry.master_rounds)
        .map_err(|_| invalid("eq-factored verifier challenge allocation failed"))?;
    let coefficients =
        absorb_claims_and_sample_coefficients(input_claims, transcript, &mut sample_challenge)?;
    let mut scaled_claim = input_claims
        .iter()
        .zip(&coefficients)
        .fold(E::zero(), |sum, (&claim, &coefficient)| {
            sum + claim * coefficient
        });
    let mut claim_scale = E::one();
    let mut equality_scalar = E::one();

    for (master_round, polynomial) in proof.round_polys.iter().enumerate() {
        let tau = plan.equality_coordinate(master_round)?;
        let factor_at_one = equality_scalar * tau;
        let factor_at_zero = equality_scalar - factor_at_one;
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, polynomial);
        let challenge = sample_challenge(transcript);
        (scaled_claim, claim_scale) = advance_eq_factored_claim(
            scaled_claim,
            claim_scale,
            factor_at_zero,
            factor_at_one,
            polynomial,
            challenge,
        );
        equality_scalar *= tau * challenge + (E::one() - tau) * (E::one() - challenge);
        master_point.push(challenge);
    }

    let mut terminal_obligations = Vec::new();
    terminal_obligations
        .try_reserve_exact(plan.geometry.members.len())
        .map_err(|_| invalid("eq-factored terminal obligation allocation failed"))?;
    for member_index in 0..plan.geometry.members.len() {
        let group_index = plan.member_group_index(member_index)?;
        let group = plan
            .geometry
            .groups
            .get(group_index)
            .ok_or_else(|| invalid("sumcheck member group is missing"))?;
        let batching_coefficient = coefficients
            .get(member_index)
            .copied()
            .ok_or_else(|| invalid("sumcheck batching coefficient is missing"))?;
        terminal_obligations.push(EqFactoredTerminalObligation {
            member_index,
            suffix_offset: group.suffix_offset(),
            local_rounds: group.num_rounds(),
            batching_coefficient,
            prefix_equality_scalar: plan.group_prefix_scalar(&master_point, group_index)?,
        });
    }

    Ok(EqFactoredBatchRoundResult {
        master_point,
        batching_coefficients: coefficients,
        terminal_obligations,
        final_scaled_claim: scaled_claim,
        final_claim_scale: claim_scale,
    })
}
