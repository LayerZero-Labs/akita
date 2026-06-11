//! Verifier-side ring-switch replay.

use akita_algebra::ring::scalar_powers;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, MulBase, RandomSampling,
};
use akita_transcript::labels::{
    ABSORB_NEXT_LEVEL_WITNESS_BINDING, ABSORB_TERMINAL_W_REMAINDER, CHALLENGE_RING_SWITCH,
    CHALLENGE_TAU0, CHALLENGE_TAU1,
};
use akita_transcript::{sample_ext_challenge, Transcript};
use akita_types::{
    validate_opening_points_for_claims, FlatRingVec, LevelParams, MRowLayout, RingRelationInstance,
    RingRelationSegmentLayout, RingSubfieldEncoding, TerminalWitnessTranscriptParts,
};

use super::validate_ring_dispatch;
#[cfg(test)]
use akita_challenges::Challenges;
use row_eval::prepare_ring_switch_row_eval_inner;
#[cfg(test)]
pub(crate) use row_eval::summarize_pow2_block_carries_base;
#[cfg(test)]
use row_eval::summarize_pow2_multiplier_block_carries;
pub(crate) use tensor_challenges::PreparedChallengeEvals;

mod row_eval;
mod tensor_challenges;
#[cfg(test)]
mod tests;

/// Verifier-side ring-switch output, carrying only the data needed to replay
/// the fused stage-1/stage-2 checks.
pub(crate) struct RingSwitchVerifyOutput<E: FieldCore> {
    /// Prepared data for deferred ring-switch row MLE evaluation.
    pub prepared_row_eval: RingSwitchDeferredRowEval<E>,
    /// Evaluation table of alpha powers over the ring-coordinate dimension.
    pub alpha_evals_y: Vec<E>,
    /// Number of upper variable bits.
    pub col_bits: usize,
    /// Number of lower variable bits.
    pub ring_bits: usize,
    /// Challenge tau0 for the stage-1 sumcheck.
    pub tau0: Vec<E>,
    /// Challenge tau1 for the stage-2 M-row combination.
    pub tau1: Vec<E>,
    /// Basis size `b = 2^log_basis`.
    pub b: usize,
    /// Ring-switch challenge alpha.
    pub alpha: E,
}

struct RingSwitchVerifyCoreOutput<E: FieldCore> {
    prepared_row_eval: RingSwitchDeferredRowEval<E>,
    alpha_evals_y: Vec<E>,
    col_bits: usize,
    ring_bits: usize,
    tau0: Option<Vec<E>>,
    tau1: Vec<E>,
    b: usize,
    alpha: E,
}

impl<E: FieldCore> RingSwitchVerifyCoreOutput<E> {
    fn into_intermediate(self) -> Result<RingSwitchVerifyOutput<E>, AkitaError> {
        let tau0 = self.tau0.ok_or(AkitaError::InvalidProof)?;
        Ok(RingSwitchVerifyOutput {
            prepared_row_eval: self.prepared_row_eval,
            alpha_evals_y: self.alpha_evals_y,
            col_bits: self.col_bits,
            ring_bits: self.ring_bits,
            tau0,
            tau1: self.tau1,
            b: self.b,
            alpha: self.alpha,
        })
    }

    fn into_terminal_as_output(self) -> Result<RingSwitchVerifyOutput<E>, AkitaError> {
        if self.tau0.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(RingSwitchVerifyOutput {
            prepared_row_eval: self.prepared_row_eval,
            alpha_evals_y: self.alpha_evals_y,
            col_bits: self.col_bits,
            ring_bits: self.ring_bits,
            tau0: Vec::new(),
            tau1: self.tau1,
            b: self.b,
            alpha: self.alpha,
        })
    }
}

/// Precomputed challenge-derived data for deferred ring-switch row MLE evaluation.
///
/// Stores only data that cannot be derived from context at evaluation time:
/// alpha-evaluated folding challenges and the tau1 eq-polynomial expansion.
/// Everything else is passed by reference at evaluation time to avoid
/// duplicating setup matrix views, opening points, and gadget vectors.
#[derive(Clone)]
pub struct RingSwitchDeferredRowEval<F: FieldCore> {
    pub(crate) c_alphas: PreparedChallengeEvals<F>,
    pub(crate) eq_tau1: Vec<F>,
    pub(crate) num_t_vectors: usize,
    pub(crate) num_blocks: usize,
    pub(crate) num_claims: usize,
    pub(crate) depth_open: usize,
    pub(crate) depth_commit: usize,
    pub(crate) depth_fold: usize,
    #[cfg(feature = "zk")]
    pub(crate) d_blinding_segment_len: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_digit_planes_per_point: usize,
    #[cfg(feature = "zk")]
    pub(crate) b_blinding_segment_len: usize,
    pub(crate) block_len: usize,
    pub(crate) inner_width: usize,
    pub(crate) log_basis: u32,
    pub(crate) n_a: usize,
    pub(crate) n_d: usize,
    pub(crate) m_row_layout: MRowLayout,
    pub(crate) n_b: usize,
    /// Tiered split factor `f` (`1` = single-tier).
    pub(crate) tier_split: usize,
    /// Second-tier `F` rank (`0` = single-tier); the sent-commitment length.
    pub(crate) n_f: usize,
    pub(crate) num_points: usize,
    pub(crate) rows: usize,
    pub(crate) claim_to_commitment_group_poly: Vec<(usize, usize)>,
    pub(crate) num_polys_per_commitment_group: Vec<usize>,
    pub(crate) num_public_rows: usize,
    pub(crate) gamma: Vec<F>,
    pub(crate) claim_to_point: Vec<usize>,
    pub(crate) witness_segment_layout: RingRelationSegmentLayout,
}

pub(crate) type RingSwitchSegmentLayout = RingRelationSegmentLayout;

/// Fixed public relation inputs for verifier ring-switch replay.
pub struct RingSwitchReplay<'a, F: FieldCore, E, const D: usize> {
    pub relation: &'a RingRelationInstance<F, D>,
    pub row_coefficients: &'a [E],
    pub lp: &'a LevelParams,
}

/// Replay the verifier half of ring switching.
///
/// This handles multiple opening points, arbitrary claim-to-point mapping, and
/// point-local polynomial bundles. The recursive/single-point path is the
/// `opening_points = [pt]`, `claim_to_point = [0]`,
/// `num_polys_per_commitment_group = [1]`, `num_public_rows = 1` specialization.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or ring-switch row-eval
/// preparation fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier")]
#[inline(never)]
pub(crate) fn ring_switch_verifier<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    w_commitment: &FlatRingVec<F>,
    transcript: &mut T,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    // `validate_ring_dispatch` is called inside `ring_switch_verifier_core`;
    // the outer wrapper just performs the witness absorb before delegating.
    transcript.absorb_and_record_serde(ABSORB_NEXT_LEVEL_WITNESS_BINDING, w_commitment);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithDBlock)?
        .into_intermediate()
}

/// Terminal variant of [`ring_switch_verifier`].
///
/// This owns the required terminal final-witness remainder absorb before
/// sampling ring-switch challenges.
///
/// # Errors
///
/// Returns an error if the claim shape is invalid, opening-point routing is
/// inconsistent, transcript-bound challenge data has the wrong size, or
/// ring-switch row-eval preparation fails.
#[tracing::instrument(skip_all, name = "ring_switch_verifier_terminal")]
#[inline(never)]
pub(crate) fn ring_switch_verifier_terminal<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    transcript: &mut T,
    terminal_parts: &TerminalWitnessTranscriptParts,
) -> Result<RingSwitchVerifyOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_W_REMAINDER, &terminal_parts.remainder);
    ring_switch_verifier_core::<F, E, T, D>(replay, w_len, transcript, MRowLayout::WithoutDBlock)?
        .into_terminal_as_output()
}

#[tracing::instrument(skip_all, name = "ring_switch_verifier_core")]
#[inline(never)]
fn ring_switch_verifier_core<F, E, T, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    w_len: usize,
    transcript: &mut T,
    m_row_layout: MRowLayout,
) -> Result<RingSwitchVerifyCoreOutput<E>, AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt,
    T: Transcript<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let opening_points = relation.opening_points();
    let claim_to_point = relation.claim_to_point();
    let num_public_rows = relation.num_public_rows();

    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let num_claims = claim_to_point.len();
    validate_opening_points_for_claims(opening_points, claim_to_point, lp, num_claims)?;
    let num_points = relation
        .commitment_routing()
        .num_polys_per_commitment_group()
        .len();

    if w_len == 0 || !w_len.is_multiple_of(D) {
        return Err(AkitaError::InvalidProof);
    }
    let num_ring_elems = w_len / D;
    let col_bits = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch column count overflow".to_string()))?
        .trailing_zeros() as usize;
    let ring_bits = validate_ring_dispatch::<D>()?;
    let m_rows = lp.m_row_count_for(num_points, num_public_rows, m_row_layout)?;
    let num_sc_vars = col_bits + ring_bits;
    let num_i = m_rows
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch row count overflow".to_string()))?
        .trailing_zeros() as usize;

    let tau0 = match m_row_layout {
        MRowLayout::WithDBlock => Some(
            (0..num_sc_vars)
                .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
                .collect(),
        ),
        MRowLayout::WithoutDBlock => None,
    };
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    let alpha_evals_y = scalar_powers(alpha, D);
    let prepared_row_eval = prepare_ring_switch_row_eval::<F, E, D>(replay, alpha, &tau1)?;

    Ok(RingSwitchVerifyCoreOutput {
        prepared_row_eval,
        alpha_evals_y,
        col_bits,
        ring_bits,
        tau0,
        tau1,
        b: 1usize
            .checked_shl(lp.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("basis size overflow".to_string()))?,
        alpha,
    })
}

/// Prepare deferred verifier ring-switch row evaluation data from a fixed
/// [`RingRelationInstance`] and transcript-sampled row coefficients.
///
/// # Errors
///
/// Returns an error if gamma/challenge lengths do not match the claim shape,
/// the expanded tau1 table is too short for the level layout, or sparse
/// challenge evaluation fails.
#[tracing::instrument(skip_all, name = "prepare_ring_switch_row_eval")]
pub fn prepare_ring_switch_row_eval<F, E, const D: usize>(
    replay: &RingSwitchReplay<'_, F, E, D>,
    alpha: E,
    tau1: &[E],
) -> Result<RingSwitchDeferredRowEval<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F> + FromPrimitiveInt + MulBase<F>,
{
    let relation = replay.relation;
    let lp = replay.lp;
    let witness_segment_layout = relation.segment_layout(lp)?;
    let routing = relation.commitment_routing();
    prepare_ring_switch_row_eval_inner::<F, E, D>(
        &relation.challenges,
        alpha,
        lp,
        tau1,
        routing.num_polys_per_commitment_group(),
        routing.claim_to_commitment_group(),
        routing.claim_poly_in_commitment_group(),
        replay.row_coefficients,
        relation.num_public_rows(),
        relation.m_row_layout(),
        relation.opening_points().len(),
        relation.ring_multiplier_points(),
        relation.claim_to_point(),
        witness_segment_layout,
    )
}
