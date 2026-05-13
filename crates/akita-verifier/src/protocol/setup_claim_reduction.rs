//! Setup-side claim-reduction sumcheck wiring.
//!
//! The fourth-root verifier optimization splits the stage-2 closing
//! `M`-table evaluation `m(r_x)` into an algebraic part `m_alg(r_x)` that the
//! verifier can compute cheaply and a setup-dependent part `m_setup(r_x)`
//! that is reduced to a single point claim on the shared setup polynomial
//! `S` via this sumcheck.
//!
//! The sumcheck instance is
//!
//! ```text
//! m_setup(r_x) = sum_{i,j,k} w_setup(i, j, k; r_x) * S(i, j, k)
//! ```
//!
//! where the summation index decomposes as `row | col | coeff` (the bit layout
//! matches [`PreparedMEval::setup_weight_table_at_point`] and
//! [`SetupMatrixPolynomialView::materialize_table`]).

use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_sumcheck::{verify_sumcheck_rounds_only, SumcheckInstanceVerifier, SumcheckProof};
use akita_transcript::labels::{CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND, CHALLENGE_SUMCHECK_ROUND};
use akita_transcript::Transcript;
use akita_types::{AkitaExpandedSetup, SetupClaimReductionPayload};

use crate::stages::AkitaStage2Verifier;
use crate::PreparedMEval;

/// Materialize the setup weights and setup polynomial table used by the
/// claim-reduction sumcheck. Reference / debug helper: production callers
/// should prefer the structured evaluator
/// [`PreparedMEval::eval_setup_weight_at_point`] for the weight side and
/// [`SetupMatrixPolynomialView::mle`] for the setup side.
///
/// Both vectors have length `2^(row_bits + col_bits + coeff_bits)` and share
/// the same `row | col | coeff` index layout. Their inner product equals the
/// setup contribution `prepared.eval_split_at_point(...).setup`.
///
/// # Errors
///
/// Returns an error if `alpha` does not match this prepared M-eval, or if the
/// padded setup polynomial dimensions disagree with the weight table.
pub fn materialize_setup_claim_tables<F, const D: usize>(
    prepared: &PreparedMEval<F>,
    x_challenges: &[F],
    setup: &AkitaExpandedSetup<F>,
    alpha: F,
) -> Result<(Vec<F>, Vec<F>), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let setup_weights = prepared.setup_weight_table_at_point::<D>(x_challenges, setup, alpha)?;
    let row_count = prepared.setup_polynomial_row_count();
    let col_count = setup.seed.max_stride.max(1);
    let setup_view = setup
        .shared_matrix
        .setup_polynomial_view::<D>(row_count, col_count);
    let setup_table = setup_view.materialize_table();
    if setup_table.len() != setup_weights.len() {
        return Err(AkitaError::InvalidSize {
            expected: setup_weights.len(),
            actual: setup_table.len(),
        });
    }
    Ok((setup_weights, setup_table))
}

/// Number of sumcheck rounds for the setup-side claim-reduction sumcheck.
///
/// Matches `log2(2^(row_bits + col_bits + coeff_bits))` derived from
/// [`PreparedMEval::setup_polynomial_padded_dims`].
fn setup_claim_reduction_rounds<F: FieldCore + CanonicalField>(
    prepared: &PreparedMEval<F>,
    max_stride: usize,
) -> usize {
    let (row_bits, col_bits, coeff_bits) = prepared.setup_polynomial_padded_dims(max_stride);
    row_bits + col_bits + coeff_bits
}

/// Verify the setup-side claim-reduction sumcheck and close the final point
/// equality on the prover-claimed `S(r_setup)`.
///
/// The function replays the sumcheck with `payload.m_setup_eval` as input
/// claim, then closes the protocol against the prover-claimed
/// `payload.s_opening_value`:
///
/// ```text
/// weight_at_point * payload.s_opening_value == final_running_claim
/// ```
///
/// Returns `(r_setup, s_opening_value)`: the sumcheck-bound point and the
/// prover-claimed `S(r_setup)`. Phase D-full v2 (per book §5.3 lines
/// 627-660) routes this pair forward as a deferred claim discharged by
/// the next fold level's recursive open of `S` as a second handle in
/// the multi-claim batch (slices E-F of `specs/phase-d-full-handoff.md`).
/// Until slice F lands, the transitional per-level `mle` check below
/// anchors soundness.
///
/// **Phase D-full v2 seam**: when slice F wires the S-routing through,
/// the per-level `mle` check should be dropped at intermediate levels
/// (where a recursive next fold can discharge the deferred claim). At
/// terminal levels with `lp.use_setup_claim_reduction = true` the `mle`
/// check stays — there is no next level to discharge the claim.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if the sumcheck rejects, the
/// rounds disagree with the expected layout, the closing oracle equality
/// fails, or the transitional `mle` equality fails.
pub fn verify_setup_claim_reduction<F, T, const D: usize>(
    prepared: &PreparedMEval<F>,
    setup: &AkitaExpandedSetup<F>,
    x_challenges: &[F],
    alpha: F,
    payload: &SetupClaimReductionPayload<F>,
    transcript: &mut T,
) -> Result<(Vec<F>, F), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let max_stride = setup.seed.max_stride.max(1);
    let num_rounds = setup_claim_reduction_rounds(prepared, max_stride);

    let (challenges, final_running_claim) = verify_sumcheck_rounds_only::<F, T, F, _>(
        &payload.sumcheck,
        num_rounds,
        2,
        payload.m_setup_eval,
        transcript,
        |tr| tr.challenge_scalar(CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND),
    )?;

    let weight_at_point =
        prepared.eval_setup_weight_at_point::<D>(x_challenges, setup, alpha, &challenges)?;

    if weight_at_point * payload.s_opening_value != final_running_claim {
        return Err(AkitaError::InvalidProof);
    }

    // Transitional consistency check: prover's `s_opening_value` matches
    // the cleartext mle of `S` at `r_setup`. This is what today's
    // verifier hot path computes; Phase D-full v2's slice F replaces it
    // with a multi-claim recursive opening of `S` as a second handle in
    // the next level's batch (book §5.3 lines 627-660).
    let (row_bits, col_bits, _coeff_bits) = prepared.setup_polynomial_padded_dims(max_stride);
    let row_challenges = &challenges[..row_bits];
    let col_challenges = &challenges[row_bits..row_bits + col_bits];
    let coeff_challenges = &challenges[row_bits + col_bits..];
    let row_count = prepared.setup_polynomial_row_count();
    let setup_view = setup
        .shared_matrix
        .setup_polynomial_view::<D>(row_count, max_stride);
    let setup_at_point = setup_view.mle(row_challenges, col_challenges, coeff_challenges)?;
    if setup_at_point != payload.s_opening_value {
        return Err(AkitaError::InvalidProof);
    }

    Ok((challenges, payload.s_opening_value))
}

/// Verify the stage-2 main sumcheck together with the setup-side claim
/// reduction.
///
/// This is the verifier dual of the prover-side stage-2 path with claim
/// reduction enabled: the main sumcheck is replayed round-by-round without
/// the closing oracle equality, the setup-dependent residual `m_setup(r_x)`
/// is read from the payload, the closing equality is checked using the
/// algebraic part plus that residual, and the residual itself is then
/// validated by the claim-reduction sumcheck.
///
/// Returns `(stage2_challenges, r_setup, s_opening_value)`: the stage-2
/// sumcheck-sampled point (= the recursive opening point for the next
/// level's `w`), the claim-reduction sumcheck-sampled point, and the
/// prover-claimed `S(r_setup)`. Callers route the latter two as a joint
/// recursive opening claim on `S`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if any of the per-round messages, the
/// closing equality, or the claim-reduction sumcheck rejects.
pub fn verify_stage2_with_setup_claim_reduction<F, T, const D: usize>(
    main_sumcheck: &SumcheckProof<F>,
    payload: &SetupClaimReductionPayload<F>,
    stage2_verifier: &AkitaStage2Verifier<'_, F, D>,
    transcript: &mut T,
) -> Result<(Vec<F>, Vec<F>, F), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    T: Transcript<F>,
{
    let num_rounds = stage2_verifier.num_rounds();
    let degree_bound = stage2_verifier.degree_bound();
    let input_claim = stage2_verifier.input_claim();

    let (challenges, final_running_claim) = verify_sumcheck_rounds_only::<F, T, F, _>(
        main_sumcheck,
        num_rounds,
        degree_bound,
        input_claim,
        transcript,
        |tr| tr.challenge_scalar(CHALLENGE_SUMCHECK_ROUND),
    )?;

    let expected_main =
        stage2_verifier.expected_output_claim_with_m_setup(&challenges, payload.m_setup_eval)?;
    if expected_main != final_running_claim {
        return Err(AkitaError::InvalidProof);
    }

    let x_challenges = &challenges[stage2_verifier.ring_bits()..];
    let (r_setup, s_opening_value) = verify_setup_claim_reduction::<F, T, D>(
        stage2_verifier.prepared_m_eval(),
        stage2_verifier.setup(),
        x_challenges,
        stage2_verifier.alpha(),
        payload,
        transcript,
    )?;

    Ok((challenges, r_setup, s_opening_value))
}
