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

use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_sumcheck::{multilinear_eval, verify_sumcheck_rounds_only, SumcheckProof};
use akita_transcript::labels::CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND;
use akita_transcript::Transcript;
use akita_types::AkitaExpandedSetup;

use crate::PreparedMEval;

/// Materialize the setup weights and setup polynomial table used by the
/// claim-reduction sumcheck.
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

/// Verify the setup-side claim-reduction sumcheck and close the final point
/// equality on the materialized setup polynomial.
///
/// `m_setup_claim` is the value the verifier reduces to: the setup-dependent
/// contribution to the closing M-table evaluation at the stage-2 challenge
/// point. It is typically derived from the main stage-2 running claim minus
/// the algebraic part `m_alg(r_x)`.
///
/// On success, returns the sampled sumcheck challenges so callers can record
/// them for downstream batching.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if the sumcheck rejects, the rounds
/// disagree with the expected layout, or the closing oracle equality
/// `S(r_setup) * w(r_setup) = final_running_claim` does not hold.
pub fn verify_setup_claim_reduction<F, T, const D: usize>(
    prepared: &PreparedMEval<F>,
    setup: &AkitaExpandedSetup<F>,
    x_challenges: &[F],
    alpha: F,
    proof: &SumcheckProof<F>,
    m_setup_claim: F,
    transcript: &mut T,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let (setup_weights, setup_table) =
        materialize_setup_claim_tables::<F, D>(prepared, x_challenges, setup, alpha)?;
    let num_rounds = setup_weights.len().trailing_zeros() as usize;

    let (challenges, final_running_claim) = verify_sumcheck_rounds_only::<F, T, F, _>(
        proof,
        num_rounds,
        2,
        m_setup_claim,
        transcript,
        |tr| tr.challenge_scalar(CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND),
    )?;

    let weight_at_point = multilinear_eval(&setup_weights, &challenges)?;
    let setup_at_point = multilinear_eval(&setup_table, &challenges)?;
    if weight_at_point * setup_at_point != final_running_claim {
        return Err(AkitaError::InvalidProof);
    }
    Ok(challenges)
}
