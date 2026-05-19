//! Setup-side claim-reduction sumcheck wiring.
//!
//! The fourth-root verifier optimization splits the stage-2 closing
//! `M`-table evaluation `m(r_x)` into an algebraic part `m_alg(r_x)` that the
//! verifier can compute cheaply and a setup-dependent part `m_setup(r_x)`
//! that is reduced to a single point claim on the shared setup polynomial
//! `S` via this sumcheck.
//!
//! The sumcheck instance fixes the main stage-2 column point `r_x` and is
//!
//! ```text
//! lambda * m_setup(r_x) =
//!     sum_{i,k} eq(tau_1, i) * lambda * alpha^k * S(i, r_x, k)
//! ```
//!
//! where `S(i, r_x, k)` is the shared setup matrix contribution with the
//! setup-column/M-table dimension evaluated structurally at the already-sampled
//! stage-2 point `r_x`.

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_sumcheck::multilinear_eval;
use akita_sumcheck::{verify_sumcheck_rounds_only, SumcheckInstanceVerifier, SumcheckProof};
use akita_transcript::labels::{CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND, CHALLENGE_SUMCHECK_ROUND};
use akita_transcript::Transcript;
use akita_types::{AkitaExpandedSetup, SetupClaimReductionPayload};

use crate::stages::AkitaStage2Verifier;
use crate::PreparedMEval;

/// Materialize the setup weights and setup polynomial table used by the
/// prover-side claim-reduction sumcheck.
///
/// Both vectors have length `2^(row_bits + coeff_bits)` and share the same
/// `row | coeff` index layout. The setup table is the shared setup matrix
/// contribution with the main stage-2 column point `r_x` fixed; their inner
/// product equals the scaled setup contribution
/// `claim_scale * prepared.eval_split_at_point(...).setup`.
///
/// The prover currently materializes them via `WeightedTableProver` (see
/// `akita-prover/src/protocol/setup_claim_reduction.rs`).
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
    claim_scale: F,
) -> Result<(Vec<F>, Vec<F>), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let setup_weights = prepared.setup_claim_weight_table::<D>(alpha, claim_scale)?;
    let setup_table = flatten_setup_claim_polynomial(
        prepared,
        &materialize_setup_claim_polynomial::<F, D>(prepared, x_challenges, setup)?,
    );
    if setup_table.len() != setup_weights.len() {
        return Err(AkitaError::InvalidSize {
            expected: setup_weights.len(),
            actual: setup_table.len(),
        });
    }
    Ok((setup_weights, setup_table))
}

/// Materialize the setup polynomial used by the book-shaped reducer.
///
/// The returned rings are indexed by logical M-row family; each ring contains
/// the coefficient vector of `S(i, r_x, k)` with `r_x` fixed.
///
/// # Errors
///
/// Returns an error if the setup matrix cannot be viewed at this ring
/// dimension or if `x_challenges` is malformed for the prepared M-table.
pub fn materialize_setup_claim_polynomial<F, const D: usize>(
    prepared: &PreparedMEval<F>,
    x_challenges: &[F],
    setup: &AkitaExpandedSetup<F>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    prepared.materialize_setup_claim_polynomial_at_point::<D>(x_challenges, setup)
}

/// Flatten a row-indexed ring polynomial into `row | coeff` Boolean-table
/// order, padding the row dimension to the next power of two.
pub fn flatten_setup_claim_polynomial<F: FieldCore + CanonicalField, const D: usize>(
    prepared: &PreparedMEval<F>,
    rings: &[CyclotomicRing<F, D>],
) -> Vec<F> {
    let (row_bits, _coeff_bits) = prepared.setup_claim_reduction_dims();
    let row_pow = 1usize << row_bits;
    let mut table = vec![F::zero(); row_pow * D];
    for (row, ring) in rings.iter().enumerate().take(row_pow) {
        for coeff in 0..D {
            table[row | (coeff << row_bits)] = ring.coeffs[coeff];
        }
    }
    table
}

/// Number of sumcheck rounds for the setup-side claim-reduction sumcheck.
///
/// Matches book §5.4 lines 615-621: row-family variables plus ring
/// coefficient variables, with the stage-2 column point already fixed.
fn setup_claim_reduction_rounds<F: FieldCore + CanonicalField>(
    prepared: &PreparedMEval<F>,
    _max_stride: usize,
) -> usize {
    let (row_bits, coeff_bits) = prepared.setup_claim_reduction_dims();
    row_bits + coeff_bits
}

/// Verify the setup-side claim-reduction sumcheck and close the final point
/// equality on the prover-claimed `S(r_i, r_x, r_k)`.
///
/// The function replays the sumcheck with the scaled
/// `payload.m_setup_eval = lambda * m_setup(r_x)` as input claim, then closes
/// the protocol against the prover-claimed
/// `payload.s_opening_value`:
///
/// ```text
/// (lambda * weight_at_point) * payload.s_opening_value == final_running_claim
/// ```
///
/// Returns `(r_setup, s_opening_value)`: the sumcheck-bound row/coeff point
/// and the prover-claimed `S(r_i, r_x, r_k)`. Recursive routes carry the
/// derived, `r_x`-fixed setup polynomial forward as a deferred claim.
///
/// `routes_recursively` controls the closing-oracle check on the
/// prover-claimed `s_opening_value`:
///
///  - `routes_recursively == false`: the cleartext mle of the `r_x`-fixed
///    setup polynomial is computed and compared against `s_opening_value`. This
///    is the historical "self-contained" verifier check used at
///    terminal levels and any level whose deferred claim is not picked
///    up by a recursive next fold.
///  - `routes_recursively == true`: the cleartext mle check is dropped.
///    Soundness is anchored by the next fold level's joint multi-group
///    opening of `S` alongside the folded witness (see book §5.3
///    "split commitment" lines 643-660); the closing-oracle equality
///    `weight_at_point * s_opening_value == final_running_claim` still
///    runs and binds `s_opening_value` to the sumcheck-bound point.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if the sumcheck rejects, the
/// rounds disagree with the expected layout, the closing oracle equality
/// fails, or — when `routes_recursively == false` — the cleartext mle
/// check fails.
#[allow(clippy::too_many_arguments)]
pub fn verify_setup_claim_reduction<F, T, const D: usize>(
    prepared: &PreparedMEval<F>,
    setup: &AkitaExpandedSetup<F>,
    x_challenges: &[F],
    alpha: F,
    claim_scale: F,
    payload: &SetupClaimReductionPayload<F>,
    transcript: &mut T,
    routes_recursively: bool,
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
        prepared.eval_setup_claim_weight_at_point::<D>(alpha, claim_scale, &challenges)?;

    if weight_at_point * payload.s_opening_value != final_running_claim {
        return Err(AkitaError::InvalidProof);
    }

    if !routes_recursively {
        akita_field::op_counter::with_category(
            akita_field::op_counter::OpCategory::Setup,
            || -> Result<(), AkitaError> {
                let setup_table = flatten_setup_claim_polynomial(
                    prepared,
                    &materialize_setup_claim_polynomial::<F, D>(prepared, x_challenges, setup)?,
                );
                let setup_at_point = multilinear_eval(&setup_table, &challenges)?;
                if setup_at_point != payload.s_opening_value {
                    return Err(AkitaError::InvalidProof);
                }
                Ok(())
            },
        )?;
    }

    Ok((challenges, payload.s_opening_value))
}

/// Verify the stage-2 main sumcheck together with the setup-side claim
/// reduction.
///
/// This is the verifier dual of the prover-side stage-2 path with claim
/// reduction enabled: the main sumcheck is replayed round-by-round without
/// the closing oracle equality, the scaled setup-dependent residual
/// `lambda * m_setup(r_x)` is read from the payload, the closing equality is
/// checked using the algebraic part plus that scaled residual, and the
/// residual itself is then validated by the claim-reduction sumcheck.
///
/// `routes_recursively` is forwarded to
/// [`verify_setup_claim_reduction`]: pass `true` when the deferred
/// `S(r_i, r_x, r_k)` claim will be discharged by the next fold level's
/// recursive open (skipping the cleartext mle check), `false` when this
/// level is the last consumer of the claim (the cleartext mle check
/// stays).
///
/// Returns `(stage2_challenges, r_setup, s_opening_value)`: the stage-2
/// sumcheck-sampled point (= the recursive opening point for the next
/// level's `w`), the claim-reduction sumcheck-sampled row/coeff point, and
/// the prover-claimed `S(r_i, r_x, r_k)`. Callers route the latter as a joint
/// recursive opening claim on the `r_x`-fixed setup polynomial.
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
    routes_recursively: bool,
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
    let claim_scale = stage2_verifier.setup_claim_scale(&challenges)?;
    let (r_setup, s_opening_value) = verify_setup_claim_reduction::<F, T, D>(
        stage2_verifier.prepared_m_eval(),
        stage2_verifier.setup(),
        x_challenges,
        stage2_verifier.alpha(),
        claim_scale,
        payload,
        transcript,
        routes_recursively,
    )?;

    Ok((challenges, r_setup, s_opening_value))
}
