//! Prover-side setup-claim-reduction sumcheck construction.
//!
//! Reduces the setup-dependent contribution of the stage-2 `M`-table evaluation
//! to a single point opening on the shared setup polynomial `S`. The shared
//! algebraic state with the verifier is consumed via [`PreparedMEval`].

use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_sumcheck::{prove_sumcheck, SumcheckProof, WeightedTableProver};
use akita_transcript::labels::CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND;
use akita_transcript::Transcript;
use akita_types::AkitaExpandedSetup;
use akita_verifier::{materialize_setup_claim_tables, PreparedMEval};

/// Prover output of the setup-side claim-reduction sumcheck.
pub struct SetupClaimReductionProof<F: FieldCore> {
    /// Sumcheck proof on `m_setup(r_x) = sum_z w_setup(z; r_x) * S(z)`.
    pub proof: SumcheckProof<F>,
    /// The reduced setup-side claim that the proof witnesses.
    pub input_claim: F,
    /// Prover-claimed evaluation `S(r_setup)`. The closing-oracle
    /// equality of the claim-reduction sumcheck is
    /// `weight(r_x, r_setup) * s_opening_value == final_running_claim`,
    /// and the value itself is discharged by the next-level recursive
    /// opening of `S`.
    pub s_opening_value: F,
    /// Sampled sumcheck challenges, recorded for downstream batching.
    pub challenges: Vec<F>,
}

/// Build a setup-claim-reduction proof for the closing stage-2 setup
/// contribution `m_setup(r_x)`.
///
/// `x_challenges` is the column-side challenge slice from the main stage-2
/// sumcheck. The returned proof reduces the setup-dependent contribution at
/// `r_x` to a single point opening on `S`.
///
/// # Errors
///
/// Returns an error if the prepared M-eval state, the setup, or the sumcheck
/// driver disagree on shape or fail to construct the table.
pub fn prove_setup_claim_reduction<F, T, const D: usize>(
    prepared: &PreparedMEval<F>,
    setup: &AkitaExpandedSetup<F>,
    x_challenges: &[F],
    alpha: F,
    transcript: &mut T,
) -> Result<SetupClaimReductionProof<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let (setup_weights, setup_table) =
        materialize_setup_claim_tables::<F, D>(prepared, x_challenges, setup, alpha)?;
    let mut prover = WeightedTableProver::new(setup_table, setup_weights)?;
    let (proof, challenges, _final_running_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND)
        })?;
    let input_claim = {
        use akita_sumcheck::SumcheckInstanceProver;
        prover.input_claim()
    };

    let max_stride = setup.seed.max_stride.max(1);
    let (row_bits, col_bits, _coeff_bits) = prepared.setup_polynomial_padded_dims(max_stride);
    let row_count = prepared.setup_polynomial_row_count();
    let setup_view = setup
        .shared_matrix
        .setup_polynomial_view::<D>(row_count, max_stride);
    let row_challenges = &challenges[..row_bits];
    let col_challenges = &challenges[row_bits..row_bits + col_bits];
    let coeff_challenges = &challenges[row_bits + col_bits..];
    let s_opening_value = setup_view.mle(row_challenges, col_challenges, coeff_challenges)?;

    Ok(SetupClaimReductionProof {
        proof,
        input_claim,
        s_opening_value,
        challenges,
    })
}
