//! Prover-side setup-claim-reduction sumcheck construction.
//!
//! Reduces the setup-dependent contribution of the stage-2 `M`-table evaluation
//! to a single point opening on the shared setup polynomial `S`. The shared
//! algebraic state with the verifier is consumed via [`PreparedMEval`].

use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_sumcheck::{multilinear_eval, prove_sumcheck, SumcheckProof, WeightedTableProver};
use akita_transcript::labels::CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND;
use akita_transcript::Transcript;
use akita_types::AkitaExpandedSetup;
use akita_verifier::{materialize_setup_claim_tables, PreparedMEval};

/// Prover output of the setup-side claim-reduction sumcheck.
pub struct SetupClaimReductionProof<F: FieldCore> {
    /// Sumcheck proof on
    /// `lambda * m_setup(r_x) = sum_{i,k} eq(tau_1,i) lambda * alpha^k S(i,r_x,k)`.
    pub proof: SumcheckProof<F>,
    /// The scaled setup-side claim that the proof witnesses.
    pub input_claim: F,
    /// Prover-claimed evaluation `S(r_i, r_x, r_k)`. The closing-oracle
    /// equality of the claim-reduction sumcheck is
    /// `weight(r_i, r_k) * s_opening_value == final_running_claim`, and the
    /// value itself is discharged by the next-level recursive opening of the
    /// `r_x`-fixed setup polynomial when routed.
    pub s_opening_value: F,
    /// Sampled sumcheck challenges, recorded for downstream batching.
    pub challenges: Vec<F>,
}

/// Build a setup-claim-reduction proof for the scaled closing stage-2 setup
/// contribution `lambda * m_setup(r_x)`.
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
    claim_scale: F,
    transcript: &mut T,
) -> Result<SetupClaimReductionProof<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let (setup_weights, setup_table) =
        materialize_setup_claim_tables::<F, D>(prepared, x_challenges, setup, alpha, claim_scale)?;
    let mut prover = WeightedTableProver::new(setup_table.clone(), setup_weights)?;
    let (proof, challenges, _final_running_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, transcript, |tr| {
            tr.challenge_scalar(CHALLENGE_SETUP_CLAIM_REDUCTION_ROUND)
        })?;
    let input_claim = {
        use akita_sumcheck::SumcheckInstanceProver;
        prover.input_claim()
    };

    let s_opening_value = multilinear_eval(&setup_table, &challenges)?;

    Ok(SetupClaimReductionProof {
        proof,
        input_claim,
        s_opening_value,
        challenges,
    })
}
