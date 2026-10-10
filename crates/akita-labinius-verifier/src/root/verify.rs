use super::{
    bind_root_statement, exchange_root_auxiliary, zero_vec, RootEvaluationClaims,
    RootVerifierOracle,
};
use crate::{
    channel::{self, RootChallengeChannel, RootSumcheckVerifierChannel},
    codec::{exchange_binary, exchange_extension},
    endpoint::verify_left_expansion,
    frontend::verify_frontend,
    grinding::RootGrindingSite,
    lowered::{
        image_weight_mle, prime_row_weight_mle, witness_weight_mle, LoweredChallenges,
        LoweredParity, LoweredPrime, LoweredPublic,
    },
    root_sumcheck::{
        combined_terminal, product_terminal, verify_combined_rounds, verify_product_rounds,
    },
    statement::RootStatement,
    AdmittedRootSetup,
};
use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField162 as B},
    TrinomialModulus,
};
use akita_challenges::BinaryChallengeSampler;
use akita_error::AkitaError;
use akita_sumcheck::SumcheckVerifierChannel;
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Replay one root reduction and discharge the committed-table evaluations.
/// Oracle errors reject; the channel is left open for a surrounding protocol.
///
/// The image and response tables hold stored base-16 digits, and each gets one
/// combined instance: the alphabet check on every entry plus its linear term
/// of the lowered relation. The prover's `y_Y` splits the relation's constant
/// between the two instances.
///
/// The statement fixes the opening mode. With a binary claim the frontend, the
/// binary left opening and the parity row run. With a prime claim the oracle
/// binds the prime left opening before the fold challenges, the prime row
/// joins the lowered relation with its own batching scalar, and one product
/// instance proves the value claim and the row's claim `y_P` on that table.
/// Nothing of a claim the statement lacks is read from the proof.
///
/// `F` is the base field of the committed tables; its characteristic is the
/// proof prime of every integer no-wrap condition. `E` is the challenge field:
/// all field challenges, sumcheck messages, evaluation points and evaluation
/// claims are elements of `E`, encoded as canonical base coordinates.
///
/// Every challenge in `E` is a site of the grinding plan bound with the
/// statement. The channel receives and checks the site's proof-of-work nonce
/// before the challenge; a missing, non-canonical, out-of-range or failing
/// nonce rejects.
pub fn verify_root_reduction<'proof, H, F, E, const D: usize, M, O, S>(
    admitted: &AdmittedRootSetup<D, M>,
    statement: &RootStatement<'_, H, E>,
    oracle: &mut O,
    channel: &mut S,
) -> Result<RootEvaluationClaims<E>, AkitaError>
where
    H: SwitchField,
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    M: TrinomialModulus,
    O: RootVerifierOracle<E>,
    S: RootChallengeChannel<E> + SumcheckVerifierChannel<'proof, E>,
{
    let layout = bind_root_statement::<H, F, E, D, M, S>(
        admitted,
        statement,
        channel,
        |layout, channel| oracle.bind_image(layout, channel),
    )?;
    let setup = admitted.setup();
    // The frontend's claim, the binary left opening, and storage for the
    // parity quotient and carry.
    let mut binary = match statement.binary {
        Some((point, value)) => {
            let claim = verify_frontend(point, value, channel)?;
            let mut u = Vec::new();
            u.try_reserve_exact(setup.columns())
                .map_err(|_| AkitaError::InvalidProof)?;
            for _ in 0..setup.columns() {
                let mut element = B::ZERO;
                exchange_binary(channel, &mut element)?;
                u.push(element);
            }
            verify_left_expansion(setup, &claim, &u)?;
            let quotient = zero_vec::<i128>(layout.encoding().parity_quotient_len())?;
            let carry = zero_vec::<i128>(layout.encoding().parity_carry_len())?;
            Some((claim, u, quotient, carry))
        }
        None => None,
    };
    if statement.prime.is_some() {
        oracle.bind_prime_opening(&layout, channel)?;
    }
    let mut sampler = BinaryChallengeSampler::new(setup.profile().clone());
    // The nonce only selects the challenges; the digit alphabet check on the
    // committed response is what binds the prover.
    let fold = channel.fold_challenges(
        &mut sampler,
        b"akita/labinius/root-fold/v1",
        setup.columns(),
        &mut 0,
    )?;
    oracle.bind_response(&layout, channel)?;
    let mut ka = zero_vec(layout.encoding().a_carry_len())?;
    exchange_root_auxiliary(
        &layout,
        channel,
        &mut ka,
        binary
            .as_mut()
            .map(|(_, _, quotient, carry)| (quotient.as_mut_slice(), carry.as_mut_slice())),
    )?;
    let alpha = channel.field_challenge(RootGrindingSite::Alpha)?;
    let parity = match &binary {
        Some((claim, u, quotient, carry)) => Some(LoweredParity {
            claim,
            u,
            quotient,
            carry,
            xi: channel.field_challenge(RootGrindingSite::Xi)?,
        }),
        None => None,
    };
    let gamma = channel.field_challenge(RootGrindingSite::Gamma)?;
    let prime_row = match statement.prime {
        Some(claim) => Some(LoweredPrime {
            ring_point: &claim.ring_point,
            eta: channel.field_challenge(RootGrindingSite::PrimeRow)?,
        }),
        None => None,
    };
    let public = LoweredPublic::new(
        &layout,
        setup,
        &fold,
        &ka,
        LoweredChallenges { alpha, gamma },
        parity,
        prime_row,
    )?;
    let mut y_y = E::zero();
    exchange_extension::<F, E, S>(channel, &mut y_y)?;
    let y_p = match statement.prime {
        Some(_) => {
            let mut y_p = E::zero();
            exchange_extension::<F, E, S>(channel, &mut y_p)?;
            Some(y_p)
        }
        None => None,
    };
    let s = public.response_claim(y_y, y_p)?;
    let tau = channel.field_point(
        RootGrindingSite::EqualityPoint { invocation: 0 },
        layout.witness_log_len(),
    )?;
    let beta = channel.field_challenge(RootGrindingSite::Batch { invocation: 0 })?;
    let combined =
        verify_combined_rounds::<F, E, S>(channel, 0, layout.witness_log_len(), beta, s)?;
    let mut w_eval = E::zero();
    exchange_extension::<F, E, S>(channel, &mut w_eval)?;
    let kw = witness_weight_mle(&layout, &public, setup, &combined.challenges)?;
    if combined.output_claim != combined_terminal(&tau, &combined.challenges, beta, w_eval, kw)? {
        return Err(AkitaError::InvalidProof);
    }
    let image_tau = channel.field_point(
        RootGrindingSite::EqualityPoint { invocation: 1 },
        layout.image_log_len(),
    )?;
    let image_beta = channel.field_challenge(RootGrindingSite::Batch { invocation: 1 })?;
    let image =
        verify_combined_rounds::<F, E, S>(channel, 1, layout.image_log_len(), image_beta, y_y)?;
    let mut y_eval = E::zero();
    exchange_extension::<F, E, S>(channel, &mut y_eval)?;
    let ky = image_weight_mle(&layout, &public, &image.challenges)?;
    if image.output_claim
        != combined_terminal(&image_tau, &image.challenges, image_beta, y_eval, ky)?
    {
        return Err(AkitaError::InvalidProof);
    }
    // Both linear claims on the prime left opening in one product instance:
    // the statement's value and the prime row's `y_P`.
    let prime = match (statement.prime, y_p) {
        (Some(claim), Some(y_p)) => {
            let theta = channel.field_challenge(RootGrindingSite::Batch { invocation: 2 })?;
            let product = verify_product_rounds::<F, E, S>(
                channel,
                2,
                layout.prime_log_len(),
                claim.value + theta * y_p,
            )?;
            let mut p_eval = E::zero();
            exchange_extension::<F, E, S>(channel, &mut p_eval)?;
            let weight = claim.value_weight_mle(&layout, &product.challenges)?
                + theta * prime_row_weight_mle(&layout, &public, &product.challenges)?;
            if product.output_claim != product_terminal(p_eval, weight) {
                return Err(AkitaError::InvalidProof);
            }
            Some((product.challenges, p_eval))
        }
        _ => None,
    };
    let claims = RootEvaluationClaims {
        response_point: combined.challenges,
        response_value: w_eval,
        image_point: image.challenges,
        image_value: y_eval,
        prime,
    };
    channel.finish_schedule()?;
    oracle.discharge(&claims, channel)?;
    Ok(claims)
}

/// Verify an entire standalone root proof, rejecting truncation and trailing bytes.
pub fn verify_root_reduction_bytes<H, F, E, const D: usize, M, O>(
    admitted: &AdmittedRootSetup<D, M>,
    statement: &RootStatement<'_, H, E>,
    oracle: &mut O,
    proof: &[u8],
) -> Result<RootEvaluationClaims<E>, AkitaError>
where
    H: SwitchField,
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    M: TrinomialModulus,
    O: RootVerifierOracle<E>,
{
    let mut state = channel::new_root_verifier(proof)?;
    let claims = verify_root_reduction::<H, F, E, D, M, O, _>(
        admitted,
        statement,
        oracle,
        &mut RootSumcheckVerifierChannel::<F>::new(&mut state),
    )?;
    channel::finish_verifier(state)?;
    Ok(claims)
}
