use super::{
    bind_root_statement, exchange_root_auxiliary, zero_vec, RootEvaluationClaims,
    RootVerifierOracle,
};
use crate::{
    channel::{self, RootChallengeChannel, RootFieldSite, RootSumcheckVerifierChannel},
    codec::{exchange_binary, exchange_field},
    endpoint::verify_left_expansion,
    frontend::verify_frontend,
    lowered::{image_weight_mle, witness_weight_mle, LoweredChallenges, LoweredPublic},
    root_sumcheck::{
        combined_terminal, product_terminal, verify_combined_rounds, verify_product_rounds,
    },
    AdmittedRootSetup,
};
use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField162 as B},
    SmoothFftField, TrinomialModulus,
};
use akita_challenges::BinaryChallengeSampler;
use akita_error::AkitaError;
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::SumcheckVerifierChannel;
use jolt_field::ExtField;

/// Replay one root reduction and discharge both committed-table evaluations.
/// Oracle errors reject; the channel is left open for a surrounding protocol.
#[allow(clippy::too_many_arguments)]
pub fn verify_root_reduction<'proof, H, F, const D: usize, M, O, S>(
    admitted: &AdmittedRootSetup<F, D, M>,
    base: LabiniusDigitBase,
    point: &[H],
    value: H,
    oracle: &mut O,
    channel: &mut S,
) -> Result<RootEvaluationClaims<F>, AkitaError>
where
    H: SwitchField,
    F: SmoothFftField + ExtField<F>,
    M: TrinomialModulus,
    O: RootVerifierOracle<F>,
    S: RootChallengeChannel<F> + SumcheckVerifierChannel<'proof, F>,
{
    let layout = bind_root_statement(admitted, base, point, value, channel, |layout, channel| {
        oracle.bind_image(layout, channel)
    })?;
    let setup = admitted.setup();
    let binary = verify_frontend(point, value, channel)?;
    let mut u = Vec::new();
    u.try_reserve_exact(setup.columns())
        .map_err(|_| AkitaError::InvalidProof)?;
    for _ in 0..setup.columns() {
        let mut element = B::ZERO;
        exchange_binary(channel, &mut element)?;
        u.push(element);
    }
    verify_left_expansion(setup, &binary, &u)?;
    let mut sampler = BinaryChallengeSampler::new(setup.profile().clone());
    let fold = channel.fold_challenges(
        &mut sampler,
        b"akita/labinius/root-fold/v1",
        setup.columns(),
    )?;
    oracle.bind_response(&layout, channel)?;
    let mut ka = zero_vec(layout.encoding().a_carry_len())?;
    let mut q = zero_vec(layout.encoding().parity_quotient_len())?;
    let mut k = zero_vec(layout.encoding().parity_carry_len())?;
    exchange_root_auxiliary(&layout, channel, &mut ka, &mut q, &mut k)?;
    let challenges = LoweredChallenges {
        alpha: channel.field_challenge(RootFieldSite::Alpha)?,
        xi: channel.field_challenge(RootFieldSite::Xi)?,
        gamma: channel.field_challenge(RootFieldSite::Gamma)?,
    };
    let public = LoweredPublic::new(&layout, setup, &binary, &u, &fold, &ka, &q, &k, challenges)?;
    let mut y_y = F::zero();
    exchange_field(channel, &mut y_y)?;
    let s = public.c_pub() - y_y;
    let mut tau = Vec::new();
    tau.try_reserve_exact(layout.witness_log_len())
        .map_err(|_| AkitaError::InvalidProof)?;
    for index in 0..layout.witness_log_len() {
        tau.push(channel.field_challenge(RootFieldSite::Tau(
            u32::try_from(index).map_err(|_| AkitaError::InvalidProof)?,
        ))?);
    }
    let beta = channel.field_challenge(RootFieldSite::Beta)?;
    let combined = verify_combined_rounds(channel, 0, layout.witness_log_len(), base, beta, s)?;
    let mut w_eval = F::zero();
    exchange_field(channel, &mut w_eval)?;
    let kw = witness_weight_mle(&layout, &public, setup, &combined.challenges)?;
    if combined.output_claim
        != combined_terminal(base, &tau, &combined.challenges, beta, w_eval, kw)?
    {
        return Err(AkitaError::InvalidProof);
    }
    let product = verify_product_rounds(channel, 1, layout.image_log_len(), y_y)?;
    let mut y_eval = F::zero();
    exchange_field(channel, &mut y_eval)?;
    let ky = image_weight_mle(&layout, &public, &product.challenges)?;
    if product.output_claim != product_terminal(y_eval, ky) {
        return Err(AkitaError::InvalidProof);
    }
    let claims = RootEvaluationClaims {
        response_point: combined.challenges,
        response_value: w_eval,
        image_point: product.challenges,
        image_value: y_eval,
    };
    oracle.discharge(&claims, channel)?;
    Ok(claims)
}

/// Verify an entire standalone root proof, rejecting truncation and trailing bytes.
pub fn verify_root_reduction_bytes<H, F, const D: usize, M, O>(
    admitted: &AdmittedRootSetup<F, D, M>,
    base: LabiniusDigitBase,
    point: &[H],
    value: H,
    oracle: &mut O,
    proof: &[u8],
) -> Result<RootEvaluationClaims<F>, AkitaError>
where
    H: SwitchField,
    F: SmoothFftField + ExtField<F>,
    M: TrinomialModulus,
    O: RootVerifierOracle<F>,
{
    let mut state = channel::new_root_verifier(proof)?;
    let claims = verify_root_reduction(
        admitted,
        base,
        point,
        value,
        oracle,
        &mut RootSumcheckVerifierChannel::new(&mut state),
    )?;
    channel::finish_verifier(state)?;
    Ok(claims)
}
