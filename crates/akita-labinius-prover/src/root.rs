//! Root reduction from a binary opening to two committed-table evaluations.
//!
//! The oracle supplies the table commitment and evaluation protocol; this
//! module composes the admitted root, frontend and coefficient-field sumchecks.

mod transparent;

pub use transparent::TransparentRootProverOracle;

use akita_algebra::{binary::field_switch::SwitchField, SmoothFftField, TrinomialModulus};
use akita_challenges::BinaryChallengeSampler;
use akita_error::AkitaError;
use akita_labinius_verifier::{
    channel::{
        finish_prover, new_root_prover, RootChallengeChannel, RootFieldSite,
        RootSumcheckProverChannel,
    },
    codec::{exchange_binary, exchange_field},
    endpoint::{fold_integer, left_expansion, verify_left_expansion},
    frontend::prove_frontend,
    lowered::{image_weights_dense, witness_weights_dense, LoweredChallenges, LoweredPublic},
    root::{bind_root_statement, exchange_root_auxiliary, RootEvaluationClaims, RootProverOracle},
    AdmittedRootSetup, BinaryClearCommitment,
};
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::SumcheckProverChannel;
use jolt_field::ExtField;

use crate::{
    lowered::{a_relation_quotients, encode_witness, flatten_image, parity_quotient_and_carry},
    root_sumcheck::{
        prove_combined_rounds, prove_product_rounds, CombinedRootSumcheck, ProductSumcheck,
    },
};

/// Prove a root reduction on the caller's channel, returning its opening claims.
///
/// The image oracle MUST bind the image table corresponding to `commitment`.
/// Response coefficients outside the admitted interval return an error without
/// retrying challenges. The last call discharges the returned table evaluations.
#[allow(clippy::too_many_arguments)]
pub fn prove_root_reduction<H, F, const D: usize, M, O, S>(
    admitted: &AdmittedRootSetup<F, D, M>,
    base: LabiniusDigitBase,
    source: &[H::Source],
    commitment: &BinaryClearCommitment<F, D, M>,
    point: &[H],
    value: H,
    oracle: &mut O,
    channel: &mut S,
) -> Result<RootEvaluationClaims<F>, AkitaError>
where
    H: SwitchField,
    F: SmoothFftField + ExtField<F>,
    M: TrinomialModulus,
    O: RootProverOracle<F>,
    S: RootChallengeChannel<F> + SumcheckProverChannel<F>,
{
    let setup = admitted.setup();
    if source.len() != setup.source_len() {
        return Err(AkitaError::InvalidSize {
            expected: setup.source_len(),
            actual: source.len(),
        });
    }
    let layout = bind_root_statement(admitted, base, point, value, channel, |layout, channel| {
        oracle.bind_image(layout, channel)
    })?;
    let binary = prove_frontend::<H, S>(source, point, value, channel)?;
    let mut u = left_expansion::<H>(source, &binary.point, setup.scalar_rows(), setup.columns())?;
    for element in &mut u {
        exchange_binary(channel, element)?;
    }
    verify_left_expansion(setup, &binary, &u)?;
    let mut sampler = BinaryChallengeSampler::new(setup.profile().clone());
    let fold = channel.fold_challenges(
        &mut sampler,
        b"akita/labinius/root-fold/v1",
        setup.columns(),
    )?;
    let response = fold_integer::<H>(
        source,
        setup.scalar_rows(),
        setup.columns(),
        &fold,
        setup.profile(),
    )?;
    if response
        .iter()
        .flatten()
        .any(|&coefficient| coefficient < setup.lower() || coefficient > setup.upper())
    {
        return Err(AkitaError::InvalidInput(
            "root response leaves the admitted interval".into(),
        ));
    }
    let digits = encode_witness(&layout, &response)?;
    oracle.commit_response(&layout, &digits, channel)?;
    let mut qa = a_relation_quotients(setup, commitment, &fold, &response)?;
    let (mut q, mut k) = parity_quotient_and_carry(setup, &binary, &u, &fold, &response)?;
    exchange_root_auxiliary(&layout, channel, &mut qa, &mut q, &mut k)?;
    let challenges = LoweredChallenges {
        alpha: channel.field_challenge(RootFieldSite::Alpha)?,
        xi: channel.field_challenge(RootFieldSite::Xi)?,
        gamma: channel.field_challenge(RootFieldSite::Gamma)?,
    };
    let public = LoweredPublic::new(&layout, setup, &binary, &u, &fold, &qa, &q, &k, challenges)?;
    let image = flatten_image(&layout, commitment)?;
    let ky = image_weights_dense(&layout, &public)?;
    let mut y_y = image
        .iter()
        .zip(&ky)
        .fold(F::zero(), |sum, (&y, &weight)| sum + y * weight);
    exchange_field(channel, &mut y_y)?;
    let s = public.c_pub() - y_y;
    let mut tau = Vec::new();
    tau.try_reserve_exact(layout.witness_log_len())
        .map_err(|_| AkitaError::InvalidInput("root equality-point allocation failed".into()))?;
    for coordinate in 0..layout.witness_log_len() {
        let site = u32::try_from(coordinate).map_err(|_| AkitaError::InvalidProof)?;
        tau.push(channel.field_challenge(RootFieldSite::Tau(site))?);
    }
    let beta = channel.field_challenge(RootFieldSite::Beta)?;
    let kw = witness_weights_dense(&layout, &public)?;
    let mut combined = CombinedRootSumcheck::new(base, &digits, kw, &tau, beta, s)?;
    let (response_point, _) = prove_combined_rounds(&mut combined, channel, 0)?;
    let (mut response_value, _) = combined
        .final_evaluations()
        .ok_or(AkitaError::InvalidProof)?;
    exchange_field(channel, &mut response_value)?;
    let mut product = ProductSumcheck::new(image, ky, y_y)?;
    let (image_point, _) = prove_product_rounds(&mut product, channel, 1)?;
    let (mut image_value, _) = product
        .final_evaluations()
        .ok_or(AkitaError::InvalidProof)?;
    exchange_field(channel, &mut image_value)?;
    let claims = RootEvaluationClaims {
        response_point,
        response_value,
        image_point,
        image_value,
    };
    oracle.discharge(&claims, channel)?;
    Ok(claims)
}

/// Create and finish a root session, returning proof bytes and opening claims.
#[allow(clippy::too_many_arguments)]
pub fn prove_root_reduction_bytes<H, F, const D: usize, M, O>(
    admitted: &AdmittedRootSetup<F, D, M>,
    base: LabiniusDigitBase,
    source: &[H::Source],
    commitment: &BinaryClearCommitment<F, D, M>,
    point: &[H],
    value: H,
    oracle: &mut O,
) -> Result<(Vec<u8>, RootEvaluationClaims<F>), AkitaError>
where
    H: SwitchField,
    F: SmoothFftField + ExtField<F>,
    M: TrinomialModulus,
    O: RootProverOracle<F>,
{
    let mut state = new_root_prover()?;
    let mut channel = RootSumcheckProverChannel::new(&mut state);
    let claims = prove_root_reduction(
        admitted,
        base,
        source,
        commitment,
        point,
        value,
        oracle,
        &mut channel,
    )?;
    Ok((finish_prover(state), claims))
}
