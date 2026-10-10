//! Challenge-field sumchecks for the succinct LaBinius root.
//!
//! Round replay returns an opening obligation, not a complete root proof. The
//! caller supplies authenticated witness evaluations to the terminal formulas.
//! Public-table construction and statement binding belong to the enclosing
//! protocol; these routines accept their values explicitly.
//!
//! Claims, round messages and challenges are elements of the challenge field
//! `E`. The base field `F` only fixes their transcript encoding: canonical
//! base coordinates, as in `akita_sumcheck`.

use crate::channel::ClearChannel;
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::LABINIUS_BALANCED_LOG_BASIS;
use akita_sumcheck::{
    verify_sumcheck_rounds, SumcheckRoundResult, SumcheckShape, SumcheckVerifierChannel,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Stored digits of either committed table lie in `[0, DIGIT_ALPHABET)`.
const DIGIT_ALPHABET: u64 = 1 << LABINIUS_BALANCED_LOG_BASIS;

/// Public kind of a root sumcheck instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootSumcheckInstance {
    Combined,
    Product,
}

/// Absorb the canonical versioned instance identity before any round replay.
///
/// The header consists of the domain, kind byte (0 combined, 1 product),
/// invocation as four little-endian bytes and dimension as eight little-endian
/// bytes. Diagnostic site identifiers do not bind any of these values
/// themselves.
pub fn bind_root_sumcheck_instance<C: ClearChannel>(
    channel: &mut C,
    kind: RootSumcheckInstance,
    invocation: u32,
    num_vars: usize,
) -> Result<(), AkitaError> {
    let dimension = u64::try_from(num_vars)
        .map_err(|_| AkitaError::InvalidInput("root sumcheck dimension does not fit u64".into()))?;
    let tag = match kind {
        RootSumcheckInstance::Combined => 0u8,
        RootSumcheckInstance::Product => 1u8,
    };
    let mut header = Vec::new();
    header.try_reserve_exact(64).map_err(|_| {
        AkitaError::InvalidInput("root sumcheck instance header allocation failed".into())
    })?;
    header.extend_from_slice(b"akita/labinius/root-sumcheck-instance/v1");
    header.push(tag);
    header.extend_from_slice(&invocation.to_le_bytes());
    header.extend_from_slice(&dimension.to_le_bytes());
    channel.public(&header)
}

/// Evaluate `P(w) = product_{a=0}^{15} (w-a)`, which vanishes exactly on the
/// stored digit alphabet. The proof prime exceeds the alphabet size.
pub fn alphabet_polynomial<F: Field>(w: F) -> F {
    (0..DIGIT_ALPHABET).fold(F::one(), |product, digit| {
        product * (w - F::from_u64(digit))
    })
}

/// Checked public grammar for the combined alphabet-and-relation instance.
///
/// The round degree is the alphabet polynomial's degree plus one for the
/// equality factor.
pub fn combined_shape(num_vars: usize) -> Result<SumcheckShape, AkitaError> {
    checked::pow2(num_vars).ok_or_else(|| {
        AkitaError::InvalidInput("combined root table length overflows usize".into())
    })?;
    SumcheckShape::new(num_vars, DIGIT_ALPHABET as usize + 1)
        .map_err(|_| AkitaError::InvalidInput("invalid combined root sumcheck shape".into()))
}

/// Checked public grammar for a product instance over a field table.
pub fn product_shape(num_vars: usize) -> Result<SumcheckShape, AkitaError> {
    checked::pow2(num_vars).ok_or_else(|| {
        AkitaError::InvalidInput("product root table length overflows usize".into())
    })?;
    SumcheckShape::new(num_vars, 2)
        .map_err(|_| AkitaError::InvalidInput("invalid product root sumcheck shape".into()))
}

/// Claimed combined sum for the linear claim `s`: `c_pub - y_Y` on the
/// response table, `y_Y` on the image table.
pub fn combined_input_claim<F: Field>(beta: F, s: F) -> F {
    beta * s
}

/// Compute the combined terminal using externally authenticated `w_eval`.
///
/// Equality evaluation takes `O(tau.len())` field operations and constant
/// additional space. Both points use the little-endian table convention.
pub fn combined_terminal<F: Field>(
    tau: &[F],
    rho: &[F],
    beta: F,
    w_eval: F,
    kw_eval: F,
) -> Result<F, AkitaError> {
    if tau.len() != rho.len() {
        return Err(AkitaError::InvalidInput(
            "combined root equality point dimension mismatch".into(),
        ));
    }
    let equality = tau.iter().zip(rho).fold(F::one(), |product, (&t, &r)| {
        product * ((F::one() - t) * (F::one() - r) + t * r)
    });
    Ok(equality * alphabet_polynomial(w_eval) + beta * w_eval * kw_eval)
}

/// Compute the product terminal using an externally authenticated table
/// evaluation and the public weight evaluation at the same point.
pub fn product_terminal<F: Field>(table_eval: F, weight_eval: F) -> F {
    table_eval * weight_eval
}

/// Replay combined rounds; the caller must check the returned terminal claim.
pub fn verify_combined_rounds<'proof, F, E, C>(
    channel: &mut C,
    invocation: u32,
    num_vars: usize,
    beta: E,
    s: E,
) -> Result<SumcheckRoundResult<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: SumcheckVerifierChannel<'proof, E> + ClearChannel,
{
    let shape = combined_shape(num_vars)?;
    bind_root_sumcheck_instance(
        channel,
        RootSumcheckInstance::Combined,
        invocation,
        num_vars,
    )?;
    verify_sumcheck_rounds::<F, E, C>(channel, invocation, combined_input_claim(beta, s), shape)
}

/// Replay product rounds for the linear claim `claim = <table, weights>`; the
/// caller must check the returned terminal claim.
pub fn verify_product_rounds<'proof, F, E, C>(
    channel: &mut C,
    invocation: u32,
    num_vars: usize,
    claim: E,
) -> Result<SumcheckRoundResult<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: SumcheckVerifierChannel<'proof, E> + ClearChannel,
{
    let shape = product_shape(num_vars)?;
    bind_root_sumcheck_instance(channel, RootSumcheckInstance::Product, invocation, num_vars)?;
    verify_sumcheck_rounds::<F, E, C>(channel, invocation, claim, shape)
}
