//! Coefficient-field sumchecks for the succinct LaBinius root.
//!
//! Round replay returns an opening obligation, not a complete root proof. The
//! caller supplies authenticated witness evaluations to the terminal formulas.
//! Public-table construction and statement binding belong to the enclosing
//! protocol; these routines accept their values explicitly.

use crate::channel::ClearChannel;
use akita_algebra::fft::SmoothFftField;
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::{
    verify_sumcheck_rounds, SumcheckRoundResult, SumcheckShape, SumcheckVerifierChannel,
};
use jolt_field::ExtField;

/// Public kind and digit alphabet of a root sumcheck instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootSumcheckInstance {
    Combined(LabiniusDigitBase),
    Product,
}

/// Absorb the canonical versioned instance identity before any round replay.
///
/// The header consists of the domain, kind byte (0 combined, 1 product),
/// invocation as four little-endian bytes, dimension as eight little-endian
/// bytes, and a digit-base byte (1, 2, 4 for combined; 0 for product).
/// Diagnostic site identifiers do not bind any of these values themselves.
pub fn bind_root_sumcheck_instance<C: ClearChannel>(
    channel: &mut C,
    kind: RootSumcheckInstance,
    invocation: u32,
    num_vars: usize,
) -> Result<(), AkitaError> {
    let dimension = u64::try_from(num_vars)
        .map_err(|_| AkitaError::InvalidInput("root sumcheck dimension does not fit u64".into()))?;
    let (tag, base) = match kind {
        RootSumcheckInstance::Combined(base) => (0u8, base.bits() as u8),
        RootSumcheckInstance::Product => (1u8, 0),
    };
    let mut header = Vec::new();
    header.try_reserve_exact(64).map_err(|_| {
        AkitaError::InvalidInput("root sumcheck instance header allocation failed".into())
    })?;
    header.extend_from_slice(b"akita/labinius/root-sumcheck-instance/v1");
    header.push(tag);
    header.extend_from_slice(&invocation.to_le_bytes());
    header.extend_from_slice(&dimension.to_le_bytes());
    header.push(base);
    channel.public(&header)
}

/// Evaluate `P_b(w) = product_{a=0}^{2^b-1} (w-a)`.
///
/// The coefficient prime must exceed the alphabet size, as it does for the
/// supported LaBinius coefficient fields.
pub fn alphabet_polynomial<F: SmoothFftField>(base: LabiniusDigitBase, w: F) -> F {
    (0..(1u64 << base.bits())).fold(F::one(), |product, digit| {
        product * (w - F::from_u64(digit))
    })
}

/// Checked public grammar for the combined alphabet-and-relation instance.
pub fn combined_shape(
    num_vars: usize,
    base: LabiniusDigitBase,
) -> Result<SumcheckShape, AkitaError> {
    checked::pow2(num_vars).ok_or_else(|| {
        AkitaError::InvalidInput("combined root table length overflows usize".into())
    })?;
    let degree = match base {
        LabiniusDigitBase::Bits1 => 3,
        LabiniusDigitBase::Bits2 => 5,
        LabiniusDigitBase::Bits4 => 17,
    };
    SumcheckShape::new(num_vars, degree)
        .map_err(|_| AkitaError::InvalidInput("invalid combined root sumcheck shape".into()))
}

/// Checked public grammar for the image-table product instance.
pub fn product_shape(num_vars: usize) -> Result<SumcheckShape, AkitaError> {
    checked::pow2(num_vars).ok_or_else(|| {
        AkitaError::InvalidInput("product root table length overflows usize".into())
    })?;
    SumcheckShape::new(num_vars, 2)
        .map_err(|_| AkitaError::InvalidInput("invalid product root sumcheck shape".into()))
}

/// Claimed combined sum after setting `s = c_pub - y_Y`.
pub fn combined_input_claim<F: SmoothFftField>(beta: F, s: F) -> F {
    beta * s
}

/// Compute the combined terminal using externally authenticated `w_eval`.
///
/// Equality evaluation takes `O(tau.len())` field operations and constant
/// additional space. Both points use the little-endian table convention.
pub fn combined_terminal<F: SmoothFftField>(
    base: LabiniusDigitBase,
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
    Ok(equality * alphabet_polynomial(base, w_eval) + beta * w_eval * kw_eval)
}

/// Compute the product terminal using externally authenticated `y_eval`.
pub fn product_terminal<F: SmoothFftField>(y_eval: F, ky_eval: F) -> F {
    y_eval * ky_eval
}

/// Replay combined rounds; the caller must check the returned terminal claim.
pub fn verify_combined_rounds<'proof, F, C>(
    channel: &mut C,
    invocation: u32,
    num_vars: usize,
    base: LabiniusDigitBase,
    beta: F,
    s: F,
) -> Result<SumcheckRoundResult<F>, AkitaError>
where
    F: SmoothFftField + ExtField<F>,
    C: SumcheckVerifierChannel<'proof, F> + ClearChannel,
{
    let shape = combined_shape(num_vars, base)?;
    bind_root_sumcheck_instance(
        channel,
        RootSumcheckInstance::Combined(base),
        invocation,
        num_vars,
    )?;
    verify_sumcheck_rounds::<F, F, C>(channel, invocation, combined_input_claim(beta, s), shape)
}

/// Replay product rounds; the caller must check the returned terminal claim.
pub fn verify_product_rounds<'proof, F, C>(
    channel: &mut C,
    invocation: u32,
    num_vars: usize,
    y_y: F,
) -> Result<SumcheckRoundResult<F>, AkitaError>
where
    F: SmoothFftField + ExtField<F>,
    C: SumcheckVerifierChannel<'proof, F> + ClearChannel,
{
    let shape = product_shape(num_vars)?;
    bind_root_sumcheck_instance(channel, RootSumcheckInstance::Product, invocation, num_vars)?;
    verify_sumcheck_rounds::<F, F, C>(channel, invocation, y_y, shape)
}
