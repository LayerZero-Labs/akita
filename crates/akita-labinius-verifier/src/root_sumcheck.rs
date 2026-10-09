//! Coefficient-field sumchecks for the succinct LaBinius root.
//!
//! Round replay returns an opening obligation, not a complete root proof. The
//! caller supplies authenticated witness evaluations to the terminal formulas.
//! Public-table construction and statement binding belong to the enclosing
//! protocol; these routines accept their values explicitly.

use akita_algebra::fft::SmoothFftField;
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::{
    verify_sumcheck_rounds, SumcheckRoundResult, SumcheckShape, SumcheckVerifierChannel,
};
use jolt_field::ExtField;

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
    C: SumcheckVerifierChannel<'proof, F>,
{
    verify_sumcheck_rounds::<F, F, C>(
        channel,
        invocation,
        combined_input_claim(beta, s),
        combined_shape(num_vars, base)?,
    )
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
    C: SumcheckVerifierChannel<'proof, F>,
{
    verify_sumcheck_rounds::<F, F, C>(channel, invocation, y_y, product_shape(num_vars)?)
}
