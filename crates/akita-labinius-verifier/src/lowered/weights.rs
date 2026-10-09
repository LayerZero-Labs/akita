use super::{signed_field, zero_vec, LoweredPublic, LoweredRootLayout};
use crate::BinaryClearSetup;
use akita_algebra::{fft::SmoothFftField, offset_eq::eq_eval_at_index, ring::TrinomialModulus};
use akita_error::AkitaError;

/// Dense reference response weights, with zero coefficient-tail weights.
pub fn witness_weights_dense<F: SmoothFftField>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
) -> Result<Vec<F>, AkitaError> {
    public.validate_layout(layout)?;
    let mut weights = zero_vec(layout.witness_len())?;
    for j in 0..layout.m() {
        for t in 0..layout.degree() {
            let coefficient = public.coefficient_weight(j, t)?;
            for (l, &digit) in public.digit_powers.iter().enumerate() {
                let address = layout.response_layout().address(j, l, t)?;
                *weights.get_mut(address).ok_or(AkitaError::InvalidProof)? = digit * coefficient;
            }
        }
    }
    Ok(weights)
}

/// Dense reference image weights, with zero coefficient and entry padding.
pub fn image_weights_dense<F: SmoothFftField>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
) -> Result<Vec<F>, AkitaError> {
    public.validate_layout(layout)?;
    let mut weights = zero_vec(layout.image_len())?;
    for e in 0..layout.image_count() {
        for (t, &power) in public.alpha_powers.iter().enumerate() {
            let address = layout.image_address(e, t)?;
            *weights.get_mut(address).ok_or(AkitaError::InvalidProof)? =
                public.image_entry_weight(e)? * power;
        }
    }
    Ok(weights)
}

/// Structured response-weight MLE with canonical A setup-weight preparation.
///
/// The explicit matrix contraction costs linear work in the setup domain.
/// A later protocol may supply this quantity as a setup-contribution claim.
/// Preparation's modulus evaluation is separate; it is not a weight factor.
pub fn witness_weight_mle<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    setup: &BinaryClearSetup<F, D, M>,
    rho: &[F],
) -> Result<F, AkitaError> {
    public.validate_layout(layout)?;
    layout.validate_setup(setup)?;
    if rho.len() != layout.witness_log_len() {
        return Err(AkitaError::InvalidInput(
            "response MLE point dimension mismatch".into(),
        ));
    }
    let row_weights = public
        .gamma_powers
        .get(..layout.n_a())
        .ok_or(AkitaError::InvalidProof)?;
    let prepared = layout.setup_view().prepare(
        rho,
        public.challenges.alpha,
        row_weights,
        &public.digit_powers,
    )?;
    let weights = prepared.materialize_setup_weights()?;
    let mut a_part = F::zero();
    for (element, coefficients) in setup.matrix().iter().enumerate() {
        for (t, &coefficient) in coefficients.coefficients().iter().enumerate() {
            let address =
                layout
                    .setup_view()
                    .setup_address(element / layout.m(), element % layout.m(), t)?;
            a_part += coefficient * *weights.get(address).ok_or(AkitaError::InvalidProof)?;
        }
    }
    let digit_bits = layout.response_layout().digit_depth().trailing_zeros() as usize;
    let coefficient_bits = layout.padded_coefficients().trailing_zeros() as usize;
    let component_bits = layout.k().trailing_zeros() as usize;
    let (rho_l, rest) = rho
        .split_at_checked(digit_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let (rho_t, rho_j) = rest
        .split_at_checked(coefficient_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let (rho_c, rho_s) = rho_t
        .split_at_checked(component_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let gadget = public
        .digit_powers
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (l, &digit)| {
            acc + eq_eval_at_index(rho_l, l) * digit
        });
    let mut scalar = F::zero();
    for (s, &power) in public.xi_powers.iter().enumerate() {
        let t = akita_error::checked::product([s, layout.k()]).ok_or(AkitaError::InvalidProof)?;
        scalar += eq_eval_at_index(rho_s, s) * signed_field::<F>(layout.sigma(t)?) * power;
    }
    let mut binary = F::zero();
    for (r, &value) in public.binary_rows.iter().enumerate() {
        binary += eq_eval_at_index(rho_c, r % layout.k())
            * eq_eval_at_index(rho_j, r / layout.k())
            * value;
    }
    Ok(a_part + public.g()? * gadget * scalar * binary)
}

/// Structured image-weight MLE in coefficient-low, image-entry-high order.
pub fn image_weight_mle<F: SmoothFftField>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    rho_y: &[F],
) -> Result<F, AkitaError> {
    public.validate_layout(layout)?;
    if rho_y.len() != layout.image_log_len() {
        return Err(AkitaError::InvalidInput(
            "image MLE point dimension mismatch".into(),
        ));
    }
    let (rho_t, rho_e) = rho_y
        .split_at_checked(layout.padded_coefficients().trailing_zeros() as usize)
        .ok_or(AkitaError::InvalidProof)?;
    let coefficient = public
        .alpha_powers
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (t, &power)| {
            acc + eq_eval_at_index(rho_t, t) * power
        });
    let mut entries = F::zero();
    for e in 0..layout.image_count() {
        entries += eq_eval_at_index(rho_e, e) * public.image_entry_weight(e)?;
    }
    Ok(coefficient * entries)
}
