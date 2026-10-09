//! Compact coefficient factor of the digit-innermost response weights.

use akita_algebra::SmoothFftField;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::lowered::{LoweredPublic, LoweredRootLayout};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Build the response-weight factor in coefficient-low, ring-element-high order.
///
/// The caller supplies the layout used to construct `public`. Multiplying this
/// table by `public.digit_powers()` in digit-innermost order gives the dense
/// response weights, including their zero coefficient padding.
pub fn coefficient_weights<F: SmoothFftField>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
) -> Result<Vec<F>, AkitaError> {
    let invalid = || AkitaError::InvalidInput("invalid compact response-weight geometry".into());
    let digit_depth = layout.response_layout().digit_depth();
    let len = checked::product([layout.m(), layout.padded_coefficients()]).ok_or_else(invalid)?;
    if public.digit_powers().len() != digit_depth
        || checked::product([len, digit_depth]) != Some(layout.witness_len())
    {
        return Err(invalid());
    }
    let mut weights = Vec::new();
    weights.try_reserve_exact(len).map_err(|_| {
        AkitaError::InvalidInput("compact response-weight allocation failed".into())
    })?;
    weights.resize(len, F::zero());
    let fill_element = |(j, coefficients): (usize, &mut [F])| -> Result<(), AkitaError> {
        for (t, weight) in coefficients.iter_mut().take(layout.degree()).enumerate() {
            *weight = public.coefficient_weight(j, t)?;
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    weights
        .par_chunks_mut(layout.padded_coefficients())
        .enumerate()
        .try_for_each(fill_element)?;
    #[cfg(not(feature = "parallel"))]
    weights
        .chunks_mut(layout.padded_coefficients())
        .enumerate()
        .try_for_each(fill_element)?;
    Ok(weights)
}
