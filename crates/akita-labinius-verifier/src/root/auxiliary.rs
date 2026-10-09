use crate::{
    channel::ClearChannel,
    codec::{exchange_bounded_integer, exchange_field},
    lowered::{ARelationRows, LoweredRootLayout},
};
use akita_algebra::SmoothFftField;
use akita_error::AkitaError;

/// Exchange QA, optional KA, parity quotient, and parity carry, in that order.
/// Canonical field encodings and the admitted offset integer ranges are enforced.
pub fn exchange_root_auxiliary<F: SmoothFftField, S: ClearChannel>(
    layout: &LoweredRootLayout,
    channel: &mut S,
    a_rows: &mut (impl ARelationRows<F> + ?Sized),
    parity_quotient: &mut [i128],
    parity_carry: &mut [i128],
) -> Result<(), AkitaError> {
    let quotient_len = layout.polynomial().quotient_coefficient_len()?;
    let encoding = layout.encoding();
    let a_quotients = a_rows.quotients();
    if a_quotients.len() != layout.n_a()
        || a_quotients.iter().any(|row| row.len() != quotient_len)
        || a_rows.a_carry().len() != encoding.a_carry_len()
        || parity_quotient.len() != encoding.parity_quotient_len()
        || parity_carry.len() != encoding.parity_carry_len()
    {
        return Err(AkitaError::InvalidProof);
    }
    for value in a_rows.quotients_mut().iter_mut().flatten() {
        exchange_field(channel, value)?;
    }
    if let Some(range) = encoding.a_carry() {
        for value in a_rows.a_carry_mut() {
            exchange_bounded_integer(channel, value, range.bits())?;
        }
    }
    for value in parity_quotient {
        exchange_bounded_integer(channel, value, encoding.quotient().bits())?;
    }
    for value in parity_carry {
        exchange_bounded_integer(channel, value, encoding.carry().bits())?;
    }
    Ok(())
}
