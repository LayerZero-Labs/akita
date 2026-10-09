use crate::{channel::ClearChannel, codec::exchange_bounded_integer, lowered::LoweredRootLayout};
use akita_error::AkitaError;

/// Exchange optional KA, parity quotient Q, and parity carry K, in that order.
/// The admitted offset integer ranges and canonical encodings are enforced.
pub fn exchange_root_auxiliary<S: ClearChannel>(
    layout: &LoweredRootLayout,
    channel: &mut S,
    a_carry: &mut [i128],
    parity_quotient: &mut [i128],
    parity_carry: &mut [i128],
) -> Result<(), AkitaError> {
    let encoding = layout.encoding();
    if a_carry.len() != encoding.a_carry_len()
        || parity_quotient.len() != encoding.parity_quotient_len()
        || parity_carry.len() != encoding.parity_carry_len()
    {
        return Err(AkitaError::InvalidProof);
    }
    if let Some(range) = encoding.a_carry() {
        for value in a_carry {
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
