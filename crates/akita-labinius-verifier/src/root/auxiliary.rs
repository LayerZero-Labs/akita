use crate::{channel::ClearChannel, codec::exchange_bounded_integer, lowered::LoweredRootLayout};
use akita_error::AkitaError;

/// Exchange the commitment carry `K_A`, then, when the statement has a binary
/// claim, the parity quotient `Q` and the parity carry `K`. The admitted
/// signed ranges and canonical encodings are enforced.
pub fn exchange_root_auxiliary<S: ClearChannel>(
    layout: &LoweredRootLayout,
    channel: &mut S,
    a_carry: &mut [i128],
    parity: Option<(&mut [i128], &mut [i128])>,
) -> Result<(), AkitaError> {
    let encoding = layout.encoding();
    if a_carry.len() != encoding.a_carry_len() {
        return Err(AkitaError::InvalidProof);
    }
    for value in a_carry {
        exchange_bounded_integer(channel, value, encoding.a_carry().bits())?;
    }
    if let Some((quotient, carry)) = parity {
        if quotient.len() != encoding.parity_quotient_len()
            || carry.len() != encoding.parity_carry_len()
        {
            return Err(AkitaError::InvalidProof);
        }
        for value in quotient {
            exchange_bounded_integer(channel, value, encoding.quotient().bits())?;
        }
        for value in carry {
            exchange_bounded_integer(channel, value, encoding.carry().bits())?;
        }
    }
    Ok(())
}
