//! Shared wire atoms for the carrier-level binary field switch.

use akita_algebra::binary::{
    field_switch::{transparent_weight, SwitchPartials},
    BinaryField128 as H, BinaryField162 as B,
};
use akita_error::AkitaError;
use akita_transcript::ProofChannel;

use super::{BinarySourceDescriptor, BinaryTerminalClaim, BinaryWordClaim};

/// Exchange exactly 128 canonical H carrier partials before the row selector.
pub fn field_switch_partials(
    channel: &mut (impl ProofChannel + ?Sized),
    values: &[u128],
) -> Result<SwitchPartials<H>, AkitaError> {
    if values.len() != 128 {
        return Err(AkitaError::Internal("field-switch partial count".into()));
    }
    let mut received = Vec::with_capacity(128);
    for value in values {
        let mut bytes = value.to_le_bytes();
        channel.exchange(&mut bytes)?;
        received.push(u128::from_le_bytes(bytes));
    }
    SwitchPartials::try_from_values(received)
}

/// Draw exactly 162 uniform bits, without reduction modulo an odd prime.
pub fn field_switch_challenge(channel: &mut (impl ProofChannel + ?Sized)) -> Result<B, AkitaError> {
    let block = channel.challenge_block()?;
    let mut bytes = [0; 21];
    bytes.copy_from_slice(block.get(..21).ok_or(AkitaError::Internal(
        "field-switch challenge block width".into(),
    ))?);
    *bytes.last_mut().ok_or(AkitaError::Internal(
        "field-switch challenge encoding width".into(),
    ))? &= 3;
    B::from_bytes(&bytes).ok_or(AkitaError::InvalidProof)
}

/// Exchange the two free coefficients of a characteristic-two quadratic.
/// Its linear coefficient is determined by `g(0)+g(1)=claim`.
pub fn field_switch_round(
    channel: &mut (impl ProofChannel + ?Sized),
    claim: &mut B,
    c0: B,
    c2: B,
) -> Result<B, AkitaError> {
    let mut bytes0 = c0.to_bytes();
    let mut bytes2 = c2.to_bytes();
    channel.exchange(&mut bytes0)?;
    channel.exchange(&mut bytes2)?;
    let c0 = B::from_bytes(&bytes0).ok_or(AkitaError::InvalidProof)?;
    let c2 = B::from_bytes(&bytes2).ok_or(AkitaError::InvalidProof)?;
    let z = field_switch_challenge(channel)?;
    *claim = c0 + (*claim + c2) * z + c2 * z.square();
    Ok(z)
}

/// Read the mandatory source value and check its product with the public weight.
///
/// Even when the weight is zero, retain this source value for root authentication.
/// The low `log2(C)` carrier variables select columns; the rest select rows.
#[allow(clippy::too_many_arguments)]
pub fn field_switch_terminal(
    channel: &mut (impl ProofChannel + ?Sized),
    descriptor: &BinarySourceDescriptor,
    word_claim: &BinaryWordClaim,
    host_point: &[H],
    lambda: &[B],
    point: &[B],
    product: B,
    source_value: B,
) -> Result<BinaryTerminalClaim, AkitaError> {
    let mut bytes = source_value.to_bytes();
    channel.exchange(&mut bytes)?;
    let value = B::from_bytes(&bytes).ok_or(AkitaError::InvalidProof)?;
    if product != value * transparent_weight::<H>(host_point, point, lambda)? {
        return Err(AkitaError::InvalidProof);
    }
    let columns = descriptor.index_map().columns().trailing_zeros() as usize;
    let column_point = point.get(..columns).ok_or(AkitaError::InvalidProof)?;
    let row_point = point.get(columns..).ok_or(AkitaError::InvalidProof)?;
    BinaryTerminalClaim::new(
        descriptor,
        *word_claim.polynomial(),
        row_point.iter().map(|x| x.to_bytes()).collect(),
        column_point.iter().map(|x| x.to_bytes()).collect(),
        bytes,
    )
}
