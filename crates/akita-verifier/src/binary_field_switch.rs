//! Replay the carrier-level field switch before binary column challenges.

use akita_algebra::binary::BinaryField162 as B;
use akita_error::AkitaError;
use akita_transcript::ProofChannel;
use akita_types::binary_source::{
    field_switch_challenge, field_switch_partials, field_switch_round, field_switch_terminal,
    BinarySourceDescriptor, BinaryTerminalClaim, BinaryWordClaim,
};

/// Verify one word claim after its statement has been bound by the opening.
/// The returned source obligation is mandatory even for a zero public multiplier;
/// this function alone does not authenticate a commitment opening.
pub fn verify_field_switch(
    descriptor: &BinarySourceDescriptor,
    claim: &BinaryWordClaim,
    channel: &mut (impl ProofChannel + ?Sized),
) -> Result<BinaryTerminalClaim, AkitaError> {
    claim.validate(descriptor)?;
    let point = claim.carrier_point(descriptor)?;
    let partials = field_switch_partials(channel, &[0; 128])?;
    if claim.reconstructed_value(descriptor, &partials)? != claim.decoded_value() {
        return Err(AkitaError::InvalidProof);
    }
    let lambda = (0..7)
        .map(|_| field_switch_challenge(channel))
        .collect::<Result<Vec<_>, _>>()?;
    let mut product = partials.batch(&lambda)?;
    let mut z = Vec::with_capacity(point.len());
    for _ in &point {
        z.push(field_switch_round(channel, &mut product, B::ZERO, B::ZERO)?);
    }
    field_switch_terminal(
        channel,
        descriptor,
        claim,
        &point,
        &lambda,
        &z,
        product,
        B::ZERO,
    )
}
