//! Carrier-level binary field switching inside the caller's transcript.

use akita_algebra::binary::{
    field_switch::{batched_weights, partial_evaluations},
    BinaryField128 as H, PackedBinary162,
};
use akita_error::AkitaError;
use akita_transcript::ProofChannel;
use akita_types::binary_source::{
    field_switch_challenge, field_switch_partials, field_switch_round, field_switch_terminal,
    BinarySourceDescriptor, BinaryTerminalClaim, BinaryWordClaim,
};

/// Prove one word claim after its statement has been bound by the opening.
/// `source` contains padded carriers in the descriptor's low-column-bit order.
/// The returned terminal must be authenticated by the owning grouped root.
pub fn prove_field_switch(
    descriptor: &BinarySourceDescriptor,
    claim: &BinaryWordClaim,
    source: &[u128],
    channel: &mut (impl ProofChannel + ?Sized),
) -> Result<BinaryTerminalClaim, AkitaError> {
    claim.validate(descriptor)?;
    let point = claim.carrier_point(descriptor)?;
    let mut equality = Vec::new();
    let partials = partial_evaluations::<H>(source, &point, &mut equality)?;
    if claim.reconstructed_value(descriptor, &partials)? != claim.decoded_value() {
        return Err(AkitaError::InvalidInput(
            "incorrect binary word claim".into(),
        ));
    }
    let partials = field_switch_partials(channel, partials.values())?;
    let lambda = (0..7)
        .map(|_| field_switch_challenge(channel))
        .collect::<Result<Vec<_>, _>>()?;
    let mut product = partials.batch(&lambda)?;
    let mut a = PackedBinary162::new();
    a.refill_binary_words(source);
    let mut b = PackedBinary162::new();
    batched_weights::<H>(&equality, &lambda, &mut b)?;
    let mut z = Vec::with_capacity(point.len());
    for _ in &point {
        let [c0, _, c2] = a.round_product(&b, product).ok_or_else(|| {
            AkitaError::Internal("invalid field-switch sumcheck table shape".into())
        })?;
        let coordinate = field_switch_round(channel, &mut product, c0, c2)?;
        a.fold_in_place(coordinate);
        b.fold_in_place(coordinate);
        z.push(coordinate);
    }
    let value = a
        .get(0)
        .ok_or_else(|| AkitaError::Internal("empty carrier terminal".into()))?;
    field_switch_terminal(
        channel, descriptor, claim, &point, &lambda, &z, product, value,
    )
}
