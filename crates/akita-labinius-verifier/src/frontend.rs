//! Field switch and F162 product sumcheck, reusable before any source opening.
use crate::{channel::ClearChannel, codec::exchange_binary};
use akita_algebra::binary::{
    field_switch::{
        batched_weights, partial_evaluations, transparent_weight, SwitchField, SwitchPartials,
    },
    BinaryField162 as B, PackedBinary162,
};
use akita_error::{checked, AkitaError};

/// Source evaluation obligation produced by the host-field frontend.
/// The owner must open this value even when the transparent weight is zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryEvaluationClaim {
    pub point: Vec<B>,
    pub value: B,
}

/// Exactly uniform F162 challenge: mask the six unused bits of byte 20.
pub fn binary_challenge<S: ClearChannel>(channel: &mut S) -> Result<B, AkitaError> {
    let block = channel.challenge_block()?;
    let mut bytes = [0u8; 21];
    bytes.copy_from_slice(block.get(..21).ok_or(AkitaError::InvalidProof)?);
    *bytes.last_mut().ok_or(AkitaError::InvalidProof)? &= 3;
    B::from_bytes(&bytes).ok_or(AkitaError::InvalidProof)
}

fn exchange_partials<H: SwitchField, S: ClearChannel>(
    channel: &mut S,
    partials: &SwitchPartials<H>,
) -> Result<SwitchPartials<H>, AkitaError> {
    let width = std::mem::size_of::<H::Source>();
    let mut values = Vec::new();
    values
        .try_reserve_exact(1 << H::BATCH_BITS)
        .map_err(|_| AkitaError::InvalidProof)?;
    // Only live rows travel on the wire. F192's zero padding is reconstructed.
    for &value in partials
        .values()
        .get(..H::ROWS)
        .ok_or(AkitaError::InvalidProof)?
    {
        let mut bytes = [0u8; 16];
        let mut encoded = Vec::from(
            value
                .into()
                .to_le_bytes()
                .get(..width)
                .ok_or(AkitaError::InvalidProof)?,
        );
        channel.message(&mut encoded)?;
        bytes
            .get_mut(..width)
            .ok_or(AkitaError::InvalidProof)?
            .copy_from_slice(&encoded);
        values.push(
            H::Source::try_from(u128::from_le_bytes(bytes))
                .map_err(|_| AkitaError::InvalidProof)?,
        );
    }
    values.resize(1 << H::BATCH_BITS, H::Source::default());
    SwitchPartials::try_from_values(values).map_err(|_| AkitaError::InvalidProof)
}

fn batch_point<S: ClearChannel>(channel: &mut S, bits: usize) -> Result<Vec<B>, AkitaError> {
    (0..bits).map(|_| binary_challenge(channel)).collect()
}

// One transcript and characteristic-two update for both protocol roles.
fn sumcheck_round<S: ClearChannel>(
    channel: &mut S,
    claim: &mut B,
    mut c0: B,
    mut c2: B,
) -> Result<B, AkitaError> {
    exchange_binary(channel, &mut c0)?;
    exchange_binary(channel, &mut c2)?;
    let c1 = *claim + c2;
    let z = binary_challenge(channel)?;
    *claim = c0 + c1 * z + c2 * z.square();
    Ok(z)
}

/// Produce steps 1-4. Source and public statement must already be bound.
pub fn prove_frontend<H: SwitchField, S: ClearChannel>(
    source: &[H::Source],
    point: &[H],
    value: H,
    channel: &mut S,
) -> Result<BinaryEvaluationClaim, AkitaError> {
    let mut equality = Vec::new();
    let partials = partial_evaluations::<H>(source, point, &mut equality)?;
    if partials.reconstruct() != value {
        return Err(AkitaError::InvalidInput(
            "incorrect host evaluation claim".into(),
        ));
    }
    let partials = exchange_partials(channel, &partials)?;
    let s = batch_point(channel, H::BATCH_BITS)?;
    let mut claim = partials.batch(&s)?;
    let mut a = PackedBinary162::new();
    a.refill_binary_words(source);
    let mut b = PackedBinary162::new();
    batched_weights::<H>(&equality, &s, &mut b)?;
    let mut z = Vec::new();
    z.try_reserve_exact(point.len())
        .map_err(|_| AkitaError::InvalidInput("frontend point allocation failed".into()))?;
    for _ in point {
        let [c0, _, c2] = a
            .round_product(&b, claim)
            .ok_or_else(|| AkitaError::InvalidInput("invalid sumcheck table shape".into()))?;
        let zi = sumcheck_round(channel, &mut claim, c0, c2)?;
        a.fold_in_place(zi);
        b.fold_in_place(zi);
        z.push(zi);
    }
    let mut terminal = a
        .get(0)
        .ok_or_else(|| AkitaError::InvalidInput("empty source table".into()))?;
    exchange_binary(channel, &mut terminal)?;
    if claim != terminal * transparent_weight::<H>(point, &z, &s)? {
        return Err(AkitaError::InvalidInput(
            "inconsistent frontend terminal".into(),
        ));
    }
    Ok(BinaryEvaluationClaim {
        point: z,
        value: terminal,
    })
}

/// Verify steps 1-4 and return the mandatory binary source-opening obligation.
/// This function alone does not verify an opening against a commitment.
pub fn verify_frontend<H: SwitchField, S: ClearChannel>(
    point: &[H],
    value: H,
    channel: &mut S,
) -> Result<BinaryEvaluationClaim, AkitaError> {
    checked::pow2(point.len())
        .ok_or_else(|| AkitaError::InvalidInput("frontend dimension overflow".into()))?;
    let zeros =
        SwitchPartials::<H>::try_from_values(vec![H::Source::default(); 1 << H::BATCH_BITS])?;
    let partials = exchange_partials(channel, &zeros)?;
    if partials.reconstruct() != value {
        return Err(AkitaError::InvalidProof);
    }
    let s = batch_point(channel, H::BATCH_BITS)?;
    let mut claim = partials.batch(&s).map_err(|_| AkitaError::InvalidProof)?;
    let mut z = Vec::new();
    z.try_reserve_exact(point.len())
        .map_err(|_| AkitaError::InvalidInput("frontend point allocation failed".into()))?;
    for _ in point {
        let zi = sumcheck_round(channel, &mut claim, B::ZERO, B::ZERO)?;
        z.push(zi);
    }
    let mut terminal = B::ZERO;
    exchange_binary(channel, &mut terminal)?;
    if claim != terminal * transparent_weight::<H>(point, &z, &s)? {
        return Err(AkitaError::InvalidProof);
    }
    Ok(BinaryEvaluationClaim {
        point: z,
        value: terminal,
    })
}
