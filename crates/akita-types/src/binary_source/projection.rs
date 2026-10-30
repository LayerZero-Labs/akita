//! Exact integer bounds and wire encoding for seven divisibility projections.

use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing, FoldChallengeFamily};
use akita_error::{checked, AkitaError};
use akita_transcript::{ProofChannel, ProtocolSiteId, SITE_FAMILY_ROOT_STATEMENT};

use super::{
    BINARY_RESPONSE_INTERVAL, BINARY_SOURCE_Q, CANONICAL_IMAGE_WINDOW, ENFORCED_IMAGE_WINDOW,
};

pub const BINARY_PROJECTION_COUNT: usize = 7;
pub const BINARY_PROJECTION_BYTES: usize = 12;
const MAGNITUDE_LIMIT: u128 = 1u128 << 95;

/// Enforced residual envelope and exact-lift admission for projection messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryProjectionBounds {
    residual_bound: u128,
    projection_bound: u128,
    rank: usize,
}

impl BinaryProjectionBounds {
    /// Admit geometry only when `2 U_J < P` and `U_J < 2^95`.
    ///
    /// `L_R = 972 m B_A B_Z + C Gamma B_T`, where the trinomial
    /// coefficient kernel has maximum absolute row sum 972. The other bounds
    /// are the canonical centered matrix interval, enforced response interval,
    /// challenge multiplication operator norm, and enforced image interval.
    pub fn new(
        source_rows: usize,
        columns: usize,
        rank: usize,
        proof_modulus: u128,
    ) -> Result<Self, AkitaError> {
        let overflow = || AkitaError::InvalidInput("binary projection bound overflow".into());
        if source_rows == 0 || columns == 0 || rank == 0 {
            return Err(AkitaError::InvalidInput(
                "binary projection geometry must be nonempty".into(),
            ));
        }
        checked::product([source_rows, 648]).ok_or_else(overflow)?;
        checked::product([columns, 648]).ok_or_else(overflow)?;
        let coefficients = checked::product([rank, 648]).ok_or_else(overflow)?;
        let challenge = BinaryChallengeProfile::bounded_weight(
            BinaryScalarRing::Cyclotomic243,
            FoldChallengeFamily::BinaryRoot.weight(),
        )?;
        let gamma = u128::from(challenge.multiplication_linf_operator_bound());
        let matrix_bound = u128::from(CANONICAL_IMAGE_WINDOW.0.unsigned_abs());
        let response_bound = u128::from(BINARY_RESPONSE_INTERVAL.0.unsigned_abs());
        let image_bound = u128::from(ENFORCED_IMAGE_WINDOW.0.unsigned_abs());
        let residual_bound = 972u128
            .checked_mul(source_rows as u128)
            .and_then(|x| x.checked_mul(matrix_bound))
            .and_then(|x| x.checked_mul(response_bound))
            .zip(
                (columns as u128)
                    .checked_mul(gamma)
                    .and_then(|x| x.checked_mul(image_bound)),
            )
            .and_then(|(a, b)| a.checked_add(b))
            .ok_or_else(overflow)?;
        let projection_bound = u128::from(BINARY_SOURCE_Q - 1)
            .checked_mul(coefficients as u128)
            .and_then(|x| x.checked_mul(residual_bound))
            .ok_or_else(overflow)?;
        if projection_bound >= MAGNITUDE_LIMIT
            || projection_bound.checked_mul(2).ok_or_else(overflow)? >= proof_modulus
        {
            return Err(AkitaError::InvalidInput(
                "binary projection requires 2 U_J < P and U_J < 2^95".into(),
            ));
        }
        Ok(Self {
            residual_bound,
            projection_bound,
            rank,
        })
    }

    pub fn residual_bound(&self) -> u128 {
        self.residual_bound
    }
    pub fn projection_bound(&self) -> u128 {
        self.projection_bound
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
}

/// Encode one bounded multiple of q as little-endian 95-bit sign-magnitude.
/// The high bit is the sign; zero has its sign bit clear.
pub fn encode_binary_projection(
    value: i128,
    bounds: &BinaryProjectionBounds,
) -> Result<[u8; BINARY_PROJECTION_BYTES], AkitaError> {
    if value.unsigned_abs() > bounds.projection_bound() || value % i128::from(BINARY_SOURCE_Q) != 0
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut bytes = [0; BINARY_PROJECTION_BYTES];
    bytes.copy_from_slice(
        value
            .unsigned_abs()
            .to_le_bytes()
            .get(..BINARY_PROJECTION_BYTES)
            .ok_or_else(|| AkitaError::Internal("binary projection encoding width".into()))?,
    );
    if value < 0 {
        *bytes.last_mut().ok_or(AkitaError::InvalidProof)? |= 0x80;
    }
    Ok(bytes)
}

/// Decode one canonical, bounded multiple of q, rejecting negative zero.
pub fn decode_binary_projection(
    bytes: &[u8],
    bounds: &BinaryProjectionBounds,
) -> Result<i128, AkitaError> {
    if bytes.len() != BINARY_PROJECTION_BYTES {
        return Err(AkitaError::InvalidProof);
    }
    let mut magnitude_bytes = [0; 16];
    magnitude_bytes
        .get_mut(..BINARY_PROJECTION_BYTES)
        .ok_or(AkitaError::InvalidProof)?
        .copy_from_slice(bytes);
    let last = magnitude_bytes
        .get_mut(BINARY_PROJECTION_BYTES - 1)
        .ok_or(AkitaError::InvalidProof)?;
    let negative = *last & 0x80 != 0;
    *last &= 0x7f;
    let magnitude = u128::from_le_bytes(magnitude_bytes);
    if magnitude > bounds.projection_bound()
        || (negative && magnitude == 0)
        || !magnitude.is_multiple_of(u128::from(BINARY_SOURCE_Q))
    {
        return Err(AkitaError::InvalidProof);
    }
    let value = i128::try_from(magnitude).map_err(|_| AkitaError::InvalidProof)?;
    Ok(if negative { -value } else { value })
}

/// Exchange all seven signed values before sampling any output batching scalar.
/// Verifier callers supply seven zero placeholders, which the channel replaces.
pub fn binary_projection_values(
    channel: &mut (impl ProofChannel + ?Sized),
    bounds: &BinaryProjectionBounds,
    values: &[i128; BINARY_PROJECTION_COUNT],
) -> Result<[i128; BINARY_PROJECTION_COUNT], AkitaError> {
    akita_transcript::public_bytes(
        channel,
        ProtocolSiteId {
            family: SITE_FAMILY_ROOT_STATEMENT,
            stage: 6,
            ..ProtocolSiteId::default()
        },
        b"binary exact divisibility values",
    )?;
    let mut received = [0; BINARY_PROJECTION_COUNT];
    for (value, received) in values.iter().zip(&mut received) {
        let mut bytes = encode_binary_projection(*value, bounds)?;
        channel.exchange(&mut bytes)?;
        *received = decode_binary_projection(&bytes, bounds)?;
    }
    Ok(received)
}

#[cfg(test)]
#[path = "projection_tests.rs"]
mod tests;
