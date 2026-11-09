//! Profile-selected residual bounds: seven P128 projections or P64 quotients.
//! The legacy sign-magnitude wire encoding is available only for P128.

use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing, FoldChallengeDistribution};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::{BinaryFieldProfile, BINARY_SETUP_P64};
use akita_params::sis::checked_balanced_digit_representable_bounds;
use akita_transcript::{ProofChannel, ProtocolSiteId, SITE_FAMILY_ROOT_STATEMENT};

use super::{
    BINARY_RESPONSE_INTERVAL, BINARY_SOURCE_Q, CANONICAL_IMAGE_WINDOW, ENFORCED_IMAGE_WINDOW,
};

pub const BINARY_PROJECTION_COUNT: usize = 7;
pub const BINARY_PROJECTION_BYTES: usize = 12;
const MAGNITUDE_LIMIT: u128 = 1u128 << 95;

/// Enforced residual envelope and profile-selected exact-lift admission.
/// A quotient selection has no projection messages or encoding-based owner cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryResidualBounds {
    residual_bound: u128,
    projection_bound: u128,
    rank: usize,
    polynomials: usize,
    residual_rows: usize,
    quotient: Option<BinaryQuotientBounds>,
}

impl BinaryResidualBounds {
    /// Admit geometry only when `2 U_J < P` and `U_J < 2^95`.
    ///
    /// `L_R = 972 m B_A B_Z + C Gamma B_T`, where the trinomial
    /// coefficient kernel has maximum absolute row sum 972. Each projection
    /// spans `a = rank * polynomials * 648` coefficients in polynomial-major
    /// order, and `U_J = (q-1) * a * L_R`. The other bounds
    /// are the canonical centered matrix interval, enforced response interval,
    /// challenge multiplication operator norm, and enforced image interval.
    pub fn new(
        source_rows: usize,
        columns: usize,
        rank: usize,
        polynomials: usize,
        proof_modulus: u128,
    ) -> Result<Self, AkitaError> {
        let overflow = || AkitaError::InvalidInput("binary projection bound overflow".into());
        if source_rows == 0 || columns == 0 || rank == 0 || polynomials == 0 {
            return Err(AkitaError::InvalidInput(
                "binary projection geometry must be nonempty".into(),
            ));
        }
        checked::product([source_rows, 648]).ok_or_else(overflow)?;
        checked::product([columns, 648]).ok_or_else(overflow)?;
        let residual_rows = checked::product([rank, polynomials]).ok_or_else(overflow)?;
        let coefficients = checked::product([residual_rows, 648]).ok_or_else(overflow)?;
        let residual_bound = residual_bound(source_rows, columns)?;
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
            polynomials,
            residual_rows,
            quotient: None,
        })
    }

    /// Select coefficientwise quotient admission without the P128 wire cap.
    pub fn for_profile(
        source_rows: usize,
        columns: usize,
        rank: usize,
        polynomials: usize,
        profile: BinaryFieldProfile,
    ) -> Result<Self, AkitaError> {
        if profile == BinaryFieldProfile::P128 {
            return Self::new(source_rows, columns, rank, polynomials, profile.modulus());
        }
        if rank != 3 || polynomials == 0 {
            return Err(AkitaError::InvalidInput(
                "p64 residuals require rank three and owners".into(),
            ));
        }
        let residual_rows = checked::product([rank, polynomials])
            .filter(|&rows| checked::product([rows, 648]).is_some())
            .ok_or(AkitaError::InvalidProof)?;
        let quotient = BinaryQuotientBounds::new(source_rows, columns)?;
        Ok(Self {
            residual_bound: quotient.residual_bound,
            projection_bound: 0,
            rank,
            polynomials,
            residual_rows,
            quotient: Some(quotient),
        })
    }

    pub fn quotient(&self) -> Option<&BinaryQuotientBounds> {
        self.quotient.as_ref()
    }

    /// No J rows or y messages exist in the quotient profile.
    pub fn projection_count(&self) -> usize {
        if self.quotient.is_some() {
            0
        } else {
            BINARY_PROJECTION_COUNT
        }
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
    pub fn polynomials(&self) -> usize {
        self.polynomials
    }
    /// Number of rank rows across all polynomials, in polynomial-major order.
    pub fn residual_rows(&self) -> usize {
        self.residual_rows
    }
}

/// Full accepted residual envelope shared by both field profiles.
fn residual_bound(source_rows: usize, columns: usize) -> Result<u128, AkitaError> {
    let overflow = || AkitaError::InvalidInput("binary residual bound overflow".into());
    if source_rows == 0 || columns == 0 {
        return Err(AkitaError::InvalidInput(
            "empty binary residual geometry".into(),
        ));
    }
    checked::product([source_rows, 648]).ok_or_else(overflow)?;
    checked::product([columns, 648]).ok_or_else(overflow)?;
    let challenge = BinaryChallengeProfile::bounded_weight(
        BinaryScalarRing::Cyclotomic243,
        FoldChallengeDistribution::BinaryRoot.weight(),
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
    Ok(residual_bound)
}

/// Coefficient lift and the minimal complete balanced base-eight quotient depth.
/// Soundness uses the full negative reach, including malicious accepted digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryQuotientBounds {
    pub residual_bound: u128,
    pub quotient_floor: u128,
    pub digits: usize,
    pub positive_reach: u128,
    pub quotient_bound: u128,
    pub no_wrap_bound: u128,
}

impl BinaryQuotientBounds {
    pub fn new(source_rows: usize, columns: usize) -> Result<Self, AkitaError> {
        let bad = || AkitaError::InvalidInput("binary quotient lift admission failed".into());
        let residual_bound = residual_bound(source_rows, columns)?;
        let quotient_floor = residual_bound / u128::from(BINARY_SOURCE_Q);
        let mut digits = 1usize;
        let (quotient_bound, positive_reach) = loop {
            let (negative, positive) = checked_balanced_digit_representable_bounds(3, digits);
            let positive = positive.ok_or_else(bad)?;
            if positive >= quotient_floor {
                break (negative.ok_or_else(bad)?, positive);
            }
            digits = digits.checked_add(1).ok_or_else(bad)?;
        };
        let no_wrap_bound = quotient_bound
            .checked_mul(u128::from(BINARY_SOURCE_Q))
            .and_then(|bound| bound.checked_add(residual_bound))
            .ok_or_else(bad)?;
        if no_wrap_bound >= u128::from(BINARY_SETUP_P64) {
            return Err(bad());
        }
        Ok(Self {
            residual_bound,
            quotient_floor,
            digits,
            positive_reach,
            quotient_bound,
            no_wrap_bound,
        })
    }
}

/// Encode one bounded multiple of q as little-endian 95-bit sign-magnitude.
/// The high bit is the sign; zero has its sign bit clear.
pub fn encode_binary_projection(
    value: i128,
    bounds: &BinaryResidualBounds,
) -> Result<[u8; BINARY_PROJECTION_BYTES], AkitaError> {
    if bounds.quotient().is_some()
        || value.unsigned_abs() > bounds.projection_bound()
        || value % i128::from(BINARY_SOURCE_Q) != 0
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
    bounds: &BinaryResidualBounds,
) -> Result<i128, AkitaError> {
    if bounds.quotient().is_some() || bytes.len() != BINARY_PROJECTION_BYTES {
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
    channel: &mut impl ProofChannel,
    bounds: &BinaryResidualBounds,
    values: &[i128; BINARY_PROJECTION_COUNT],
) -> Result<[i128; BINARY_PROJECTION_COUNT], AkitaError> {
    if bounds.quotient().is_some() {
        return Err(AkitaError::InvalidInput(
            "quotient profile has no projection messages".into(),
        ));
    }
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
