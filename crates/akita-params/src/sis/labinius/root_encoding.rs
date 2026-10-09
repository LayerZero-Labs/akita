use akita_error::{checked, AkitaError};

use super::{parity_no_wrap_total, LabiniusRootShape};

/// Closed digit alphabets; a tag selects the number of bits in each digit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LabiniusDigitBase {
    /// Digits in `[0, 1]`.
    Bits1,
    /// Digits in `[0, 3]`.
    Bits2,
    /// Digits in `[0, 15]`.
    Bits4,
}

impl LabiniusDigitBase {
    /// Stable wire tag, never reassigned.
    pub const fn tag(self) -> u8 {
        match self {
            Self::Bits1 => 0,
            Self::Bits2 => 1,
            Self::Bits4 => 2,
        }
    }

    /// Decode the closed digit alphabet.
    pub fn from_tag(tag: u8) -> Result<Self, AkitaError> {
        match tag {
            0 => Ok(Self::Bits1),
            1 => Ok(Self::Bits2),
            2 => Ok(Self::Bits4),
            _ => Err(AkitaError::InvalidSetup(format!(
                "unknown LaBinius digit base tag {tag}"
            ))),
        }
    }

    /// Bits per digit `b`.
    pub const fn bits(self) -> u32 {
        match self {
            Self::Bits1 => 1,
            Self::Bits2 => 2,
            Self::Bits4 => 4,
        }
    }
}

/// One signed map: `value = sum_i digit_i * 2^(b*i) - offset`.
///
/// The exact accepted interval is `[-offset, offset-1]`. The magnitude used
/// for no-wrap admission is `offset`, the larger absolute endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabiniusSignedDigitRange {
    bits: u32,
    digit_count: usize,
    offset: u128,
    interval: (i128, i128),
}

impl LabiniusSignedDigitRange {
    /// Enforced bit width `e`.
    pub const fn bits(self) -> u32 {
        self.bits
    }
    /// Number of digits `e/b`.
    pub const fn digit_count(self) -> usize {
        self.digit_count
    }
    /// Signed offset and enforced magnitude `2^(e-1)`.
    pub const fn offset(self) -> u128 {
        self.offset
    }
    /// Inclusive accepted endpoints.
    pub const fn interval(self) -> (i128, i128) {
        self.interval
    }
}

/// Accepted integer ranges and natural/padded lengths for the root digit vector.
///
/// Segment order is response, parity quotient, parity carry. Within-segment
/// address order belongs to the lowered relation. All padded digits have the
/// same alphabet, zero public relation weights and honest value zero. The image
/// is a coefficient-field vector and needs no range proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabiniusRootEncoding {
    base: LabiniusDigitBase,
    response: LabiniusSignedDigitRange,
    quotient: LabiniusSignedDigitRange,
    carry: LabiniusSignedDigitRange,
    no_wrap_total: u128,
    segment_lengths: [usize; 3],
    natural_len: usize,
    padded_log_len: usize,
    image_padded_log_len: usize,
}

impl LabiniusRootEncoding {
    /// Selected digit alphabet.
    pub const fn base(&self) -> LabiniusDigitBase {
        self.base
    }
    /// Accepted response map.
    pub const fn response(&self) -> LabiniusSignedDigitRange {
        self.response
    }
    /// Accepted parity quotient map.
    pub const fn quotient(&self) -> LabiniusSignedDigitRange {
        self.quotient
    }
    /// Accepted parity carry map.
    pub const fn carry(&self) -> LabiniusSignedDigitRange {
        self.carry
    }
    /// Checked `H + 3*quotient.offset + 2*carry.offset`, strictly below P.
    pub const fn no_wrap_total(&self) -> u128 {
        self.no_wrap_total
    }
    /// Natural digit lengths, in response/quotient/carry order.
    pub const fn segment_lengths(&self) -> [usize; 3] {
        self.segment_lengths
    }
    /// Sum of the three natural segment lengths.
    pub const fn natural_len(&self) -> usize {
        self.natural_len
    }
    /// Smallest `n` with `2^n >= natural_len`.
    pub const fn padded_log_len(&self) -> usize {
        self.padded_log_len
    }
    /// Smallest `n` with `2^n >= shape.image_len`.
    pub const fn image_padded_log_len(&self) -> usize {
        self.image_padded_log_len
    }
}

impl LabiniusRootShape {
    /// Derive enforced ranges and witness lengths for one digit alphabet.
    /// No allocation occurs; overflow or unsupported accepted intervals reject.
    pub fn derive_encoding(
        &self,
        base: LabiniusDigitBase,
    ) -> Result<LabiniusRootEncoding, AkitaError> {
        let (lower, upper) = self.response_interval;
        let offset = lower.unsigned_abs();
        if lower >= 0 || !offset.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "LaBinius response interval is not a centered power of two".into(),
            ));
        }
        let response_bits = offset.trailing_zeros().checked_add(1).ok_or_else(|| {
            AkitaError::InvalidSetup("LaBinius response bit width overflow".into())
        })?;
        let response = signed_digit_range(response_bits, base)?;
        if response.interval != (lower, upper) {
            return Err(AkitaError::InvalidSetup(
                "LaBinius response interval is not a centered power of two".into(),
            ));
        }
        let quotient = envelope_for_bound(self.honest_quotient_bound, base)?;
        let carry = envelope_for_bound(self.honest_carry_bound, base)?;
        self.check_parity_no_wrap(quotient.offset, carry.offset)?;
        let no_wrap_total =
            parity_no_wrap_total(self.parity_residual_bound, quotient.offset, carry.offset)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("LaBinius parity no-wrap sum overflow".into())
                })?;
        let size_overflow =
            || AkitaError::InvalidSetup("LaBinius root digit witness size overflow".into());
        let segment_lengths = [
            checked::product([
                self.ring_elements_per_column,
                self.commitment_degree,
                response.digit_count,
            ])
            .ok_or_else(size_overflow)?,
            checked::product([
                self.scalar_degree
                    .checked_sub(1)
                    .ok_or_else(size_overflow)?,
                quotient.digit_count,
            ])
            .ok_or_else(size_overflow)?,
            checked::product([self.scalar_degree, carry.digit_count]).ok_or_else(size_overflow)?,
        ];
        let natural_len = checked::sum(segment_lengths).ok_or_else(size_overflow)?;
        let padded_log_len = checked::ceil_log2(natural_len).ok_or_else(size_overflow)?;
        let image_padded_log_len = checked::ceil_log2(self.image_len).ok_or_else(size_overflow)?;
        Ok(LabiniusRootEncoding {
            base,
            response,
            quotient,
            carry,
            no_wrap_total,
            segment_lengths,
            natural_len,
            padded_log_len,
            image_padded_log_len,
        })
    }
}

// Canonical signed map metadata, shared by all three integer families.
fn signed_digit_range(
    bits: u32,
    base: LabiniusDigitBase,
) -> Result<LabiniusSignedDigitRange, AkitaError> {
    if bits == 0 || !bits.is_multiple_of(base.bits()) {
        return Err(AkitaError::InvalidSetup(
            "LaBinius enforced bits must be a positive multiple of digit bits".into(),
        ));
    }
    let overflow = || AkitaError::InvalidSetup("LaBinius signed digit range overflow".into());
    let offset = 1u128
        .checked_shl(bits.checked_sub(1).ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    let upper =
        i128::try_from(offset.checked_sub(1).ok_or_else(overflow)?).map_err(|_| overflow())?;
    let lower = upper
        .checked_neg()
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(overflow)?;
    let digit_count = checked::exact_div(
        usize::try_from(bits).map_err(|_| overflow())?,
        usize::try_from(base.bits()).map_err(|_| overflow())?,
    )
    .ok_or_else(overflow)?;
    Ok(LabiniusSignedDigitRange {
        bits,
        digit_count,
        offset,
        interval: (lower, upper),
    })
}

fn envelope_for_bound(
    bound: u128,
    base: LabiniusDigitBase,
) -> Result<LabiniusSignedDigitRange, AkitaError> {
    let overflow = || AkitaError::InvalidSetup("LaBinius honest digit envelope overflow".into());
    let required = bound
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(overflow)?;
    let bits = u128::BITS
        .checked_sub(
            required
                .checked_sub(1)
                .ok_or_else(overflow)?
                .leading_zeros(),
        )
        .ok_or_else(overflow)?
        .max(1);
    let digit_bits = usize::try_from(base.bits()).map_err(|_| overflow())?;
    let rounded_bits = checked::product([
        checked::div_ceil(usize::try_from(bits).map_err(|_| overflow())?, digit_bits)
            .ok_or_else(overflow)?,
        digit_bits,
    ])
    .ok_or_else(overflow)?;
    signed_digit_range(u32::try_from(rounded_bits).map_err(|_| overflow())?, base)
}

#[cfg(test)]
#[path = "root_encoding_tests.rs"]
mod tests;
