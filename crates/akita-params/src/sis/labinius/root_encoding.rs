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

/// Enforced integer ranges and canonical root response-table lengths.
///
/// The committed table contains only response digits, addressed as
/// `digit + digit_count * (coefficient + padded_coefficient_len * ring_element)`.
/// Every position has the same alphabet; padding has zero public relation
/// weights and honest digit zero. Parity quotient, parity carry, and A-row carry
/// lengths count integers outside this table; the root protocol chooses how to
/// enforce their ranges. The image uses the same padded coefficient index over the
/// coefficient field and needs no range proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabiniusRootEncoding {
    base: LabiniusDigitBase,
    response: LabiniusSignedDigitRange,
    quotient: LabiniusSignedDigitRange,
    carry: LabiniusSignedDigitRange,
    a_carry: Option<LabiniusSignedDigitRange>,
    a_carry_len: usize,
    no_wrap_total: u128,
    padded_coefficient_len: usize,
    response_table_len: usize,
    response_table_log_len: usize,
    parity_quotient_len: usize,
    parity_carry_len: usize,
    image_table_len: usize,
    image_table_log_len: usize,
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
    /// Enforced parity quotient range, independent of its protocol representation.
    pub const fn quotient(&self) -> LabiniusSignedDigitRange {
        self.quotient
    }
    /// Enforced parity carry range, independent of its protocol representation.
    pub const fn carry(&self) -> LabiniusSignedDigitRange {
        self.carry
    }
    /// Enforced foreign-modulus A-row carry range, when the primes differ.
    pub const fn a_carry(&self) -> Option<LabiniusSignedDigitRange> {
        self.a_carry
    }
    /// A-row carry integers `n_A * D`, or zero for a shared-prime profile.
    pub const fn a_carry_len(&self) -> usize {
        self.a_carry_len
    }
    /// Checked `H + 3*quotient.offset + 2*carry.offset`, strictly below P.
    pub const fn no_wrap_total(&self) -> u128 {
        self.no_wrap_total
    }
    /// Coefficient-axis length `D` rounded up to a power of two.
    pub const fn padded_coefficient_len(&self) -> usize {
        self.padded_coefficient_len
    }
    /// Committed response digits `m * padded_coefficient_len * response.digit_count`.
    pub const fn response_table_len(&self) -> usize {
        self.response_table_len
    }
    /// Smallest `n` with `2^n >= response_table_len`.
    pub const fn response_table_log_len(&self) -> usize {
        self.response_table_log_len
    }
    /// Parity quotient integers `d - 1`, outside the committed response table.
    pub const fn parity_quotient_len(&self) -> usize {
        self.parity_quotient_len
    }
    /// Parity carry integers `d`, outside the committed response table.
    pub const fn parity_carry_len(&self) -> usize {
        self.parity_carry_len
    }
    /// Image field entries `padded_coefficient_len * C * n_A`, with no range proof.
    pub const fn image_table_len(&self) -> usize {
        self.image_table_len
    }
    /// Smallest `n` with `2^n >= image_table_len`.
    pub const fn image_table_log_len(&self) -> usize {
        self.image_table_log_len
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
        let a_carry = self
            .honest_a_carry_bound()
            .map(|bound| envelope_for_bound(bound, base))
            .transpose()?;
        if let Some(range) = a_carry {
            self.check_a_carry_no_wrap(range.bits())?;
        }
        let a_carry_len = self.a_carry_len();
        let size_overflow =
            || AkitaError::InvalidSetup("LaBinius root digit witness size overflow".into());
        let padded_coefficient_len = self
            .commitment_degree
            .checked_next_power_of_two()
            .ok_or_else(size_overflow)?;
        let response_table_len = checked::product([
            self.ring_elements_per_column,
            padded_coefficient_len,
            response.digit_count,
        ])
        .ok_or_else(size_overflow)?;
        let response_table_log_len =
            checked::ceil_log2(response_table_len).ok_or_else(size_overflow)?;
        let parity_quotient_len = self
            .scalar_degree
            .checked_sub(1)
            .ok_or_else(size_overflow)?;
        let parity_carry_len = self.scalar_degree;
        let image_table_len = checked::product([
            padded_coefficient_len,
            self.fold_width,
            usize::try_from(self.rank_a).map_err(|_| size_overflow())?,
        ])
        .ok_or_else(size_overflow)?;
        let image_table_log_len = checked::ceil_log2(image_table_len).ok_or_else(size_overflow)?;
        Ok(LabiniusRootEncoding {
            base,
            response,
            quotient,
            carry,
            a_carry,
            a_carry_len,
            no_wrap_total,
            padded_coefficient_len,
            response_table_len,
            response_table_log_len,
            parity_quotient_len,
            parity_carry_len,
            image_table_len,
            image_table_log_len,
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
