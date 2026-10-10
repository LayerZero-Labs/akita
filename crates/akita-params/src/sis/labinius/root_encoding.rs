use akita_error::{checked, AkitaError};

use super::LabiniusRootShape;
use crate::sis::labinius::{check_labinius_proof_prime_units, labinius_image_digit_bound};

/// Signed range of one clear integer: `[-2^(bits-1), 2^(bits-1) - 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabiniusSignedRange {
    bits: u32,
}

impl LabiniusSignedRange {
    /// Narrowest range containing `[-bound, bound]`.
    fn covering(bound: u128) -> Result<Self, AkitaError> {
        let bits = u128::BITS - bound.leading_zeros() + 1;
        // Both endpoints must be `i128` values.
        if bits > i128::BITS {
            return Err(AkitaError::InvalidSetup(
                "LaBinius clear-integer range overflow".into(),
            ));
        }
        Ok(Self { bits })
    }

    /// Bit width `e` of the two's-complement range.
    pub const fn bits(self) -> u32 {
        self.bits
    }

    /// Enforced magnitude `2^(e-1)`, the larger absolute endpoint.
    pub const fn magnitude(self) -> u128 {
        1 << (self.bits - 1)
    }

    /// Inclusive accepted endpoints.
    pub const fn interval(self) -> (i128, i128) {
        let upper = (self.magnitude() - 1) as i128;
        (-upper - 1, upper)
    }
}

/// Enforced integer ranges and canonical root table lengths for one admitted
/// proof prime.
///
/// The committed response table holds stored digits in `[0, 15]`, addressed as
/// `digit + digit_slots * (coefficient + padded_coefficient_len * ring_element)`.
/// A coefficient `v` with packing offset `o` has
/// `v + o = sum_{l < digit_count} digit_l * 16^l`. The digit axis is padded to
/// the power of two `digit_slots` and the coefficient axis to
/// `padded_coefficient_len`; padding positions have zero public relation
/// weights and honest digit zero. The parity quotient, parity carry and
/// commitment carry are clear integers outside this table.
///
/// The committed image table has the same shape with its own digit axis:
/// `digit + image_digit_slots * (coefficient + padded_coefficient_len *
/// (column * n_A + row))`. An image coefficient `T`, honestly the canonical
/// residue in `[0, q)`, has `T + image_offset = sum_{l < image_digit_count}
/// digit_l * 16^l`. The same padding rule applies on all three axes.
///
/// The prime left opening, bound only when the statement has a prime claim,
/// is a table of challenge-field elements addressed as
/// `coefficient + padded_coefficient_len * column`, with no digit axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabiniusRootEncoding {
    response_interval: (i128, i128),
    response_digit_count: usize,
    response_digit_slots: usize,
    quotient: LabiniusSignedRange,
    carry: LabiniusSignedRange,
    a_carry: LabiniusSignedRange,
    a_carry_len: usize,
    padded_coefficient_len: usize,
    response_table_len: usize,
    response_table_log_len: usize,
    parity_quotient_len: usize,
    parity_carry_len: usize,
    image_digit_count: usize,
    image_digit_slots: usize,
    image_offset: u128,
    image_table_log_len: usize,
    prime_table_log_len: usize,
}

impl LabiniusRootEncoding {
    /// Inclusive accepted response coefficient interval `[L, U]`.
    pub const fn response_interval(&self) -> (i128, i128) {
        self.response_interval
    }
    /// Balanced base-16 digits `k` of one response coefficient.
    pub const fn response_digit_count(&self) -> usize {
        self.response_digit_count
    }
    /// Digit-axis length: `k` rounded up to a power of two.
    pub const fn response_digit_slots(&self) -> usize {
        self.response_digit_slots
    }
    /// Enforced parity quotient range.
    pub const fn quotient(&self) -> LabiniusSignedRange {
        self.quotient
    }
    /// Enforced parity carry range.
    pub const fn carry(&self) -> LabiniusSignedRange {
        self.carry
    }
    /// Enforced commitment-row carry range.
    pub const fn a_carry(&self) -> LabiniusSignedRange {
        self.a_carry
    }
    /// Commitment-row carry integers `n_A * D`.
    pub const fn a_carry_len(&self) -> usize {
        self.a_carry_len
    }
    /// Coefficient-axis length `D` rounded up to a power of two.
    pub const fn padded_coefficient_len(&self) -> usize {
        self.padded_coefficient_len
    }
    /// Committed response digits `m * padded_coefficient_len * digit_slots`.
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
    /// Balanced base-16 digits `k_T` of one image coefficient.
    pub const fn image_digit_count(&self) -> usize {
        self.image_digit_count
    }
    /// Image digit-axis length: `k_T` rounded up to a power of two.
    pub const fn image_digit_slots(&self) -> usize {
        self.image_digit_slots
    }
    /// Image offset `B_T = 8 * (16^k_T - 1) / 15`: the digits store
    /// `T + B_T`, and `B_T` is the magnitude they enforce on `T`.
    pub const fn image_offset(&self) -> u128 {
        self.image_offset
    }
    /// Smallest `n` with
    /// `2^n >= image_digit_slots * padded_coefficient_len * C * n_A`.
    pub const fn image_table_log_len(&self) -> usize {
        self.image_table_log_len
    }
    /// Smallest `n` with `2^n >= padded_coefficient_len * C`: the prime left
    /// opening holds one padded ring element per column.
    pub const fn prime_table_log_len(&self) -> usize {
        self.prime_table_log_len
    }
}

impl LabiniusRootShape {
    /// Admit `proof_prime`, the characteristic of the proof field, and derive
    /// the enforced ranges and table lengths of the lowered relation.
    ///
    /// Three conditions are checked, in this order:
    ///
    /// 1. every nonzero difference of two fold challenges is a unit modulo
    ///    `proof_prime` ([`check_labinius_proof_prime_units`]);
    /// 2. the commitment rows cannot wrap
    ///    ([`super::LabiniusCommitmentLift::check_no_wrap`]);
    /// 3. the binary parity row cannot wrap: `H + 3 * B_Q + 2 * B_K < P`.
    ///
    /// Each clear-integer range is the narrowest signed bit width covering its
    /// honest bound, and conditions 2 and 3 use the enforced magnitudes.
    /// Primality of `proof_prime` is the caller's premise.
    pub fn derive_encoding(&self, proof_prime: u128) -> Result<LabiniusRootEncoding, AkitaError> {
        let challenge = self.profile.challenge_profile()?;
        check_labinius_proof_prime_units(proof_prime, &challenge)?;
        let a_carry = LabiniusSignedRange::covering(self.honest_a_carry_bound)?;
        self.commitment_lift(&challenge)
            .check_no_wrap(proof_prime, a_carry.magnitude())?;
        let quotient = LabiniusSignedRange::covering(self.honest_quotient_bound)?;
        let carry = LabiniusSignedRange::covering(self.honest_carry_bound)?;
        let parity_total = quotient
            .magnitude()
            .checked_mul(3)
            .zip(carry.magnitude().checked_mul(2))
            .and_then(|(quotient, carry)| {
                self.parity_residual_bound
                    .checked_add(quotient)?
                    .checked_add(carry)
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup("LaBinius parity no-wrap sum overflow".into())
            })?;
        if parity_total >= proof_prime {
            return Err(AkitaError::InvalidSetup(
                "LaBinius parity envelopes do not satisfy H + 3*B_Q + 2*B_K < P".into(),
            ));
        }
        let size_overflow =
            || AkitaError::InvalidSetup("LaBinius root digit witness size overflow".into());
        let rank = usize::try_from(self.rank_a).map_err(|_| size_overflow())?;
        let a_carry_len =
            checked::product([rank, self.commitment_degree]).ok_or_else(size_overflow)?;
        let padded_coefficient_len = self
            .commitment_degree
            .checked_next_power_of_two()
            .ok_or_else(size_overflow)?;
        let response_digit_count = self.response.digit_count();
        let response_digit_slots = response_digit_count
            .checked_next_power_of_two()
            .ok_or_else(size_overflow)?;
        let response_table_len = checked::product([
            self.ring_elements_per_column,
            padded_coefficient_len,
            response_digit_slots,
        ])
        .ok_or_else(size_overflow)?;
        let response_table_log_len =
            checked::ceil_log2(response_table_len).ok_or_else(size_overflow)?;
        let parity_quotient_len = self
            .scalar_degree
            .checked_sub(1)
            .ok_or_else(size_overflow)?;
        let (image_digit_count, image_offset) =
            labinius_image_digit_bound(self.profile.commitment_modulus().modulus());
        let image_digit_slots = image_digit_count
            .checked_next_power_of_two()
            .ok_or_else(size_overflow)?;
        let image_table_log_len = checked::product([
            image_digit_slots,
            padded_coefficient_len,
            self.fold_width,
            rank,
        ])
        .and_then(checked::ceil_log2)
        .ok_or_else(size_overflow)?;
        let prime_table_log_len = checked::product([padded_coefficient_len, self.fold_width])
            .and_then(checked::ceil_log2)
            .ok_or_else(size_overflow)?;
        Ok(LabiniusRootEncoding {
            response_interval: self.response.interval(),
            response_digit_count,
            response_digit_slots,
            quotient,
            carry,
            a_carry,
            a_carry_len,
            padded_coefficient_len,
            response_table_len,
            response_table_log_len,
            parity_quotient_len,
            parity_carry_len: self.scalar_degree,
            image_digit_count,
            image_digit_slots,
            image_offset,
            image_table_log_len,
            prime_table_log_len,
        })
    }
}

#[cfg(test)]
#[path = "root_encoding_tests.rs"]
mod tests;
