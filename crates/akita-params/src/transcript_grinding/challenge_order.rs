//! Exact challenge-set cardinality and checked grinding-target arithmetic.

use super::{push_u32, MAX_GRINDING_BITS, TRANSCRIPT_SECURITY_BITS};
use akita_error::AkitaError;

const CHALLENGE_ORDER_COMPARISON_CAP: GrindingUint = GrindingUint([0, 0, 0, 1]);

/// Exact cardinality of a challenge set used when pricing transcript grinding.
///
/// A field-backed order records the base modulus and extension degree, so its
/// cardinality is exactly `modulus^extension_degree`. A full-capacity order is
/// available for protocols whose challenge set has exactly a power-of-two
/// cardinality. The fixed-width internal comparison is capped at 2^192, which
/// exceeds every loss factor multiplied by Akita's 128-bit security target.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChallengeFieldOrder {
    nominal_capacity_bits: u32,
    field_modulus: u128,
    extension_degree: u32,
    exact_cardinality: GrindingUint,
}

impl ChallengeFieldOrder {
    /// Build the exact extension-field order from a prime-field modulus.
    pub fn from_field(
        modulus_bits: u32,
        extension_degree: usize,
        field_modulus: u128,
    ) -> Result<Self, AkitaError> {
        let actual_modulus_bits = u128::BITS - field_modulus.leading_zeros();
        if field_modulus <= 1 || actual_modulus_bits != modulus_bits {
            return Err(AkitaError::InvalidSetup(
                "challenge modulus does not match its declared bit width".into(),
            ));
        }
        let extension_degree = u32::try_from(extension_degree).map_err(|_| {
            AkitaError::InvalidSetup("challenge extension degree exceeds u32".into())
        })?;
        if extension_degree == 0 || !extension_degree.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "challenge extension degree must be a nonzero power of two".into(),
            ));
        }
        let nominal_capacity_bits = modulus_bits
            .checked_mul(extension_degree)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge capacity overflow".into()))?;
        let exact_cardinality = GrindingUint::pow_capped(
            GrindingUint::from_u128(field_modulus),
            extension_degree,
            CHALLENGE_ORDER_COMPARISON_CAP,
        );
        Ok(Self {
            nominal_capacity_bits,
            field_modulus,
            extension_degree,
            exact_cardinality,
        })
    }

    /// Build a challenge set whose cardinality is exactly `2^capacity_bits`.
    pub fn from_full_capacity(capacity_bits: u32) -> Result<Self, AkitaError> {
        if capacity_bits == 0 {
            return Err(AkitaError::InvalidSetup(
                "grinding nominal capacity must be nonzero".into(),
            ));
        }
        let exact_cardinality = if capacity_bits >= 192 {
            CHALLENGE_ORDER_COMPARISON_CAP
        } else {
            GrindingUint::power_of_two(capacity_bits)
        };
        Ok(Self {
            nominal_capacity_bits: capacity_bits,
            field_modulus: 0,
            extension_degree: 0,
            exact_cardinality,
        })
    }

    /// Nominal bit width for diagnostics only; grinding uses exact cardinality.
    #[must_use]
    pub const fn nominal_capacity_bits(self) -> u32 {
        self.nominal_capacity_bits
    }

    pub(super) fn append_canonical_bytes(self, out: &mut Vec<u8>) {
        if let Some(base_modulus_bits) = self
            .nominal_capacity_bits
            .checked_div(self.extension_degree)
        {
            out.push(1);
            push_u32(out, base_modulus_bits);
            push_u32(out, self.extension_degree);
            out.extend_from_slice(&self.field_modulus.to_le_bytes());
        } else {
            out.push(0);
            push_u32(out, self.nominal_capacity_bits);
        }
    }

    fn satisfies_target(self, loss_factor: u64, grind_bits: u8) -> bool {
        let target =
            GrindingUint::from_u64_shifted(loss_factor, u32::from(TRANSCRIPT_SECURITY_BITS));
        self.exact_cardinality
            .cmp(&target.ceil_shift_right(u32::from(grind_bits)))
            .is_ge()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GrindingUint([u64; 4]);

impl GrindingUint {
    const ZERO: Self = Self([0; 4]);
    const ONE: Self = Self([1, 0, 0, 0]);

    fn from_u128(value: u128) -> Self {
        Self([value as u64, (value >> 64) as u64, 0, 0])
    }

    fn from_u64_shifted(value: u64, shift: u32) -> Self {
        let mut limbs = [0; 4];
        let word_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;
        if word_shift >= limbs.len() {
            return Self(limbs);
        }
        limbs[word_shift] = value << bit_shift;
        if bit_shift != 0 && word_shift + 1 < limbs.len() {
            limbs[word_shift + 1] = value >> (64 - bit_shift);
        }
        Self(limbs)
    }

    fn power_of_two(bit: u32) -> Self {
        let mut limbs = [0; 4];
        let bit = bit as usize;
        limbs[bit / 64] = 1u64 << (bit % 64);
        Self(limbs)
    }

    fn cmp(self, other: &Self) -> std::cmp::Ordering {
        for index in (0..self.0.len()).rev() {
            match self.0[index].cmp(&other.0[index]) {
                std::cmp::Ordering::Equal => {}
                ordering => return ordering,
            }
        }
        std::cmp::Ordering::Equal
    }

    fn add_capped(self, other: Self, cap: Self) -> Self {
        let mut result = [0u64; 4];
        let mut carry = 0u128;
        for (index, value) in result.iter_mut().enumerate() {
            let sum = u128::from(self.0[index]) + u128::from(other.0[index]) + carry;
            *value = sum as u64;
            carry = sum >> 64;
        }
        if carry != 0 {
            cap
        } else {
            let result = Self(result);
            if result.cmp(&cap).is_ge() {
                cap
            } else {
                result
            }
        }
    }

    fn mul_capped(self, other: Self, cap: Self) -> Self {
        let mut product = Self::ZERO;
        let mut addend = self;
        for index in 0..256 {
            if other.0[index / 64] & (1u64 << (index % 64)) != 0 {
                product = product.add_capped(addend, cap);
                if product == cap {
                    return cap;
                }
            }
            if index != 255 {
                addend = addend.add_capped(addend, cap);
            }
        }
        product
    }

    fn pow_capped(mut base: Self, mut exponent: u32, cap: Self) -> Self {
        let mut result = Self::ONE;
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.mul_capped(base, cap);
                if result == cap {
                    return cap;
                }
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.mul_capped(base, cap);
            }
        }
        result
    }

    fn ceil_shift_right(self, shift: u32) -> Self {
        if shift == 0 {
            return self;
        }
        let word_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;
        let mut result = [0u64; 4];
        if word_shift < self.0.len() {
            for (destination, output) in result
                .iter_mut()
                .enumerate()
                .take(self.0.len() - word_shift)
            {
                let source = destination + word_shift;
                *output = self.0[source] >> bit_shift;
                if bit_shift != 0 && source + 1 < self.0.len() {
                    *output |= self.0[source + 1] << (64 - bit_shift);
                }
            }
        }
        let mut rounded = Self(result);
        let has_remainder = (0..word_shift.min(self.0.len())).any(|index| self.0[index] != 0)
            || (bit_shift != 0
                && word_shift < self.0.len()
                && self.0[word_shift] & ((1u64 << bit_shift) - 1) != 0);
        if has_remainder {
            rounded = rounded.add_capped(Self::ONE, CHALLENGE_ORDER_COMPARISON_CAP);
        }
        rounded
    }
}

/// Assign the least public proof-of-work target satisfying the exact challenge-set bound.
pub fn grind_bits_for_loss(
    loss_factor: u64,
    challenge_order: ChallengeFieldOrder,
) -> Result<u8, AkitaError> {
    if loss_factor == 0 {
        return Err(AkitaError::InvalidSetup(
            "proof-of-work loss factor must be nonzero".into(),
        ));
    }
    for grind_bits in 0..=MAX_GRINDING_BITS {
        if challenge_order.satisfies_target(loss_factor, grind_bits) {
            return Ok(grind_bits);
        }
    }
    Err(AkitaError::InvalidSetup(format!(
        "grinding target exceeds supported maximum {MAX_GRINDING_BITS}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;

    fn oracle_grind_bits(loss: u64, modulus: u128, degree: u32) -> Option<u8> {
        let cardinality = BigUint::from(modulus).pow(degree);
        let target = BigUint::from(loss) << usize::from(TRANSCRIPT_SECURITY_BITS);
        (0..=MAX_GRINDING_BITS).find(|&bits| target <= (&cardinality << usize::from(bits)))
    }

    #[test]
    fn production_orders_price_the_last_supported_targets_exactly() {
        let fields = [
            (128, 1, 340_282_366_920_938_463_463_374_607_427_473_266_697),
            (64, 2, 18_446_744_073_709_551_557),
            (32, 4, 4_294_967_197),
        ];
        let losses = [
            (1u64 << 24) - 1,
            1u64 << 24,
            (1u64 << 24) + 1,
            (1u64 << 25) - 1,
            1u64 << 25,
            (1u64 << 25) + 1,
        ];
        for (modulus_bits, degree, modulus) in fields {
            let order = ChallengeFieldOrder::from_field(modulus_bits, degree, modulus).unwrap();
            for loss in losses {
                let expected = oracle_grind_bits(loss, modulus, degree as u32);
                assert_eq!(
                    grind_bits_for_loss(loss, order).ok(),
                    expected,
                    "modulus_bits={modulus_bits}, degree={degree}, loss={loss}"
                );
            }
        }
    }

    #[test]
    fn cardinality_comparison_saturates_at_two_to_the_192() {
        let below_modulus = (1u128 << 96) - 1;
        let below = ChallengeFieldOrder::from_field(96, 2, below_modulus).unwrap();
        let expected = BigUint::from(below_modulus).pow(2).to_u64_digits();
        assert_eq!(
            &below.exact_cardinality.0[..expected.len()],
            expected.as_slice()
        );
        assert!(below.exact_cardinality.0[expected.len()..]
            .iter()
            .all(|&limb| limb == 0));
        assert_ne!(below.exact_cardinality, CHALLENGE_ORDER_COMPARISON_CAP);
        assert_eq!(grind_bits_for_loss(u64::MAX, below).unwrap(), 0);

        let above_modulus = (1u128 << 96) + 1;
        let above = ChallengeFieldOrder::from_field(97, 2, above_modulus).unwrap();
        assert!(BigUint::from(above_modulus).pow(2) > BigUint::from(1u8) << 192usize);
        assert_eq!(above.exact_cardinality, CHALLENGE_ORDER_COMPARISON_CAP);
        assert_eq!(grind_bits_for_loss(u64::MAX, above).unwrap(), 0);
    }
}
