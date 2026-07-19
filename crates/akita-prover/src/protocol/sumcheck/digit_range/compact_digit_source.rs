use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_types::FlatBooleanDomain;
use std::sync::Arc;

/// Collision class of one balanced digit under `digit * (digit + 1)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RangeImageClass(u8);

impl RangeImageClass {
    pub(super) const PADDING: Self = Self(0);

    /// Derive the range-image class from a digit emitted by ring-switch decomposition.
    #[inline(always)]
    pub(super) fn from_balanced_digit(digit: i8, class_count: usize) -> Self {
        let class = if digit >= 0 {
            digit as u8
        } else {
            (-i16::from(digit) - 1) as u8
        };
        debug_assert!(usize::from(class) < class_count);
        Self(class)
    }

    #[inline(always)]
    pub(super) fn index(self) -> usize {
        usize::from(self.0)
    }

    #[inline(always)]
    pub(super) fn range_image<E: FieldCore + FromPrimitiveInt>(self) -> E {
        let class = i64::from(self.0);
        E::from_i64(class * (class + 1))
    }
}

/// Shared compact digits plus their checked flat-domain layout.
///
/// Ring-switch decomposition owns the balanced-digit invariant. This type checks only
/// layout and derives range-image classes without rescanning honest-prover digits.
#[derive(Clone)]
pub(super) struct CompactDigitSource {
    digits: Arc<[i8]>,
    domain: FlatBooleanDomain,
    class_count: usize,
}

impl CompactDigitSource {
    pub(super) fn new(
        digits: Arc<[i8]>,
        domain: FlatBooleanDomain,
        basis: usize,
    ) -> Result<Self, AkitaError> {
        if digits.len() != domain.live_len() {
            return Err(AkitaError::InvalidSize {
                expected: domain.live_len(),
                actual: digits.len(),
            });
        }
        Ok(Self {
            digits,
            domain,
            class_count: basis / 2,
        })
    }

    pub(super) fn digits(&self) -> Arc<[i8]> {
        Arc::clone(&self.digits)
    }

    pub(super) fn live_len(&self) -> usize {
        self.digits.len()
    }

    pub(super) fn domain_len(&self) -> usize {
        self.domain.domain_len()
    }

    #[inline(always)]
    pub(super) fn class_or_padding(&self, index: usize) -> RangeImageClass {
        self.digits
            .get(index)
            .copied()
            .map(|digit| RangeImageClass::from_balanced_digit(digit, self.class_count))
            .unwrap_or(RangeImageClass::PADDING)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_symmetry_maps_to_one_range_class() {
        for basis in [4, 8, 16, 32, 64] {
            let class_count = basis / 2;
            for class in 0..class_count {
                let positive = i8::try_from(class).unwrap();
                let negative = -positive - 1;
                assert_eq!(
                    RangeImageClass::from_balanced_digit(positive, class_count),
                    RangeImageClass::from_balanced_digit(negative, class_count)
                );
            }
        }
    }
}
