use akita_error::AkitaError;
use akita_params::{DigitRangePlan, FlatBooleanDomain};
use jolt_field::{solinas::parallel::*, Field, Ring};
use std::sync::Arc;

use crate::sources::packed_digits::PackedSignedDigits;

/// Collision class of one balanced digit under `digit * (digit + 1)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RangeImageClass(pub(super) u8);

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
    #[cfg(test)]
    pub(super) fn index(self) -> usize {
        usize::from(self.0)
    }

    #[inline(always)]
    pub(super) fn range_image<E: Field + Ring>(self) -> E {
        let class = i64::from(self.0);
        E::from_i64(class * (class + 1))
    }
}

/// Shared compact digits plus their checked flat-domain layout.
///
/// Ring-switch decomposition owns the balanced-digit invariant. This type checks only
/// layout and derives range-image classes without rescanning honest-prover digits.
#[derive(Clone)]
pub(crate) struct CompactDigitSource {
    digits: PackedSignedDigits,
    ordered_range_class_pairs: Arc<[u16]>,
    domain: FlatBooleanDomain,
    class_count: u8,
}

impl CompactDigitSource {
    pub(super) fn new(
        digits: PackedSignedDigits,
        domain: FlatBooleanDomain,
        plan: DigitRangePlan,
    ) -> Result<Self, AkitaError> {
        if digits.len() != domain.live_len() {
            return Err(AkitaError::InvalidSize {
                expected: domain.live_len(),
                actual: digits.len(),
            });
        }
        // DigitRangePlan admits log bases 2..=6, so the class count is 2..=32.
        let class_count = 1u8 << (plan.log_basis() - 1);
        let ordered_range_class_pairs = if !plan.product_stage_arities().is_empty() {
            Self::ordered_range_class_pairs(&digits, class_count)
        } else {
            Arc::from([])
        };
        Ok(Self {
            digits,
            ordered_range_class_pairs,
            domain,
            class_count,
        })
    }

    /// Prepare the ordered class pairs consumed by the class-indexed leaf.
    ///
    /// Low-basis L-infinity routes use the direct leaf and deliberately avoid
    /// this allocation. A physical L2 proof uses the class-indexed fused leaf
    /// even when the range plan has no product-prefix stage, so it requests the
    /// same pairs explicitly before entering that leaf.
    pub(super) fn prepare_class_indexed_leaf(&mut self) {
        if self.ordered_range_class_pairs.is_empty() && !self.digits.is_empty() {
            self.ordered_range_class_pairs =
                Self::ordered_range_class_pairs(&self.digits, self.class_count);
        }
    }

    fn ordered_range_class_pairs(digits: &PackedSignedDigits, class_count: u8) -> Arc<[u16]> {
        /// Pairs per task: a whole number of packed blocks, long enough that
        /// a task outweighs its scheduling.
        const TASK_PAIRS: usize = 1 << 14;
        let view = digits.view();
        let class_of =
            |digit| RangeImageClass::from_balanced_digit(digit, usize::from(class_count)).0;
        let mut pairs = vec![0u16; view.len().div_ceil(2)];
        cfg_chunks_mut!(pairs, TASK_PAIRS)
            .enumerate()
            .for_each(|(task, pairs)| {
                let first = 2 * task * TASK_PAIRS;
                let end = view.len().min(first + 2 * pairs.len());
                let mut digits = view
                    .slice(first..end)
                    .expect("a task's digits lie inside the witness")
                    .iter();
                for pair in pairs {
                    let left = digits.next().map_or(RangeImageClass::PADDING.0, class_of);
                    let right = digits.next().map_or(RangeImageClass::PADDING.0, class_of);
                    // An i8 digit's class is at most 127; even with a u8 class
                    // count, 127 * 255 + 127 fits u16.
                    *pair = u16::from(left) * u16::from(class_count) + u16::from(right);
                }
            });
        Arc::from(pairs)
    }

    pub(super) fn digits(&self) -> PackedSignedDigits {
        self.digits.clone()
    }

    pub(super) fn live_len(&self) -> usize {
        self.digits.len()
    }

    pub(super) fn domain_len(&self) -> usize {
        self.domain.domain_len()
    }

    pub(super) fn class_count(&self) -> usize {
        usize::from(self.class_count)
    }

    pub(super) fn classes(&self) -> std::ops::Range<u8> {
        0..self.class_count
    }

    pub(super) fn pair_count(&self) -> usize {
        self.ordered_range_class_pairs.len()
    }

    pub(super) fn quartet_count(&self) -> usize {
        self.pair_count().div_ceil(2)
    }

    #[inline(always)]
    pub(super) fn ordered_pair_index(&self, pair_index: usize) -> usize {
        usize::from(self.ordered_range_class_pairs[pair_index])
    }

    #[inline(always)]
    pub(super) fn ordered_pair_indices_for_quartet(&self, quartet_index: usize) -> (usize, usize) {
        let first_pair_index = 2 * quartet_index;
        let first = usize::from(self.ordered_range_class_pairs[first_pair_index]);
        let second = self
            .ordered_range_class_pairs
            .get(first_pair_index + 1)
            .copied()
            .map(usize::from)
            .unwrap_or(0);
        (first, second)
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

    #[test]
    fn compact_ordered_pairs_include_odd_prefix_padding() {
        let digits = PackedSignedDigits::from_i8_digits_auto(vec![0, -2, 3]);
        let source = CompactDigitSource::new(
            digits,
            FlatBooleanDomain::new(3, 2).unwrap(),
            DigitRangePlan::new(16).unwrap(),
        )
        .unwrap();
        assert_eq!(source.pair_count(), 2);
        assert_eq!(source.ordered_pair_index(0), 1);
        assert_eq!(source.ordered_pair_index(1), 3 * source.class_count());
    }

    #[test]
    fn low_basis_source_prepares_pairs_for_the_fused_l2_leaf_on_demand() {
        let digits = PackedSignedDigits::from_i8_digits_auto(vec![0, -2, 3]);
        let mut source = CompactDigitSource::new(
            digits,
            FlatBooleanDomain::new(3, 2).unwrap(),
            DigitRangePlan::new(8).unwrap(),
        )
        .unwrap();
        assert_eq!(source.pair_count(), 0);

        source.prepare_class_indexed_leaf();

        assert_eq!(source.pair_count(), 2);
        assert_eq!(source.ordered_pair_index(0), 1);
        assert_eq!(source.ordered_pair_index(1), 3 * source.class_count());
    }
}
