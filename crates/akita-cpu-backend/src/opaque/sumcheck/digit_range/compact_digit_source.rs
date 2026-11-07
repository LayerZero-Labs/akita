use akita_error::AkitaError;
use akita_params::{DigitRangePlan, FlatBooleanDomain};
use jolt_algebra::cfg_chunks_mut;
#[cfg(feature = "parallel")]
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::{Field, Ring};
use std::sync::Arc;

use crate::sources::packed_digits::PackedSignedDigits;

/// Digit pairs one task of [`CompactDigitSource::ordered_range_class_pairs`]
/// classifies.
const PAIRS_PER_TASK: usize = 4096;

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
            Self::ordered_range_class_pairs(&digits, class_count)?
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
    pub(super) fn prepare_class_indexed_leaf(&mut self) -> Result<(), AkitaError> {
        if self.ordered_range_class_pairs.is_empty() && !self.digits.is_empty() {
            self.ordered_range_class_pairs =
                Self::ordered_range_class_pairs(&self.digits, self.class_count)?;
        }
        Ok(())
    }

    /// Class pairs of consecutive digits, a trailing odd digit paired with
    /// padding, decoded and classified in parallel runs.
    fn ordered_range_class_pairs(
        digits: &PackedSignedDigits,
        class_count: u8,
    ) -> Result<Arc<[u16]>, AkitaError> {
        let view = digits.view();
        let mut pairs = vec![0u16; digits.len().div_ceil(2)];
        cfg_chunks_mut!(pairs, PAIRS_PER_TASK)
            .enumerate()
            .try_for_each(|(task, pairs)| {
                let start = 2 * task * PAIRS_PER_TASK;
                let mut decoded = [0i8; 2 * PAIRS_PER_TASK];
                let len = (2 * pairs.len()).min(digits.len() - start);
                view.decode_range(start, &mut decoded[..len])?;
                let class =
                    |digit| RangeImageClass::from_balanced_digit(digit, usize::from(class_count)).0;
                for (index, pair) in pairs.iter_mut().enumerate() {
                    let left = class(decoded[2 * index]);
                    let right = if 2 * index + 1 < len {
                        class(decoded[2 * index + 1])
                    } else {
                        RangeImageClass::PADDING.0
                    };
                    // An i8 digit's class is at most 127; even with a u8 class
                    // count, 127 * 255 + 127 fits u16.
                    *pair = u16::from(left) * u16::from(class_count) + u16::from(right);
                }
                Ok::<_, AkitaError>(())
            })?;
        Ok(pairs.into())
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

        source.prepare_class_indexed_leaf().unwrap();

        assert_eq!(source.pair_count(), 2);
        assert_eq!(source.ordered_pair_index(0), 1);
        assert_eq!(source.ordered_pair_index(1), 3 * source.class_count());
    }

    /// Parallel runs classify the same pairs as one serial pass, across
    /// several tasks and an odd trailing digit.
    #[test]
    fn parallel_pairs_match_a_serial_pass() {
        let class_count = 16u8;
        let len = 3 * 2 * PAIRS_PER_TASK + 101;
        let digits = (0..len)
            .map(|index| ((index * 7 + index / 5) % 32) as i8 - 16)
            .collect::<Vec<_>>();
        let packed = PackedSignedDigits::from_i8_digits_auto(digits.clone());
        let class = |digit| RangeImageClass::from_balanced_digit(digit, 16).0;
        let expected = digits
            .chunks(2)
            .map(|pair| {
                let right = pair
                    .get(1)
                    .map_or(RangeImageClass::PADDING.0, |&d| class(d));
                u16::from(class(pair[0])) * u16::from(class_count) + u16::from(right)
            })
            .collect::<Vec<_>>();
        let pairs = CompactDigitSource::ordered_range_class_pairs(&packed, class_count).unwrap();
        assert_eq!(&pairs[..], &expected[..]);
    }
}
