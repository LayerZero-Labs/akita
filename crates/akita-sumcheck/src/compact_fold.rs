//! Precomputed lookup table for folding pairs of small integer values.
//!
//! Used by Akita stage sumchecks for compact-witness folding.

use akita_algebra::fields::HasUnreducedOps;
use akita_field::{FieldCore, FromSmallInt};

/// Precomputed lookup table for folding pairs of small integer values at a
/// fixed challenge `r`.
///
/// This is useful for the round-0 compact tables in Hachi's stage-1 and
/// stage-2 sumchecks: the table entries are small integers, the fold formula is
/// always `left + r * (right - left)`, and the set of possible `(left, right)`
/// pairs is tiny.
pub struct CompactPairFoldLut<E: FieldCore> {
    min_value: i16,
    value_to_index: Vec<usize>,
    pair_values: Vec<E>,
    num_values: usize,
}

impl<E: FieldCore + FromSmallInt + HasUnreducedOps> CompactPairFoldLut<E> {
    /// Build a lookup table from an explicit set of allowed small integer
    /// values.
    ///
    /// # Panics
    ///
    /// Panics if `allowed_values` is empty.
    pub fn from_allowed_values(allowed_values: &[i16], r: E) -> Self {
        assert!(
            !allowed_values.is_empty(),
            "allowed_values must be non-empty"
        );
        let min_value = *allowed_values.iter().min().expect("non-empty");
        let max_value = *allowed_values.iter().max().expect("non-empty");
        let mut value_to_index = vec![usize::MAX; (max_value - min_value + 1) as usize];
        for (idx, &value) in allowed_values.iter().enumerate() {
            let offset = (value - min_value) as usize;
            debug_assert_eq!(
                value_to_index[offset],
                usize::MAX,
                "allowed_values must be unique"
            );
            value_to_index[offset] = idx;
        }

        let num_values = allowed_values.len();
        let mut pair_values = Vec::with_capacity(num_values * num_values);
        for &left in allowed_values {
            let left_field = E::from_i64(left as i64);
            for &right in allowed_values {
                let delta = i64::from(right) - i64::from(left);
                let delta_abs = delta.unsigned_abs();
                let r_delta = E::reduce_mul_u64_accum(r.mul_u64_unreduced(delta_abs));
                pair_values.push(if delta < 0 {
                    left_field - r_delta
                } else {
                    left_field + r_delta
                });
            }
        }

        Self {
            min_value,
            value_to_index,
            pair_values,
            num_values,
        }
    }

    /// Build a lookup table for every integer in `min_value..=max_value`.
    ///
    /// # Panics
    ///
    /// Panics if `min_value > max_value`.
    pub fn from_contiguous_range(min_value: i16, max_value: i16, r: E) -> Self {
        assert!(min_value <= max_value, "invalid compact fold range");
        let allowed_values: Vec<i16> = (min_value..=max_value).collect();
        Self::from_allowed_values(&allowed_values, r)
    }
}

impl<E: FieldCore> CompactPairFoldLut<E> {
    #[inline]
    fn index_of(&self, value: i16) -> usize {
        let offset = (value - self.min_value) as usize;
        let idx = self.value_to_index[offset];
        debug_assert_ne!(idx, usize::MAX, "value missing from compact fold LUT");
        idx
    }

    /// Fold the pair `(left, right)` at the challenge used to construct this
    /// table.
    #[inline]
    pub fn fold(&self, left: i16, right: i16) -> E {
        let left_idx = self.index_of(left);
        let right_idx = self.index_of(right);
        self.pair_values[left_idx * self.num_values + right_idx]
    }
}
