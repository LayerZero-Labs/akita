use super::*;

/// Value-weighted suffix-factor sums grouped by the original low index.
///
/// During merge-free sparse rounds, challenges change only the low-index
/// factor state. The high suffix contribution is fixed, so it can be summed
/// once and reused even after sparse rows begin to merge.
#[derive(Debug, Clone)]
pub(super) struct SparseSuffixSums<E: FieldCore> {
    values: Vec<E>,
    fold_weights: Vec<E>,
    width: usize,
}

impl<E: FieldCore> SparseSuffixSums<E> {
    pub(super) fn build<I>(
        rows: I,
        materialize_at: usize,
        palette_values: &[E],
        suffix_tables: &[Vec<E>],
    ) -> Self
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let low_count = 1usize << materialize_at;
        let width = suffix_tables.len();
        let palette_len = palette_values.len();
        let mut tagged = vec![E::zero(); low_count * palette_len * width];
        let low_mask = low_count - 1;
        for (index, tag) in rows {
            let low = index & low_mask;
            let suffix = index >> materialize_at;
            let start = (low * palette_len + tag) * width;
            for column in 0..width {
                tagged[start + column] += suffix_tables[column][suffix];
            }
        }
        let mut values = vec![E::zero(); low_count * width];
        for low in 0..low_count {
            for (tag, &palette_value) in palette_values.iter().enumerate() {
                let tagged_start = (low * palette_len + tag) * width;
                let output_start = low * width;
                for column in 0..width {
                    values[output_start + column] += palette_value * tagged[tagged_start + column];
                }
            }
        }
        Self {
            values,
            fold_weights: vec![E::one(); low_count],
            width,
        }
    }

    pub(super) fn low_count(&self) -> usize {
        self.values.len() / self.width
    }

    pub(super) fn bind(&mut self, round: usize, challenge: E) {
        let one_minus = E::one() - challenge;
        for low in 0..self.low_count() {
            let weight = if (low >> round) & 1 == 0 {
                one_minus
            } else {
                challenge
            };
            self.fold_weights[low] *= weight;
            for value in &mut self.values[low * self.width..(low + 1) * self.width] {
                *value *= weight;
            }
        }
    }

    pub(super) fn fold_weights(&self) -> &[E] {
        &self.fold_weights
    }

    pub(super) fn get(&self, original_low: usize) -> &[E] {
        let start = original_low * self.width;
        &self.values[start..start + self.width]
    }
}
