use super::*;

/// Suffix-factor sums grouped by the original low index and witness value tag.
///
/// During merge-free sparse rounds, challenges change only the low-index
/// factor state and the small witness palette. The high suffix contribution is
/// fixed, so it can be summed once and reused until the first merging fold.
#[derive(Debug, Clone)]
pub(super) struct MergeFreeSuffixSums<E: FieldCore> {
    values: Vec<E>,
    low_count: usize,
    palette_len: usize,
    width: usize,
}

impl<E: FieldCore> MergeFreeSuffixSums<E> {
    pub(super) fn build<I>(
        rows: I,
        materialize_at: usize,
        palette_len: usize,
        suffix_tables: &[Vec<E>],
    ) -> Self
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let low_count = 1usize << materialize_at;
        let width = suffix_tables.len();
        let mut values = vec![E::zero(); low_count * palette_len * width];
        let low_mask = low_count - 1;
        for (index, tag) in rows {
            let low = index & low_mask;
            let suffix = index >> materialize_at;
            let start = (low * palette_len + tag) * width;
            for column in 0..width {
                values[start + column] += suffix_tables[column][suffix];
            }
        }
        Self {
            values,
            low_count,
            palette_len,
            width,
        }
    }

    pub(super) fn low_count(&self) -> usize {
        self.low_count
    }

    pub(super) fn palette_len(&self) -> usize {
        self.palette_len
    }

    pub(super) fn get(&self, original_low: usize, tag: usize) -> &[E] {
        let start = (original_low * self.palette_len + tag) * self.width;
        &self.values[start..start + self.width]
    }
}
