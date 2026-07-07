//! Prover-side trace table: sparse columns for `K = 1`, dense flat slice for `K > 1`.

use akita_field::{AkitaError, FieldCore};

#[inline]
fn fold_pair<E: FieldCore>(a: E, b: E, r: E) -> E {
    a + r * (b - a)
}

#[inline]
fn fold_quad<E: FieldCore>(v00: E, v10: E, v01: E, v11: E, r0: E, r1: E) -> E {
    let x0 = fold_pair(v00, v10, r0);
    let x1 = fold_pair(v01, v11, r0);
    fold_pair(x0, x1, r1)
}

/// One active opening-digit column of a sparse (`K = 1`) trace table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceSparseColumn<E: FieldCore> {
    pub col: usize,
    pub values: Vec<E>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceSparseTable<E: FieldCore> {
    columns: Vec<TraceSparseColumn<E>>,
    live_x_cols: usize,
    y_len: usize,
}

impl<E: FieldCore> TraceSparseTable<E> {
    fn new(mut columns: Vec<TraceSparseColumn<E>>, live_x_cols: usize, y_len: usize) -> Self {
        columns.retain(|column| column.col < live_x_cols);
        columns.sort_by_key(|column| column.col);

        let mut merged: Vec<TraceSparseColumn<E>> = Vec::with_capacity(columns.len());
        for mut column in columns {
            debug_assert_eq!(column.values.len(), y_len);
            if let Some(last) = merged.last_mut() {
                if last.col == column.col {
                    for (dst, src) in last.values.iter_mut().zip(column.values.drain(..)) {
                        *dst += src;
                    }
                    continue;
                }
            }
            merged.push(column);
        }

        Self {
            columns: merged,
            live_x_cols,
            y_len,
        }
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> E {
        match self.columns.binary_search_by_key(&x, |column| column.col) {
            Ok(idx) => self.columns[idx]
                .values
                .get(y)
                .copied()
                .unwrap_or_else(E::zero),
            Err(_) => E::zero(),
        }
    }

    fn fold_y(&mut self, r: E) {
        for column in &mut self.columns {
            let half = column.values.len() / 2;
            for i in 0..half {
                let a = column.values[2 * i];
                let b = column.values[2 * i + 1];
                column.values[i] = fold_pair(a, b, r);
            }
            column.values.truncate(half);
        }
        self.y_len /= 2;
    }

    fn fold_y2(&mut self, r0: E, r1: E) {
        let next_y_len = self.y_len >> 2;
        for column in &mut self.columns {
            for quad_y in 0..next_y_len {
                let base = 4 * quad_y;
                column.values[quad_y] = fold_quad(
                    column.values[base],
                    column.values[base + 1],
                    column.values[base + 2],
                    column.values[base + 3],
                    r0,
                    r1,
                );
            }
            column.values.truncate(next_y_len);
        }
        self.y_len = next_y_len;
    }

    fn fold_x(&mut self, r: E) {
        let live_x_cols = self.live_x_cols;
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let y_len = self.y_len;
        let mut folded = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let next_col = column.col / 2;
            if next_col >= next_live_x_cols {
                continue;
            }
            let scale = if column.col % 2 == 0 { E::one() - r } else { r };
            let values = column.values.iter().map(|&value| scale * value).collect();
            folded.push(TraceSparseColumn {
                col: next_col,
                values,
            });
        }
        *self = Self::new(folded, next_live_x_cols, y_len);
    }

    fn materialize_dense(&self) -> Vec<E> {
        let mut dense = vec![E::zero(); self.live_x_cols * self.y_len];
        for column in &self.columns {
            let dst = column.col * self.y_len;
            for (y, value) in column.values.iter().enumerate() {
                dense[dst + y] += *value;
            }
        }
        dense
    }
}

/// Trace addend folded alongside the stage-2 witness table.
///
/// `FieldSparse` is the production `K = 1` representation (active opening-digit columns only).
/// `RingDense` is the flat `live_x_cols · y_len` table used for `K > 1` ring block weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceTable<E: FieldCore> {
    FieldSparse(TraceSparseTable<E>),
    RingDense(Vec<E>),
}

impl<E: FieldCore> TraceTable<E> {
    pub fn field_sparse(
        columns: Vec<TraceSparseColumn<E>>,
        live_x_cols: usize,
        y_len: usize,
    ) -> Self {
        Self::FieldSparse(TraceSparseTable::new(columns, live_x_cols, y_len))
    }

    pub fn ring_dense(dense: Vec<E>) -> Self {
        Self::RingDense(dense)
    }

    /// Extract the flat `col ⊗ ring` table backing a dense trace table.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] for a sparse (`K = 1`) table.
    pub fn into_ring_dense(self) -> Result<Vec<E>, AkitaError> {
        match self {
            Self::RingDense(dense) => Ok(dense),
            Self::FieldSparse(_) => Err(AkitaError::InvalidProof),
        }
    }

    pub fn materialize_dense(&self, live_x_cols: usize, y_len: usize) -> Vec<E> {
        match self {
            Self::FieldSparse(table) => {
                debug_assert_eq!(table.live_x_cols, live_x_cols);
                debug_assert_eq!(table.y_len, y_len);
                table.materialize_dense()
            }
            Self::RingDense(dense) => dense.clone(),
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, y_len: usize) -> E {
        match self {
            Self::RingDense(dense) => dense.get(x * y_len + y).copied().unwrap_or_else(E::zero),
            Self::FieldSparse(table) => {
                debug_assert_eq!(table.y_len, y_len);
                table.get(x, y)
            }
        }
    }

    #[inline]
    pub fn pair_at_columns(&self, x0: usize, x1: usize, y: usize, y_len: usize) -> (E, E) {
        (self.get(x0, y, y_len), self.get(x1, y, y_len))
    }

    #[inline]
    pub fn pair_flat(&self, idx0: usize, idx1: usize, y_len: usize) -> (E, E) {
        (
            self.get(idx0 / y_len, idx0 % y_len, y_len),
            self.get(idx1 / y_len, idx1 % y_len, y_len),
        )
    }

    pub fn quad_at(&self, x: usize, base: usize, y_len: usize) -> [E; 4] {
        std::array::from_fn(|offset| self.get(x, base + offset, y_len))
    }

    pub fn validate_len(&self, witness_len: usize) -> Result<(), AkitaError> {
        match self {
            Self::RingDense(dense) => {
                if dense.len() != witness_len {
                    return Err(AkitaError::InvalidSize {
                        expected: witness_len,
                        actual: dense.len(),
                    });
                }
            }
            Self::FieldSparse(table) => {
                if table.live_x_cols * table.y_len > witness_len {
                    return Err(AkitaError::InvalidSize {
                        expected: witness_len,
                        actual: table.live_x_cols * table.y_len,
                    });
                }
                for column in &table.columns {
                    if column.col >= table.live_x_cols {
                        return Err(AkitaError::InvalidInput(
                            "sparse trace column index out of live range".to_string(),
                        ));
                    }
                    if column.values.len() != table.y_len {
                        return Err(AkitaError::InvalidSize {
                            expected: table.y_len,
                            actual: column.values.len(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    pub fn fold_y(&mut self, r: E) {
        match self {
            Self::RingDense(dense) => {
                let half = dense.len() / 2;
                for i in 0..half {
                    dense[i] = fold_pair(dense[2 * i], dense[2 * i + 1], r);
                }
                dense.truncate(half);
            }
            Self::FieldSparse(table) => table.fold_y(r),
        }
    }

    pub fn fold_y2(&mut self, live_x_cols: usize, y_len: usize, r0: E, r1: E) {
        match self {
            Self::RingDense(dense) => {
                let next_y_len = y_len >> 2;
                let mut out = vec![E::zero(); live_x_cols * next_y_len];
                for x in 0..live_x_cols {
                    let src_start = x * y_len;
                    let dst_start = x * next_y_len;
                    for quad_y in 0..next_y_len {
                        let base = src_start + 4 * quad_y;
                        out[dst_start + quad_y] = fold_quad(
                            dense[base],
                            dense[base + 1],
                            dense[base + 2],
                            dense[base + 3],
                            r0,
                            r1,
                        );
                    }
                }
                *dense = out;
            }
            Self::FieldSparse(table) => {
                debug_assert_eq!(table.live_x_cols, live_x_cols);
                debug_assert_eq!(table.y_len, y_len);
                table.fold_y2(r0, r1);
            }
        }
    }

    pub fn fold_x(&mut self, live_x_cols: usize, y_len: usize, r: E) {
        match self {
            Self::RingDense(dense) => {
                let next_live_x_cols = live_x_cols.div_ceil(2);
                let mut out = vec![E::zero(); y_len * next_live_x_cols];
                for pair_x in 0..next_live_x_cols {
                    let left = 2 * pair_x;
                    let dst_start = pair_x * y_len;
                    let left_start = left * y_len;
                    let right_start = (left + 1) * y_len;
                    for y in 0..y_len {
                        let a = dense[left_start + y];
                        let b = if left + 1 < live_x_cols {
                            dense[right_start + y]
                        } else {
                            E::zero()
                        };
                        out[dst_start + y] = fold_pair(a, b, r);
                    }
                }
                *dense = out;
            }
            Self::FieldSparse(table) => {
                debug_assert_eq!(table.live_x_cols, live_x_cols);
                debug_assert_eq!(table.y_len, y_len);
                table.fold_x(r);
            }
        }
    }

    pub fn fold_for_w_update(
        &mut self,
        live_x_cols: usize,
        y_len: usize,
        r: E,
        folding_x_round: bool,
    ) {
        if folding_x_round {
            self.fold_x(live_x_cols, y_len, r);
        } else {
            self.fold_y(r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp32;

    type F = Fp32<251>;

    #[test]
    fn ring_dense_fold_x_preserves_column_major_layout() {
        let dense = (0..12).map(F::from_u64).collect();
        let mut table = TraceTable::ring_dense(dense);
        let r = F::from_u64(3);

        table.fold_x(3, 4, r);

        for y in 0..4 {
            let y_value = F::from_u64(y as u64);
            assert_eq!(
                table.get(0, y, 4),
                fold_pair(y_value, F::from_u64((4 + y) as u64), r)
            );
            assert_eq!(
                table.get(1, y, 4),
                fold_pair(F::from_u64((8 + y) as u64), F::zero(), r)
            );
        }
    }
}
