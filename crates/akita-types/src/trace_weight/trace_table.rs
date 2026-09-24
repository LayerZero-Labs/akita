//! Prover-side trace table: sparse columns for `K = 1`, dense flat slice for `K > 1`.

use jolt_field::Field;

#[inline]
fn fold_pair<E: Field>(a: E, b: E, r: E) -> E {
    a + r * (b - a)
}

/// One active opening-digit column of a sparse (`K = 1`) trace table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceSparseColumn<E: Field> {
    pub col: usize,
    pub values: Vec<E>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceSparseTable<E: Field> {
    columns: Vec<TraceSparseColumn<E>>,
    live_x_cols: usize,
    y_len: usize,
}

impl<E: Field> TraceSparseTable<E> {
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
pub enum TraceTable<E: Field> {
    FieldSparse(TraceSparseTable<E>),
    RingDense(Vec<E>),
}

impl<E: Field> TraceTable<E> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fp32;
    use jolt_field::{Ring, Zero};

    type F = Fp32<251>;

    #[test]
    fn ring_dense_fold_x_preserves_x_outer_layout() {
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
