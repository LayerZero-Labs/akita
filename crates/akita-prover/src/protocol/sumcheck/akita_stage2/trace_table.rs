use super::*;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SparseTraceColumn<E: FieldCore> {
    pub col: usize,
    pub values: Vec<E>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SparseTraceTable<E: FieldCore> {
    columns: Vec<SparseTraceColumn<E>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TraceTable<E: FieldCore> {
    Dense(Vec<E>),
    Sparse(SparseTraceTable<E>),
}

impl<E: FieldCore> SparseTraceTable<E> {
    pub(crate) fn new(
        mut columns: Vec<SparseTraceColumn<E>>,
        live_x_cols: usize,
        y_len: usize,
    ) -> Self {
        columns.retain(|column| column.col < live_x_cols);
        columns.sort_by_key(|column| column.col);

        let mut merged: Vec<SparseTraceColumn<E>> = Vec::with_capacity(columns.len());
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

        Self { columns: merged }
    }

    #[inline]
    pub(crate) fn get(&self, x: usize, y: usize) -> E {
        match self.columns.binary_search_by_key(&x, |column| column.col) {
            Ok(idx) => self.columns[idx]
                .values
                .get(y)
                .copied()
                .unwrap_or_else(E::zero),
            Err(_) => E::zero(),
        }
    }

    pub(crate) fn fold_y(&mut self, r: E) {
        for column in &mut self.columns {
            let half = column.values.len() / 2;
            for i in 0..half {
                let a = column.values[2 * i];
                let b = column.values[2 * i + 1];
                column.values[i] = a + r * (b - a);
            }
            column.values.truncate(half);
        }
    }

    pub(crate) fn fold_y2(&mut self, r0: E, r1: E) {
        for column in &mut self.columns {
            let next_y_len = column.values.len() >> 2;
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
    }

    pub(crate) fn fold_x(&mut self, live_x_cols: usize, r: E) {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut folded: Vec<SparseTraceColumn<E>> = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let next_col = column.col / 2;
            if next_col >= next_live_x_cols {
                continue;
            }
            let scale = if column.col % 2 == 0 { E::one() - r } else { r };
            let values = column.values.iter().map(|&value| scale * value).collect();
            folded.push(SparseTraceColumn {
                col: next_col,
                values,
            });
        }
        let y_len = self.columns.first().map_or(0, |column| column.values.len());
        *self = Self::new(folded, next_live_x_cols, y_len);
    }
}

impl<E: FieldCore> TraceTable<E> {
    #[inline]
    pub(crate) fn dense(trace: Vec<E>) -> Self {
        Self::Dense(trace)
    }

    #[inline]
    pub(crate) fn sparse(
        columns: Vec<SparseTraceColumn<E>>,
        live_x_cols: usize,
        y_len: usize,
    ) -> Self {
        Self::Sparse(SparseTraceTable::new(columns, live_x_cols, y_len))
    }

    #[inline]
    pub(crate) fn get(&self, x: usize, y: usize, y_len: usize) -> E {
        match self {
            Self::Dense(trace) => trace.get(x * y_len + y).copied().unwrap_or_else(E::zero),
            Self::Sparse(trace) => trace.get(x, y),
        }
    }

    #[inline]
    pub(crate) fn get_flat(&self, idx: usize, y_len: usize) -> E {
        self.get(idx / y_len, idx % y_len, y_len)
    }

    pub(crate) fn fold_y(&mut self, r: E) {
        match self {
            Self::Dense(trace) => {
                let half = trace.len() / 2;
                for i in 0..half {
                    trace[i] = fold_pair(trace[2 * i], trace[2 * i + 1], r);
                }
                trace.truncate(half);
            }
            Self::Sparse(trace) => trace.fold_y(r),
        }
    }

    pub(crate) fn fold_y2(&mut self, live_x_cols: usize, y_len: usize, r0: E, r1: E) {
        match self {
            Self::Dense(trace) => {
                let next_y_len = y_len >> 2;
                let mut out = vec![E::zero(); live_x_cols * next_y_len];
                for x in 0..live_x_cols {
                    let src_start = x * y_len;
                    let dst_start = x * next_y_len;
                    for quad_y in 0..next_y_len {
                        let base = src_start + 4 * quad_y;
                        out[dst_start + quad_y] = fold_quad(
                            trace[base],
                            trace[base + 1],
                            trace[base + 2],
                            trace[base + 3],
                            r0,
                            r1,
                        );
                    }
                }
                *trace = out;
            }
            Self::Sparse(trace) => trace.fold_y2(r0, r1),
        }
    }

    pub(crate) fn fold_x(&mut self, live_x_cols: usize, y_len: usize, r: E) {
        match self {
            Self::Dense(trace) => {
                let next_live_x_cols = live_x_cols.div_ceil(2);
                let mut out = vec![E::zero(); y_len * next_live_x_cols];
                for y in 0..y_len {
                    let src_start = y * live_x_cols;
                    let dst_start = y * next_live_x_cols;
                    for pair_x in 0..next_live_x_cols {
                        let left = 2 * pair_x;
                        let a = trace[src_start + left];
                        let b = if left + 1 < live_x_cols {
                            trace[src_start + left + 1]
                        } else {
                            E::zero()
                        };
                        out[dst_start + pair_x] = fold_pair(a, b, r);
                    }
                }
                *trace = out;
            }
            Self::Sparse(trace) => trace.fold_x(live_x_cols, r),
        }
    }

    pub(crate) fn fold_for_w_update(
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
