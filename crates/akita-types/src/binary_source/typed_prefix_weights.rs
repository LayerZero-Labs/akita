//! The Stage 3 weight of a typed prefix in aligned blocks of 64 cells.
//!
//! The dense weight is `16 * 2^k'` bytes, but it has little information. On
//! the A tensor it is the row equality times one coefficient factor. On a
//! B/D piece, copy column `j` of period `q` is `beta * K(e)`, and `e` splits
//! by the period's lane-high value: `e = G_0(q) E_0[j] + G_1(q) E_1[j]`, where
//! `G_s(q) = eq(r_hi, lane_high + s + q)` and `E_s[j]` does not depend on `q`
//! (except in a partial last period). `K` is linear, so every 64-cell block is
//! `sum scale * row` over at most two rows of a small pool.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::ResidueKernelPoint;
use akita_error::{checked, AkitaError};
use jolt_field::Field;

use super::{BinaryAProduct, BinaryTypedPrefix, A_DEGREE, A_STRIDE, SLOTS, SLOT_BITS};

/// Pool row of zeros; a block term on it is absent.
const ZERO_ROW: u32 = 0;

/// The Stage 3 weight of a typed prefix, served in aligned blocks of 64 cells.
///
/// A *live* block holds the A cells of one live rank and 64-coefficient chunk
/// of one source row, or one column copy of a B/D piece. Every other block is
/// zero in the weight and in the typed setup table. The weights of live block
/// `k` are `scales[k][0] * pool[rows[k][0]] + scales[k][1] * pool[rows[k][1]]`,
/// with row 0 of the pool all zeros and an absent term scaled by zero. Binding the lowest address bit folds the
/// pool rows; the block records do not change.
#[derive(Clone, Debug)]
pub struct TypedPrefixWeights<E> {
    domain_len: usize,
    width: usize,
    pool: Vec<E>,
    blocks: Vec<usize>,
    scales: Vec<[E; 2]>,
    rows: Vec<[u32; 2]>,
}

impl<E: Field> TypedPrefixWeights<E> {
    /// Cells of the prefix domain before any binding.
    pub fn domain_len(&self) -> usize {
        self.domain_len
    }

    /// Current cells per block: 64 halved once per bound bit.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Index of every live block, at 64 cells per block.
    pub fn blocks(&self) -> &[usize] {
        &self.blocks
    }

    /// The two `(scale, row)` terms of live block `live` at the current
    /// width; its weights are `scale_0 * row_0 + scale_1 * row_1`. An absent
    /// term has scale zero.
    pub fn block_terms(&self, live: usize) -> Result<[(E, &[E]); 2], AkitaError> {
        let (scales, rows) = self
            .scales
            .get(live)
            .zip(self.rows.get(live))
            .ok_or(AkitaError::InvalidProof)?;
        Ok([
            (scales[0], self.row(rows[0])?),
            (scales[1], self.row(rows[1])?),
        ])
    }

    /// Bind the lowest unbound address bit to `challenge`.
    pub fn bind(&mut self, challenge: E) -> Result<(), AkitaError> {
        if self.width < 2 {
            return Err(AkitaError::InvalidInput(
                "typed prefix weight blocks are fully bound".into(),
            ));
        }
        // Rows are contiguous and of even width, so one pairwise fold over
        // the pool folds every row.
        let half = self.pool.len() / 2;
        for index in 0..half {
            let (low, high) = (self.pool[2 * index], self.pool[2 * index + 1]);
            self.pool[index] = low + challenge * (high - low);
        }
        self.pool.truncate(half);
        self.width /= 2;
        Ok(())
    }

    fn row(&self, row: u32) -> Result<&[E], AkitaError> {
        let start = checked::product([row as usize, self.width]).ok_or(AkitaError::InvalidProof)?;
        self.pool
            .get(start..)
            .and_then(|rows| rows.get(..self.width))
            .ok_or(AkitaError::InvalidProof)
    }
}

/// Pool and block records under construction.
struct Builder<E> {
    pool: Vec<E>,
    blocks: Vec<usize>,
    scales: Vec<[E; 2]>,
    rows: Vec<[u32; 2]>,
}

impl<E: Field> Builder<E> {
    fn push_row(&mut self, row: &[E]) -> Result<u32, AkitaError> {
        let index = u32::try_from(self.pool.len() / SLOTS).map_err(|_| AkitaError::InvalidProof)?;
        if row.len() != SLOTS {
            return Err(AkitaError::InvalidProof);
        }
        self.pool.extend_from_slice(row);
        Ok(index)
    }

    fn push_block(&mut self, cell: usize, scales: [E; 2], rows: [u32; 2]) {
        self.blocks.push(cell >> SLOT_BITS);
        self.scales.push(scales);
        self.rows.push(rows);
    }
}

impl BinaryTypedPrefix {
    /// The Stage 3 weights of [`Self::prover_weights`], as live 64-cell blocks
    /// over a small pool of rows. Expanding the blocks gives the dense table
    /// exactly.
    pub fn prover_weight_blocks<E: Field>(
        &self,
        a: &BinaryAProduct<E>,
        root_point: &[E],
        alpha: E,
        batches: [&[E]; 2],
        prefix_len: usize,
    ) -> Result<TypedPrefixWeights<E>, AkitaError> {
        self.validate_prefix(prefix_len)?;
        self.validate_batches(batches)?;
        let (index, coefficient) = a.factors()?;
        if coefficient.len() != self.a.row_stride || index.len() < self.a.rows {
            return Err(AkitaError::InvalidProof);
        }
        let mut builder = Builder {
            pool: vec![E::zero(); SLOTS],
            blocks: Vec::new(),
            scales: Vec::new(),
            rows: Vec::new(),
        };
        // A: one pool row per live rank and coefficient chunk, shared by
        // every source row.
        let chunks = A_DEGREE.div_ceil(SLOTS);
        let mut a_rows = Vec::new();
        for rank in 0..self.a_rank {
            for chunk in 0..chunks {
                let start = checked::mul_add(rank, A_STRIDE, chunk * SLOTS)
                    .ok_or(AkitaError::InvalidProof)?;
                let row = coefficient
                    .get(start..)
                    .and_then(|row| row.get(..SLOTS))
                    .ok_or(AkitaError::InvalidProof)?;
                a_rows.push((start, builder.push_row(row)?));
            }
        }
        for (source_row, &row_weight) in index.iter().take(self.a.rows).enumerate() {
            let base = checked::product([source_row, self.a.row_stride])
                .ok_or(AkitaError::InvalidProof)?;
            for &(start, row) in &a_rows {
                builder.push_block(base + start, [row_weight, E::zero()], [row, ZERO_ROW]);
            }
        }
        let kernel = ResidueKernelPoint::new(alpha, SLOTS)?;
        for (segment_index, segment) in self.segments.iter().enumerate() {
            let stride_bits = segment.lanes.stride().trailing_zeros() as usize;
            let period_bits = segment.period.trailing_zeros() as usize;
            let lane_end =
                checked::sum([stride_bits, period_bits]).ok_or(AkitaError::InvalidProof)?;
            let coefficient_eq = EqPolynomial::evals(
                root_point
                    .get(..stride_bits)
                    .ok_or(AkitaError::InvalidProof)?,
            )?;
            let lane_eq = EqPolynomial::evals(
                root_point
                    .get(stride_bits..lane_end)
                    .ok_or(AkitaError::InvalidProof)?,
            )?;
            let lane_point = root_point.get(lane_end..).ok_or(AkitaError::InvalidProof)?;
            let first_lane = segment.lanes.start() / segment.lanes.stride();
            let low_lane = first_lane & (segment.period - 1);
            let count = segment.lane_count();
            let degree = segment.lanes.degree();
            let shift = segment.offset % SLOTS;
            // Variant 0 is a full period; variant 1, when present, is the
            // partial last period with `count mod P` live lanes.
            let partial = count % segment.period;
            let mut profiles = Vec::new();
            for live in [segment.period, partial]
                .into_iter()
                .filter(|&live| live > 0)
            {
                for column in 0..segment.columns {
                    let mut equality = [[E::zero(); SLOTS]; 2];
                    let mut present = [false; 2];
                    for slot in 0..SLOTS {
                        let Some(local) = checked::mul_add(column, SLOTS, slot)
                            .and_then(|local| local.checked_sub(shift))
                        else {
                            continue;
                        };
                        let (lane, coefficient) = (local / degree, local % degree);
                        if lane >= live {
                            continue;
                        }
                        // The lane-high value of `first_lane + q P + lane` is
                        // `(first_lane >> period_bits) + high + q`.
                        let high = (low_lane + lane) >> period_bits;
                        let side = equality.get_mut(high).ok_or(AkitaError::InvalidProof)?;
                        *side.get_mut(slot).ok_or(AkitaError::InvalidProof)? = *coefficient_eq
                            .get(coefficient)
                            .ok_or(AkitaError::InvalidProof)?
                            * *lane_eq
                                .get((first_lane + lane) & (segment.period - 1))
                                .ok_or(AkitaError::InvalidProof)?;
                        present[high] = true;
                    }
                    let mut rows = [ZERO_ROW; 2];
                    for ((row, equality), present) in rows.iter_mut().zip(&equality).zip(present) {
                        if present {
                            *row = builder.push_row(&kernel.field_kernel(equality)?)?;
                        }
                    }
                    profiles.push(rows);
                }
            }
            let lane_high = first_lane >> period_bits;
            let period_factors = (0..segment.periods)
                .map(|q| [0, 1].map(|side| eq_eval_at_index(lane_point, lane_high + side + q)))
                .collect::<Vec<_>>();
            let row_batches = *batches.get(segment.role).ok_or(AkitaError::InvalidProof)?;
            for piece in self
                .pieces
                .iter()
                .filter(|piece| piece.segment == segment_index)
            {
                let region = self
                    .regions
                    .get(piece.region)
                    .ok_or(AkitaError::InvalidProof)?;
                let batch = *row_batches.get(piece.row).ok_or(AkitaError::InvalidProof)?;
                let width = 1usize << piece.width_bits;
                for chunk in 0..1usize << (piece.period_bits + piece.width_bits) {
                    let column = piece.first + (chunk & (width - 1));
                    let q = piece.q0 + (chunk >> piece.width_bits);
                    let variant = usize::from(partial > 0 && q + 1 == segment.periods);
                    let rows = *profiles
                        .get(variant * segment.columns + column)
                        .ok_or(AkitaError::InvalidProof)?;
                    let factors = period_factors.get(q).ok_or(AkitaError::InvalidProof)?;
                    let scales = [0, 1].map(|side| {
                        if rows[side] == ZERO_ROW {
                            E::zero()
                        } else {
                            batch * factors[side]
                        }
                    });
                    let cell = checked::mul_add(chunk, SLOTS, piece.base)
                        .and_then(|local| region.deposit(local))
                        .ok_or(AkitaError::InvalidProof)?;
                    builder.push_block(cell, scales, rows);
                }
            }
        }
        // Live blocks are disjoint and inside the domain.
        let mut seen = vec![false; prefix_len >> SLOT_BITS];
        for &block in &builder.blocks {
            let seen = seen.get_mut(block).ok_or(AkitaError::InvalidProof)?;
            if std::mem::replace(seen, true) {
                return Err(AkitaError::InvalidProof);
            }
        }
        Ok(TypedPrefixWeights {
            domain_len: prefix_len,
            width: SLOTS,
            pool: builder.pool,
            blocks: builder.blocks,
            scales: builder.scales,
            rows: builder.rows,
        })
    }
}
