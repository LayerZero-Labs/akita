//! Block profiles of a root weight table for the prover's low Stage 2 rounds.
//!
//! Stage 2 binds address bits least significant first. A recomposed family
//! gives every digit lane of an element the same coefficient profile, scaled
//! by `8^digit`, and every lane is a whole number of aligned blocks of
//! `2^block_bits` addresses. While only in-block bits are bound, the weight of
//! a block is therefore `scale * shared_row + own_row`: one profile row shared
//! by every digit of its element, plus the block's per-address weights. The
//! rows fold like the dense table, so each pair and each folded entry is the
//! same field element as in the materialized table, at a fraction of its size.

#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;

use akita_error::{checked, AkitaError};
use jolt_field::{Field, Fold};

use super::SegmentLanes;

/// One aligned block: `scale * row[shared] + row[own]`. Row zero is all zeros.
#[derive(Clone, Copy, Debug)]
struct BlockWeight<E> {
    scale: E,
    shared: u32,
    own: u32,
}

/// A root weight table as block profiles, bound in its low variables.
#[derive(Clone, Debug)]
pub struct RootWeightProfiles<E: Field> {
    /// Every row has `2^width_bits` entries; `width_bits` starts at the
    /// block bits and drops by one per bound variable.
    width_bits: usize,
    blocks: Vec<BlockWeight<E>>,
    rows: Vec<E>,
}

impl<E: Field> RootWeightProfiles<E> {
    /// Length of the partially bound table.
    pub fn len(&self) -> usize {
        self.blocks.len() << self.width_bits
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Weights of the even address `left` and its neighbour, while each block
    /// still has an unbound variable.
    #[inline]
    pub fn pair(&self, left: usize) -> (E, E) {
        debug_assert!(self.width_bits > 0 && left.is_multiple_of(2));
        let block = &self.blocks[left >> self.width_bits];
        let offset = left & ((1 << self.width_bits) - 1);
        let own = ((block.own as usize) << self.width_bits) + offset;
        let (own0, own1) = (self.rows[own], self.rows[own + 1]);
        if block.shared == 0 {
            return (own0, own1);
        }
        let shared = ((block.shared as usize) << self.width_bits) + offset;
        (
            block.scale * self.rows[shared] + own0,
            block.scale * self.rows[shared + 1] + own1,
        )
    }

    /// Bind the lowest live variable. Pairs never cross a row, so the rows
    /// fold as one table.
    pub fn bind(&mut self, challenge: E)
    where
        E: Fold,
    {
        debug_assert!(self.width_bits > 0);
        akita_algebra::poly::fold_evals_in_place(&mut self.rows, challenge);
        self.width_bits -= 1;
    }

    /// The partially bound table in dense form, leaving the profiles empty.
    pub fn take_dense(&mut self) -> Vec<E> {
        let bits = self.width_bits;
        let rows = std::mem::take(&mut self.rows);
        let blocks = std::mem::take(&mut self.blocks);
        let entry = |index: usize| {
            let block = &blocks[index >> bits];
            let offset = index & ((1 << bits) - 1);
            let own = rows[((block.own as usize) << bits) + offset];
            if block.shared == 0 {
                own
            } else {
                block.scale * rows[((block.shared as usize) << bits) + offset] + own
            }
        };
        let len = blocks.len() << bits;
        #[cfg(feature = "parallel")]
        let dense = (0..len).into_par_iter().map(entry).collect();
        #[cfg(not(feature = "parallel"))]
        let dense = (0..len).map(entry).collect();
        dense
    }
}

/// Lanes one parallel task of [`ProfileBuilder::add_lanes`] fills.
#[cfg(feature = "parallel")]
const LANES_PER_TASK: usize = 32;

/// Consecutive blocks whose rows one task of
/// [`ProfileBuilder::add_intervals`] fills.
const BLOCKS_PER_TASK: usize = 64;

/// Builds [`RootWeightProfiles`] from explicit terms and structured families.
pub(crate) struct ProfileBuilder<E: Field> {
    block_bits: usize,
    blocks: Vec<BlockWeight<E>>,
    rows: Vec<E>,
}

impl<E: Field> ProfileBuilder<E> {
    pub(crate) fn new(table_len: usize, block_bits: usize) -> Result<Self, AkitaError> {
        let width = checked::pow2(block_bits).ok_or(AkitaError::InvalidProof)?;
        if !table_len.is_power_of_two() || width > table_len {
            return Err(AkitaError::InvalidSetup(
                "root weight profile blocks differ from the table".into(),
            ));
        }
        let zero = BlockWeight {
            scale: E::zero(),
            shared: 0,
            own: 0,
        };
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(table_len / width)
            .map_err(|_| AkitaError::InvalidSetup("root weight profile allocation".into()))?;
        blocks.resize(table_len / width, zero);
        let mut builder = Self {
            block_bits,
            blocks,
            rows: Vec::new(),
        };
        builder.row()?;
        Ok(builder)
    }

    pub(crate) fn width(&self) -> usize {
        1 << self.block_bits
    }

    /// Append one zero row and return its index.
    fn row(&mut self) -> Result<u32, AkitaError> {
        let width = self.width();
        let index = u32::try_from(self.rows.len() / width).map_err(|_| AkitaError::InvalidProof)?;
        self.rows
            .try_reserve(width)
            .map_err(|_| AkitaError::InvalidSetup("root weight profile allocation".into()))?;
        self.rows.resize(self.rows.len() + width, E::zero());
        Ok(index)
    }

    /// Add `value` to one table address.
    pub(crate) fn add(&mut self, address: usize, value: E) -> Result<(), AkitaError> {
        let block = address >> self.block_bits;
        let own = match self.blocks.get(block) {
            Some(weight) if weight.own != 0 => weight.own,
            Some(_) => {
                let own = self.row()?;
                self.blocks[block].own = own;
                own
            }
            None => return Err(AkitaError::InvalidProof),
        };
        let index = own as usize * self.width() + (address & (self.width() - 1));
        self.rows[index] += value;
        Ok(())
    }

    /// Add per-address weights to whole lanes, in parallel.
    ///
    /// Every lane starts at a block boundary in `bases` and covers `degree`
    /// addresses, so it spans `degree.div_ceil(width)` consecutive blocks.
    /// Blocks without their own row receive one first. `add(lanes, rows)`
    /// then adds the weights of a run of consecutive lanes: `rows` holds the
    /// own rows of their blocks, lane by lane, in address order.
    pub(crate) fn add_lanes(
        &mut self,
        bases: &[usize],
        degree: usize,
        add: impl Fn(std::ops::Range<usize>, &mut [&mut [E]]) -> Result<(), AkitaError> + Sync + Send,
    ) -> Result<(), AkitaError> {
        let width = self.width();
        let lane_rows = degree.div_ceil(width);
        if bases.is_empty() || lane_rows == 0 {
            return Ok(());
        }
        let (block_bits, block_count) = (self.block_bits, self.blocks.len());
        let blocks = move |base: usize| {
            let first = base >> block_bits;
            (base.is_multiple_of(width) && first + lane_rows <= block_count)
                .then_some(first..first + lane_rows)
        };
        for &base in bases {
            let range = blocks(base).ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "root weight profile lanes differ from their blocks".into(),
                )
            })?;
            for block in range {
                if self.blocks[block].own == 0 {
                    self.blocks[block].own = self.row()?;
                }
            }
        }
        let mut rows = self
            .rows
            .chunks_exact_mut(width)
            .map(Some)
            .collect::<Vec<_>>();
        let mut lanes = Vec::with_capacity(bases.len() * lane_rows);
        for &base in bases {
            for block in blocks(base).ok_or(AkitaError::InvalidProof)? {
                lanes.push(
                    rows.get_mut(self.blocks[block].own as usize)
                        .and_then(Option::take)
                        .ok_or(AkitaError::InvalidProof)?,
                );
            }
        }
        #[cfg(feature = "parallel")]
        lanes
            .par_chunks_mut(LANES_PER_TASK * lane_rows)
            .enumerate()
            .try_for_each(|(task, rows)| {
                let start = task * LANES_PER_TASK;
                add(start..start + rows.len() / lane_rows, rows)
            })?;
        #[cfg(not(feature = "parallel"))]
        add(0..bases.len(), &mut lanes)?;
        Ok(())
    }

    /// Add per-address weights over address intervals, in parallel.
    ///
    /// `intervals` holds `(start, len)` pairs, which may overlap. Every
    /// block an interval meets receives its own row first. One task then
    /// owns the rows of a run of consecutive blocks and adds every piece of
    /// every interval inside that run: `add(interval, first, row)` adds the
    /// weights of `interval` from its `first` address onward to `row`, the
    /// addresses of one block that the interval covers.
    pub(crate) fn add_intervals(
        &mut self,
        intervals: &[(usize, usize)],
        add: impl Fn(usize, usize, &mut [E]) -> Result<(), AkitaError> + Sync + Send,
    ) -> Result<(), AkitaError> {
        let (width, block_bits) = (self.width(), self.block_bits);
        let table_len = self.blocks.len() << block_bits;
        let mut order = Vec::with_capacity(intervals.len());
        let mut longest = 0;
        for (index, &(start, len)) in intervals.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let end = checked::sum([start, len])
                .filter(|&end| end <= table_len)
                .ok_or(AkitaError::InvalidProof)?;
            for block in start >> block_bits..=(end - 1) >> block_bits {
                if self.blocks[block].own == 0 {
                    self.blocks[block].own = self.row()?;
                }
            }
            order.push(index);
            longest = longest.max(len);
        }
        if order.is_empty() {
            return Ok(());
        }
        order.sort_unstable_by_key(|&index| intervals[index].0);
        let mut rows = self
            .rows
            .chunks_exact_mut(width)
            .map(Some)
            .collect::<Vec<_>>();
        let mut owned = self
            .blocks
            .iter()
            .map(|block| match block.own {
                0 => None,
                own => rows.get_mut(own as usize).and_then(Option::take),
            })
            .collect::<Vec<_>>();
        let fill = |task: usize, owned: &mut [Option<&mut [E]>]| {
            let first_block = task * BLOCKS_PER_TASK;
            let (low, high) = (
                first_block << block_bits,
                (first_block + owned.len()) << block_bits,
            );
            // Intervals are sorted by start and none is longer than
            // `longest`, so every interval that reaches `low` starts here.
            let from = order.partition_point(|&index| intervals[index].0 + longest <= low);
            for &index in &order[from..] {
                let (start, len) = intervals[index];
                if start >= high {
                    break;
                }
                let mut address = start.max(low);
                let end = (start + len).min(high);
                while address < end {
                    let offset = address & (width - 1);
                    let run = (width - offset).min(end - address);
                    let row = owned
                        .get_mut((address >> block_bits) - first_block)
                        .and_then(Option::as_deref_mut)
                        .ok_or(AkitaError::InvalidProof)?;
                    add(index, address - start, &mut row[offset..offset + run])?;
                    address += run;
                }
            }
            Ok(())
        };
        #[cfg(feature = "parallel")]
        owned
            .par_chunks_mut(BLOCKS_PER_TASK)
            .enumerate()
            .try_for_each(|(task, owned)| fill(task, owned))?;
        #[cfg(not(feature = "parallel"))]
        owned
            .chunks_mut(BLOCKS_PER_TASK)
            .enumerate()
            .try_for_each(|(task, owned)| fill(task, owned))?;
        Ok(())
    }

    /// Add one profile to every element of `lanes`, in parallel.
    ///
    /// `add(element, profile)` adds to the element's `degree` natural
    /// coefficients, which digit lane `d` weighs by `8^d`. Lanes whose stride
    /// is a multiple of the block width keep one shared profile per element;
    /// narrower lanes stream the nonzero entries as per-address terms.
    pub(crate) fn add_shared(
        &mut self,
        lanes: &SegmentLanes,
        add: impl Fn(usize, &mut [E]) -> Result<(), AkitaError> + Sync + Send,
    ) -> Result<(), AkitaError> {
        let degree = lanes.degree();
        if lanes.stride() & (self.width() - 1) != 0 {
            let mut profile = vec![E::zero(); degree];
            for element in 0..lanes.elements() {
                profile.fill(E::zero());
                add(element, &mut profile)?;
                for (offset, &value) in profile.iter().enumerate() {
                    if !value.is_zero() {
                        lanes.recompose(element, offset, value, &mut |address, value| {
                            self.add(address, value)
                        })?;
                    }
                }
            }
            return Ok(());
        }
        let mut shared = Vec::with_capacity(lanes.elements());
        for element in 0..lanes.elements() {
            let first = lanes.base(element, 0)? >> self.block_bits;
            let row = match self.blocks.get(first) {
                Some(weight) if weight.shared != 0 => weight.shared,
                Some(_) => self.share(lanes, element)?,
                None => return Err(AkitaError::InvalidProof),
            };
            shared.push((row as usize, element));
        }
        // Every element owns distinct contiguous rows; split them off in
        // row order.
        shared.sort_unstable();
        let span = degree.div_ceil(self.width()) << self.block_bits;
        let mut profiles = Vec::with_capacity(shared.len());
        let mut rest = self.rows.as_mut_slice();
        let mut position = 0;
        for (row, element) in shared {
            let start = (row << self.block_bits)
                .checked_sub(position)
                .ok_or(AkitaError::InvalidProof)?;
            let (_, tail) = rest
                .split_at_mut_checked(start)
                .ok_or(AkitaError::InvalidProof)?;
            let (rows, tail) = tail
                .split_at_mut_checked(span)
                .ok_or(AkitaError::InvalidProof)?;
            profiles.push((
                element,
                rows.get_mut(..degree).ok_or(AkitaError::InvalidProof)?,
            ));
            rest = tail;
            position = (row << self.block_bits) + span;
        }
        #[cfg(feature = "parallel")]
        profiles
            .par_iter_mut()
            .try_for_each(|(element, profile)| add(*element, profile))?;
        #[cfg(not(feature = "parallel"))]
        profiles
            .iter_mut()
            .try_for_each(|(element, profile)| add(*element, profile))?;
        Ok(())
    }

    /// Allocate the shared rows of one element and point its lanes at them.
    fn share(&mut self, lanes: &SegmentLanes, element: usize) -> Result<u32, AkitaError> {
        let width = self.width();
        let rows = lanes.degree().div_ceil(width);
        let first = self.row()?;
        for _ in 1..rows {
            self.row()?;
        }
        let mut scale = E::one();
        for digit in 0..lanes.digits() {
            let block = lanes.base(element, digit)? >> self.block_bits;
            for (row, weight) in (first..).zip(
                self.blocks
                    .get_mut(block..block + rows)
                    .ok_or(AkitaError::InvalidProof)?,
            ) {
                if weight.shared != 0 {
                    return Err(AkitaError::InvalidSetup(
                        "root weight profile lanes overlap".into(),
                    ));
                }
                *weight = BlockWeight {
                    scale,
                    shared: row,
                    own: weight.own,
                };
            }
            scale *= E::from_u64(8);
        }
        Ok(first)
    }

    pub(crate) fn finish(self) -> RootWeightProfiles<E> {
        RootWeightProfiles {
            width_bits: self.block_bits,
            blocks: self.blocks,
            rows: self.rows,
        }
    }
}

#[cfg(test)]
#[path = "profiles_tests.rs"]
mod tests;
