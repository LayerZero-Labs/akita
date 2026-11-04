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

    fn width(&self) -> usize {
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

    /// Add `8^d * value` at `offset` of every digit lane `d` of `element`.
    ///
    /// Lanes whose stride is a multiple of the block width keep one shared
    /// profile per element; narrower lanes fall back to per-address terms.
    pub(crate) fn add_recomposed(
        &mut self,
        lanes: &SegmentLanes,
        element: usize,
        offset: usize,
        value: E,
    ) -> Result<(), AkitaError> {
        if offset >= lanes.degree() {
            return Err(AkitaError::InvalidProof);
        }
        if lanes.stride() & (self.width() - 1) != 0 {
            return lanes.recompose(element, offset, value, &mut |address, value| {
                self.add(address, value)
            });
        }
        let first = lanes.base(element, 0)? >> self.block_bits;
        let shared = match self.blocks.get(first) {
            Some(weight) if weight.shared != 0 => weight.shared,
            Some(_) => self.share(lanes, element)?,
            None => return Err(AkitaError::InvalidProof),
        };
        // The element's shared rows are contiguous, so `offset` indexes them
        // directly.
        self.rows[((shared as usize) << self.block_bits) + offset] += value;
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
