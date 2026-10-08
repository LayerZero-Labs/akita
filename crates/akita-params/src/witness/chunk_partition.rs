use std::ops::Range;

use crate::CommittedGroupParams;
use akita_error::{checked, AkitaError};

use super::MAX_WITNESS_CHUNKS;

/// Chunk-major body lengths and shared tail before successor alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WitnessChunkShape {
    pub body_lengths: [usize; MAX_WITNESS_CHUNKS],
    pub num_chunks: usize,
    pub tail_prefix_len: usize,
    pub tail_alignment: usize,
    pub tail_len: usize,
    pub tail_suffix_alignment: usize,
}

impl std::hash::Hash for WitnessChunkShape {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.num_chunks.hash(state);
        self.body_lengths[..self.num_chunks.min(MAX_WITNESS_CHUNKS)].hash(state);
        self.tail_prefix_len.hash(state);
        self.tail_alignment.hash(state);
        self.tail_len.hash(state);
        self.tail_suffix_alignment.hash(state);
    }
}

impl WitnessChunkShape {
    /// Pad multi-chunk bodies to the consumer's block width; merge owners when
    /// chunking contracts, retaining the complete shared tail in the last owner.
    pub fn align(
        self,
        block_len: usize,
        num_chunks: usize,
    ) -> Result<(usize, Vec<usize>), AkitaError> {
        if !self.tail_alignment.is_power_of_two()
            || !self.tail_suffix_alignment.is_power_of_two()
            || !block_len.is_power_of_two()
            || num_chunks == 0
            || !num_chunks.is_power_of_two()
            || !self.num_chunks.is_power_of_two()
            || self.num_chunks > MAX_WITNESS_CHUNKS
            || !self.num_chunks.is_multiple_of(num_chunks)
        {
            return Err(AkitaError::InvalidSetup(
                "invalid inherited witness chunks".into(),
            ));
        }
        let body_alignment = if self.num_chunks > 1 { block_len } else { 1 };
        let mut cursor = 0usize;
        let mut ends = Vec::with_capacity(num_chunks);
        for (index, &len) in self.body_lengths[..self.num_chunks].iter().enumerate() {
            let padded = checked::align_up(len, body_alignment)
                .and_then(|len| cursor.checked_add(len))
                .ok_or_else(|| AkitaError::InvalidSetup("witness chunk padding overflow".into()))?;
            cursor = padded;
            if (index + 1).is_multiple_of(self.num_chunks / num_chunks) {
                ends.push(cursor / block_len);
            }
        }
        let len = cursor
            .checked_add(self.tail_prefix_len)
            .and_then(|len| checked::align_up(len, self.tail_alignment))
            .and_then(|len| len.checked_add(self.tail_len))
            .and_then(|len| checked::align_up(len, self.tail_suffix_alignment))
            .ok_or_else(|| AkitaError::InvalidSetup("witness tail overflow".into()))?;
        if let Some(last) = ends.last_mut() {
            *last = len.div_ceil(block_len);
        }
        if num_chunks == 1 {
            ends.clear();
        }
        Ok((len, ends))
    }
}

impl CommittedGroupParams {
    /// Schedule-owned partition of one source group. Frozen groups retain the
    /// proportional root partition; a recursive witness inherits its owners.
    pub fn witness_block_ranges(
        &self,
        group_index: usize,
        num_chunks: usize,
    ) -> Result<Vec<Range<usize>>, AkitaError> {
        let params = if group_index == self.preceding_group_count() {
            self.final_group()
        } else {
            *self
                .preceding_group_params(group_index)
                .ok_or(AkitaError::InvalidSetup("missing witness group".into()))?
        };
        if group_index != self.preceding_group_count() || self.witness_chunk_ends.is_empty() {
            return dyadic_block_ranges(params.num_live_blocks(), num_chunks);
        }
        if self.witness_chunk_ends.len() != num_chunks
            || !num_chunks.is_power_of_two()
            || num_chunks > MAX_WITNESS_CHUNKS
            || self.witness_chunk_ends.last() != Some(&params.num_live_blocks())
        {
            return Err(AkitaError::InvalidSetup(
                "inherited witness partition has wrong extent".into(),
            ));
        }
        let mut start = 0;
        self.witness_chunk_ends
            .iter()
            .map(|&end| {
                if end < start {
                    return Err(AkitaError::InvalidSetup(
                        "inherited witness partition is unordered".into(),
                    ));
                }
                let range = start..end;
                start = end;
                Ok(range)
            })
            .collect()
    }
}

/// Partition an exact live block prefix into canonical dyadic ranges.
///
/// Part `i` of `P` owns
/// `[floor(i * B / P), floor((i + 1) * B / P))`, where `B` is
/// `num_live_blocks`. The quotient and remainder calculation below evaluates
/// those endpoints without forming the overflow prone product `i * B`. Every
/// product in the equivalent calculation uses checked arithmetic. When one
/// supported part count divides another, every boundary of the coarser
/// partition is a boundary of the finer partition.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when either count is zero, the part
/// count is not a power of two, the part count exceeds the verifier cap, or
/// endpoint arithmetic overflows. When there are more parts than live blocks,
/// consecutive equal boundaries produce empty ranges while preserving all part
/// indices.
pub fn dyadic_block_ranges(
    num_live_blocks: usize,
    num_parts: usize,
) -> Result<Vec<Range<usize>>, AkitaError> {
    if num_parts == 0 || num_parts > MAX_WITNESS_CHUNKS || num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "dyadic block partition geometry is malformed".into(),
        ));
    }
    if !num_parts.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "dyadic block partition count must be a power of two".into(),
        ));
    }
    let base = num_live_blocks / num_parts;
    let remainder = num_live_blocks % num_parts;
    let boundary = |index: usize| -> Result<usize, AkitaError> {
        let base_offset = base.checked_mul(index).ok_or_else(|| {
            AkitaError::InvalidSetup("dyadic block partition boundary overflow".into())
        })?;
        let remainder_offset = remainder.checked_mul(index).ok_or_else(|| {
            AkitaError::InvalidSetup("dyadic block partition boundary overflow".into())
        })? / num_parts;
        base_offset.checked_add(remainder_offset).ok_or_else(|| {
            AkitaError::InvalidSetup("dyadic block partition boundary overflow".into())
        })
    };

    let mut ranges = Vec::with_capacity(num_parts);
    for part_index in 0..num_parts {
        let start = boundary(part_index)?;
        let end = boundary(part_index + 1)?;
        if start > end {
            return Err(AkitaError::InvalidSetup(
                "dyadic block partition boundaries are not ordered".into(),
            ));
        }
        ranges.push(start..end);
    }
    if ranges.first().is_none_or(|range| range.start != 0)
        || ranges
            .last()
            .is_none_or(|range| range.end != num_live_blocks)
    {
        return Err(AkitaError::InvalidSetup(
            "dyadic block partition does not cover the live blocks".into(),
        ));
    }
    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::*;

    #[test]
    fn dyadic_chunks_use_proportional_boundaries() {
        assert_eq!(
            dyadic_block_ranges(13, 4).expect("chunk ranges"),
            vec![0..3, 3..6, 6..9, 9..13]
        );
        assert_eq!(
            dyadic_block_ranges(10, 4).expect("chunk ranges"),
            vec![0..2, 2..5, 5..7, 7..10]
        );
        assert_eq!(
            dyadic_block_ranges(4, 8).expect("chunk ranges with empty slots"),
            vec![0..0, 0..1, 1..1, 1..2, 2..2, 2..3, 3..3, 3..4]
        );
    }

    #[test]
    fn dyadic_chunk_partitions_are_balanced_and_nested() {
        let supported_parts = [1usize, 2, 4, 8, 16, 32, 64];
        for num_live_blocks in 1usize..=512 {
            let counts = supported_parts;
            let partitions = counts
                .iter()
                .map(|&parts| {
                    dyadic_block_ranges(num_live_blocks, parts).expect("dyadic partition")
                })
                .collect::<Vec<_>>();

            for (&parts, ranges) in counts.iter().zip(&partitions) {
                assert_eq!(ranges.first().expect("first range").start, 0);
                assert_eq!(ranges.last().expect("last range").end, num_live_blocks);
                assert_eq!(ranges.len(), parts);
                assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
                let min_len = ranges.iter().map(Range::len).min().expect("minimum");
                let max_len = ranges.iter().map(Range::len).max().expect("maximum");
                assert!(max_len - min_len <= 1);
                for (part_index, range) in ranges.iter().enumerate() {
                    let expected_start = ((part_index as u128) * (num_live_blocks as u128)
                        / (parts as u128)) as usize;
                    let expected_end = (((part_index + 1) as u128) * (num_live_blocks as u128)
                        / (parts as u128)) as usize;
                    assert_eq!(range, &(expected_start..expected_end));
                }
            }

            for (coarse_index, coarse) in partitions.iter().enumerate() {
                for fine in partitions.iter().skip(coarse_index) {
                    let fine_boundaries = fine
                        .iter()
                        .map(|range| range.start)
                        .chain(std::iter::once(num_live_blocks))
                        .collect::<Vec<_>>();
                    assert!(coarse
                        .iter()
                        .map(|range| range.start)
                        .chain(std::iter::once(num_live_blocks))
                        .all(|boundary| fine_boundaries.contains(&boundary)));
                }
            }
        }
    }

    #[test]
    fn dyadic_chunk_partition_validates_counts_without_overflow() {
        for (blocks, parts) in [(0, 1), (8, 0), (8, 3), (128, 128)] {
            assert!(matches!(
                dyadic_block_ranges(blocks, parts),
                Err(AkitaError::InvalidSetup(_))
            ));
        }
        let ranges =
            dyadic_block_ranges(usize::MAX, MAX_WITNESS_CHUNKS).expect("maximum block count");
        assert_eq!(ranges.first().expect("first range").start, 0);
        assert_eq!(ranges.last().expect("last range").end, usize::MAX);
    }
}
