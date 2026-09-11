//! Canonical B commitment slice counts and derived geometry.

use std::ops::Range;

use akita_error::{checked, AkitaError};

use crate::compression::CommitmentPayloadMode;
use crate::witness::dyadic_block_ranges;

/// Largest B commitment slice count admitted by the protocol.
pub const MAX_COMMITMENT_SLICES: usize = 8;

/// Checked number of logical inputs committed through one physical B matrix.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct CommitmentSliceCount(u8);

impl Default for CommitmentSliceCount {
    fn default() -> Self {
        Self::ONE
    }
}

impl CommitmentSliceCount {
    /// Unsliced commitment geometry.
    pub const ONE: Self = Self(1);
    /// Two B slices.
    pub const TWO: Self = Self(2);
    /// Four B slices.
    pub const FOUR: Self = Self(4);
    /// Eight B slices.
    pub const EIGHT: Self = Self(8);

    /// All counts admitted by planner enumeration, in canonical order.
    pub const ALL: [Self; 4] = [Self::ONE, Self::TWO, Self::FOUR, Self::EIGHT];

    /// Construct a checked B slice count.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] unless `count` is one, two, four,
    /// or eight.
    pub fn try_new(count: usize) -> Result<Self, AkitaError> {
        match count {
            1 => Ok(Self::ONE),
            2 => Ok(Self::TWO),
            4 => Ok(Self::FOUR),
            8 => Ok(Self::EIGHT),
            _ => Err(AkitaError::InvalidSetup(format!(
                "B commitment slice count {count} is outside the supported set {{1, 2, 4, 8}}"
            ))),
        }
    }

    /// Return the count as `usize`.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0 as usize
    }

    /// Whether this count represents a sliced B commitment.
    #[must_use]
    pub const fn is_sliced(self) -> bool {
        self.0 > 1
    }

    /// Validate commitment-level policy and live block admission.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] for empty B slices, sliced raw
    /// payloads, or slicing after absolute commitment level one.
    pub fn validate_for_commitment(
        self,
        absolute_commitment_level: usize,
        payload_mode: CommitmentPayloadMode,
        num_live_blocks: usize,
    ) -> Result<(), AkitaError> {
        if self.get() > num_live_blocks {
            return Err(AkitaError::InvalidSetup(format!(
                "B commitment slice count {} exceeds {num_live_blocks} live blocks",
                self.get()
            )));
        }
        if self.is_sliced() && absolute_commitment_level >= 2 {
            return Err(AkitaError::InvalidSetup(format!(
                "B commitment slicing is not supported at absolute level {absolute_commitment_level}"
            )));
        }
        if self.is_sliced() && !payload_mode.is_compressed() {
            return Err(AkitaError::InvalidSetup(
                "B commitment slicing requires compressed payload mode".into(),
            ));
        }
        Ok(())
    }

    /// Derive canonical nonempty B slice ranges over the live block prefix.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the live block count is less
    /// than the slice count or the shared dyadic partition rejects the input.
    pub fn block_ranges(self, num_live_blocks: usize) -> Result<Vec<Range<usize>>, AkitaError> {
        if self.get() > num_live_blocks {
            return Err(AkitaError::InvalidSetup(format!(
                "B commitment slice count {} exceeds {num_live_blocks} live blocks",
                self.get()
            )));
        }
        let ranges = dyadic_block_ranges(num_live_blocks, self.get())?;
        if ranges.iter().any(Range::is_empty) {
            return Err(AkitaError::InvalidSetup(
                "B commitment slices must be nonempty".into(),
            ));
        }
        Ok(ranges)
    }

    /// Logical B relation rows in the complete stacked image.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the row count overflows or the
    /// physical rank is zero.
    pub fn logical_output_rows(self, physical_output_rank: usize) -> Result<usize, AkitaError> {
        if physical_output_rank == 0 {
            return Err(AkitaError::InvalidSetup(
                "physical B output rank must be nonzero".into(),
            ));
        }
        physical_output_rank
            .checked_mul(self.get())
            .ok_or_else(|| AkitaError::InvalidSetup("logical B row count overflow".into()))
    }

    /// Field coefficients in the complete stacked B image.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the coefficient count
    /// overflows or either physical dimension is zero.
    pub fn complete_source_coefficients(
        self,
        physical_output_rank: usize,
        outer_ring_dimension: usize,
    ) -> Result<usize, AkitaError> {
        if outer_ring_dimension == 0 {
            return Err(AkitaError::InvalidSetup(
                "B ring dimension must be nonzero".into(),
            ));
        }
        self.logical_output_rows(physical_output_rank)?
            .checked_mul(outer_ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("complete B source size overflow".into()))
    }

    pub(crate) fn append_descriptor_bytes(self, bytes: &mut Vec<u8>) {
        bytes.push(self.0);
    }
}

impl TryFrom<usize> for CommitmentSliceCount {
    type Error = AkitaError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::try_new(value)
    }
}

impl From<CommitmentSliceCount> for usize {
    fn from(value: CommitmentSliceCount) -> Self {
        value.get()
    }
}

/// Checked physical and logical B geometry for one committed group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitmentSliceGeometry {
    slice_count: CommitmentSliceCount,
    block_ranges: Vec<Range<usize>>,
    num_live_blocks: usize,
    num_polynomials: usize,
    max_blocks_per_slice: usize,
    ring_elements_per_block_per_polynomial: usize,
    logical_input_width: usize,
    physical_input_width: usize,
    outer_ring_dimension: usize,
}

impl CommitmentSliceGeometry {
    /// Derive B slice geometry from one committed group's source shape.
    ///
    /// `inner_output_rank * (inner_ring_dimension / outer_ring_dimension) *
    /// num_digits_outer` is the number of B input ring elements contributed by
    /// one live block of one polynomial.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] for zero dimensions, incompatible
    /// role dimensions, empty slices, or arithmetic overflow.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        slice_count: CommitmentSliceCount,
        num_live_blocks: usize,
        num_polynomials: usize,
        inner_output_rank: usize,
        num_digits_outer: usize,
        inner_ring_dimension: usize,
        outer_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if num_polynomials == 0
            || inner_output_rank == 0
            || num_digits_outer == 0
            || inner_ring_dimension == 0
            || outer_ring_dimension == 0
            || !inner_ring_dimension.is_multiple_of(outer_ring_dimension)
        {
            return Err(AkitaError::InvalidSetup(
                "B commitment slice geometry is malformed".into(),
            ));
        }
        let block_ranges = slice_count.block_ranges(num_live_blocks)?;
        let max_blocks_per_slice = block_ranges
            .iter()
            .map(Range::len)
            .max()
            .ok_or_else(|| AkitaError::InvalidSetup("B commitment has no slices".into()))?;
        let projection_ratio = inner_ring_dimension / outer_ring_dimension;
        let ring_elements_per_block_per_polynomial = inner_output_rank
            .checked_mul(projection_ratio)
            .and_then(|count| count.checked_mul(num_digits_outer))
            .ok_or_else(|| AkitaError::InvalidSetup("B slice block width overflow".into()))?;
        let physical_input_width = ring_elements_per_block_per_polynomial
            .checked_mul(max_blocks_per_slice)
            .and_then(|count| count.checked_mul(num_polynomials))
            .ok_or_else(|| AkitaError::InvalidSetup("physical B width overflow".into()))?;
        let logical_input_width = ring_elements_per_block_per_polynomial
            .checked_mul(num_live_blocks)
            .and_then(|count| count.checked_mul(num_polynomials))
            .ok_or_else(|| AkitaError::InvalidSetup("logical B width overflow".into()))?;

        Ok(Self {
            slice_count,
            block_ranges,
            num_live_blocks,
            num_polynomials,
            max_blocks_per_slice,
            ring_elements_per_block_per_polynomial,
            logical_input_width,
            physical_input_width,
            outer_ring_dimension,
        })
    }

    /// Checked slice count used by this geometry.
    #[must_use]
    pub const fn slice_count(&self) -> CommitmentSliceCount {
        self.slice_count
    }

    /// Canonical block ranges in slice order.
    #[must_use]
    pub fn block_ranges(&self) -> &[Range<usize>] {
        &self.block_ranges
    }

    /// Number of live blocks partitioned by this geometry.
    #[must_use]
    pub const fn num_live_blocks(&self) -> usize {
        self.num_live_blocks
    }

    /// Number of polynomials sharing each sliced block partition.
    #[must_use]
    pub const fn num_polynomials(&self) -> usize {
        self.num_polynomials
    }

    /// Largest number of live blocks assigned to one slice.
    #[must_use]
    pub const fn max_blocks_per_slice(&self) -> usize {
        self.max_blocks_per_slice
    }

    /// B input ring elements for one live block of one polynomial.
    #[must_use]
    pub const fn ring_elements_per_block_per_polynomial(&self) -> usize {
        self.ring_elements_per_block_per_polynomial
    }

    /// Unsliced logical B input width across every live block and polynomial.
    #[must_use]
    pub const fn logical_input_width(&self) -> usize {
        self.logical_input_width
    }

    /// Input width of the one physical B matrix.
    #[must_use]
    pub const fn physical_input_width(&self) -> usize {
        self.physical_input_width
    }

    /// Convert one global live-block index to `(slice_index, block_in_slice)`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the block lies outside this
    /// geometry's live prefix.
    pub fn block_coordinates(&self, global_block: usize) -> Result<(usize, usize), AkitaError> {
        self.block_ranges
            .iter()
            .enumerate()
            .find_map(|(slice_index, range)| {
                range
                    .contains(&global_block)
                    .then_some((slice_index, global_block - range.start))
            })
            .ok_or_else(|| AkitaError::InvalidSetup("B block lies outside slice geometry".into()))
    }

    /// Convert one logical stacked B row to `(slice_index, physical_row)`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] for a zero physical rank or a row
    /// outside the complete logical stack.
    pub fn logical_row_coordinates(
        &self,
        logical_row: usize,
        physical_output_rank: usize,
    ) -> Result<(usize, usize), AkitaError> {
        let logical_rows = self.logical_output_rows(physical_output_rank)?;
        if logical_row >= logical_rows {
            return Err(AkitaError::InvalidSetup(
                "logical B row lies outside slice geometry".into(),
            ));
        }
        Ok((
            logical_row / physical_output_rank,
            logical_row % physical_output_rank,
        ))
    }

    /// Flatten `(slice_index, physical_row)` into the complete logical B row.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when either coordinate is outside
    /// this geometry or the flattened index overflows.
    pub fn logical_row_index(
        &self,
        slice_index: usize,
        physical_row: usize,
        physical_output_rank: usize,
    ) -> Result<usize, AkitaError> {
        if slice_index >= self.slice_count.get()
            || physical_output_rank == 0
            || physical_row >= physical_output_rank
        {
            return Err(AkitaError::InvalidSetup(
                "B slice row coordinates are outside the physical matrix".into(),
            ));
        }
        checked::mul_add(slice_index, physical_output_rank, physical_row)
            .ok_or_else(|| AkitaError::InvalidSetup("logical B row index overflow".into()))
    }

    /// Logical B relation rows in the complete stacked image.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the row count overflows.
    pub fn logical_output_rows(&self, physical_output_rank: usize) -> Result<usize, AkitaError> {
        self.slice_count.logical_output_rows(physical_output_rank)
    }

    /// Field coefficients in the complete stacked B image.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the coefficient count
    /// overflows.
    pub fn complete_source_coefficients(
        &self,
        physical_output_rank: usize,
    ) -> Result<usize, AkitaError> {
        self.slice_count
            .complete_source_coefficients(physical_output_rank, self.outer_ring_dimension)
    }

    /// Ring elements stored by the one physical B matrix.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the matrix size overflows.
    pub fn physical_matrix_ring_elements(
        &self,
        physical_output_rank: usize,
    ) -> Result<usize, AkitaError> {
        if physical_output_rank == 0 {
            return Err(AkitaError::InvalidSetup(
                "physical B output rank must be nonzero".into(),
            ));
        }
        physical_output_rank
            .checked_mul(self.physical_input_width)
            .ok_or_else(|| AkitaError::InvalidSetup("physical B matrix size overflow".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_count_accepts_only_the_protocol_domain() {
        assert_eq!(CommitmentSliceCount::default(), CommitmentSliceCount::ONE);
        for count in CommitmentSliceCount::ALL {
            assert_eq!(CommitmentSliceCount::try_new(count.get()), Ok(count));
        }
        for count in [0, 3, 6, 16, usize::MAX] {
            assert!(CommitmentSliceCount::try_new(count).is_err());
        }
        assert_eq!(MAX_COMMITMENT_SLICES, CommitmentSliceCount::EIGHT.get());
    }

    #[test]
    fn slice_count_enforces_level_payload_and_nonempty_policy() {
        CommitmentSliceCount::ONE
            .validate_for_commitment(12, CommitmentPayloadMode::Raw, 1)
            .expect("unsliced raw tail");
        CommitmentSliceCount::FOUR
            .validate_for_commitment(1, CommitmentPayloadMode::Compressed, 13)
            .expect("sliced compressed prefix");

        assert!(CommitmentSliceCount::TWO
            .validate_for_commitment(2, CommitmentPayloadMode::Compressed, 13)
            .is_err());
        assert!(CommitmentSliceCount::TWO
            .validate_for_commitment(1, CommitmentPayloadMode::Raw, 13)
            .is_err());
        assert!(CommitmentSliceCount::EIGHT
            .validate_for_commitment(0, CommitmentPayloadMode::Compressed, 7)
            .is_err());
    }

    #[test]
    fn slice_ranges_use_the_shared_dyadic_partition() {
        assert_eq!(
            CommitmentSliceCount::FOUR
                .block_ranges(13)
                .expect("slice ranges"),
            vec![0..3, 3..6, 6..9, 9..13]
        );
        assert!(CommitmentSliceCount::EIGHT.block_ranges(5).is_err());
    }

    #[test]
    fn geometry_separates_physical_and_logical_b_sizes() {
        let geometry =
            CommitmentSliceGeometry::try_new(CommitmentSliceCount::FOUR, 13, 2, 3, 5, 128, 64)
                .expect("slice geometry");

        assert_eq!(geometry.block_ranges(), &[0..3, 3..6, 6..9, 9..13]);
        assert_eq!(geometry.max_blocks_per_slice(), 4);
        assert_eq!(geometry.ring_elements_per_block_per_polynomial(), 30);
        assert_eq!(geometry.logical_input_width(), 780);
        assert_eq!(geometry.physical_input_width(), 240);
        assert_eq!(geometry.logical_output_rows(7).expect("logical rows"), 28);
        assert_eq!(
            geometry
                .complete_source_coefficients(7)
                .expect("source coefficients"),
            1_792
        );
        assert_eq!(
            geometry
                .physical_matrix_ring_elements(7)
                .expect("physical matrix"),
            1_680
        );
        assert_eq!(geometry.block_coordinates(11).expect("block"), (3, 2));
        assert_eq!(
            geometry.logical_row_coordinates(23, 7).expect("row"),
            (3, 2)
        );
        assert_eq!(geometry.logical_row_index(3, 2, 7).expect("row"), 23);
        assert!(geometry.block_coordinates(13).is_err());
        assert!(geometry.logical_row_coordinates(28, 7).is_err());
    }

    #[test]
    fn one_slice_matches_unsliced_width() {
        let geometry =
            CommitmentSliceGeometry::try_new(CommitmentSliceCount::ONE, 13, 2, 3, 5, 128, 64)
                .expect("unsliced geometry");

        assert_eq!(geometry.block_ranges().len(), 1);
        assert_eq!(geometry.block_ranges()[0], 0..13);
        assert_eq!(geometry.physical_input_width(), 2 * 13 * 3 * 2 * 5);
        assert_eq!(geometry.logical_output_rows(7).expect("rows"), 7);
    }

    #[test]
    fn geometry_rejects_malformed_and_overflowing_inputs() {
        assert!(
            CommitmentSliceGeometry::try_new(CommitmentSliceCount::ONE, 1, 1, 1, 1, 64, 128,)
                .is_err()
        );
        assert!(CommitmentSliceGeometry::try_new(
            CommitmentSliceCount::ONE,
            1,
            usize::MAX,
            2,
            1,
            64,
            64,
        )
        .is_err());
    }

    #[test]
    fn slice_and_chunk_intersections_are_the_finer_partition() {
        for num_live_blocks in 1usize..=512 {
            for slice_count in CommitmentSliceCount::ALL {
                if slice_count.get() > num_live_blocks {
                    continue;
                }
                let slices = slice_count
                    .block_ranges(num_live_blocks)
                    .expect("slice ranges");
                assert!(slices.iter().all(|range| !range.is_empty()));
                assert_eq!(
                    slices
                        .iter()
                        .flat_map(|range| range.clone())
                        .collect::<Vec<_>>(),
                    (0..num_live_blocks).collect::<Vec<_>>()
                );
                for chunk_count in [1usize, 2, 4, 8] {
                    let chunks =
                        dyadic_block_ranges(num_live_blocks, chunk_count).expect("chunk ranges");
                    let finer = if slice_count.get() >= chunk_count {
                        &slices
                    } else {
                        &chunks
                    };
                    let intersections = slices
                        .iter()
                        .flat_map(|slice| {
                            chunks.iter().filter_map(move |chunk| {
                                let start = slice.start.max(chunk.start);
                                let end = slice.end.min(chunk.end);
                                (start < end).then_some(start..end)
                            })
                        })
                        .collect::<Vec<_>>();
                    let expected = finer
                        .iter()
                        .filter(|range| !range.is_empty())
                        .cloned()
                        .collect::<Vec<_>>();
                    assert_eq!(
                        intersections,
                        expected,
                        "B={num_live_blocks}, S={}, W={chunk_count}",
                        slice_count.get()
                    );
                }
            }
        }
    }
}
