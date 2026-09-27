use super::compiler::RelationWeightGroupPlan;
use super::lane_weights::{lane_windows_mut, LaneWindow};
use akita_error::AkitaError;
use akita_types::{WitnessLayout, WitnessUnitLayout};
use jolt_field::Field;
use std::ops::Range;

/// Blocks whose E/T addresses one task compiles.
const ET_BLOCKS_PER_TASK: usize = 8;

/// Positions whose Z addresses one task compiles.
const Z_POSITIONS_PER_TASK: usize = 64;

/// Consecutive blocks of one witness unit under one claim.
pub(super) struct EtBlockRange<'a> {
    pub(super) unit: &'a WitnessUnitLayout,
    pub(super) claim: usize,
    pub(super) blocks: Range<usize>,
}

/// One E/T scatter task: a block range with its E and T windows.
pub(super) struct EtScatterTask<'l, 'w, E> {
    pub(super) range: EtBlockRange<'l>,
    pub(super) e: LaneWindow<'w, E>,
    pub(super) t: LaneWindow<'w, E>,
}

pub(super) struct ZScatterTask<'l, 'w, E> {
    pub(super) range: ZPositionRange<'l>,
    pub(super) z: LaneWindow<'w, E>,
}

/// Consecutive positions of one witness unit.
pub(super) struct ZPositionRange<'a> {
    pub(super) unit: &'a WitnessUnitLayout,
    pub(super) positions: Range<usize>,
}

impl<E: Field> RelationWeightGroupPlan<E> {
    /// Every `(claim, block)` of this group, in ranges of at most
    /// `max_blocks` blocks that share an owning unit.
    fn et_block_ranges<'a>(
        &self,
        witness_layout: &'a WitnessLayout,
        max_blocks: usize,
    ) -> Result<Vec<EtBlockRange<'a>>, AkitaError> {
        if max_blocks == 0 || self.rows.a_row_weights.len() != self.witness.n_a {
            return Err(AkitaError::InvalidProof);
        }
        let mut ranges: Vec<EtBlockRange<'a>> = Vec::new();
        for claim in 0..self.witness.num_claims {
            for block in 0..self.witness.num_live_blocks {
                let unit = witness_layout.unit_for_block(self.group_index, block)?;
                match ranges.last_mut() {
                    Some(range)
                        if range.claim == claim
                            && std::ptr::eq(range.unit, unit)
                            && range.blocks.len() < max_blocks =>
                    {
                        range.blocks.end = block + 1;
                    }
                    _ => ranges.push(EtBlockRange {
                        unit,
                        claim,
                        blocks: block..block + 1,
                    }),
                }
            }
        }
        Ok(ranges)
    }

    /// Every position of every unit of this group, in ranges of at most
    /// `max_positions` positions.
    fn z_position_ranges<'a>(
        &self,
        witness_layout: &'a WitnessLayout,
        max_positions: usize,
    ) -> Result<Vec<ZPositionRange<'a>>, AkitaError> {
        if max_positions == 0 {
            return Err(AkitaError::InvalidProof);
        }
        let mut ranges = Vec::new();
        for unit in witness_layout.units_for_group(self.group_index)? {
            let mut start = 0;
            while start < self.witness.num_positions {
                let end = start
                    .saturating_add(max_positions)
                    .min(self.witness.num_positions);
                ranges.push(ZPositionRange {
                    unit,
                    positions: start..end,
                });
                start = end;
            }
        }
        Ok(ranges)
    }

    /// Physical E and T coefficients from the first to the last address of
    /// `range`.
    ///
    /// Sinks reject any address outside these extents.
    fn et_extents(&self, range: &EtBlockRange<'_>) -> Result<[Range<usize>; 2], AkitaError> {
        let last_block = range.blocks.end.checked_sub(1);
        let e = match (
            last_block,
            self.roles.d_subcolumns.checked_sub(1),
            self.witness.depth_open.checked_sub(1),
        ) {
            (Some(last_block), Some(last_subcolumn), Some(last_digit)) => {
                let e_index = |block, subcolumn, digit| {
                    range.unit.e_coefficient_index(
                        self.roles.d_d,
                        self.witness.num_claims,
                        self.witness.depth_open,
                        range.claim,
                        block,
                        subcolumn,
                        digit,
                        0,
                    )
                };
                address_extent(
                    e_index(range.blocks.start, 0, 0)?,
                    e_index(last_block, last_subcolumn, last_digit)?,
                    self.roles.d_d,
                )?
            }
            _ => 0..0,
        };
        let t = match (
            last_block,
            self.witness.n_a.checked_sub(1),
            self.roles.b_subcolumns.checked_sub(1),
            self.witness.depth_commit.checked_sub(1),
        ) {
            (Some(last_block), Some(last_row), Some(last_subcolumn), Some(last_digit)) => {
                let t_index = |block, a_row, subcolumn, digit| {
                    range.unit.t_coefficient_index(
                        self.roles.d_a,
                        self.roles.d_b,
                        self.witness.num_claims,
                        self.witness.n_a,
                        self.witness.depth_commit,
                        range.claim,
                        block,
                        a_row,
                        subcolumn,
                        digit,
                        0,
                    )
                };
                address_extent(
                    t_index(range.blocks.start, 0, 0, 0)?,
                    t_index(last_block, last_row, last_subcolumn, last_digit)?,
                    self.roles.d_b,
                )?
            }
            _ => 0..0,
        };
        Ok([e, t])
    }

    /// Physical Z coefficients from the first to the last address of `range`.
    ///
    /// Sinks reject any address outside this extent.
    fn z_extent(&self, range: &ZPositionRange<'_>) -> Result<Range<usize>, AkitaError> {
        match (
            range.positions.end.checked_sub(1),
            self.witness.depth_witness.checked_sub(1),
            self.witness.depth_fold.checked_sub(1),
        ) {
            (Some(last_position), Some(last_witness_digit), Some(last_fold_digit)) => {
                let z_index = |position, witness_digit, fold_digit| {
                    range.unit.z_coefficient_index(
                        self.roles.d_a,
                        self.witness.num_positions,
                        self.witness.depth_witness,
                        self.witness.depth_fold,
                        position,
                        witness_digit,
                        fold_digit,
                        0,
                    )
                };
                address_extent(
                    z_index(range.positions.start, 0, 0)?,
                    z_index(last_position, last_witness_digit, last_fold_digit)?,
                    self.roles.d_a,
                )
            }
            _ => Ok(0..0),
        }
    }

    /// E/T scatter tasks over `lanes`, a table of `lane_len`-coefficient
    /// lanes: every range of `ET_BLOCKS_PER_TASK` blocks with the windows over
    /// its E and T extents.
    pub(super) fn et_scatter_tasks<'l, 'w>(
        &self,
        witness_layout: &'l WitnessLayout,
        lanes: &'w mut [E],
        lane_len: usize,
    ) -> Result<Vec<EtScatterTask<'l, 'w, E>>, AkitaError> {
        let ranges = self.et_block_ranges(witness_layout, ET_BLOCKS_PER_TASK)?;
        let extents = ranges
            .iter()
            .map(|range| self.et_extents(range))
            .collect::<Result<Vec<_>, _>>()?;
        let mut windows = lane_windows_mut(lanes, extents.as_flattened(), lane_len)?.into_iter();
        ranges
            .into_iter()
            .map(|range| match (windows.next(), windows.next()) {
                (Some(e), Some(t)) => Ok(EtScatterTask { range, e, t }),
                _ => Err(AkitaError::InvalidProof),
            })
            .collect()
    }

    /// Z scatter tasks over `lanes`, a table of `lane_len`-coefficient lanes:
    /// every range of `Z_POSITIONS_PER_TASK` positions with the window over
    /// its Z extent.
    pub(super) fn z_scatter_tasks<'l, 'w>(
        &self,
        witness_layout: &'l WitnessLayout,
        lanes: &'w mut [E],
        lane_len: usize,
    ) -> Result<Vec<ZScatterTask<'l, 'w, E>>, AkitaError> {
        let ranges = self.z_position_ranges(witness_layout, Z_POSITIONS_PER_TASK)?;
        let extents = ranges
            .iter()
            .map(|range| self.z_extent(range))
            .collect::<Result<Vec<_>, _>>()?;
        let windows = lane_windows_mut(lanes, &extents, lane_len)?;
        Ok(ranges
            .into_iter()
            .zip(windows)
            .map(|(range, z)| ZScatterTask { range, z })
            .collect())
    }
}

/// Coefficients from `first` through the `width` coefficients at `last`.
fn address_extent(first: usize, last: usize, width: usize) -> Result<Range<usize>, AkitaError> {
    last.checked_add(width)
        .filter(|_| first <= last)
        .map(|end| first..end)
        .ok_or(AkitaError::InvalidProof)
}
