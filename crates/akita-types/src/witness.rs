//! Canonical opening-batch witness layout shared by the planner, prover, and verifier.
//!
//! [`ChunkedWitnessCfg`] describes the multi-chunk witness layout used by the
//! distributed prover: how many chunks the witness is split into and for how
//! many leading fold levels the chunked layout stays active before the schedule
//! reverts to single-chunk sizing.

use akita_field::AkitaError;
use std::ops::Range;

/// Checked block geometry for compact storage and its zero-padded opening MLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpeningBlockLayout {
    num_blocks: usize,
    block_len: usize,
    position_stride: usize,
    physical_len: usize,
    opening_len: usize,
}

impl OpeningBlockLayout {
    /// Build the canonical compact-to-opening address mapping.
    ///
    /// # Errors
    ///
    /// Returns an error unless the block count is a non-zero power of two and
    /// all derived lengths fit in `usize`.
    pub fn new(num_blocks: usize, block_len: usize) -> Result<Self, AkitaError> {
        if !num_blocks.is_power_of_two() || block_len == 0 {
            return Err(AkitaError::InvalidSetup(
                "opening block layout requires power-of-two blocks and non-zero block length"
                    .into(),
            ));
        }
        let position_stride = block_len
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("opening position stride overflow".into()))?;
        let physical_len = num_blocks
            .checked_mul(block_len)
            .ok_or_else(|| AkitaError::InvalidSetup("physical opening length overflow".into()))?;
        let opening_len = num_blocks
            .checked_mul(position_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("virtual opening length overflow".into()))?;
        if physical_len.checked_next_power_of_two() != Some(opening_len) {
            return Err(AkitaError::InvalidSetup(
                "virtual opening domain does not match padded physical domain".into(),
            ));
        }
        Ok(Self {
            num_blocks,
            block_len,
            position_stride,
            physical_len,
            opening_len,
        })
    }

    pub fn num_blocks(self) -> usize {
        self.num_blocks
    }

    pub fn block_len(self) -> usize {
        self.block_len
    }

    pub fn position_stride(self) -> usize {
        self.position_stride
    }

    pub fn physical_len(self) -> usize {
        self.physical_len
    }

    pub fn opening_len(self) -> usize {
        self.opening_len
    }

    /// Compact physical address `block * block_len + position`.
    ///
    /// # Errors
    ///
    /// Returns an error when either coordinate is out of range.
    pub fn physical_index(self, block: usize, position: usize) -> Result<usize, AkitaError> {
        if block >= self.num_blocks || position >= self.block_len {
            return Err(AkitaError::InvalidInput(
                "physical opening coordinate out of range".into(),
            ));
        }
        block
            .checked_mul(self.block_len)
            .and_then(|base| base.checked_add(position))
            .ok_or_else(|| AkitaError::InvalidSetup("physical opening address overflow".into()))
    }

    /// Zero-padded MLE address `block * position_stride + position`.
    ///
    /// # Errors
    ///
    /// Returns an error when either live coordinate is out of range.
    pub fn opening_index(self, block: usize, position: usize) -> Result<usize, AkitaError> {
        if block >= self.num_blocks || position >= self.block_len {
            return Err(AkitaError::InvalidInput(
                "virtual opening coordinate out of range".into(),
            ));
        }
        block
            .checked_mul(self.position_stride)
            .and_then(|base| base.checked_add(position))
            .ok_or_else(|| AkitaError::InvalidSetup("virtual opening address overflow".into()))
    }

    /// Map one compact physical index into its zero-padded MLE address.
    ///
    /// # Errors
    ///
    /// Returns an error when `physical_index` is not live.
    pub fn opening_index_for_physical(self, physical_index: usize) -> Result<usize, AkitaError> {
        if physical_index >= self.physical_len {
            return Err(AkitaError::InvalidInput(
                "physical opening index out of range".into(),
            ));
        }
        self.opening_index(
            physical_index / self.block_len,
            physical_index % self.block_len,
        )
    }
}

/// Typed semantic commitment-group axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemanticGroupId(pub usize);

/// Typed machine ownership-chunk axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MachineChunkId(pub usize);

/// Semantic dimensions for one commitment group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningBatchWitnessGroup {
    pub id: SemanticGroupId,
    pub num_claims: usize,
    pub num_blocks: usize,
    pub block_len: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub n_a: usize,
    /// First D-role setup column owned by this group.
    pub e_setup_col_offset: usize,
}

/// One physical `[z | e | t]` ownership stride.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessOwnershipUnit {
    pub group: SemanticGroupId,
    pub machine_chunk: MachineChunkId,
    pub global_block_base: usize,
    pub blocks: usize,
    pub z_range: Range<usize>,
    pub e_range: Range<usize>,
    pub t_range: Range<usize>,
}

/// Canonical physical and semantic witness descriptor.
///
/// Ownership units are emitted in relation processing order.
/// Each unit owns one contiguous `[z | e | t]` stride.
/// The single shared `r` tail follows all ownership units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningBatchWitnessLayout {
    pub groups: Vec<OpeningBatchWitnessGroup>,
    pub machine_chunks: Vec<MachineChunkId>,
    pub transcript_group_order: Vec<SemanticGroupId>,
    pub relation_group_order: Vec<SemanticGroupId>,
    pub ownership_units: Vec<WitnessOwnershipUnit>,
    pub r_range: Range<usize>,
    pub relation_rows: usize,
    pub quotient_depth: usize,
}

impl OpeningBatchWitnessLayout {
    /// Resolve the complete compact witness layout.
    ///
    /// Multi-group and multi-chunk are distinct axes.
    /// Their product remains rejected until the proof protocol supports it.
    pub fn new(
        mut groups: Vec<OpeningBatchWitnessGroup>,
        transcript_group_order: Vec<SemanticGroupId>,
        relation_group_order: Vec<SemanticGroupId>,
        num_machine_chunks: usize,
        relation_rows: usize,
        quotient_depth: usize,
    ) -> Result<Self, AkitaError> {
        if groups.is_empty() || num_machine_chunks == 0 || quotient_depth == 0 {
            return Err(AkitaError::InvalidSetup(
                "witness layout requires non-empty groups, chunks, and quotient depth".into(),
            ));
        }
        if num_machine_chunks > MAX_WITNESS_CHUNKS {
            return Err(AkitaError::InvalidSetup(
                "witness machine chunk count exceeds verifier cap".into(),
            ));
        }
        if groups.len() > 1 && num_machine_chunks > 1 {
            return Err(AkitaError::InvalidSetup(
                "multi-group and multi-chunk witness product is unsupported".into(),
            ));
        }
        validate_group_order(&transcript_group_order, groups.len(), "transcript")?;
        validate_group_order(&relation_group_order, groups.len(), "relation")?;
        for (index, group) in groups.iter().enumerate() {
            if group.id != SemanticGroupId(index)
                || group.num_claims == 0
                || group.num_blocks == 0
                || group.block_len == 0
                || group.depth_open == 0
                || group.depth_commit == 0
                || group.depth_fold == 0
                || group.n_a == 0
            {
                return Err(AkitaError::InvalidSetup(
                    "witness semantic group has malformed dimensions".into(),
                ));
            }
            if groups.len() == 1 && !group.num_blocks.is_multiple_of(num_machine_chunks) {
                return Err(AkitaError::InvalidSetup(
                    "witness machine chunks must divide the block axis".into(),
                ));
            }
        }

        let mut e_setup_cursor = 0usize;
        for &group_id in &relation_group_order {
            let group = groups
                .get_mut(group_id.0)
                .ok_or_else(|| AkitaError::InvalidSetup("witness group id out of range".into()))?;
            group.e_setup_col_offset = e_setup_cursor;
            e_setup_cursor = e_setup_cursor
                .checked_add(checked_mul3(
                    group.num_claims,
                    group.num_blocks,
                    group.depth_open,
                    "witness setup E width overflow",
                )?)
                .ok_or_else(|| AkitaError::InvalidSetup("witness setup E width overflow".into()))?;
        }

        let machine_chunks = (0..num_machine_chunks)
            .map(MachineChunkId)
            .collect::<Vec<_>>();
        let mut ownership_units = Vec::new();
        let mut cursor = 0usize;
        for &group_id in &relation_group_order {
            let group = groups
                .get(group_id.0)
                .ok_or_else(|| AkitaError::InvalidSetup("witness group id out of range".into()))?;
            let chunks_for_group = if groups.len() == 1 {
                num_machine_chunks
            } else {
                1
            };
            let blocks = group.num_blocks / chunks_for_group;
            for chunk_index in 0..chunks_for_group {
                let z_len = checked_mul3(
                    group.block_len,
                    group.depth_commit,
                    group.depth_fold,
                    "witness Z width overflow",
                )?;
                let e_len = checked_mul3(
                    group.num_claims,
                    blocks,
                    group.depth_open,
                    "witness E width overflow",
                )?;
                let t_len = group
                    .num_claims
                    .checked_mul(blocks)
                    .and_then(|n| n.checked_mul(group.n_a))
                    .and_then(|n| n.checked_mul(group.depth_open))
                    .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".into()))?;
                let z_range = checked_range(cursor, z_len, "witness Z range overflow")?;
                let e_range = checked_range(z_range.end, e_len, "witness E range overflow")?;
                let t_range = checked_range(e_range.end, t_len, "witness T range overflow")?;
                cursor = t_range.end;
                ownership_units.push(WitnessOwnershipUnit {
                    group: group_id,
                    machine_chunk: MachineChunkId(chunk_index),
                    global_block_base: chunk_index.checked_mul(blocks).ok_or_else(|| {
                        AkitaError::InvalidSetup("witness block base overflow".into())
                    })?,
                    blocks,
                    z_range,
                    e_range,
                    t_range,
                });
            }
        }
        let r_len = relation_rows
            .checked_mul(quotient_depth)
            .ok_or_else(|| AkitaError::InvalidSetup("witness R width overflow".into()))?;
        let r_range = checked_range(cursor, r_len, "witness R range overflow")?;
        Ok(Self {
            groups,
            machine_chunks,
            transcript_group_order,
            relation_group_order,
            ownership_units,
            r_range,
            relation_rows,
            quotient_depth,
        })
    }

    pub fn num_machine_chunks(&self) -> usize {
        self.machine_chunks.len()
    }

    pub fn total_len(&self) -> usize {
        self.r_range.end
    }

    pub fn group(&self, id: SemanticGroupId) -> Result<&OpeningBatchWitnessGroup, AkitaError> {
        self.groups
            .get(id.0)
            .ok_or_else(|| AkitaError::InvalidSetup("witness group id out of range".into()))
    }

    pub fn ownership_unit(
        &self,
        group: SemanticGroupId,
        chunk: MachineChunkId,
    ) -> Result<&WitnessOwnershipUnit, AkitaError> {
        self.ownership_units
            .iter()
            .find(|unit| unit.group == group && unit.machine_chunk == chunk)
            .ok_or_else(|| AkitaError::InvalidSetup("witness ownership unit is missing".into()))
    }

    /// Ownership units for one semantic commitment group, in machine-chunk order.
    pub fn units_for_group(
        &self,
        group: SemanticGroupId,
    ) -> Result<Vec<&WitnessOwnershipUnit>, AkitaError> {
        self.group(group)?;
        let mut units = self
            .ownership_units
            .iter()
            .filter(|unit| unit.group == group)
            .collect::<Vec<_>>();
        units.sort_by_key(|unit| unit.machine_chunk.0);
        if units.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "witness ownership unit is missing".into(),
            ));
        }
        Ok(units)
    }

    /// Ownership unit containing `global_block` for one semantic group.
    pub fn unit_for_block(
        &self,
        group: SemanticGroupId,
        global_block: usize,
    ) -> Result<&WitnessOwnershipUnit, AkitaError> {
        let descriptor = self.group(group)?;
        if global_block >= descriptor.num_blocks {
            return Err(AkitaError::InvalidInput(
                "witness block index out of range".into(),
            ));
        }
        self.ownership_units
            .iter()
            .find(|unit| {
                unit.group == group
                    && unit
                        .global_block_base
                        .checked_add(unit.blocks)
                        .is_some_and(|end| {
                            global_block >= unit.global_block_base && global_block < end
                        })
            })
            .ok_or_else(|| AkitaError::InvalidSetup("witness block has no ownership unit".into()))
    }

    pub fn e_index(
        &self,
        unit: &WitnessOwnershipUnit,
        claim: usize,
        global_block: usize,
        digit: usize,
    ) -> Result<usize, AkitaError> {
        let group = self.group(unit.group)?;
        let local_block = checked_owned_block(unit, global_block)?;
        if claim >= group.num_claims || digit >= group.depth_open {
            return Err(AkitaError::InvalidInput(
                "witness E semantic index out of range".into(),
            ));
        }
        let block_claim = unit
            .blocks
            .checked_mul(claim)
            .and_then(|base| base.checked_add(local_block))
            .ok_or_else(|| AkitaError::InvalidSetup("witness E index overflow".into()))?;
        let local = group
            .depth_open
            .checked_mul(block_claim)
            .and_then(|base| base.checked_add(digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness E index overflow".into()))?;
        checked_range_index(&unit.e_range, local, "witness E")
    }

    pub fn t_index(
        &self,
        unit: &WitnessOwnershipUnit,
        claim: usize,
        global_block: usize,
        a_row: usize,
        digit: usize,
    ) -> Result<usize, AkitaError> {
        let group = self.group(unit.group)?;
        let local_block = checked_owned_block(unit, global_block)?;
        if claim >= group.num_claims || a_row >= group.n_a || digit >= group.depth_open {
            return Err(AkitaError::InvalidInput(
                "witness T semantic index out of range".into(),
            ));
        }
        let block_claim = local_block
            .checked_add(
                unit.blocks
                    .checked_mul(claim)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness T index overflow".into()))?,
            )
            .ok_or_else(|| AkitaError::InvalidSetup("witness T index overflow".into()))?;
        let row_block_claim = group
            .n_a
            .checked_mul(block_claim)
            .and_then(|base| base.checked_add(a_row))
            .ok_or_else(|| AkitaError::InvalidSetup("witness T index overflow".into()))?;
        let local = group
            .depth_open
            .checked_mul(row_block_claim)
            .and_then(|base| base.checked_add(digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness T index overflow".into()))?;
        checked_range_index(&unit.t_range, local, "witness T")
    }

    pub fn z_index(
        &self,
        unit: &WitnessOwnershipUnit,
        position: usize,
        commit_digit: usize,
        fold_digit: usize,
    ) -> Result<usize, AkitaError> {
        let group = self.group(unit.group)?;
        if position >= group.block_len
            || commit_digit >= group.depth_commit
            || fold_digit >= group.depth_fold
        {
            return Err(AkitaError::InvalidInput(
                "witness Z semantic index out of range".into(),
            ));
        }
        let position_commit = group
            .depth_commit
            .checked_mul(position)
            .and_then(|base| base.checked_add(commit_digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness Z index overflow".into()))?;
        let local = group
            .depth_fold
            .checked_mul(position_commit)
            .and_then(|base| base.checked_add(fold_digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness Z index overflow".into()))?;
        checked_range_index(&unit.z_range, local, "witness Z")
    }

    /// D-role setup column in `(claim, global_block, opening_digit)` order.
    pub fn e_setup_col_index(
        &self,
        group_id: SemanticGroupId,
        claim: usize,
        global_block: usize,
        digit: usize,
    ) -> Result<usize, AkitaError> {
        let group = self.group(group_id)?;
        if claim >= group.num_claims
            || global_block >= group.num_blocks
            || digit >= group.depth_open
        {
            return Err(AkitaError::InvalidInput(
                "setup E semantic index out of range".into(),
            ));
        }
        let block_claim = group
            .num_blocks
            .checked_mul(claim)
            .and_then(|base| base.checked_add(global_block))
            .ok_or_else(|| AkitaError::InvalidSetup("setup E index overflow".into()))?;
        group
            .depth_open
            .checked_mul(block_claim)
            .and_then(|base| base.checked_add(digit))
            .and_then(|local| group.e_setup_col_offset.checked_add(local))
            .ok_or_else(|| AkitaError::InvalidSetup("setup E index overflow".into()))
    }

    /// B-role setup column in `(claim, global_block, A_row, opening_digit)` order.
    pub fn t_setup_col_index(
        &self,
        group_id: SemanticGroupId,
        claim: usize,
        global_block: usize,
        a_row: usize,
        digit: usize,
    ) -> Result<usize, AkitaError> {
        let group = self.group(group_id)?;
        if claim >= group.num_claims
            || global_block >= group.num_blocks
            || a_row >= group.n_a
            || digit >= group.depth_open
        {
            return Err(AkitaError::InvalidInput(
                "setup T semantic index out of range".into(),
            ));
        }
        let block_claim = group
            .num_blocks
            .checked_mul(claim)
            .and_then(|base| base.checked_add(global_block))
            .ok_or_else(|| AkitaError::InvalidSetup("setup T index overflow".into()))?;
        group
            .n_a
            .checked_mul(block_claim)
            .and_then(|base| base.checked_add(a_row))
            .and_then(|base| group.depth_open.checked_mul(base))
            .and_then(|base| base.checked_add(digit))
            .ok_or_else(|| AkitaError::InvalidSetup("setup T index overflow".into()))
    }

    /// A-role setup column in `(position, commitment_digit)` order.
    pub fn z_setup_col_index(
        &self,
        group_id: SemanticGroupId,
        position: usize,
        commit_digit: usize,
    ) -> Result<usize, AkitaError> {
        let group = self.group(group_id)?;
        if position >= group.block_len || commit_digit >= group.depth_commit {
            return Err(AkitaError::InvalidInput(
                "setup Z semantic index out of range".into(),
            ));
        }
        group
            .depth_commit
            .checked_mul(position)
            .and_then(|base| base.checked_add(commit_digit))
            .ok_or_else(|| AkitaError::InvalidSetup("setup Z index overflow".into()))
    }

    pub fn r_index(&self, relation_row: usize, quotient_digit: usize) -> Result<usize, AkitaError> {
        if relation_row >= self.relation_rows || quotient_digit >= self.quotient_depth {
            return Err(AkitaError::InvalidInput(
                "witness R semantic index out of range".into(),
            ));
        }
        let local = quotient_digit
            .checked_add(
                self.quotient_depth
                    .checked_mul(relation_row)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness R index overflow".into()))?,
            )
            .ok_or_else(|| AkitaError::InvalidSetup("witness R index overflow".into()))?;
        checked_range_index(&self.r_range, local, "witness R")
    }

    pub fn r_offset(&self) -> usize {
        self.r_range.start
    }
}

fn validate_group_order(
    order: &[SemanticGroupId],
    num_groups: usize,
    name: &str,
) -> Result<(), AkitaError> {
    if order.len() != num_groups {
        return Err(AkitaError::InvalidSetup(format!(
            "witness {name} group order has the wrong length"
        )));
    }
    let mut seen = vec![false; num_groups];
    for &id in order {
        let Some(slot) = seen.get_mut(id.0) else {
            return Err(AkitaError::InvalidSetup(format!(
                "witness {name} group id out of range"
            )));
        };
        if *slot {
            return Err(AkitaError::InvalidSetup(format!(
                "witness {name} group order contains a duplicate"
            )));
        }
        *slot = true;
    }
    Ok(())
}

fn checked_range(start: usize, len: usize, context: &str) -> Result<Range<usize>, AkitaError> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| AkitaError::InvalidSetup(context.into()))?;
    Ok(start..end)
}

fn checked_range_index(
    range: &Range<usize>,
    local: usize,
    name: &str,
) -> Result<usize, AkitaError> {
    let index = range
        .start
        .checked_add(local)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{name} index overflow")))?;
    if index >= range.end {
        return Err(AkitaError::InvalidInput(format!(
            "{name} semantic index exceeds its ownership range"
        )));
    }
    Ok(index)
}

fn checked_owned_block(
    unit: &WitnessOwnershipUnit,
    global_block: usize,
) -> Result<usize, AkitaError> {
    let local = global_block
        .checked_sub(unit.global_block_base)
        .ok_or_else(|| AkitaError::InvalidInput("witness block is not owned by unit".into()))?;
    if local >= unit.blocks {
        return Err(AkitaError::InvalidInput(
            "witness block is not owned by unit".into(),
        ));
    }
    Ok(local)
}

fn checked_mul3(a: usize, b: usize, c: usize, context: &str) -> Result<usize, AkitaError> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(c))
        .ok_or_else(|| AkitaError::InvalidSetup(context.into()))
}

/// Upper bound on [`ChunkedWitnessCfg::num_chunks`] enforced at layout validation
/// and planner policy entry. Replicated `ẑ` scales witness width linearly in the
/// chunk count; this cap closes a DoS vector from arbitrarily large layouts.
pub const MAX_WITNESS_CHUNKS: usize = 64;

/// Indexed multi-chunk preset on the shipped `num_chunks × num_activated_levels`
/// grid (`num_chunks ∈ {2, 4, 8}`, `num_activated_levels ∈ {1, 2}`).
///
/// `num_chunks` must be a power of two; non-power-of-two chunk counts are rejected
/// by [`ChunkedWitnessCfg::validate`] and are not part of this grid.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultiChunkProfileId {
    /// `num_chunks = 2`, `num_activated_levels = 1`.
    W2R1 = 0,
    /// `num_chunks = 2`, `num_activated_levels = 2`.
    W2R2 = 1,
    /// `num_chunks = 4`, `num_activated_levels = 1`.
    W4R1 = 2,
    /// `num_chunks = 4`, `num_activated_levels = 2`.
    W4R2 = 3,
    /// `num_chunks = 8`, `num_activated_levels = 1`.
    W8R1 = 4,
    /// `num_chunks = 8`, `num_activated_levels = 2` (D64 production default).
    W8R2 = 5,
}

impl MultiChunkProfileId {
    /// Number of profiles in [`Self::ALL`].
    pub const COUNT: usize = 6;

    /// Every supported profile, in stable index order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::W2R1,
        Self::W2R2,
        Self::W4R1,
        Self::W4R2,
        Self::W8R1,
        Self::W8R2,
    ];

    /// Shipped D64 multi-chunk preset (`8` chunks, `2` leading fold levels).
    pub const PRODUCTION: Self = Self::W8R2;

    /// Stable dense index in `0 .. COUNT`.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Resolve a profile from its stable index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= COUNT` (test-only helper; presets use the named
    /// variants or [`Self::PRODUCTION`]).
    pub const fn from_index(index: usize) -> Self {
        assert!(index < Self::COUNT);
        Self::ALL[index]
    }

    pub const fn num_chunks(self) -> usize {
        match self {
            Self::W2R1 | Self::W2R2 => 2,
            Self::W4R1 | Self::W4R2 => 4,
            Self::W8R1 | Self::W8R2 => 8,
        }
    }

    pub const fn num_activated_levels(self) -> usize {
        match self {
            Self::W2R1 | Self::W4R1 | Self::W8R1 => 1,
            Self::W2R2 | Self::W4R2 | Self::W8R2 => 2,
        }
    }

    pub const fn cfg(self) -> ChunkedWitnessCfg {
        ChunkedWitnessCfg {
            num_chunks: self.num_chunks(),
            num_activated_levels: self.num_activated_levels(),
        }
    }
}

/// Chunk-based witness layout parameters.
///
/// `num_chunks = 1` is the single-chunk (standard) case; `num_chunks` must be a
/// power of two. `num_activated_levels` is how many leading protocol levels the
/// multi-chunk layout is active; it is ignored when `num_chunks = 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkedWitnessCfg {
    /// Number of witness chunks / replicated ẑ segments while the multi-chunk
    /// layout is active. `1` means single-chunk (default).
    pub num_chunks: usize,
    /// Count of leading fold levels (absolute levels `0, 1, …, R−1`) priced
    /// under the chunked layout. `0` disables multi-chunk planning.
    pub num_activated_levels: usize,
}

impl Default for ChunkedWitnessCfg {
    fn default() -> Self {
        Self {
            num_chunks: 1,
            num_activated_levels: 0,
        }
    }
}

impl ChunkedWitnessCfg {
    /// Const equivalent of [`Default::default`], usable in const contexts such as
    /// generated catalog-identity literals.
    pub const fn default_non_chunked() -> Self {
        Self {
            num_chunks: 1,
            num_activated_levels: 0,
        }
    }

    /// True iff the planner should price the chunked layout for the leading
    /// levels. Both `num_chunks > 1` and `num_activated_levels > 0` are required;
    /// any other combination is either single-chunk or an invalid config caught
    /// by [`Self::validate`].
    pub const fn uses_multi_chunk(self) -> bool {
        self.num_chunks > 1 && self.num_activated_levels > 0
    }

    /// Shipped D64 multi-chunk preset (`8` chunks, `2` leading fold levels).
    pub const fn d64_production() -> Self {
        MultiChunkProfileId::PRODUCTION.cfg()
    }

    /// Build a config from a canonical [`MultiChunkProfileId`].
    pub const fn from_profile(profile: MultiChunkProfileId) -> Self {
        profile.cfg()
    }

    /// Recover the profile id when this config matches a grid entry.
    pub fn profile_id(self) -> Option<MultiChunkProfileId> {
        MultiChunkProfileId::ALL
            .into_iter()
            .find(|profile| profile.cfg() == self)
    }

    /// Layout-only validation (no dependency on planner internals).
    ///
    /// The depth bound against the planner's recursion cap is enforced
    /// separately at the planner entry, where the constant is in scope.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] for `num_chunks == 0`,
    /// `num_chunks > [`MAX_WITNESS_CHUNKS`]`, a non-power-of-two `num_chunks > 1`,
    /// or an inconsistent `(num_chunks, num_activated_levels)` pair.
    pub fn validate(self) -> Result<(), AkitaError> {
        if self.num_chunks == 0 {
            return Err(AkitaError::InvalidSetup(
                "ChunkedWitnessCfg: num_chunks must be >= 1".to_string(),
            ));
        }
        if self.num_chunks > MAX_WITNESS_CHUNKS {
            return Err(AkitaError::InvalidSetup(format!(
                "ChunkedWitnessCfg: num_chunks={} exceeds cap {MAX_WITNESS_CHUNKS}",
                self.num_chunks
            )));
        }
        if self.num_chunks > 1 && !self.num_chunks.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "ChunkedWitnessCfg: num_chunks must be a power of two".to_string(),
            ));
        }
        if self.num_activated_levels > 0 && self.num_chunks == 1 {
            return Err(AkitaError::InvalidSetup(
                "ChunkedWitnessCfg: num_activated_levels > 0 requires num_chunks > 1".to_string(),
            ));
        }
        if self.num_chunks > 1 && self.num_activated_levels == 0 {
            return Err(AkitaError::InvalidSetup(
                "ChunkedWitnessCfg: num_chunks > 1 requires num_activated_levels > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Append canonical Fiat-Shamir descriptor bytes.
    ///
    /// Only invoked by [`crate::LevelParams`] descriptor binding when the level
    /// is chunked, so single-chunk levels stay byte-for-byte identical to the
    /// historical layout (the flag-off no-op invariant).
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        crate::descriptor_bytes::push_usize(bytes, self.num_chunks);
        crate::descriptor_bytes::push_usize(bytes, self.num_activated_levels);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_single_chunk() {
        let cfg = ChunkedWitnessCfg::default();
        assert_eq!(cfg, ChunkedWitnessCfg::default_non_chunked());
        assert_eq!(cfg.num_chunks, 1);
        assert_eq!(cfg.num_activated_levels, 0);
        assert!(!cfg.uses_multi_chunk());
        cfg.validate().expect("default config is valid");
    }

    #[test]
    fn d64_production_uses_multi_chunk() {
        let cfg = ChunkedWitnessCfg::d64_production();
        assert_eq!(cfg, MultiChunkProfileId::PRODUCTION.cfg());
        assert_eq!(cfg.num_chunks, 8);
        assert_eq!(cfg.num_activated_levels, 2);
        assert!(cfg.uses_multi_chunk());
        cfg.validate().expect("d64_production is valid");
    }

    #[test]
    fn multi_chunk_profile_grid_roundtrip() {
        for (index, profile) in MultiChunkProfileId::ALL.into_iter().enumerate() {
            assert_eq!(profile.index(), index);
            assert_eq!(MultiChunkProfileId::from_index(index), profile);
            let cfg = ChunkedWitnessCfg::from_profile(profile);
            assert_eq!(cfg.profile_id(), Some(profile));
            cfg.validate().expect("grid profile is valid");
        }
    }

    fn test_group() -> OpeningBatchWitnessGroup {
        OpeningBatchWitnessGroup {
            id: SemanticGroupId(0),
            num_claims: 2,
            num_blocks: 4,
            block_len: 3,
            depth_open: 2,
            depth_commit: 2,
            depth_fold: 2,
            n_a: 2,
            e_setup_col_offset: 0,
        }
    }

    #[test]
    fn layout_indexing_matches_compact_semantics() {
        let layout = OpeningBatchWitnessLayout::new(
            vec![test_group()],
            vec![SemanticGroupId(0)],
            vec![SemanticGroupId(0)],
            1,
            3,
            2,
        )
        .expect("layout");
        let unit = layout
            .ownership_unit(SemanticGroupId(0), MachineChunkId(0))
            .expect("unit");
        assert_eq!(
            layout.e_index(unit, 1, 2, 1).expect("e"),
            unit.e_range.start + 1 + 2 * (2 + unit.blocks)
        );
        assert_eq!(
            layout.t_index(unit, 0, 3, 1, 1).expect("t"),
            unit.t_range.start + 1 + 2 * (1 + 2 * 3)
        );
        assert_eq!(
            layout.z_index(unit, 1, 1, 0).expect("z"),
            unit.z_range.start + 2 * (1 + 2)
        );
        assert_eq!(layout.r_index(2, 1).expect("r"), layout.r_range.start + 5);
        assert_eq!(
            layout
                .e_setup_col_index(SemanticGroupId(0), 1, 2, 1)
                .expect("setup e"),
            1 + 2 * (2 + 4)
        );
        assert_eq!(
            layout
                .t_setup_col_index(SemanticGroupId(0), 1, 2, 1, 1)
                .expect("setup t"),
            1 + 2 * (1 + 2 * (2 + 4))
        );
        assert_eq!(
            layout
                .z_setup_col_index(SemanticGroupId(0), 1, 1)
                .expect("setup z"),
            3
        );
    }

    #[test]
    fn layout_rejects_out_of_range_semantic_indices() {
        let layout = OpeningBatchWitnessLayout::new(
            vec![test_group()],
            vec![SemanticGroupId(0)],
            vec![SemanticGroupId(0)],
            1,
            1,
            1,
        )
        .expect("layout");
        let unit = layout
            .ownership_unit(SemanticGroupId(0), MachineChunkId(0))
            .expect("unit");
        assert!(layout.e_index(unit, 2, 0, 0).is_err());
        assert!(layout.t_index(unit, 0, 0, 2, 0).is_err());
        assert!(layout.z_index(unit, 3, 0, 0).is_err());
        assert!(layout.r_index(1, 0).is_err());
    }

    #[test]
    fn validate_rejects_invalid_configs() {
        assert!(ChunkedWitnessCfg {
            num_chunks: 0,
            num_activated_levels: 0,
        }
        .validate()
        .is_err());
        assert!(ChunkedWitnessCfg {
            num_chunks: 6,
            num_activated_levels: 2,
        }
        .validate()
        .is_err());
        assert!(ChunkedWitnessCfg {
            num_chunks: 1,
            num_activated_levels: 2,
        }
        .validate()
        .is_err());
        assert!(ChunkedWitnessCfg {
            num_chunks: 8,
            num_activated_levels: 0,
        }
        .validate()
        .is_err());
        assert!(ChunkedWitnessCfg {
            num_chunks: 128,
            num_activated_levels: 1,
        }
        .validate()
        .is_err());
        ChunkedWitnessCfg {
            num_chunks: MAX_WITNESS_CHUNKS,
            num_activated_levels: 1,
        }
        .validate()
        .expect("max chunk count is valid");
        for n in [2usize, 4, 8, 16] {
            ChunkedWitnessCfg {
                num_chunks: n,
                num_activated_levels: 1,
            }
            .validate()
            .expect("power-of-two chunk counts validate");
        }
    }
}
