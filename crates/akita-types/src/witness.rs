//! Witness-layout configuration shared by the planner, prover, and verifier.
//!
//! [`ChunkedWitnessCfg`] is the planner/catalog policy describing how many
//! machines remain active for how many leading outputs. The checked resolved
//! transition is [`DistributedOwnershipGeometry`]. Until the atomic
//! `LevelParams` cutover lands, legacy protocol code still reads
//! `LevelParams::witness_chunk`; new consumers must not add another derived
//! production authority.
//!
//! `num_chunks = 1` is the single-chunk (standard) case and is byte-identical to
//! the historical layout.

use akita_field::AkitaError;

/// Per-chunk witness segment ring-column counts (emission order `z ‖ e ‖ t ‖ r`).
///
/// `z_len` is **replicated** (the same in every chunk); `e_len`/`t_len` are
/// **partitioned** (each chunk covers `blocks_per_chunk = num_blocks /
/// num_chunks` blocks). `r_len` is `Some` only in the last chunk and `None`
/// elsewhere, so a call site cannot treat an absent segment as length `0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessChunkLengths {
    /// Replicated folded-response width: `num_digits_fold · num_digits_commit · block_len`.
    pub z_len: usize,
    /// Partitioned opening-digit width: `num_digits_open · num_claims · blocks_per_chunk`.
    pub e_len: usize,
    /// Partitioned inner-Ajtai width: `num_digits_open · n_a · num_t_vectors · blocks_per_chunk`.
    pub t_len: usize,
    /// Shared quotient-tail width (`num_rows · r_decomp_levels`); `Some` only in
    /// the last chunk.
    pub r_len: Option<usize>,
}

/// Per-chunk witness segment column offsets.
///
/// `offset_r` mirrors [`WitnessChunkLengths::r_len`]: `None` when the segment is
/// absent from this chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WitnessChunkLayout {
    /// Column offset of the replicated folded response `zᵢ`.
    pub offset_z: usize,
    /// Column offset of the partitioned opening digits `êᵢ`.
    pub offset_e: usize,
    /// Column offset of the partitioned inner-Ajtai digits `t̂ᵢ`.
    pub offset_t: usize,
    /// Column offset of the shared quotient tail; `Some` only in the last chunk.
    pub offset_r: Option<usize>,
    /// First global block index owned by this chunk (`chunk_idx · blocks_per_chunk`).
    pub global_block_base: usize,
}

/// Resolved, layout-agnostic witness column description consumed by the
/// ring-switch row-MLE evaluation and the setup-contribution planner.
///
/// `num_chunks = 1` is the single-chunk (historical) case: one chunk spanning
/// all `num_blocks` with `global_block_base = 0`, byte-identical to the legacy
/// `z ‖ e ‖ t ‖ r` layout. `num_chunks = W` lays out `W` contiguous
/// `[zᵢ | eᵢ | t̂ᵢ]` strides followed by a single shared `r̂` tail.
///
/// `chunks` and `chunk_lengths` are parallel vectors of length `num_chunks`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessLayout {
    /// Blocks owned by each chunk (`num_blocks / num_chunks`); equals
    /// `num_blocks` for the single-chunk case.
    pub blocks_per_chunk: usize,
    /// Per-chunk offsets; `len == num_chunks`.
    pub chunks: Vec<WitnessChunkLayout>,
    /// Per-chunk lengths; parallel to [`Self::chunks`].
    pub chunk_lengths: Vec<WitnessChunkLengths>,
}

impl WitnessLayout {
    /// Number of resolved chunks (`1` for the single-chunk layout).
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// The last chunk's offsets/lengths, which alone carry the shared `r̂` segment.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] if the layout has no chunks (a
    /// malformed layout that resolution should never produce).
    pub fn last_chunk(&self) -> Result<(&WitnessChunkLayout, &WitnessChunkLengths), AkitaError> {
        match (self.chunks.last(), self.chunk_lengths.last()) {
            (Some(layout), Some(lengths)) => Ok((layout, lengths)),
            _ => Err(AkitaError::InvalidSetup(
                "witness layout has no chunks".to_string(),
            )),
        }
    }

    /// Column offset of the shared quotient tail (always carried by the last chunk).
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] if the layout is empty or the last
    /// chunk is missing its `r̂` offset.
    pub fn r_offset(&self) -> Result<usize, AkitaError> {
        let (layout, _) = self.last_chunk()?;
        layout.offset_r.ok_or_else(|| {
            AkitaError::InvalidSetup("last witness chunk is missing the r-tail offset".to_string())
        })
    }
}

/// Upper bound on distributed machine counts enforced at policy, descriptor,
/// and layout boundaries. Replicated partial-fold state scales witness width
/// linearly in this count, so the cap also closes a verifier DoS vector.
pub const MAX_DISTRIBUTED_MACHINES: usize = 64;

/// One machine's contiguous window on a protocol block axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MachineBlockWindow {
    machine: usize,
    global_block_base: usize,
    blocks: usize,
}

impl MachineBlockWindow {
    /// Zero-based machine index in canonical order.
    #[must_use]
    pub const fn machine(self) -> usize {
        self.machine
    }

    /// First global block owned by this machine.
    #[must_use]
    pub const fn global_block_base(self) -> usize {
        self.global_block_base
    }

    /// Number of contiguous global blocks owned by this machine.
    #[must_use]
    pub const fn blocks(self) -> usize {
        self.blocks
    }

    /// Exclusive end of the global block window.
    pub fn global_block_end(self) -> Result<usize, AkitaError> {
        self.global_block_base
            .checked_add(self.blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("machine block window overflow".to_string()))
    }
}

/// Descriptor-bound machine ownership entering and leaving one fold level.
///
/// Input ownership determines which machines hold the current witness. Output
/// ownership determines the machine count of the newly committed recursive
/// witness. Keeping both counts prevents the W-to-1 cutover from being inferred
/// ambiguously from one level-local chunk count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DistributedOwnershipGeometry {
    input_machines: usize,
    output_machines: usize,
}

impl Default for DistributedOwnershipGeometry {
    fn default() -> Self {
        Self::single_machine()
    }
}

impl DistributedOwnershipGeometry {
    /// Construct checked input/output machine ownership.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] unless both counts are non-zero
    /// powers of two at most [`MAX_DISTRIBUTED_MACHINES`].
    pub fn new(input_machines: usize, output_machines: usize) -> Result<Self, AkitaError> {
        validate_machine_count(input_machines, "input")?;
        validate_machine_count(output_machines, "output")?;
        Ok(Self {
            input_machines,
            output_machines,
        })
    }

    /// Ordinary single-machine ownership.
    #[must_use]
    pub const fn single_machine() -> Self {
        Self {
            input_machines: 1,
            output_machines: 1,
        }
    }

    /// Number of machines owning the current fold input.
    #[must_use]
    pub const fn input_machines(self) -> usize {
        self.input_machines
    }

    /// Number of machines owning the newly committed fold output.
    #[must_use]
    pub const fn output_machines(self) -> usize {
        self.output_machines
    }

    /// Whether this fold performs an explicit distributed-to-single cutover.
    #[must_use]
    pub const fn is_cutover(self) -> bool {
        self.input_machines > 1 && self.output_machines == 1
    }

    /// Validate ownership continuity from the preceding fold.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the predecessor's output
    /// machine count differs from this fold's input machine count.
    pub fn validate_predecessor(self, predecessor: Self) -> Result<(), AkitaError> {
        if predecessor.output_machines != self.input_machines {
            return Err(AkitaError::InvalidSetup(format!(
                "distributed ownership discontinuity: predecessor output_machines={} but current input_machines={}",
                predecessor.output_machines, self.input_machines
            )));
        }
        Ok(())
    }

    /// Partition an input block axis into canonical contiguous machine windows.
    pub fn input_block_windows(
        self,
        num_blocks: usize,
    ) -> Result<Vec<MachineBlockWindow>, AkitaError> {
        block_windows(num_blocks, self.input_machines, "input")
    }

    /// Partition an output block axis into canonical contiguous machine windows.
    pub fn output_block_windows(
        self,
        num_blocks: usize,
    ) -> Result<Vec<MachineBlockWindow>, AkitaError> {
        block_windows(num_blocks, self.output_machines, "output")
    }
}

fn validate_machine_count(count: usize, role: &str) -> Result<(), AkitaError> {
    if count == 0 || !count.is_power_of_two() || count > MAX_DISTRIBUTED_MACHINES {
        return Err(AkitaError::InvalidSetup(format!(
            "distributed {role}_machines must be a non-zero power of two at most {MAX_DISTRIBUTED_MACHINES}, got {count}"
        )));
    }
    Ok(())
}

fn block_windows(
    num_blocks: usize,
    machines: usize,
    role: &str,
) -> Result<Vec<MachineBlockWindow>, AkitaError> {
    if num_blocks == 0 || !num_blocks.is_power_of_two() || !num_blocks.is_multiple_of(machines) {
        return Err(AkitaError::InvalidSetup(format!(
            "distributed {role} block count must be a non-zero power of two divisible by machines: blocks={num_blocks}, machines={machines}"
        )));
    }
    let blocks = num_blocks / machines;
    (0..machines)
        .map(|machine| {
            let global_block_base = machine.checked_mul(blocks).ok_or_else(|| {
                AkitaError::InvalidSetup("machine block base overflow".to_string())
            })?;
            Ok(MachineBlockWindow {
                machine,
                global_block_base,
                blocks,
            })
        })
        .collect()
}

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
    /// `num_chunks > [`MAX_DISTRIBUTED_MACHINES`]`, a non-power-of-two `num_chunks > 1`,
    /// or an inconsistent `(num_chunks, num_activated_levels)` pair.
    pub fn validate(self) -> Result<(), AkitaError> {
        if self.num_chunks == 0 {
            return Err(AkitaError::InvalidSetup(
                "ChunkedWitnessCfg: num_chunks must be >= 1".to_string(),
            ));
        }
        if self.num_chunks > MAX_DISTRIBUTED_MACHINES {
            return Err(AkitaError::InvalidSetup(format!(
                "ChunkedWitnessCfg: num_chunks={} exceeds cap {MAX_DISTRIBUTED_MACHINES}",
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
    fn ownership_geometry_partitions_input_and_output_independently() {
        let geometry = DistributedOwnershipGeometry::new(8, 2).unwrap();
        let input = geometry.input_block_windows(32).unwrap();
        let output = geometry.output_block_windows(8).unwrap();
        assert_eq!(input.len(), 8);
        assert_eq!(input[3].global_block_base(), 12);
        assert_eq!(input[3].blocks(), 4);
        assert_eq!(input[3].global_block_end().unwrap(), 16);
        assert_eq!(output.len(), 2);
        assert_eq!(output[1].global_block_base(), 4);
        assert_eq!(output[1].blocks(), 4);
    }

    #[test]
    fn ownership_geometry_validates_cutover_and_predecessor() {
        let distributed = DistributedOwnershipGeometry::new(8, 8).unwrap();
        let cutover = DistributedOwnershipGeometry::new(8, 1).unwrap();
        let suffix = DistributedOwnershipGeometry::single_machine();
        assert!(cutover.is_cutover());
        cutover.validate_predecessor(distributed).unwrap();
        suffix.validate_predecessor(cutover).unwrap();
        assert!(distributed.validate_predecessor(cutover).is_err());
    }

    #[test]
    fn ownership_geometry_rejects_bad_counts_and_windows() {
        for (input, output) in [(0, 1), (1, 0), (3, 1), (1, 6), (128, 1)] {
            assert!(DistributedOwnershipGeometry::new(input, output).is_err());
        }
        let geometry = DistributedOwnershipGeometry::new(8, 2).unwrap();
        assert!(geometry.input_block_windows(4).is_err());
        assert!(geometry.input_block_windows(24).is_err());
        assert!(geometry.output_block_windows(0).is_err());
    }

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
            num_chunks: MAX_DISTRIBUTED_MACHINES,
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
