//! Canonical witness ranges shared by the planner, prover, and verifier.
//!
//! [`ChunkedWitnessCfg`] describes the multi-chunk witness layout used by the
//! distributed prover: how many chunks the witness is split into and for how
//! many leading fold levels the chunked layout stays active before the schedule
//! reverts to single-chunk sizing.

use std::ops::Range;

use akita_field::AkitaError;

use crate::{LevelParams, OpeningClaimsLayout};

/// One physical `[z_hat | e_hat | t_hat]` group-and-shard unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessUnitLayout {
    group_index: usize,
    shard_index: usize,
    global_fold_start: usize,
    live_fold_count: usize,
    z_range: Range<usize>,
    e_range: Range<usize>,
    t_range: Range<usize>,
}

/// Canonical physical witness descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessLayout {
    units: Vec<WitnessUnitLayout>,
    r_range: Range<usize>,
}

impl WitnessUnitLayout {
    #[cfg(test)]
    pub(crate) fn new_for_test(
        group_index: usize,
        shard_index: usize,
        global_fold_start: usize,
        live_fold_count: usize,
        z_range: Range<usize>,
        e_range: Range<usize>,
        t_range: Range<usize>,
    ) -> Self {
        Self {
            group_index,
            shard_index,
            global_fold_start,
            live_fold_count,
            z_range,
            e_range,
            t_range,
        }
    }

    pub fn group_index(&self) -> usize {
        self.group_index
    }

    pub fn shard_index(&self) -> usize {
        self.shard_index
    }

    pub fn global_fold_start(&self) -> usize {
        self.global_fold_start
    }

    pub fn live_fold_count(&self) -> usize {
        self.live_fold_count
    }

    pub fn global_fold_range(&self) -> Range<usize> {
        self.global_fold_start..self.global_fold_start + self.live_fold_count
    }

    pub fn z_range(&self) -> Range<usize> {
        self.z_range.clone()
    }

    pub fn e_range(&self) -> Range<usize> {
        self.e_range.clone()
    }

    pub fn t_range(&self) -> Range<usize> {
        self.t_range.clone()
    }
}

impl WitnessLayout {
    #[cfg(test)]
    pub(crate) fn new_for_test(units: Vec<WitnessUnitLayout>, r_range: Range<usize>) -> Self {
        Self { units, r_range }
    }

    /// Resolve exact group-major, shard-minor witness ranges from the canonical
    /// level parameters and opening claims layout.
    pub fn new(
        lp: &LevelParams,
        opening_batch: &OpeningClaimsLayout,
        num_shards: usize,
        relation_rows: usize,
        quotient_depth: usize,
    ) -> Result<Self, AkitaError> {
        let num_groups = opening_batch.num_groups();
        if num_groups == 0 || num_shards == 0 || quotient_depth == 0 {
            return Err(AkitaError::InvalidSetup(
                "witness layout requires non-empty groups, shards, and quotient depth".into(),
            ));
        }
        if num_shards > MAX_WITNESS_CHUNKS {
            return Err(AkitaError::InvalidSetup(
                "witness shard count exceeds verifier cap".into(),
            ));
        }
        lp.validate_root_opening_batch(opening_batch)?;
        let relation_group_order = opening_batch.root_group_order()?;

        let mut units = Vec::with_capacity(
            num_groups
                .checked_mul(num_shards)
                .ok_or_else(|| AkitaError::InvalidSetup("witness unit count overflow".into()))?,
        );
        let mut cursor = 0usize;
        for group_index in relation_group_order {
            let params = lp.root_group_params(opening_batch, group_index)?;
            let group = opening_batch.group_layout(group_index)?;
            let num_claims = group.num_polynomials();
            let live_fold_count = params.live_fold_count();
            let shard_granule = params.shard_granule();
            let depth_open = params.num_digits_open();
            let depth_commit = params.num_digits_commit();
            let depth_fold =
                lp.num_digits_fold_for_params(params, num_claims, lp.field_bits_for_cache())?;
            if num_claims == 0
                || live_fold_count == 0
                || params.fold_position_count() == 0
                || depth_open == 0
                || depth_commit == 0
                || depth_fold == 0
                || params.a_rows_len() == 0
                || shard_granule == 0
                || !shard_granule.is_power_of_two()
            {
                return Err(AkitaError::InvalidSetup(
                    "witness group has malformed dimensions".into(),
                ));
            }
            if num_shards
                .checked_mul(shard_granule)
                .is_none_or(|minimum| minimum > live_fold_count)
            {
                return Err(AkitaError::InvalidSetup(
                    "witness shards exceed the live fold granules".into(),
                ));
            }

            let full_granules = live_fold_count / shard_granule;
            let residual = live_fold_count % shard_granule;
            let base_granules = full_granules / num_shards;
            let extra_granules = full_granules % num_shards;
            let z_len = checked_mul3(
                params.fold_position_count(),
                depth_commit,
                depth_fold,
                "witness Z width overflow",
            )?;
            let mut global_fold_start = 0usize;
            for shard_index in 0..num_shards {
                let owned_granules = base_granules + usize::from(shard_index < extra_granules);
                let mut shard_live_fold_count =
                    owned_granules.checked_mul(shard_granule).ok_or_else(|| {
                        AkitaError::InvalidSetup("witness shard width overflow".into())
                    })?;
                if shard_index + 1 == num_shards {
                    shard_live_fold_count =
                        shard_live_fold_count.checked_add(residual).ok_or_else(|| {
                            AkitaError::InvalidSetup("witness shard width overflow".into())
                        })?;
                }
                let e_len = checked_mul3(
                    num_claims,
                    shard_live_fold_count,
                    depth_open,
                    "witness E width overflow",
                )?;
                let t_len = num_claims
                    .checked_mul(shard_live_fold_count)
                    .and_then(|n| n.checked_mul(params.a_rows_len()))
                    .and_then(|n| n.checked_mul(depth_open))
                    .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".into()))?;
                let z_range = checked_range(cursor, z_len, "witness Z range overflow")?;
                let e_range = checked_range(z_range.end, e_len, "witness E range overflow")?;
                let t_range = checked_range(e_range.end, t_len, "witness T range overflow")?;
                cursor = t_range.end;
                units.push(WitnessUnitLayout {
                    group_index,
                    shard_index,
                    global_fold_start,
                    live_fold_count: shard_live_fold_count,
                    z_range,
                    e_range,
                    t_range,
                });
                global_fold_start = global_fold_start
                    .checked_add(shard_live_fold_count)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("witness fold coverage overflow".into())
                    })?;
            }
            if global_fold_start != live_fold_count {
                return Err(AkitaError::InvalidSetup(
                    "witness shards do not cover the live folds".into(),
                ));
            }
        }
        let r_len = relation_rows
            .checked_mul(quotient_depth)
            .ok_or_else(|| AkitaError::InvalidSetup("witness R width overflow".into()))?;
        let r_range = checked_range(cursor, r_len, "witness R range overflow")?;
        Ok(Self { units, r_range })
    }

    pub fn units(&self) -> &[WitnessUnitLayout] {
        &self.units
    }

    pub fn first_group_index(&self) -> Result<usize, AkitaError> {
        self.units
            .first()
            .map(WitnessUnitLayout::group_index)
            .ok_or_else(|| AkitaError::InvalidSetup("witness layout has no units".into()))
    }

    pub fn num_groups(&self) -> usize {
        self.units
            .iter()
            .map(WitnessUnitLayout::group_index)
            .max()
            .map_or(0, |max| max + 1)
    }

    pub fn r_range(&self) -> Range<usize> {
        self.r_range.clone()
    }

    pub fn total_len(&self) -> usize {
        self.r_range.end
    }

    pub fn num_shards_for_group(&self, group_index: usize) -> usize {
        self.units
            .iter()
            .filter(|unit| unit.group_index == group_index)
            .count()
    }

    pub fn group_live_fold_count(&self, group_index: usize) -> Result<usize, AkitaError> {
        let mut total = 0usize;
        let mut found = false;
        for unit in self
            .units
            .iter()
            .filter(|unit| unit.group_index == group_index)
        {
            found = true;
            total = total
                .checked_add(unit.live_fold_count)
                .ok_or_else(|| AkitaError::InvalidSetup("witness fold coverage overflow".into()))?;
        }
        if !found {
            return Err(AkitaError::InvalidSetup("witness group is missing".into()));
        }
        Ok(total)
    }

    pub fn unit(
        &self,
        group_index: usize,
        shard_index: usize,
    ) -> Result<&WitnessUnitLayout, AkitaError> {
        self.units
            .iter()
            .find(|unit| unit.group_index == group_index && unit.shard_index == shard_index)
            .ok_or_else(|| AkitaError::InvalidSetup("witness unit is missing".into()))
    }

    pub fn units_for_group(
        &self,
        group_index: usize,
    ) -> Result<Vec<&WitnessUnitLayout>, AkitaError> {
        let units = self
            .units
            .iter()
            .filter(|unit| unit.group_index == group_index)
            .collect::<Vec<_>>();
        if units.is_empty() {
            return Err(AkitaError::InvalidSetup("witness group is missing".into()));
        }
        Ok(units)
    }

    pub fn unit_for_fold(
        &self,
        group_index: usize,
        global_fold: usize,
    ) -> Result<&WitnessUnitLayout, AkitaError> {
        self.units
            .iter()
            .filter(|unit| unit.group_index == group_index)
            .find(|unit| unit.global_fold_range().contains(&global_fold))
            .ok_or_else(|| AkitaError::InvalidInput("witness fold has no owning unit".into()))
    }

    pub fn e_index(
        &self,
        unit: &WitnessUnitLayout,
        num_claims: usize,
        depth_open: usize,
        claim: usize,
        global_fold: usize,
        digit: usize,
    ) -> Result<usize, AkitaError> {
        self.validate_unit_membership(unit)?;
        let expected_len = checked_mul3(
            num_claims,
            unit.live_fold_count,
            depth_open,
            "witness E shape overflow",
        )?;
        if unit.e_range.len() != expected_len {
            return Err(AkitaError::InvalidSetup(
                "witness E shape disagrees with resolved range".into(),
            ));
        }
        let local_fold = checked_owned_fold(unit, global_fold)?;
        if claim >= num_claims || digit >= depth_open {
            return Err(AkitaError::InvalidInput(
                "witness E semantic index out of range".into(),
            ));
        }
        let fold_claim = unit
            .live_fold_count
            .checked_mul(claim)
            .and_then(|base| base.checked_add(local_fold))
            .ok_or_else(|| AkitaError::InvalidSetup("witness E index overflow".into()))?;
        let local = depth_open
            .checked_mul(fold_claim)
            .and_then(|base| base.checked_add(digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness E index overflow".into()))?;
        checked_range_index(&unit.e_range, local, "witness E")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn t_index(
        &self,
        unit: &WitnessUnitLayout,
        num_claims: usize,
        n_a: usize,
        depth_open: usize,
        claim: usize,
        global_fold: usize,
        a_row: usize,
        digit: usize,
    ) -> Result<usize, AkitaError> {
        self.validate_unit_membership(unit)?;
        let expected_len = num_claims
            .checked_mul(unit.live_fold_count)
            .and_then(|len| len.checked_mul(n_a))
            .and_then(|len| len.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("witness T shape overflow".into()))?;
        if unit.t_range.len() != expected_len {
            return Err(AkitaError::InvalidSetup(
                "witness T shape disagrees with resolved range".into(),
            ));
        }
        let local_fold = checked_owned_fold(unit, global_fold)?;
        if claim >= num_claims || a_row >= n_a || digit >= depth_open {
            return Err(AkitaError::InvalidInput(
                "witness T semantic index out of range".into(),
            ));
        }
        let fold_claim = unit
            .live_fold_count
            .checked_mul(claim)
            .and_then(|base| base.checked_add(local_fold))
            .ok_or_else(|| AkitaError::InvalidSetup("witness T index overflow".into()))?;
        let row_fold_claim = n_a
            .checked_mul(fold_claim)
            .and_then(|base| base.checked_add(a_row))
            .ok_or_else(|| AkitaError::InvalidSetup("witness T index overflow".into()))?;
        let local = depth_open
            .checked_mul(row_fold_claim)
            .and_then(|base| base.checked_add(digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness T index overflow".into()))?;
        checked_range_index(&unit.t_range, local, "witness T")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn z_index(
        &self,
        unit: &WitnessUnitLayout,
        fold_position_count: usize,
        depth_commit: usize,
        depth_fold: usize,
        position: usize,
        commit_digit: usize,
        fold_digit: usize,
    ) -> Result<usize, AkitaError> {
        self.validate_unit_membership(unit)?;
        let expected_len = checked_mul3(
            fold_position_count,
            depth_commit,
            depth_fold,
            "witness Z shape overflow",
        )?;
        if unit.z_range.len() != expected_len {
            return Err(AkitaError::InvalidSetup(
                "witness Z shape disagrees with resolved range".into(),
            ));
        }
        if position >= fold_position_count
            || commit_digit >= depth_commit
            || fold_digit >= depth_fold
        {
            return Err(AkitaError::InvalidInput(
                "witness Z semantic index out of range".into(),
            ));
        }
        let position_commit = depth_commit
            .checked_mul(position)
            .and_then(|base| base.checked_add(commit_digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness Z index overflow".into()))?;
        let local = depth_fold
            .checked_mul(position_commit)
            .and_then(|base| base.checked_add(fold_digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness Z index overflow".into()))?;
        checked_range_index(&unit.z_range, local, "witness Z")
    }

    pub fn r_index(
        &self,
        quotient_depth: usize,
        relation_row: usize,
        quotient_digit: usize,
    ) -> Result<usize, AkitaError> {
        if quotient_depth == 0
            || !self.r_range.len().is_multiple_of(quotient_depth)
            || relation_row >= self.r_range.len() / quotient_depth
            || quotient_digit >= quotient_depth
        {
            return Err(AkitaError::InvalidInput(
                "witness R semantic index out of range".into(),
            ));
        }
        let local = quotient_depth
            .checked_mul(relation_row)
            .and_then(|base| base.checked_add(quotient_digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness R index overflow".into()))?;
        checked_range_index(&self.r_range, local, "witness R")
    }

    pub fn r_offset(&self) -> usize {
        self.r_range.start
    }

    fn validate_unit_membership(&self, unit: &WitnessUnitLayout) -> Result<(), AkitaError> {
        if !self.units.contains(unit) {
            return Err(AkitaError::InvalidInput(
                "witness unit does not belong to this layout".into(),
            ));
        }
        Ok(())
    }
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
            "{name} semantic index exceeds its unit range"
        )));
    }
    Ok(index)
}

fn checked_owned_fold(unit: &WitnessUnitLayout, global_fold: usize) -> Result<usize, AkitaError> {
    let local = global_fold
        .checked_sub(unit.global_fold_start)
        .ok_or_else(|| AkitaError::InvalidInput("witness fold is not owned by unit".into()))?;
    if local >= unit.live_fold_count {
        return Err(AkitaError::InvalidInput(
            "witness fold is not owned by unit".into(),
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
    use crate::SisModulusFamily;

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

    fn test_layout(num_shards: usize) -> (LevelParams, OpeningClaimsLayout, WitnessLayout) {
        let mut lp = LevelParams::params_only(
            SisModulusFamily::Q32,
            32,
            2,
            1,
            1,
            1,
            akita_challenges::SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 25, 2, 2)
        .expect("test params");
        lp.shard_granule = 2;
        let opening_batch = OpeningClaimsLayout::new(0, 2).expect("opening batch");
        let layout =
            WitnessLayout::new(&lp, &opening_batch, num_shards, 3, 2).expect("witness layout");
        (lp, opening_batch, layout)
    }

    #[test]
    fn layout_indexing_matches_digit_innermost_semantics() {
        let (lp, opening_batch, layout) = test_layout(2);
        let unit = layout.unit(0, 1).expect("unit");
        let depth_fold = lp
            .num_digits_fold(2, lp.field_bits_for_cache())
            .expect("fold depth");
        assert_eq!(unit.global_fold_range(), 4..7);
        assert_eq!(
            layout.e_index(unit, 2, 2, 1, 6, 1).expect("e"),
            unit.e_range().start + 1 + 2 * (2 + 3)
        );
        assert_eq!(
            layout.t_index(unit, 2, 1, 2, 0, 5, 0, 1).expect("t"),
            unit.t_range().start + 1 + 2
        );
        assert_eq!(
            layout.z_index(unit, 4, 2, depth_fold, 1, 1, 0).expect("z"),
            unit.z_range().start + depth_fold * (1 + 2)
        );
        assert_eq!(
            layout.r_index(2, 2, 1).expect("r"),
            layout.r_range().start + 5
        );
        assert_eq!(opening_batch.num_total_polynomials(), 2);
    }

    #[test]
    fn granule_aligned_shards_are_exact_and_contiguous() {
        let (_, _, layout) = test_layout(2);
        let units = layout.units_for_group(0).expect("units");
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].global_fold_range(), 0..4);
        assert_eq!(units[1].global_fold_range(), 4..7);
        assert_eq!(units[0].t_range().end, units[1].z_range().start);
        assert_eq!(units[1].t_range().end, layout.r_range().start);
        assert_eq!(layout.group_live_fold_count(0).expect("fold count"), 7);
    }

    #[test]
    fn layout_rejects_out_of_range_semantic_indices() {
        let (lp, _, layout) = test_layout(2);
        let unit = layout.unit(0, 0).expect("unit");
        let depth_fold = lp
            .num_digits_fold(2, lp.field_bits_for_cache())
            .expect("fold depth");
        assert!(layout.e_index(unit, 2, 2, 2, 0, 0).is_err());
        assert!(layout.t_index(unit, 2, 1, 2, 0, 0, 1, 0).is_err());
        assert!(layout.z_index(unit, 4, 2, depth_fold, 4, 0, 0).is_err());
        assert!(layout.r_index(2, 3, 0).is_err());
    }

    #[test]
    fn layout_rejects_mismatched_shapes_and_foreign_units() {
        let (_, _, layout) = test_layout(2);
        let unit = layout.unit(0, 0).expect("unit");
        assert!(layout.e_index(unit, 1, 2, 0, 0, 0).is_err());
        assert!(layout.t_index(unit, 2, 2, 2, 0, 0, 0, 0).is_err());
        assert!(layout.z_index(unit, 1, 1, 1, 0, 0, 0).is_err());

        let foreign = WitnessUnitLayout::new_for_test(0, 0, 0, 1, 0..1, 1..2, 2..3);
        assert!(layout.e_index(&foreign, 1, 1, 0, 0, 0).is_err());
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
