//! Canonical witness ranges shared by the planner, prover, and verifier.
//!
//! [`ChunkedWitnessCfg`] describes the multi-chunk witness layout used by the
//! distributed prover: how many chunks the witness is split into and for how
//! many leading fold levels the chunked layout stays active before the schedule
//! reverts to single-chunk sizing.

use std::ops::Range;

use akita_error::{checked, AkitaError};

use crate::{
    CommitmentRingDims, CommittedGroupParams, CompressionMapPlan, OpeningClaimsLayout,
    RelationRowGeometry, RelationWitnessGeometry,
};

mod chunk_partition;
mod quotient_breakdown;
mod scalar_len;
mod tail;

pub use chunk_partition::dyadic_block_ranges;
pub use quotient_breakdown::QuotientCoefficientBreakdown;
pub use tail::RelationQuotientPlan;

/// Exact physical coefficient count of the grouped `[Z | E | T]` witness body.
///
/// This excludes relation quotients, compression layers, and alignment. It is
/// the shared sizing authority for runtime witness layout and planner
/// contraction/scoring decisions.
pub fn grouped_witness_body_coefficients(
    params: &crate::GroupOpenPhaseParams,
    source_encoding: crate::CommittedSourceEncoding,
    role_dims: CommitmentRingDims,
    extension_degree: usize,
    num_claims: usize,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    role_dims.validate_role_projection()?;
    if params.inner_commit_matrix_params().ring_dimension() != role_dims.d_a() {
        return Err(AkitaError::InvalidSetup(
            "grouped witness body role dimensions disagree with its parameters".into(),
        ));
    }
    if num_claims == 0
        || params.num_live_blocks() == 0
        || params.num_positions_per_block() == 0
        || params.num_digits_open() == 0
        || params.num_digits_inner() == 0
        || params.num_digits_outer() == 0
        || params.num_digits_fold() == 0
        || params.a_rows_len() == 0
    {
        return Err(AkitaError::InvalidSetup(
            "witness group has malformed dimensions".into(),
        ));
    }
    let opening_geometry =
        crate::proof::relation::opening_row_geometry(params, source_encoding, extension_degree)?;
    let mut total = 0usize;
    for block_range in dyadic_block_ranges(params.num_live_blocks(), num_chunks)? {
        let (z_len, e_len, t_len) = witness_unit_lengths(
            params,
            role_dims,
            opening_geometry,
            num_claims,
            block_range.len(),
        )?;
        total = total
            .checked_add(z_len)
            .and_then(|len| len.checked_add(e_len))
            .and_then(|len| len.checked_add(t_len))
            .ok_or_else(|| AkitaError::InvalidSetup("grouped witness body overflow".into()))?;
    }
    Ok(total)
}

/// One physical `[z_hat | e_hat | t_hat]` group-and-chunk unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessUnitLayout {
    group_index: usize,
    chunk_index: usize,
    global_block_start: usize,
    num_live_blocks: usize,
    z_range: Range<usize>,
    e_range: Range<usize>,
    e_geometry: RelationRowGeometry,
    t_range: Range<usize>,
}

/// Canonical physical witness descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessLayout {
    units: Vec<WitnessUnitLayout>,
    compression_layers: Vec<CompressionWitnessLayerLayout>,
    compression_alignment_ranges: Vec<Range<usize>>,
    relation_quotients: RelationQuotientLayout,
    /// Envelope containing everything after the chunk-major Z/E/T body.
    tail_range: Range<usize>,
}

/// Mode-aware ownership of polynomial-modulus quotient witness rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationQuotientLayout {
    /// Ordinary and compression relation rows are represented by quotient
    /// digits at one shared decomposition depth.
    QuotientLift {
        quotient_depth: std::num::NonZeroUsize,
        rows: Vec<WitnessQuotientRowLayout>,
    },
    /// Ring equalities are checked after negacyclic reduction, so no quotient
    /// row is part of the committed witness.
    ReducedEvaluation,
}

/// One padded negative-binary digit span in the global witness tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionWitnessSpan {
    map: CompressionMapPlan,
    range: Range<usize>,
}

/// One layer-major family of F spans followed by the shared H span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionWitnessLayerLayout {
    map_index: usize,
    f_spans: Vec<(usize, CompressionWitnessSpan)>,
    h_span: CompressionWitnessSpan,
    f_quotient_rows: Option<Vec<(usize, usize)>>,
    h_quotient_row: Option<usize>,
}

/// One native relation-quotient row in the shared R tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessQuotientRowLayout {
    geometry: RelationRowGeometry,
    range: Range<usize>,
}

impl WitnessUnitLayout {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_for_test(
        group_index: usize,
        chunk_index: usize,
        global_block_start: usize,
        num_live_blocks: usize,
        z_range: Range<usize>,
        e_range: Range<usize>,
        e_geometry: RelationRowGeometry,
        t_range: Range<usize>,
    ) -> Self {
        Self {
            group_index,
            chunk_index,
            global_block_start,
            num_live_blocks,
            z_range,
            e_range,
            e_geometry,
            t_range,
        }
    }

    pub fn group_index(&self) -> usize {
        self.group_index
    }

    pub fn chunk_index(&self) -> usize {
        self.chunk_index
    }

    pub fn global_block_start(&self) -> usize {
        self.global_block_start
    }

    pub fn num_live_blocks(&self) -> usize {
        self.num_live_blocks
    }

    pub fn global_block_range(&self) -> Range<usize> {
        self.global_block_start..self.global_block_start + self.num_live_blocks
    }

    pub fn z_range(&self) -> Range<usize> {
        self.z_range.clone()
    }

    pub fn e_range(&self) -> Range<usize> {
        self.e_range.clone()
    }

    #[must_use]
    pub const fn e_geometry(&self) -> RelationRowGeometry {
        self.e_geometry
    }

    pub fn t_range(&self) -> Range<usize> {
        self.t_range.clone()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn e_coefficient_index(
        &self,
        role_ring_dim: usize,
        num_claims: usize,
        depth_open: usize,
        claim: usize,
        global_block: usize,
        role_subcolumn: usize,
        digit: usize,
        role_coefficient: usize,
    ) -> Result<usize, AkitaError> {
        let expected_len = checked::product([num_claims, self.num_live_blocks, depth_open])
            .ok_or_else(|| AkitaError::InvalidSetup("witness E shape overflow".into()))?
            .checked_mul(self.e_geometry.physical_coefficient_width())
            .ok_or_else(|| AkitaError::InvalidSetup("witness E shape overflow".into()))?;
        if self.e_range.len() != expected_len {
            return Err(AkitaError::InvalidSetup(
                "witness E shape disagrees with resolved range".into(),
            ));
        }
        let local_block = checked_owned_block(self, global_block)?;
        if claim >= num_claims || digit >= depth_open {
            return Err(AkitaError::InvalidInput(
                "witness E semantic index out of range".into(),
            ));
        }
        let block_claim = self
            .num_live_blocks
            .checked_mul(claim)
            .and_then(|base| base.checked_add(local_block))
            .ok_or_else(|| AkitaError::InvalidSetup("witness E index overflow".into()))?;
        projected_coefficient_index(
            &self.e_range,
            "witness E",
            self.e_geometry.physical_coefficient_width(),
            role_ring_dim,
            num_claims
                .checked_mul(self.num_live_blocks)
                .ok_or_else(|| AkitaError::InvalidSetup("witness E shape overflow".into()))?,
            depth_open,
            block_claim,
            role_subcolumn,
            digit,
            role_coefficient,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn t_coefficient_index(
        &self,
        source_ring_dim: usize,
        role_ring_dim: usize,
        num_claims: usize,
        n_a: usize,
        depth_outer: usize,
        claim: usize,
        global_block: usize,
        a_row: usize,
        role_subcolumn: usize,
        digit: usize,
        role_coefficient: usize,
    ) -> Result<usize, AkitaError> {
        let expected_len = num_claims
            .checked_mul(self.num_live_blocks)
            .and_then(|len| len.checked_mul(n_a))
            .and_then(|len| len.checked_mul(depth_outer))
            .and_then(|len| len.checked_mul(source_ring_dim))
            .ok_or_else(|| AkitaError::InvalidSetup("witness T shape overflow".into()))?;
        if self.t_range.len() != expected_len {
            return Err(AkitaError::InvalidSetup(
                "witness T shape disagrees with resolved range".into(),
            ));
        }
        let local_block = checked_owned_block(self, global_block)?;
        if claim >= num_claims || a_row >= n_a || digit >= depth_outer {
            return Err(AkitaError::InvalidInput(
                "witness T semantic index out of range".into(),
            ));
        }
        let block_claim = self
            .num_live_blocks
            .checked_mul(claim)
            .and_then(|base| base.checked_add(local_block))
            .ok_or_else(|| AkitaError::InvalidSetup("witness T index overflow".into()))?;
        let row_block_claim = n_a
            .checked_mul(block_claim)
            .and_then(|base| base.checked_add(a_row))
            .ok_or_else(|| AkitaError::InvalidSetup("witness T index overflow".into()))?;
        projected_coefficient_index(
            &self.t_range,
            "witness T",
            source_ring_dim,
            role_ring_dim,
            num_claims
                .checked_mul(self.num_live_blocks)
                .and_then(|count| count.checked_mul(n_a))
                .ok_or_else(|| AkitaError::InvalidSetup("witness T shape overflow".into()))?,
            depth_outer,
            row_block_claim,
            role_subcolumn,
            digit,
            role_coefficient,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn z_coefficient_index(
        &self,
        source_ring_dim: usize,
        num_positions_per_block: usize,
        depth_witness: usize,
        depth_fold: usize,
        position: usize,
        witness_digit: usize,
        fold_digit: usize,
        coefficient: usize,
    ) -> Result<usize, AkitaError> {
        let expected_len = checked::product([num_positions_per_block, depth_witness, depth_fold])
            .ok_or_else(|| AkitaError::InvalidSetup("witness Z shape overflow".into()))?
            .checked_mul(source_ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("witness Z shape overflow".into()))?;
        if self.z_range.len() != expected_len {
            return Err(AkitaError::InvalidSetup(
                "witness Z shape disagrees with resolved range".into(),
            ));
        }
        if position >= num_positions_per_block
            || witness_digit >= depth_witness
            || fold_digit >= depth_fold
            || coefficient >= source_ring_dim
        {
            return Err(AkitaError::InvalidInput(
                "witness Z semantic index out of range".into(),
            ));
        }
        let position_witness = depth_witness
            .checked_mul(position)
            .and_then(|base| base.checked_add(witness_digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness Z index overflow".into()))?;
        let ring_index = depth_fold
            .checked_mul(position_witness)
            .and_then(|base| base.checked_add(fold_digit))
            .ok_or_else(|| AkitaError::InvalidSetup("witness Z index overflow".into()))?;
        let local = ring_index
            .checked_mul(source_ring_dim)
            .and_then(|base| base.checked_add(coefficient))
            .ok_or_else(|| AkitaError::InvalidSetup("witness Z index overflow".into()))?;
        checked_range_index(&self.z_range, local, "witness Z")
    }
}

impl WitnessQuotientRowLayout {
    #[cfg(test)]
    pub(crate) fn new_for_test(geometry: RelationRowGeometry, range: Range<usize>) -> Self {
        Self { geometry, range }
    }

    #[must_use]
    pub const fn geometry(&self) -> RelationRowGeometry {
        self.geometry
    }

    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

impl CompressionWitnessSpan {
    #[cfg(test)]
    pub(crate) fn new_for_test(map: CompressionMapPlan, range: Range<usize>) -> Self {
        Self { map, range }
    }

    /// Checked canonical map represented by this span.
    #[must_use]
    pub fn map(&self) -> CompressionMapPlan {
        self.map
    }

    /// Complete padded coefficient range, with digit coordinates innermost.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    fn coefficient_index(&self, row: usize, coefficient: usize) -> Result<usize, AkitaError> {
        if row >= self.map.input_width() || coefficient >= self.map.ring_dimension() {
            return Err(AkitaError::InvalidInput(
                "compression witness semantic index out of range".into(),
            ));
        }
        let local = row
            .checked_mul(self.map.ring_dimension())
            .and_then(|base| base.checked_add(coefficient))
            .ok_or_else(|| AkitaError::InvalidSetup("compression witness index overflow".into()))?;
        checked_range_index(&self.range, local, "compression witness")
    }
}

impl CompressionWitnessLayerLayout {
    /// Zero-based compression map index.
    #[must_use]
    pub fn map_index(&self) -> usize {
        self.map_index
    }

    /// Relation-ordered F spans, tagged by opening group index.
    #[must_use]
    pub fn f_spans(&self) -> &[(usize, CompressionWitnessSpan)] {
        &self.f_spans
    }

    /// Shared H span for this layer.
    #[must_use]
    pub fn h_span(&self) -> &CompressionWitnessSpan {
        &self.h_span
    }

    /// Relation-ordered F quotient-row indices, tagged by opening group.
    /// Absent when this layer belongs to a reduced-evaluation witness.
    #[must_use]
    pub fn f_quotient_rows(&self) -> Option<&[(usize, usize)]> {
        self.f_quotient_rows.as_deref()
    }

    /// Shared H quotient-row index for this layer. Absent in reduced mode.
    #[must_use]
    pub fn h_quotient_row(&self) -> Option<usize> {
        self.h_quotient_row
    }
}

impl WitnessLayout {
    #[cfg(test)]
    pub(crate) fn new_for_test(
        units: Vec<WitnessUnitLayout>,
        r_rows: Vec<WitnessQuotientRowLayout>,
        quotient_depth: usize,
    ) -> Self {
        let r_start = r_rows.first().map_or_else(
            || units.last().map_or(0, |unit| unit.t_range.end),
            |row| row.range.start,
        );
        let r_end = r_rows.last().map_or(r_start, |row| row.range.end);
        Self {
            units,
            compression_layers: Vec::new(),
            compression_alignment_ranges: Vec::new(),
            relation_quotients: RelationQuotientLayout::QuotientLift {
                quotient_depth: std::num::NonZeroUsize::new(quotient_depth)
                    .expect("test quotient depth must be nonzero"),
                rows: r_rows,
            },
            tail_range: r_start..r_end,
        }
    }

    pub(crate) fn validate_internal_ranges(&self) -> Result<(), AkitaError> {
        let body_end = self.units.last().map_or(0, |unit| unit.t_range.end);
        if self.tail_range.start != body_end || self.tail_range.start > self.tail_range.end {
            return Err(AkitaError::InvalidSetup(
                "witness tail disagrees with its chunk-major body".into(),
            ));
        }
        if let RelationQuotientLayout::QuotientLift {
            quotient_depth,
            rows,
        } = &self.relation_quotients
        {
            let mut previous_row_end = None;
            for row in rows {
                let expected_len = quotient_depth
                    .get()
                    .checked_mul(row.geometry.physical_coefficient_width())
                    .ok_or_else(|| AkitaError::InvalidSetup("witness R width overflow".into()))?;
                if row.range.len() != expected_len
                    || previous_row_end.is_some_and(|end| row.range.start < end)
                {
                    return Err(AkitaError::InvalidSetup(
                        "witness quotient rows are not canonically ordered".into(),
                    ));
                }
                previous_row_end = Some(row.range.end);
            }
        }

        let mut ranges = Vec::new();
        for unit in &self.units {
            ranges.extend([unit.z_range(), unit.e_range(), unit.t_range()]);
        }
        ranges.extend(self.r_rows().iter().map(WitnessQuotientRowLayout::range));
        for layer in &self.compression_layers {
            ranges.extend(layer.f_spans.iter().map(|(_, span)| span.range()));
            ranges.push(layer.h_span.range());
        }
        ranges.extend(self.compression_alignment_ranges.iter().cloned());
        ranges.retain(|range| !range.is_empty());
        ranges.sort_unstable_by_key(|range| range.start);
        let mut cursor = 0usize;
        for range in ranges {
            if range.start != cursor || range.end > self.live_coeff_len() {
                return Err(AkitaError::InvalidSetup(
                    "witness ranges do not form one exact live partition".into(),
                ));
            }
            cursor = range.end;
        }
        if cursor != self.live_coeff_len() {
            return Err(AkitaError::InvalidSetup(
                "witness ranges do not cover the declared live prefix".into(),
            ));
        }
        Ok(())
    }

    /// Resolve exact chunk-major coefficient ranges from the canonical level
    /// parameters and opening claims layout.
    pub fn new(
        lp: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        relation_geometry: &RelationWitnessGeometry,
        num_chunks: usize,
        quotient_plan: RelationQuotientPlan,
    ) -> Result<Self, AkitaError> {
        let num_groups = opening_batch.num_groups();
        if num_groups == 0 || num_chunks == 0 {
            return Err(AkitaError::InvalidSetup(
                "witness layout requires non-empty groups and chunks".into(),
            ));
        }
        if num_chunks > MAX_WITNESS_CHUNKS {
            return Err(AkitaError::InvalidSetup(
                "witness chunk count exceeds verifier cap".into(),
            ));
        }
        let expected_relation_geometry = RelationWitnessGeometry::for_level(
            lp,
            opening_batch,
            relation_geometry.extension_degree(),
        )?;
        if &expected_relation_geometry != relation_geometry {
            return Err(AkitaError::InvalidSetup(
                "witness layout received relation geometry for different level parameters".into(),
            ));
        }
        let relation_group_order = opening_batch.root_group_order()?;
        let final_group_index = opening_batch.root_final_group_index()?;

        let mut units = Vec::with_capacity(
            num_groups
                .checked_mul(num_chunks)
                .ok_or_else(|| AkitaError::InvalidSetup("witness unit count overflow".into()))?,
        );
        let mut cursor = 0usize;
        let group_geometry = relation_group_order
            .iter()
            .map(|&group_index| {
                // `RelationWitnessGeometry::for_level` above already validated
                // the complete batch. Resolve per-group views directly.
                let (params, role_dims) = if group_index == final_group_index {
                    (lp.final_group(), lp.role_dims())
                } else {
                    let params = *lp
                        .preceding_group_params(group_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    (params, params.role_dims(lp.open().matrix.ring_dimension()))
                };
                role_dims.validate_role_projection()?;
                let group = opening_batch.group_layout(group_index)?;
                let opening_geometry = relation_geometry.group_opening_geometry(group_index)?;
                let num_claims = group.num_polynomials();
                let depth_witness = params.num_digits_inner();
                let depth_commit = params.num_digits_outer();
                let depth_open = params.num_digits_open();
                let depth_fold = params.num_digits_fold();
                if num_claims == 0
                    || params.num_live_blocks() == 0
                    || params.num_positions_per_block() == 0
                    || depth_open == 0
                    || depth_witness == 0
                    || depth_commit == 0
                    || depth_fold == 0
                    || params.a_rows_len() == 0
                {
                    return Err(AkitaError::InvalidSetup(
                        "witness group has malformed dimensions".into(),
                    ));
                }
                let chunk_block_ranges = dyadic_block_ranges(params.num_live_blocks(), num_chunks)?;
                Ok((
                    group_index,
                    params,
                    role_dims,
                    opening_geometry,
                    num_claims,
                    chunk_block_ranges,
                ))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;

        // Chunk is the outer physical key. Within each chunk, groups are laid
        // out in relation order, and each unit is `[Z | E | T]`.
        for chunk_index in 0..num_chunks {
            for &(
                group_index,
                params,
                role_dims,
                opening_geometry,
                num_claims,
                ref chunk_block_ranges,
            ) in &group_geometry
            {
                let global_block_range = chunk_block_ranges
                    .get(chunk_index)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness chunk is missing".into()))?
                    .clone();
                let global_block_start = global_block_range.start;
                let chunk_num_live_blocks = global_block_range.len();
                let (z_len, e_len, t_len) = witness_unit_lengths(
                    &params,
                    role_dims,
                    opening_geometry,
                    num_claims,
                    chunk_num_live_blocks,
                )?;
                let z_range = witness_range(cursor, z_len, "witness Z range overflow")?;
                let e_range = witness_range(z_range.end, e_len, "witness E range overflow")?;
                let t_range = witness_range(e_range.end, t_len, "witness T range overflow")?;
                cursor = t_range.end;
                units.push(WitnessUnitLayout {
                    group_index,
                    chunk_index,
                    global_block_start,
                    num_live_blocks: chunk_num_live_blocks,
                    z_range,
                    e_range,
                    e_geometry: opening_geometry,
                    t_range,
                });
            }
        }
        let tail_start = cursor;
        let successor_a_alignment = if num_groups > 1 {
            group_geometry
                .iter()
                .map(|(_, _, role_dims, _, _, _)| role_dims.d_a())
                .max()
                .ok_or_else(|| AkitaError::InvalidSetup("witness groups are empty".into()))?
        } else {
            relation_geometry.relation_coefficient_block_len()?
        };
        let tail = tail::materialize(
            lp,
            relation_geometry,
            num_groups,
            successor_a_alignment,
            tail_start,
            quotient_plan,
        )?;
        Ok(Self {
            units,
            compression_layers: tail.compression_layers,
            compression_alignment_ranges: tail.compression_alignment_ranges,
            relation_quotients: tail.relation_quotients,
            tail_range: tail_start..tail.end,
        })
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

    pub fn tail_range(&self) -> Range<usize> {
        self.tail_range.clone()
    }

    /// Layer-major compression witness geometry.
    #[must_use]
    pub fn compression_layers(&self) -> &[CompressionWitnessLayerLayout] {
        &self.compression_layers
    }

    /// Sorted support intervals for the negative-binary constraint.
    #[must_use]
    pub fn negative_binary_support_intervals(&self) -> Vec<Range<usize>> {
        self.compression_layers
            .iter()
            .filter_map(|layer| {
                let start = layer.f_spans.first().map(|(_, span)| span.range.start)?;
                Some(start..layer.h_span.range.end)
            })
            .collect()
    }

    /// Zero ranges used only to preserve the existing A/B/D coefficient block.
    #[must_use]
    pub fn compression_alignment_ranges(&self) -> &[Range<usize>] {
        &self.compression_alignment_ranges
    }

    /// F digit coefficient address for one group and map.
    pub fn f_compression_coefficient_index(
        &self,
        group_index: usize,
        map_index: usize,
        row: usize,
        coefficient: usize,
    ) -> Result<usize, AkitaError> {
        let layer = self
            .compression_layers
            .get(map_index)
            .ok_or_else(|| AkitaError::InvalidInput("compression map index is invalid".into()))?;
        let span = layer
            .f_spans
            .iter()
            .find_map(|(candidate, span)| (*candidate == group_index).then_some(span))
            .ok_or_else(|| AkitaError::InvalidInput("compression group index is invalid".into()))?;
        span.coefficient_index(row, coefficient)
    }

    /// Shared H digit coefficient address for one map.
    pub fn h_compression_coefficient_index(
        &self,
        map_index: usize,
        row: usize,
        coefficient: usize,
    ) -> Result<usize, AkitaError> {
        self.compression_layers
            .get(map_index)
            .ok_or_else(|| AkitaError::InvalidInput("compression map index is invalid".into()))?
            .h_span
            .coefficient_index(row, coefficient)
    }

    pub fn r_rows(&self) -> &[WitnessQuotientRowLayout] {
        match &self.relation_quotients {
            RelationQuotientLayout::QuotientLift { rows, .. } => rows,
            RelationQuotientLayout::ReducedEvaluation => &[],
        }
    }

    #[must_use]
    pub const fn relation_quotient_layout(&self) -> &RelationQuotientLayout {
        &self.relation_quotients
    }

    #[must_use]
    pub fn quotient_depth(&self) -> Option<usize> {
        match self.relation_quotients {
            RelationQuotientLayout::QuotientLift { quotient_depth, .. } => {
                Some(quotient_depth.get())
            }
            RelationQuotientLayout::ReducedEvaluation => None,
        }
    }

    pub fn live_coeff_len(&self) -> usize {
        self.tail_range.end
    }

    pub fn num_chunks_for_group(&self, group_index: usize) -> usize {
        self.units
            .iter()
            .filter(|unit| unit.group_index == group_index)
            .count()
    }

    pub fn group_num_live_blocks(&self, group_index: usize) -> Result<usize, AkitaError> {
        let mut total = 0usize;
        let mut found = false;
        for unit in self
            .units
            .iter()
            .filter(|unit| unit.group_index == group_index)
        {
            found = true;
            total = total
                .checked_add(unit.num_live_blocks)
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
        chunk_index: usize,
    ) -> Result<&WitnessUnitLayout, AkitaError> {
        self.units
            .iter()
            .find(|unit| unit.group_index == group_index && unit.chunk_index == chunk_index)
            .ok_or_else(|| AkitaError::InvalidSetup("witness unit is missing".into()))
    }

    pub fn units_for_group(
        &self,
        group_index: usize,
    ) -> Result<impl Iterator<Item = &WitnessUnitLayout> + Clone, AkitaError> {
        let single_group = self.units.first().is_some_and(|unit| unit.group_index == 0);
        if (single_group && group_index != 0)
            || (!single_group
                && !self
                    .units
                    .iter()
                    .any(|unit| unit.group_index == group_index))
        {
            return Err(AkitaError::InvalidSetup("witness group is missing".into()));
        }
        let empty = self.units[..0].iter();
        let (direct, filtered) = if single_group {
            (self.units.iter(), empty)
        } else {
            (empty, self.units.iter())
        };
        Ok(direct.chain(filtered.filter(move |unit| unit.group_index == group_index)))
    }

    pub fn unit_for_block(
        &self,
        group_index: usize,
        global_block: usize,
    ) -> Result<&WitnessUnitLayout, AkitaError> {
        self.units
            .iter()
            .filter(|unit| unit.group_index == group_index)
            .find(|unit| unit.global_block_range().contains(&global_block))
            .ok_or_else(|| AkitaError::InvalidInput("witness fold has no owning unit".into()))
    }

    pub fn r_coefficient_index(
        &self,
        relation_row: usize,
        quotient_digit: usize,
        coordinate_plane: usize,
        modulus_coefficient: usize,
    ) -> Result<usize, AkitaError> {
        let RelationQuotientLayout::QuotientLift {
            quotient_depth,
            rows,
        } = &self.relation_quotients
        else {
            return Err(AkitaError::InvalidInput(
                "reduced-evaluation witness has no quotient coordinates".into(),
            ));
        };
        let row = rows.get(relation_row).ok_or_else(|| {
            AkitaError::InvalidInput("witness R semantic index out of range".into())
        })?;
        if quotient_digit >= quotient_depth.get() {
            return Err(AkitaError::InvalidInput(
                "witness R semantic index out of range".into(),
            ));
        }
        let physical_coefficient = row
            .geometry
            .physical_coefficient_index(coordinate_plane, modulus_coefficient)?;
        let local = quotient_digit
            .checked_mul(row.geometry.physical_coefficient_width())
            .and_then(|base| base.checked_add(physical_coefficient))
            .ok_or_else(|| AkitaError::InvalidSetup("witness R index overflow".into()))?;
        checked_range_index(&row.range, local, "witness R")
    }
}

fn collect_quotient_rows(
    rows: Vec<Option<WitnessQuotientRowLayout>>,
    context: &str,
) -> Result<Vec<WitnessQuotientRowLayout>, AkitaError> {
    rows.into_iter()
        .enumerate()
        .map(|(row_index, row)| {
            row.ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "{context} quotient row {row_index} was not placed"
                ))
            })
        })
        .collect()
}

fn witness_range(start: usize, len: usize, context: &str) -> Result<Range<usize>, AkitaError> {
    checked::range(start, len).ok_or_else(|| AkitaError::InvalidSetup(context.into()))
}

fn align_witness_offset(
    value: usize,
    alignment: usize,
    context: &str,
) -> Result<usize, AkitaError> {
    checked::align_up(value, alignment).ok_or_else(|| {
        let message = if alignment.is_power_of_two() {
            context.into()
        } else {
            "witness alignment must be a nonzero power of two".into()
        };
        AkitaError::InvalidSetup(message)
    })
}

fn witness_unit_lengths(
    params: &crate::GroupOpenPhaseParams,
    role_dims: CommitmentRingDims,
    opening_geometry: RelationRowGeometry,
    num_claims: usize,
    chunk_num_live_blocks: usize,
) -> Result<(usize, usize, usize), AkitaError> {
    // Every response chunk retains the same full ambient Z buffer. The number
    // of assigned live blocks affects only the chunk-local E and T material.
    let z_len = checked::product([
        params.num_positions_per_block(),
        params.num_digits_inner(),
        params.num_digits_fold(),
    ])
    .ok_or_else(|| AkitaError::InvalidSetup("witness Z width overflow".into()))?
    .checked_mul(role_dims.d_a())
    .ok_or_else(|| AkitaError::InvalidSetup("witness Z width overflow".into()))?;
    let e_len = checked::product([num_claims, chunk_num_live_blocks, params.num_digits_open()])
        .ok_or_else(|| AkitaError::InvalidSetup("witness E width overflow".into()))?
        .checked_mul(opening_geometry.physical_coefficient_width())
        .ok_or_else(|| AkitaError::InvalidSetup("witness E width overflow".into()))?;
    let t_len = checked::product([
        num_claims,
        chunk_num_live_blocks,
        params.a_rows_len(),
        params.num_digits_outer(),
        role_dims.d_a(),
    ])
    .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".into()))?;
    Ok((z_len, e_len, t_len))
}

#[allow(clippy::too_many_arguments)]
fn projected_coefficient_index(
    range: &Range<usize>,
    label: &str,
    source_ring_dim: usize,
    role_ring_dim: usize,
    semantic_count: usize,
    digit_count: usize,
    semantic_index: usize,
    role_subcolumn: usize,
    digit: usize,
    role_coefficient: usize,
) -> Result<usize, AkitaError> {
    if source_ring_dim == 0 || role_ring_dim == 0 || !source_ring_dim.is_multiple_of(role_ring_dim)
    {
        return Err(AkitaError::InvalidSetup(format!(
            "{label} projected ring dimensions must satisfy role | source"
        )));
    }
    let expected_len = semantic_count
        .checked_mul(digit_count)
        .and_then(|len| len.checked_mul(source_ring_dim))
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{label} shape overflow")))?;
    if range.len() != expected_len {
        return Err(AkitaError::InvalidSetup(format!(
            "{label} shape disagrees with resolved range"
        )));
    }
    let live_subcolumns = source_ring_dim / role_ring_dim;
    if semantic_index >= semantic_count
        || role_subcolumn >= live_subcolumns
        || digit >= digit_count
        || role_coefficient >= role_ring_dim
    {
        return Err(AkitaError::InvalidInput(format!(
            "{label} projected coefficient index out of range"
        )));
    }
    let local_coefficient = semantic_index
        .checked_mul(live_subcolumns)
        .and_then(|index| index.checked_add(role_subcolumn))
        .and_then(|index| index.checked_mul(digit_count))
        .and_then(|index| index.checked_add(digit))
        .and_then(|index| index.checked_mul(role_ring_dim))
        .and_then(|index| index.checked_add(role_coefficient))
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{label} index overflow")))?;
    let index = range
        .start
        .checked_add(local_coefficient)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{label} index overflow")))?;
    if index >= range.end {
        return Err(AkitaError::InvalidSetup(format!(
            "{label} index exceeds resolved range"
        )));
    }
    Ok(index)
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

fn checked_owned_block(unit: &WitnessUnitLayout, global_block: usize) -> Result<usize, AkitaError> {
    let ownership_error = || {
        AkitaError::InvalidInput(format!(
            "witness fold block {global_block} is not owned by group {} chunk {} range {:?}",
            unit.group_index,
            unit.chunk_index,
            unit.global_block_range()
        ))
    };
    let local = global_block
        .checked_sub(unit.global_block_start)
        .ok_or_else(&ownership_error)?;
    if local >= unit.num_live_blocks {
        return Err(ownership_error());
    }
    Ok(local)
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
    /// Only invoked by [`crate::CommittedGroupParams`] descriptor binding when the level
    /// is chunked, so single-chunk levels stay byte-for-byte identical to the
    /// historical layout (the flag-off no-op invariant).
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        crate::descriptor_bytes::push_usize(bytes, self.num_chunks);
        crate::descriptor_bytes::push_usize(bytes, self.num_activated_levels);
    }
}

#[cfg(test)]
#[path = "witness_tests.rs"]
mod tests;
