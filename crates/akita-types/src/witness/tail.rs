use std::num::NonZeroUsize;

use akita_error::AkitaError;

use super::{
    align_witness_offset, collect_quotient_rows, witness_range, CompressionWitnessLayerLayout,
    CompressionWitnessSpan, RelationQuotientLayout, WitnessQuotientRowLayout,
};
use crate::{
    CommittedGroupParams, CompressionMapPlan, RelationRowFamily, RelationRowGeometry,
    RelationWitnessGeometry, RingRelationMode, COMPRESSION_MAP_COUNT,
};

/// Quotient witness state required to construct one canonical witness layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationQuotientPlan {
    /// Lifted relations carry quotient rows at a nonzero decomposition depth.
    QuotientLift { quotient_depth: NonZeroUsize },
    /// Reduced relations carry no quotient metadata or rows.
    ReducedEvaluation,
}

impl RelationQuotientPlan {
    /// Construct checked quotient-lift state.
    pub fn quotient_lift(quotient_depth: usize) -> Result<Self, AkitaError> {
        let quotient_depth = NonZeroUsize::new(quotient_depth).ok_or_else(|| {
            AkitaError::InvalidSetup("quotient-lift depth must be nonzero".into())
        })?;
        Ok(Self::QuotientLift { quotient_depth })
    }

    /// Construct the canonical plan for a committed parameter set and field width.
    pub fn for_field_bits(
        params: &CommittedGroupParams,
        field_bits: u32,
    ) -> Result<Self, AkitaError> {
        match params.ring_relation_mode {
            RingRelationMode::QuotientLift => {
                Self::quotient_lift(crate::sis::compute_num_digits_field_width(
                    field_bits,
                    params.open().digits.log_basis,
                ))
            }
            RingRelationMode::ReducedEvaluation => Ok(Self::ReducedEvaluation),
        }
    }

    fn validate_mode(self, mode: RingRelationMode) -> Result<(), AkitaError> {
        if matches!(
            (self, mode),
            (Self::QuotientLift { .. }, RingRelationMode::QuotientLift)
                | (Self::ReducedEvaluation, RingRelationMode::ReducedEvaluation)
        ) {
            Ok(())
        } else {
            Err(AkitaError::InvalidSetup(
                "witness quotient plan disagrees with the authenticated relation mode".into(),
            ))
        }
    }

    fn quotient_depth(self) -> Option<NonZeroUsize> {
        match self {
            Self::QuotientLift { quotient_depth } => Some(quotient_depth),
            Self::ReducedEvaluation => None,
        }
    }
}

pub(super) struct MaterializedWitnessTail {
    pub(super) compression_layers: Vec<CompressionWitnessLayerLayout>,
    pub(super) compression_alignment_ranges: Vec<std::ops::Range<usize>>,
    pub(super) relation_quotients: RelationQuotientLayout,
    pub(super) end: usize,
}

trait TailSink: Sized {
    type Output;

    fn prepare_quotient_rows(&mut self, row_count: usize);
    fn align(&mut self, alignment: usize, overflow: &'static str) -> Result<(), AkitaError>;
    fn place_f_span(
        &mut self,
        group_index: usize,
        map: CompressionMapPlan,
    ) -> Result<(), AkitaError>;
    fn place_h_span(&mut self, map: CompressionMapPlan) -> Result<(), AkitaError>;
    fn place_quotient_row(
        &mut self,
        row_index: usize,
        geometry: RelationRowGeometry,
        quotient_depth: NonZeroUsize,
        width_overflow: &'static str,
        range_overflow: &'static str,
    ) -> Result<(), AkitaError>;
    fn record_f_quotient_row(&mut self, group_index: usize, row_index: usize);
    fn finish_compression_layer(
        &mut self,
        map_index: usize,
        h_quotient_row: Option<usize>,
        plan: RelationQuotientPlan,
    ) -> Result<(), AkitaError>;
    fn finish(self, start: usize, plan: RelationQuotientPlan) -> Result<Self::Output, AkitaError>;
}

struct MaterializingTailSink {
    cursor: usize,
    compression_layers: Vec<CompressionWitnessLayerLayout>,
    compression_alignment_ranges: Vec<std::ops::Range<usize>>,
    quotient_rows: Vec<Option<WitnessQuotientRowLayout>>,
    current_f_spans: Vec<(usize, CompressionWitnessSpan)>,
    current_f_quotient_rows: Vec<(usize, usize)>,
    current_h_span: Option<CompressionWitnessSpan>,
}

impl MaterializingTailSink {
    fn new(start: usize) -> Self {
        Self {
            cursor: start,
            compression_layers: Vec::new(),
            compression_alignment_ranges: Vec::new(),
            quotient_rows: Vec::new(),
            current_f_spans: Vec::new(),
            current_f_quotient_rows: Vec::new(),
            current_h_span: None,
        }
    }
}

impl TailSink for MaterializingTailSink {
    type Output = MaterializedWitnessTail;

    fn prepare_quotient_rows(&mut self, row_count: usize) {
        self.quotient_rows = vec![None; row_count];
    }

    fn align(&mut self, alignment: usize, overflow: &'static str) -> Result<(), AkitaError> {
        let aligned = align_witness_offset(self.cursor, alignment, overflow)?;
        if aligned != self.cursor {
            self.compression_alignment_ranges.push(self.cursor..aligned);
            self.cursor = aligned;
        }
        Ok(())
    }

    fn place_f_span(
        &mut self,
        group_index: usize,
        map: CompressionMapPlan,
    ) -> Result<(), AkitaError> {
        let range = witness_range(
            self.cursor,
            map.padded_digit_count(),
            "witness F range overflow",
        )?;
        self.cursor = range.end;
        self.current_f_spans
            .push((group_index, CompressionWitnessSpan { map, range }));
        Ok(())
    }

    fn place_h_span(&mut self, map: CompressionMapPlan) -> Result<(), AkitaError> {
        let range = witness_range(
            self.cursor,
            map.padded_digit_count(),
            "witness H range overflow",
        )?;
        self.cursor = range.end;
        self.current_h_span = Some(CompressionWitnessSpan { map, range });
        Ok(())
    }

    fn place_quotient_row(
        &mut self,
        row_index: usize,
        geometry: RelationRowGeometry,
        quotient_depth: NonZeroUsize,
        width_overflow: &'static str,
        range_overflow: &'static str,
    ) -> Result<(), AkitaError> {
        let len = quotient_depth
            .get()
            .checked_mul(geometry.physical_coefficient_width())
            .ok_or_else(|| AkitaError::InvalidSetup(width_overflow.into()))?;
        let range = witness_range(self.cursor, len, range_overflow)?;
        self.cursor = range.end;
        let slot = self.quotient_rows.get_mut(row_index).ok_or_else(|| {
            AkitaError::InvalidSetup("witness quotient row index is invalid".into())
        })?;
        *slot = Some(WitnessQuotientRowLayout { geometry, range });
        Ok(())
    }

    fn record_f_quotient_row(&mut self, group_index: usize, row_index: usize) {
        self.current_f_quotient_rows.push((group_index, row_index));
    }

    fn finish_compression_layer(
        &mut self,
        map_index: usize,
        h_quotient_row: Option<usize>,
        plan: RelationQuotientPlan,
    ) -> Result<(), AkitaError> {
        let h_span = self.current_h_span.take().ok_or_else(|| {
            AkitaError::InvalidSetup("compression H witness span is missing".into())
        })?;
        let f_spans = std::mem::take(&mut self.current_f_spans);
        let canonical_f_quotient_rows = std::mem::take(&mut self.current_f_quotient_rows);
        if plan.quotient_depth().is_some() && canonical_f_quotient_rows.len() != f_spans.len() {
            return Err(AkitaError::InvalidSetup(
                "compression F quotient ownership disagrees with witness spans".into(),
            ));
        }
        let lifted = plan.quotient_depth().is_some();
        self.compression_layers.push(CompressionWitnessLayerLayout {
            map_index,
            f_spans,
            h_span,
            f_quotient_rows: lifted.then_some(canonical_f_quotient_rows),
            h_quotient_row,
        });
        Ok(())
    }

    fn finish(self, _start: usize, plan: RelationQuotientPlan) -> Result<Self::Output, AkitaError> {
        let relation_quotients = match plan {
            RelationQuotientPlan::QuotientLift { quotient_depth } => {
                RelationQuotientLayout::QuotientLift {
                    quotient_depth,
                    rows: collect_quotient_rows(self.quotient_rows, "witness")?,
                }
            }
            RelationQuotientPlan::ReducedEvaluation => RelationQuotientLayout::ReducedEvaluation,
        };
        Ok(MaterializedWitnessTail {
            compression_layers: self.compression_layers,
            compression_alignment_ranges: self.compression_alignment_ranges,
            relation_quotients,
            end: self.cursor,
        })
    }
}

struct MeasuringTailSink {
    cursor: usize,
}

impl TailSink for MeasuringTailSink {
    type Output = usize;

    fn prepare_quotient_rows(&mut self, _row_count: usize) {}

    fn align(&mut self, alignment: usize, overflow: &'static str) -> Result<(), AkitaError> {
        self.cursor = align_witness_offset(self.cursor, alignment, overflow)?;
        Ok(())
    }

    fn place_f_span(
        &mut self,
        _group_index: usize,
        map: CompressionMapPlan,
    ) -> Result<(), AkitaError> {
        self.cursor = witness_range(
            self.cursor,
            map.padded_digit_count(),
            "witness F range overflow",
        )?
        .end;
        Ok(())
    }

    fn place_h_span(&mut self, map: CompressionMapPlan) -> Result<(), AkitaError> {
        self.cursor = witness_range(
            self.cursor,
            map.padded_digit_count(),
            "witness H range overflow",
        )?
        .end;
        Ok(())
    }

    fn place_quotient_row(
        &mut self,
        _row_index: usize,
        geometry: RelationRowGeometry,
        quotient_depth: NonZeroUsize,
        width_overflow: &'static str,
        range_overflow: &'static str,
    ) -> Result<(), AkitaError> {
        let len = quotient_depth
            .get()
            .checked_mul(geometry.physical_coefficient_width())
            .ok_or_else(|| AkitaError::InvalidSetup(width_overflow.into()))?;
        self.cursor = witness_range(self.cursor, len, range_overflow)?.end;
        Ok(())
    }

    fn record_f_quotient_row(&mut self, _group_index: usize, _row_index: usize) {}

    fn finish_compression_layer(
        &mut self,
        _map_index: usize,
        _h_quotient_row: Option<usize>,
        _plan: RelationQuotientPlan,
    ) -> Result<(), AkitaError> {
        Ok(())
    }

    fn finish(
        self,
        _start: usize,
        _plan: RelationQuotientPlan,
    ) -> Result<Self::Output, AkitaError> {
        Ok(self.cursor)
    }
}

pub(super) fn materialize(
    params: &CommittedGroupParams,
    relation_geometry: &RelationWitnessGeometry,
    num_groups: usize,
    successor_a_alignment: usize,
    start: usize,
    plan: RelationQuotientPlan,
) -> Result<MaterializedWitnessTail, AkitaError> {
    resolve(
        params,
        relation_geometry,
        num_groups,
        successor_a_alignment,
        start,
        plan,
        MaterializingTailSink::new(start),
    )
}

pub(super) fn measure(
    params: &CommittedGroupParams,
    relation_geometry: &RelationWitnessGeometry,
    num_groups: usize,
    successor_a_alignment: usize,
    start: usize,
    plan: RelationQuotientPlan,
) -> Result<usize, AkitaError> {
    resolve(
        params,
        relation_geometry,
        num_groups,
        successor_a_alignment,
        start,
        plan,
        MeasuringTailSink { cursor: start },
    )
}

#[allow(clippy::too_many_arguments)]
fn resolve<S: TailSink>(
    params: &CommittedGroupParams,
    relation_geometry: &RelationWitnessGeometry,
    num_groups: usize,
    successor_a_alignment: usize,
    start: usize,
    plan: RelationQuotientPlan,
    mut sink: S,
) -> Result<S::Output, AkitaError> {
    plan.validate_mode(params.ring_relation_mode)?;
    let relation_layout = relation_geometry.rhs_layout();
    let row_families = if plan.quotient_depth().is_some() {
        relation_layout.row_families()?
    } else {
        Vec::new()
    };
    let first_compression_row = row_families
        .iter()
        .position(|row| {
            matches!(
                row,
                RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
            )
        })
        .unwrap_or(row_families.len());
    if let Some(quotient_depth) = plan.quotient_depth() {
        sink.prepare_quotient_rows(row_families.len());
        for (row_index, row) in row_families[..first_compression_row].iter().enumerate() {
            sink.place_quotient_row(
                row_index,
                row.geometry(),
                quotient_depth,
                "witness R width overflow",
                "witness R range overflow",
            )?;
        }
    }
    if !params.payload_mode.is_compressed() {
        return sink.finish(start, plan);
    }

    let relation_coefficient_block = relation_geometry.relation_coefficient_block_len()?;
    sink.align(
        relation_coefficient_block,
        "compression witness alignment overflow",
    )?;
    for map_index in 0..COMPRESSION_MAP_COUNT {
        let mut layer_alignment =
            relation_layout.opening_compression_plan()?.maps()[map_index].ring_dimension();
        for relation_group_index in 0..num_groups {
            let (_, group_plan) = relation_layout.group_compression_plan(relation_group_index)?;
            layer_alignment = layer_alignment.max(group_plan.maps()[map_index].ring_dimension());
        }
        sink.align(layer_alignment, "compression layer alignment overflow")?;

        for relation_group_index in 0..num_groups {
            let (group_index, group_plan) =
                relation_layout.group_compression_plan(relation_group_index)?;
            sink.place_f_span(group_index, group_plan.maps()[map_index])?;
        }
        let h_map = relation_layout.opening_compression_plan()?.maps()[map_index];
        sink.place_h_span(h_map)?;

        let h_quotient_row = if let Some(quotient_depth) = plan.quotient_depth() {
            for relation_group_index in 0..num_groups {
                let row_index = first_compression_row
                    .checked_add(map_index * (num_groups + 1) + relation_group_index)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("compression quotient index overflow".into())
                    })?;
                let row = *row_families.get(row_index).ok_or_else(|| {
                    AkitaError::InvalidSetup("compression F quotient row is missing".into())
                })?;
                let (group_index, geometry) = match row {
                    RelationRowFamily::CompressionF {
                        group_index,
                        map_index: row_map_index,
                        geometry,
                    } if row_map_index == map_index => (group_index, geometry),
                    _ => {
                        return Err(AkitaError::InvalidSetup(
                            "compression F quotient order disagrees with relation rows".into(),
                        ))
                    }
                };
                let (planned_group_index, _) =
                    relation_layout.group_compression_plan(relation_group_index)?;
                if group_index != planned_group_index {
                    return Err(AkitaError::InvalidSetup(
                        "compression F quotient group disagrees with its compression plan".into(),
                    ));
                }
                sink.place_quotient_row(
                    row_index,
                    geometry,
                    quotient_depth,
                    "compression quotient width overflow",
                    "compression quotient range overflow",
                )?;
                sink.record_f_quotient_row(group_index, row_index);
            }
            let h_quotient_row = first_compression_row
                .checked_add(map_index * (num_groups + 1) + num_groups)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("compression quotient index overflow".into())
                })?;
            let h_row = *row_families.get(h_quotient_row).ok_or_else(|| {
                AkitaError::InvalidSetup("compression H quotient row is missing".into())
            })?;
            let h_geometry = match h_row {
                RelationRowFamily::CompressionH {
                    map_index: row_map_index,
                    geometry,
                } if row_map_index == map_index => geometry,
                _ => {
                    return Err(AkitaError::InvalidSetup(
                        "compression H quotient order disagrees with relation rows".into(),
                    ))
                }
            };
            sink.place_quotient_row(
                h_quotient_row,
                h_geometry,
                quotient_depth,
                "compression quotient width overflow",
                "compression quotient range overflow",
            )?;
            Some(h_quotient_row)
        } else {
            None
        };
        sink.finish_compression_layer(map_index, h_quotient_row, plan)?;
    }
    sink.align(
        successor_a_alignment,
        "compression witness suffix alignment overflow",
    )?;
    sink.finish(start, plan)
}
