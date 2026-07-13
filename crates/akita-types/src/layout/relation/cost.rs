//! Exact compression accounting derived from the canonical relation graph.

use std::collections::BTreeMap;

use akita_field::AkitaError;

use super::{
    CompressionSourceId, RelationLayout, RelationRowId, RelationRowInputs, RelationRowRhs,
    RelationSegmentId,
};

#[cfg(test)]
mod tests;

/// Compression accounting for one native ring dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionDimensionCost {
    native_ring_dim: usize,
    map_count: usize,
    native_rows: usize,
    logical_setup_coeffs: usize,
    max_setup_prefix_coeffs: usize,
}

/// Structural geometry of one compression map, keyed by its semantic identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionMapStructuralCost {
    source: CompressionSourceId,
    map: usize,
    native_ring_dim: usize,
    rows: usize,
    input_coeffs: usize,
    output_coeffs: usize,
    logical_setup_coeffs: usize,
    terminal_payload_coeffs: Option<usize>,
    negative_binary: bool,
}

impl CompressionMapStructuralCost {
    pub fn source(&self) -> CompressionSourceId {
        self.source
    }
    pub fn map(&self) -> usize {
        self.map
    }
    pub fn native_ring_dim(&self) -> usize {
        self.native_ring_dim
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn input_coeffs(&self) -> usize {
        self.input_coeffs
    }
    pub fn output_coeffs(&self) -> usize {
        self.output_coeffs
    }
    pub fn logical_setup_coeffs(&self) -> usize {
        self.logical_setup_coeffs
    }
    pub fn terminal_payload_coeffs(&self) -> Option<usize> {
        self.terminal_payload_coeffs
    }
    pub fn is_negative_binary(&self) -> bool {
        self.negative_binary
    }
}

impl CompressionDimensionCost {
    pub fn native_ring_dim(&self) -> usize {
        self.native_ring_dim
    }

    pub fn map_count(&self) -> usize {
        self.map_count
    }

    pub fn native_rows(&self) -> usize {
        self.native_rows
    }

    pub fn logical_setup_coeffs(&self) -> usize {
        self.logical_setup_coeffs
    }

    pub fn max_setup_prefix_coeffs(&self) -> usize {
        self.max_setup_prefix_coeffs
    }
}

/// Exact compression-only structural witness, setup, and sparse-scan units.
///
/// Counts are field coefficients. `relation_*_scan_coeffs` count live cells
/// visited at round zero after the row challenge has collapsed native rows;
/// they deliberately do not pretend to be operation counts, cycle estimates,
/// total prover overhead, or serialized proof bytes. The relation sparse state
/// visits their sum, while the independent binary state visits
/// `negative_binary_support_coeffs` cells. Both states can only shrink after
/// binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionRelationStructuralCost {
    map_count: usize,
    native_rows: usize,
    terminal_payload_coeffs: usize,
    witness_input_coeffs: usize,
    witness_quotient_coeffs: usize,
    witness_coeffs: usize,
    negative_binary_support_runs: usize,
    negative_binary_support_coeffs: usize,
    relation_input_scan_coeffs: usize,
    relation_gadget_scan_coeffs: usize,
    relation_quotient_scan_coeffs: usize,
    relation_scan_coeffs: usize,
    logical_setup_coeffs: usize,
    max_setup_prefix_coeffs: usize,
    coalesced_setup_cache_coeffs: usize,
    maps: Vec<CompressionMapStructuralCost>,
    dimensions: Vec<CompressionDimensionCost>,
}

impl CompressionRelationStructuralCost {
    pub(super) fn derive(layout: &RelationLayout) -> Result<Self, AkitaError> {
        let mut witness_input_coeffs = 0usize;
        let mut witness_quotient_coeffs = 0usize;
        for segment in layout.segments() {
            match segment.id() {
                RelationSegmentId::CompressionInput { .. } => {
                    checked_add(
                        &mut witness_input_coeffs,
                        segment.span().len(),
                        "input witness",
                    )?;
                }
                RelationSegmentId::CompressionQuotient { .. } => checked_add(
                    &mut witness_quotient_coeffs,
                    segment.span().len(),
                    "quotient witness",
                )?,
                _ => {}
            }
        }

        let negative_binary_support_runs = layout.negative_binary_support().len();
        let mut negative_binary_support_coeffs = 0usize;
        for span in layout.negative_binary_support() {
            checked_add(
                &mut negative_binary_support_coeffs,
                span.len(),
                "negative-binary support",
            )?;
        }

        let mut map_count = 0usize;
        let mut native_rows = 0usize;
        let mut terminal_payload_coeffs = 0usize;
        let mut relation_input_scan_coeffs = 0usize;
        let mut relation_gadget_scan_coeffs = 0usize;
        let mut relation_quotient_scan_coeffs = 0usize;
        let mut logical_setup_coeffs = 0usize;
        let mut max_setup_prefix_coeffs = 0usize;
        let mut dimensions = BTreeMap::<usize, CompressionDimensionCost>::new();
        let mut maps = Vec::new();

        for family in layout.row_plan().families() {
            match family.inputs() {
                RelationRowInputs::B {
                    compression_input: Some(input),
                    ..
                }
                | RelationRowInputs::D {
                    compression_input: Some(input),
                    ..
                } => checked_add(
                    &mut relation_gadget_scan_coeffs,
                    layout.segment(input.segment())?.span().len(),
                    "base-row compression scan",
                )?,
                RelationRowInputs::Compression { input, successor } => {
                    let provider = layout.family_provider(family.id())?;
                    let d = provider.native_ring_dim();
                    let rows = provider.rows().len();
                    checked_add(&mut map_count, 1, "map count")?;
                    checked_add(&mut native_rows, rows, "native rows")?;
                    checked_add(
                        &mut relation_input_scan_coeffs,
                        layout.segment(*input)?.span().len(),
                        "relation input scan",
                    )?;
                    if let Some(successor) = successor {
                        checked_add(
                            &mut relation_gadget_scan_coeffs,
                            layout.segment(successor.segment())?.span().len(),
                            "relation successor scan",
                        )?;
                    }
                    checked_add(
                        &mut relation_quotient_scan_coeffs,
                        provider.quotient_span()?.len(),
                        "relation quotient scan",
                    )?;
                    if let RelationRowRhs::TerminalPayload { coeffs } = provider.rhs() {
                        checked_add(&mut terminal_payload_coeffs, coeffs, "terminal payload")?;
                    }
                    let view = provider.compression_setup_view()?;
                    let (source, map) = match family.id() {
                        RelationRowId::Compression { source, map } => (source, map),
                        _ => {
                            return Err(AkitaError::InvalidSetup(
                                "compression inputs disagree with row identity".into(),
                            ));
                        }
                    };
                    let output_coeffs = rows.checked_mul(d).ok_or_else(|| {
                        AkitaError::InvalidSetup("compression map output overflow".into())
                    })?;
                    let terminal_payload = match provider.rhs() {
                        RelationRowRhs::TerminalPayload { coeffs } => Some(coeffs),
                        _ => None,
                    };
                    let input_span = layout.segment(*input)?.span();
                    let input_end = input_span
                        .start()
                        .checked_add(input_span.len())
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("compression input span overflow".into())
                        })?;
                    let negative_binary = layout.negative_binary_support().iter().any(|support| {
                        support.start() <= input_span.start()
                            && support
                                .start()
                                .checked_add(support.len())
                                .is_some_and(|end| end >= input_end)
                    });
                    maps.push(CompressionMapStructuralCost {
                        source,
                        map,
                        native_ring_dim: d,
                        rows,
                        input_coeffs: input_span.len(),
                        output_coeffs,
                        logical_setup_coeffs: view.flat_footprint(),
                        terminal_payload_coeffs: terminal_payload,
                        negative_binary,
                    });
                    checked_add(
                        &mut logical_setup_coeffs,
                        view.flat_footprint(),
                        "logical setup",
                    )?;
                    max_setup_prefix_coeffs = max_setup_prefix_coeffs.max(view.flat_footprint());
                    let entry = dimensions.entry(d).or_insert(CompressionDimensionCost {
                        native_ring_dim: d,
                        map_count: 0,
                        native_rows: 0,
                        logical_setup_coeffs: 0,
                        max_setup_prefix_coeffs: 0,
                    });
                    checked_add(&mut entry.map_count, 1, "dimension map count")?;
                    checked_add(&mut entry.native_rows, rows, "dimension native rows")?;
                    checked_add(
                        &mut entry.logical_setup_coeffs,
                        view.flat_footprint(),
                        "dimension logical setup",
                    )?;
                    entry.max_setup_prefix_coeffs =
                        entry.max_setup_prefix_coeffs.max(view.flat_footprint());
                }
                _ => {}
            }
        }

        let dimensions = dimensions.into_values().collect::<Vec<_>>();
        let coalesced_setup_cache_coeffs = dimensions.iter().try_fold(0usize, |total, cost| {
            total
                .checked_add(cost.max_setup_prefix_coeffs)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("coalesced compression setup cache overflow".into())
                })
        })?;
        let compression_witness_coeffs = witness_input_coeffs
            .checked_add(witness_quotient_coeffs)
            .ok_or_else(|| AkitaError::InvalidSetup("compression witness cost overflow".into()))?;
        if compression_witness_coeffs != layout.compression_witness_coeffs {
            return Err(AkitaError::InvalidSetup(
                "compression witness segments disagree with the compiled suffix".into(),
            ));
        }
        let relation_scan_coeffs = relation_input_scan_coeffs
            .checked_add(relation_gadget_scan_coeffs)
            .and_then(|total| total.checked_add(relation_quotient_scan_coeffs))
            .ok_or_else(|| AkitaError::InvalidSetup("compression relation scan overflow".into()))?;

        Ok(Self {
            map_count,
            native_rows,
            terminal_payload_coeffs,
            witness_input_coeffs,
            witness_quotient_coeffs,
            witness_coeffs: compression_witness_coeffs,
            negative_binary_support_runs,
            negative_binary_support_coeffs,
            relation_input_scan_coeffs,
            relation_gadget_scan_coeffs,
            relation_quotient_scan_coeffs,
            relation_scan_coeffs,
            logical_setup_coeffs,
            max_setup_prefix_coeffs,
            coalesced_setup_cache_coeffs,
            maps,
            dimensions,
        })
    }

    pub fn map_count(&self) -> usize {
        self.map_count
    }
    pub fn native_rows(&self) -> usize {
        self.native_rows
    }
    pub fn terminal_payload_coeffs(&self) -> usize {
        self.terminal_payload_coeffs
    }
    pub fn witness_input_coeffs(&self) -> usize {
        self.witness_input_coeffs
    }
    pub fn witness_quotient_coeffs(&self) -> usize {
        self.witness_quotient_coeffs
    }
    pub fn witness_coeffs(&self) -> usize {
        self.witness_coeffs
    }
    pub fn negative_binary_support_runs(&self) -> usize {
        self.negative_binary_support_runs
    }
    pub fn negative_binary_support_coeffs(&self) -> usize {
        self.negative_binary_support_coeffs
    }
    pub fn relation_input_scan_coeffs(&self) -> usize {
        self.relation_input_scan_coeffs
    }
    /// Coefficients visited by gadget terms, including B/D augmentation and
    /// nonterminal F/H successors.
    pub fn relation_gadget_scan_coeffs(&self) -> usize {
        self.relation_gadget_scan_coeffs
    }
    pub fn relation_quotient_scan_coeffs(&self) -> usize {
        self.relation_quotient_scan_coeffs
    }
    pub fn relation_scan_coeffs(&self) -> usize {
        self.relation_scan_coeffs
    }
    pub fn logical_setup_coeffs(&self) -> usize {
        self.logical_setup_coeffs
    }
    pub fn max_setup_prefix_coeffs(&self) -> usize {
        self.max_setup_prefix_coeffs
    }
    pub fn coalesced_setup_cache_coeffs(&self) -> usize {
        self.coalesced_setup_cache_coeffs
    }
    pub fn dimensions(&self) -> &[CompressionDimensionCost] {
        &self.dimensions
    }
    pub fn maps(&self) -> &[CompressionMapStructuralCost] {
        &self.maps
    }
}

fn checked_add(total: &mut usize, add: usize, label: &str) -> Result<(), AkitaError> {
    *total = total
        .checked_add(add)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("compression {label} overflow")))?;
    Ok(())
}
