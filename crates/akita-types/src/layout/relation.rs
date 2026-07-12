//! Canonical logical geometry for the ring relation.
//!
//! [`RelationLayout`] is compiled from authenticated level, opening, and
//! compression inputs. It is derived protocol state, not itself an
//! authenticated or serialized object. It is the single checked description of both axes of the
//! relation. Its coefficient axis is a contiguous logical arena of
//! [`RelationSegment`]s; its row axis is a [`RelationRowPlan`] of typed row
//! families followed by one field-level trace row and power-of-two padding.
//!
//! ```text
//! coefficient axis (logical; lengths below are illustrative)
//! ┌──── current ────┬── precommitted[0] ─┬─ base quotients ─┬── Xi inputs ──┬─ Xi quotients ─┐
//! │ Z │ E │ T       │ Z │ E │ T          │ q[family] ...    │ Xi[0] ...      │ qXi[0] ...     │
//! └─────────────────┴─────────────────────┴──────────────────┴────────────────┴────────────────┘
//!
//! row axis
//! ┌ consistency ┬ A(cur) ┬ B(cur) ┬ A(pre0) ┬ B(pre0) ┬ D? ┬ compression* ┐ trace │ pad
//! └─────────────┴────────┴────────┴─────────┴─────────┴────┴──────────────┘       │
//!       matrix row families, each with its own native ring dimension              │
//! ```
//!
//! Compression inputs and then compression quotients use layer-major order;
//! within a layer the source order is current, precommitted by increasing
//! index, then opening, with absent sources omitted. Compression rows use the
//! same order.
//!
//! Paper terminology differs: [`RelationRowId::Consistency`] is the paper's
//! fold-evaluation `Z/E` row, while [`RelationRowId::A`] is its
//! fold-consistency/`A` family associated with `Z/T`. The paper's compact
//! `[z | e | t | r | u1 | v1 | ...]` witness is algebraic notation, not an
//! assertion about this module's coefficient order.
//!
//! These are *logical field-coefficient addresses*, not physical witness
//! storage. [`WitnessLayout`] is a projection of the base `Z/E/T` segments and
//! base quotient tails into ring-element chunks. Compression inputs and
//! compression quotients deliberately remain outside that projection.

use akita_field::AkitaError;
use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;
use std::ops::Range;

use crate::witness::{WitnessChunkLayout, WitnessChunkLengths, WitnessLayout};
use crate::{sis::compute_num_digits_full_field, LevelParams, OpeningClaimsLayout};

use super::compression::semantics::{
    self as compression_semantics, CompressionRowRhs, SegmentId as CompressionSegmentId,
};
pub use super::compression::CompressionSourceId;
use super::RelationMatrixRowLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A half-open run in the logical field-coefficient arena.
pub struct CoeffSpan {
    start: usize,
    len: usize,
}

impl CoeffSpan {
    fn end(self) -> Result<usize, AkitaError> {
        self.start
            .checked_add(self.len)
            .ok_or_else(|| AkitaError::InvalidSetup("logical coefficient span overflow".into()))
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn range(&self) -> Range<usize> {
        self.start..self.start.saturating_add(self.len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Stable identity of a polynomial group in the relation.
///
/// The canonical order is always [`Current`](Self::Current) first, followed by
/// precommitted groups in increasing index order. This is intentionally not the
/// source ordering used by every paper-level presentation.
pub enum RelationGroupId {
    Current,
    Precommitted { index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Stable identity of a logical coefficient segment.
///
/// An identity is independent of its current offset. Consumers should resolve
/// it through [`RelationLayout::segment`] instead of reproducing offsets.
pub enum RelationSegmentId {
    Z {
        group: RelationGroupId,
    },
    E {
        group: RelationGroupId,
    },
    T {
        group: RelationGroupId,
    },
    BaseQuotient {
        row: RelationRowId,
    },
    CompressionInput {
        source: CompressionSourceId,
        map: usize,
    },
    CompressionQuotient {
        source: CompressionSourceId,
        map: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// One named allocation in the logical coefficient arena.
pub struct RelationSegment {
    id: RelationSegmentId,
    span: CoeffSpan,
}

impl RelationSegment {
    pub fn id(&self) -> RelationSegmentId {
        self.id
    }

    pub fn span(&self) -> CoeffSpan {
        self.span
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Stable identity of a relation-matrix row family.
pub enum RelationRowId {
    Consistency,
    A {
        group: RelationGroupId,
    },
    B {
        group: RelationGroupId,
    },
    D,
    Compression {
        source: CompressionSourceId,
        map: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A half-open run of rows in the unpadded relation matrix.
pub struct RowSpan {
    start: usize,
    len: usize,
}

impl RowSpan {
    pub fn start(&self) -> usize {
        self.start
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn range(&self) -> Range<usize> {
        self.start..self.start.saturating_add(self.len)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// The right-hand-side meaning derived for a row family.
///
/// Compression moves a commitment/opening payload to the last compression
/// family in its chain. Every augmented base `B`/`D` family and every
/// intermediate compression family has [`Zero`](Self::Zero) RHS.
pub enum RelationRowRhs {
    Zero,
    Commitment { group: RelationGroupId },
    Opening,
    TerminalPayload { coeffs: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A coefficient segment interpreted through a particular gadget basis.
pub struct GadgetInput {
    segment: RelationSegmentId,
    log_basis: u32,
}

impl GadgetInput {
    pub fn segment(&self) -> RelationSegmentId {
        self.segment
    }

    pub fn log_basis(&self) -> u32 {
        self.log_basis
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Typed coefficient dependencies of a row family.
///
/// Keeping these edges explicit makes the layout a checked relation graph, not
/// merely a collection of offsets.
pub enum RelationRowInputs {
    Consistency {
        z: Vec<RelationSegmentId>,
        e: Vec<RelationSegmentId>,
    },
    A {
        z: RelationSegmentId,
    },
    B {
        t: RelationSegmentId,
        compression_input: Option<GadgetInput>,
    },
    D {
        e: Vec<RelationSegmentId>,
        compression_input: Option<GadgetInput>,
    },
    Compression {
        input: RelationSegmentId,
        successor: Option<GadgetInput>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// A contiguous family of rows with one algebraic role and native ring.
///
/// `native_ring_dim` belongs to the family, not to the entire matrix. Existing
/// fused execution is uniform-dimension and must call
/// [`RelationRowPlan::validate_uniform_execution`]; future per-family execution
/// can consume this richer description directly.
pub struct RelationRowFamily {
    id: RelationRowId,
    rows: RowSpan,
    native_ring_dim: usize,
    quotient: RelationSegmentId,
    inputs: RelationRowInputs,
    rhs: RelationRowRhs,
}

impl RelationRowFamily {
    pub fn id(&self) -> RelationRowId {
        self.id
    }

    pub fn rows(&self) -> RowSpan {
        self.rows
    }

    pub fn native_ring_dim(&self) -> usize {
        self.native_ring_dim
    }

    pub fn quotient(&self) -> RelationSegmentId {
        self.quotient
    }

    pub fn inputs(&self) -> &RelationRowInputs {
        &self.inputs
    }

    pub fn rhs(&self) -> RelationRowRhs {
        self.rhs
    }
}

/// Complete logical row schedule, including the non-matrix trace row.
///
/// Families occupy rows `0..trace_row`. `trace_row` itself is a field-level
/// evaluation row: it is not a quotient-bearing [`RelationRowFamily`]. The
/// domain is then padded to `padded_row_count`, a checked power of two.
///
/// ```text
/// 0                                              trace_row       padded_row_count
/// │ families (live relation-matrix rows)         │ trace │ unused padding │
/// ├──────────────────────────────────────────────┼───────┼────────────────┤
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationRowPlan {
    /// Stable current-first order used by body segments, rows, witness
    /// execution, and proof interpretation.
    group_order: Vec<RelationGroupId>,
    families: Vec<RelationRowFamily>,
    trace_row: usize,
    padded_row_count: usize,
}

impl RelationRowPlan {
    /// Compile the checked base relation-row schedule from authenticated level context.
    pub fn compile_base(
        lp: &LevelParams,
        opening: &OpeningClaimsLayout,
        row_layout: RelationMatrixRowLayout,
    ) -> Result<Self, AkitaError> {
        lp.validate_root_opening_batch(opening)?;
        let mut group_order = Vec::with_capacity(opening.num_groups());
        group_order.push(RelationGroupId::Current);
        for index in 0..lp.precommitted_groups.len() {
            group_order.push(RelationGroupId::Precommitted { index });
        }

        let mut families = Vec::new();
        let mut row_cursor = 0usize;
        let mut push = |id, rows, native_ring_dim, inputs, rhs| -> Result<(), AkitaError> {
            if rows == 0 || native_ring_dim == 0 {
                return Err(AkitaError::InvalidSetup(
                    "relation row family has zero rows or ring dimension".into(),
                ));
            }
            let start = row_cursor;
            row_cursor = checked_add(row_cursor, rows, "row count")?;
            families.push(RelationRowFamily {
                id,
                rows: RowSpan { start, len: rows },
                native_ring_dim,
                quotient: RelationSegmentId::BaseQuotient { row: id },
                inputs,
                rhs,
            });
            Ok(())
        };
        let a_dim = lp.a_key.sis_table_key().ring_dimension as usize;
        push(
            RelationRowId::Consistency,
            1,
            a_dim,
            RelationRowInputs::Consistency {
                z: group_order
                    .iter()
                    .map(|&group| RelationSegmentId::Z { group })
                    .collect(),
                e: group_order
                    .iter()
                    .map(|&group| RelationSegmentId::E { group })
                    .collect(),
            },
            RelationRowRhs::Zero,
        )?;
        for &group in &group_order {
            let (a_rows, a_dim, b_rows, b_dim) = match group {
                RelationGroupId::Current => (
                    lp.a_key.row_len(),
                    lp.a_key.sis_table_key().ring_dimension as usize,
                    lp.b_key.row_len(),
                    lp.b_key.sis_table_key().ring_dimension as usize,
                ),
                RelationGroupId::Precommitted { index } => {
                    let pre = lp.precommitted_groups.get(index).ok_or_else(|| {
                        AkitaError::InvalidSetup("relation precommitted group is missing".into())
                    })?;
                    (
                        pre.a_key.row_len(),
                        pre.a_key.sis_table_key().ring_dimension as usize,
                        pre.b_key.row_len(),
                        pre.b_key.sis_table_key().ring_dimension as usize,
                    )
                }
            };
            push(
                RelationRowId::A { group },
                a_rows,
                a_dim,
                RelationRowInputs::A {
                    z: RelationSegmentId::Z { group },
                },
                RelationRowRhs::Zero,
            )?;
            push(
                RelationRowId::B { group },
                b_rows,
                b_dim,
                RelationRowInputs::B {
                    t: RelationSegmentId::T { group },
                    compression_input: None,
                },
                RelationRowRhs::Commitment { group },
            )?;
        }
        if row_layout == RelationMatrixRowLayout::WithDBlock {
            push(
                RelationRowId::D,
                lp.d_key.row_len(),
                lp.d_key.sis_table_key().ring_dimension as usize,
                RelationRowInputs::D {
                    e: group_order
                        .iter()
                        .map(|&group| RelationSegmentId::E { group })
                        .collect(),
                    compression_input: None,
                },
                RelationRowRhs::Opening,
            )?;
        }
        let trace_row = row_cursor;
        Ok(Self {
            group_order,
            families,
            trace_row,
            padded_row_count: checked_padded_row_count(trace_row)?,
        })
    }

    pub fn group_order(&self) -> &[RelationGroupId] {
        &self.group_order
    }

    pub fn families(&self) -> &[RelationRowFamily] {
        &self.families
    }

    /// Validate that this plan can be consumed by a fused, uniform-dimension kernel.
    ///
    /// Compression rows require per-family execution and are therefore rejected at
    /// this boundary even when their native dimension happens to equal `dimension`.
    pub fn validate_uniform_execution(&self, dimension: usize) -> Result<(), AkitaError> {
        for family in &self.families {
            if matches!(family.id, RelationRowId::Compression { .. }) {
                return Err(AkitaError::InvalidInput(
                    "compressed relation rows require per-family execution".into(),
                ));
            }
            if family.native_ring_dim != dimension {
                return Err(AkitaError::InvalidInput(format!(
                    "relation row family {:?} has native dimension {}, not uniform execution dimension {dimension}",
                    family.id, family.native_ring_dim
                )));
            }
        }
        Ok(())
    }

    pub fn trace_row(&self) -> usize {
        self.trace_row
    }

    pub fn padded_row_count(&self) -> usize {
        self.padded_row_count
    }

    pub fn family(&self, id: RelationRowId) -> Result<&RelationRowFamily, AkitaError> {
        self.families
            .iter()
            .find(|family| family.id == id)
            .ok_or_else(|| AkitaError::InvalidSetup("relation row family is absent".into()))
    }

    pub fn rhs_coeff_len(&self) -> Result<usize, AkitaError> {
        self.families.iter().try_fold(0usize, |total, family| {
            let len = family
                .rows
                .len
                .checked_mul(family.native_ring_dim)
                .ok_or_else(|| AkitaError::InvalidSetup("relation RHS length overflow".into()))?;
            let next = total
                .checked_add(len)
                .ok_or_else(|| AkitaError::InvalidSetup("relation RHS length overflow".into()))?;
            if next > DEFAULT_MAX_SEQUENCE_LEN {
                return Err(AkitaError::InvalidSetup(
                    "relation RHS length exceeds sequence cap".into(),
                ));
            }
            Ok(next)
        })
    }
}

/// Single checked authority for relation coefficients, rows, and range support.
///
/// The layout connects three views which must not be conflated:
///
/// ```text
/// RelationLayout
/// ├─ segments: logical coefficient addresses (includes compression)
/// ├─ row_plan: typed matrix families + separate trace row + padding
/// ├─ negative_binary_support: normalized runs within `segments`
/// └─ witness_layout: physical chunk projection (base Z/E/T + base q only)
/// ```
///
/// Negative-binary support contains exactly the `Xi` input segments whose
/// authenticated map alphabet is negative binary; quotient segments are never
/// included. The spans are stored as sorted, disjoint [`CoeffSpan`]s, and the
/// digit range-check augments its ordinary equality support with exactly these
/// runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationLayout {
    segments: Vec<RelationSegment>,
    row_plan: RelationRowPlan,
    /// Sorted, disjoint coefficient runs that augment the ordinary digit check.
    negative_binary_support: Vec<CoeffSpan>,
    support_derivation_version: u8,
    total_coeffs: usize,
    witness_layout: WitnessLayout,
}

fn checked_add(current: usize, add: usize, label: &str) -> Result<usize, AkitaError> {
    let next = current
        .checked_add(add)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("logical {label} overflow")))?;
    if next > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(format!(
            "logical {label} {next} exceeds cap {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    Ok(next)
}

fn allocate(cursor: &mut usize, len: usize, label: &str) -> Result<CoeffSpan, AkitaError> {
    if len == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "logical {label} must be non-zero"
        )));
    }
    let start = *cursor;
    // This cursor describes a logical, streamed coefficient arena; it is not a
    // verifier allocation or serialized sequence and may exceed the per-carrier
    // allocation cap. Concrete row domains and compact carriers enforce their
    // own caps at their allocation boundaries.
    *cursor = start
        .checked_add(len)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("logical {label} overflow")))?;
    Ok(CoeffSpan { start, len })
}

fn checked_product(values: &[usize], label: &str) -> Result<usize, AkitaError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| AkitaError::InvalidSetup(format!("logical {label} overflow")))
    })
}

fn compression_id(id: CompressionSegmentId) -> RelationSegmentId {
    match id {
        CompressionSegmentId::Xi { source, map } => {
            RelationSegmentId::CompressionInput { source, map }
        }
        CompressionSegmentId::Quotient { source, map } => {
            RelationSegmentId::CompressionQuotient { source, map }
        }
    }
}

fn segment_span(
    segments: &[RelationSegment],
    id: RelationSegmentId,
) -> Result<CoeffSpan, AkitaError> {
    segments
        .iter()
        .find_map(|segment| (segment.id == id).then_some(segment.span))
        .ok_or_else(|| AkitaError::InvalidSetup("relation segment reference is missing".into()))
}

fn normalize_support(
    mut spans: Vec<CoeffSpan>,
    total: usize,
) -> Result<Vec<CoeffSpan>, AkitaError> {
    spans.sort_unstable_by_key(|span| span.start);
    let mut normalized: Vec<CoeffSpan> = Vec::with_capacity(spans.len());
    for span in spans {
        if span.len == 0 || span.end()? > total {
            return Err(AkitaError::InvalidSetup(
                "negative-binary support is empty or outside the logical arena".into(),
            ));
        }
        if let Some(last) = normalized.last_mut() {
            let last_end = last.end()?;
            if span.start < last_end {
                return Err(AkitaError::InvalidSetup(
                    "negative-binary support overlaps".into(),
                ));
            }
            if span.start == last_end {
                last.len = last.len.checked_add(span.len).ok_or_else(|| {
                    AkitaError::InvalidSetup("negative-binary support overflow".into())
                })?;
                continue;
            }
        }
        normalized.push(span);
    }
    Ok(normalized)
}

fn checked_padded_row_count(trace_row: usize) -> Result<usize, AkitaError> {
    let padded = trace_row
        .checked_add(1)
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(|| AkitaError::InvalidSetup("relation padded row count overflow".into()))?;
    if padded > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(format!(
            "relation padded row count {padded} exceeds cap {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    Ok(padded)
}

fn family_for_source(source: CompressionSourceId) -> RelationRowId {
    match source {
        CompressionSourceId::CurrentOuter => RelationRowId::B {
            group: RelationGroupId::Current,
        },
        CompressionSourceId::PrecommittedOuter { index } => RelationRowId::B {
            group: RelationGroupId::Precommitted { index },
        },
        CompressionSourceId::Opening => RelationRowId::D,
    }
}

/// Compile the logical relation and its physical base-witness projection.
fn compile_relation_layout(
    lp: &LevelParams,
    opening: &OpeningClaimsLayout,
    row_layout: RelationMatrixRowLayout,
    field_bits: u32,
    compression: Option<&compression_semantics::CompiledCompressionSemantics>,
) -> Result<RelationLayout, AkitaError> {
    if lp.log_basis == 0 || lp.log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(
            "relation log_basis must be in 1..128".into(),
        ));
    }
    if field_bits != lp.field_bits_for_cache() {
        return Err(AkitaError::InvalidSetup(
            "relation field width disagrees with authenticated level parameters".into(),
        ));
    }
    let base_plan = RelationRowPlan::compile_base(lp, opening, row_layout)?;
    let final_group = opening.root_final_group_index()?;
    let quotient_levels = compute_num_digits_full_field(field_bits, lp.log_basis);
    if quotient_levels == 0 {
        return Err(AkitaError::InvalidSetup(
            "relation quotient decomposition is empty".into(),
        ));
    }

    let mut coeff_cursor = 0usize;
    let mut segments = Vec::new();
    let groups = base_plan
        .group_order()
        .iter()
        .copied()
        .map(|group| {
            let opening_index = match group {
                RelationGroupId::Current => final_group,
                RelationGroupId::Precommitted { index } => index,
            };
            (group, opening_index)
        })
        .collect::<Vec<_>>();

    // Logical group-major body. This is independent of WitnessLayout chunks.
    for &(group, opening_index) in &groups {
        let params = lp.root_group_params(opening, opening_index)?;
        let claims = opening.group_layout(opening_index)?.num_polynomials();
        let fold_digits = lp.num_digits_fold_for_params(params, claims, field_bits)?;
        let (a_dim, b_dim) = match group {
            RelationGroupId::Current => (
                lp.a_key.sis_table_key().ring_dimension as usize,
                lp.b_key.sis_table_key().ring_dimension as usize,
            ),
            RelationGroupId::Precommitted { index } => {
                let pre = lp.precommitted_groups.get(index).ok_or_else(|| {
                    AkitaError::InvalidSetup("relation precommitted group is missing".into())
                })?;
                (
                    pre.a_key.sis_table_key().ring_dimension as usize,
                    pre.b_key.sis_table_key().ring_dimension as usize,
                )
            }
        };
        let d_dim = lp.d_key.sis_table_key().ring_dimension as usize;
        let body = [
            (
                RelationSegmentId::Z { group },
                checked_product(
                    &[
                        params.block_len(),
                        params.num_digits_commit(),
                        fold_digits,
                        a_dim,
                    ],
                    "Z coefficients",
                )?,
            ),
            (
                RelationSegmentId::E { group },
                checked_product(
                    &[claims, params.num_blocks(), params.num_digits_open(), d_dim],
                    "E coefficients",
                )?,
            ),
            (
                RelationSegmentId::T { group },
                checked_product(
                    &[
                        claims,
                        params.num_blocks(),
                        params.a_rows_len(),
                        params.num_digits_open(),
                        b_dim,
                    ],
                    "T coefficients",
                )?,
            ),
        ];
        for (id, len) in body {
            let span = allocate(&mut coeff_cursor, len, "body coefficients")?;
            segments.push(RelationSegment { id, span });
        }
    }

    let mut families = base_plan.families().to_vec();
    let mut row_cursor = base_plan.trace_row();
    for family in &families {
        let quotient_len = checked_product(
            &[
                family.rows().len(),
                family.native_ring_dim(),
                quotient_levels,
            ],
            "quotient",
        )?;
        let span = allocate(&mut coeff_cursor, quotient_len, "quotient coefficients")?;
        segments.push(RelationSegment {
            id: family.quotient(),
            span,
        });
    }

    let mut support_ids = Vec::new();
    let support_derivation_version = compression_semantics::BINARY_SUPPORT_DERIVATION_VERSION;
    if let Some(local) = compression {
        if local.binary_support_derivation_version != support_derivation_version {
            return Err(AkitaError::InvalidSetup(
                "compression support derivation version mismatch".into(),
            ));
        }
        let compression_base = coeff_cursor;
        let mut expected_local = 0usize;
        for segment in &local.segments {
            if segment.span.start != expected_local || segment.span.len == 0 {
                return Err(AkitaError::InvalidSetup(
                    "compression segments are not canonical and contiguous".into(),
                ));
            }
            expected_local = segment.span.end()?;
            let start = compression_base
                .checked_add(segment.span.start)
                .ok_or_else(|| AkitaError::InvalidSetup("compression offset overflow".into()))?;
            segments.push(RelationSegment {
                id: compression_id(segment.id),
                span: CoeffSpan {
                    start,
                    len: segment.span.len,
                },
            });
        }
        if expected_local != local.total_coeffs {
            return Err(AkitaError::InvalidSetup(
                "compression coefficient total disagrees with its segments".into(),
            ));
        }
        coeff_cursor = compression_base
            .checked_add(local.total_coeffs)
            .ok_or_else(|| AkitaError::InvalidSetup("coefficient total overflow".into()))?;

        for augmentation in &local.augmentations {
            let target = family_for_source(augmentation.source);
            let input = compression_id(augmentation.compression_input);
            let family = families
                .iter_mut()
                .find(|family| family.id == target)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("compression source row family is absent".into())
                })?;
            let expected_rhs = match augmentation.source {
                CompressionSourceId::CurrentOuter => RelationRowRhs::Commitment {
                    group: RelationGroupId::Current,
                },
                CompressionSourceId::PrecommittedOuter { index } => RelationRowRhs::Commitment {
                    group: RelationGroupId::Precommitted { index },
                },
                CompressionSourceId::Opening => RelationRowRhs::Opening,
            };
            if family.rhs != expected_rhs {
                return Err(AkitaError::InvalidSetup(
                    "compression source RHS is not the expected base payload".into(),
                ));
            }
            family.rhs = RelationRowRhs::Zero;
            let compression_input = match &mut family.inputs {
                RelationRowInputs::B {
                    compression_input, ..
                }
                | RelationRowInputs::D {
                    compression_input, ..
                } => compression_input,
                _ => {
                    return Err(AkitaError::InvalidSetup(
                        "compression source does not name an augmentable B/D family".into(),
                    ));
                }
            };
            if compression_input
                .replace(GadgetInput {
                    segment: input,
                    log_basis: augmentation.compression_log_basis,
                })
                .is_some()
            {
                return Err(AkitaError::InvalidSetup(
                    "compression source row family is augmented twice".into(),
                ));
            }
            let _ = segment_span(&segments, input)?;
        }

        let compression_row_base = row_cursor;
        let mut expected_local_row = 0usize;
        for row in &local.rows {
            if row.rows.start != expected_local_row || row.rows.len == 0 {
                return Err(AkitaError::InvalidSetup(
                    "compression row spans are not canonical and contiguous".into(),
                ));
            }
            expected_local_row = checked_add(expected_local_row, row.rows.len, "local rows")?;
            let rows = RowSpan {
                start: checked_add(
                    compression_row_base,
                    row.rows.start,
                    "compression row offset",
                )?,
                len: row.rows.len,
            };
            let input = compression_id(row.input);
            let successor = row.successor.map(compression_id);
            let successor_input = match (successor, row.successor_log_basis) {
                (Some(segment), Some(log_basis)) => Some(GadgetInput { segment, log_basis }),
                (None, None) => None,
                _ => {
                    return Err(AkitaError::InvalidSetup(
                        "compression successor and gadget basis disagree".into(),
                    ));
                }
            };
            let quotient = compression_id(row.quotient);
            let _ = segment_span(&segments, input)?;
            let _ = segment_span(&segments, quotient)?;
            if let Some(successor) = successor {
                let _ = segment_span(&segments, successor)?;
            }
            families.push(RelationRowFamily {
                id: RelationRowId::Compression {
                    source: row.id.source,
                    map: row.id.map,
                },
                rows,
                native_ring_dim: row.native_ring_dim,
                quotient,
                inputs: RelationRowInputs::Compression {
                    input,
                    successor: successor_input,
                },
                rhs: match row.rhs {
                    CompressionRowRhs::Zero => RelationRowRhs::Zero,
                    CompressionRowRhs::TerminalPayload { coeffs } => {
                        RelationRowRhs::TerminalPayload { coeffs }
                    }
                },
            });
        }
        if expected_local_row != local.total_rows {
            return Err(AkitaError::InvalidSetup(
                "compression row total disagrees with its spans".into(),
            ));
        }
        row_cursor = checked_add(compression_row_base, local.total_rows, "relation rows")?;
        support_ids.extend(
            local
                .negative_binary_inputs
                .iter()
                .copied()
                .map(compression_id),
        );
    }

    let support = support_ids
        .into_iter()
        .map(|id| segment_span(&segments, id))
        .collect::<Result<Vec<_>, _>>()?;
    let negative_binary_support = normalize_support(support, coeff_cursor)?;
    let trace_row = row_cursor;
    let padded_row_count = checked_padded_row_count(trace_row)?;

    let row_plan = RelationRowPlan {
        group_order: groups.iter().map(|(group, _)| *group).collect(),
        families,
        trace_row,
        padded_row_count,
    };
    let witness_layout =
        RelationLayout::compile_witness_layout(&segments, &row_plan, lp, field_bits)?;
    Ok(RelationLayout {
        segments,
        row_plan,
        negative_binary_support,
        support_derivation_version,
        total_coeffs: coeff_cursor,
        witness_layout,
    })
}

impl RelationLayout {
    /// Derive the canonical relation layout from authenticated statement inputs.
    ///
    /// The returned layout is checked derived state; it is not itself serialized
    /// or independently authenticated.
    pub fn from_authenticated_statement(
        lp: &LevelParams,
        opening: &OpeningClaimsLayout,
        row_layout: RelationMatrixRowLayout,
        field_bits: u32,
    ) -> Result<Self, AkitaError> {
        compile_relation_layout(lp, opening, row_layout, field_bits, None)
    }

    pub(in crate::layout) fn compile_compressed(
        lp: &LevelParams,
        opening: &OpeningClaimsLayout,
        compression: &compression_semantics::CompiledCompressionSemantics,
        field_bits: u32,
    ) -> Result<Self, AkitaError> {
        compile_relation_layout(
            lp,
            opening,
            RelationMatrixRowLayout::WithDBlock,
            field_bits,
            Some(compression),
        )
    }

    pub(in crate::layout) fn compile_terminal_compressed(
        lp: &LevelParams,
        opening: &OpeningClaimsLayout,
        compression: &compression_semantics::CompiledCompressionSemantics,
        field_bits: u32,
    ) -> Result<Self, AkitaError> {
        compile_relation_layout(
            lp,
            opening,
            RelationMatrixRowLayout::WithoutDBlock,
            field_bits,
            Some(compression),
        )
    }

    pub fn segments(&self) -> &[RelationSegment] {
        &self.segments
    }

    pub fn segment(&self, id: RelationSegmentId) -> Result<&RelationSegment, AkitaError> {
        self.segments
            .iter()
            .find(|segment| segment.id == id)
            .ok_or_else(|| AkitaError::InvalidSetup("relation segment is absent".into()))
    }

    pub fn row_plan(&self) -> &RelationRowPlan {
        &self.row_plan
    }

    pub fn negative_binary_support(&self) -> &[CoeffSpan] {
        &self.negative_binary_support
    }

    pub fn support_derivation_version(&self) -> u8 {
        self.support_derivation_version
    }

    pub fn total_coeffs(&self) -> usize {
        self.total_coeffs
    }

    fn body_ring_len(
        segments: &[RelationSegment],
        id: RelationSegmentId,
        native_ring_dim: usize,
    ) -> Result<usize, AkitaError> {
        let len = segments
            .iter()
            .find_map(|segment| (segment.id == id).then_some(segment.span.len))
            .ok_or_else(|| AkitaError::InvalidSetup("relation body segment is absent".into()))?;
        if native_ring_dim == 0 || !len.is_multiple_of(native_ring_dim) {
            return Err(AkitaError::InvalidSetup(
                "relation body segment disagrees with its native ring dimension".into(),
            ));
        }
        Ok(len / native_ring_dim)
    }

    /// Project base `Z/E/T` and base quotient geometry into physical chunks.
    ///
    /// For one group, multiple chunks repeat `Z`, partition `E/T`, and place the
    /// shared base-quotient `r` tail only in the last chunk. For multiple groups,
    /// configured multi-chunk mode is rejected; one full `Z/E/T` chunk is emitted
    /// per group in current-first order, with the shared `r` tail only in the last
    /// group chunk. Compression inputs and quotients remain outside
    /// [`WitnessLayout`].
    fn compile_witness_layout(
        segments: &[RelationSegment],
        row_plan: &RelationRowPlan,
        lp: &LevelParams,
        field_bits: u32,
    ) -> Result<WitnessLayout, AkitaError> {
        lp.witness_chunk.validate()?;
        let num_chunks = lp.witness_chunk.num_chunks;
        if num_chunks == 0 {
            return Err(AkitaError::InvalidSetup(
                "witness chunk count must be at least one".into(),
            ));
        }
        let base_rows = row_plan.families.iter().try_fold(0usize, |total, family| {
            if matches!(family.quotient, RelationSegmentId::BaseQuotient { .. }) {
                total.checked_add(family.rows.len).ok_or_else(|| {
                    AkitaError::InvalidSetup("base quotient row count overflow".into())
                })
            } else {
                Ok(total)
            }
        })?;
        let r_len = base_rows
            .checked_mul(compute_num_digits_full_field(field_bits, lp.log_basis))
            .ok_or_else(|| AkitaError::InvalidSetup("r-tail length overflow".into()))?;
        let mut group_lens = Vec::with_capacity(row_plan.group_order.len());
        for &group in &row_plan.group_order {
            let (a_dim, b_dim) = match group {
                RelationGroupId::Current => (
                    lp.a_key.sis_table_key().ring_dimension as usize,
                    lp.b_key.sis_table_key().ring_dimension as usize,
                ),
                RelationGroupId::Precommitted { index } => {
                    let pre = lp.precommitted_groups.get(index).ok_or_else(|| {
                        AkitaError::InvalidSetup("relation witness group is absent".into())
                    })?;
                    (
                        pre.a_key.sis_table_key().ring_dimension as usize,
                        pre.b_key.sis_table_key().ring_dimension as usize,
                    )
                }
            };
            group_lens.push((
                Self::body_ring_len(segments, RelationSegmentId::Z { group }, a_dim)?,
                Self::body_ring_len(
                    segments,
                    RelationSegmentId::E { group },
                    lp.d_key.sis_table_key().ring_dimension as usize,
                )?,
                Self::body_ring_len(segments, RelationSegmentId::T { group }, b_dim)?,
            ));
        }

        let layout = if group_lens.len() > 1 {
            lp.reject_multi_group_multi_chunk("RelationLayout::witness_layout")?;
            let mut base = 0usize;
            let mut chunks = Vec::with_capacity(group_lens.len());
            let mut chunk_lengths = Vec::with_capacity(group_lens.len());
            for (position, &(z_len, e_len, t_len)) in group_lens.iter().enumerate() {
                let offset_e = base.checked_add(z_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("multi-group e offset overflow".into())
                })?;
                let offset_t = offset_e.checked_add(e_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("multi-group t offset overflow".into())
                })?;
                let after_t = offset_t.checked_add(t_len).ok_or_else(|| {
                    AkitaError::InvalidSetup("multi-group witness stride overflow".into())
                })?;
                let last = position + 1 == group_lens.len();
                chunks.push(WitnessChunkLayout {
                    offset_z: base,
                    offset_e,
                    offset_t,
                    offset_r: last.then_some(after_t),
                    global_block_base: 0,
                });
                chunk_lengths.push(WitnessChunkLengths {
                    z_len,
                    e_len,
                    t_len,
                    r_len: last.then_some(r_len),
                });
                base = after_t;
            }
            WitnessLayout {
                blocks_per_chunk: lp.num_blocks,
                chunks,
                chunk_lengths,
            }
        } else {
            let &(z_len, e_len, t_len) = group_lens.first().ok_or_else(|| {
                AkitaError::InvalidSetup("relation witness group is missing".into())
            })?;
            if num_chunks > 1
                && (!num_chunks.is_power_of_two()
                    || num_chunks > lp.num_blocks
                    || !lp.num_blocks.is_multiple_of(num_chunks)
                    || !e_len.is_multiple_of(num_chunks)
                    || !t_len.is_multiple_of(num_chunks))
            {
                return Err(AkitaError::InvalidSetup(
                    "invalid partitioned witness chunk geometry".into(),
                ));
            }
            let blocks_per_chunk = lp.num_blocks / num_chunks;
            if !blocks_per_chunk.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "witness chunk block window must be a power of two".into(),
                ));
            }
            let e_chunk = e_len / num_chunks;
            let t_chunk = t_len / num_chunks;
            let stride = z_len
                .checked_add(e_chunk)
                .and_then(|n| n.checked_add(t_chunk))
                .ok_or_else(|| AkitaError::InvalidSetup("witness chunk stride overflow".into()))?;
            let mut chunks = Vec::with_capacity(num_chunks);
            let mut chunk_lengths = Vec::with_capacity(num_chunks);
            for index in 0..num_chunks {
                let base = index.checked_mul(stride).ok_or_else(|| {
                    AkitaError::InvalidSetup("witness chunk base overflow".into())
                })?;
                let offset_e = base
                    .checked_add(z_len)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness e offset overflow".into()))?;
                let offset_t = offset_e
                    .checked_add(e_chunk)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness t offset overflow".into()))?;
                let after_t = offset_t
                    .checked_add(t_chunk)
                    .ok_or_else(|| AkitaError::InvalidSetup("witness r offset overflow".into()))?;
                let last = index + 1 == num_chunks;
                chunks.push(WitnessChunkLayout {
                    offset_z: base,
                    offset_e,
                    offset_t,
                    offset_r: last.then_some(after_t),
                    global_block_base: index.checked_mul(blocks_per_chunk).ok_or_else(|| {
                        AkitaError::InvalidSetup("global block base overflow".into())
                    })?,
                });
                chunk_lengths.push(WitnessChunkLengths {
                    z_len,
                    e_len: e_chunk,
                    t_len: t_chunk,
                    r_len: last.then_some(r_len),
                });
            }
            WitnessLayout {
                blocks_per_chunk,
                chunks,
                chunk_lengths,
            }
        };
        Ok(layout)
    }

    pub fn witness_layout(
        &self,
        witness_ring_len: Option<usize>,
    ) -> Result<&WitnessLayout, AkitaError> {
        if let Some(capacity) = witness_ring_len {
            let r_len = self
                .witness_layout
                .chunk_lengths
                .last()
                .and_then(|lengths| lengths.r_len)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation witness r tail is absent".into())
                })?;
            let needed = self
                .witness_layout
                .r_offset()?
                .checked_add(r_len)
                .ok_or_else(|| AkitaError::InvalidSetup("witness capacity overflow".into()))?;
            if needed > capacity {
                return Err(AkitaError::InvalidSetup(format!(
                    "resolved witness layout requires {needed} ring columns but only {capacity} are committed"
                )));
            }
        }
        Ok(&self.witness_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{CanonicalField, Prime128OffsetA7F7 as F};

    use crate::layout::compression::{
        validate_and_compile, CompressionAlphabet, CompressionCatalogContext, CompressionChainSpec,
        CompressionMapSpec,
    };
    use crate::sis::{sis_table_key_for_linf_bound, AjtaiKeyParams, DEFAULT_SIS_SECURITY_BITS};
    fn level() -> (LevelParams, OpeningClaimsLayout) {
        let lp = LevelParams::params_only(
            crate::SisModulusFamily::Q128,
            64,
            6,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(64),
        )
        .with_decomp(1, 1, 1, 1, 0)
        .unwrap();
        (lp, OpeningClaimsLayout::new(2, 1).unwrap())
    }

    fn certified_key(d: usize, raw_bound: u128, cols: usize) -> AjtaiKeyParams {
        let table = sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_BITS,
            crate::SisModulusFamily::Q128,
            d as u32,
            raw_bound,
        )
        .unwrap();
        AjtaiKeyParams::try_new_with_min_rank(table, cols).unwrap()
    }

    fn chain(
        source: CompressionSourceId,
        source_key: &AjtaiKeyParams,
        alphabets: [CompressionAlphabet; 2],
    ) -> CompressionChainSpec {
        let mut previous =
            source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
        let maps = alphabets
            .into_iter()
            .enumerate()
            .map(|(index, alphabet)| {
                let d = if index == 0 { 64 } else { 32 };
                let depth = match alphabet {
                    CompressionAlphabet::NegativeBinary => 128,
                    CompressionAlphabet::OpeningBase { log_basis } => {
                        crate::sis::num_digits_for_bound(128, 128, log_basis)
                    }
                };
                let input = previous * depth;
                let key = certified_key(
                    d,
                    if alphabet == CompressionAlphabet::NegativeBinary {
                        1
                    } else {
                        63
                    },
                    input / d,
                );
                previous = key.row_len() * d;
                CompressionMapSpec { key, alphabet }
            })
            .collect();
        CompressionChainSpec { source, maps }
    }

    #[test]
    fn empty_compression_is_byte_order_identical_to_the_single_group_oracle() {
        let (lp, opening) = level();
        let layout = compile_relation_layout(
            &lp,
            &opening,
            RelationMatrixRowLayout::WithDBlock,
            lp.field_bits_for_cache(),
            None,
        )
        .unwrap();
        let base_plan =
            RelationRowPlan::compile_base(&lp, &opening, RelationMatrixRowLayout::WithDBlock)
                .unwrap();
        let lens = layout.witness_layout(None).unwrap().chunk_lengths[0];
        let d = lp.ring_dimension;
        assert_eq!(
            layout.segments[0].span,
            CoeffSpan {
                start: 0,
                len: lens.z_len * d
            }
        );
        assert_eq!(layout.segments[1].span.len, lens.e_len * d);
        assert_eq!(layout.segments[2].span.len, lens.t_len * d);
        assert_eq!(layout.row_plan, base_plan);
        assert_eq!(
            layout.row_plan.families[0].rows,
            RowSpan { start: 0, len: 1 }
        );
        assert_eq!(layout.row_plan.families[1].rows.start, 1);
        assert_eq!(
            layout.row_plan.families[2].rows.start,
            1 + layout.row_plan.families[1].rows.len
        );
        assert!(layout.negative_binary_support.is_empty());
        let quotient_coeffs = layout.row_plan.trace_row
            * compute_num_digits_full_field(F::modulus_bits(), lp.log_basis)
            * d;
        assert_eq!(
            layout.total_coeffs,
            (lens.z_len + lens.e_len + lens.t_len) * d + quotient_coeffs
        );
        assert_eq!(
            layout.row_plan.padded_row_count,
            (layout.row_plan.trace_row + 1).next_power_of_two()
        );
    }

    #[test]
    fn base_plan_scalar_layouts_and_padding_are_exact() {
        let lp = LevelParams::params_only(
            crate::SisModulusFamily::Q128,
            64,
            6,
            2,
            2,
            3,
            SparseChallengeConfig::pm1_only(64),
        )
        .with_decomp(1, 1, 1, 1, 0)
        .unwrap();
        let opening = OpeningClaimsLayout::new(2, 1).unwrap();
        let with_d =
            RelationRowPlan::compile_base(&lp, &opening, RelationMatrixRowLayout::WithDBlock)
                .unwrap();
        assert_eq!(with_d.trace_row(), 8);
        assert_eq!(with_d.padded_row_count(), 16);
        assert_eq!(
            with_d.families().iter().map(|f| f.id()).collect::<Vec<_>>(),
            vec![
                RelationRowId::Consistency,
                RelationRowId::A {
                    group: RelationGroupId::Current
                },
                RelationRowId::B {
                    group: RelationGroupId::Current
                },
                RelationRowId::D,
            ]
        );
        let without_d =
            RelationRowPlan::compile_base(&lp, &opening, RelationMatrixRowLayout::WithoutDBlock)
                .unwrap();
        assert_eq!(without_d.trace_row(), 5);
        assert_eq!(without_d.padded_row_count(), 8);
        assert!(without_d.family(RelationRowId::D).is_err());
    }

    #[test]
    fn relation_compiler_rejects_stale_field_width_and_invalid_basis_without_panicking() {
        let (mut lp, opening) = level();
        assert!(RelationLayout::from_authenticated_statement(
            &lp,
            &opening,
            RelationMatrixRowLayout::WithDBlock,
            32,
        )
        .is_err());
        assert!(RelationLayout::from_authenticated_statement(
            &lp,
            &opening,
            RelationMatrixRowLayout::WithDBlock,
            lp.field_bits_for_cache(),
        )
        .is_ok());
        for invalid in [0, 128] {
            lp.log_basis = invalid;
            let result = std::panic::catch_unwind(|| {
                RelationLayout::from_authenticated_statement(
                    &lp,
                    &opening,
                    RelationMatrixRowLayout::WithDBlock,
                    lp.field_bits_for_cache(),
                )
            });
            assert!(result.is_ok());
            assert!(result.unwrap().is_err());
        }
    }

    #[test]
    fn malformed_support_is_rejected_and_adjacent_runs_are_normalized() {
        assert!(normalize_support(
            vec![
                CoeffSpan { start: 3, len: 2 },
                CoeffSpan { start: 4, len: 2 }
            ],
            8
        )
        .is_err());
        assert_eq!(
            normalize_support(
                vec![
                    CoeffSpan { start: 3, len: 2 },
                    CoeffSpan { start: 5, len: 2 }
                ],
                8
            )
            .unwrap(),
            vec![CoeffSpan { start: 3, len: 4 }]
        );
        assert!(normalize_support(vec![CoeffSpan { start: 7, len: 2 }], 8).is_err());
        assert_eq!(
            checked_padded_row_count(DEFAULT_MAX_SEQUENCE_LEN - 1).unwrap(),
            DEFAULT_MAX_SEQUENCE_LEN
        );
        assert!(checked_padded_row_count(DEFAULT_MAX_SEQUENCE_LEN).is_err());
    }

    #[test]
    fn checked_compression_extends_rows_and_directly_augments_existing_sources() {
        let (mut lp, opening) = level();
        lp.b_key = certified_key(64, 63, 1);
        lp.d_key = certified_key(64, 63, 1);
        let specs = vec![
            chain(
                CompressionSourceId::CurrentOuter,
                &lp.b_key,
                [
                    CompressionAlphabet::NegativeBinary,
                    CompressionAlphabet::NegativeBinary,
                ],
            ),
            chain(
                CompressionSourceId::Opening,
                &lp.d_key,
                [
                    CompressionAlphabet::OpeningBase { log_basis: 6 },
                    CompressionAlphabet::NegativeBinary,
                ],
            ),
        ];
        let catalog = validate_and_compile::<F>(
            &lp,
            CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
            64,
            specs,
        )
        .unwrap();
        let base = compile_relation_layout(
            &lp,
            &opening,
            RelationMatrixRowLayout::WithDBlock,
            lp.field_bits_for_cache(),
            None,
        )
        .unwrap();
        let layout = catalog.co_generated_relation_layout().unwrap();

        assert!(layout.row_plan.trace_row > base.row_plan.trace_row);
        assert!(matches!(
            &layout.row_plan.families[2].inputs,
            RelationRowInputs::B {
                compression_input: Some(GadgetInput {
                    segment: RelationSegmentId::CompressionInput {
                        source: CompressionSourceId::CurrentOuter,
                        map: 0
                    },
                    log_basis: 1,
                }),
                ..
            }
        ));
        assert!(matches!(
            &layout.row_plan.families[3].inputs,
            RelationRowInputs::D {
                compression_input: Some(GadgetInput {
                    segment: RelationSegmentId::CompressionInput {
                        source: CompressionSourceId::Opening,
                        map: 0
                    },
                    log_basis: 6,
                }),
                ..
            }
        ));
        let first_compression = &layout.row_plan.families[base.row_plan.families.len()];
        assert!(layout.row_plan.validate_uniform_execution(64).is_err());
        assert_eq!(first_compression.rows.start, base.row_plan.trace_row);
        assert!(matches!(
            first_compression.inputs,
            RelationRowInputs::Compression {
                successor: Some(_),
                ..
            }
        ));
        assert!(!layout.negative_binary_support.is_empty());
        assert!(layout
            .negative_binary_support
            .windows(2)
            .all(|pair| pair[0].end().unwrap() < pair[1].start));
    }
}
