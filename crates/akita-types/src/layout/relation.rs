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
use std::ops::Range;

use crate::witness::WitnessLayout;
use crate::{LevelParams, OpeningClaimsLayout};

use super::compression::semantics as compression_semantics;
pub use super::compression::CompressionSourceId;
use super::RelationMatrixRowLayout;

mod compiler;
mod rows;
#[cfg(test)]
mod tests;

use compiler::compile_relation_layout;

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
    /// Flat F/H input and quotient coefficients appended after the physical
    /// base z/e/t/r witness. Set only by the relation compiler.
    compression_witness_coeffs: usize,
    /// Authenticated carrier dimension for the physical successor witness.
    carrier_ring_dim: usize,
    witness_layout: WitnessLayout,
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

    /// Physical successor-witness length in field coefficients.
    ///
    /// The existing z/e/t/r witness is stored in authenticated carrier-ring
    /// columns. Compression inputs and quotients are heterogeneous flat field
    /// segments appended after that base. Their combined extent is padded once
    /// so the complete successor witness again occupies whole carrier rings.
    ///
    /// This is the sole physical *length* authority. Semantic segments and
    /// negative-binary support remain in the logical coordinates exposed by
    /// [`Self::segments`], [`Self::total_coeffs`], and
    /// [`Self::negative_binary_support`]. Translating those semantic spans into
    /// physical witness coordinates belongs to the witness-emission and sparse-
    /// provider slice. Carrier padding creates no semantic relation cells.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when any base conversion,
    /// compression addition, or final ring padding overflows `usize`.
    pub fn physical_witness_field_coeff_len(&self) -> Result<usize, AkitaError> {
        let carrier_ring_dim = self.carrier_ring_dim;
        let base_coeffs = self
            .witness_layout
            .ring_len()?
            .checked_mul(carrier_ring_dim)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "relation base witness field coefficient count overflow".into(),
                )
            })?;
        let unpadded_coeffs = base_coeffs
            .checked_add(self.compression_witness_coeffs)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "relation compression witness field coefficient count overflow".into(),
                )
            })?;
        let rings = unpadded_coeffs
            .checked_div(carrier_ring_dim)
            .and_then(|rings| {
                rings.checked_add(usize::from(
                    !unpadded_coeffs.is_multiple_of(carrier_ring_dim),
                ))
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation witness carrier ring count overflow".into())
            })?;
        rings.checked_mul(carrier_ring_dim).ok_or_else(|| {
            AkitaError::InvalidSetup("relation padded witness field coefficient overflow".into())
        })
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
