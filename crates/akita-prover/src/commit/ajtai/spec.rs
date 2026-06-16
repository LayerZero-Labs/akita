//! Matrix-window selection for the Ajtai commit primitive.
//!
//! An Ajtai commitment is `commitment = commitment_key · message`. [`MatrixSpec`]
//! selects which `rows × cols` window of the shared commitment key the multiply
//! reads, plus the ring domain. Every role reads from offset 0 of the shared
//! setup matrix (the prefix-sharing layout); [`MatrixRole`] is retained for
//! validation/tracing only.

/// Which protocol matrix a commit reads from the shared setup matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixRole {
    /// Inner Ajtai matrix `A` (`t = A · s`).
    AInner,
    /// Outer matrix `B` for a single-tier commit.
    BOuter,
    /// First-tier matrix `B'` slice for a tiered commit.
    BOuterTierSlice,
    /// Second-tier matrix `F` for a tiered commit.
    FOuterTier,
    /// Relation matrix `D` (cyclic relation rows).
    DRelation,
}

/// Ring multiplication domain for the commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingDomain {
    /// Negacyclic ring (the default for `A`/`B`/`B'`/`F`).
    Negacyclic,
    /// Cyclic ring (ring-switch relation rows).
    Cyclic,
}

/// Selects a `rows × cols` window of the commitment key (the shared setup
/// matrix). Both dimensions are explicit (the fix for the previous implicit
/// `cols == digits.len()` coupling).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixSpec {
    /// Which protocol matrix this window represents (validation/tracing only).
    pub role: MatrixRole,
    /// Number of committed rows to produce.
    pub rows: usize,
    /// Logical column width read from the shared matrix.
    pub cols: usize,
    /// Ring multiplication domain.
    pub domain: RingDomain,
}
