use crate::backend::onehot::{MultiChunkEntry, SingleChunkEntry};
use crate::backend::sparse_ring::SparseRingBlockEntry;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, FieldCore};

/// Flat block table handed to a compute backend.
///
/// `entries[offsets[i]..offsets[i + 1]]` is the entry slice for block `i`.
/// This is the canonical compact representation for sparse per-block work:
/// CPU code may recover per-block slices, while accelerator backends can upload
/// one contiguous entry table plus one offsets table.
#[derive(Debug, Clone, Copy)]
pub struct FlatBlockTable<'a, E> {
    entries: &'a [E],
    offsets: &'a [u32],
}

impl<'a, E> FlatBlockTable<'a, E> {
    /// Build a flat block table from validated storage.
    #[inline]
    pub(crate) fn new(entries: &'a [E], offsets: &'a [u32]) -> Self {
        Self { entries, offsets }
    }

    /// Contiguous sparse entries.
    #[inline]
    pub fn entries(&self) -> &'a [E] {
        self.entries
    }

    /// Block offsets into [`Self::entries`].
    #[inline]
    pub fn offsets(&self) -> &'a [u32] {
        self.offsets
    }

    /// Number of logical blocks.
    #[inline]
    pub fn num_live_blocks(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Entry slice for one block.
    pub fn block(&self, idx: usize) -> Result<&'a [E], AkitaError> {
        let lo = self.offsets.get(idx).copied().ok_or_else(|| {
            AkitaError::InvalidSetup(format!("flat block table missing offset {idx}"))
        })? as usize;
        let hi = self.offsets.get(idx + 1).copied().ok_or_else(|| {
            AkitaError::InvalidSetup(format!("flat block table missing offset {}", idx + 1))
        })? as usize;
        if lo > hi || hi > self.entries.len() {
            return Err(AkitaError::InvalidSetup(format!(
                "flat block table has malformed offsets for block {idx}: {lo}..{hi} over {} entries",
                self.entries.len()
            )));
        }
        Ok(&self.entries[lo..hi])
    }

    pub(crate) fn block_slices(&self) -> Result<Vec<&'a [E]>, AkitaError> {
        (0..self.num_live_blocks())
            .map(|idx| self.block(idx))
            .collect()
    }
}

/// Dense polynomial commit representation handed to the compute backend.
pub enum DenseCommitInput<'a, F: FieldCore, const D: usize> {
    /// Balanced digit planes are already cached by the polynomial.
    CachedDigits {
        /// Per-block digit slices.
        digit_block_slices: Vec<&'a [[i8; D]]>,
        /// Logarithm of the gadget basis used to produce the cached digits.
        log_basis: u32,
    },
    /// Ring coefficients need backend-side digit decomposition.
    CoeffBlocks {
        /// Per-block coefficient slices.
        block_slices: Vec<&'a [CyclotomicRing<F, D>]>,
        /// Number of balanced digits used for the A-side commit.
        num_digits_commit: usize,
        /// Logarithm of the gadget basis.
        log_basis: u32,
    },
}

/// Dense commit operation plan.
pub struct DenseCommitRowsPlan<'a, F: FieldCore, const D: usize> {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Dense polynomial input representation.
    pub input: DenseCommitInput<'a, F, D>,
}

/// One-hot commit input representation.
///
/// The contained entry slices are read-only plan views. They are public so
/// accelerator crates can implement [`super::backend::CommitmentComputeBackend`] without
/// depending on CPU-prepared storage, while construction remains owned by the
/// polynomial representations.
pub enum OneHotCommitBlocks<'a> {
    /// One ring has at most one hot coefficient.
    SingleChunk(FlatBlockTable<'a, SingleChunkEntry>),
    /// One ring may contain several hot coefficients.
    MultiChunk(FlatBlockTable<'a, MultiChunkEntry>),
}

/// One-hot commit operation plan.
pub struct OneHotCommitRowsPlan<'a> {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Number of ring-element positions in each root block.
    pub num_positions_per_block: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Per-block one-hot entries.
    pub(crate) blocks: OneHotCommitBlocks<'a>,
}

impl<'a> OneHotCommitRowsPlan<'a> {
    /// Per-block one-hot entries.
    #[inline]
    pub fn blocks(&self) -> &OneHotCommitBlocks<'a> {
        &self.blocks
    }
}

/// Sparse signed-ring commit operation plan.
pub struct SparseRingCommitRowsPlan<'a> {
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Number of ring-element positions in each root block.
    pub num_positions_per_block: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Per-block sparse signed coefficients.
    pub(crate) blocks: FlatBlockTable<'a, SparseRingBlockEntry>,
}

impl<'a> SparseRingCommitRowsPlan<'a> {
    /// Per-block sparse signed coefficients.
    #[inline]
    pub fn blocks(&self) -> FlatBlockTable<'a, SparseRingBlockEntry> {
        self.blocks
    }
}

/// Recursive witness commit operation plan.
pub struct RecursiveWitnessCommitRowsPlan<'a, const D: usize> {
    /// Recursive witness digit rows, chunked at `D`.
    pub coeffs: &'a [[i8; D]],
    /// Number of rows to produce.
    pub n_rows: usize,
    /// Number of positions in each recursive block.
    pub num_positions_per_block: usize,
    /// Number of logical blocks.
    pub num_live_blocks: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_commit: usize,
    /// Logarithm of the gadget basis.
    pub log_basis: u32,
    /// Known source digit basis, when construction proves every coefficient is
    /// balanced for that basis. A commit basis at least this large can skip a
    /// redundant full witness range scan.
    pub known_balanced_log_basis: Option<u32>,
}

/// Full ring-switch relation operation input.
pub struct RingSwitchRelationRowsPlan<'a, const D: usize> {
    /// Number of D-side cyclic rows to produce.
    pub n_d: usize,
    /// Number of B-side cyclic rows to produce.
    pub n_b: usize,
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// Flat decomposed `e_hat` digits for the D-side relation rows.
    pub e_hat: &'a [[i8; D]],
    /// Flat decomposed inner-commitment digits for the B-side relation rows.
    pub t_hat: &'a [[i8; D]],
    /// One centered `z` segment contributing to A-side quotient rows.
    pub z_segment: &'a [[i32; D]],
    /// Infinity norm of the full centered `z_folded_rings` witness.
    pub z_folded_centered_inf_norm: u32,
    /// Logarithm of the gadget basis used to produce `e_hat` and `t_hat`.
    pub log_basis: u32,
}

/// Additional public-row quotient operation input.
pub struct RingSwitchQuotientRowsPlan<'a, const D: usize> {
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// One centered `z` segment contributing to A-side quotient rows.
    pub z_segment: &'a [[i32; D]],
    /// Infinity norm of the full centered `z_folded_rings` witness.
    pub z_folded_centered_inf_norm: u32,
}

/// Named ring-switch relation rows returned by a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingSwitchRelationRows<F: FieldCore, const D: usize> {
    /// D-side cyclic rows.
    pub d_cyclic: Vec<CyclotomicRing<F, D>>,
    /// B-side cyclic rows.
    pub b_cyclic: Vec<CyclotomicRing<F, D>>,
    /// A-side quotient rows.
    pub a_quotients: Vec<CyclotomicRing<F, D>>,
}
