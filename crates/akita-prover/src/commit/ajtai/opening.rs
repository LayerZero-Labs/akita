//! The Ajtai opening: the message being committed, in its native
//! representation.
//!
//! [`AjtaiOpeningType`] is the closed enum that replaces the four
//! per-representation `*CommitRowsPlan` structs. The matrix window
//! ([`MatrixSpec`](super::spec::MatrixSpec)) is passed separately to
//! `ajtai_commit`; the opening only carries the message data.

use akita_algebra::CyclotomicRing;
use akita_field::FieldCore;

pub(crate) use crate::backend::sparse_ring::SparseRingBlockEntry;
pub(crate) use crate::compute::{FlatBlockTable, OneHotCommitBlocks};

/// Whether a dense A-side commit may skip all-zero plane scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroScan {
    /// Trusted dense input: no zero-plane rescans.
    Dense,
    /// Untrusted input: skip all-zero planes during the scan.
    SkipZeros,
}

/// The opening (message) being committed, in its native representation.
///
/// Each arm corresponds to one concrete kernel family. The CPU backend matches
/// this enum once and dispatches; there is no per-element dispatch.
pub enum AjtaiOpeningType<'a, F: FieldCore, const D: usize> {
    /// Raw ring coeffs per block; decomposed on the fly. (dense `A` fallback)
    CoeffBlocks {
        /// Per-block coefficient slices.
        blocks: Vec<&'a [CyclotomicRing<F, D>]>,
        /// Number of balanced digits used for the A-side commit.
        num_digits: usize,
        /// Logarithm of the gadget basis.
        log_basis: u32,
        /// Whether zero-plane scans may be skipped.
        zero_scan: ZeroScan,
    },
    /// Pre-decomposed trusted i8 digit planes per block. (dense `A` fast path)
    DigitBlocks {
        /// Per-block digit plane slices.
        blocks: Vec<&'a [[i8; D]]>,
        /// Logarithm of the gadget basis used to produce the cached digits.
        log_basis: u32,
        /// Whether zero-plane scans may be skipped.
        zero_scan: ZeroScan,
    },
    /// One flat digit vector = a single block. (`B` / `B'` / `F`)
    DigitVector {
        /// Flat digit planes for the single block.
        digits: &'a [[i8; D]],
        /// Logarithm of the gadget basis.
        log_basis: u32,
    },
    /// Strided recursive witness. (`raw == true` ⇔ `num_digits_commit == 1`,
    /// a signed-i8 stream.)
    StridedDigits {
        /// Recursive witness digit rows, chunked at `D`.
        coeffs: &'a [[i8; D]],
        /// Number of logical blocks.
        num_blocks: usize,
        /// Recursive block length.
        block_len: usize,
        /// Number of balanced digits used for the A-side commit.
        num_digits: usize,
        /// Logarithm of the gadget basis.
        log_basis: u32,
        /// Whether the witness is a raw `δ_commit == 1` signed-i8 stream.
        raw: bool,
    },
    /// One-hot shifted monomials (sparse column selection). (one-hot `A`)
    OneHot {
        /// Per-block one-hot entries.
        blocks: OneHotCommitBlocks<'a>,
        /// Number of balanced digits used for the A-side commit.
        num_digits_commit: usize,
    },
    /// Sparse signed-ring entries (sparse column selection). (sparse-ring `A`)
    SparseRing {
        /// Per-block sparse signed coefficients.
        blocks: FlatBlockTable<'a, SparseRingBlockEntry>,
        /// Number of balanced digits used for the A-side commit.
        num_digits_commit: usize,
    },
}
