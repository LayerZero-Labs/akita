//! Commitment modulo an admitted limb prime, with signed binary sources.
//!
//! The raw arithmetic kernel is independent of root-profile admission. Its output
//! is reduced coefficient storage in the same column/row/coefficient order as
//! `BinaryClearCommitment::images`; it has no proof or serialization format.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

use akita_algebra::{
    binary::field_switch::SwitchField, TrinomialLimbAccumulator, TrinomialLimbDomain,
    TrinomialLimbSlots,
};
use akita_error::{checked, AkitaError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::commit_kernel::{check_source_len, source_element_rank};

mod setup;

pub use setup::commit_binary_clear_small_modulus_prepared;

const DEGREE: usize = 648;
const K: usize = 4;

/// Row-major transforms of a reduced integer matrix over a limb prime.
///
/// Each entry holds 648 signed 32-bit slots and a `u32` prime tag. No original
/// coefficients or source transforms are retained. The reusable domain owns
/// fixed-degree lookup tables and its transform plan. Column counts are chosen
/// at commitment time and do not affect preparation.
#[derive(Debug)]
pub struct PreparedLimbCommitMatrix {
    domain: TrinomialLimbDomain,
    matrix: Vec<TrinomialLimbSlots>,
    n_a: usize,
    m: usize,
    matrix_bytes: usize,
    prepared_bytes: usize,
    workspace_bytes: usize,
    setup_digest: Option<[u8; 32]>,
}

impl PreparedLimbCommitMatrix {
    /// Transform a matrix stored as `(row, element, coefficient)` in `[0,q0)`.
    ///
    /// Reject degrees other than 648, unadmitted primes, zero rank or width,
    /// incorrect length, unreduced coefficients, overflowing sizes, and failed
    /// matrix allocations. Preparation initializes the signed-bit tables too.
    /// The result is unbound and cannot serve the setup-bound entry point; use
    /// [`Self::prepare_for_setup`] for small-modulus setups.
    pub fn prepare(
        q0: u32,
        degree: usize,
        n_a: usize,
        m: usize,
        coefficients: &[u32],
    ) -> Result<Self, AkitaError> {
        if degree != DEGREE {
            return Err(AkitaError::InvalidSetup(
                "limb commitment degree must be 648".into(),
            ));
        }
        if n_a == 0 || m == 0 {
            return Err(AkitaError::InvalidSetup(
                "limb commitment rank and width must be nonzero".into(),
            ));
        }
        let domain = TrinomialLimbDomain::new(q0)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let expected = checked::product([n_a, m, DEGREE])
            .ok_or_else(|| AkitaError::InvalidSetup("limb matrix size overflow".into()))?;
        if coefficients.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: coefficients.len(),
            });
        }
        if coefficients.iter().any(|&coefficient| coefficient >= q0) {
            return Err(AkitaError::InvalidInput(
                "matrix coefficient must be below the limb prime".into(),
            ));
        }
        Self::prepare_elements(
            domain,
            n_a,
            m,
            coefficients.chunks_exact(DEGREE).map(|element| {
                let mut values = [0; DEGREE];
                for (out, &value) in values.iter_mut().zip(element) {
                    *out = value;
                }
                Ok(values)
            }),
        )
    }

    fn prepare_elements(
        domain: TrinomialLimbDomain,
        n_a: usize,
        m: usize,
        elements: impl ExactSizeIterator<Item = Result<[u32; DEGREE], AkitaError>>,
    ) -> Result<Self, AkitaError> {
        if n_a == 0 || m == 0 {
            return Err(AkitaError::InvalidSetup(
                "limb commitment rank and width must be nonzero".into(),
            ));
        }
        let q0 = domain.prime();
        let entries = checked::product([n_a, m])
            .ok_or_else(|| AkitaError::InvalidSetup("limb matrix size overflow".into()))?;
        if elements.len() != entries {
            return Err(AkitaError::InvalidSize {
                expected: entries,
                actual: elements.len(),
            });
        }
        let matrix_bytes = checked::product([entries, size_of::<TrinomialLimbSlots>()])
            .ok_or_else(|| AkitaError::InvalidSetup("limb matrix size overflow".into()))?;
        let mut matrix = Vec::new();
        matrix
            .try_reserve_exact(entries)
            .map_err(|_| AkitaError::InvalidSetup("limb matrix allocation failed".into()))?;
        let mut centered = [0i32; DEGREE];
        for element in elements {
            for (out, value) in centered.iter_mut().zip(element?) {
                *out = if value > q0 / 2 {
                    (i64::from(value) - i64::from(q0)) as i32
                } else {
                    value as i32
                };
            }
            let mut slots = domain.zero_slots();
            domain
                .forward_centered(&centered, &mut slots)
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
            matrix.push(slots);
        }
        domain
            .forward_interleaved_bits(&[0; 11], &mut domain.zero_slots())
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let prepared_bytes = checked::sum([matrix_bytes, domain.table_storage_bytes()])
            .ok_or_else(|| AkitaError::InvalidSetup("limb prepared size overflow".into()))?;
        let workspace_bytes = checked::product([n_a, size_of::<TrinomialLimbAccumulator>()])
            .and_then(|accumulators| checked::sum([accumulators, size_of::<Workspace>()]))
            .ok_or_else(|| AkitaError::InvalidSetup("limb workspace size overflow".into()))?;
        Ok(Self {
            domain,
            matrix,
            n_a,
            m,
            matrix_bytes,
            prepared_bytes,
            workspace_bytes,
            setup_digest: None,
        })
    }

    /// Matrix allocation bytes, including the four-byte prime tag per entry.
    ///
    /// At rank 3, width 4096: 31,850,496 slot bytes plus 49,152 tag bytes.
    pub fn matrix_bytes(&self) -> usize {
        self.matrix_bytes
    }

    /// Matrix and lookup-table payload bytes, excluding inline metadata and the
    /// domain's small radix-three split plan allocation.
    pub fn prepared_bytes(&self) -> usize {
        self.prepared_bytes
    }

    /// Retained workspace bytes per worker, including accumulator heap storage.
    ///
    /// Peak arithmetic scratch additionally includes an inverse copy (2592
    /// bytes), packed bits (88 bytes), and gather indices (81 bytes). These are
    /// bounded by degree, independent of width and column count. The result's
    /// `columns * rank * 648 * 4` bytes are allocated once, shared across workers.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// The reusable limb transform domain.
    pub fn domain(&self) -> &TrinomialLimbDomain {
        &self.domain
    }

    /// Transformed matrix entries in row-major order, with per-entry prime tags.
    pub fn matrix_ntt(&self) -> &[TrinomialLimbSlots] {
        &self.matrix
    }
}

/// Pack the unsigned magnitudes of the canonical four-word signed interleaving.
///
/// Bit `4*s+c` is coordinate `s` of `embed_source::<H>(words[c])`. The limb
/// domain applies its fixed sign `(-1)^s` through lookup tables. This agrees
/// exactly with [`crate::commit_kernel::pack_binary_element_i8`], without a
/// degree-sized array of signed bytes. Both supported source types embed into
/// the low 128 coordinates of F162, so the final 136 bits are zero.
pub fn pack_binary_element_bits<H: SwitchField>(
    words: &[H::Source],
) -> Result<[u64; 11], AkitaError> {
    source_element_rank::<DEGREE>(words.len())?;
    let mut packed = [0u64; 11];
    for (component, &word) in words.iter().enumerate() {
        let mut source: u128 = word.into();
        for destination in packed.iter_mut().take(8) {
            // Spread sixteen source bits into every fourth bit of one word.
            let mut bits = (source & 0xffff) as u64;
            bits = (bits | (bits << 24)) & 0x0000_00ff_0000_00ff;
            bits = (bits | (bits << 12)) & 0x000f_000f_000f_000f;
            bits = (bits | (bits << 6)) & 0x0303_0303_0303_0303;
            bits = (bits | (bits << 3)) & 0x1111_1111_1111_1111;
            *destination |= bits << component;
            source >>= 16;
        }
    }
    Ok(packed)
}

/// Commit exactly `columns * m * 4` source words using a prepared limb matrix.
///
/// Columns contain groups of four words per element, in precisely the layout
/// accepted by [`crate::commit_binary_clear_prepared`]. Return reduced `u32`
/// coefficients ordered by column, row, then coefficient. Each source element
/// is transformed once for all rows. Parallel workers reuse private workspaces
/// and write disjoint column slices, preserving output for any thread count.
pub fn commit_binary_clear_limb_prepared<H: SwitchField>(
    prepared: &PreparedLimbCommitMatrix,
    columns: usize,
    source: &[H::Source],
) -> Result<Vec<u32>, AkitaError>
where
    H::Source: Sync,
{
    let column_words = checked::product([prepared.m, K])
        .ok_or_else(|| AkitaError::InvalidInput("limb source size overflow".into()))?;
    let expected = checked::product([columns, column_words])
        .ok_or_else(|| AkitaError::InvalidInput("limb source size overflow".into()))?;
    check_source_len(expected, source.len())?;
    let column_coefficients = checked::product([prepared.n_a, DEGREE])
        .ok_or_else(|| AkitaError::InvalidInput("limb commitment size overflow".into()))?;
    let output_len = checked::product([columns, column_coefficients])
        .ok_or_else(|| AkitaError::InvalidInput("limb commitment size overflow".into()))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| AkitaError::InvalidInput("limb image allocation failed".into()))?;
    output.resize(output_len, 0);
    #[cfg(feature = "parallel")]
    output
        .par_chunks_mut(column_coefficients)
        .zip(source.par_chunks(column_words))
        .try_for_each_init(
            || Workspace::new(prepared),
            |workspace, (image, words)| {
                let workspace = workspace.as_mut().map_err(|error| error.clone())?;
                workspace.commit_column::<H>(prepared, words, image)
            },
        )?;
    #[cfg(not(feature = "parallel"))]
    {
        let mut workspace = Workspace::new(prepared)?;
        for (image, words) in output
            .chunks_mut(column_coefficients)
            .zip(source.chunks(column_words))
        {
            workspace.commit_column::<H>(prepared, words, image)?;
        }
    }
    Ok(output)
}

struct Workspace {
    transformed: TrinomialLimbSlots,
    accumulators: Vec<TrinomialLimbAccumulator>,
    coefficients: [i32; DEGREE],
}

impl Workspace {
    fn new(prepared: &PreparedLimbCommitMatrix) -> Result<Self, AkitaError> {
        let mut accumulators = Vec::new();
        accumulators
            .try_reserve_exact(prepared.n_a)
            .map_err(|_| AkitaError::InvalidInput("limb accumulator allocation failed".into()))?;
        accumulators.resize(
            prepared.n_a,
            TrinomialLimbAccumulator::new(&prepared.domain),
        );
        Ok(Self {
            transformed: prepared.domain.zero_slots(),
            accumulators,
            coefficients: [0; DEGREE],
        })
    }

    fn commit_column<H: SwitchField>(
        &mut self,
        prepared: &PreparedLimbCommitMatrix,
        words: &[H::Source],
        output: &mut [u32],
    ) -> Result<(), AkitaError> {
        for (element, chunk) in words.chunks_exact(K).enumerate() {
            let bits = pack_binary_element_bits::<H>(chunk)?;
            prepared
                .domain
                .forward_interleaved_bits(&bits, &mut self.transformed)
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            for (accumulator, row) in self
                .accumulators
                .iter_mut()
                .zip(prepared.matrix.chunks_exact(prepared.m))
            {
                let matrix_element = row.get(element).ok_or(AkitaError::InvalidProof)?;
                accumulator
                    .add_product(matrix_element, &self.transformed)
                    .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            }
        }
        for (accumulator, image) in self
            .accumulators
            .iter_mut()
            .zip(output.chunks_exact_mut(DEGREE))
        {
            // finish clears the accumulator for its next column.
            accumulator
                .finish(&mut self.transformed)
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            prepared
                .domain
                .inverse_centered(&self.transformed, &mut self.coefficients)
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
            for (out, &value) in image.iter_mut().zip(&self.coefficients) {
                *out = if value < 0 {
                    (i64::from(value) + i64::from(prepared.domain.prime())) as u32
                } else {
                    value as u32
                };
            }
        }
        Ok(())
    }
}
