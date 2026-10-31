//! Packed q24 matrix storage and delayed-reduction accumulation for the
//! prepared commitment kernel, behind the `q24-packed-commit` feature.
//!
//! When the commitment prime is the q24 prime `14_253_409`, preparation also
//! stores every transformed matrix entry as packed slots (three bytes per slot
//! instead of four). The commitment loop then packs each transformed source
//! element once per batch and accumulates with the q24 accumulator, which
//! reduces modulo q once every `2^15` products instead of checkpointing every
//! 128. Its finished slots are the same canonical centered residues the
//! unpacked accumulator returns, so the inverse transform and the commitment
//! images are identical. Other primes keep the unpacked path.
//!
//! The unpacked transformed matrix is kept for `matrix_ntt`, so a q24
//! matrix uses about 1.75 times the unpacked matrix memory.

use akita_algebra::ring::trinomial::{
    image::Q24, TrinomialQ24Accumulator as Q24Accumulator, TrinomialQ24PackedSlots as Q24Slots,
    Q24_REDUCTION_TERMS,
};

use super::*;

const Q: u128 = Q24 as u128;
const MAX_PRODUCT: u128 = (Q - 1) * (Q - 1);
// The accumulator's lane bound, restated where the commitment relies on it:
// (q - 1) + 2^15 (q - 1)^2 < 2^63 (see the q24 module of `akita_algebra`).
const _: () = assert!((Q - 1) + Q24_REDUCTION_TERMS as u128 * MAX_PRODUCT < 1 << 63);

fn limb(error: akita_algebra::TrinomialError) -> AkitaError {
    AkitaError::InvalidInput(error.to_string())
}

fn overflow() -> AkitaError {
    AkitaError::InvalidSetup("q24 packed size overflow".into())
}

/// Pack a transformed q24 matrix; `None` for every other prime.
///
/// Return the packed entries together with the prepared and workspace byte
/// counts increased by the packed matrix and the per-worker q24 workspace.
pub(super) fn prepare_packed(
    domain: &TrinomialLimbDomain,
    matrix: &[TrinomialLimbSlots],
    n_a: usize,
    prepared_bytes: usize,
    workspace_bytes: usize,
) -> Result<(Option<Vec<Q24Slots>>, usize, usize), AkitaError> {
    if domain.prime() != Q24 {
        return Ok((None, prepared_bytes, workspace_bytes));
    }
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(matrix.len())
        .map_err(|_| AkitaError::InvalidSetup("q24 packed matrix allocation failed".into()))?;
    for slots in matrix {
        let slots = Q24Slots::from_slots(slots)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        packed.push(slots);
    }
    let slot_bytes = size_of::<Q24Slots>();
    let accumulator_bytes = size_of::<Q24Accumulator>();
    let packed_bytes = checked::product([packed.len(), slot_bytes]);
    let packed_bytes = packed_bytes.ok_or_else(overflow)?;
    let accumulators = checked::product([n_a, BLOCK_COLUMNS, accumulator_bytes]);
    let accumulators = accumulators.ok_or_else(overflow)?;
    let sources = checked::product([BLOCK_COLUMNS, ELEMENT_BATCH, slot_bytes]);
    let sources = sources.ok_or_else(overflow)?;
    let extra = checked::sum([accumulators, sources]).ok_or_else(overflow)?;
    let prepared_total = checked::sum([prepared_bytes, packed_bytes]);
    let prepared_total = prepared_total.ok_or_else(overflow)?;
    let workspace_total = checked::sum([workspace_bytes, extra]);
    let workspace_total = workspace_total.ok_or_else(overflow)?;
    Ok((Some(packed), prepared_total, workspace_total))
}

/// Per-worker packed sources and q24 accumulators for one column block.
pub(super) struct Q24Workspace<const B: usize> {
    sources: [Vec<Q24Slots>; B],
    // Row-major, with one accumulator per column in the block.
    accumulators: Vec<Q24Accumulator>,
}

impl<const B: usize> Q24Workspace<B> {
    /// A workspace for a packed q24 matrix; `None` when the matrix is unpacked.
    pub(super) fn new(prepared: &PreparedLimbCommitMatrix) -> Result<Option<Self>, AkitaError> {
        if prepared.packed.is_none() {
            return Ok(None);
        }
        let count = checked::product([prepared.n_a, B]).ok_or_else(overflow)?;
        let accumulator = Q24Accumulator::new(&prepared.domain).map_err(limb)?;
        let mut accumulators = Vec::new();
        accumulators
            .try_reserve_exact(count)
            .map_err(|_| AkitaError::InvalidInput("q24 accumulator allocation failed".into()))?;
        accumulators.resize(count, accumulator);
        let zero = prepared.domain.zero_slots();
        let zero = Q24Slots::from_slots(&zero).map_err(limb)?;
        let mut sources = std::array::from_fn(|_| Vec::new());
        for column in &mut sources {
            column
                .try_reserve_exact(ELEMENT_BATCH)
                .map_err(|_| AkitaError::InvalidInput("q24 source allocation failed".into()))?;
            column.resize(ELEMENT_BATCH, zero.clone());
        }
        Ok(Some(Self {
            sources,
            accumulators,
        }))
    }

    /// Pack the batch's transformed sources and add every row/column product
    /// over matrix elements `start..end`.
    pub(super) fn accumulate(
        &mut self,
        prepared: &PreparedLimbCommitMatrix,
        start: usize,
        end: usize,
        transformed: &[Vec<TrinomialLimbSlots>; B],
        columns: usize,
    ) -> Result<(), AkitaError> {
        let count = end - start;
        let packed = prepared.packed.as_deref();
        let packed = packed.ok_or(AkitaError::InvalidProof)?;
        for (sources, slots) in self.sources.iter_mut().zip(transformed).take(columns) {
            let slots = slots.get(..count).ok_or(AkitaError::InvalidProof)?;
            for (source, slots) in sources.iter_mut().zip(slots) {
                *source = Q24Slots::from_slots(slots).map_err(limb)?;
            }
        }
        let rows = self.accumulators.chunks_exact_mut(B);
        for (accumulators, row) in rows.zip(packed.chunks_exact(prepared.m)) {
            let row = row.get(start..end).ok_or(AkitaError::InvalidProof)?;
            let columns = accumulators.iter_mut().zip(&self.sources).take(columns);
            for (accumulator, sources) in columns {
                let sources = sources.get(..count).ok_or(AkitaError::InvalidProof)?;
                accumulator.add_product_batch(row, sources).map_err(limb)?;
            }
        }
        Ok(())
    }

    /// Finish every accumulator of the block into reduced images ordered by
    /// column, row, then coefficient, exactly as the unpacked path writes them.
    pub(super) fn finish(
        &mut self,
        prepared: &PreparedLimbCommitMatrix,
        transformed: &mut [Vec<TrinomialLimbSlots>; B],
        coefficients: &mut [i32; DEGREE],
        output: &mut [u32],
    ) -> Result<(), AkitaError> {
        let prime = i64::from(prepared.domain.prime());
        let column_coefficients = checked::product([prepared.n_a, DEGREE]);
        let column_coefficients = column_coefficients.ok_or_else(overflow)?;
        for (column, image) in output.chunks_exact_mut(column_coefficients).enumerate() {
            let slots = transformed
                .get_mut(column)
                .and_then(|slots| slots.first_mut())
                .ok_or(AkitaError::InvalidProof)?;
            let rows = self.accumulators.chunks_exact_mut(B);
            for (accumulators, image) in rows.zip(image.chunks_exact_mut(DEGREE)) {
                let accumulator = accumulators
                    .get_mut(column)
                    .ok_or(AkitaError::InvalidProof)?;
                // finish clears the accumulator for the next block.
                accumulator.finish(slots).map_err(limb)?;
                prepared
                    .domain
                    .inverse_centered(slots, coefficients)
                    .map_err(limb)?;
                for (out, &value) in image.iter_mut().zip(coefficients.iter()) {
                    *out = if value < 0 {
                        (i64::from(value) + prime) as u32
                    } else {
                        value as u32
                    };
                }
            }
        }
        Ok(())
    }
}
