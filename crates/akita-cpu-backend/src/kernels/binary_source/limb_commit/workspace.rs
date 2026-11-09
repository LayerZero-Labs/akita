use super::*;

// Zen 4 end-to-end measurements favor eight elements across four columns.
// Dispatch further reduces the column block to keep workers busy.
pub(super) const BLOCK_COLUMNS: usize = 4;

pub(super) const ELEMENT_BATCH: usize = 8;

pub(super) struct Workspace<const B: usize> {
    transformed: [Vec<TrinomialLimbSlots>; B],
    // Row-major, with one accumulator per column in the block.
    accumulators: Vec<TrinomialLimbAccumulator>,
    coefficients: [i32; DEGREE],
}

impl<const B: usize> Workspace<B> {
    pub(super) fn new(prepared: &PreparedLimbCommitMatrix) -> Result<Self, AkitaError> {
        let count = checked::product([prepared.n_a, B])
            .ok_or_else(|| AkitaError::InvalidInput("limb workspace size overflow".into()))?;
        let mut accumulators = Vec::new();
        accumulators
            .try_reserve_exact(count)
            .map_err(|_| AkitaError::InvalidInput("limb accumulator allocation failed".into()))?;
        accumulators.resize(count, TrinomialLimbAccumulator::new(&prepared.domain));
        let mut transformed = std::array::from_fn(|_| Vec::new());
        for column in &mut transformed {
            column
                .try_reserve_exact(ELEMENT_BATCH)
                .map_err(|_| AkitaError::InvalidInput("limb transform allocation failed".into()))?;
            column.resize(ELEMENT_BATCH, prepared.domain.zero_slots());
        }
        Ok(Self {
            transformed,
            accumulators,
            coefficients: [0; DEGREE],
        })
    }

    pub(super) fn commit_block<H: SwitchField>(
        &mut self,
        prepared: &PreparedLimbCommitMatrix,
        words: &[H::Source],
        output: &mut [u32],
        column_words: usize,
        column_coefficients: usize,
    ) -> Result<(), AkitaError> {
        let columns = words.len() / column_words;
        for start in (0..prepared.m).step_by(ELEMENT_BATCH) {
            let end = start.saturating_add(ELEMENT_BATCH).min(prepared.m);
            let count = end - start;
            for (transformed, column) in self
                .transformed
                .iter_mut()
                .zip(words.chunks_exact(column_words))
            {
                let active = transformed
                    .get_mut(..count)
                    .ok_or(AkitaError::InvalidProof)?;
                let mut chunks: [&[H::Source]; ELEMENT_BATCH] = [&[]; ELEMENT_BATCH];
                let mut sources = [[0u128; K]; ELEMENT_BATCH];
                for ((chunk, source), element) in
                    chunks.iter_mut().zip(&mut sources).zip(start..end)
                {
                    *chunk = column
                        .get(element * K..(element + 1) * K)
                        .ok_or(AkitaError::InvalidProof)?;
                    for (source, &word) in source.iter_mut().zip(*chunk) {
                        *source = word.into();
                    }
                }
                let sources = sources.get(..count).ok_or(AkitaError::InvalidProof)?;
                if prepared
                    .domain
                    .try_forward_interleaved_words_batch(sources, active)
                    .map_err(|error| AkitaError::InvalidInput(error.to_string()))?
                {
                    continue;
                }
                for (chunk, slots) in chunks.into_iter().zip(active) {
                    let bits = pack_binary_element_bits::<H>(chunk)?;
                    prepared
                        .domain
                        .forward_interleaved_bits(&bits, slots)
                        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
                }
            }
            let source_columns = self.transformed.each_ref().map(Vec::as_slice);
            let sources = source_columns
                .get(..columns)
                .ok_or(AkitaError::InvalidProof)?;
            let mut accumulator_rows = self.accumulators.chunks_exact_mut(B);
            let mut matrix_rows = prepared.matrix.chunks_exact(prepared.m);
            while let (Some(first), Some(first_matrix)) =
                (accumulator_rows.next(), matrix_rows.next())
            {
                let first_matrix = first_matrix
                    .get(start..end)
                    .ok_or(AkitaError::InvalidProof)?;
                match (accumulator_rows.next(), matrix_rows.next()) {
                    (Some(second), Some(second_matrix)) => {
                        let second_matrix = second_matrix
                            .get(start..end)
                            .ok_or(AkitaError::InvalidProof)?;
                        TrinomialLimbAccumulator::add_product_tile(
                            [
                                first.get_mut(..columns).ok_or(AkitaError::InvalidProof)?,
                                second.get_mut(..columns).ok_or(AkitaError::InvalidProof)?,
                            ],
                            [first_matrix, second_matrix],
                            sources,
                        )
                        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
                    }
                    (None, None) => {
                        for (accumulator, source) in first.iter_mut().zip(sources) {
                            accumulator
                                .add_product_batch(
                                    first_matrix,
                                    source.get(..count).ok_or(AkitaError::InvalidProof)?,
                                )
                                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
                        }
                    }
                    _ => return Err(AkitaError::InvalidProof),
                }
            }
        }
        for (column, image) in output.chunks_exact_mut(column_coefficients).enumerate() {
            let transformed = self
                .transformed
                .get_mut(column)
                .and_then(|slots| slots.first_mut())
                .ok_or(AkitaError::InvalidProof)?;
            for (accumulators, image) in self
                .accumulators
                .chunks_exact_mut(B)
                .zip(image.chunks_exact_mut(DEGREE))
            {
                let accumulator = accumulators
                    .get_mut(column)
                    .ok_or(AkitaError::InvalidProof)?;
                // finish clears the accumulator for the next block.
                accumulator
                    .finish(transformed)
                    .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
                prepared
                    .domain
                    .inverse_centered(transformed, &mut self.coefficients)
                    .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
                for (out, &value) in image.iter_mut().zip(&self.coefficients) {
                    *out = if value < 0 {
                        (i64::from(value) + i64::from(prepared.domain.prime())) as u32
                    } else {
                        value as u32
                    };
                }
            }
        }
        Ok(())
    }
}

pub(super) fn commit_blocks<H: SwitchField, const B: usize>(
    prepared: &PreparedLimbCommitMatrix,
    source: &[H::Source],
    output: &mut [u32],
    column_words: usize,
    column_coefficients: usize,
) -> Result<(), AkitaError>
where
    H::Source: Sync,
{
    let block_words = checked::product([B, column_words])
        .ok_or_else(|| AkitaError::InvalidInput("limb source size overflow".into()))?;
    let block_coefficients = checked::product([B, column_coefficients])
        .ok_or_else(|| AkitaError::InvalidInput("limb commitment size overflow".into()))?;
    #[cfg(feature = "parallel")]
    output
        .par_chunks_mut(block_coefficients)
        .zip(source.par_chunks(block_words))
        .try_for_each_init(
            || Workspace::<B>::new(prepared),
            |workspace, (image, words)| {
                let workspace = workspace.as_mut().map_err(|error| error.clone())?;
                workspace.commit_block::<H>(
                    prepared,
                    words,
                    image,
                    column_words,
                    column_coefficients,
                )
            },
        )?;
    #[cfg(not(feature = "parallel"))]
    {
        let mut workspace = Workspace::<B>::new(prepared)?;
        for (image, words) in output
            .chunks_mut(block_coefficients)
            .zip(source.chunks(block_words))
        {
            workspace.commit_block::<H>(
                prepared,
                words,
                image,
                column_words,
                column_coefficients,
            )?;
        }
    }
    Ok(())
}
