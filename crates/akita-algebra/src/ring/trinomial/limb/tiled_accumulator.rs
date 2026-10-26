//! Validated two-row product tiles with the existing checkpoint schedule.

use super::*;

impl TrinomialLimbAccumulator {
    /// Add two matrix rows against all source columns in one tile.
    ///
    /// Rows contain one accumulator per source column. Matrix row lengths
    /// must agree; source buffers may be longer, but must contain every consumed
    /// element. All consumed slot tags and accumulator primes must agree.
    /// Invalid shapes or tags leave every accumulator unchanged.
    pub fn add_product_tile(
        rows: [&mut [Self]; 2],
        matrix: [&[TrinomialLimbSlots]; 2],
        sources: &[&[TrinomialLimbSlots]],
    ) -> Result<(), TrinomialError> {
        let [row0, row1] = rows;
        let columns = sources.len();
        let terms = matrix[0].len();
        if row0.len() != columns
            || row1.len() != columns
            || matrix[1].len() != terms
            || sources.iter().any(|column| column.len() < terms)
        {
            return Err(TrinomialError::LimbInput {
                reason: "product tile shapes do not match",
            });
        }
        let Some(first) = row0.first() else {
            return Ok(());
        };
        let prime = first.arithmetic.prime;
        if row0.iter().chain(row1.iter()).any(|accumulator| {
            accumulator.arithmetic.prime != prime || accumulator.terms >= Self::REDUCTION_TERMS
        }) || matrix
            .iter()
            .flat_map(|row| row.iter())
            .chain(sources.iter().flat_map(|column| column.iter().take(terms)))
            .any(|slot| slot.prime != prime)
        {
            return Err(TrinomialError::LimbInput {
                reason: "product tile prime or checkpoint phase does not match",
            });
        }

        #[cfg(target_arch = "x86_64")]
        {
            let backend = first.backend;
            if matches!(backend, Backend::Avx2 | Backend::Avx512VbmiBatch)
                && row0
                    .iter()
                    .chain(row1.iter())
                    .all(|accumulator| accumulator.backend == backend)
            {
                let mut offset = 0;
                while offset < terms {
                    // Different incoming phases are valid. Stop every raw
                    // tile at the earliest checkpoint of any participant.
                    let count = row0
                        .iter()
                        .chain(row1.iter())
                        .map(|accumulator| usize::from(Self::REDUCTION_TERMS - accumulator.terms))
                        .fold(terms - offset, usize::min);
                    let end = offset + count;
                    let matrix = matrix.map(|row| &row[offset..end]);
                    let (pairs0, _) = row0.as_chunks_mut::<2>();
                    let (pairs1, _) = row1.as_chunks_mut::<2>();
                    for (column, ([a, b], [c, d])) in pairs0.iter_mut().zip(pairs1).enumerate() {
                        let column = 2 * column;
                        // SAFETY: All x86 variants detect AVX2. Validation
                        // bounds every slice, and count preserves the existing
                        // checkpoint bound independently for each accumulator.
                        unsafe {
                            tile::accumulate_2x2(
                                [&mut a.sums, &mut b.sums, &mut c.sums, &mut d.sums],
                                matrix,
                                [
                                    &sources[column][offset..end],
                                    &sources[column + 1][offset..end],
                                ],
                            );
                        }
                        for accumulator in [a, b, c, d] {
                            accumulator.advance(count as u16);
                        }
                    }
                    if !columns.is_multiple_of(2) {
                        // The batch fallback advances itself; never advance
                        // this final column a second time after its raw chunk.
                        let column = columns - 1;
                        let source = &sources[column][offset..end];
                        row0[column].add_product_batch(matrix[0], source)?;
                        row1[column].add_product_batch(matrix[1], source)?;
                    }
                    offset = end;
                }
                return Ok(());
            }
        }
        // Scalar, NEON, or mixed backend rows retain their existing batch
        // implementation and its own checkpoint and phase accounting.
        for ((a, b), source) in row0.iter_mut().zip(row1).zip(sources) {
            a.add_product_batch(matrix[0], &source[..terms])?;
            b.add_product_batch(matrix[1], &source[..terms])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(prime: u32, phase: usize) -> TrinomialLimbSlots {
        let bound = (prime - 1) as i32;
        TrinomialLimbSlots {
            prime,
            values: std::array::from_fn(|i| match (phase + i) % 4 {
                0 => bound,
                1 => -bound,
                2 => 0,
                _ => 1,
            }),
        }
    }

    fn assert_accumulators(
        actual: &[TrinomialLimbAccumulator],
        expected: &[TrinomialLimbAccumulator],
    ) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.sums, expected.sums);
            assert_eq!(actual.terms, expected.terms);
            assert_eq!(actual.arithmetic.prime, expected.arithmetic.prime);
            assert_eq!(actual.backend, expected.backend);
        }
    }

    #[test]
    fn limb_product_tiles_preserve_raw_sums_phases_tails_and_finish_reset() {
        for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
            for backend in available_backends() {
                let totals = [127, 128, 129, 255, 256];
                let cases = (1..=8)
                    .map(|batch| (batch, 1 + (batch - 1) % 4, totals[(batch - 1) % 5]))
                    .chain(totals.map(|total| (8, 4, total)));
                for (batch, columns, total) in cases {
                    let mut domain = TrinomialLimbDomain::new(prime).unwrap();
                    domain.backend = backend;
                    let mut scalar = domain.clone();
                    scalar.backend = Backend::Scalar;
                    let mut actual: [Vec<_>; 2] = std::array::from_fn(|_| {
                        vec![TrinomialLimbAccumulator::new(&domain); columns]
                    });
                    let mut expected: [Vec<_>; 2] = std::array::from_fn(|_| {
                        vec![TrinomialLimbAccumulator::new(&scalar); columns]
                    });
                    let matrix: [Vec<_>; 2] = std::array::from_fn(|row| {
                        (0..total).map(|term| slot(prime, row + term)).collect()
                    });
                    let sources: [Vec<_>; 4] = std::array::from_fn(|column| {
                        (0..total + 8)
                            .map(|term| slot(prime, column + 2 * term))
                            .collect()
                    });
                    // Deliberately give participants different valid phases.
                    for row in 0..2 {
                        for column in 0..columns {
                            for _ in 0..(row + column) % 3 {
                                actual[row][column]
                                    .add_product(&matrix[row][0], &sources[column][0])
                                    .unwrap();
                                expected[row][column]
                                    .add_product(&matrix[row][0], &sources[column][0])
                                    .unwrap();
                            }
                        }
                    }
                    for offset in (0..total).step_by(batch) {
                        let end = (offset + batch).min(total);
                        let source_slices: Vec<_> = sources
                            .iter()
                            .take(columns)
                            .map(|source| &source[offset..(offset + 8).min(source.len())])
                            .collect();
                        let [row0, row1] = &mut actual;
                        TrinomialLimbAccumulator::add_product_tile(
                            [row0, row1],
                            [&matrix[0][offset..end], &matrix[1][offset..end]],
                            &source_slices,
                        )
                        .unwrap();
                        for row in 0..2 {
                            for column in 0..columns {
                                expected[row][column]
                                    .add_product_batch(
                                        &matrix[row][offset..end],
                                        &sources[column][offset..end],
                                    )
                                    .unwrap();
                                assert_eq!(actual[row][column].sums, expected[row][column].sums);
                                assert_eq!(actual[row][column].terms, expected[row][column].terms);
                            }
                        }
                    }
                    for row in 0..2 {
                        for column in 0..columns {
                            let mut a = domain.zero_slots();
                            let mut b = scalar.zero_slots();
                            actual[row][column].finish(&mut a).unwrap();
                            expected[row][column].finish(&mut b).unwrap();
                            assert_eq!(a.values, b.values);
                            assert_eq!(actual[row][column].sums, [0; DEGREE]);
                            assert_eq!(actual[row][column].terms, 0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn limb_product_tiles_reject_invalid_shapes_and_consumed_tags_atomically() {
        for prime in TrinomialLimbDomain::ADMITTED_PRIMES {
            for backend in available_backends() {
                let mut domain = TrinomialLimbDomain::new(prime).unwrap();
                domain.backend = backend;
                let other = TrinomialLimbDomain::ADMITTED_PRIMES
                    .into_iter()
                    .find(|&p| p != prime)
                    .unwrap();
                for invalid in 0..6 {
                    let mut rows = [
                        vec![TrinomialLimbAccumulator::new(&domain); 2],
                        vec![TrinomialLimbAccumulator::new(&domain); 2],
                    ];
                    for row in &mut rows {
                        for accumulator in row {
                            accumulator
                                .add_product(&slot(prime, 0), &slot(prime, 1))
                                .unwrap();
                        }
                    }
                    let mut matrix = [vec![slot(prime, 0); 3], vec![slot(prime, 1); 3]];
                    let mut sources = [vec![slot(prime, 2); 8], vec![slot(prime, 3); 8]];
                    match invalid {
                        0 => {
                            rows[1].pop();
                        }
                        1 => {
                            matrix[1].pop();
                        }
                        2 => {
                            sources[1].truncate(2);
                        }
                        3 => {
                            matrix[1][2].prime = other;
                        }
                        4 => {
                            sources[1][2].prime = other;
                        }
                        _ => {
                            rows[1][1].arithmetic.prime = other;
                        }
                    }
                    let before = rows.clone();
                    let [row0, row1] = &mut rows;
                    assert!(TrinomialLimbAccumulator::add_product_tile(
                        [row0, row1],
                        [matrix[0].as_slice(), matrix[1].as_slice()],
                        &[sources[0].as_slice(), sources[1].as_slice()],
                    )
                    .is_err());
                    assert_accumulators(&rows[0], &before[0]);
                    assert_accumulators(&rows[1], &before[1]);
                }
            }
        }
    }
}
