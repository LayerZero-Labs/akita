use akita_error::AkitaError;
#[cfg(test)]
use akita_types::DigitBlocks;

/// Validate one committed group's per-polynomial plane counts, then stream its
/// canonical B slices through one reusable physical-width buffer.
pub fn for_each_outer_slice_input<'a, const D_B: usize>(
    polynomial_planes: impl IntoIterator<Item = &'a [[i8; D_B]]>,
    geometry: &akita_types::CommitmentSliceGeometry,
    mut consume: impl FnMut(&[[i8; D_B]]) -> Result<(), AkitaError>,
) -> Result<(), AkitaError> {
    let per_block = geometry.ring_elements_per_block_per_polynomial();
    let num_live_blocks = geometry
        .block_ranges()
        .last()
        .map(|range| range.end)
        .ok_or_else(|| AkitaError::InvalidSetup("B commitment has no slices".into()))?;
    let expected_planes = num_live_blocks
        .checked_mul(per_block)
        .ok_or_else(|| AkitaError::InvalidSetup("B slice plane count overflow".into()))?;
    let polynomial_planes = polynomial_planes.into_iter().collect::<Vec<_>>();
    if polynomial_planes.is_empty()
        || polynomial_planes
            .iter()
            .any(|planes| planes.len() != expected_planes)
    {
        return Err(AkitaError::InvalidSetup(
            "B slice input does not match the frozen block geometry".into(),
        ));
    }

    let max_blocks = geometry.max_blocks_per_slice();
    let expected_width = geometry.physical_input_width();
    let mut input = Vec::with_capacity(expected_width);
    for range in geometry.block_ranges() {
        input.clear();
        let plane_start = range
            .start
            .checked_mul(per_block)
            .ok_or_else(|| AkitaError::InvalidSetup("B slice input offset overflow".into()))?;
        let plane_end = range
            .end
            .checked_mul(per_block)
            .ok_or_else(|| AkitaError::InvalidSetup("B slice input offset overflow".into()))?;
        for planes in &polynomial_planes {
            input.extend_from_slice(planes.get(plane_start..plane_end).ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "B slice input does not match the frozen block geometry".into(),
                )
            })?);
            let padding = (max_blocks - range.len())
                .checked_mul(per_block)
                .ok_or_else(|| AkitaError::InvalidSetup("B slice padding overflow".into()))?;
            let padded_len = input
                .len()
                .checked_add(padding)
                .filter(|len| *len <= expected_width)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "B slice input width does not match the physical matrix".into(),
                    )
                })?;
            input.resize(padded_len, [0i8; D_B]);
        }
        if input.len() != expected_width {
            return Err(AkitaError::InvalidSetup(
                "B slice input width does not match the physical matrix".into(),
            ));
        }
        consume(&input)?;
    }
    Ok(())
}

#[cfg(test)]
fn validate_outer_slice_digits<'a, const D_B: usize>(
    polynomial_digits: impl IntoIterator<Item = &'a DigitBlocks>,
    geometry: &akita_types::CommitmentSliceGeometry,
) -> Result<Vec<&'a [[i8; D_B]]>, AkitaError> {
    let per_block = geometry.ring_elements_per_block_per_polynomial();
    let num_live_blocks = geometry
        .block_ranges()
        .last()
        .map(|range| range.end)
        .ok_or_else(|| AkitaError::InvalidSetup("B commitment has no slices".into()))?;
    polynomial_digits
        .into_iter()
        .map(|digits| {
            if digits.block_count() != num_live_blocks
                || digits.block_sizes().iter().any(|&size| size != per_block)
            {
                return Err(AkitaError::InvalidSetup(
                    "B slice input does not match the frozen block geometry".into(),
                ));
            }
            digits.typed_planes::<D_B>()
        })
        .collect()
}

#[cfg(test)]
fn outer_slice_inputs<const D_B: usize>(
    polynomial_digits: &[&DigitBlocks],
    geometry: &akita_types::CommitmentSliceGeometry,
) -> Result<Vec<Vec<[i8; D_B]>>, AkitaError> {
    let mut inputs = Vec::with_capacity(geometry.slice_count().get());
    let polynomial_planes =
        validate_outer_slice_digits::<D_B>(polynomial_digits.iter().copied(), geometry)?;
    for_each_outer_slice_input::<D_B>(polynomial_planes, geometry, |input| {
        inputs.push(input.to_vec());
        Ok(())
    })?;
    Ok(inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::CommitmentSliceGeometry;

    #[test]
    fn outer_slice_inputs_are_polynomial_major_and_zero_padded() {
        let first =
            DigitBlocks::new(vec![10, 11, 12, 13, 14], vec![1; 5], 1).expect("first digit blocks");
        let second =
            DigitBlocks::new(vec![20, 21, 22, 23, 24], vec![1; 5], 1).expect("second digit blocks");
        let geometry = CommitmentSliceGeometry::try_new(
            akita_types::CommitmentSliceCount::TWO,
            5,
            2,
            1,
            1,
            1,
            1,
        )
        .expect("slice geometry");

        let inputs = outer_slice_inputs::<1>(&[&first, &second], &geometry).expect("slice inputs");
        assert_eq!(
            inputs,
            vec![
                vec![[10], [11], [0], [20], [21], [0]],
                vec![[12], [13], [14], [22], [23], [24]],
            ]
        );
    }

    #[test]
    fn outer_slice_stream_reuses_one_physical_width_buffer() {
        let digits = DigitBlocks::new((0..13).collect(), vec![1; 13], 1).expect("digit blocks");
        let geometry = CommitmentSliceGeometry::try_new(
            akita_types::CommitmentSliceCount::FOUR,
            13,
            1,
            1,
            1,
            1,
            1,
        )
        .expect("slice geometry");
        let planes = digits.typed_planes::<1>().expect("typed planes");
        let mut addresses = Vec::new();

        for_each_outer_slice_input::<1>(std::iter::once(planes), &geometry, |input| {
            assert_eq!(input.len(), geometry.physical_input_width());
            addresses.push(input.as_ptr());
            Ok(())
        })
        .expect("stream slices");

        assert_eq!(addresses.len(), 4);
        assert!(addresses.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn sliced_b_images_match_independent_block_diagonal_oracle_for_all_counts() {
        const BLOCKS: usize = 9;
        const POLYS: usize = 2;
        const PER_BLOCK: usize = 2;
        const ROWS: usize = 3;

        let polynomial_digits = (0..POLYS)
            .map(|polynomial| {
                let digits = (0..BLOCKS * PER_BLOCK)
                    .map(|index| (1 + polynomial * 31 + index) as i8)
                    .collect::<Vec<_>>();
                DigitBlocks::new(digits, vec![PER_BLOCK; BLOCKS], 1).unwrap()
            })
            .collect::<Vec<_>>();

        for slice_count in akita_types::CommitmentSliceCount::ALL {
            let geometry =
                CommitmentSliceGeometry::try_new(slice_count, BLOCKS, POLYS, PER_BLOCK, 1, 1, 1)
                    .unwrap();
            let matrix = (0..ROWS)
                .map(|row| {
                    (0..geometry.physical_input_width())
                        .map(|column| 1 + (row as i64 + 1) * 17 + column as i64)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let production_inputs =
                outer_slice_inputs::<1>(&polynomial_digits.iter().collect::<Vec<_>>(), &geometry)
                    .unwrap();
            let production_image = production_inputs
                .iter()
                .flat_map(|input| {
                    let matrix = &matrix;
                    matrix.iter().map(move |row| {
                        row.iter()
                            .zip(input)
                            .map(|(&matrix_entry, digit)| matrix_entry * i64::from(digit[0]))
                            .sum::<i64>()
                    })
                })
                .collect::<Vec<_>>();

            let slices = slice_count.get();
            let max_blocks = BLOCKS.div_ceil(slices);
            let mut oracle_image = Vec::with_capacity(slices * ROWS);
            for slice_index in 0..slices {
                let start = BLOCKS * slice_index / slices;
                let end = BLOCKS * (slice_index + 1) / slices;
                for row in &matrix {
                    let mut image = 0i64;
                    for polynomial in 0..POLYS {
                        for global_block in start..end {
                            let local_block = global_block - start;
                            for offset in 0..PER_BLOCK {
                                let physical_column =
                                    (polynomial * max_blocks + local_block) * PER_BLOCK + offset;
                                let digit = 1 + polynomial * 31 + global_block * PER_BLOCK + offset;
                                image += row[physical_column] * digit as i64;
                            }
                        }
                    }
                    oracle_image.push(image);
                }
            }
            assert_eq!(production_image, oracle_image);

            let production_compressed = production_image
                .iter()
                .enumerate()
                .map(|(index, &value)| (index as i64 + 3) * value)
                .sum::<i64>();
            let oracle_compressed = oracle_image
                .iter()
                .enumerate()
                .map(|(index, &value)| (index as i64 + 3) * value)
                .sum::<i64>();
            assert_eq!(production_compressed, oracle_compressed);
        }
    }
}
