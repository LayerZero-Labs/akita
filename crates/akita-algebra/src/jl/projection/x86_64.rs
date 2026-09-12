//! x86-64 paired-Rademacher lookup kernels.

use super::{
    build_lookup_table_i32, build_lookup_table_i64, SmallLookupInput, TernaryProjectionMatrix,
    WideLookupInput, LOOKUP_GROUPS_PER_TILE,
};
use std::arch::x86_64::*;

/// Accumulate doubled projections for inputs whose four-value lookup tables fit in `i32`.
///
/// # Safety
///
/// The caller must ensure that AVX2 is available, `input.len() == matrix.cols`,
/// `output.len() == matrix.rows`, and `output` is zeroed before the first tile.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn project_lookup_i32_avx2<T: SmallLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
) {
    let shape = matrix.shape();
    let mut tables = [[0i32; 16]; LOOKUP_GROUPS_PER_TILE];
    for group_base in (0..shape.col_groups()).step_by(LOOKUP_GROUPS_PER_TILE) {
        let groups = (shape.col_groups() - group_base).min(LOOKUP_GROUPS_PER_TILE);
        for (local_group, table) in tables.iter_mut().take(groups).enumerate() {
            *table = build_lookup_table_i32(input, group_base + local_group);
        }

        let mut row = 0;
        while row + 8 <= shape.rows() {
            let mut accumulator = _mm256_setzero_si256();
            let row_pair = row >> 1;
            for (local_group, table) in tables.iter().take(groups).enumerate() {
                let (first, second) = matrix.sign_groups_unchecked(group_base + local_group);
                accumulator = _mm256_add_epi32(
                    accumulator,
                    lookup_i32x8(table, selector_indices_8_i32(first.as_ptr().add(row_pair))),
                );
                accumulator = _mm256_add_epi32(
                    accumulator,
                    lookup_i32x8(table, selector_indices_8_i32(second.as_ptr().add(row_pair))),
                );
            }
            add_i32x8_to_i64_output(accumulator, output.as_mut_ptr().add(row));
            row += 8;
        }
        accumulate_scalar_tail_i32(matrix, group_base, groups, &tables, row, output);
    }
}

/// Accumulate doubled projections using `i64` lookup tables.
///
/// # Safety
///
/// The caller must ensure that AVX2 is available, the input and output lengths
/// exactly match the matrix, `output` is zeroed before the first tile, and all
/// doubled results fit in `i64`.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn project_lookup_i64_avx2<T: WideLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
) {
    let shape = matrix.shape();
    let mut tables = [[0i64; 16]; LOOKUP_GROUPS_PER_TILE];
    for group_base in (0..shape.col_groups()).step_by(LOOKUP_GROUPS_PER_TILE) {
        let groups = (shape.col_groups() - group_base).min(LOOKUP_GROUPS_PER_TILE);
        for (local_group, table) in tables.iter_mut().take(groups).enumerate() {
            *table = build_lookup_table_i64(input, group_base + local_group);
        }

        let mut row = 0;
        while row + 4 <= shape.rows() {
            let mut accumulator = _mm256_loadu_si256(output.as_ptr().add(row).cast());
            let row_pair = row >> 1;
            for (local_group, table) in tables.iter().take(groups).enumerate() {
                let (first, second) = matrix.sign_groups_unchecked(group_base + local_group);
                let first_indices = selector_indices_4_i64(first.as_ptr().add(row_pair));
                let second_indices = selector_indices_4_i64(second.as_ptr().add(row_pair));
                accumulator = _mm256_add_epi64(
                    accumulator,
                    _mm256_i64gather_epi64::<8>(table.as_ptr(), first_indices),
                );
                accumulator = _mm256_add_epi64(
                    accumulator,
                    _mm256_i64gather_epi64::<8>(table.as_ptr(), second_indices),
                );
            }
            _mm256_storeu_si256(output.as_mut_ptr().add(row).cast(), accumulator);
            row += 4;
        }
        accumulate_scalar_tail_i64(matrix, group_base, groups, &tables, row, output);
    }
}

/// Accumulate doubled projections for inputs whose four-value lookup tables fit in `i32`.
///
/// # Safety
///
/// The caller must ensure that AVX-512F is available, the input and output
/// lengths exactly match the matrix, and `output` is zeroed before the first
/// tile.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn project_lookup_i32_avx512<T: SmallLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
) {
    let shape = matrix.shape();
    let mut tables = [[0i32; 16]; LOOKUP_GROUPS_PER_TILE];
    for group_base in (0..shape.col_groups()).step_by(LOOKUP_GROUPS_PER_TILE) {
        let groups = (shape.col_groups() - group_base).min(LOOKUP_GROUPS_PER_TILE);
        for (local_group, table) in tables.iter_mut().take(groups).enumerate() {
            *table = build_lookup_table_i32(input, group_base + local_group);
        }

        let mut row = 0;
        while row + 16 <= shape.rows() {
            let mut accumulator = _mm512_setzero_si512();
            let row_pair = row >> 1;
            for (local_group, table) in tables.iter().take(groups).enumerate() {
                let table = _mm512_loadu_si512(table.as_ptr().cast());
                let (first, second) = matrix.sign_groups_unchecked(group_base + local_group);
                let first_indices = selector_indices_16_i32(first.as_ptr().add(row_pair));
                let second_indices = selector_indices_16_i32(second.as_ptr().add(row_pair));
                accumulator =
                    _mm512_add_epi32(accumulator, _mm512_permutexvar_epi32(first_indices, table));
                accumulator =
                    _mm512_add_epi32(accumulator, _mm512_permutexvar_epi32(second_indices, table));
            }
            add_i32x16_to_i64_output(accumulator, output.as_mut_ptr().add(row));
            row += 16;
        }
        accumulate_scalar_tail_i32(matrix, group_base, groups, &tables, row, output);
    }
}

/// Accumulate doubled projections using `i64` lookup tables.
///
/// # Safety
///
/// The caller must ensure that AVX-512F is available, the input and output
/// lengths exactly match the matrix, `output` is zeroed before the first tile,
/// and all doubled results fit in `i64`.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn project_lookup_i64_avx512<T: WideLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
) {
    let shape = matrix.shape();
    let mut tables = [[0i64; 16]; LOOKUP_GROUPS_PER_TILE];
    for group_base in (0..shape.col_groups()).step_by(LOOKUP_GROUPS_PER_TILE) {
        let groups = (shape.col_groups() - group_base).min(LOOKUP_GROUPS_PER_TILE);
        for (local_group, table) in tables.iter_mut().take(groups).enumerate() {
            *table = build_lookup_table_i64(input, group_base + local_group);
        }

        let mut row = 0;
        while row + 8 <= shape.rows() {
            let mut accumulator = _mm512_loadu_si512(output.as_ptr().add(row).cast());
            let row_pair = row >> 1;
            for (local_group, table) in tables.iter().take(groups).enumerate() {
                let low = _mm512_loadu_si512(table.as_ptr().cast());
                let high = _mm512_loadu_si512(table.as_ptr().add(8).cast());
                let (first, second) = matrix.sign_groups_unchecked(group_base + local_group);
                let first_indices = selector_indices_8_i64(first.as_ptr().add(row_pair));
                let second_indices = selector_indices_8_i64(second.as_ptr().add(row_pair));
                accumulator = _mm512_add_epi64(
                    accumulator,
                    _mm512_permutex2var_epi64(low, first_indices, high),
                );
                accumulator = _mm512_add_epi64(
                    accumulator,
                    _mm512_permutex2var_epi64(low, second_indices, high),
                );
            }
            _mm512_storeu_si512(output.as_mut_ptr().add(row).cast(), accumulator);
            row += 8;
        }
        accumulate_scalar_tail_i64(matrix, group_base, groups, &tables, row, output);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn lookup_i32x8(table: &[i32; 16], indices: __m256i) -> __m256i {
    let base_indices = _mm256_and_si256(indices, _mm256_set1_epi32(7));
    let low = _mm256_loadu_si256(table.as_ptr().cast());
    let high = _mm256_loadu_si256(table.as_ptr().add(8).cast());
    let use_high = _mm256_cmpgt_epi32(indices, _mm256_set1_epi32(7));
    _mm256_blendv_epi8(
        _mm256_permutevar8x32_epi32(low, base_indices),
        _mm256_permutevar8x32_epi32(high, base_indices),
        use_high,
    )
}

#[target_feature(enable = "avx2")]
unsafe fn selector_indices_8_i32(packed: *const u8) -> __m256i {
    let bytes = _mm_cvtsi32_si128(std::ptr::read_unaligned(packed.cast::<i32>()));
    let nibble_mask = _mm_set1_epi8(0x0f);
    let low = _mm_and_si128(bytes, nibble_mask);
    let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), nibble_mask);
    _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(low, high))
}

#[target_feature(enable = "avx2")]
unsafe fn selector_indices_4_i64(packed: *const u8) -> __m256i {
    let bytes = _mm_cvtsi32_si128(i32::from(std::ptr::read_unaligned(packed.cast::<u16>())));
    let nibble_mask = _mm_set1_epi8(0x0f);
    let low = _mm_and_si128(bytes, nibble_mask);
    let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), nibble_mask);
    _mm256_cvtepu8_epi64(_mm_unpacklo_epi8(low, high))
}

#[target_feature(enable = "avx512f")]
unsafe fn selector_indices_16_i32(packed: *const u8) -> __m512i {
    let bytes = _mm_loadl_epi64(packed.cast());
    let nibble_mask = _mm_set1_epi8(0x0f);
    let low = _mm_and_si128(bytes, nibble_mask);
    let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), nibble_mask);
    _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(low, high))
}

#[target_feature(enable = "avx512f")]
unsafe fn selector_indices_8_i64(packed: *const u8) -> __m512i {
    let bytes = _mm_cvtsi32_si128(std::ptr::read_unaligned(packed.cast::<i32>()));
    let nibble_mask = _mm_set1_epi8(0x0f);
    let low = _mm_and_si128(bytes, nibble_mask);
    let high = _mm_and_si128(_mm_srli_epi16::<4>(bytes), nibble_mask);
    _mm512_cvtepu8_epi64(_mm_unpacklo_epi8(low, high))
}

#[target_feature(enable = "avx2")]
unsafe fn add_i32x8_to_i64_output(values: __m256i, output: *mut i64) {
    let low = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(values));
    let high = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(values));
    let low_output = _mm256_loadu_si256(output.cast());
    let high_output = _mm256_loadu_si256(output.add(4).cast());
    _mm256_storeu_si256(output.cast(), _mm256_add_epi64(low_output, low));
    _mm256_storeu_si256(output.add(4).cast(), _mm256_add_epi64(high_output, high));
}

#[target_feature(enable = "avx512f")]
unsafe fn add_i32x16_to_i64_output(values: __m512i, output: *mut i64) {
    let low = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(values));
    let high = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(values));
    let low_output = _mm512_loadu_si512(output.cast());
    let high_output = _mm512_loadu_si512(output.add(8).cast());
    _mm512_storeu_si512(output.cast(), _mm512_add_epi64(low_output, low));
    _mm512_storeu_si512(output.add(8).cast(), _mm512_add_epi64(high_output, high));
}

fn accumulate_scalar_tail_i32(
    matrix: &TernaryProjectionMatrix,
    group_base: usize,
    groups: usize,
    tables: &[[i32; 16]; LOOKUP_GROUPS_PER_TILE],
    first_row: usize,
    output: &mut [i64],
) {
    for (row, value) in output.iter_mut().enumerate().skip(first_row) {
        let row_pair = row >> 1;
        let shift = (row & 1) << 2;
        let mut sum = *value;
        for (local_group, table) in tables.iter().take(groups).enumerate() {
            let (first, second) = matrix.sign_groups_unchecked(group_base + local_group);
            let first_selector = usize::from((first[row_pair] >> shift) & 0x0f);
            let second_selector = usize::from((second[row_pair] >> shift) & 0x0f);
            sum += i64::from(table[first_selector]) + i64::from(table[second_selector]);
        }
        *value = sum;
    }
}

fn accumulate_scalar_tail_i64(
    matrix: &TernaryProjectionMatrix,
    group_base: usize,
    groups: usize,
    tables: &[[i64; 16]; LOOKUP_GROUPS_PER_TILE],
    first_row: usize,
    output: &mut [i64],
) {
    for (row, value) in output.iter_mut().enumerate().skip(first_row) {
        let row_pair = row >> 1;
        let shift = (row & 1) << 2;
        let mut sum = *value;
        for (local_group, table) in tables.iter().take(groups).enumerate() {
            let (first, second) = matrix.sign_groups_unchecked(group_base + local_group);
            let first_selector = usize::from((first[row_pair] >> shift) & 0x0f);
            let second_selector = usize::from((second[row_pair] >> shift) & 0x0f);
            sum += table[first_selector] + table[second_selector];
        }
        *value = sum;
    }
}

#[cfg(test)]
#[target_feature(enable = "avx512f")]
unsafe fn project_lookup_i32_avx512_baseline<T: SmallLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
) {
    let shape = matrix.shape();
    let mut tables = [[0i32; 16]; LOOKUP_GROUPS_PER_TILE];
    for group_base in (0..shape.col_groups()).step_by(LOOKUP_GROUPS_PER_TILE) {
        let groups = (shape.col_groups() - group_base).min(LOOKUP_GROUPS_PER_TILE);
        for (local_group, table) in tables.iter_mut().take(groups).enumerate() {
            *table = build_lookup_table_i32(input, group_base + local_group);
        }
        let mut row = 0;
        while row + 16 <= shape.rows() {
            let mut accumulator = _mm512_setzero_si512();
            let row_pair = row >> 1;
            for (local_group, table) in tables.iter().take(groups).enumerate() {
                let table = _mm512_loadu_si512(table.as_ptr().cast());
                let (first, second) = matrix.sign_groups_unchecked(group_base + local_group);
                accumulator = _mm512_add_epi32(
                    accumulator,
                    _mm512_permutexvar_epi32(
                        selector_indices_16_i32_baseline(first.as_ptr().add(row_pair)),
                        table,
                    ),
                );
                accumulator = _mm512_add_epi32(
                    accumulator,
                    _mm512_permutexvar_epi32(
                        selector_indices_16_i32_baseline(second.as_ptr().add(row_pair)),
                        table,
                    ),
                );
            }
            add_i32x16_to_i64_output(accumulator, output.as_mut_ptr().add(row));
            row += 16;
        }
        accumulate_scalar_tail_i32(matrix, group_base, groups, &tables, row, output);
    }
}

#[cfg(test)]
#[target_feature(enable = "avx512f")]
unsafe fn project_lookup_i64_avx512_baseline<T: WideLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
) {
    let shape = matrix.shape();
    let mut tables = [[0i64; 16]; LOOKUP_GROUPS_PER_TILE];
    for group_base in (0..shape.col_groups()).step_by(LOOKUP_GROUPS_PER_TILE) {
        let groups = (shape.col_groups() - group_base).min(LOOKUP_GROUPS_PER_TILE);
        for (local_group, table) in tables.iter_mut().take(groups).enumerate() {
            *table = build_lookup_table_i64(input, group_base + local_group);
        }
        let mut row = 0;
        while row + 8 <= shape.rows() {
            let mut accumulator = _mm512_loadu_si512(output.as_ptr().add(row).cast());
            let row_pair = row >> 1;
            for (local_group, table) in tables.iter().take(groups).enumerate() {
                let low = _mm512_loadu_si512(table.as_ptr().cast());
                let high = _mm512_loadu_si512(table.as_ptr().add(8).cast());
                let (first, second) = matrix.sign_groups_unchecked(group_base + local_group);
                accumulator = _mm512_add_epi64(
                    accumulator,
                    _mm512_permutex2var_epi64(
                        low,
                        selector_indices_8_i64_baseline(first.as_ptr().add(row_pair)),
                        high,
                    ),
                );
                accumulator = _mm512_add_epi64(
                    accumulator,
                    _mm512_permutex2var_epi64(
                        low,
                        selector_indices_8_i64_baseline(second.as_ptr().add(row_pair)),
                        high,
                    ),
                );
            }
            _mm512_storeu_si512(output.as_mut_ptr().add(row).cast(), accumulator);
            row += 8;
        }
        accumulate_scalar_tail_i64(matrix, group_base, groups, &tables, row, output);
    }
}

#[cfg(test)]
#[target_feature(enable = "avx512f")]
unsafe fn selector_indices_16_i32_baseline(packed: *const u8) -> __m512i {
    let bytes = std::ptr::read_unaligned(packed.cast::<u64>()).to_le_bytes();
    let expanded = [
        expand_selector_pair_baseline(bytes[0]),
        expand_selector_pair_baseline(bytes[1]),
        expand_selector_pair_baseline(bytes[2]),
        expand_selector_pair_baseline(bytes[3]),
        expand_selector_pair_baseline(bytes[4]),
        expand_selector_pair_baseline(bytes[5]),
        expand_selector_pair_baseline(bytes[6]),
        expand_selector_pair_baseline(bytes[7]),
    ];
    _mm512_cvtepu8_epi32(_mm_loadu_si128(expanded.as_ptr().cast()))
}

#[cfg(test)]
const fn expand_selector_pair_baseline(byte: u8) -> u16 {
    (byte & 0x0f) as u16 | (((byte >> 4) as u16) << 8)
}

#[cfg(test)]
#[target_feature(enable = "avx512f")]
unsafe fn selector_indices_8_i64_baseline(packed: *const u8) -> __m512i {
    let bytes = std::ptr::read_unaligned(packed.cast::<u32>()).to_le_bytes();
    let expanded = u64::from(bytes[0] & 0x0f)
        | (u64::from(bytes[0] >> 4) << 8)
        | (u64::from(bytes[1] & 0x0f) << 16)
        | (u64::from(bytes[1] >> 4) << 24)
        | (u64::from(bytes[2] & 0x0f) << 32)
        | (u64::from(bytes[2] >> 4) << 40)
        | (u64::from(bytes[3] & 0x0f) << 48)
        | (u64::from(bytes[3] >> 4) << 56);
    _mm512_cvtepu8_epi64(_mm_cvtsi64_si128(expanded as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jl::TernaryProjectionShape;
    use std::{hint::black_box, time::Instant};

    fn patterned_matrix(rows: usize, cols: usize) -> TernaryProjectionMatrix {
        let shape = TernaryProjectionShape::new(rows, cols).unwrap();
        let mut first = vec![0u8; shape.plane_len()];
        let mut second = vec![0u8; shape.plane_len()];
        for row in 0..rows {
            for col in 0..cols {
                let index = (col >> 2) * shape.row_pairs() + (row >> 1);
                let bit = 1u8 << (((row & 1) << 2) | (col & 3));
                match (row * 37 + col * 19 + row * col) % 4 {
                    0 => {}
                    1 | 3 => {
                        first[index] |= bit;
                        second[index] |= bit;
                    }
                    _ => first[index] |= bit,
                }
            }
        }
        TernaryProjectionMatrix::from_rademacher_bitplanes(shape, first, second).unwrap()
    }

    fn packed_small_avx2<T: SmallLookupInput>(
        matrix: &TernaryProjectionMatrix,
        input: &[T],
    ) -> Vec<i64> {
        let mut output = vec![0; matrix.shape().rows()];
        // SAFETY: every caller is guarded by runtime AVX2 detection.
        unsafe { project_lookup_i32_avx2(matrix, input, &mut output) };
        super::super::finish_doubled_projection(&mut output).unwrap();
        output
    }

    fn packed_small_avx512<T: SmallLookupInput>(
        matrix: &TernaryProjectionMatrix,
        input: &[T],
    ) -> Vec<i64> {
        let mut output = vec![0; matrix.shape().rows()];
        // SAFETY: every caller is guarded by runtime AVX-512F detection.
        unsafe { project_lookup_i32_avx512(matrix, input, &mut output) };
        super::super::finish_doubled_projection(&mut output).unwrap();
        output
    }

    fn packed_wide_avx2<T: WideLookupInput>(
        matrix: &TernaryProjectionMatrix,
        input: &[T],
    ) -> Vec<i64> {
        let mut output = vec![0; matrix.shape().rows()];
        // SAFETY: every caller is guarded by runtime AVX2 detection and test
        // inputs have bounded doubled L1 norms.
        unsafe { project_lookup_i64_avx2(matrix, input, &mut output) };
        super::super::finish_doubled_projection(&mut output).unwrap();
        output
    }

    fn packed_wide_avx512<T: WideLookupInput>(
        matrix: &TernaryProjectionMatrix,
        input: &[T],
    ) -> Vec<i64> {
        let mut output = vec![0; matrix.shape().rows()];
        // SAFETY: every caller is guarded by runtime AVX-512F detection and
        // test inputs have bounded doubled L1 norms.
        unsafe { project_lookup_i64_avx512(matrix, input, &mut output) };
        super::super::finish_doubled_projection(&mut output).unwrap();
        output
    }

    fn packed_small_avx512_baseline<T: SmallLookupInput>(
        matrix: &TernaryProjectionMatrix,
        input: &[T],
    ) -> Vec<i64> {
        let mut output = vec![0; matrix.shape().rows()];
        // SAFETY: every caller is guarded by runtime AVX-512F detection.
        unsafe { project_lookup_i32_avx512_baseline(matrix, input, &mut output) };
        super::super::finish_doubled_projection(&mut output).unwrap();
        output
    }

    fn packed_wide_avx512_baseline<T: WideLookupInput>(
        matrix: &TernaryProjectionMatrix,
        input: &[T],
    ) -> Vec<i64> {
        let mut output = vec![0; matrix.shape().rows()];
        // SAFETY: every caller is guarded by runtime AVX-512F detection and
        // test inputs have bounded doubled L1 norms.
        unsafe { project_lookup_i64_avx512_baseline(matrix, input, &mut output) };
        super::super::finish_doubled_projection(&mut output).unwrap();
        output
    }

    #[test]
    fn forced_x86_backends_match_scalar_for_widths_blocks_and_tails() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for &(rows, cols) in &[(1, 1), (3, 5), (7, 67), (17, 259)] {
            let matrix = patterned_matrix(rows, cols);
            let input_i8 = (0..2 * cols)
                .map(|index| (index % 127) as i8 - 63)
                .collect::<Vec<_>>();
            let input_i16 = input_i8
                .iter()
                .copied()
                .map(|value| i16::from(value) * 509)
                .collect::<Vec<_>>();
            let input_i32 = input_i8
                .iter()
                .copied()
                .map(|value| i32::from(value) * 1_000_003)
                .collect::<Vec<_>>();
            let input_i64 = input_i8
                .iter()
                .copied()
                .map(|value| i64::from(value) * 10_000_019)
                .collect::<Vec<_>>();

            for block in 0..2 {
                let range = block * cols..(block + 1) * cols;
                let expected_i8 =
                    super::super::dense::tests::scalar_i8(&matrix, &input_i8[range.clone()]);
                let expected_i16 =
                    super::super::dense::tests::scalar_i16(&matrix, &input_i16[range.clone()]);
                let expected_i32 =
                    super::super::dense::tests::scalar_i32(&matrix, &input_i32[range.clone()]);
                let expected_i64 =
                    super::super::dense::tests::scalar_i64(&matrix, &input_i64[range.clone()]);

                assert_eq!(
                    packed_small_avx2(&matrix, &input_i8[range.clone()]),
                    expected_i8
                );
                assert_eq!(
                    packed_small_avx2(&matrix, &input_i16[range.clone()]),
                    expected_i16
                );
                assert_eq!(
                    packed_wide_avx2(&matrix, &input_i32[range.clone()]),
                    expected_i32
                );
                assert_eq!(
                    packed_wide_avx2(&matrix, &input_i64[range.clone()]),
                    expected_i64
                );
                if std::arch::is_x86_feature_detected!("avx512f") {
                    assert_eq!(
                        packed_small_avx512(&matrix, &input_i8[range.clone()]),
                        expected_i8
                    );
                    assert_eq!(
                        packed_small_avx512(&matrix, &input_i16[range.clone()]),
                        expected_i16
                    );
                    assert_eq!(
                        packed_wide_avx512(&matrix, &input_i32[range.clone()]),
                        expected_i32
                    );
                    assert_eq!(packed_wide_avx512(&matrix, &input_i64[range]), expected_i64);
                }
            }
        }
    }

    #[test]
    fn forced_x86_backends_exhaust_paired_selector_bytes() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let shape = TernaryProjectionShape::new(16, 4).unwrap();
        let input_i8 = [-17i8, 23, -31, 47];
        let input_i32 = input_i8.map(|value| i32::from(value) * 1_000_003);
        let rademacher_sum = |selector: u8, input: &[i64; 4]| {
            input.iter().enumerate().fold(0i64, |sum, (lane, &value)| {
                if selector & (1 << lane) == 0 {
                    sum - value
                } else {
                    sum + value
                }
            })
        };
        let wide_input = input_i32.map(i64::from);
        for first in 0u8..=u8::MAX {
            for second in 0u8..=u8::MAX {
                let matrix = TernaryProjectionMatrix::from_rademacher_bitplanes(
                    shape,
                    vec![first; shape.plane_len()],
                    vec![second; shape.plane_len()],
                )
                .unwrap();
                let expected_even = (rademacher_sum(first & 0x0f, &wide_input)
                    + rademacher_sum(second & 0x0f, &wide_input))
                    / 2;
                let expected_odd = (rademacher_sum(first >> 4, &wide_input)
                    + rademacher_sum(second >> 4, &wide_input))
                    / 2;
                let expected = (0..shape.rows())
                    .map(|row| {
                        if row & 1 == 0 {
                            expected_even
                        } else {
                            expected_odd
                        }
                    })
                    .collect::<Vec<_>>();
                let expected_i8 = expected
                    .iter()
                    .map(|value| value / 1_000_003)
                    .collect::<Vec<_>>();
                assert_eq!(packed_small_avx2(&matrix, &input_i8), expected_i8);
                assert_eq!(packed_wide_avx2(&matrix, &input_i32), expected);
                if std::arch::is_x86_feature_detected!("avx512f") {
                    assert_eq!(packed_small_avx512(&matrix, &input_i8), expected_i8);
                    assert_eq!(packed_wide_avx512(&matrix, &input_i32), expected);
                }
            }
        }
    }

    #[test]
    fn extreme_inputs_preserve_checked_projection_semantics() {
        let zero = TernaryProjectionMatrix::from_rademacher_bitplanes(
            TernaryProjectionShape::new(1, 4).unwrap(),
            vec![0x0f],
            vec![0],
        )
        .unwrap();
        assert_eq!(
            zero.project(&[i64::MIN, i64::MAX, i64::MIN, i64::MAX])
                .unwrap(),
            [0]
        );

        let negative = TernaryProjectionMatrix::from_rademacher_bitplanes(
            TernaryProjectionShape::new(1, 1).unwrap(),
            vec![0],
            vec![0],
        )
        .unwrap();
        assert!(negative.project(&[i64::MIN]).is_err());
        assert_eq!(
            negative.project(&[i32::MIN]).unwrap(),
            [i64::from(i32::MAX) + 1]
        );
        assert_eq!(negative.project(&[i16::MIN]).unwrap(), [32_768]);
        assert_eq!(negative.project(&[i8::MIN]).unwrap(), [128]);
    }

    #[test]
    fn cold_packed_marker_is_single_use_and_clone_local() {
        let matrix = patterned_matrix(3, 5);
        assert!(matrix.take_cold_packed_projection());
        assert!(!matrix.take_cold_packed_projection());
        let clone = matrix.clone();
        assert!(clone.take_cold_packed_projection());
        assert!(!clone.take_cold_packed_projection());
        let dense = patterned_matrix(3, 5);
        super::super::dense::tests::scalar_i8(&dense, &[1, 2, 3, 4, 5]);
        assert!(!dense.take_cold_packed_projection());
    }

    fn time_many(mut run: impl FnMut(), iterations: u32) -> u128 {
        let start = Instant::now();
        for _ in 0..iterations {
            run();
        }
        start.elapsed().as_nanos() / u128::from(iterations)
    }

    #[test]
    #[ignore = "manual x86 projection microbenchmark"]
    fn x86_projection_microbench() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 unavailable");
            return;
        }
        macro_rules! report {
            ($matrix:expr, $cols:expr, $iterations:expr, $name:literal, $input:expr, $avx2:ident, $avx512:ident, $baseline:ident, $dense:ident) => {{
                let matrix = $matrix;
                let cols = $cols;
                let iterations = $iterations;
                let input = $input;
                let hot = matrix.clone();
                let mut output = vec![0; hot.shape().rows()];
                let mut cold_output = vec![0; hot.shape().rows()];
                super::super::dense::$dense(&hot, input, &mut output).unwrap();
                for trial in 0..5 {
                    let measure_avx2 = || {
                        time_many(
                            || {
                                black_box($avx2(&matrix, input));
                            },
                            iterations,
                        )
                    };
                    let mut measure_hot = || {
                        time_many(
                            || super::super::dense::$dense(&hot, input, &mut output).unwrap(),
                            iterations,
                        )
                    };
                    let mut measure_cold = || {
                        time_many(
                            || {
                                let cold = matrix.clone();
                                super::super::dense::$dense(&cold, input, &mut cold_output).unwrap();
                            },
                            iterations,
                        )
                    };
                    let (avx2, dense_hot, dense_cold) = if trial & 1 == 0 {
                        (measure_avx2(), measure_hot(), measure_cold())
                    } else {
                        let cold = measure_cold();
                        let hot = measure_hot();
                        (measure_avx2(), hot, cold)
                    };
                    eprintln!(
                        "cols={cols} trial={trial} {} avx2_packed={avx2}ns dense_hot={dense_hot}ns dense_cold={dense_cold}ns",
                        $name
                    );
                    if std::arch::is_x86_feature_detected!("avx512f") {
                        let measure_avx512 = || {
                            time_many(
                                || {
                                    black_box($avx512(&matrix, input));
                                },
                                iterations,
                            )
                        };
                        let measure_baseline = || {
                            time_many(
                                || {
                                    black_box($baseline(&matrix, input));
                                },
                                iterations,
                            )
                        };
                        let (avx512, baseline) = if trial & 1 == 0 {
                            (measure_avx512(), measure_baseline())
                        } else {
                            let baseline = measure_baseline();
                            (measure_avx512(), baseline)
                        };
                        eprintln!(
                            "cols={cols} trial={trial} {} avx512_packed={avx512}ns avx512_baseline={baseline}ns",
                            $name
                        );
                    }
                }
            }};
        }
        for cols in [1 << 12, 1 << 14, 1 << 16] {
            let matrix = patterned_matrix(256, cols);
            let input_i8 = (0..cols)
                .map(|index| (index % 127) as i8 - 63)
                .collect::<Vec<_>>();
            let input_i16 = input_i8
                .iter()
                .map(|&value| i16::from(value) * 509)
                .collect::<Vec<_>>();
            let input_i32 = input_i8
                .iter()
                .map(|&value| i32::from(value) * 1_000_003)
                .collect::<Vec<_>>();
            let input_i64 = input_i8
                .iter()
                .map(|&value| i64::from(value) * 10_000_019)
                .collect::<Vec<_>>();
            let iterations = (200 * (1 << 14) / cols).max(20) as u32;
            report!(
                &matrix,
                cols,
                iterations,
                "i8",
                &input_i8,
                packed_small_avx2,
                packed_small_avx512,
                packed_small_avx512_baseline,
                project_i8
            );
            report!(
                &matrix,
                cols,
                iterations,
                "i16",
                &input_i16,
                packed_small_avx2,
                packed_small_avx512,
                packed_small_avx512_baseline,
                project_i16
            );
            report!(
                &matrix,
                cols,
                iterations,
                "i32",
                &input_i32,
                packed_wide_avx2,
                packed_wide_avx512,
                packed_wide_avx512_baseline,
                project_i32
            );
            report!(
                &matrix,
                cols,
                iterations,
                "i64",
                &input_i64,
                packed_wide_avx2,
                packed_wide_avx512,
                packed_wide_avx512_baseline,
                project_i64
            );
        }
    }
}
