//! x86-64 paired-Rademacher lookup kernels.

use super::{
    build_lookup_table_i32, build_lookup_table_i64, expand_selector_pair, SmallLookupInput,
    TernaryProjectionMatrix, LOOKUP_GROUPS_PER_TILE,
};
use std::arch::x86_64::*;

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

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn project_lookup_i64_avx512(
    matrix: &TernaryProjectionMatrix,
    input: &[i64],
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

#[target_feature(enable = "avx512f")]
unsafe fn selector_indices_16_i32(packed: *const u8) -> __m512i {
    let bytes = std::ptr::read_unaligned(packed.cast::<u64>()).to_le_bytes();
    let expanded = [
        expand_selector_pair(bytes[0]),
        expand_selector_pair(bytes[1]),
        expand_selector_pair(bytes[2]),
        expand_selector_pair(bytes[3]),
        expand_selector_pair(bytes[4]),
        expand_selector_pair(bytes[5]),
        expand_selector_pair(bytes[6]),
        expand_selector_pair(bytes[7]),
    ];
    _mm512_cvtepu8_epi32(_mm_loadu_si128(expanded.as_ptr().cast()))
}

#[target_feature(enable = "avx512f")]
unsafe fn selector_indices_8_i64(packed: *const u8) -> __m512i {
    let bytes = std::ptr::read_unaligned(packed.cast::<u32>()).to_le_bytes();
    let expanded = u64::from(expand_selector_pair(bytes[0]))
        | (u64::from(expand_selector_pair(bytes[1])) << 16)
        | (u64::from(expand_selector_pair(bytes[2])) << 32)
        | (u64::from(expand_selector_pair(bytes[3])) << 48);
    _mm512_cvtepu8_epi64(_mm_cvtsi64_si128(expanded as i64))
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
