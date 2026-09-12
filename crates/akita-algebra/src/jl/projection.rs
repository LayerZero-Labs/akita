//! Exact paired-Rademacher lookup projection kernels.

mod dense;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use super::{ProjectionInput, TernaryProjectionMatrix};
#[cfg(feature = "parallel")]
use akita_error::checked;
use akita_error::AkitaError;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(target_arch = "aarch64")]
const AARCH64_LOOKUP_GROUPS_PER_TILE: usize = 32;
#[cfg(all(target_arch = "aarch64", feature = "parallel"))]
const AARCH64_PARALLEL_ROWS_PER_CHUNK: usize = 32;
#[cfg(all(target_arch = "aarch64", feature = "parallel"))]
const AARCH64_PARALLEL_PACKED_WORK_THRESHOLD: usize = 1 << 26;
#[cfg(target_arch = "x86_64")]
pub(super) const LOOKUP_GROUPS_PER_TILE: usize = 16;

pub(super) mod private {
    use super::{AkitaError, TernaryProjectionMatrix};

    #[cfg(target_arch = "aarch64")]
    pub trait AarchColdProjection: Sized {
        fn project_cold(
            matrix: &TernaryProjectionMatrix,
            input: &[Self],
            output: &mut [i64],
        ) -> Result<(), AkitaError>;
    }

    #[cfg(target_arch = "aarch64")]
    pub trait Sealed: Sized + AarchColdProjection {
        fn project(
            matrix: &TernaryProjectionMatrix,
            input: &[Self],
            output: &mut [i64],
        ) -> Result<(), AkitaError>;
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub trait Sealed: Sized {
        fn project(
            matrix: &TernaryProjectionMatrix,
            input: &[Self],
            output: &mut [i64],
        ) -> Result<(), AkitaError>;
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(super) trait WideLookupInput: Copy + Send + Sync {
    fn to_i64(self) -> i64;
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
impl WideLookupInput for i32 {
    #[inline(always)]
    fn to_i64(self) -> i64 {
        i64::from(self)
    }
}

#[cfg(target_arch = "x86_64")]
impl WideLookupInput for i64 {
    #[inline(always)]
    fn to_i64(self) -> i64 {
        self
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) trait SmallLookupInput: Copy + Send + Sync {
    fn to_i32(self) -> i32;
}

macro_rules! impl_projection_input {
    ($integer:ty, $project:ident) => {
        impl private::Sealed for $integer {
            fn project(
                matrix: &TernaryProjectionMatrix,
                input: &[Self],
                output: &mut [i64],
            ) -> Result<(), AkitaError> {
                $project(matrix, input, output)
            }
        }
    };
}

impl_projection_input!(i8, project_i8);
impl_projection_input!(i16, project_i16);
impl_projection_input!(i32, project_i32);

#[cfg(target_arch = "aarch64")]
impl private::AarchColdProjection for i8 {
    fn project_cold(
        matrix: &TernaryProjectionMatrix,
        input: &[Self],
        output: &mut [i64],
    ) -> Result<(), AkitaError> {
        dense::project_i8(matrix, input, output)
    }
}

#[cfg(target_arch = "aarch64")]
impl private::AarchColdProjection for i16 {
    fn project_cold(
        matrix: &TernaryProjectionMatrix,
        input: &[Self],
        output: &mut [i64],
    ) -> Result<(), AkitaError> {
        dense::project_i16(matrix, input, output)
    }
}

#[cfg(target_arch = "aarch64")]
impl private::AarchColdProjection for i32 {
    fn project_cold(
        matrix: &TernaryProjectionMatrix,
        input: &[Self],
        output: &mut [i64],
    ) -> Result<(), AkitaError> {
        project_wide_packed(matrix, input, output)
    }
}

#[cfg(target_arch = "x86_64")]
impl SmallLookupInput for i8 {
    #[inline(always)]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }
}

#[cfg(target_arch = "x86_64")]
impl SmallLookupInput for i16 {
    #[inline(always)]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }
}

impl private::Sealed for i64 {
    fn project(
        matrix: &TernaryProjectionMatrix,
        input: &[Self],
        output: &mut [i64],
    ) -> Result<(), AkitaError> {
        project_i64(matrix, input, output)
    }
}

#[cfg(target_arch = "aarch64")]
impl private::AarchColdProjection for i64 {
    fn project_cold(
        matrix: &TernaryProjectionMatrix,
        input: &[Self],
        output: &mut [i64],
    ) -> Result<(), AkitaError> {
        dense::project_i64(matrix, input, output)
    }
}

#[cfg(feature = "parallel")]
const PARALLEL_WORK_THRESHOLD: usize = 1 << 18;

pub(super) fn project_i8(
    matrix: &TernaryProjectionMatrix,
    input: &[i8],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return project_small_x86(matrix, input, output, X86LookupBackend::Avx512);
        }
        if std::arch::is_x86_feature_detected!("avx2") && matrix.take_cold_packed_projection() {
            return project_small_x86(matrix, input, output, X86LookupBackend::Avx2);
        }
    }
    dense::project_i8(matrix, input, output)
}

pub(super) fn project_i16(
    matrix: &TernaryProjectionMatrix,
    input: &[i16],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return project_small_x86(matrix, input, output, X86LookupBackend::Avx512);
        }
        if std::arch::is_x86_feature_detected!("avx2") && matrix.take_cold_packed_projection() {
            return project_small_x86(matrix, input, output, X86LookupBackend::Avx2);
        }
    }
    dense::project_i16(matrix, input, output)
}

pub(super) fn project_i32(
    matrix: &TernaryProjectionMatrix,
    input: &[i32],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    #[cfg(target_arch = "aarch64")]
    {
        if matrix.take_cold_packed_projection() {
            return project_wide_packed(matrix, input, output);
        }
        dense::project_i32(matrix, input, output)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return project_wide_x86(matrix, input, output, X86LookupBackend::Avx512);
            }
            if std::arch::is_x86_feature_detected!("avx2") && matrix.take_cold_packed_projection() {
                return project_wide_x86(matrix, input, output, X86LookupBackend::Avx2);
            }
        }
        dense::project_i32(matrix, input, output)
    }
}

pub(super) fn project_i64(
    matrix: &TernaryProjectionMatrix,
    input: &[i64],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    #[cfg(target_arch = "x86_64")]
    {
        if doubled_l1_fits_i64(input) {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return project_wide_x86(matrix, input, output, X86LookupBackend::Avx512);
            }
            if std::arch::is_x86_feature_detected!("avx2") && matrix.take_cold_packed_projection() {
                return project_wide_x86(matrix, input, output, X86LookupBackend::Avx2);
            }
        }
    }
    dense::project_i64(matrix, input, output)
}

#[cfg(target_arch = "aarch64")]
fn project_wide_packed<T: WideLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    output.fill(0);
    #[cfg(feature = "parallel")]
    if matrix.shape().dense_len() >= AARCH64_PARALLEL_PACKED_WORK_THRESHOLD
        && output.len() > AARCH64_PARALLEL_ROWS_PER_CHUNK
    {
        output
            .par_chunks_mut(AARCH64_PARALLEL_ROWS_PER_CHUNK)
            .enumerate()
            .for_each(|(chunk, rows)| {
                project_wide_packed_rows(
                    matrix,
                    input,
                    rows,
                    chunk * AARCH64_PARALLEL_ROWS_PER_CHUNK,
                );
            });
        return finish_doubled_projection(output);
    }
    project_wide_packed_rows(matrix, input, output, 0);
    finish_doubled_projection(output)
}

#[cfg(target_arch = "aarch64")]
fn project_wide_packed_rows<T: WideLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
    first_row: usize,
) {
    let shape = matrix.shape();
    let mut tables = [[0i64; 16]; AARCH64_LOOKUP_GROUPS_PER_TILE];
    for group_base in (0..shape.col_groups()).step_by(AARCH64_LOOKUP_GROUPS_PER_TILE) {
        let groups = (shape.col_groups() - group_base).min(AARCH64_LOOKUP_GROUPS_PER_TILE);
        for (local_group, table) in tables.iter_mut().take(groups).enumerate() {
            *table = build_lookup_table_i64(input, group_base + local_group);
        }
        accumulate_packed_i64_tile(matrix, group_base, groups, &tables, first_row, output);
    }
}

#[cfg(target_arch = "aarch64")]
fn accumulate_packed_i64_tile(
    matrix: &TernaryProjectionMatrix,
    group_base: usize,
    groups: usize,
    tables: &[[i64; 16]; AARCH64_LOOKUP_GROUPS_PER_TILE],
    first_row: usize,
    output: &mut [i64],
) {
    for (row_pair, rows) in output.chunks_mut(2).enumerate() {
        let row_pair = (first_row >> 1) + row_pair;
        let mut even = rows[0];
        let mut odd = rows.get(1).copied().unwrap_or_default();
        for (local_group, table) in tables.iter().take(groups).enumerate() {
            let (first, second) = matrix.sign_groups_unchecked(group_base + local_group);
            let first = first[row_pair];
            let second = second[row_pair];
            even += table[usize::from(first & 0x0f)] + table[usize::from(second & 0x0f)];
            odd += table[usize::from(first >> 4)] + table[usize::from(second >> 4)];
        }
        rows[0] = even;
        if let Some(value) = rows.get_mut(1) {
            *value = odd;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
enum X86LookupBackend {
    Avx2,
    Avx512,
}

#[cfg(target_arch = "x86_64")]
fn project_small_x86<T: SmallLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
    backend: X86LookupBackend,
) -> Result<(), AkitaError> {
    output.fill(0);
    // SAFETY: each caller runtime-detects the feature required by its selected
    // backend; public projection methods validated all slice lengths.
    unsafe {
        match backend {
            X86LookupBackend::Avx2 => x86_64::project_lookup_i32_avx2(matrix, input, output),
            X86LookupBackend::Avx512 => x86_64::project_lookup_i32_avx512(matrix, input, output),
        }
    }
    finish_doubled_projection(output)
}

#[cfg(target_arch = "x86_64")]
fn project_wide_x86<T: WideLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
    backend: X86LookupBackend,
) -> Result<(), AkitaError> {
    output.fill(0);
    // SAFETY: each caller runtime-detects the feature required by its selected
    // backend. The i64 caller proves that doubled intermediates fit. For i32,
    // the 1-GiB materialization limit bounds the number of columns below 2^30,
    // so twice the L1 norm is at most 2^62 and fits in i64. Public projection
    // methods validated exact input and output lengths before reaching here.
    unsafe {
        match backend {
            X86LookupBackend::Avx2 => x86_64::project_lookup_i64_avx2(matrix, input, output),
            X86LookupBackend::Avx512 => x86_64::project_lookup_i64_avx512(matrix, input, output),
        }
    }
    finish_doubled_projection(output)
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn finish_doubled_projection(output: &mut [i64]) -> Result<(), AkitaError> {
    for value in output {
        if *value & 1 != 0 {
            return Err(AkitaError::InvalidInput(
                "paired-Rademacher projection produced an odd doubled coordinate".into(),
            ));
        }
        *value >>= 1;
    }
    Ok(())
}

pub(super) fn project_i128(
    matrix: &TernaryProjectionMatrix,
    input: &[i128],
    output: &mut [i128],
) -> Result<(), AkitaError> {
    dense::project_i128(matrix, input, output)
}

pub(super) fn project_blocks<T: ProjectionInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    let cols = matrix.shape().cols();
    let rows = matrix.shape().rows();
    #[cfg(target_arch = "x86_64")]
    if output.len() > rows
        && std::arch::is_x86_feature_detected!("avx2")
        && !std::arch::is_x86_feature_detected!("avx512f")
    {
        // Multiple blocks immediately reuse the matrix, so materialize once
        // instead of paying for a packed cold pass before the dense hot path.
        matrix.dense_rows()?;
    }
    #[cfg(target_arch = "aarch64")]
    if matrix.take_cold_packed_projection() {
        #[cfg(feature = "parallel")]
        if output.len() > rows
            && checked::product([output.len(), cols])
                .is_none_or(|work| work >= PARALLEL_WORK_THRESHOLD)
        {
            return input
                .par_chunks_exact(cols)
                .zip(output.par_chunks_exact_mut(rows))
                .try_for_each(|(block_input, block_output)| {
                    <T as private::AarchColdProjection>::project_cold(
                        matrix,
                        block_input,
                        block_output,
                    )
                });
        }
        for (block_input, block_output) in
            input.chunks_exact(cols).zip(output.chunks_exact_mut(rows))
        {
            <T as private::AarchColdProjection>::project_cold(matrix, block_input, block_output)?;
        }
        return Ok(());
    }
    #[cfg(feature = "parallel")]
    if output.len() > rows
        && checked::product([output.len(), cols]).is_none_or(|work| work >= PARALLEL_WORK_THRESHOLD)
    {
        return input
            .par_chunks_exact(cols)
            .zip(output.par_chunks_exact_mut(rows))
            .try_for_each(|(block_input, block_output)| {
                <T as private::Sealed>::project(matrix, block_input, block_output)
            });
    }
    for (block_input, block_output) in input.chunks_exact(cols).zip(output.chunks_exact_mut(rows)) {
        <T as private::Sealed>::project(matrix, block_input, block_output)?;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn build_lookup_table_i32<T: SmallLookupInput>(input: &[T], group: usize) -> [i32; 16] {
    let start = group * 4;
    let mut values = [0i32; 4];
    for (lane, value) in values.iter_mut().enumerate() {
        if let Some(&input_value) = input.get(start + lane) {
            *value = input_value.to_i32();
        }
    }
    let first = [
        -values[0] - values[1],
        values[0] - values[1],
        -values[0] + values[1],
        values[0] + values[1],
    ];
    let second = [
        -values[2] - values[3],
        values[2] - values[3],
        -values[2] + values[3],
        values[2] + values[3],
    ];
    std::array::from_fn(|selector| first[selector & 3] + second[selector >> 2])
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
pub(super) fn build_lookup_table_i64<T: WideLookupInput>(input: &[T], group: usize) -> [i64; 16] {
    let start = group * 4;
    let mut values = [0i64; 4];
    for (lane, value) in values.iter_mut().enumerate() {
        if let Some(&input_value) = input.get(start + lane) {
            *value = input_value.to_i64();
        }
    }
    finish_lookup_table_i64(values)
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[inline]
fn finish_lookup_table_i64(values: [i64; 4]) -> [i64; 16] {
    let first = [
        -values[0] - values[1],
        values[0] - values[1],
        -values[0] + values[1],
        values[0] + values[1],
    ];
    let second = [
        -values[2] - values[3],
        values[2] - values[3],
        -values[2] + values[3],
        values[2] + values[3],
    ];
    std::array::from_fn(|selector| first[selector & 3] + second[selector >> 2])
}

#[cfg(target_arch = "x86_64")]
fn doubled_l1_fits_i64(input: &[i64]) -> bool {
    let limit = (i64::MAX as u64) >> 1;
    let mut sum = 0u64;
    for &value in input {
        let Some(next) = sum.checked_add(value.unsigned_abs()) else {
            return false;
        };
        if next > limit {
            return false;
        }
        sum = next;
    }
    true
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn avx512_small<T: SmallLookupInput>(
        matrix: &TernaryProjectionMatrix,
        input: &[T],
    ) -> Vec<i64> {
        let mut output = vec![0i64; matrix.shape().rows()];
        // SAFETY: callers guard this test helper with runtime AVX-512F detection.
        unsafe { x86_64::project_lookup_i32_avx512(matrix, input, &mut output) };
        finish_doubled_projection(&mut output).unwrap();
        output
    }

    pub(crate) fn dense_scalar_i32(matrix: &TernaryProjectionMatrix, input: &[i32]) -> Vec<i64> {
        dense::tests::scalar_i32(matrix, input)
    }

    pub(crate) fn dense_scalar_i64(matrix: &TernaryProjectionMatrix, input: &[i64]) -> Vec<i64> {
        dense::tests::scalar_i64(matrix, input)
    }

    pub(crate) fn dense_scalar_i8(matrix: &TernaryProjectionMatrix, input: &[i8]) -> Vec<i64> {
        dense::tests::scalar_i8(matrix, input)
    }

    pub(crate) fn dense_scalar_i16(matrix: &TernaryProjectionMatrix, input: &[i16]) -> Vec<i64> {
        dense::tests::scalar_i16(matrix, input)
    }
}
