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

#[cfg(target_arch = "x86_64")]
pub(super) const LOOKUP_GROUPS_PER_TILE: usize = 16;
#[cfg(target_arch = "x86_64")]
const I16_LOOKUP_MIN_COLS: usize = 1 << 16;

pub(super) mod private {
    use super::{AkitaError, TernaryProjectionMatrix};

    pub trait Sealed: Sized {
        fn project(
            matrix: &TernaryProjectionMatrix,
            input: &[Self],
            output: &mut [i64],
        ) -> Result<(), AkitaError>;
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
            return project_small_avx512(matrix, input, output);
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
        if matrix.shape().cols() >= I16_LOOKUP_MIN_COLS
            && std::arch::is_x86_feature_detected!("avx512f")
        {
            return project_small_avx512(matrix, input, output);
        }
    }
    dense::project_i16(matrix, input, output)
}

pub(super) fn project_i32(
    matrix: &TernaryProjectionMatrix,
    input: &[i32],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    dense::project_i32(matrix, input, output)
}

pub(super) fn project_i64(
    matrix: &TernaryProjectionMatrix,
    input: &[i64],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") && doubled_l1_fits_i64(input) {
            output.fill(0);
            // SAFETY: runtime dispatch checked AVX-512F; public projection
            // methods validated all slice lengths.
            unsafe { x86_64::project_lookup_i64_avx512(matrix, input, output) };
            return finish_doubled_projection(output);
        }
    }
    dense::project_i64(matrix, input, output)
}

#[cfg(target_arch = "x86_64")]
fn project_small_avx512<T: SmallLookupInput>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    output.fill(0);
    // SAFETY: callers reach this function only after runtime AVX-512F
    // detection; public projection methods validated all slice lengths.
    unsafe { x86_64::project_lookup_i32_avx512(matrix, input, output) };
    finish_doubled_projection(output)
}

#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn build_lookup_table_i64(input: &[i64], group: usize) -> [i64; 16] {
    let start = group * 4;
    let mut values = [0i64; 4];
    for (lane, value) in values.iter_mut().enumerate() {
        if let Some(&input_value) = input.get(start + lane) {
            *value = input_value;
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

#[inline]
#[cfg(target_arch = "x86_64")]
pub(super) const fn expand_selector_pair(byte: u8) -> u16 {
    (byte & 0x0f) as u16 | (((byte >> 4) as u16) << 8)
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
