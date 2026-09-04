//! Exact scalar and architecture-specific integer projection kernels.

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use super::{ProjectionInput, TernaryProjectionMatrix};
#[cfg(feature = "parallel")]
use akita_error::checked;
use akita_error::AkitaError;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
impl_projection_input!(i64, project_i64);

type Dot<T> = fn(&[i8], &[T]) -> i64;

#[cfg(feature = "parallel")]
const PARALLEL_WORK_THRESHOLD: usize = 1 << 18;
#[cfg(feature = "parallel")]
const PARALLEL_I8_WORK_THRESHOLD: usize = 1 << 22;

pub(super) fn project_i8(
    matrix: &TernaryProjectionMatrix,
    input: &[i8],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    fill_rows(
        matrix,
        input,
        output,
        selected_i8_dot(),
        parallel_i8_threshold(),
    );
    Ok(())
}

pub(super) fn project_i16(
    matrix: &TernaryProjectionMatrix,
    input: &[i16],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    fill_rows(
        matrix,
        input,
        output,
        selected_i16_dot(),
        parallel_threshold(),
    );
    Ok(())
}

pub(super) fn project_i32(
    matrix: &TernaryProjectionMatrix,
    input: &[i32],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    fill_rows(
        matrix,
        input,
        output,
        selected_i32_dot(),
        parallel_threshold(),
    );
    Ok(())
}

pub(super) fn project_i64(
    matrix: &TernaryProjectionMatrix,
    input: &[i64],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    if l1_fits_i64(input) {
        fill_rows(
            matrix,
            input,
            output,
            selected_i64_dot(),
            parallel_threshold(),
        );
        return Ok(());
    }
    fill_rows_checked(matrix, input, output)
}

pub(super) fn project_i128(
    matrix: &TernaryProjectionMatrix,
    input: &[i128],
    output: &mut [i128],
) -> Result<(), AkitaError> {
    if l1_fits_i128(input) {
        fill_rows(matrix, input, output, dot_i128_scalar, parallel_threshold());
        return Ok(());
    }
    fill_rows_i128_checked(matrix, input, output)
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

fn fill_rows<T: Sync, U: Send>(
    matrix: &TernaryProjectionMatrix,
    input: &[T],
    output: &mut [U],
    dot: fn(&[i8], &[T]) -> U,
    parallel_work_threshold: usize,
) {
    let cols = matrix.shape().cols();
    debug_assert_eq!(output.len(), matrix.shape().rows());
    #[cfg(not(feature = "parallel"))]
    let _ = parallel_work_threshold;
    #[cfg(feature = "parallel")]
    if matrix.dense_rows().len() >= parallel_work_threshold && output.len() > 1 {
        output
            .par_iter_mut()
            .zip(matrix.dense_rows().par_chunks_exact(cols))
            .for_each(|(value, weights)| *value = dot(weights, input));
        return;
    }
    for (value, weights) in output
        .iter_mut()
        .zip(matrix.dense_rows().chunks_exact(cols))
    {
        *value = dot(weights, input);
    }
}

const fn parallel_threshold() -> usize {
    #[cfg(feature = "parallel")]
    {
        PARALLEL_WORK_THRESHOLD
    }
    #[cfg(not(feature = "parallel"))]
    {
        usize::MAX
    }
}

const fn parallel_i8_threshold() -> usize {
    #[cfg(feature = "parallel")]
    {
        PARALLEL_I8_WORK_THRESHOLD
    }
    #[cfg(not(feature = "parallel"))]
    {
        usize::MAX
    }
}

fn fill_rows_checked(
    matrix: &TernaryProjectionMatrix,
    input: &[i64],
    output: &mut [i64],
) -> Result<(), AkitaError> {
    let cols = matrix.shape().cols();
    #[cfg(feature = "parallel")]
    if matrix.dense_rows().len() >= PARALLEL_WORK_THRESHOLD && output.len() > 1 {
        return output
            .par_iter_mut()
            .zip(matrix.dense_rows().par_chunks_exact(cols))
            .try_for_each(|(value, weights)| {
                *value = dot_i64_wide(weights, input)?;
                Ok(())
            });
    }
    for (value, weights) in output
        .iter_mut()
        .zip(matrix.dense_rows().chunks_exact(cols))
    {
        *value = dot_i64_wide(weights, input)?;
    }
    Ok(())
}

fn fill_rows_i128_checked(
    matrix: &TernaryProjectionMatrix,
    input: &[i128],
    output: &mut [i128],
) -> Result<(), AkitaError> {
    let cols = matrix.shape().cols();
    #[cfg(feature = "parallel")]
    if matrix.dense_rows().len() >= PARALLEL_WORK_THRESHOLD && output.len() > 1 {
        return output
            .par_iter_mut()
            .zip(matrix.dense_rows().par_chunks_exact(cols))
            .try_for_each(|(value, weights)| {
                *value = dot_i128_checked(weights, input)?;
                Ok(())
            });
    }
    for (value, weights) in output
        .iter_mut()
        .zip(matrix.dense_rows().chunks_exact(cols))
    {
        *value = dot_i128_checked(weights, input)?;
    }
    Ok(())
}

fn l1_fits_i64(input: &[i64]) -> bool {
    let limit = i64::MAX as u64;
    let mut sum = 0u64;
    for &value in input {
        let magnitude = value.unsigned_abs();
        let Some(next) = sum.checked_add(magnitude) else {
            return false;
        };
        if next > limit {
            return false;
        }
        sum = next;
    }
    true
}

fn l1_fits_i128(input: &[i128]) -> bool {
    let limit = i128::MAX as u128;
    let mut sum = 0u128;
    for &value in input {
        let magnitude = value.unsigned_abs();
        let Some(next) = sum.checked_add(magnitude) else {
            return false;
        };
        if next > limit {
            return false;
        }
        sum = next;
    }
    true
}

#[cfg_attr(all(target_arch = "aarch64", not(test)), allow(dead_code))]
#[inline]
fn dot_i8_scalar(weights: &[i8], input: &[i8]) -> i64 {
    weights
        .iter()
        .zip(input)
        .fold(0i64, |sum, (&weight, &value)| {
            sum + i64::from(weight) * i64::from(value)
        })
}

#[cfg_attr(all(target_arch = "aarch64", not(test)), allow(dead_code))]
#[inline]
fn dot_i16_scalar(weights: &[i8], input: &[i16]) -> i64 {
    weights
        .iter()
        .zip(input)
        .fold(0i64, |sum, (&weight, &value)| {
            sum + i64::from(weight) * i64::from(value)
        })
}

#[cfg_attr(all(target_arch = "aarch64", not(test)), allow(dead_code))]
#[inline]
fn dot_i32_scalar(weights: &[i8], input: &[i32]) -> i64 {
    weights
        .iter()
        .zip(input)
        .fold(0i64, |sum, (&weight, &value)| {
            sum + i64::from(weight) * i64::from(value)
        })
}

#[cfg_attr(all(target_arch = "aarch64", not(test)), allow(dead_code))]
#[inline]
fn dot_i64_scalar(weights: &[i8], input: &[i64]) -> i64 {
    weights
        .iter()
        .zip(input)
        .fold(0i64, |sum, (&weight, &value)| match weight {
            -1 => sum.wrapping_sub(value),
            1 => sum.wrapping_add(value),
            _ => sum,
        })
}

#[inline]
fn dot_i128_scalar(weights: &[i8], input: &[i128]) -> i128 {
    weights
        .iter()
        .zip(input)
        .fold(0i128, |sum, (&weight, &value)| match weight {
            -1 => sum.wrapping_sub(value),
            1 => sum.wrapping_add(value),
            _ => sum,
        })
}

fn dot_i64_wide(weights: &[i8], input: &[i64]) -> Result<i64, AkitaError> {
    let wide = weights
        .iter()
        .zip(input)
        .fold(0i128, |sum, (&weight, &value)| match weight {
            -1 => sum - i128::from(value),
            1 => sum + i128::from(value),
            _ => sum,
        });
    i64::try_from(wide).map_err(|_| projection_overflow())
}

fn dot_i128_checked(weights: &[i8], input: &[i128]) -> Result<i128, AkitaError> {
    weights
        .iter()
        .zip(input)
        .try_fold(0i128, |sum, (&weight, &value)| {
            let next = match weight {
                -1 => sum.checked_sub(value),
                1 => sum.checked_add(value),
                _ => Some(sum),
            };
            next.ok_or_else(projection_overflow)
        })
}

fn projection_overflow() -> AkitaError {
    AkitaError::InvalidInput("ternary integer projection overflow".into())
}

fn selected_i8_dot() -> Dot<i8> {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::selected_i8_dot()
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return x86_64::dot_i8_avx2_dispatch;
        }
        dot_i8_scalar
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_i8_scalar
    }
}

fn selected_i16_dot() -> Dot<i16> {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::dot_i16_neon_dispatch
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return x86_64::dot_i16_avx2_dispatch;
        }
        dot_i16_scalar
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_i16_scalar
    }
}

fn selected_i32_dot() -> Dot<i32> {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::dot_i32_neon_dispatch
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return x86_64::dot_i32_avx2_dispatch;
        }
        dot_i32_scalar
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_i32_scalar
    }
}

fn selected_i64_dot() -> Dot<i64> {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::dot_i64_neon_dispatch
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return x86_64::dot_i64_avx2_dispatch;
        }
        dot_i64_scalar
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_i64_scalar
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(crate) fn scalar_i8(weights: &[i8], input: &[i8]) -> i64 {
        dot_i8_scalar(weights, input)
    }

    pub(crate) fn scalar_i16(weights: &[i8], input: &[i16]) -> i64 {
        dot_i16_scalar(weights, input)
    }

    pub(crate) fn scalar_i32(weights: &[i8], input: &[i32]) -> i64 {
        dot_i32_scalar(weights, input)
    }

    pub(crate) fn scalar_i64(weights: &[i8], input: &[i64]) -> i64 {
        dot_i64_scalar(weights, input)
    }
}
