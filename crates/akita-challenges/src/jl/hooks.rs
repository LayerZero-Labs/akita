//! Bench and differential hooks for the JL projection prototype.

#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use akita_field::AkitaError;

use super::kernels;
use super::{validate_digit_witness, JlImage, JlProjectionMatrix};

/// Project pre-centered digits with the scalar kernel, bypassing SIMD dispatch.
pub fn project_digits_scalar(
    matrix: &JlProjectionMatrix,
    digits: &[i32],
) -> Result<JlImage, AkitaError> {
    validate_digit_witness(digits, matrix.cols())?;
    let coords = kernels::project_rows_scalar(
        matrix.n_rows(),
        matrix.row_bytes(),
        matrix.packed_rows(),
        digits,
        matrix.cols(),
    );
    Ok(JlImage { coords })
}

/// Checked `i64` reference projection for tests and differential benches.
pub fn project_digits_reference(
    matrix: &JlProjectionMatrix,
    digits: &[i32],
) -> Result<JlImage, AkitaError> {
    validate_digit_witness(digits, matrix.cols())?;
    let centered: Vec<i64> = digits.iter().map(|&d| i64::from(d)).collect();
    let project_row = |row_idx: usize| {
        kernels::project_row_reference(matrix.row_slice(row_idx), &centered, matrix.cols())
    };
    let coords = if super::use_parallel_projection(matrix.n_rows(), matrix.cols()) {
        akita_field::cfg_into_iter!(0..matrix.n_rows())
            .map(project_row)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..matrix.n_rows())
            .map(project_row)
            .collect::<Result<Vec<_>, _>>()?
    };
    let coords: Vec<i32> = coords
        .into_iter()
        .map(|c| {
            i32::try_from(c).map_err(|_| {
                AkitaError::InvalidInput(
                    "JL reference coordinate exceeds i32 range for digit witness".to_string(),
                )
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(JlImage { coords })
}
