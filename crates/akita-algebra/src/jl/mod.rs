//! Seed-independent algebra for balanced-ternary JL projections.
//!
//! The Fiat--Shamir law that samples these matrices lives in
//! `akita-challenges`. This module owns only the checked packed matrix,
//! projection arithmetic, and multilinear evaluations shared by prover and
//! verifier code.

mod mle;
mod projection;

#[cfg(test)]
mod tests;

use akita_error::{checked, AkitaError};
use jolt_field::Field;
use std::mem;

pub use mle::{
    build_ternary_column_weights, eval_power_of_two_block_diagonal_mle, eval_ternary_matrix_mle,
    eval_ternary_matrix_mle_from_eq_tables,
};

/// Maximum storage accepted for one materialized JL allocation.
///
/// For a matrix this covers both balanced-ternary bitplanes and the signed-byte
/// compute plane. It also bounds projection outputs and partial MLE tables
/// allocated by this module. Protocol plans may choose smaller limits.
pub const MAX_MATERIALIZED_JL_BYTES: usize = 1 << 30;

const BYTE_LANES_ONE: u64 = 0x0101_0101_0101_0101;
const EXPANDED_BIT_MASKS: [u64; 256] = expanded_bit_masks();

const fn expanded_bit_masks() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut byte = 0usize;
    while byte < table.len() {
        let mut bit = 0usize;
        while bit < 8 {
            if byte & (1 << bit) != 0 {
                table[byte] |= 0xff << (bit * 8);
            }
            bit += 1;
        }
        byte += 1;
    }
    table
}

/// Checked shape of one materialized row-major balanced-ternary matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TernaryProjectionShape {
    rows: usize,
    cols: usize,
    row_bytes: usize,
    plane_len: usize,
    dense_len: usize,
}

impl TernaryProjectionShape {
    /// Construct a nonempty packed matrix shape.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] if a dimension is zero, any size
    /// computation overflows, or the packed planes plus the compute plane
    /// exceed the materialized matrix budget.
    pub fn new(rows: usize, cols: usize) -> Result<Self, AkitaError> {
        if rows == 0 || cols == 0 {
            return Err(AkitaError::InvalidInput(
                "ternary projection matrix dimensions must be nonzero".into(),
            ));
        }
        let row_bytes = checked::div_ceil(cols, 8)
            .ok_or_else(|| AkitaError::InvalidInput("ternary row width overflow".into()))?;
        let plane_len = checked::product([rows, row_bytes])
            .ok_or_else(|| AkitaError::InvalidInput("ternary matrix size overflow".into()))?;
        let dense_len = checked::product([rows, cols])
            .ok_or_else(|| AkitaError::InvalidInput("ternary matrix size overflow".into()))?;
        let packed_len = checked::product([2, plane_len])
            .ok_or_else(|| AkitaError::InvalidInput("ternary matrix size overflow".into()))?;
        let total_bytes = checked::sum([packed_len, dense_len])
            .ok_or_else(|| AkitaError::InvalidInput("ternary matrix size overflow".into()))?;
        if total_bytes > MAX_MATERIALIZED_JL_BYTES {
            return Err(AkitaError::InvalidInput(format!(
                "ternary matrix requires {total_bytes} materialized bytes, exceeding the budget of {MAX_MATERIALIZED_JL_BYTES} bytes"
            )));
        }
        Ok(Self {
            rows,
            cols,
            row_bytes,
            plane_len,
            dense_len,
        })
    }

    /// Number of matrix rows.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Number of live matrix columns.
    #[must_use]
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Packed bytes per row in one bitplane.
    #[must_use]
    pub const fn row_bytes(self) -> usize {
        self.row_bytes
    }

    /// Packed bytes in one bitplane.
    #[must_use]
    pub const fn plane_len(self) -> usize {
        self.plane_len
    }

    /// Total packed bytes in the nonzero and positive bitplanes.
    #[must_use]
    pub const fn packed_len(self) -> usize {
        self.plane_len * 2
    }

    /// Signed-byte entries in the row-major compute plane.
    #[must_use]
    pub const fn dense_len(self) -> usize {
        self.dense_len
    }

    /// Total bytes retained by the canonical and compute representations.
    #[must_use]
    pub const fn materialized_len(self) -> usize {
        self.packed_len() + self.dense_len
    }

    /// Boolean variables in the padded row MLE domain.
    pub fn row_num_vars(self) -> Result<usize, AkitaError> {
        checked::ceil_log2(self.rows)
            .ok_or_else(|| AkitaError::InvalidInput("ternary row domain overflow".into()))
    }

    /// Boolean variables in the padded column MLE domain.
    pub fn col_num_vars(self) -> Result<usize, AkitaError> {
        checked::ceil_log2(self.cols)
            .ok_or_else(|| AkitaError::InvalidInput("ternary column domain overflow".into()))
    }

    pub(super) fn row_domain_len(self) -> Result<usize, AkitaError> {
        checked::pow2(self.row_num_vars()?)
            .ok_or_else(|| AkitaError::InvalidInput("ternary row domain overflow".into()))
    }

    pub(super) fn col_domain_len(self) -> Result<usize, AkitaError> {
        checked::pow2(self.col_num_vars()?)
            .ok_or_else(|| AkitaError::InvalidInput("ternary column domain overflow".into()))
    }

    /// Mask selecting live columns in the final byte of every packed row.
    #[must_use]
    pub const fn final_byte_live_mask(self) -> u8 {
        let tail = self.cols & 7;
        if tail == 0 {
            u8::MAX
        } else {
            ((1u16 << tail) - 1) as u8
        }
    }
}

/// Row-major balanced-ternary matrix with canonical bitplanes and a compute plane.
///
/// A set nonzero bit marks an entry in `{-1, +1}`. A positive bit selects
/// `+1`; a clear positive bit selects `-1`. Positive bits must be a subset of
/// nonzero bits, and padding bits must be zero. A derived signed-byte plane is
/// retained for branch-free scalar and SIMD projection; transcript and MLE
/// semantics continue to use the canonical bitplanes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TernaryProjectionMatrix {
    shape: TernaryProjectionShape,
    nonzero: Box<[u8]>,
    positive: Box<[u8]>,
    dense: Box<[i8]>,
}

impl TernaryProjectionMatrix {
    /// Construct a matrix from canonical packed bitplanes.
    ///
    /// # Errors
    ///
    /// Returns an error if a plane length disagrees with the shape, a positive
    /// bit is set for a zero entry, or any row has nonzero padding bits.
    pub fn from_bitplanes(
        shape: TernaryProjectionShape,
        nonzero: Vec<u8>,
        positive: Vec<u8>,
    ) -> Result<Self, AkitaError> {
        for plane in [&nonzero, &positive] {
            if plane.len() != shape.plane_len() {
                return Err(AkitaError::InvalidSize {
                    expected: shape.plane_len(),
                    actual: plane.len(),
                });
            }
        }
        if nonzero
            .iter()
            .zip(&positive)
            .any(|(&nonzero_byte, &positive_byte)| positive_byte & !nonzero_byte != 0)
        {
            return Err(AkitaError::InvalidInput(
                "ternary positive bitplane is not a subset of the nonzero bitplane".into(),
            ));
        }
        let padding_mask = !shape.final_byte_live_mask();
        if padding_mask != 0 {
            for row in 0..shape.rows() {
                let final_index = checked::mul_add(row, shape.row_bytes(), shape.row_bytes() - 1)
                    .ok_or_else(|| {
                    AkitaError::InvalidInput("ternary row offset overflow".into())
                })?;
                if (nonzero[final_index] | positive[final_index]) & padding_mask != 0 {
                    return Err(AkitaError::InvalidInput(
                        "ternary matrix padding bits must be zero".into(),
                    ));
                }
            }
        }
        let mut dense = try_zeroed_vec(shape.dense_len(), 0i8)?;
        for row in 0..shape.rows() {
            let packed_start = row * shape.row_bytes();
            let dense_start = row * shape.cols();
            let full_bytes = shape.cols() / 8;
            for byte_index in 0..full_bytes {
                let nonzero_byte = nonzero[packed_start + byte_index];
                let positive_byte = positive[packed_start + byte_index];
                let nonzero_lanes = EXPANDED_BIT_MASKS[usize::from(nonzero_byte)];
                let positive_lanes = EXPANDED_BIT_MASKS[usize::from(positive_byte)];
                let signed_lanes =
                    (positive_lanes & BYTE_LANES_ONE) | (nonzero_lanes & !positive_lanes);
                let bytes = signed_lanes.to_le_bytes().map(|byte| byte as i8);
                let start = dense_start + byte_index * 8;
                dense[start..start + 8].copy_from_slice(&bytes);
            }
            if shape.cols() & 7 != 0 {
                let byte_index = full_bytes;
                let nonzero_byte = nonzero[packed_start + byte_index];
                let positive_byte = positive[packed_start + byte_index];
                let column_start = full_bytes * 8;
                let live = shape.cols() - column_start;
                for bit in 0..live {
                    let mask = 1u8 << bit;
                    dense[dense_start + column_start + bit] = if nonzero_byte & mask == 0 {
                        0
                    } else if positive_byte & mask != 0 {
                        1
                    } else {
                        -1
                    };
                }
            }
        }
        Ok(Self {
            shape,
            nonzero: nonzero.into_boxed_slice(),
            positive: positive.into_boxed_slice(),
            dense: dense.into_boxed_slice(),
        })
    }

    /// Checked matrix shape.
    #[must_use]
    pub const fn shape(&self) -> TernaryProjectionShape {
        self.shape
    }

    /// Read one entry as `-1`, `0`, or `1`.
    pub fn entry(&self, row: usize, col: usize) -> Result<i8, AkitaError> {
        if row >= self.shape.rows() || col >= self.shape.cols() {
            return Err(AkitaError::InvalidInput(format!(
                "ternary matrix entry ({row}, {col}) is outside {} by {}",
                self.shape.rows(),
                self.shape.cols()
            )));
        }
        let byte = checked::mul_add(row, self.shape.row_bytes(), col >> 3)
            .ok_or_else(|| AkitaError::InvalidInput("ternary entry offset overflow".into()))?;
        let bit = 1u8 << (col & 7);
        Ok(if self.nonzero[byte] & bit == 0 {
            0
        } else if self.positive[byte] & bit != 0 {
            1
        } else {
            -1
        })
    }

    /// Apply the matrix to one exact integer vector.
    ///
    /// Accumulation is checked so malformed or unsupported values return an
    /// error rather than wrapping.
    pub fn project_i128(&self, input: &[i128]) -> Result<Vec<i128>, AkitaError> {
        if input.len() != self.shape.cols() {
            return Err(AkitaError::InvalidSize {
                expected: self.shape.cols(),
                actual: input.len(),
            });
        }
        let mut output = try_zeroed_vec(self.shape.rows(), i128::default())?;
        projection::project_i128(self, input, &mut output)?;
        Ok(output)
    }

    /// Apply the matrix to one signed integer vector using the narrowest
    /// native input representation available to the caller.
    ///
    /// `i8`, `i16`, `i32`, and `i64` inputs use architecture-specific SIMD
    /// where available and return exact `i64` outputs. An output outside the
    /// signed 64-bit range is rejected. Use [`Self::project_i128`] when the
    /// protocol plan genuinely requires wider coordinates.
    pub fn project<T: ProjectionInput>(&self, input: &[T]) -> Result<Vec<i64>, AkitaError> {
        if input.len() != self.shape.cols() {
            return Err(AkitaError::InvalidSize {
                expected: self.shape.cols(),
                actual: input.len(),
            });
        }
        let mut output = try_zeroed_vec(self.shape.rows(), i64::default())?;
        <T as projection::private::Sealed>::project(self, input, &mut output)?;
        Ok(output)
    }

    /// Apply `I_blocks tensor self` to exact integer blocks.
    pub fn project_i128_blocks(&self, input: &[i128]) -> Result<Vec<i128>, AkitaError> {
        let blocks = checked::exact_div(input.len(), self.shape.cols()).ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "block projection input length {} is not divisible by matrix width {}",
                input.len(),
                self.shape.cols()
            ))
        })?;
        if blocks == 0 {
            return Err(AkitaError::InvalidInput(
                "block projection requires at least one input block".into(),
            ));
        }
        let output_len = checked::product([blocks, self.shape.rows()])
            .ok_or_else(|| AkitaError::InvalidInput("block projection size overflow".into()))?;
        let mut output = try_zeroed_vec(output_len, i128::default())?;
        for (block, block_input) in input.chunks_exact(self.shape.cols()).enumerate() {
            let output_start = block.checked_mul(self.shape.rows()).ok_or_else(|| {
                AkitaError::InvalidInput("block projection offset overflow".into())
            })?;
            projection::project_i128(
                self,
                block_input,
                &mut output[output_start..output_start + self.shape.rows()],
            )?;
        }
        Ok(output)
    }

    /// Apply `I_blocks tensor self` to narrow signed integer blocks.
    pub fn project_blocks<T: ProjectionInput>(&self, input: &[T]) -> Result<Vec<i64>, AkitaError> {
        let blocks = checked::exact_div(input.len(), self.shape.cols()).ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "block projection input length {} is not divisible by matrix width {}",
                input.len(),
                self.shape.cols()
            ))
        })?;
        if blocks == 0 {
            return Err(AkitaError::InvalidInput(
                "block projection requires at least one input block".into(),
            ));
        }
        let output_len = checked::product([blocks, self.shape.rows()])
            .ok_or_else(|| AkitaError::InvalidInput("block projection size overflow".into()))?;
        let mut output = try_zeroed_vec(output_len, i64::default())?;
        projection::project_blocks(self, input, &mut output)?;
        Ok(output)
    }

    /// Apply the matrix over a field.
    pub fn project_field<F: Field>(&self, input: &[F]) -> Result<Vec<F>, AkitaError> {
        if input.len() != self.shape.cols() {
            return Err(AkitaError::InvalidSize {
                expected: self.shape.cols(),
                actual: input.len(),
            });
        }
        let mut output = try_zeroed_vec(self.shape.rows(), F::zero())?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.project_field_row(row, input);
        }
        Ok(output)
    }

    /// Apply `I_blocks tensor self` over a field.
    pub fn project_field_blocks<F: Field>(&self, input: &[F]) -> Result<Vec<F>, AkitaError> {
        let blocks = checked::exact_div(input.len(), self.shape.cols()).ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "block projection input length {} is not divisible by matrix width {}",
                input.len(),
                self.shape.cols()
            ))
        })?;
        if blocks == 0 {
            return Err(AkitaError::InvalidInput(
                "block projection requires at least one input block".into(),
            ));
        }
        let output_len = checked::product([blocks, self.shape.rows()])
            .ok_or_else(|| AkitaError::InvalidInput("block projection size overflow".into()))?;
        let mut output = try_zeroed_vec(output_len, F::zero())?;
        for (block, block_input) in input.chunks_exact(self.shape.cols()).enumerate() {
            let output_start = block.checked_mul(self.shape.rows()).ok_or_else(|| {
                AkitaError::InvalidInput("block projection offset overflow".into())
            })?;
            for row in 0..self.shape.rows() {
                output[output_start + row] = self.project_field_row(row, block_input);
            }
        }
        Ok(output)
    }

    fn project_field_row<F: Field>(&self, row: usize, input: &[F]) -> F {
        let (nonzero, positive) = self.row_bitplanes_unchecked(row);
        let mut accumulator = F::zero();
        for (byte_index, (&nonzero_byte, &positive_byte)) in
            nonzero.iter().zip(positive).enumerate()
        {
            let mut live = nonzero_byte;
            while live != 0 {
                let bit = live.trailing_zeros() as usize;
                let col = byte_index * 8 + bit;
                if col < self.shape.cols() {
                    if positive_byte & (1u8 << bit) != 0 {
                        accumulator += input[col];
                    } else {
                        accumulator -= input[col];
                    }
                }
                live &= live - 1;
            }
        }
        accumulator
    }

    pub(super) fn row_bitplanes_unchecked(&self, row: usize) -> (&[u8], &[u8]) {
        let start = row * self.shape.row_bytes();
        let end = start + self.shape.row_bytes();
        (&self.nonzero[start..end], &self.positive[start..end])
    }

    pub(super) fn dense_rows(&self) -> &[i8] {
        &self.dense
    }
}

/// Signed input types supported by the architecture-specific JL kernels.
///
/// This trait is sealed; its public role is to let [`TernaryProjectionMatrix::project`]
/// preserve the caller's compact coefficient representation without separate
/// pass-through methods for every integer width.
pub trait ProjectionInput: projection::private::Sealed + Copy + Send + Sync {}

impl<T> ProjectionInput for T where T: projection::private::Sealed + Copy + Send + Sync {}

/// Exact squared Euclidean norm of signed integer coordinates.
///
/// # Errors
///
/// Returns an error if a square or the accumulated sum does not fit `u128`.
pub fn squared_l2_i128(values: &[i128]) -> Result<u128, AkitaError> {
    values.iter().try_fold(0u128, |sum, &value| {
        let magnitude = value.unsigned_abs();
        let square = magnitude.checked_mul(magnitude).ok_or_else(|| {
            AkitaError::InvalidInput("integer squared-norm coordinate overflow".into())
        })?;
        sum.checked_add(square)
            .ok_or_else(|| AkitaError::InvalidInput("integer squared-norm sum overflow".into()))
    })
}

pub(super) fn try_zeroed_vec<T: Clone>(len: usize, zero: T) -> Result<Vec<T>, AkitaError> {
    let bytes = len
        .checked_mul(mem::size_of::<T>().max(1))
        .ok_or_else(|| AkitaError::InvalidInput("ternary projection allocation overflow".into()))?;
    if bytes > MAX_MATERIALIZED_JL_BYTES {
        return Err(AkitaError::InvalidInput(format!(
            "ternary projection allocation requires {bytes} bytes, exceeding the materialized budget of {MAX_MATERIALIZED_JL_BYTES} bytes"
        )));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|_| {
        AkitaError::InvalidInput(format!(
            "ternary projection allocation failed for {len} elements"
        ))
    })?;
    output.resize(len, zero);
    Ok(output)
}
