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
use jolt_field::{CanonicalEncoding, Field};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use std::sync::atomic::{AtomicBool, Ordering};
use std::{mem, sync::OnceLock};

pub use mle::{
    build_block_projection_weight_table, build_ternary_column_weights, eval_block_tensor_mle,
    eval_power_of_two_block_diagonal_mle, eval_power_of_two_block_projection_reduction_factor,
    eval_ternary_matrix_mle, eval_ternary_matrix_mle_from_eq_tables,
};

/// Maximum storage accepted for one materialized JL allocation.
///
/// For a matrix this covers both packed Rademacher sign planes and the optional
/// lazily derived signed-byte compute plane. It also bounds projection outputs
/// and partial MLE tables allocated by this module. Protocol plans may choose
/// smaller limits.
pub const MAX_MATERIALIZED_JL_BYTES: usize = 1 << 30;

const DENSE_TRANSPOSE_GROUPS_PER_TILE: usize = 16;
#[cfg(target_arch = "aarch64")]
// Measured at the 256-by-16K cold crossover; keep tiny matrices on their existing path.
const COLD_CENTERED_FIELD_MIN_WORK: usize = 1 << 22;
const TERNARY_NIBBLE_DECODE: [u32; 256] = ternary_nibble_decode();

const fn ternary_nibble_decode() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut encoded = 0usize;
    while encoded < table.len() {
        let mut decoded = [0u8; 4];
        let mut lane = 0usize;
        while lane < 4 {
            let first = (encoded >> lane) & 1;
            let second = (encoded >> (lane + 4)) & 1;
            decoded[lane] = match first + second {
                0 => -1,
                2 => 1,
                _ => 0,
            } as i8 as u8;
            lane += 1;
        }
        table[encoded] = u32::from_ne_bytes(decoded);
        encoded += 1;
    }
    table
}

/// Checked shape of one materialized balanced-ternary matrix.
///
/// Each Rademacher plane is tiled by four adjacent columns and two adjacent
/// rows. A byte stores one four-bit sign selector for each row, with the even
/// row in the low nibble. This is the canonical projection layout: selectors
/// for every row are contiguous for a fixed four-column group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TernaryProjectionShape {
    rows: usize,
    cols: usize,
    col_groups: usize,
    row_pairs: usize,
    plane_len: usize,
    dense_len: usize,
}

impl TernaryProjectionShape {
    /// Construct a nonempty packed matrix shape.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] if a dimension is zero, any size
    /// computation overflows, or the packed sign planes plus the compute plane
    /// exceed the materialized matrix budget.
    pub fn new(rows: usize, cols: usize) -> Result<Self, AkitaError> {
        if rows == 0 || cols == 0 {
            return Err(AkitaError::InvalidInput(
                "ternary projection matrix dimensions must be nonzero".into(),
            ));
        }
        let col_groups = checked::div_ceil(cols, 4)
            .ok_or_else(|| AkitaError::InvalidInput("ternary column-group overflow".into()))?;
        let row_pairs = checked::div_ceil(rows, 2)
            .ok_or_else(|| AkitaError::InvalidInput("ternary row-pair overflow".into()))?;
        let plane_len = checked::product([col_groups, row_pairs])
            .ok_or_else(|| AkitaError::InvalidInput("ternary matrix size overflow".into()))?;
        let packed_len = checked::product([2, plane_len])
            .ok_or_else(|| AkitaError::InvalidInput("ternary matrix size overflow".into()))?;
        let dense_len = checked::product([rows, cols])
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
            col_groups,
            row_pairs,
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

    /// Number of four-column selector groups.
    #[must_use]
    pub const fn col_groups(self) -> usize {
        self.col_groups
    }

    /// Number of row pairs packed into each column group.
    #[must_use]
    pub const fn row_pairs(self) -> usize {
        self.row_pairs
    }

    /// Packed bytes in one bitplane.
    #[must_use]
    pub const fn plane_len(self) -> usize {
        self.plane_len
    }

    /// Total packed bytes in the two Rademacher sign planes.
    #[must_use]
    pub const fn packed_len(self) -> usize {
        self.plane_len * 2
    }

    /// Signed-byte entries in the row-major compute plane if materialized.
    #[must_use]
    pub const fn dense_len(self) -> usize {
        self.dense_len
    }

    /// Maximum bytes retained after the optional compute plane is materialized.
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

    /// Mask selecting live columns in the final four-column selector.
    #[must_use]
    pub const fn final_selector_live_mask(self) -> u8 {
        let tail = self.cols & 3;
        if tail == 0 {
            0x0f
        } else {
            ((1u16 << tail) - 1) as u8
        }
    }
}

/// Balanced-ternary matrix with packed Rademacher signs and a lazy compute plane.
///
/// A set bit denotes `+1` and a clear bit denotes `-1` in each plane. The
/// balanced-ternary entry is half the sum of the two signs, giving the exact
/// distribution `Pr[0] = 1/2` and `Pr[-1] = Pr[+1] = 1/4`. Padding bits are
/// canonical zeroes and are never interpreted as live signs. A derived
/// row-major signed-byte plane is materialized when repeated projection makes
/// that representation profitable. On AArch64 and AVX2-only x86-64, the first
/// eligible projection scans the canonical planes directly and leaves the lazy
/// dense cache empty; a later projection materializes the dense plane for its
/// faster hot kernel. AVX-512 lookup kernels remain packed.
/// Transcript and MLE semantics use only the canonical sign planes and never
/// pay that cost.
#[derive(Debug)]
pub struct TernaryProjectionMatrix {
    shape: TernaryProjectionShape,
    first_signs: Box<[u8]>,
    second_signs: Box<[u8]>,
    dense: OnceLock<Result<Box<[i8]>, AkitaError>>,
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    cold_packed_projection_used: AtomicBool,
}

impl Clone for TernaryProjectionMatrix {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape,
            first_signs: self.first_signs.clone(),
            second_signs: self.second_signs.clone(),
            dense: OnceLock::new(),
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            cold_packed_projection_used: AtomicBool::new(false),
        }
    }
}

impl PartialEq for TernaryProjectionMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.first_signs == other.first_signs
            && self.second_signs == other.second_signs
    }
}

impl Eq for TernaryProjectionMatrix {}

impl TernaryProjectionMatrix {
    /// Construct a matrix from canonical packed Rademacher sign planes.
    ///
    /// # Errors
    ///
    /// Returns an error if a plane length disagrees with the shape or any
    /// padded row or column bit is nonzero.
    pub fn from_rademacher_bitplanes(
        shape: TernaryProjectionShape,
        first_signs: Vec<u8>,
        second_signs: Vec<u8>,
    ) -> Result<Self, AkitaError> {
        for plane in [&first_signs, &second_signs] {
            if plane.len() != shape.plane_len() {
                return Err(AkitaError::InvalidSize {
                    expected: shape.plane_len(),
                    actual: plane.len(),
                });
            }
        }
        let final_selector_padding = !shape.final_selector_live_mask() & 0x0f;
        for plane in [&first_signs, &second_signs] {
            if shape.rows() & 1 != 0
                && plane
                    .chunks_exact(shape.row_pairs())
                    .any(|group| group[shape.row_pairs() - 1] & 0xf0 != 0)
            {
                return Err(AkitaError::InvalidInput(
                    "Rademacher plane has nonzero padded-row bits".into(),
                ));
            }
            if final_selector_padding != 0 {
                let final_group_start = (shape.col_groups() - 1) * shape.row_pairs();
                let padding_mask = final_selector_padding | (final_selector_padding << 4);
                if plane[final_group_start..]
                    .iter()
                    .any(|&byte| byte & padding_mask != 0)
                {
                    return Err(AkitaError::InvalidInput(
                        "Rademacher plane has nonzero padded-column bits".into(),
                    ));
                }
            }
        }
        Ok(Self {
            shape,
            first_signs: first_signs.into_boxed_slice(),
            second_signs: second_signs.into_boxed_slice(),
            dense: OnceLock::new(),
            #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
            cold_packed_projection_used: AtomicBool::new(false),
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
        Ok(self.entry_unchecked(row, col))
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
        mle::contract_columns_to_rows(self, input, &mut output)?;
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
            mle::contract_columns_to_rows(
                self,
                block_input,
                &mut output[output_start..output_start + self.shape.rows()],
            )?;
        }
        Ok(output)
    }

    pub(super) fn entry_unchecked(&self, row: usize, col: usize) -> i8 {
        let group = col >> 2;
        let row_pair = row >> 1;
        let shift = ((row & 1) << 2) | (col & 3);
        let index = group * self.shape.row_pairs() + row_pair;
        let first = (self.first_signs[index] >> shift) & 1;
        let second = (self.second_signs[index] >> shift) & 1;
        match first + second {
            0 => -1,
            2 => 1,
            _ => 0,
        }
    }

    pub(super) fn sign_groups_unchecked(&self, group: usize) -> (&[u8], &[u8]) {
        let start = group * self.shape.row_pairs();
        let end = start + self.shape.row_pairs();
        (
            &self.first_signs[start..end],
            &self.second_signs[start..end],
        )
    }

    pub(super) fn dense_rows(&self) -> Result<&[i8], AkitaError> {
        match self.dense.get_or_init(|| self.build_dense()) {
            Ok(dense) => Ok(dense),
            Err(error) => Err(error.clone()),
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    pub(super) fn take_cold_packed_projection(&self) -> bool {
        self.dense.get().is_none()
            && !self
                .cold_packed_projection_used
                .swap(true, Ordering::Relaxed)
    }

    fn build_dense(&self) -> Result<Box<[i8]>, AkitaError> {
        let shape = self.shape;
        let mut dense = try_zeroed_vec(shape.dense_len(), 0i8)?;
        for group_base in (0..shape.col_groups()).step_by(DENSE_TRANSPOSE_GROUPS_PER_TILE) {
            let groups = (shape.col_groups() - group_base).min(DENSE_TRANSPOSE_GROUPS_PER_TILE);
            let col_base = group_base * 4;
            for row_pair in 0..shape.row_pairs() {
                let even = row_pair * 2;
                let even_start = even * shape.cols() + col_base;
                let odd = even + 1;
                let odd_start = odd * shape.cols() + col_base;
                for local_group in 0..groups {
                    let index = (group_base + local_group) * shape.row_pairs() + row_pair;
                    let first = self.first_signs[index];
                    let second = self.second_signs[index];
                    let even_code = usize::from((first & 0x0f) | ((second & 0x0f) << 4));
                    let odd_code = usize::from((first >> 4) | (second & 0xf0));
                    let offset = local_group * 4;
                    let live = (shape.cols() - (col_base + offset)).min(4);
                    if live == 4 {
                        // SAFETY: the checked shape and `live == 4` establish
                        // that each four-byte store remains inside its row and
                        // the allocated dense plane. Unaligned stores are used
                        // because row widths need not be word-aligned.
                        unsafe {
                            std::ptr::write_unaligned(
                                dense.as_mut_ptr().add(even_start + offset).cast::<u32>(),
                                TERNARY_NIBBLE_DECODE[even_code],
                            );
                            if odd < shape.rows() {
                                std::ptr::write_unaligned(
                                    dense.as_mut_ptr().add(odd_start + offset).cast::<u32>(),
                                    TERNARY_NIBBLE_DECODE[odd_code],
                                );
                            }
                        }
                    } else {
                        let even_decoded = TERNARY_NIBBLE_DECODE[even_code].to_ne_bytes();
                        let odd_decoded = TERNARY_NIBBLE_DECODE[odd_code].to_ne_bytes();
                        for lane in 0..live {
                            dense[even_start + offset + lane] = even_decoded[lane] as i8;
                            if odd < shape.rows() {
                                dense[odd_start + offset + lane] = odd_decoded[lane] as i8;
                            }
                        }
                    }
                }
            }
        }
        Ok(dense.into_boxed_slice())
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

/// Recover the odd base-field modulus as a checked `u128`.
pub fn base_field_modulus<F: Field + CanonicalEncoding>() -> Result<u128, AkitaError> {
    let modulus = (-F::one())
        .to_u128_checked()
        .and_then(|minus_one| minus_one.checked_add(1))
        .ok_or_else(|| {
            AkitaError::InvalidInput("JL base-field modulus does not fit u128".into())
        })?;
    if modulus <= 2 || modulus & 1 == 0 {
        return Err(AkitaError::InvalidInput(
            "JL base-field modulus must be an odd integer greater than two".into(),
        ));
    }
    Ok(modulus)
}

/// Reject a signed coordinate outside the base field's unique centered window.
pub fn validate_centered_i128<F: Field + CanonicalEncoding>(value: i128) -> Result<(), AkitaError> {
    let half_modulus = base_field_modulus::<F>()? / 2;
    validate_centered_with_half_modulus(value, half_modulus)
}

/// Convert one base-field value to its unique centered signed representative.
pub fn centered_i128_from_field<F: Field + CanonicalEncoding>(
    value: F,
) -> Result<i128, AkitaError> {
    let modulus = base_field_modulus::<F>()?;
    centered_i128_from_field_with_modulus(value, modulus)
}

fn validate_centered_with_half_modulus(value: i128, half_modulus: u128) -> Result<(), AkitaError> {
    if value.unsigned_abs() > half_modulus {
        return Err(AkitaError::InvalidInput(
            "JL coordinate is outside the canonical centered field window".into(),
        ));
    }
    Ok(())
}

fn centered_i128_from_field_with_modulus<F: Field + CanonicalEncoding>(
    value: F,
    modulus: u128,
) -> Result<i128, AkitaError> {
    let residue = value.to_u128_checked().ok_or_else(|| {
        AkitaError::InvalidInput("JL projected value is not a base-field scalar".into())
    })?;
    centered_i128_from_residue(residue, modulus)
}

fn centered_i128_from_residue(residue: u128, modulus: u128) -> Result<i128, AkitaError> {
    if residue >= modulus {
        return Err(AkitaError::InvalidInput(
            "JL canonical residue exceeds its modulus".into(),
        ));
    }
    let magnitude = if residue <= modulus / 2 {
        return i128::try_from(residue)
            .map_err(|_| AkitaError::InvalidInput("JL centered coordinate overflow".into()));
    } else {
        modulus - residue
    };
    i128::try_from(magnitude)
        .map(|centered| -centered)
        .map_err(|_| AkitaError::InvalidInput("JL centered coordinate overflow".into()))
}

fn centered_i128_from_i64(value: i64, modulus: u128) -> Result<i128, AkitaError> {
    let magnitude = u128::from(value.unsigned_abs());
    if magnitude <= modulus / 2 {
        return Ok(i128::from(value));
    }
    let reduced = magnitude % modulus;
    let residue = if value >= 0 || reduced == 0 {
        reduced
    } else {
        modulus - reduced
    };
    centered_i128_from_residue(residue, modulus)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CenteredProjectionKernel {
    I8,
    I16,
    I32,
    Field,
}

fn centered_projection_kernel(
    input: &[i128],
    cols: usize,
    half_modulus: u128,
) -> Result<CenteredProjectionKernel, AkitaError> {
    let mut fits_i8 = true;
    let mut fits_i16 = true;
    let mut fits_i32 = true;
    let mut outputs_fit_i64 = true;
    for block in input.chunks_exact(cols) {
        let mut l1 = 0u128;
        for &coordinate in block {
            validate_centered_with_half_modulus(coordinate, half_modulus)?;
            fits_i8 &= i8::try_from(coordinate).is_ok();
            fits_i16 &= i16::try_from(coordinate).is_ok();
            fits_i32 &= i32::try_from(coordinate).is_ok();
            if outputs_fit_i64 {
                match l1.checked_add(coordinate.unsigned_abs()) {
                    Some(next) if next <= i64::MAX as u128 => l1 = next,
                    _ => outputs_fit_i64 = false,
                }
            }
        }
    }
    Ok(if !outputs_fit_i64 {
        CenteredProjectionKernel::Field
    } else if fits_i8 {
        CenteredProjectionKernel::I8
    } else if fits_i16 {
        CenteredProjectionKernel::I16
    } else if fits_i32 {
        CenteredProjectionKernel::I32
    } else {
        CenteredProjectionKernel::Field
    })
}

impl TernaryProjectionMatrix {
    /// Apply `I_blocks tensor self` modulo the base field and center every output.
    ///
    /// Inputs must already use the unique centered representation. Reducing and
    /// centering after each layer prevents raw-integer aliases and intermediate
    /// overflow from changing the iterated projection semantics.
    pub fn project_centered_i128_blocks<F: Field + CanonicalEncoding>(
        &self,
        input: &[i128],
    ) -> Result<Vec<i128>, AkitaError> {
        if input.is_empty() || !input.len().is_multiple_of(self.shape.cols()) {
            return Err(AkitaError::InvalidSize {
                expected: self.shape.cols(),
                actual: input.len(),
            });
        }
        let blocks = input.len() / self.shape.cols();
        let output_len = checked::product([blocks, self.shape.rows()])
            .ok_or_else(|| AkitaError::InvalidInput("JL projection output overflow".into()))?;
        let modulus = base_field_modulus::<F>()?;
        let half_modulus = modulus / 2;
        let kernel = centered_projection_kernel(input, self.shape.cols(), half_modulus)?;
        #[cfg(target_arch = "aarch64")]
        let kernel = if matches!(
            kernel,
            CenteredProjectionKernel::I8 | CenteredProjectionKernel::I16
        ) && self.prefer_field_for_cold_single_block::<F>(blocks)
        {
            CenteredProjectionKernel::Field
        } else {
            kernel
        };
        match kernel {
            CenteredProjectionKernel::I8 => {
                return self.project_centered_narrow_blocks::<i8>(input, output_len, modulus);
            }
            CenteredProjectionKernel::I16 => {
                return self.project_centered_narrow_blocks::<i16>(input, output_len, modulus);
            }
            CenteredProjectionKernel::I32 => {
                return self.project_centered_narrow_blocks::<i32>(input, output_len, modulus);
            }
            CenteredProjectionKernel::Field => {}
        }

        let mut field_input = try_zeroed_vec(input.len(), F::zero())?;
        for (&coordinate, embedded) in input.iter().zip(&mut field_input) {
            *embedded = F::from_i128(coordinate);
        }
        let projected = self.project_field_blocks(&field_input)?;
        let mut centered = try_zeroed_vec(projected.len(), 0i128)?;
        for (coordinate, output) in projected.into_iter().zip(&mut centered) {
            *output = centered_i128_from_field_with_modulus(coordinate, modulus)?;
        }
        Ok(centered)
    }

    #[cfg(target_arch = "aarch64")]
    fn prefer_field_for_cold_single_block<F: Field>(&self, blocks: usize) -> bool {
        #[cfg(feature = "parallel")]
        let parallel = rayon::current_num_threads() > 1
            && self
                .shape
                .field_contraction_scratch_len()
                .is_ok_and(|elements| elements != 0);
        #[cfg(not(feature = "parallel"))]
        let parallel = false;
        blocks == 1
            && self.shape.dense_len() >= COLD_CENTERED_FIELD_MIN_WORK
            && self.dense.get().is_none()
            && (parallel || mem::size_of::<F>() <= mem::size_of::<u64>())
    }

    fn project_centered_narrow_blocks<T>(
        &self,
        input: &[i128],
        output_len: usize,
        modulus: u128,
    ) -> Result<Vec<i128>, AkitaError>
    where
        T: ProjectionInput + Default + TryFrom<i128>,
    {
        let mut narrow = try_zeroed_vec(input.len(), T::default())?;
        for (&coordinate, converted) in input.iter().zip(&mut narrow) {
            *converted = T::try_from(coordinate).map_err(|_| {
                AkitaError::InvalidInput("JL narrow projection selection is inconsistent".into())
            })?;
        }
        let mut projected = try_zeroed_vec(output_len, 0i64)?;
        projection::project_blocks(self, &narrow, &mut projected)?;
        let mut centered = try_zeroed_vec(output_len, 0i128)?;
        for (coordinate, output) in projected.into_iter().zip(&mut centered) {
            *output = centered_i128_from_i64(coordinate, modulus)?;
        }
        Ok(centered)
    }

    /// Materialize the literal upper-left rectangular prefix of this matrix.
    ///
    /// This preserves two-dimensional prefix semantics even when the requested
    /// row count is smaller than the envelope row count; truncating a flat
    /// row-major representation would not.
    pub fn upper_left_prefix(&self, rows: usize, cols: usize) -> Result<Self, AkitaError> {
        if rows == 0 || cols == 0 || rows > self.shape.rows() || cols > self.shape.cols() {
            return Err(AkitaError::InvalidInput(format!(
                "ternary prefix {rows} by {cols} is outside envelope {} by {}",
                self.shape.rows(),
                self.shape.cols()
            )));
        }
        let shape = TernaryProjectionShape::new(rows, cols)?;
        let mut first = try_zeroed_vec(shape.plane_len(), 0u8)?;
        let mut second = try_zeroed_vec(shape.plane_len(), 0u8)?;
        copy_upper_left_plane_prefix(&self.first_signs, self.shape, &mut first, shape)?;
        copy_upper_left_plane_prefix(&self.second_signs, self.shape, &mut second, shape)?;
        Self::from_rademacher_bitplanes(shape, first, second)
    }
}

fn copy_upper_left_plane_prefix(
    source: &[u8],
    source_shape: TernaryProjectionShape,
    target: &mut [u8],
    target_shape: TernaryProjectionShape,
) -> Result<(), AkitaError> {
    for (source_group, target_group) in source
        .chunks_exact(source_shape.row_pairs())
        .zip(target.chunks_exact_mut(target_shape.row_pairs()))
    {
        let prefix = source_group
            .get(..target_shape.row_pairs())
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "ternary matrix prefix row geometry is inconsistent".into(),
                )
            })?;
        target_group.copy_from_slice(prefix);
        if target_shape.rows() & 1 != 0 {
            let final_pair = target_group.last_mut().ok_or_else(|| {
                AkitaError::InvalidInput("ternary matrix prefix has no live row pair".into())
            })?;
            *final_pair &= 0x0f;
        }
    }
    let live = target_shape.final_selector_live_mask();
    let live_pair = live | (live << 4);
    let final_group = target
        .chunks_exact_mut(target_shape.row_pairs())
        .last()
        .ok_or_else(|| {
            AkitaError::InvalidInput("ternary matrix prefix has no column group".into())
        })?;
    for byte in final_group {
        *byte &= live_pair;
    }
    Ok(())
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
