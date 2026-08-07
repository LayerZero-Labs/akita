//! Runtime-selected AArch64 NEON arithmetic over coefficient-oriented slices.

use super::neon::{PackedFp32Neon, PackedFp64Neon};
use super::runtime_common::{
    compute_class_coded_affine_polynomial_round_packed,
    compute_compact_affine_product_round_packed, compute_product_round_fp_ext2_fp64_packed,
    compute_product_round_packed, compute_weighted_affine_polynomial_round_packed,
    compute_weighted_affine_product_round_packed,
    fold_and_compute_product_round_fp_ext2_fp64_packed, fold_and_compute_product_round_packed,
    fold_and_compute_sparse_affine_polynomial_round_packed, fold_fp_ext2_fp64_packed,
    fold_fp_ext4_fp32_packed,
};
use crate::{Fp32, Fp64, FpExt2, FpExt2Config, FpExt4};

/// Fold fp32 quartic-extension slices four rows at a time with NEON.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn fold_fp_ext4_fp32_neon<const P: u32>(
    left: [&mut [Fp32<P>]; 4],
    right: [&[Fp32<P>]; 4],
    challenge: FpExt4<Fp32<P>>,
) {
    // SAFETY: this target function enables the feature required by the packed
    // backend; the shared traversal validates every slice before access.
    unsafe { fold_fp_ext4_fp32_packed::<P, PackedFp32Neon<P>>(left, right, challenge) };
}

/// Compute an fp32 quartic-extension product round four rows at a time.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn compute_product_round_fp_ext4_fp32_neon<const P: u32>(
    witness_0: [&[Fp32<P>]; 4],
    witness_1: [&[Fp32<P>]; 4],
    factor_0: [&[Fp32<P>]; 4],
    factor_1: [&[Fp32<P>]; 4],
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    // SAFETY: this target function enables the feature required by the packed
    // backend; the shared traversal validates every slice before access.
    unsafe {
        compute_product_round_packed::<P, PackedFp32Neon<P>>(
            witness_0, witness_1, factor_0, factor_1,
        )
    }
}

/// Compute an equality-weighted affine-product round four rows at a time.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn compute_weighted_affine_product_round_fp_ext4_fp32_neon<
    const P: u32,
    const LANES: usize,
>(
    left: [[&[Fp32<P>]; 4]; LANES],
    right: [[&[Fp32<P>]; 4]; LANES],
    equality: [&[Fp32<P>]; 4],
    arity: usize,
    parent_weights: &[FpExt4<Fp32<P>>],
) -> [FpExt4<Fp32<P>>; 5] {
    // SAFETY: the target feature matches the packed backend, and the shared
    // traversal validates every slice before reading it.
    unsafe {
        compute_weighted_affine_product_round_packed::<P, PackedFp32Neon<P>, LANES>(
            left,
            right,
            equality,
            arity,
            parent_weights,
        )
    }
}

/// Compute an equality-weighted polynomial-composition round with NEON.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn compute_weighted_affine_polynomial_round_fp_ext4_fp32_neon<const P: u32>(
    left: [&[Fp32<P>]; 4],
    right: [&[Fp32<P>]; 4],
    equality: [&[Fp32<P>]; 4],
    polynomial_coefficients: &[FpExt4<Fp32<P>>],
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        compute_weighted_affine_polynomial_round_packed::<P, PackedFp32Neon<P>>(
            left,
            right,
            equality,
            polynomial_coefficients,
        )
    }
}

/// Compute a compact class-indexed product prefix four rows at a time.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn compute_compact_affine_product_round_fp_ext4_fp32_neon<
    const P: u32,
    const LANES: usize,
>(
    ordered_pair_indices: &[u16],
    folded_pair_rows: &[[FpExt4<Fp32<P>>; LANES]],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    arity: usize,
    parent_weights: &[FpExt4<Fp32<P>>],
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        compute_compact_affine_product_round_packed::<P, PackedFp32Neon<P>, LANES>(
            ordered_pair_indices,
            folded_pair_rows,
            first_equality,
            second_equality,
            arity,
            parent_weights,
        )
    }
}

/// Compute a class-coded polynomial round with NEON.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn compute_class_coded_affine_polynomial_round_fp_ext4_fp32_neon<const P: u32>(
    class_codes: &[u16],
    class_values: &[FpExt4<Fp32<P>>],
    class_taylor_coefficients: &[[FpExt4<Fp32<P>>; 4]],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        compute_class_coded_affine_polynomial_round_packed::<P, PackedFp32Neon<P>>(
            class_codes,
            class_values,
            class_taylor_coefficients,
            first_equality,
            second_equality,
            degree,
        )
    }
}

/// Fold a sparse-prefix value table and compute its next round with NEON.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn fold_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_neon<const P: u32>(
    values: &[FpExt4<Fp32<P>>],
    folded_values: &mut [FpExt4<Fp32<P>>],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    challenge: FpExt4<Fp32<P>>,
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        fold_and_compute_sparse_affine_polynomial_round_packed::<P, PackedFp32Neon<P>>(
            values,
            folded_values,
            first_equality,
            second_equality,
            challenge,
            degree,
        )
    }
}

/// Fold two fp32 quartic-extension tables and compute the next product round.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn fold_and_compute_product_round_fp_ext4_fp32_neon<const P: u32>(
    witness_left: [&mut [Fp32<P>]; 4],
    witness_right: [&[Fp32<P>]; 4],
    factor_left: [&mut [Fp32<P>]; 4],
    factor_right: [&[Fp32<P>]; 4],
    challenge: FpExt4<Fp32<P>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    // SAFETY: this target function enables the feature required by the packed
    // backend; the shared traversal validates every slice before access.
    unsafe {
        fold_and_compute_product_round_packed::<P, PackedFp32Neon<P>>(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    }
}

/// Fold fp64 quadratic-extension slices two rows at a time with NEON.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn fold_fp_ext2_fp64_neon<const P: u64, C>(
    left: [&mut [Fp64<P>]; 2],
    right: [&[Fp64<P>]; 2],
    challenge: FpExt2<Fp64<P>, C>,
) where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    // SAFETY: this target function enables the feature required by the packed
    // backend; the shared traversal validates every slice before access.
    unsafe { fold_fp_ext2_fp64_packed::<P, C, PackedFp64Neon<P>>(left, right, challenge) };
}

/// Compute an fp64 quadratic-extension product round two rows at a time.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn compute_product_round_fp_ext2_fp64_neon<const P: u64, C>(
    witness_0: [&[Fp64<P>]; 2],
    witness_1: [&[Fp64<P>]; 2],
    factor_0: [&[Fp64<P>]; 2],
    factor_1: [&[Fp64<P>]; 2],
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    // SAFETY: this target function enables the feature required by the packed
    // backend; the shared traversal validates every slice before access.
    unsafe {
        compute_product_round_fp_ext2_fp64_packed::<P, C, PackedFp64Neon<P>>(
            witness_0, witness_1, factor_0, factor_1,
        )
    }
}

/// Fold two fp64 quadratic-extension tables and compute the next product round.
///
/// # Safety
///
/// The caller must establish that NEON is available on the current CPU.
#[target_feature(enable = "neon")]
pub unsafe fn fold_and_compute_product_round_fp_ext2_fp64_neon<const P: u64, C>(
    witness_left: [&mut [Fp64<P>]; 2],
    witness_right: [&[Fp64<P>]; 2],
    factor_left: [&mut [Fp64<P>]; 2],
    factor_right: [&[Fp64<P>]; 2],
    challenge: FpExt2<Fp64<P>, C>,
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    // SAFETY: this target function enables the feature required by the packed
    // backend; the shared traversal validates every slice before access.
    unsafe {
        fold_and_compute_product_round_fp_ext2_fp64_packed::<P, C, PackedFp64Neon<P>>(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    }
}
