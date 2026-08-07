//! Runtime-selected x86 arithmetic over coefficient-oriented field slices.

use super::avx2::{PackedFp32Avx2, PackedFp64Avx2};
use super::avx512::{PackedFp32Avx512, PackedFp64Avx512};
use super::runtime_common::{
    compute_class_coded_affine_polynomial_round_packed,
    compute_compact_affine_product_round_packed, compute_product_round_fp_ext2_fp64_packed,
    compute_product_round_packed, compute_weighted_affine_polynomial_round_packed,
    compute_weighted_affine_product_round_packed,
    fold_and_compute_product_round_fp_ext2_fp64_packed, fold_and_compute_product_round_packed,
    fold_and_compute_sparse_affine_polynomial_round_packed,
    fold_class_coded_and_compute_sparse_affine_polynomial_round_packed, fold_fp_ext2_fp64_packed,
};
use super::{PackedField, PackedFpExt4};
use crate::{Fp32, Fp64, FpExt2, FpExt2Config, FpExt4};

type CoefficientSlices<'a, const P: u32> = [&'a [Fp32<P>]; 4];

/// Fold fp32 quartic-extension slices eight rows at a time with AVX2.
///
/// `left` contains both the even input and the output. `right` contains the odd
/// input. Every slice must have the same length, and that length must be a
/// multiple of eight.
///
/// # Panics
///
/// Panics if the slice lengths differ or are not a multiple of eight.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn fold_fp_ext4_fp32_avx2<const P: u32>(
    left: [&mut [Fp32<P>]; 4],
    right: [&[Fp32<P>]; 4],
    challenge: FpExt4<Fp32<P>>,
) {
    const WIDTH: usize = 8;
    let len = validate_fold_slices(&left, &right, WIDTH);
    let challenge = PackedFpExt4::<Fp32<P>, PackedFp32Avx2<P>>::broadcast(challenge);
    let left = left.map(|slice| slice.as_mut_ptr());
    let right = right.map(|slice| slice.as_ptr());

    for row in (0..len).step_by(WIDTH) {
        // SAFETY: validation above establishes a complete WIDTH-row chunk in
        // every input and output slice. Fp32 arrays have the same element
        // layout as their source slices.
        let even = unsafe {
            PackedFpExt4::new(std::array::from_fn(|coefficient| {
                PackedFp32Avx2(read_array::<_, WIDTH>(left[coefficient], row))
            }))
        };
        // SAFETY: the same validated chunk bound applies to every right slice.
        let odd = unsafe {
            PackedFpExt4::new(std::array::from_fn(|coefficient| {
                PackedFp32Avx2(read_array::<_, WIDTH>(right[coefficient], row))
            }))
        };
        let folded = even + (odd - even) * challenge;
        for (&output, packed) in left.iter().zip(folded.coeffs) {
            // SAFETY: validation establishes a complete output chunk, and the
            // source packed value is a distinct local array.
            unsafe {
                packed
                    .0
                    .as_ptr()
                    .copy_to_nonoverlapping(output.add(row), WIDTH)
            };
        }
    }
}

/// Fold fp32 quartic-extension slices sixteen rows at a time with AVX-512 IFMA.
///
/// `left` contains both the even input and the output. `right` contains the odd
/// input. Every slice must have the same length, and that length must be a
/// multiple of sixteen.
///
/// # Panics
///
/// Panics if the slice lengths differ or are not a multiple of sixteen.
///
/// # Safety
///
/// The caller must establish that AVX-512F, AVX-512DQ, and AVX-512IFMA are
/// available on the current CPU.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn fold_fp_ext4_fp32_avx512_ifma<const P: u32>(
    left: [&mut [Fp32<P>]; 4],
    right: [&[Fp32<P>]; 4],
    challenge: FpExt4<Fp32<P>>,
) {
    const WIDTH: usize = 16;
    let len = validate_fold_slices(&left, &right, WIDTH);
    let challenge = PackedFpExt4::<Fp32<P>, PackedFp32Avx512<P>>::broadcast(challenge);
    let left = left.map(|slice| slice.as_mut_ptr());
    let right = right.map(|slice| slice.as_ptr());

    for row in (0..len).step_by(WIDTH) {
        // SAFETY: validation above establishes a complete WIDTH-row chunk in
        // every input and output slice. Fp32 arrays have the same element
        // layout as their source slices.
        let even = unsafe {
            PackedFpExt4::new(std::array::from_fn(|coefficient| {
                PackedFp32Avx512(read_array::<_, WIDTH>(left[coefficient], row))
            }))
        };
        // SAFETY: the same validated chunk bound applies to every right slice.
        let odd = unsafe {
            PackedFpExt4::new(std::array::from_fn(|coefficient| {
                PackedFp32Avx512(read_array::<_, WIDTH>(right[coefficient], row))
            }))
        };
        let folded = even + (odd - even) * challenge;
        for (&output, packed) in left.iter().zip(folded.coeffs) {
            // SAFETY: validation establishes a complete output chunk, and the
            // source packed value is a distinct local array.
            unsafe {
                packed
                    .0
                    .as_ptr()
                    .copy_to_nonoverlapping(output.add(row), WIDTH)
            };
        }
    }
}

/// Compute an fp32 quartic-extension product round eight rows at a time.
///
/// Each table is split into the two binding-order halves for the current
/// variable. All sixteen coefficient slices must have the same length, and
/// that length must be a multiple of eight.
///
/// # Panics
///
/// Panics if the slice lengths differ or are not a multiple of eight.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn compute_product_round_fp_ext4_fp32_avx2<const P: u32>(
    witness_0: CoefficientSlices<'_, P>,
    witness_1: CoefficientSlices<'_, P>,
    factor_0: CoefficientSlices<'_, P>,
    factor_1: CoefficientSlices<'_, P>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    // SAFETY: the target feature above matches the packed backend, and the
    // generic loop validates every input before reading complete chunks.
    unsafe {
        compute_product_round_packed::<P, PackedFp32Avx2<P>>(
            witness_0, witness_1, factor_0, factor_1,
        )
    }
}

/// Compute an fp32 quartic-extension product round sixteen rows at a time.
///
/// Each table is split into the two binding-order halves for the current
/// variable. All sixteen coefficient slices must have the same length, and
/// that length must be a multiple of sixteen.
///
/// # Panics
///
/// Panics if the slice lengths differ or are not a multiple of sixteen.
///
/// # Safety
///
/// The caller must establish that AVX-512F, AVX-512DQ, and AVX-512IFMA are
/// available on the current CPU.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn compute_product_round_fp_ext4_fp32_avx512_ifma<const P: u32>(
    witness_0: CoefficientSlices<'_, P>,
    witness_1: CoefficientSlices<'_, P>,
    factor_0: CoefficientSlices<'_, P>,
    factor_1: CoefficientSlices<'_, P>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    // SAFETY: the target features above match the packed backend, and the
    // generic loop validates every input before reading complete chunks.
    unsafe {
        compute_product_round_packed::<P, PackedFp32Avx512<P>>(
            witness_0, witness_1, factor_0, factor_1,
        )
    }
}

/// Compute an equality-weighted affine-product round eight rows at a time.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn compute_weighted_affine_product_round_fp_ext4_fp32_avx2<
    const P: u32,
    const LANES: usize,
>(
    left: [CoefficientSlices<'_, P>; LANES],
    right: [CoefficientSlices<'_, P>; LANES],
    equality: CoefficientSlices<'_, P>,
    arity: usize,
    parent_weights: &[FpExt4<Fp32<P>>],
) -> [FpExt4<Fp32<P>>; 5] {
    // SAFETY: the target feature matches the packed backend, and the shared
    // traversal validates every slice before reading it.
    unsafe {
        compute_weighted_affine_product_round_packed::<P, PackedFp32Avx2<P>, LANES>(
            left,
            right,
            equality,
            arity,
            parent_weights,
        )
    }
}

/// Compute an equality-weighted affine-product round sixteen rows at a time.
///
/// # Safety
///
/// The caller must establish that AVX-512F, DQ, and IFMA are available.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn compute_weighted_affine_product_round_fp_ext4_fp32_avx512_ifma<
    const P: u32,
    const LANES: usize,
>(
    left: [CoefficientSlices<'_, P>; LANES],
    right: [CoefficientSlices<'_, P>; LANES],
    equality: CoefficientSlices<'_, P>,
    arity: usize,
    parent_weights: &[FpExt4<Fp32<P>>],
) -> [FpExt4<Fp32<P>>; 5] {
    // SAFETY: the target features match the packed backend, and the shared
    // traversal validates every slice before reading it.
    unsafe {
        compute_weighted_affine_product_round_packed::<P, PackedFp32Avx512<P>, LANES>(
            left,
            right,
            equality,
            arity,
            parent_weights,
        )
    }
}

/// Compute an equality-weighted polynomial-composition round with AVX2.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn compute_weighted_affine_polynomial_round_fp_ext4_fp32_avx2<const P: u32>(
    left: CoefficientSlices<'_, P>,
    right: CoefficientSlices<'_, P>,
    equality: CoefficientSlices<'_, P>,
    polynomial_coefficients: &[FpExt4<Fp32<P>>],
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        compute_weighted_affine_polynomial_round_packed::<P, PackedFp32Avx2<P>>(
            left,
            right,
            equality,
            polynomial_coefficients,
        )
    }
}

/// Compute an equality-weighted polynomial-composition round with AVX-512 IFMA.
///
/// # Safety
///
/// The caller must establish that AVX-512F, DQ, and IFMA are available.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn compute_weighted_affine_polynomial_round_fp_ext4_fp32_avx512_ifma<const P: u32>(
    left: CoefficientSlices<'_, P>,
    right: CoefficientSlices<'_, P>,
    equality: CoefficientSlices<'_, P>,
    polynomial_coefficients: &[FpExt4<Fp32<P>>],
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        compute_weighted_affine_polynomial_round_packed::<P, PackedFp32Avx512<P>>(
            left,
            right,
            equality,
            polynomial_coefficients,
        )
    }
}

/// Compute a compact class-indexed product prefix with AVX2.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn compute_compact_affine_product_round_fp_ext4_fp32_avx2<
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
        compute_compact_affine_product_round_packed::<P, PackedFp32Avx2<P>, LANES>(
            ordered_pair_indices,
            folded_pair_rows,
            first_equality,
            second_equality,
            arity,
            parent_weights,
        )
    }
}

/// Compute a compact class-indexed product prefix with AVX-512 IFMA.
///
/// # Safety
///
/// The caller must establish that AVX-512F, DQ, and IFMA are available.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn compute_compact_affine_product_round_fp_ext4_fp32_avx512_ifma<
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
        compute_compact_affine_product_round_packed::<P, PackedFp32Avx512<P>, LANES>(
            ordered_pair_indices,
            folded_pair_rows,
            first_equality,
            second_equality,
            arity,
            parent_weights,
        )
    }
}

/// Compute a class-coded polynomial round with AVX2.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn compute_class_coded_affine_polynomial_round_fp_ext4_fp32_avx2<const P: u32>(
    class_codes: &[u16],
    class_values: &[FpExt4<Fp32<P>>],
    class_taylor_coefficients: &[[FpExt4<Fp32<P>>; 4]],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        compute_class_coded_affine_polynomial_round_packed::<P, PackedFp32Avx2<P>>(
            class_codes,
            class_values,
            class_taylor_coefficients,
            first_equality,
            second_equality,
            degree,
        )
    }
}

/// Compute a class-coded polynomial round with AVX-512 IFMA.
///
/// # Safety
///
/// The caller must establish that AVX-512F, DQ, and IFMA are available.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn compute_class_coded_affine_polynomial_round_fp_ext4_fp32_avx512_ifma<const P: u32>(
    class_codes: &[u16],
    class_values: &[FpExt4<Fp32<P>>],
    class_taylor_coefficients: &[[FpExt4<Fp32<P>>; 4]],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        compute_class_coded_affine_polynomial_round_packed::<P, PackedFp32Avx512<P>>(
            class_codes,
            class_values,
            class_taylor_coefficients,
            first_equality,
            second_equality,
            degree,
        )
    }
}

/// Fold a sparse-prefix value table and compute its next round with AVX2.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn fold_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_avx2<const P: u32>(
    values: &[FpExt4<Fp32<P>>],
    folded_values: &mut [FpExt4<Fp32<P>>],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    challenge: FpExt4<Fp32<P>>,
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        fold_and_compute_sparse_affine_polynomial_round_packed::<P, PackedFp32Avx2<P>>(
            values,
            folded_values,
            first_equality,
            second_equality,
            challenge,
            degree,
        )
    }
}

/// Fold a sparse-prefix value table and compute its next round with AVX-512 IFMA.
///
/// # Safety
///
/// The caller must establish that AVX-512F, DQ, and IFMA are available.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn fold_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_avx512_ifma<
    const P: u32,
>(
    values: &[FpExt4<Fp32<P>>],
    folded_values: &mut [FpExt4<Fp32<P>>],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    challenge: FpExt4<Fp32<P>>,
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        fold_and_compute_sparse_affine_polynomial_round_packed::<P, PackedFp32Avx512<P>>(
            values,
            folded_values,
            first_equality,
            second_equality,
            challenge,
            degree,
        )
    }
}

/// Fold class-coded values and compute their next sparse-prefix round with AVX2.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn fold_class_coded_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_avx2<
    const P: u32,
>(
    class_codes: &[u16],
    class_values: &[FpExt4<Fp32<P>>],
    folded_values: &mut [FpExt4<Fp32<P>>],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    challenge: FpExt4<Fp32<P>>,
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        fold_class_coded_and_compute_sparse_affine_polynomial_round_packed::<P, PackedFp32Avx2<P>>(
            class_codes,
            class_values,
            folded_values,
            first_equality,
            second_equality,
            challenge,
            degree,
        )
    }
}

/// Fold class-coded values and compute their next sparse-prefix round with AVX-512 IFMA.
///
/// # Safety
///
/// The caller must establish that AVX-512F, DQ, and IFMA are available.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn fold_class_coded_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_avx512_ifma<
    const P: u32,
>(
    class_codes: &[u16],
    class_values: &[FpExt4<Fp32<P>>],
    folded_values: &mut [FpExt4<Fp32<P>>],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    challenge: FpExt4<Fp32<P>>,
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5] {
    unsafe {
        fold_class_coded_and_compute_sparse_affine_polynomial_round_packed::<P, PackedFp32Avx512<P>>(
            class_codes,
            class_values,
            folded_values,
            first_equality,
            second_equality,
            challenge,
            degree,
        )
    }
}

/// Fold two fp32 quartic-extension tables and compute the next product round
/// eight rows at a time.
///
/// Each left and right table contains one binding-order half for the variable
/// being folded. All sixteen coefficient slices must have the same even length.
/// Half of that length must be a multiple of eight.
///
/// # Panics
///
/// Panics if the slice shapes do not satisfy these requirements.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn fold_and_compute_product_round_fp_ext4_fp32_avx2<const P: u32>(
    witness_left: [&mut [Fp32<P>]; 4],
    witness_right: CoefficientSlices<'_, P>,
    factor_left: [&mut [Fp32<P>]; 4],
    factor_right: CoefficientSlices<'_, P>,
    challenge: FpExt4<Fp32<P>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    // SAFETY: the target feature above matches the packed backend, and the
    // generic loop validates every input and output before accessing it.
    unsafe {
        fold_and_compute_product_round_packed::<P, PackedFp32Avx2<P>>(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    }
}

/// Fold two fp32 quartic-extension tables and compute the next product round
/// sixteen rows at a time.
///
/// Each left and right table contains one binding-order half for the variable
/// being folded. All sixteen coefficient slices must have the same even length.
/// Half of that length must be a multiple of sixteen.
///
/// # Panics
///
/// Panics if the slice shapes do not satisfy these requirements.
///
/// # Safety
///
/// The caller must establish that AVX-512F, AVX-512DQ, and AVX-512IFMA are
/// available on the current CPU.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn fold_and_compute_product_round_fp_ext4_fp32_avx512_ifma<const P: u32>(
    witness_left: [&mut [Fp32<P>]; 4],
    witness_right: CoefficientSlices<'_, P>,
    factor_left: [&mut [Fp32<P>]; 4],
    factor_right: CoefficientSlices<'_, P>,
    challenge: FpExt4<Fp32<P>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    // SAFETY: the target features above match the packed backend, and the
    // generic loop validates every input and output before accessing it.
    unsafe {
        fold_and_compute_product_round_packed::<P, PackedFp32Avx512<P>>(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    }
}

/// Fold fp64 quadratic-extension slices four rows at a time with AVX2.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn fold_fp_ext2_fp64_avx2<const P: u64, C>(
    left: [&mut [Fp64<P>]; 2],
    right: [&[Fp64<P>]; 2],
    challenge: FpExt2<Fp64<P>, C>,
) where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    // SAFETY: this target function enables the feature required by the packed
    // backend; the shared traversal validates every slice before access.
    unsafe { fold_fp_ext2_fp64_packed::<P, C, PackedFp64Avx2<P>>(left, right, challenge) };
}

/// Fold fp64 quadratic-extension slices eight rows at a time with AVX-512.
///
/// # Safety
///
/// The caller must establish that AVX-512F and AVX-512DQ are available.
#[target_feature(enable = "avx512f,avx512dq")]
pub unsafe fn fold_fp_ext2_fp64_avx512<const P: u64, C>(
    left: [&mut [Fp64<P>]; 2],
    right: [&[Fp64<P>]; 2],
    challenge: FpExt2<Fp64<P>, C>,
) where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    // SAFETY: this target function enables the features required by the packed
    // backend; the shared traversal validates every slice before access.
    unsafe { fold_fp_ext2_fp64_packed::<P, C, PackedFp64Avx512<P>>(left, right, challenge) };
}

/// Compute an fp64 quadratic-extension product round four rows at a time.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn compute_product_round_fp_ext2_fp64_avx2<const P: u64, C>(
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
        compute_product_round_fp_ext2_fp64_packed::<P, C, PackedFp64Avx2<P>>(
            witness_0, witness_1, factor_0, factor_1,
        )
    }
}

/// Compute an fp64 quadratic-extension product round eight rows at a time.
///
/// # Safety
///
/// The caller must establish that AVX-512F and AVX-512DQ are available.
#[target_feature(enable = "avx512f,avx512dq")]
pub unsafe fn compute_product_round_fp_ext2_fp64_avx512<const P: u64, C>(
    witness_0: [&[Fp64<P>]; 2],
    witness_1: [&[Fp64<P>]; 2],
    factor_0: [&[Fp64<P>]; 2],
    factor_1: [&[Fp64<P>]; 2],
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    // SAFETY: this target function enables the features required by the packed
    // backend; the shared traversal validates every slice before access.
    unsafe {
        compute_product_round_fp_ext2_fp64_packed::<P, C, PackedFp64Avx512<P>>(
            witness_0, witness_1, factor_0, factor_1,
        )
    }
}

/// Fold two fp64 tables and compute the next product round with AVX2.
///
/// # Safety
///
/// The caller must establish that AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn fold_and_compute_product_round_fp_ext2_fp64_avx2<const P: u64, C>(
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
        fold_and_compute_product_round_fp_ext2_fp64_packed::<P, C, PackedFp64Avx2<P>>(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    }
}

/// Fold two fp64 tables and compute the next product round with AVX-512.
///
/// # Safety
///
/// The caller must establish that AVX-512F and AVX-512DQ are available.
#[target_feature(enable = "avx512f,avx512dq")]
pub unsafe fn fold_and_compute_product_round_fp_ext2_fp64_avx512<const P: u64, C>(
    witness_left: [&mut [Fp64<P>]; 2],
    witness_right: [&[Fp64<P>]; 2],
    factor_left: [&mut [Fp64<P>]; 2],
    factor_right: [&[Fp64<P>]; 2],
    challenge: FpExt2<Fp64<P>, C>,
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    // SAFETY: this target function enables the features required by the packed
    // backend; the shared traversal validates every slice before access.
    unsafe {
        fold_and_compute_product_round_fp_ext2_fp64_packed::<P, C, PackedFp64Avx512<P>>(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    }
}

fn validate_fold_slices<const P: u32>(
    left: &[&mut [Fp32<P>]; 4],
    right: &[&[Fp32<P>]; 4],
    width: usize,
) -> usize {
    let len = left[0].len();
    assert!(
        left.iter().all(|slice| slice.len() == len) && right.iter().all(|slice| slice.len() == len),
        "fp32 extension fold slices must have equal lengths"
    );
    assert!(
        len.is_multiple_of(width),
        "fp32 extension fold slice length must be a multiple of the SIMD width"
    );
    len
}

#[inline(always)]
unsafe fn read_array<T: Copy, const N: usize>(start: *const T, offset: usize) -> [T; N] {
    // SAFETY: the caller establishes that `offset..offset + N` lies in the
    // allocation referenced by `start`.
    unsafe { (start.add(offset) as *const [T; N]).read() }
}
