//! Runtime-selected x86 arithmetic over coefficient-oriented field slices.

use super::avx2::PackedFp32Avx2;
use super::avx512::PackedFp32Avx512;
use super::{PackedField, PackedFpExt4};
use crate::{Fp32, FpExt4};

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
