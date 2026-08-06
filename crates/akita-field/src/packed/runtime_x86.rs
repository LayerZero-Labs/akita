//! Runtime-selected x86 arithmetic over coefficient-oriented field slices.

use super::avx2::PackedFp32Avx2;
use super::avx512::PackedFp32Avx512;
use super::{PackedField, PackedFpExt4, PackedValue};
use crate::{Fp32, FpExt4};

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

#[inline(always)]
unsafe fn compute_product_round_packed<const P: u32, PF>(
    witness_0: CoefficientSlices<'_, P>,
    witness_1: CoefficientSlices<'_, P>,
    factor_0: CoefficientSlices<'_, P>,
    factor_1: CoefficientSlices<'_, P>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>)
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let slices = [&witness_0, &witness_1, &factor_0, &factor_1];
    let len = validate_product_round_slices(&slices, PF::WIDTH);
    let witness_0 = witness_0.map(|slice| slice.as_ptr());
    let witness_1 = witness_1.map(|slice| slice.as_ptr());
    let factor_0 = factor_0.map(|slice| slice.as_ptr());
    let factor_1 = factor_1.map(|slice| slice.as_ptr());
    let zero = FpExt4::<Fp32<P>>::zero();
    let mut constant = PackedFpExt4::<Fp32<P>, PF>::broadcast(zero);
    let mut quadratic = PackedFpExt4::<Fp32<P>, PF>::broadcast(zero);

    for row in (0..len).step_by(PF::WIDTH) {
        // SAFETY: validation establishes a complete packed chunk in every
        // coefficient slice for this row.
        let witness_0 = unsafe { read_packed_fp_ext4::<P, PF>(witness_0, row) };
        let witness_1 = unsafe { read_packed_fp_ext4::<P, PF>(witness_1, row) };
        let factor_0 = unsafe { read_packed_fp_ext4::<P, PF>(factor_0, row) };
        let factor_1 = unsafe { read_packed_fp_ext4::<P, PF>(factor_1, row) };
        constant = constant + witness_0 * factor_0;
        quadratic = quadratic + (witness_1 - witness_0) * (factor_1 - factor_0);
    }

    let mut constant_sum = zero;
    let mut quadratic_sum = zero;
    for lane in 0..PF::WIDTH {
        constant_sum += constant.extract(lane);
        quadratic_sum += quadratic.extract(lane);
    }
    (constant_sum, quadratic_sum)
}

#[inline(always)]
unsafe fn read_packed_fp_ext4<const P: u32, PF>(
    coefficients: [*const Fp32<P>; 4],
    row: usize,
) -> PackedFpExt4<Fp32<P>, PF>
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    PackedFpExt4::new(std::array::from_fn(|coefficient| {
        PF::from_fn(|lane| {
            // SAFETY: the caller validated the full packed chunk starting at
            // `row` for every coefficient pointer.
            unsafe { *coefficients[coefficient].add(row + lane) }
        })
    }))
}

fn validate_product_round_slices<const P: u32>(
    tables: &[&CoefficientSlices<'_, P>; 4],
    width: usize,
) -> usize {
    let len = tables[0][0].len();
    assert!(
        tables
            .iter()
            .all(|table| table.iter().all(|slice| slice.len() == len)),
        "fp32 product round slices must have equal lengths"
    );
    assert!(
        len.is_multiple_of(width),
        "fp32 product round slice length must be a multiple of the SIMD width"
    );
    len
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
