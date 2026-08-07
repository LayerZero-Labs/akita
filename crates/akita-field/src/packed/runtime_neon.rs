//! Runtime-selected AArch64 NEON arithmetic over coefficient-oriented slices.

use super::neon::{PackedFp32Neon, PackedFp64Neon};
use super::runtime_common::{
    compute_product_round_fp_ext2_fp64_packed, compute_product_round_packed,
    fold_and_compute_product_round_fp_ext2_fp64_packed, fold_and_compute_product_round_packed,
    fold_fp_ext2_fp64_packed, fold_fp_ext4_fp32_packed,
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
