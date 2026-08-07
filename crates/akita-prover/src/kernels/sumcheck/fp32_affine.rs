//! Runtime dispatch for compact fp32 affine-product rounds.

use super::{Fp32Kernel, SumcheckKernelPlan};
use akita_field::{Fp32, FpExt4};

impl SumcheckKernelPlan {
    pub(super) fn try_compute_compact_affine_product_round_fp32<
        const P: u32,
        const LANES: usize,
    >(
        self,
        ordered_pair_indices: &[u16],
        folded_pair_rows: &[[FpExt4<Fp32<P>>; LANES]],
        first_equality: &[FpExt4<Fp32<P>>],
        second_equality: &[FpExt4<Fp32<P>>],
        arity: usize,
        parent_weights: &[FpExt4<Fp32<P>>],
    ) -> Option<[FpExt4<Fp32<P>>; 5]> {
        let quartet_count = ordered_pair_indices.len().div_ceil(2);
        if !matches!(arity, 2 | 4)
            || LANES != arity * parent_weights.len()
            || folded_pair_rows.is_empty()
            || first_equality.is_empty()
            || second_equality.is_empty()
            || quartet_count > first_equality.len().checked_mul(second_equality.len())?
            || ordered_pair_indices
                .iter()
                .any(|&index| usize::from(index) >= folded_pair_rows.len())
        {
            return None;
        }

        let width = match self.fp32_product_round {
            Fp32Kernel::Scalar => return None,
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => 4,
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => 8,
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => 16,
        };
        if !first_equality.len().is_multiple_of(width) || !quartet_count.is_multiple_of(width) {
            return None;
        }

        match self.fp32_product_round {
            Fp32Kernel::Scalar => None,
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => {
                // SAFETY: production plans select NEON only after runtime
                // detection, and the shape checks above establish full chunks.
                Some(unsafe {
                    akita_field::packed::runtime_neon::compute_compact_affine_product_round_fp_ext4_fp32_neon(
                        ordered_pair_indices,
                        folded_pair_rows,
                        first_equality,
                        second_equality,
                        arity,
                        parent_weights,
                    )
                })
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => {
                // SAFETY: production plans select AVX2 only after runtime
                // detection, and the shape checks above establish full chunks.
                Some(unsafe {
                    akita_field::packed::runtime_x86::compute_compact_affine_product_round_fp_ext4_fp32_avx2(
                        ordered_pair_indices,
                        folded_pair_rows,
                        first_equality,
                        second_equality,
                        arity,
                        parent_weights,
                    )
                })
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => {
                // SAFETY: production plans select AVX-512 only after runtime
                // detection, and the shape checks above establish full chunks.
                Some(unsafe {
                    akita_field::packed::runtime_x86::compute_compact_affine_product_round_fp_ext4_fp32_avx512_ifma(
                        ordered_pair_indices,
                        folded_pair_rows,
                        first_equality,
                        second_equality,
                        arity,
                        parent_weights,
                    )
                })
            }
        }
    }
}
