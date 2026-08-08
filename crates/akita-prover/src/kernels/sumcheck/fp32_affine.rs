//! Runtime dispatch for compact fp32 affine-product rounds.

use super::{multiple_workers_available, Fp32Kernel, SumcheckKernelPlan};
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
        if multiple_workers_available() {
            return None;
        }
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

        let width = self.fp32_affine_width()?;
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

    pub(super) fn try_compute_class_coded_affine_polynomial_round_fp32<const P: u32>(
        self,
        class_codes: &[u16],
        class_values: &[FpExt4<Fp32<P>>],
        class_taylor_coefficients: &[[FpExt4<Fp32<P>>; 4]],
        first_equality: &[FpExt4<Fp32<P>>],
        second_equality: &[FpExt4<Fp32<P>>],
        degree: usize,
    ) -> Option<[FpExt4<Fp32<P>>; 5]> {
        if multiple_workers_available() {
            return None;
        }
        if !matches!(degree, 2 | 4)
            || !class_codes.len().is_multiple_of(2)
            || class_values.is_empty()
            || class_values.len() != class_taylor_coefficients.len()
            || first_equality.is_empty()
            || second_equality.is_empty()
            || !first_equality.len().is_power_of_two()
            || !second_equality.len().is_power_of_two()
            || class_codes
                .iter()
                .any(|&class| usize::from(class) >= class_values.len())
        {
            return None;
        }
        let pair_count = class_codes.len() / 2;
        if pair_count > first_equality.len().checked_mul(second_equality.len())? {
            return None;
        }
        let width = self.fp32_affine_width()?;
        if !first_equality.len().is_multiple_of(width) {
            return None;
        }

        match self.fp32_product_round {
            Fp32Kernel::Scalar => None,
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => Some(unsafe {
                akita_field::packed::runtime_neon::compute_class_coded_affine_polynomial_round_fp_ext4_fp32_neon(
                    class_codes,
                    class_values,
                    class_taylor_coefficients,
                    first_equality,
                    second_equality,
                    degree,
                )
            }),
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => Some(unsafe {
                akita_field::packed::runtime_x86::compute_class_coded_affine_polynomial_round_fp_ext4_fp32_avx2(
                    class_codes,
                    class_values,
                    class_taylor_coefficients,
                    first_equality,
                    second_equality,
                    degree,
                )
            }),
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => Some(unsafe {
                akita_field::packed::runtime_x86::compute_class_coded_affine_polynomial_round_fp_ext4_fp32_avx512_ifma(
                    class_codes,
                    class_values,
                    class_taylor_coefficients,
                    first_equality,
                    second_equality,
                    degree,
                )
            }),
        }
    }

    pub(super) fn try_fold_and_compute_sparse_affine_polynomial_round_fp32<const P: u32>(
        self,
        values: &[FpExt4<Fp32<P>>],
        folded_values: &mut [FpExt4<Fp32<P>>],
        first_equality: &[FpExt4<Fp32<P>>],
        second_equality: &[FpExt4<Fp32<P>>],
        challenge: FpExt4<Fp32<P>>,
        degree: usize,
    ) -> Option<[FpExt4<Fp32<P>>; 5]> {
        if multiple_workers_available() {
            return None;
        }
        if !matches!(degree, 2 | 4)
            || !values.len().is_multiple_of(4)
            || folded_values.len() != values.len() / 2
            || first_equality.is_empty()
            || second_equality.is_empty()
            || !first_equality.len().is_power_of_two()
            || !second_equality.len().is_power_of_two()
            || values.len() / 4 > first_equality.len().checked_mul(second_equality.len())?
        {
            return None;
        }
        let width = self.fp32_affine_width()?;
        if !first_equality.len().is_multiple_of(width) {
            return None;
        }

        match self.fp32_product_round {
            Fp32Kernel::Scalar => None,
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => Some(unsafe {
                akita_field::packed::runtime_neon::fold_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_neon(
                    values,
                    folded_values,
                    first_equality,
                    second_equality,
                    challenge,
                    degree,
                )
            }),
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => Some(unsafe {
                akita_field::packed::runtime_x86::fold_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_avx2(
                    values,
                    folded_values,
                    first_equality,
                    second_equality,
                    challenge,
                    degree,
                )
            }),
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => Some(unsafe {
                akita_field::packed::runtime_x86::fold_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_avx512_ifma(
                    values,
                    folded_values,
                    first_equality,
                    second_equality,
                    challenge,
                    degree,
                )
            }),
        }
    }

    pub(super) fn try_fold_class_coded_and_compute_sparse_affine_polynomial_round_fp32<
        const P: u32,
    >(
        self,
        class_codes: &[u16],
        class_values: &[FpExt4<Fp32<P>>],
        folded_values: &mut [FpExt4<Fp32<P>>],
        split_equality: (&[FpExt4<Fp32<P>>], &[FpExt4<Fp32<P>>]),
        challenge: FpExt4<Fp32<P>>,
        degree: usize,
    ) -> Option<[FpExt4<Fp32<P>>; 5]> {
        if multiple_workers_available() {
            return None;
        }
        let (first_equality, second_equality) = split_equality;
        if !matches!(degree, 2 | 4)
            || !class_codes.len().is_multiple_of(4)
            || class_values.is_empty()
            || folded_values.len() != class_codes.len() / 2
            || first_equality.is_empty()
            || second_equality.is_empty()
            || !first_equality.len().is_power_of_two()
            || !second_equality.len().is_power_of_two()
            || class_codes.len() / 4 > first_equality.len().checked_mul(second_equality.len())?
            || class_codes
                .iter()
                .any(|&class| usize::from(class) >= class_values.len())
        {
            return None;
        }
        if !first_equality
            .len()
            .is_multiple_of(self.fp32_affine_width()?)
        {
            return None;
        }

        match self.fp32_product_round {
            Fp32Kernel::Scalar => None,
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => Some(unsafe {
                akita_field::packed::runtime_neon::fold_class_coded_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_neon(
                    class_codes,
                    class_values,
                    folded_values,
                    first_equality,
                    second_equality,
                    challenge,
                    degree,
                )
            }),
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => Some(unsafe {
                akita_field::packed::runtime_x86::fold_class_coded_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_avx2(
                    class_codes,
                    class_values,
                    folded_values,
                    first_equality,
                    second_equality,
                    challenge,
                    degree,
                )
            }),
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => Some(unsafe {
                akita_field::packed::runtime_x86::fold_class_coded_and_compute_sparse_affine_polynomial_round_fp_ext4_fp32_avx512_ifma(
                    class_codes,
                    class_values,
                    folded_values,
                    first_equality,
                    second_equality,
                    challenge,
                    degree,
                )
            }),
        }
    }

    fn fp32_affine_width(self) -> Option<usize> {
        match self.fp32_product_round {
            Fp32Kernel::Scalar => None,
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => Some(4),
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => Some(8),
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => Some(16),
        }
    }
}
