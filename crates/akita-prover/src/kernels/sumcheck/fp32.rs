//! Runtime-selected operations for quartic extensions of 32-bit fields.

use super::{Fp32Kernel, SumcheckKernelPlan, SumcheckTableOperations};
use akita_field::{Fp32, FpExt4};
use akita_sumcheck::{
    compute_product_round_scalar, fold_and_compute_product_round_scalar,
    fold_first_variable_scalar, EvaluationTable,
};

impl<const P: u32> SumcheckTableOperations<Fp32<P>> for FpExt4<Fp32<P>> {
    fn fold_first_variable(
        plan: SumcheckKernelPlan,
        table: &mut EvaluationTable<Fp32<P>, Self>,
        challenge: Self,
    ) {
        plan.fold_first_variable_fp32(table, challenge);
    }

    fn compute_product_round(
        plan: SumcheckKernelPlan,
        witness: &EvaluationTable<Fp32<P>, Self>,
        factor: &EvaluationTable<Fp32<P>, Self>,
    ) -> (Self, Self) {
        plan.compute_product_round_fp32(witness, factor)
    }

    fn fold_and_compute_product_round(
        plan: SumcheckKernelPlan,
        witness: &mut EvaluationTable<Fp32<P>, Self>,
        factor: &mut EvaluationTable<Fp32<P>, Self>,
        challenge: Self,
    ) -> (Self, Self) {
        plan.fold_and_compute_product_round_fp32(witness, factor, challenge)
    }
}

impl SumcheckKernelPlan {
    /// Fold one fp32 quartic-extension table using the detected operation.
    pub fn fold_first_variable_fp32<const P: u32>(
        self,
        table: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
        challenge: FpExt4<Fp32<P>>,
    ) {
        assert!(
            table.len().is_power_of_two(),
            "evaluation table length must be a power of two"
        );
        assert!(
            table.len() >= 2,
            "evaluation table must have at least two rows"
        );

        match self.fp32_fold {
            Fp32Kernel::Scalar => fold_first_variable_scalar(table, challenge),
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => {
                if table.len() / 2 < 4 {
                    fold_first_variable_scalar(table, challenge);
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking NEON support.
                    unsafe { fold_fp32_neon(table, challenge) };
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => {
                if table.len() / 2 < 8 {
                    fold_first_variable_scalar(table, challenge);
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX2 support.
                    unsafe { fold_fp32_avx2(table, challenge) };
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => {
                if table.len() / 2 < 16 {
                    fold_first_variable_scalar(table, challenge);
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX-512F, DQ,
                    // and IFMA.
                    unsafe { fold_fp32_avx512_ifma(table, challenge) };
                }
            }
        }
    }

    /// Compute one fp32 quartic-extension product round using the detected operation.
    pub fn compute_product_round_fp32<const P: u32>(
        self,
        witness: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
        factor: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    ) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
        assert_eq!(
            witness.len(),
            factor.len(),
            "product round tables must have equal lengths"
        );
        assert!(
            witness.len().is_power_of_two(),
            "product round table length must be a power of two"
        );
        assert!(
            witness.len() >= 2,
            "product round tables must have at least two rows"
        );

        match self.fp32_product_round {
            Fp32Kernel::Scalar => compute_product_round_scalar(witness, factor),
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => {
                if witness.len() / 2 < 4 {
                    compute_product_round_scalar(witness, factor)
                } else {
                    let (witness_0, witness_1) = coefficient_halves(witness);
                    let (factor_0, factor_1) = coefficient_halves(factor);
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking NEON support.
                    unsafe {
                        akita_field::packed::runtime_neon::compute_product_round_fp_ext4_fp32_neon(
                            witness_0, witness_1, factor_0, factor_1,
                        )
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => {
                if witness.len() / 2 < 8 {
                    compute_product_round_scalar(witness, factor)
                } else {
                    let (witness_0, witness_1) = coefficient_halves(witness);
                    let (factor_0, factor_1) = coefficient_halves(factor);
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX2 support.
                    unsafe {
                        akita_field::packed::runtime_x86::compute_product_round_fp_ext4_fp32_avx2(
                            witness_0, witness_1, factor_0, factor_1,
                        )
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => {
                if witness.len() / 2 < 16 {
                    compute_product_round_scalar(witness, factor)
                } else {
                    let (witness_0, witness_1) = coefficient_halves(witness);
                    let (factor_0, factor_1) = coefficient_halves(factor);
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX-512F, DQ,
                    // and IFMA.
                    unsafe {
                        akita_field::packed::runtime_x86::compute_product_round_fp_ext4_fp32_avx512_ifma(
                        witness_0, witness_1, factor_0, factor_1,
                    )
                    }
                }
            }
        }
    }

    /// Fold two fp32 tables and compute their next product round.
    pub fn fold_and_compute_product_round_fp32<const P: u32>(
        self,
        witness: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
        factor: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
        challenge: FpExt4<Fp32<P>>,
    ) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
        assert_eq!(
            witness.len(),
            factor.len(),
            "product round tables must have equal lengths"
        );
        assert!(
            witness.len().is_power_of_two(),
            "product round table length must be a power of two"
        );
        assert!(
            witness.len() >= 4,
            "fused product round tables must have at least four rows"
        );

        match self.fp32_fold_and_product_round {
            Fp32Kernel::Scalar => fold_and_compute_product_round_scalar(witness, factor, challenge),
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => {
                if witness.len() / 4 < 4 {
                    fold_and_compute_product_round_scalar(witness, factor, challenge)
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking NEON support.
                    unsafe { fold_and_compute_product_round_fp32_neon(witness, factor, challenge) }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => {
                if witness.len() / 4 < 8 {
                    fold_and_compute_product_round_scalar(witness, factor, challenge)
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX2 support.
                    unsafe { fold_and_compute_product_round_fp32_avx2(witness, factor, challenge) }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => {
                if witness.len() / 4 < 16 {
                    fold_and_compute_product_round_scalar(witness, factor, challenge)
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX-512F, DQ,
                    // and IFMA.
                    unsafe {
                        fold_and_compute_product_round_fp32_avx512_ifma(witness, factor, challenge)
                    }
                }
            }
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn coefficient_halves<const P: u32>(
    table: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
) -> ([&[Fp32<P>]; 4], [&[Fp32<P>]; 4]) {
    let half = table.len() / 2;
    (
        std::array::from_fn(|coefficient| &table.coefficient_slice(coefficient)[..half]),
        std::array::from_fn(|coefficient| &table.coefficient_slice(coefficient)[half..]),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fold_fp32_neon<const P: u32>(
    table: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) {
    let half = table.len() / 2;
    let (left, right) = coefficient_halves_mut(table);

    // SAFETY: this function requires NEON, and every left/right pair comes
    // from equal halves of a power-of-two table with `half >= 4`.
    unsafe { akita_field::packed::runtime_neon::fold_fp_ext4_fp32_neon(left, right, challenge) };
    table.truncate(half);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fold_fp32_avx2<const P: u32>(
    table: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) {
    let half = table.len() / 2;
    let (left, right) = coefficient_halves_mut(table);

    // SAFETY: this function requires AVX2, and every left/right pair comes
    // from equal halves of a power-of-two table with `half >= 8`.
    unsafe { akita_field::packed::runtime_x86::fold_fp_ext4_fp32_avx2(left, right, challenge) };
    table.truncate(half);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn fold_fp32_avx512_ifma<const P: u32>(
    table: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) {
    let half = table.len() / 2;
    let (left, right) = coefficient_halves_mut(table);

    // SAFETY: this function requires AVX-512F, DQ, and IFMA, and every
    // left/right pair comes from equal halves of a power-of-two table with
    // `half >= 16`.
    unsafe {
        akita_field::packed::runtime_x86::fold_fp_ext4_fp32_avx512_ifma(left, right, challenge)
    };
    table.truncate(half);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fold_and_compute_product_round_fp32_avx2<const P: u32>(
    witness: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    factor: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    let half = witness.len() / 2;
    let (witness_left, witness_right) = coefficient_halves_mut(witness);
    let (factor_left, factor_right) = coefficient_halves_mut(factor);
    // SAFETY: this function requires AVX2. Both tables have equal power-of-two
    // lengths, and each next-round half has at least eight rows.
    let round = unsafe {
        akita_field::packed::runtime_x86::fold_and_compute_product_round_fp_ext4_fp32_avx2(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    };
    witness.truncate(half);
    factor.truncate(half);
    round
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn fold_and_compute_product_round_fp32_avx512_ifma<const P: u32>(
    witness: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    factor: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    let half = witness.len() / 2;
    let (witness_left, witness_right) = coefficient_halves_mut(witness);
    let (factor_left, factor_right) = coefficient_halves_mut(factor);
    // SAFETY: this function requires AVX-512F, DQ, and IFMA. Both tables have
    // equal power-of-two lengths, and each next-round half has at least sixteen
    // rows.
    let round = unsafe {
        akita_field::packed::runtime_x86::fold_and_compute_product_round_fp_ext4_fp32_avx512_ifma(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    };
    witness.truncate(half);
    factor.truncate(half);
    round
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fold_and_compute_product_round_fp32_neon<const P: u32>(
    witness: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    factor: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>) {
    let half = witness.len() / 2;
    let (witness_left, witness_right) = coefficient_halves_mut(witness);
    let (factor_left, factor_right) = coefficient_halves_mut(factor);
    // SAFETY: this function requires NEON. Both tables have equal power-of-two
    // lengths, and each next-round half has at least four rows.
    let round = unsafe {
        akita_field::packed::runtime_neon::fold_and_compute_product_round_fp_ext4_fp32_neon(
            witness_left,
            witness_right,
            factor_left,
            factor_right,
            challenge,
        )
    };
    witness.truncate(half);
    factor.truncate(half);
    round
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn coefficient_halves_mut<const P: u32>(
    table: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
) -> ([&mut [Fp32<P>]; 4], [&[Fp32<P>]; 4]) {
    let half = table.len() / 2;
    let [coefficient_0, coefficient_1, coefficient_2, coefficient_3] =
        table.coefficient_slices_mut::<4>();
    let (left_0, right_0) = coefficient_0.split_at_mut(half);
    let (left_1, right_1) = coefficient_1.split_at_mut(half);
    let (left_2, right_2) = coefficient_2.split_at_mut(half);
    let (left_3, right_3) = coefficient_3.split_at_mut(half);
    (
        [left_0, left_1, left_2, left_3],
        [right_0, right_1, right_2, right_3],
    )
}
