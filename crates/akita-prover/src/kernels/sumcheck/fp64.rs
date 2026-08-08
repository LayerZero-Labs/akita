//! Runtime-selected operations for quadratic extensions of 64-bit fields.

use super::{Fp64Kernel, SumcheckKernelPlan, SumcheckTableOperations};
use akita_field::unreduced::HasUnreducedOps;
use akita_field::{Fp64, FpExt2, FpExt2Config};
use akita_sumcheck::{
    fold_and_compute_product_round_scalar, fold_first_variable_scalar,
    DelayedProductRoundAccumulator, DirectProductRoundAccumulator, EvaluationTable,
    ProductRoundAccumulator,
};

impl<const P: u64, C> SumcheckTableOperations<Fp64<P>> for FpExt2<Fp64<P>, C>
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    fn fold_first_variable(
        plan: SumcheckKernelPlan,
        table: &mut EvaluationTable<Fp64<P>, Self>,
        challenge: Self,
    ) {
        plan.fold_first_variable_fp64(table, challenge);
    }

    fn compute_product_round(
        plan: SumcheckKernelPlan,
        witness: &EvaluationTable<Fp64<P>, Self>,
        factor: &EvaluationTable<Fp64<P>, Self>,
    ) -> (Self, Self) {
        plan.compute_product_round_fp64(witness, factor)
    }

    fn fold_and_compute_product_round(
        plan: SumcheckKernelPlan,
        witness: &mut EvaluationTable<Fp64<P>, Self>,
        factor: &mut EvaluationTable<Fp64<P>, Self>,
        challenge: Self,
    ) -> (Self, Self) {
        plan.fold_and_compute_product_round_fp64(witness, factor, challenge)
    }
}

impl SumcheckKernelPlan {
    /// Fold one fp64 quadratic-extension table using the detected operation.
    pub fn fold_first_variable_fp64<const P: u64, C>(
        self,
        table: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
        challenge: FpExt2<Fp64<P>, C>,
    ) where
        C: FpExt2Config<Fp64<P>> + 'static,
    {
        assert!(
            table.len().is_power_of_two(),
            "evaluation table length must be a power of two"
        );
        assert!(
            table.len() >= 2,
            "evaluation table must have at least two rows"
        );

        match self.fp64_fold {
            Fp64Kernel::Scalar => fold_first_variable_scalar(table, challenge),
            #[cfg(target_arch = "aarch64")]
            Fp64Kernel::Neon => {
                if table.len() / 2 < 2 {
                    fold_first_variable_scalar(table, challenge);
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking NEON support.
                    unsafe { fold_fp64_neon(table, challenge) };
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp64Kernel::Avx2 => {
                if table.len() / 2 < 4 {
                    fold_first_variable_scalar(table, challenge);
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX2 support.
                    unsafe { fold_fp64_avx2(table, challenge) };
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp64Kernel::Avx512 => {
                if table.len() / 2 < 8 {
                    fold_first_variable_scalar(table, challenge);
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX-512F and DQ.
                    unsafe { fold_fp64_avx512(table, challenge) };
                }
            }
        }
    }

    /// Compute one fp64 quadratic-extension product round.
    pub fn compute_product_round_fp64<const P: u64, C>(
        self,
        witness: &EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
        factor: &EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    ) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
    where
        C: FpExt2Config<Fp64<P>> + 'static,
    {
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

        match self.fp64_product_round {
            Fp64Kernel::Scalar => compute_product_round_fp64_scalar(witness, factor),
            #[cfg(target_arch = "aarch64")]
            Fp64Kernel::Neon => {
                if witness.len() / 2 < 2 {
                    compute_product_round_fp64_scalar(witness, factor)
                } else {
                    let (witness_0, witness_1) = fp64_coefficient_halves(witness);
                    let (factor_0, factor_1) = fp64_coefficient_halves(factor);
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking NEON support.
                    unsafe {
                        akita_field::packed::runtime_neon::compute_product_round_fp_ext2_fp64_neon(
                            witness_0, witness_1, factor_0, factor_1,
                        )
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp64Kernel::Avx2 => {
                if witness.len() / 2 < 4 {
                    compute_product_round_fp64_scalar(witness, factor)
                } else {
                    let (witness_0, witness_1) = fp64_coefficient_halves(witness);
                    let (factor_0, factor_1) = fp64_coefficient_halves(factor);
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX2 support.
                    unsafe {
                        akita_field::packed::runtime_x86::compute_product_round_fp_ext2_fp64_avx2(
                            witness_0, witness_1, factor_0, factor_1,
                        )
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp64Kernel::Avx512 => {
                if witness.len() / 2 < 8 {
                    compute_product_round_fp64_scalar(witness, factor)
                } else {
                    let (witness_0, witness_1) = fp64_coefficient_halves(witness);
                    let (factor_0, factor_1) = fp64_coefficient_halves(factor);
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX-512F and DQ.
                    unsafe {
                        akita_field::packed::runtime_x86::compute_product_round_fp_ext2_fp64_avx512(
                            witness_0, witness_1, factor_0, factor_1,
                        )
                    }
                }
            }
        }
    }

    /// Fold two fp64 tables and compute their next product round.
    pub fn fold_and_compute_product_round_fp64<const P: u64, C>(
        self,
        witness: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
        factor: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
        challenge: FpExt2<Fp64<P>, C>,
    ) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
    where
        C: FpExt2Config<Fp64<P>> + 'static,
    {
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

        match self.fp64_fold_and_product_round {
            Fp64Kernel::Scalar => fold_and_compute_product_round_scalar(witness, factor, challenge),
            #[cfg(target_arch = "aarch64")]
            Fp64Kernel::Neon => {
                if witness.len() / 4 < 2 {
                    fold_and_compute_product_round_scalar(witness, factor, challenge)
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking NEON support.
                    unsafe { fold_and_compute_product_round_fp64_neon(witness, factor, challenge) }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp64Kernel::Avx2 => {
                if witness.len() / 4 < 4 {
                    fold_and_compute_product_round_scalar(witness, factor, challenge)
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX2 support.
                    unsafe { fold_and_compute_product_round_fp64_avx2(witness, factor, challenge) }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp64Kernel::Avx512 => {
                if witness.len() / 4 < 8 {
                    fold_and_compute_product_round_scalar(witness, factor, challenge)
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX-512F and DQ.
                    unsafe {
                        fold_and_compute_product_round_fp64_avx512(witness, factor, challenge)
                    }
                }
            }
        }
    }
}

fn compute_product_round_fp64_scalar<const P: u64, C>(
    witness: &EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    factor: &EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    if <FpExt2<Fp64<P>, C> as HasUnreducedOps>::DELAYED_PRODUCT_SUM_IS_EXACT {
        compute_product_round_fp64_scalar_with::<P, C, DelayedProductRoundAccumulator<_>>(
            witness, factor,
        )
    } else {
        compute_product_round_fp64_scalar_with::<P, C, DirectProductRoundAccumulator<_>>(
            witness, factor,
        )
    }
}

fn compute_product_round_fp64_scalar_with<const P: u64, C, A>(
    witness: &EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    factor: &EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
    A: ProductRoundAccumulator<FpExt2<Fp64<P>, C>>,
{
    let half = witness.len() / 2;
    let [witness_c0, witness_c1] = witness.coefficient_slices::<2>();
    let [factor_c0, factor_c1] = factor.coefficient_slices::<2>();
    let mut accumulator = A::zero();
    for row in 0..half {
        let witness_0 = FpExt2::new(witness_c0[row], witness_c1[row]);
        let witness_1 = FpExt2::new(witness_c0[row + half], witness_c1[row + half]);
        let factor_0 = FpExt2::new(factor_c0[row], factor_c1[row]);
        let factor_1 = FpExt2::new(factor_c0[row + half], factor_c1[row + half]);
        accumulator.add_constant_product(witness_0, factor_0);
        accumulator.add_quadratic_product(witness_1 - witness_0, factor_1 - factor_0);
    }
    accumulator.finish()
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn fp64_coefficient_halves<const P: u64, C>(
    table: &EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
) -> ([&[Fp64<P>]; 2], [&[Fp64<P>]; 2])
where
    C: FpExt2Config<Fp64<P>>,
{
    let half = table.len() / 2;
    (
        std::array::from_fn(|coefficient| &table.coefficient_slice(coefficient)[..half]),
        std::array::from_fn(|coefficient| &table.coefficient_slice(coefficient)[half..]),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fold_fp64_neon<const P: u64, C>(
    table: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    challenge: FpExt2<Fp64<P>, C>,
) where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    let half = table.len() / 2;
    let (left, right) = fp64_coefficient_halves_mut(table);

    // SAFETY: this function requires NEON, and every left/right pair comes
    // from equal halves of a power-of-two table with `half >= 2`.
    unsafe { akita_field::packed::runtime_neon::fold_fp_ext2_fp64_neon(left, right, challenge) };
    table.truncate(half);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fold_fp64_avx2<const P: u64, C>(
    table: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    challenge: FpExt2<Fp64<P>, C>,
) where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    let half = table.len() / 2;
    let (left, right) = fp64_coefficient_halves_mut(table);
    // SAFETY: this function requires AVX2, and the power-of-two table has at
    // least four rows in each input half.
    unsafe { akita_field::packed::runtime_x86::fold_fp_ext2_fp64_avx2(left, right, challenge) };
    table.truncate(half);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn fold_fp64_avx512<const P: u64, C>(
    table: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    challenge: FpExt2<Fp64<P>, C>,
) where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    let half = table.len() / 2;
    let (left, right) = fp64_coefficient_halves_mut(table);
    // SAFETY: this function requires AVX-512F and DQ, and the power-of-two
    // table has at least eight rows in each input half.
    unsafe { akita_field::packed::runtime_x86::fold_fp_ext2_fp64_avx512(left, right, challenge) };
    table.truncate(half);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fold_and_compute_product_round_fp64_neon<const P: u64, C>(
    witness: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    factor: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    challenge: FpExt2<Fp64<P>, C>,
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    let half = witness.len() / 2;
    let (witness_left, witness_right) = fp64_coefficient_halves_mut(witness);
    let (factor_left, factor_right) = fp64_coefficient_halves_mut(factor);
    // SAFETY: this function requires NEON. Both tables have equal power-of-two
    // lengths, and each next-round half has at least two rows.
    let round = unsafe {
        akita_field::packed::runtime_neon::fold_and_compute_product_round_fp_ext2_fp64_neon(
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
#[target_feature(enable = "avx2")]
unsafe fn fold_and_compute_product_round_fp64_avx2<const P: u64, C>(
    witness: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    factor: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    challenge: FpExt2<Fp64<P>, C>,
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    let half = witness.len() / 2;
    let (witness_left, witness_right) = fp64_coefficient_halves_mut(witness);
    let (factor_left, factor_right) = fp64_coefficient_halves_mut(factor);
    // SAFETY: this function requires AVX2. Both power-of-two tables have at
    // least four rows in each next-round half.
    let round = unsafe {
        akita_field::packed::runtime_x86::fold_and_compute_product_round_fp_ext2_fp64_avx2(
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
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn fold_and_compute_product_round_fp64_avx512<const P: u64, C>(
    witness: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    factor: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    challenge: FpExt2<Fp64<P>, C>,
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    let half = witness.len() / 2;
    let (witness_left, witness_right) = fp64_coefficient_halves_mut(witness);
    let (factor_left, factor_right) = fp64_coefficient_halves_mut(factor);
    // SAFETY: this function requires AVX-512F and DQ. Both power-of-two tables
    // have at least eight rows in each next-round half.
    let round = unsafe {
        akita_field::packed::runtime_x86::fold_and_compute_product_round_fp_ext2_fp64_avx512(
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
fn fp64_coefficient_halves_mut<const P: u64, C>(
    table: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
) -> ([&mut [Fp64<P>]; 2], [&[Fp64<P>]; 2])
where
    C: FpExt2Config<Fp64<P>>,
{
    let half = table.len() / 2;
    let [coefficient_0, coefficient_1] = table.coefficient_slices_mut::<2>();
    let (left_0, right_0) = coefficient_0.split_at_mut(half);
    let (left_1, right_1) = coefficient_1.split_at_mut(half);
    ([left_0, left_1], [right_0, right_1])
}
