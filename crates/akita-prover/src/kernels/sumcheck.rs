//! Runtime-selected kernels over canonical sumcheck evaluation tables.

use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{ExtField, FieldCore, Fp128, Fp32, Fp64, FpExt2, FpExt2Config, FpExt4, FpExt8};
use akita_sumcheck::{
    compute_product_round_scalar, fold_and_compute_product_round_scalar,
    fold_first_variable_scalar, DelayedProductRoundAccumulator, DirectProductRoundAccumulator,
    EvaluationTable, ProductRoundAccumulator,
};

/// Host-detected operation choices for sumcheck tables.
///
/// The fields and operation enums stay private so safe callers cannot select a
/// target-feature implementation that the current CPU does not support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckKernelPlan {
    fp32_fold: Fp32Kernel,
    fp32_product_round: Fp32Kernel,
    fp32_fold_and_product_round: Fp32Kernel,
    fp64_fold: Fp64Kernel,
    fp64_product_round: Fp64Kernel,
    fp64_fold_and_product_round: Fp64Kernel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fp32Kernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512Ifma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum Fp64Kernel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

/// Field-specific operations over canonical sumcheck evaluation tables.
///
/// The default methods are the portable scalar implementations. Field families
/// override only operations with a measured runtime-selected implementation.
/// This keeps protocol code generic without erasing `F` or `E` and keeps CPU
/// dispatch outside the row loop.
pub trait SumcheckTableOperations<F>: ExtField<F> + HasOptimizedFold + HasUnreducedOps
where
    F: FieldCore,
{
    /// Fold one table by its first variable.
    fn fold_first_variable(
        _plan: SumcheckKernelPlan,
        table: &mut EvaluationTable<F, Self>,
        challenge: Self,
    ) where
        Self: Sized,
    {
        fold_first_variable_scalar(table, challenge);
    }

    /// Compute the constant and quadratic coefficients of one product round.
    fn compute_product_round(
        _plan: SumcheckKernelPlan,
        witness: &EvaluationTable<F, Self>,
        factor: &EvaluationTable<F, Self>,
    ) -> (Self, Self)
    where
        Self: Sized,
    {
        compute_product_round_scalar(witness, factor)
    }

    /// Fold two tables and compute their next product round in one pass.
    fn fold_and_compute_product_round(
        _plan: SumcheckKernelPlan,
        witness: &mut EvaluationTable<F, Self>,
        factor: &mut EvaluationTable<F, Self>,
        challenge: Self,
    ) -> (Self, Self)
    where
        Self: Sized,
    {
        fold_and_compute_product_round_scalar(witness, factor, challenge)
    }
}

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

macro_rules! impl_scalar_identity_sumcheck_operations {
    ($base:ident, $modulus:ty) => {
        impl<const P: $modulus> SumcheckTableOperations<$base<P>> for $base<P> {
            fn fold_first_variable(
                _plan: SumcheckKernelPlan,
                table: &mut EvaluationTable<$base<P>, Self>,
                challenge: Self,
            ) {
                fold_first_variable_identity_scalar(table, challenge);
            }

            fn compute_product_round(
                _plan: SumcheckKernelPlan,
                witness: &EvaluationTable<$base<P>, Self>,
                factor: &EvaluationTable<$base<P>, Self>,
            ) -> (Self, Self) {
                compute_product_round_identity_scalar(witness, factor)
            }

            fn fold_and_compute_product_round(
                _plan: SumcheckKernelPlan,
                witness: &mut EvaluationTable<$base<P>, Self>,
                factor: &mut EvaluationTable<$base<P>, Self>,
                challenge: Self,
            ) -> (Self, Self) {
                fold_and_compute_product_round_identity_scalar(witness, factor, challenge)
            }
        }
    };
}

macro_rules! impl_scalar_fp_ext2_sumcheck_operations {
    ($base:ident, $modulus:ty) => {
        impl<const P: $modulus, C> SumcheckTableOperations<$base<P>> for FpExt2<$base<P>, C> where
            C: FpExt2Config<$base<P>>
        {
        }
    };
}

macro_rules! impl_scalar_fp_ext4_sumcheck_operations {
    ($base:ident, $modulus:ty) => {
        impl<const P: $modulus> SumcheckTableOperations<$base<P>> for FpExt4<$base<P>> {}
    };
}

macro_rules! impl_scalar_fp_ext8_sumcheck_operations {
    ($base:ident, $modulus:ty) => {
        impl<const P: $modulus> SumcheckTableOperations<$base<P>> for FpExt8<$base<P>> {}
    };
}

impl_scalar_identity_sumcheck_operations!(Fp32, u32);
impl_scalar_identity_sumcheck_operations!(Fp64, u64);
impl_scalar_identity_sumcheck_operations!(Fp128, u128);
impl_scalar_fp_ext2_sumcheck_operations!(Fp32, u32);
impl_scalar_fp_ext2_sumcheck_operations!(Fp128, u128);
impl_scalar_fp_ext4_sumcheck_operations!(Fp64, u64);
impl_scalar_fp_ext4_sumcheck_operations!(Fp128, u128);
impl_scalar_fp_ext8_sumcheck_operations!(Fp32, u32);
impl_scalar_fp_ext8_sumcheck_operations!(Fp64, u64);
impl_scalar_fp_ext8_sumcheck_operations!(Fp128, u128);

impl SumcheckKernelPlan {
    /// Detect the fastest supported implementation for each operation.
    pub fn detect() -> Self {
        let fp32 = detect_fp32_kernel();
        let fp64 = detect_fp64_kernel();
        Self {
            fp32_fold: fp32,
            fp32_product_round: fp32,
            fp32_fold_and_product_round: fp32,
            fp64_fold: fp64,
            fp64_product_round: Fp64Kernel::Scalar,
            fp64_fold_and_product_round: fp64,
        }
    }

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

    #[cfg(test)]
    const SCALAR: Self = Self {
        fp32_fold: Fp32Kernel::Scalar,
        fp32_product_round: Fp32Kernel::Scalar,
        fp32_fold_and_product_round: Fp32Kernel::Scalar,
        fp64_fold: Fp64Kernel::Scalar,
        fp64_product_round: Fp64Kernel::Scalar,
        fp64_fold_and_product_round: Fp64Kernel::Scalar,
    };

    #[cfg(all(test, target_arch = "aarch64"))]
    const NEON: Self = Self {
        fp32_fold: Fp32Kernel::Neon,
        fp32_product_round: Fp32Kernel::Neon,
        fp32_fold_and_product_round: Fp32Kernel::Neon,
        fp64_fold: Fp64Kernel::Neon,
        fp64_product_round: Fp64Kernel::Neon,
        fp64_fold_and_product_round: Fp64Kernel::Neon,
    };

    #[cfg(all(test, target_arch = "x86_64"))]
    const AVX2: Self = Self {
        fp32_fold: Fp32Kernel::Avx2,
        fp32_product_round: Fp32Kernel::Avx2,
        fp32_fold_and_product_round: Fp32Kernel::Avx2,
        fp64_fold: Fp64Kernel::Avx2,
        fp64_product_round: Fp64Kernel::Avx2,
        fp64_fold_and_product_round: Fp64Kernel::Avx2,
    };

    #[cfg(all(test, target_arch = "x86_64"))]
    const AVX512_IFMA: Self = Self {
        fp32_fold: Fp32Kernel::Avx512Ifma,
        fp32_product_round: Fp32Kernel::Avx512Ifma,
        fp32_fold_and_product_round: Fp32Kernel::Avx512Ifma,
        fp64_fold: Fp64Kernel::Avx512,
        fp64_product_round: Fp64Kernel::Avx512,
        fp64_fold_and_product_round: Fp64Kernel::Avx512,
    };
}

fn detect_fp32_kernel() -> Fp32Kernel {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return Fp32Kernel::Neon;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512ifma")
        {
            return Fp32Kernel::Avx512Ifma;
        }
        if std::is_x86_feature_detected!("avx2") {
            return Fp32Kernel::Avx2;
        }
    }

    Fp32Kernel::Scalar
}

fn detect_fp64_kernel() -> Fp64Kernel {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return Fp64Kernel::Neon;
    }

    Fp64Kernel::Scalar
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

fn fold_first_variable_identity_scalar<F>(table: &mut EvaluationTable<F, F>, challenge: F)
where
    F: FieldCore + ExtField<F> + HasOptimizedFold,
{
    assert!(
        table.len().is_power_of_two(),
        "evaluation table length must be a power of two"
    );
    assert!(
        table.len() >= 2,
        "evaluation table must have at least two rows"
    );
    let half = table.len() / 2;
    let fold = F::precompute_fold(challenge);
    let [values] = table.coefficient_slices_mut::<1>();
    for row in 0..half {
        values[row] = F::fold_one(&fold, values[row], values[row + half]);
    }
    table.truncate(half);
}

fn compute_product_round_identity_scalar<F>(
    witness: &EvaluationTable<F, F>,
    factor: &EvaluationTable<F, F>,
) -> (F, F)
where
    F: FieldCore + ExtField<F> + HasUnreducedOps,
{
    validate_identity_product_round_tables(witness, factor, 2);
    if F::DELAYED_PRODUCT_SUM_IS_EXACT {
        compute_product_round_identity_scalar_with::<F, DelayedProductRoundAccumulator<F>>(
            witness, factor,
        )
    } else {
        compute_product_round_identity_scalar_with::<F, DirectProductRoundAccumulator<F>>(
            witness, factor,
        )
    }
}

fn compute_product_round_identity_scalar_with<F, A>(
    witness: &EvaluationTable<F, F>,
    factor: &EvaluationTable<F, F>,
) -> (F, F)
where
    F: FieldCore + ExtField<F> + HasUnreducedOps,
    A: ProductRoundAccumulator<F>,
{
    let half = witness.len() / 2;
    let [witness] = witness.coefficient_slices::<1>();
    let [factor] = factor.coefficient_slices::<1>();
    let mut accumulator = A::zero();
    for row in 0..half {
        let witness_0 = witness[row];
        let witness_1 = witness[row + half];
        let factor_0 = factor[row];
        let factor_1 = factor[row + half];
        accumulator.add_constant_product(witness_0, factor_0);
        accumulator.add_quadratic_product(witness_1 - witness_0, factor_1 - factor_0);
    }
    accumulator.finish()
}

fn fold_and_compute_product_round_identity_scalar<F>(
    witness: &mut EvaluationTable<F, F>,
    factor: &mut EvaluationTable<F, F>,
    challenge: F,
) -> (F, F)
where
    F: FieldCore + ExtField<F> + HasOptimizedFold + HasUnreducedOps,
{
    validate_identity_product_round_tables(witness, factor, 4);
    if F::DELAYED_PRODUCT_SUM_IS_EXACT {
        fold_and_compute_product_round_identity_scalar_with::<F, DelayedProductRoundAccumulator<F>>(
            witness, factor, challenge,
        )
    } else {
        fold_and_compute_product_round_identity_scalar_with::<F, DirectProductRoundAccumulator<F>>(
            witness, factor, challenge,
        )
    }
}

fn fold_and_compute_product_round_identity_scalar_with<F, A>(
    witness: &mut EvaluationTable<F, F>,
    factor: &mut EvaluationTable<F, F>,
    challenge: F,
) -> (F, F)
where
    F: FieldCore + ExtField<F> + HasOptimizedFold + HasUnreducedOps,
    A: ProductRoundAccumulator<F>,
{
    let half = witness.len() / 2;
    let quarter = half / 2;
    let fold = F::precompute_fold(challenge);
    let [witness_values] = witness.coefficient_slices_mut::<1>();
    let [factor_values] = factor.coefficient_slices_mut::<1>();
    let mut accumulator = A::zero();

    for row in 0..quarter {
        let witness_0 = F::fold_one(&fold, witness_values[row], witness_values[row + half]);
        let witness_1 = F::fold_one(
            &fold,
            witness_values[row + quarter],
            witness_values[row + quarter + half],
        );
        let factor_0 = F::fold_one(&fold, factor_values[row], factor_values[row + half]);
        let factor_1 = F::fold_one(
            &fold,
            factor_values[row + quarter],
            factor_values[row + quarter + half],
        );
        accumulator.add_constant_product(witness_0, factor_0);
        accumulator.add_quadratic_product(witness_1 - witness_0, factor_1 - factor_0);
        witness_values[row] = witness_0;
        witness_values[row + quarter] = witness_1;
        factor_values[row] = factor_0;
        factor_values[row + quarter] = factor_1;
    }

    witness.truncate(half);
    factor.truncate(half);
    accumulator.finish()
}

fn validate_identity_product_round_tables<F>(
    witness: &EvaluationTable<F, F>,
    factor: &EvaluationTable<F, F>,
    minimum_len: usize,
) where
    F: FieldCore + ExtField<F>,
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
        witness.len() >= minimum_len,
        "product round tables are too short"
    );
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

#[cfg(test)]
mod tests {
    use super::SumcheckKernelPlan;
    use akita_field::{Ext2, ExtField, FpExt4, Prime32Offset99, Prime64Offset59};
    use akita_sumcheck::EvaluationTable;

    type F = Prime32Offset99;
    type E = FpExt4<F>;
    type F64 = Prime64Offset59;
    type E64 = Ext2<F64>;

    fn value(row: usize) -> E {
        E::from_base_fn(|coefficient| F::from_u64((row as u64 + 3) * (coefficient as u64 + 11)))
    }

    fn value64(row: usize) -> E64 {
        E64::from_base_fn(|coefficient| F64::from_u64((row as u64 + 5) * (coefficient as u64 + 13)))
    }

    fn compare_plan(plan: SumcheckKernelPlan) {
        for len in [2, 4, 8, 16, 32, 64, 256] {
            let source: Vec<_> = (0..len).map(value).collect();
            let challenge = value(len + 17);
            let mut expected = EvaluationTable::<F, E>::from_multilinear_evaluations(&source)
                .expect("valid table length");
            let mut actual = expected.clone();
            SumcheckKernelPlan::SCALAR.fold_first_variable_fp32(&mut expected, challenge);
            plan.fold_first_variable_fp32(&mut actual, challenge);
            assert_eq!(actual, expected, "fold mismatch at len={len}");
        }
    }

    fn compare_product_round_plan(plan: SumcheckKernelPlan) {
        for len in [2, 4, 8, 16, 32, 64, 256] {
            let witness = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, value)
                .expect("valid witness table");
            let factor = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, |row| {
                value(row + len + 31)
            })
            .expect("valid factor table");
            let expected = SumcheckKernelPlan::SCALAR.compute_product_round_fp32(&witness, &factor);
            let actual = plan.compute_product_round_fp32(&witness, &factor);
            assert_eq!(actual, expected, "product round mismatch at len={len}");
        }
    }

    fn compare_fused_product_round_plan(plan: SumcheckKernelPlan) {
        for len in [4, 8, 16, 32, 64, 256] {
            let witness = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, value)
                .expect("valid witness table");
            let factor = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(len, |row| {
                value(row + len + 31)
            })
            .expect("valid factor table");
            let challenge = value(len + 73);
            let mut expected_witness = witness.clone();
            let mut expected_factor = factor.clone();
            let expected = SumcheckKernelPlan::SCALAR.fold_and_compute_product_round_fp32(
                &mut expected_witness,
                &mut expected_factor,
                challenge,
            );
            let mut actual_witness = witness;
            let mut actual_factor = factor;
            let actual = plan.fold_and_compute_product_round_fp32(
                &mut actual_witness,
                &mut actual_factor,
                challenge,
            );
            assert_eq!(actual, expected, "fused round mismatch at len={len}");
            assert_eq!(
                actual_witness, expected_witness,
                "fused witness mismatch at len={len}"
            );
            assert_eq!(
                actual_factor, expected_factor,
                "fused factor mismatch at len={len}"
            );
        }
    }

    fn compare_fp64_plan(plan: SumcheckKernelPlan) {
        for len in [2, 4, 8, 16, 32, 64, 256] {
            let source: Vec<_> = (0..len).map(value64).collect();
            let challenge = value64(len + 17);
            let mut expected = EvaluationTable::<F64, E64>::from_multilinear_evaluations(&source)
                .expect("valid table length");
            let mut actual = expected.clone();
            SumcheckKernelPlan::SCALAR.fold_first_variable_fp64(&mut expected, challenge);
            plan.fold_first_variable_fp64(&mut actual, challenge);
            assert_eq!(actual, expected, "fp64 fold mismatch at len={len}");

            let witness = EvaluationTable::<F64, E64>::from_multilinear_evaluation_fn(len, value64)
                .expect("valid witness table");
            let factor = EvaluationTable::<F64, E64>::from_multilinear_evaluation_fn(len, |row| {
                value64(row + len + 31)
            })
            .expect("valid factor table");
            let expected_round =
                SumcheckKernelPlan::SCALAR.compute_product_round_fp64(&witness, &factor);
            let actual_round = plan.compute_product_round_fp64(&witness, &factor);
            assert_eq!(
                actual_round, expected_round,
                "fp64 product round mismatch at len={len}"
            );

            if len >= 4 {
                let mut expected_witness = witness.clone();
                let mut expected_factor = factor.clone();
                let expected_round = SumcheckKernelPlan::SCALAR
                    .fold_and_compute_product_round_fp64(
                        &mut expected_witness,
                        &mut expected_factor,
                        challenge,
                    );
                let mut actual_witness = witness;
                let mut actual_factor = factor;
                let actual_round = plan.fold_and_compute_product_round_fp64(
                    &mut actual_witness,
                    &mut actual_factor,
                    challenge,
                );
                assert_eq!(
                    actual_round, expected_round,
                    "fp64 fused round mismatch at len={len}"
                );
                assert_eq!(actual_witness, expected_witness);
                assert_eq!(actual_factor, expected_factor);
            }
        }
    }

    #[test]
    fn detected_fp32_fold_matches_scalar() {
        compare_plan(SumcheckKernelPlan::detect());
        compare_product_round_plan(SumcheckKernelPlan::detect());
        compare_fused_product_round_plan(SumcheckKernelPlan::detect());
        compare_fp64_plan(SumcheckKernelPlan::detect());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn supported_x86_fp32_folds_match_scalar() {
        if std::is_x86_feature_detected!("avx2") {
            compare_plan(SumcheckKernelPlan::AVX2);
            compare_product_round_plan(SumcheckKernelPlan::AVX2);
            compare_fused_product_round_plan(SumcheckKernelPlan::AVX2);
            compare_fp64_plan(SumcheckKernelPlan::AVX2);
        }
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512ifma")
        {
            compare_plan(SumcheckKernelPlan::AVX512_IFMA);
            compare_product_round_plan(SumcheckKernelPlan::AVX512_IFMA);
            compare_fused_product_round_plan(SumcheckKernelPlan::AVX512_IFMA);
            compare_fp64_plan(SumcheckKernelPlan::AVX512_IFMA);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn supported_neon_fp32_folds_match_scalar() {
        if std::arch::is_aarch64_feature_detected!("neon") {
            compare_plan(SumcheckKernelPlan::NEON);
            compare_product_round_plan(SumcheckKernelPlan::NEON);
            compare_fused_product_round_plan(SumcheckKernelPlan::NEON);
            compare_fp64_plan(SumcheckKernelPlan::NEON);
        }
    }
}
