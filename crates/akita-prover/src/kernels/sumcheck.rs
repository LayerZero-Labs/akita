//! Runtime-selected kernels over canonical sumcheck evaluation tables.

use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{ExtField, FieldCore, Fp128, Fp32, Fp64, FpExt2, FpExt2Config, FpExt4, FpExt8};
use akita_sumcheck::{
    compute_product_round_scalar, fold_and_compute_product_round_scalar,
    fold_first_variable_scalar, EvaluationTable,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fp32Kernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512Ifma,
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

macro_rules! impl_scalar_identity_sumcheck_operations {
    ($base:ident, $modulus:ty) => {
        impl<const P: $modulus> SumcheckTableOperations<$base<P>> for $base<P> {}
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
impl_scalar_fp_ext2_sumcheck_operations!(Fp64, u64);
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
        Self {
            fp32_fold: fp32,
            fp32_product_round: fp32,
            fp32_fold_and_product_round: fp32,
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

    #[cfg(test)]
    const SCALAR: Self = Self {
        fp32_fold: Fp32Kernel::Scalar,
        fp32_product_round: Fp32Kernel::Scalar,
        fp32_fold_and_product_round: Fp32Kernel::Scalar,
    };

    #[cfg(all(test, target_arch = "x86_64"))]
    const AVX2: Self = Self {
        fp32_fold: Fp32Kernel::Avx2,
        fp32_product_round: Fp32Kernel::Avx2,
        fp32_fold_and_product_round: Fp32Kernel::Avx2,
    };

    #[cfg(all(test, target_arch = "x86_64"))]
    const AVX512_IFMA: Self = Self {
        fp32_fold: Fp32Kernel::Avx512Ifma,
        fp32_product_round: Fp32Kernel::Avx512Ifma,
        fp32_fold_and_product_round: Fp32Kernel::Avx512Ifma,
    };
}

fn detect_fp32_kernel() -> Fp32Kernel {
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

#[cfg(target_arch = "x86_64")]
fn coefficient_halves<const P: u32>(
    table: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
) -> ([&[Fp32<P>]; 4], [&[Fp32<P>]; 4]) {
    let half = table.len() / 2;
    (
        std::array::from_fn(|coefficient| &table.coefficient_slice(coefficient)[..half]),
        std::array::from_fn(|coefficient| &table.coefficient_slice(coefficient)[half..]),
    )
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

#[cfg(target_arch = "x86_64")]
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

#[cfg(test)]
mod tests {
    use super::SumcheckKernelPlan;
    use akita_field::{ExtField, FpExt4, Prime32Offset99};
    use akita_sumcheck::EvaluationTable;

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn value(row: usize) -> E {
        E::from_base_fn(|coefficient| F::from_u64((row as u64 + 3) * (coefficient as u64 + 11)))
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

    #[test]
    fn detected_fp32_fold_matches_scalar() {
        compare_plan(SumcheckKernelPlan::detect());
        compare_product_round_plan(SumcheckKernelPlan::detect());
        compare_fused_product_round_plan(SumcheckKernelPlan::detect());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn supported_x86_fp32_folds_match_scalar() {
        if std::is_x86_feature_detected!("avx2") {
            compare_plan(SumcheckKernelPlan::AVX2);
            compare_product_round_plan(SumcheckKernelPlan::AVX2);
            compare_fused_product_round_plan(SumcheckKernelPlan::AVX2);
        }
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512ifma")
        {
            compare_plan(SumcheckKernelPlan::AVX512_IFMA);
            compare_product_round_plan(SumcheckKernelPlan::AVX512_IFMA);
            compare_fused_product_round_plan(SumcheckKernelPlan::AVX512_IFMA);
        }
    }
}
