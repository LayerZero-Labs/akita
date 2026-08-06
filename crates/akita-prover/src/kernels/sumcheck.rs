//! Runtime-selected kernels over canonical sumcheck evaluation tables.

use akita_field::{Fp32, FpExt4};
use akita_sumcheck::{fold_first_variable_scalar, EvaluationTable};

/// Host-detected operation choices for sumcheck tables.
///
/// The fields and operation enums stay private so safe callers cannot select a
/// target-feature implementation that the current CPU does not support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckKernelPlan {
    fp32_fold: Fp32FoldKernel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fp32FoldKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512Ifma,
}

impl SumcheckKernelPlan {
    /// Detect the fastest supported implementation for each operation.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512dq")
                && std::is_x86_feature_detected!("avx512ifma")
            {
                return Self {
                    fp32_fold: Fp32FoldKernel::Avx512Ifma,
                };
            }
            if std::is_x86_feature_detected!("avx2") {
                return Self {
                    fp32_fold: Fp32FoldKernel::Avx2,
                };
            }
        }

        Self {
            fp32_fold: Fp32FoldKernel::Scalar,
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
            Fp32FoldKernel::Scalar => fold_first_variable_scalar(table, challenge),
            #[cfg(target_arch = "x86_64")]
            Fp32FoldKernel::Avx2 => {
                if table.len() / 2 < 8 {
                    fold_first_variable_scalar(table, challenge);
                } else {
                    // SAFETY: only `detect` constructs production plans, and
                    // it selects this variant after checking AVX2 support.
                    unsafe { fold_fp32_avx2(table, challenge) };
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32FoldKernel::Avx512Ifma => {
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

    #[cfg(test)]
    const SCALAR: Self = Self {
        fp32_fold: Fp32FoldKernel::Scalar,
    };

    #[cfg(all(test, target_arch = "x86_64"))]
    const AVX2: Self = Self {
        fp32_fold: Fp32FoldKernel::Avx2,
    };

    #[cfg(all(test, target_arch = "x86_64"))]
    const AVX512_IFMA: Self = Self {
        fp32_fold: Fp32FoldKernel::Avx512Ifma,
    };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fold_fp32_avx2<const P: u32>(
    table: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) {
    let half = table.len() / 2;
    let [coefficient_0, coefficient_1, coefficient_2, coefficient_3] =
        table.coefficient_slices_mut::<4>();
    let (left_0, right_0) = coefficient_0.split_at_mut(half);
    let (left_1, right_1) = coefficient_1.split_at_mut(half);
    let (left_2, right_2) = coefficient_2.split_at_mut(half);
    let (left_3, right_3) = coefficient_3.split_at_mut(half);

    // SAFETY: this function requires AVX2, and every left/right pair comes
    // from equal halves of a power-of-two table with `half >= 8`.
    unsafe {
        akita_field::packed::runtime_x86::fold_fp_ext4_fp32_avx2(
            [left_0, left_1, left_2, left_3],
            [right_0, right_1, right_2, right_3],
            challenge,
        )
    };
    table.truncate(half);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn fold_fp32_avx512_ifma<const P: u32>(
    table: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    challenge: FpExt4<Fp32<P>>,
) {
    let half = table.len() / 2;
    let [coefficient_0, coefficient_1, coefficient_2, coefficient_3] =
        table.coefficient_slices_mut::<4>();
    let (left_0, right_0) = coefficient_0.split_at_mut(half);
    let (left_1, right_1) = coefficient_1.split_at_mut(half);
    let (left_2, right_2) = coefficient_2.split_at_mut(half);
    let (left_3, right_3) = coefficient_3.split_at_mut(half);

    // SAFETY: this function requires AVX-512F, DQ, and IFMA, and every
    // left/right pair comes from equal halves of a power-of-two table with
    // `half >= 16`.
    unsafe {
        akita_field::packed::runtime_x86::fold_fp_ext4_fp32_avx512_ifma(
            [left_0, left_1, left_2, left_3],
            [right_0, right_1, right_2, right_3],
            challenge,
        )
    };
    table.truncate(half);
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

    #[test]
    fn detected_fp32_fold_matches_scalar() {
        compare_plan(SumcheckKernelPlan::detect());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn supported_x86_fp32_folds_match_scalar() {
        if std::is_x86_feature_detected!("avx2") {
            compare_plan(SumcheckKernelPlan::AVX2);
        }
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512ifma")
        {
            compare_plan(SumcheckKernelPlan::AVX512_IFMA);
        }
    }
}
