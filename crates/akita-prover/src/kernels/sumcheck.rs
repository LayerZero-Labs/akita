//! Runtime-selected kernels over canonical sumcheck evaluation tables.

use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{ExtField, FieldCore};
use akita_sumcheck::{
    batched_affine_product_coefficients, compose_polynomial_with_affine,
    compute_product_round_scalar, fold_and_compute_product_round_scalar,
    fold_first_variable_scalar, EvaluationTable, MAX_AFFINE_POLYNOMIAL_DEGREE,
    MAX_AFFINE_PRODUCT_DEGREE,
};

mod fp32;
mod fp32_affine;
mod fp64;
mod scalar;

/// Host-detected operation choices for sumcheck tables.
///
/// The fields and operation enums stay private so safe callers cannot select a
/// target-feature implementation that the current CPU does not support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckKernelPlan {
    pub(super) fp32_fold: Fp32Kernel,
    pub(super) fp32_product_round: Fp32Kernel,
    pub(super) fp32_fold_and_product_round: Fp32Kernel,
    pub(super) fp64_fold: Fp64Kernel,
    pub(super) fp64_product_round: Fp64Kernel,
    pub(super) fp64_fold_and_product_round: Fp64Kernel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fp32Kernel {
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
pub(super) enum Fp64Kernel {
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

    /// Compute a weighted round for a batch of quadratic or quartic products.
    fn compute_weighted_affine_product_round<const LANES: usize>(
        _plan: SumcheckKernelPlan,
        lanes: &[EvaluationTable<F, Self>; LANES],
        equality: &EvaluationTable<F, Self>,
        arity: usize,
        parent_weights: &[Self],
    ) -> [Self; MAX_AFFINE_PRODUCT_DEGREE + 1]
    where
        Self: Sized,
    {
        compute_weighted_affine_product_round_scalar(lanes, equality, arity, parent_weights)
    }

    /// Compute an equality-weighted round for a degree-at-most-four polynomial.
    fn compute_weighted_affine_polynomial_round(
        _plan: SumcheckKernelPlan,
        values: &EvaluationTable<F, Self>,
        equality: &EvaluationTable<F, Self>,
        polynomial_coefficients: &[Self],
    ) -> [Self; MAX_AFFINE_POLYNOMIAL_DEGREE + 1]
    where
        Self: Sized,
    {
        compute_weighted_affine_polynomial_round_scalar(values, equality, polynomial_coefficients)
    }

    /// Try a field-specific compact class-indexed affine-product round.
    ///
    /// Returning `None` selects the protocol's exact portable blocked
    /// accumulator without changing the compact state.
    fn try_compute_compact_affine_product_round<const LANES: usize>(
        _plan: SumcheckKernelPlan,
        _ordered_pair_indices: &[u16],
        _folded_pair_rows: &[[Self; LANES]],
        _first_equality: &[Self],
        _second_equality: &[Self],
        _arity: usize,
        _parent_weights: &[Self],
    ) -> Option<[Self; MAX_AFFINE_PRODUCT_DEGREE + 1]>
    where
        Self: Sized,
    {
        None
    }

    /// Try a field-specific round over class-coded polynomial values.
    ///
    /// Returning `None` keeps the protocol on its exact scalar accumulator.
    fn try_compute_class_coded_affine_polynomial_round(
        _plan: SumcheckKernelPlan,
        _class_codes: &[u16],
        _class_values: &[Self],
        _class_taylor_coefficients: &[[Self; 4]],
        _first_equality: &[Self],
        _second_equality: &[Self],
        _degree: usize,
    ) -> Option<[Self; MAX_AFFINE_POLYNOMIAL_DEGREE + 1]>
    where
        Self: Sized,
    {
        None
    }

    /// Try folding class-coded values and computing their next sparse-prefix round.
    fn try_fold_class_coded_and_compute_sparse_affine_polynomial_round(
        _plan: SumcheckKernelPlan,
        _class_codes: &[u16],
        _class_values: &[Self],
        _folded_values: &mut [Self],
        _split_equality: (&[Self], &[Self]),
        _challenge: Self,
        _degree: usize,
    ) -> Option<[Self; MAX_AFFINE_POLYNOMIAL_DEGREE + 1]>
    where
        Self: Sized,
    {
        None
    }

    /// Try folding a sparse-prefix value table and computing its next round.
    fn try_fold_and_compute_sparse_affine_polynomial_round(
        _plan: SumcheckKernelPlan,
        _values: &[Self],
        _folded_values: &mut [Self],
        _first_equality: &[Self],
        _second_equality: &[Self],
        _challenge: Self,
        _degree: usize,
    ) -> Option<[Self; MAX_AFFINE_POLYNOMIAL_DEGREE + 1]>
    where
        Self: Sized,
    {
        None
    }
}

fn compute_weighted_affine_polynomial_round_scalar<F, E>(
    values: &EvaluationTable<F, E>,
    equality: &EvaluationTable<F, E>,
    polynomial_coefficients: &[E],
) -> [E; MAX_AFFINE_POLYNOMIAL_DEGREE + 1]
where
    F: FieldCore,
    E: ExtField<F>,
{
    assert!(
        values.len().is_power_of_two() && values.len() >= 2,
        "polynomial values must have a nontrivial power-of-two length"
    );
    assert_eq!(equality.len(), values.len() / 2);
    assert!(polynomial_coefficients.len() <= MAX_AFFINE_POLYNOMIAL_DEGREE + 1);

    let half = values.len() / 2;
    let mut result = [E::zero(); MAX_AFFINE_POLYNOMIAL_DEGREE + 1];
    for row in 0..half {
        let left = values.evaluation(row);
        let right = values.evaluation(row + half);
        let coefficients =
            compose_polynomial_with_affine(polynomial_coefficients, left, right - left);
        let equality_weight = equality.evaluation(row);
        for degree in 0..polynomial_coefficients.len() {
            result[degree] += equality_weight * coefficients[degree];
        }
    }
    result
}

fn compute_weighted_affine_product_round_scalar<F, E, const LANES: usize>(
    lanes: &[EvaluationTable<F, E>; LANES],
    equality: &EvaluationTable<F, E>,
    arity: usize,
    parent_weights: &[E],
) -> [E; MAX_AFFINE_PRODUCT_DEGREE + 1]
where
    F: FieldCore,
    E: ExtField<F>,
{
    assert!(matches!(arity, 2 | 4), "product arity must be two or four");
    assert_eq!(LANES, arity * parent_weights.len());
    let table_len = lanes[0].len();
    assert!(
        lanes.iter().all(|lane| lane.len() == table_len),
        "product lane tables must have equal lengths"
    );
    assert_eq!(equality.len(), table_len / 2);

    let half = table_len / 2;
    let mut result = [E::zero(); MAX_AFFINE_PRODUCT_DEGREE + 1];
    for row in 0..half {
        let left = std::array::from_fn::<_, LANES, _>(|lane| lanes[lane].evaluation(row));
        let right = std::array::from_fn::<_, LANES, _>(|lane| lanes[lane].evaluation(row + half));
        let coefficients =
            batched_affine_product_coefficients(&left, &right, arity, parent_weights);
        let equality_weight = equality.evaluation(row);
        for degree in 0..=arity {
            result[degree] += equality_weight * coefficients[degree];
        }
    }
    result
}

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

#[cfg(test)]
mod tests;
