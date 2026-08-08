//! Portable operations for identity fields and extension families without selected SIMD kernels.

#[cfg(feature = "parallel")]
use super::multiple_workers_available;
use super::{SumcheckKernelPlan, SumcheckTableOperations};
#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{ExtField, FieldCore, Fp128, Fp32, Fp64, FpExt2, FpExt2Config, FpExt4, FpExt8};
use akita_sumcheck::{
    DelayedProductRoundAccumulator, DirectProductRoundAccumulator, EvaluationTable,
    ProductRoundAccumulator,
};

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
    #[cfg(feature = "parallel")]
    if multiple_workers_available() {
        let (left, right) = values.split_at_mut(half);
        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(left, &right)| *left = F::fold_one(&fold, *left, right));
        table.truncate(half);
        return;
    }
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
    #[cfg(feature = "parallel")]
    if multiple_workers_available() {
        return (0..half)
            .into_par_iter()
            .fold(A::zero, |mut accumulator, row| {
                let witness_0 = witness[row];
                let witness_1 = witness[row + half];
                let factor_0 = factor[row];
                let factor_1 = factor[row + half];
                accumulator.add_constant_product(witness_0, factor_0);
                accumulator.add_quadratic_product(witness_1 - witness_0, factor_1 - factor_0);
                accumulator
            })
            .reduce(A::zero, A::merge)
            .finish();
    }
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
    #[cfg(feature = "parallel")]
    if multiple_workers_available() {
        fold_first_variable_identity_scalar(witness, challenge);
        fold_first_variable_identity_scalar(factor, challenge);
        return compute_product_round_identity_scalar_with::<F, A>(witness, factor);
    }
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
