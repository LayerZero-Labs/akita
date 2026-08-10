//! Runtime-selected kernels over canonical sumcheck evaluation tables.

use akita_algebra::SplitEqEvals;
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, ExtField, FieldCore, MulBaseUnreduced};
use akita_sumcheck::{
    batched_affine_product_coefficients, compose_polynomial_with_affine,
    compute_product_round_scalar, fold_and_compute_product_round_scalar,
    fold_first_variable_scalar, DelayedProductSum, DirectProductSum, EvaluationTable,
    ProductSumAccumulator, MAX_AFFINE_POLYNOMIAL_DEGREE, MAX_AFFINE_PRODUCT_DEGREE,
};
use akita_types::TensorFactorProjection;

mod fp32;
mod fp32_affine;
mod fp64;
mod scalar;
#[cfg(feature = "parallel")]
mod stage2_parallel;

#[inline]
fn multiple_workers_available() -> bool {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads() > 1
    }
    #[cfg(not(feature = "parallel"))]
    {
        false
    }
}

#[cfg(feature = "parallel")]
#[inline]
fn parallel_simd_rows(len: usize, simd_width: usize) -> usize {
    debug_assert!(len.is_multiple_of(simd_width));
    let target_tasks = rayon::current_num_threads().saturating_mul(4).max(1);
    len.div_ceil(target_tasks)
        .max(1_024)
        .next_multiple_of(simd_width)
}

/// Host-detected operation choices for sumcheck tables.
///
/// The fields and operation enums stay private so safe callers cannot select a
/// target-feature implementation that the current CPU does not support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckKernelPlan {
    pub(super) fp32_fold: Fp32Kernel,
    pub(super) fp32_product_round: Fp32Kernel,
    pub(super) fp32_fold_and_product_round: Fp32Kernel,
    pub(super) fp32_stage2_coefficient_round: Fp32Kernel,
    pub(super) fp32_tensor_factor_round: Fp32Kernel,
    pub(super) fp64_fold: Fp64Kernel,
    pub(super) fp64_product_round: Fp64Kernel,
    pub(super) fp64_fold_and_product_round: Fp64Kernel,
    pub(super) fp64_tensor_factor_round: Fp64Kernel,
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

/// Canonical tensor-factor table plus an optional cached first product round.
pub type TensorFactorRoundOutput<F, E> = (EvaluationTable<F, E>, Option<(E, E)>);

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
    /// Materialize the transparent tensor factor directly in canonical table
    /// form while computing its first product round with `witness`.
    fn materialize_tensor_factor_and_compute_product_round(
        _plan: SumcheckKernelPlan,
        witness: &EvaluationTable<F, Self>,
        tail_point: &[Self],
        projection: &TensorFactorProjection<F, Self>,
    ) -> Result<TensorFactorRoundOutput<F, Self>, AkitaError>
    where
        Self: Sized + MulBaseUnreduced<F>,
    {
        materialize_tensor_factor_and_compute_product_round_scalar(witness, tail_point, projection)
    }

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

    /// Fold one Stage 2 coefficient coordinate in place and compute the next
    /// norm and ordinary-relation round from the folded witness.
    #[allow(clippy::too_many_arguments)]
    fn fold_and_compute_stage2_coefficient_round(
        _plan: SumcheckKernelPlan,
        witness: &mut EvaluationTable<F, Self>,
        live_lane_count: usize,
        old_coefficient_count: usize,
        next_alpha_factor: &[Self],
        relation_lane_weights: &[Self],
        first_equality: &[Self],
        second_equality: &[Self],
        challenge: Self,
        include_norm_linear: bool,
    ) -> ([Self; 3], [Self; 3])
    where
        Self: Sized,
    {
        fold_and_compute_stage2_coefficient_round_portable(
            witness,
            live_lane_count,
            old_coefficient_count,
            next_alpha_factor,
            relation_lane_weights,
            first_equality,
            second_equality,
            challenge,
            include_norm_linear,
        )
    }
}

fn materialize_tensor_factor_and_compute_product_round_scalar<F, E>(
    witness: &EvaluationTable<F, E>,
    tail_point: &[E],
    projection: &TensorFactorProjection<F, E>,
) -> Result<TensorFactorRoundOutput<F, E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F> + HasUnreducedOps + MulBaseUnreduced<F>,
{
    let factor = materialize_tensor_factor(witness, tail_point, projection)?;
    let round = (witness.len() >= 2).then(|| compute_product_round_scalar(witness, &factor));
    Ok((factor, round))
}

pub(super) fn materialize_tensor_factor<F, E>(
    witness: &EvaluationTable<F, E>,
    tail_point: &[E],
    projection: &TensorFactorProjection<F, E>,
) -> Result<EvaluationTable<F, E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let shift = u32::try_from(tail_point.len()).map_err(|_| AkitaError::InvalidSize {
        expected: usize::BITS as usize,
        actual: tail_point.len(),
    })?;
    let expected = 1usize.checked_shl(shift).ok_or_else(|| {
        AkitaError::InvalidInput("tensor factor table length overflow".to_string())
    })?;
    if witness.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: witness.len(),
        });
    }

    let reversed_suffix = tail_point[usize::from(!tail_point.is_empty())..]
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let equality = SplitEqEvals::new(&reversed_suffix)?;
    let suffix_len = equality.len();
    Ok(EvaluationTable::from_evaluation_fn(
        expected,
        |stored_row| {
            if tail_point.is_empty() {
                return projection.project(E::one());
            }
            let suffix_row = stored_row % suffix_len;
            let suffix = equality.e_out[suffix_row / equality.in_len()]
                * equality.e_in[suffix_row % equality.in_len()];
            let branch = if stored_row < suffix_len {
                E::one() - tail_point[0]
            } else {
                tail_point[0]
            };
            projection.project(branch * suffix)
        },
    ))
}

/// Return one contiguous block that shares a value from the outer split
/// equality table.
#[inline]
pub(crate) fn stage2_equality_block(
    address_base: usize,
    block_start: usize,
    first_len: usize,
    first_bits: usize,
    block_size: usize,
    live_pairs: usize,
) -> (usize, usize) {
    debug_assert!(first_len.is_power_of_two());
    let address = address_base + block_start;
    let second_index = address >> first_bits;
    let bucket_remaining = first_len - (address & (first_len - 1));
    let block_end = (block_start + block_size.min(bucket_remaining)).min(live_pairs);
    (second_index, block_end)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fold_and_compute_stage2_coefficient_round_portable<F, E>(
    witness: &mut EvaluationTable<F, E>,
    live_lane_count: usize,
    old_coefficient_count: usize,
    next_alpha_factor: &[E],
    relation_lane_weights: &[E],
    first_equality: &[E],
    second_equality: &[E],
    challenge: E,
    include_norm_linear: bool,
) -> ([E; 3], [E; 3])
where
    F: FieldCore,
    E: ExtField<F> + HasOptimizedFold + HasUnreducedOps,
{
    #[cfg(feature = "parallel")]
    if multiple_workers_available() {
        return stage2_parallel::fold_and_compute_stage2_coefficient_round_parallel(
            witness,
            live_lane_count,
            old_coefficient_count,
            next_alpha_factor,
            relation_lane_weights,
            first_equality,
            second_equality,
            challenge,
            include_norm_linear,
        );
    }
    if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        fold_and_compute_stage2_coefficient_round_with::<F, E, DelayedProductSum<E>>(
            witness,
            live_lane_count,
            old_coefficient_count,
            next_alpha_factor,
            relation_lane_weights,
            first_equality,
            second_equality,
            challenge,
            include_norm_linear,
        )
    } else {
        fold_and_compute_stage2_coefficient_round_with::<F, E, DirectProductSum<E>>(
            witness,
            live_lane_count,
            old_coefficient_count,
            next_alpha_factor,
            relation_lane_weights,
            first_equality,
            second_equality,
            challenge,
            include_norm_linear,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn fold_and_compute_stage2_coefficient_round_with<F, E, A>(
    witness: &mut EvaluationTable<F, E>,
    live_lane_count: usize,
    old_coefficient_count: usize,
    next_alpha_factor: &[E],
    relation_lane_weights: &[E],
    first_equality: &[E],
    second_equality: &[E],
    challenge: E,
    include_norm_linear: bool,
) -> ([E; 3], [E; 3])
where
    F: FieldCore,
    E: ExtField<F> + HasOptimizedFold + HasUnreducedOps,
    A: ProductSumAccumulator<E>,
{
    assert!(old_coefficient_count.is_power_of_two() && old_coefficient_count >= 4);
    assert_eq!(witness.len(), live_lane_count * old_coefficient_count);
    assert!(relation_lane_weights.len() >= live_lane_count);
    assert!(first_equality.len().is_power_of_two());
    let next_coefficient_count = old_coefficient_count / 2;
    let next_pair_count = next_coefficient_count / 2;
    assert_eq!(next_alpha_factor.len(), next_coefficient_count);
    assert!(
        first_equality.len() * second_equality.len() >= live_lane_count * next_pair_count,
        "split equality table does not cover the live Stage 2 rows"
    );

    let fold = E::precompute_fold(challenge);
    let mut total_norm: [A; 3] = std::array::from_fn(|_| A::zero());
    let mut total_relation: [A; 3] = std::array::from_fn(|_| A::zero());

    let old_half = live_lane_count * next_coefficient_count;
    let next_half = live_lane_count * next_pair_count;
    let lanes_use_binding_order = live_lane_count == relation_lane_weights.len();
    for stored_pair in 0..next_pair_count {
        let logical_pair = reverse_power_of_two_index(stored_pair, next_pair_count);
        let alpha_0 = next_alpha_factor[stored_pair];
        let alpha_delta = next_alpha_factor[stored_pair + next_pair_count] - alpha_0;
        let pair_start = stored_pair * live_lane_count;
        for (stored_lane, &lane_weight) in relation_lane_weights
            .iter()
            .take(live_lane_count)
            .enumerate()
        {
            let row_0 = pair_start + stored_lane;
            let row_1 = row_0 + next_half;
            let witness_0 = E::fold_one(
                &fold,
                witness.evaluation(row_0),
                witness.evaluation(row_0 + old_half),
            );
            let witness_1 = E::fold_one(
                &fold,
                witness.evaluation(row_1),
                witness.evaluation(row_1 + old_half),
            );
            witness.set_evaluation(row_0, witness_0);
            witness.set_evaluation(row_1, witness_1);

            let logical_lane = if lanes_use_binding_order {
                reverse_power_of_two_index(stored_lane, live_lane_count)
            } else {
                stored_lane
            };
            let equality_address = logical_lane * next_pair_count + logical_pair;
            let equality = first_equality[equality_address & (first_equality.len() - 1)]
                * second_equality[equality_address / first_equality.len()];
            add_stage2_round_terms(
                &mut total_norm,
                &mut total_relation,
                witness_0,
                witness_1,
                equality,
                lane_weight,
                alpha_0,
                alpha_delta,
                include_norm_linear,
            );
        }
    }
    witness.truncate(live_lane_count * next_coefficient_count);
    (total_norm.map(A::finish), total_relation.map(A::finish))
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn add_stage2_round_terms<E, A>(
    norm: &mut [A; 3],
    relation: &mut [A; 3],
    witness_0: E,
    witness_1: E,
    equality: E,
    lane_weight: E,
    alpha_0: E,
    alpha_delta: E,
    include_norm_linear: bool,
) where
    E: FieldCore + HasUnreducedOps,
    A: ProductSumAccumulator<E>,
{
    let witness_delta = witness_1 - witness_0;
    norm[0].add_product(equality, witness_0 * (witness_0 + E::one()));
    if include_norm_linear {
        norm[1].add_product(equality, witness_delta * (witness_0 + witness_0 + E::one()));
    }
    norm[2].add_product(equality, witness_delta * witness_delta);

    relation[0].add_product(witness_0, lane_weight * alpha_0);
    relation[1].add_product(witness_0, lane_weight * alpha_delta);
    relation[1].add_product(witness_delta, lane_weight * alpha_0);
    relation[2].add_product(witness_delta, lane_weight * alpha_delta);
}

#[inline]
fn reverse_power_of_two_index(index: usize, len: usize) -> usize {
    debug_assert!(len.is_power_of_two());
    if len <= 1 {
        0
    } else {
        index.reverse_bits() >> (usize::BITS - len.trailing_zeros())
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
    cfg_fold_reduce!(
        0..half,
        || [E::zero(); MAX_AFFINE_POLYNOMIAL_DEGREE + 1],
        |mut result, row| {
            let left = values.evaluation(row);
            let right = values.evaluation(row + half);
            let coefficients =
                compose_polynomial_with_affine(polynomial_coefficients, left, right - left);
            let equality_weight = equality.evaluation(row);
            for degree in 0..polynomial_coefficients.len() {
                result[degree] += equality_weight * coefficients[degree];
            }
            result
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        }
    )
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
    cfg_fold_reduce!(
        0..half,
        || [E::zero(); MAX_AFFINE_PRODUCT_DEGREE + 1],
        |mut result, row| {
            let left = std::array::from_fn::<_, LANES, _>(|lane| lanes[lane].evaluation(row));
            let right =
                std::array::from_fn::<_, LANES, _>(|lane| lanes[lane].evaluation(row + half));
            let coefficients =
                batched_affine_product_coefficients(&left, &right, arity, parent_weights);
            let equality_weight = equality.evaluation(row);
            for degree in 0..=arity {
                result[degree] += equality_weight * coefficients[degree];
            }
            result
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        }
    )
}

impl SumcheckKernelPlan {
    /// Detect the fastest supported implementation for each operation.
    pub fn detect() -> Self {
        #[cfg(test)]
        if let Some(plan) = TEST_PLAN_OVERRIDE.with(std::cell::Cell::get) {
            return plan;
        }

        let fp32 = detect_fp32_kernel();
        let fp64 = detect_fp64_kernel();
        Self {
            fp32_fold: fp32,
            fp32_product_round: fp32,
            fp32_fold_and_product_round: fp32,
            fp32_stage2_coefficient_round: fp32,
            fp32_tensor_factor_round: fp32,
            fp64_fold: fp64,
            fp64_product_round: Fp64Kernel::Scalar,
            fp64_fold_and_product_round: fp64,
            fp64_tensor_factor_round: fp64,
        }
    }

    #[cfg(test)]
    pub(crate) const SCALAR: Self = Self {
        fp32_fold: Fp32Kernel::Scalar,
        fp32_product_round: Fp32Kernel::Scalar,
        fp32_fold_and_product_round: Fp32Kernel::Scalar,
        fp32_stage2_coefficient_round: Fp32Kernel::Scalar,
        fp32_tensor_factor_round: Fp32Kernel::Scalar,
        fp64_fold: Fp64Kernel::Scalar,
        fp64_product_round: Fp64Kernel::Scalar,
        fp64_fold_and_product_round: Fp64Kernel::Scalar,
        fp64_tensor_factor_round: Fp64Kernel::Scalar,
    };

    /// Run one test with a forced plan without leaking the override to other
    /// test threads or leaving it installed after a panic.
    #[cfg(test)]
    pub(crate) fn with_test_override<R>(plan: Self, test: impl FnOnce() -> R) -> R {
        TEST_PLAN_OVERRIDE.with(|slot| {
            struct ResetOverride<'a> {
                slot: &'a std::cell::Cell<Option<SumcheckKernelPlan>>,
                previous: Option<SumcheckKernelPlan>,
            }

            impl Drop for ResetOverride<'_> {
                fn drop(&mut self) {
                    self.slot.set(self.previous);
                }
            }

            let previous = slot.replace(Some(plan));
            let _reset = ResetOverride { slot, previous };
            test()
        })
    }

    #[cfg(all(test, target_arch = "aarch64"))]
    const NEON: Self = Self {
        fp32_fold: Fp32Kernel::Neon,
        fp32_product_round: Fp32Kernel::Neon,
        fp32_fold_and_product_round: Fp32Kernel::Neon,
        fp32_stage2_coefficient_round: Fp32Kernel::Neon,
        fp32_tensor_factor_round: Fp32Kernel::Neon,
        fp64_fold: Fp64Kernel::Neon,
        fp64_product_round: Fp64Kernel::Neon,
        fp64_fold_and_product_round: Fp64Kernel::Neon,
        fp64_tensor_factor_round: Fp64Kernel::Neon,
    };

    #[cfg(all(test, target_arch = "x86_64"))]
    const AVX2: Self = Self {
        fp32_fold: Fp32Kernel::Avx2,
        fp32_product_round: Fp32Kernel::Avx2,
        fp32_fold_and_product_round: Fp32Kernel::Avx2,
        fp32_stage2_coefficient_round: Fp32Kernel::Avx2,
        fp32_tensor_factor_round: Fp32Kernel::Avx2,
        fp64_fold: Fp64Kernel::Avx2,
        fp64_product_round: Fp64Kernel::Avx2,
        fp64_fold_and_product_round: Fp64Kernel::Avx2,
        fp64_tensor_factor_round: Fp64Kernel::Avx2,
    };

    #[cfg(all(test, target_arch = "x86_64"))]
    const AVX512_IFMA: Self = Self {
        fp32_fold: Fp32Kernel::Avx512Ifma,
        fp32_product_round: Fp32Kernel::Avx512Ifma,
        fp32_fold_and_product_round: Fp32Kernel::Avx512Ifma,
        fp32_stage2_coefficient_round: Fp32Kernel::Avx512Ifma,
        fp32_tensor_factor_round: Fp32Kernel::Avx512Ifma,
        fp64_fold: Fp64Kernel::Avx512,
        fp64_product_round: Fp64Kernel::Avx512,
        fp64_fold_and_product_round: Fp64Kernel::Avx512,
        fp64_tensor_factor_round: Fp64Kernel::Avx512,
    };
}

#[cfg(test)]
thread_local! {
    static TEST_PLAN_OVERRIDE: std::cell::Cell<Option<SumcheckKernelPlan>> = const {
        std::cell::Cell::new(None)
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
