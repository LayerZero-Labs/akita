//! Runtime-selected operations for quadratic extensions of 64-bit fields.

#[cfg(feature = "parallel")]
use super::{materialize_tensor_factor, multiple_workers_available, parallel_simd_rows};
use super::{
    materialize_tensor_factor_and_compute_product_round_scalar, Fp64Kernel, SumcheckKernelPlan,
    SumcheckTableOperations, TensorFactorRoundOutput,
};
use akita_algebra::SplitEqEvals;
#[cfg(feature = "parallel")]
use akita_field::parallel::*;
#[cfg(feature = "parallel")]
use akita_field::unreduced::HasOptimizedFold;
use akita_field::unreduced::HasUnreducedOps;
use akita_field::{AkitaError, Fp64, FpExt2, FpExt2Config};
use akita_sumcheck::{
    fold_and_compute_product_round_scalar, fold_first_variable_scalar,
    DelayedProductRoundAccumulator, DirectProductRoundAccumulator, EvaluationTable,
    ProductRoundAccumulator,
};
use akita_types::TensorFactorProjection;

impl<const P: u64, C> SumcheckTableOperations<Fp64<P>> for FpExt2<Fp64<P>, C>
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    fn materialize_tensor_factor_and_compute_product_round(
        plan: SumcheckKernelPlan,
        witness: &EvaluationTable<Fp64<P>, Self>,
        tail_point: &[Self],
        projection: &TensorFactorProjection<Fp64<P>, Self>,
    ) -> Result<TensorFactorRoundOutput<Fp64<P>, Self>, AkitaError> {
        plan.materialize_tensor_factor_and_compute_product_round_fp64(
            witness, tail_point, projection,
        )
    }

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
    /// Materialize an fp64 tensor factor and compute its first product round in
    /// the same traversal.
    pub fn materialize_tensor_factor_and_compute_product_round_fp64<const P: u64, C>(
        self,
        witness: &EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
        tail_point: &[FpExt2<Fp64<P>, C>],
        projection: &TensorFactorProjection<Fp64<P>, FpExt2<Fp64<P>, C>>,
    ) -> Result<TensorFactorRoundOutput<Fp64<P>, FpExt2<Fp64<P>, C>>, AkitaError>
    where
        C: FpExt2Config<Fp64<P>> + 'static,
    {
        if tail_point.is_empty() {
            return materialize_tensor_factor_and_compute_product_round_scalar(
                witness, tail_point, projection,
            );
        }
        #[cfg(feature = "parallel")]
        if super::multiple_workers_available() {
            let factor = materialize_tensor_factor(witness, tail_point, projection)?;
            let round = self.compute_product_round_fp64(witness, &factor);
            return Ok((factor, Some(round)));
        }
        if self.fp64_tensor_factor_round == Fp64Kernel::Scalar {
            return materialize_tensor_factor_and_compute_product_round_scalar(
                witness, tail_point, projection,
            );
        }

        let reversed_suffix = tail_point[1..].iter().rev().copied().collect::<Vec<_>>();
        let equality = SplitEqEvals::new(&reversed_suffix)?;
        let minimum_width = match self.fp64_tensor_factor_round {
            Fp64Kernel::Scalar => 1,
            #[cfg(target_arch = "aarch64")]
            Fp64Kernel::Neon => 2,
            #[cfg(target_arch = "x86_64")]
            Fp64Kernel::Avx2 => 4,
            #[cfg(target_arch = "x86_64")]
            Fp64Kernel::Avx512 => 8,
        };
        if equality.in_len() < minimum_width {
            return materialize_tensor_factor_and_compute_product_round_scalar(
                witness, tail_point, projection,
            );
        }

        let equality_inner = EvaluationTable::from_evaluations(&equality.e_in);
        let equality_inner = equality_inner.coefficient_slices::<2>();
        let witness_coefficients = witness.coefficient_slices::<2>();
        let [zero_weights, one_weights] =
            fp64_tensor_factor_branch_weights(projection, tail_point[0]);
        let (storage, round) = match self.fp64_tensor_factor_round {
            Fp64Kernel::Scalar => unreachable!("scalar tensor factor returned above"),
            #[cfg(target_arch = "aarch64")]
            Fp64Kernel::Neon => unsafe {
                akita_field::packed::runtime_neon::materialize_tensor_factor_and_compute_product_round_fp_ext2_fp64_neon(
                    witness_coefficients,
                    equality_inner,
                    &equality.e_out,
                    zero_weights,
                    one_weights,
                )
            },
            #[cfg(target_arch = "x86_64")]
            Fp64Kernel::Avx2 => unsafe {
                akita_field::packed::runtime_x86::materialize_tensor_factor_and_compute_product_round_fp_ext2_fp64_avx2(
                    witness_coefficients,
                    equality_inner,
                    &equality.e_out,
                    zero_weights,
                    one_weights,
                )
            },
            #[cfg(target_arch = "x86_64")]
            Fp64Kernel::Avx512 => unsafe {
                akita_field::packed::runtime_x86::materialize_tensor_factor_and_compute_product_round_fp_ext2_fp64_avx512(
                    witness_coefficients,
                    equality_inner,
                    &equality.e_out,
                    zero_weights,
                    one_weights,
                )
            },
        };
        let factor = EvaluationTable::from_coefficient_storage(storage, witness.len())?;
        Ok((factor, Some(round)))
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
            Fp64Kernel::Scalar => {
                #[cfg(feature = "parallel")]
                if multiple_workers_available() {
                    let context = FpExt2::<Fp64<P>, C>::precompute_fold(challenge);
                    fold_fp64_in_parallel(table, 1, |left, right| {
                        fold_fp64_scalar_range::<P, C>(left, right, &context);
                    });
                    return;
                }
                fold_first_variable_scalar(table, challenge);
            }
            #[cfg(target_arch = "aarch64")]
            Fp64Kernel::Neon => {
                if table.len() / 2 < 2 {
                    fold_first_variable_scalar(table, challenge);
                } else {
                    #[cfg(feature = "parallel")]
                    if multiple_workers_available() {
                        fold_fp64_in_parallel(table, 2, |left, right| unsafe {
                            akita_field::packed::runtime_neon::fold_fp_ext2_fp64_neon(
                                left, right, challenge,
                            )
                        });
                        return;
                    }
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
                    #[cfg(feature = "parallel")]
                    if multiple_workers_available() {
                        fold_fp64_in_parallel(table, 4, |left, right| unsafe {
                            akita_field::packed::runtime_x86::fold_fp_ext2_fp64_avx2(
                                left, right, challenge,
                            )
                        });
                        return;
                    }
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
                    #[cfg(feature = "parallel")]
                    if multiple_workers_available() {
                        fold_fp64_in_parallel(table, 8, |left, right| unsafe {
                            akita_field::packed::runtime_x86::fold_fp_ext2_fp64_avx512(
                                left, right, challenge,
                            )
                        });
                        return;
                    }
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

        #[cfg(feature = "parallel")]
        if multiple_workers_available() {
            self.fold_first_variable_fp64(witness, challenge);
            self.fold_first_variable_fp64(factor, challenge);
            return self.compute_product_round_fp64(witness, factor);
        }

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
    #[cfg(feature = "parallel")]
    if multiple_workers_available() {
        return (0..half)
            .into_par_iter()
            .fold(A::zero, |mut accumulator, row| {
                let witness_0 = FpExt2::new(witness_c0[row], witness_c1[row]);
                let witness_1 = FpExt2::new(witness_c0[row + half], witness_c1[row + half]);
                let factor_0 = FpExt2::new(factor_c0[row], factor_c1[row]);
                let factor_1 = FpExt2::new(factor_c0[row + half], factor_c1[row + half]);
                accumulator.add_constant_product(witness_0, factor_0);
                accumulator.add_quadratic_product(witness_1 - witness_0, factor_1 - factor_0);
                accumulator
            })
            .reduce(A::zero, A::merge)
            .finish();
    }
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

#[cfg(feature = "parallel")]
fn fold_fp64_scalar_range<const P: u64, C>(
    left: [&mut [Fp64<P>]; 2],
    right: [&[Fp64<P>]; 2],
    context: &<FpExt2<Fp64<P>, C> as HasOptimizedFold>::FoldCtx,
) where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    let [left_c0, left_c1] = left;
    let [right_c0, right_c1] = right;
    for row in 0..left_c0.len() {
        let folded = FpExt2::<Fp64<P>, C>::fold_one(
            context,
            FpExt2::new(left_c0[row], left_c1[row]),
            FpExt2::new(right_c0[row], right_c1[row]),
        );
        left_c0[row] = folded.c0();
        left_c1[row] = folded.c1();
    }
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

#[cfg(feature = "parallel")]
fn fold_fp64_in_parallel<const P: u64, C>(
    table: &mut EvaluationTable<Fp64<P>, FpExt2<Fp64<P>, C>>,
    simd_width: usize,
    fold_range: impl Fn([&mut [Fp64<P>]; 2], [&[Fp64<P>]; 2]) + Sync,
) where
    C: FpExt2Config<Fp64<P>>,
{
    let half = table.len() / 2;
    let rows_per_task = parallel_simd_rows(half, simd_width);
    let (left, right) = fp64_coefficient_halves_mut(table);
    let [left_0, left_1] = left;
    let [right_0, right_1] = right;
    left_0
        .par_chunks_mut(rows_per_task)
        .zip(left_1.par_chunks_mut(rows_per_task))
        .zip(
            right_0
                .par_chunks(rows_per_task)
                .zip(right_1.par_chunks(rows_per_task)),
        )
        .for_each(|((left_0, left_1), (right_0, right_1))| {
            fold_range([left_0, left_1], [right_0, right_1]);
        });
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

fn fp64_tensor_factor_branch_weights<const P: u64, C>(
    projection: &TensorFactorProjection<Fp64<P>, FpExt2<Fp64<P>, C>>,
    tail: FpExt2<Fp64<P>, C>,
) -> [[FpExt2<Fp64<P>, C>; 2]; 2]
where
    C: FpExt2Config<Fp64<P>> + 'static,
{
    let zero_branch = FpExt2::one() - tail;
    std::array::from_fn(|branch| {
        let branch = if branch == 0 { zero_branch } else { tail };
        std::array::from_fn(|coordinate| {
            let basis = FpExt2::new(
                if coordinate == 0 {
                    Fp64::one()
                } else {
                    Fp64::zero()
                },
                if coordinate == 1 {
                    Fp64::one()
                } else {
                    Fp64::zero()
                },
            );
            projection.project(basis * branch)
        })
    })
}
