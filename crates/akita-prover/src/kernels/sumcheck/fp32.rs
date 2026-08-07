//! Runtime-selected operations for quartic extensions of 32-bit fields.

use super::{
    compute_weighted_affine_polynomial_round_scalar, compute_weighted_affine_product_round_scalar,
    fold_and_compute_stage2_coefficient_round_portable,
    materialize_tensor_factor_and_compute_product_round_scalar, multiple_workers_available,
    Fp32Kernel, SumcheckKernelPlan, SumcheckTableOperations, TensorFactorRoundOutput,
};
use akita_algebra::SplitEqEvals;
use akita_field::{AkitaError, Fp32, FpExt4};
use akita_sumcheck::{
    compute_product_round_scalar, fold_and_compute_product_round_scalar,
    fold_first_variable_scalar, EvaluationTable,
};
use akita_types::TensorFactorProjection;

impl<const P: u32> SumcheckTableOperations<Fp32<P>> for FpExt4<Fp32<P>> {
    fn materialize_tensor_factor_and_compute_product_round(
        plan: SumcheckKernelPlan,
        witness: &EvaluationTable<Fp32<P>, Self>,
        tail_point: &[Self],
        projection: &TensorFactorProjection<Fp32<P>, Self>,
    ) -> Result<TensorFactorRoundOutput<Fp32<P>, Self>, AkitaError> {
        plan.materialize_tensor_factor_and_compute_product_round_fp32(
            witness, tail_point, projection,
        )
    }

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

    fn compute_weighted_affine_product_round<const LANES: usize>(
        plan: SumcheckKernelPlan,
        lanes: &[EvaluationTable<Fp32<P>, Self>; LANES],
        equality: &EvaluationTable<Fp32<P>, Self>,
        arity: usize,
        parent_weights: &[Self],
    ) -> [Self; 5] {
        plan.compute_weighted_affine_product_round_fp32(lanes, equality, arity, parent_weights)
    }

    fn compute_weighted_affine_polynomial_round(
        plan: SumcheckKernelPlan,
        values: &EvaluationTable<Fp32<P>, Self>,
        equality: &EvaluationTable<Fp32<P>, Self>,
        polynomial_coefficients: &[Self],
    ) -> [Self; 5] {
        plan.compute_weighted_affine_polynomial_round_fp32(
            values,
            equality,
            polynomial_coefficients,
        )
    }

    fn try_compute_compact_affine_product_round<const LANES: usize>(
        plan: SumcheckKernelPlan,
        ordered_pair_indices: &[u16],
        folded_pair_rows: &[[Self; LANES]],
        first_equality: &[Self],
        second_equality: &[Self],
        arity: usize,
        parent_weights: &[Self],
    ) -> Option<[Self; 5]> {
        plan.try_compute_compact_affine_product_round_fp32(
            ordered_pair_indices,
            folded_pair_rows,
            first_equality,
            second_equality,
            arity,
            parent_weights,
        )
    }

    fn try_compute_class_coded_affine_polynomial_round(
        plan: SumcheckKernelPlan,
        class_codes: &[u16],
        class_values: &[Self],
        class_taylor_coefficients: &[[Self; 4]],
        first_equality: &[Self],
        second_equality: &[Self],
        degree: usize,
    ) -> Option<[Self; 5]> {
        plan.try_compute_class_coded_affine_polynomial_round_fp32(
            class_codes,
            class_values,
            class_taylor_coefficients,
            first_equality,
            second_equality,
            degree,
        )
    }

    fn try_fold_and_compute_sparse_affine_polynomial_round(
        plan: SumcheckKernelPlan,
        values: &[Self],
        folded_values: &mut [Self],
        first_equality: &[Self],
        second_equality: &[Self],
        challenge: Self,
        degree: usize,
    ) -> Option<[Self; 5]> {
        plan.try_fold_and_compute_sparse_affine_polynomial_round_fp32(
            values,
            folded_values,
            first_equality,
            second_equality,
            challenge,
            degree,
        )
    }

    fn try_fold_class_coded_and_compute_sparse_affine_polynomial_round(
        plan: SumcheckKernelPlan,
        class_codes: &[u16],
        class_values: &[Self],
        folded_values: &mut [Self],
        split_equality: (&[Self], &[Self]),
        challenge: Self,
        degree: usize,
    ) -> Option<[Self; 5]> {
        plan.try_fold_class_coded_and_compute_sparse_affine_polynomial_round_fp32(
            class_codes,
            class_values,
            folded_values,
            split_equality,
            challenge,
            degree,
        )
    }

    fn fold_and_compute_stage2_coefficient_round(
        plan: SumcheckKernelPlan,
        witness: &mut EvaluationTable<Fp32<P>, Self>,
        live_lane_count: usize,
        old_coefficient_count: usize,
        next_alpha_factor: &[Self],
        relation_lane_weights: &[Self],
        first_equality: &[Self],
        second_equality: &[Self],
        challenge: Self,
        include_norm_linear: bool,
    ) -> ([Self; 3], [Self; 3]) {
        plan.fold_and_compute_stage2_coefficient_round_fp32(
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

impl SumcheckKernelPlan {
    /// Materialize an fp32 tensor factor and compute its first product round in
    /// the same traversal.
    pub fn materialize_tensor_factor_and_compute_product_round_fp32<const P: u32>(
        self,
        witness: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
        tail_point: &[FpExt4<Fp32<P>>],
        projection: &TensorFactorProjection<Fp32<P>, FpExt4<Fp32<P>>>,
    ) -> Result<TensorFactorRoundOutput<Fp32<P>, FpExt4<Fp32<P>>>, AkitaError> {
        if tail_point.is_empty() || self.fp32_tensor_factor_round == Fp32Kernel::Scalar {
            return materialize_tensor_factor_and_compute_product_round_scalar(
                witness, tail_point, projection,
            );
        }

        let reversed_suffix = tail_point[1..].iter().rev().copied().collect::<Vec<_>>();
        let equality = SplitEqEvals::new(&reversed_suffix)?;
        let minimum_width = match self.fp32_tensor_factor_round {
            Fp32Kernel::Scalar => 1,
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => 4,
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => 8,
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => 16,
        };
        if equality.in_len() < minimum_width {
            return materialize_tensor_factor_and_compute_product_round_scalar(
                witness, tail_point, projection,
            );
        }

        let equality_inner = EvaluationTable::from_evaluations(&equality.e_in);
        let equality_inner = equality_inner.coefficient_slices::<4>();
        let witness_coefficients = witness.coefficient_slices::<4>();
        let [zero_weights, one_weights] = tensor_factor_branch_weights(projection, tail_point[0]);
        let (storage, round) = match self.fp32_tensor_factor_round {
            Fp32Kernel::Scalar => unreachable!("scalar tensor factor returned above"),
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => unsafe {
                akita_field::packed::runtime_neon::materialize_tensor_factor_and_compute_product_round_fp_ext4_fp32_neon(
                    witness_coefficients,
                    equality_inner,
                    &equality.e_out,
                    zero_weights,
                    one_weights,
                )
            },
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => unsafe {
                akita_field::packed::runtime_x86::materialize_tensor_factor_and_compute_product_round_fp_ext4_fp32_avx2(
                    witness_coefficients,
                    equality_inner,
                    &equality.e_out,
                    zero_weights,
                    one_weights,
                )
            },
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => unsafe {
                akita_field::packed::runtime_x86::materialize_tensor_factor_and_compute_product_round_fp_ext4_fp32_avx512_ifma(
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

    /// Fold a binding-order fp32 Stage 2 coefficient coordinate and compute
    /// its next norm and ordinary-relation round.
    #[allow(clippy::too_many_arguments)]
    pub fn fold_and_compute_stage2_coefficient_round_fp32<const P: u32>(
        self,
        witness: &mut EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
        live_lane_count: usize,
        old_coefficient_count: usize,
        next_alpha_factor: &[FpExt4<Fp32<P>>],
        relation_lane_weights: &[FpExt4<Fp32<P>>],
        first_equality: &[FpExt4<Fp32<P>>],
        second_equality: &[FpExt4<Fp32<P>>],
        challenge: FpExt4<Fp32<P>>,
        include_norm_linear: bool,
    ) -> ([FpExt4<Fp32<P>>; 3], [FpExt4<Fp32<P>>; 3]) {
        let next_len = live_lane_count * (old_coefficient_count / 2);
        #[cfg(feature = "parallel")]
        if multiple_workers_available() {
            return fold_and_compute_stage2_coefficient_round_portable(
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
        let result = match self.fp32_stage2_coefficient_round {
            Fp32Kernel::Scalar => fold_and_compute_stage2_coefficient_round_portable(
                witness,
                live_lane_count,
                old_coefficient_count,
                next_alpha_factor,
                relation_lane_weights,
                first_equality,
                second_equality,
                challenge,
                include_norm_linear,
            ),
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => {
                let coefficients = witness.coefficient_slices_mut::<4>();
                unsafe {
                    akita_field::packed::runtime_neon::fold_and_compute_stage2_coefficient_round_fp_ext4_fp32_neon(
                        coefficients,
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
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => {
                let coefficients = witness.coefficient_slices_mut::<4>();
                unsafe {
                    akita_field::packed::runtime_x86::fold_and_compute_stage2_coefficient_round_fp_ext4_fp32_avx2(
                        coefficients,
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
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => {
                let coefficients = witness.coefficient_slices_mut::<4>();
                unsafe {
                    akita_field::packed::runtime_x86::fold_and_compute_stage2_coefficient_round_fp_ext4_fp32_avx512_ifma(
                        coefficients,
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
        };
        witness.truncate(next_len);
        result
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

    /// Compute a weighted quadratic or quartic product round over fp32 tables.
    pub fn compute_weighted_affine_product_round_fp32<const P: u32, const LANES: usize>(
        self,
        lanes: &[EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>; LANES],
        equality: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
        arity: usize,
        parent_weights: &[FpExt4<Fp32<P>>],
    ) -> [FpExt4<Fp32<P>>; 5] {
        validate_weighted_affine_product_tables(lanes, equality, arity, parent_weights.len());
        if multiple_workers_available() {
            return compute_weighted_affine_product_round_scalar(
                lanes,
                equality,
                arity,
                parent_weights,
            );
        }
        match self.fp32_product_round {
            Fp32Kernel::Scalar => {
                compute_weighted_affine_product_round_scalar(lanes, equality, arity, parent_weights)
            }
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => {
                if equality.len() < 4 {
                    compute_weighted_affine_product_round_scalar(
                        lanes,
                        equality,
                        arity,
                        parent_weights,
                    )
                } else {
                    let (left, right) = affine_product_halves(lanes);
                    let equality = equality.coefficient_slices::<4>();
                    // SAFETY: production plans select NEON only after runtime
                    // detection, and validation establishes complete chunks.
                    unsafe {
                        akita_field::packed::runtime_neon::compute_weighted_affine_product_round_fp_ext4_fp32_neon(
                            left,
                            right,
                            equality,
                            arity,
                            parent_weights,
                        )
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => {
                if equality.len() < 8 {
                    compute_weighted_affine_product_round_scalar(
                        lanes,
                        equality,
                        arity,
                        parent_weights,
                    )
                } else {
                    let (left, right) = affine_product_halves(lanes);
                    let equality = equality.coefficient_slices::<4>();
                    // SAFETY: production plans select AVX2 only after runtime
                    // detection, and validation establishes complete chunks.
                    unsafe {
                        akita_field::packed::runtime_x86::compute_weighted_affine_product_round_fp_ext4_fp32_avx2(
                            left,
                            right,
                            equality,
                            arity,
                            parent_weights,
                        )
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => {
                if equality.len() < 16 {
                    compute_weighted_affine_product_round_scalar(
                        lanes,
                        equality,
                        arity,
                        parent_weights,
                    )
                } else {
                    let (left, right) = affine_product_halves(lanes);
                    let equality = equality.coefficient_slices::<4>();
                    // SAFETY: production plans select AVX-512 only after
                    // runtime detection, and validation establishes chunks.
                    unsafe {
                        akita_field::packed::runtime_x86::compute_weighted_affine_product_round_fp_ext4_fp32_avx512_ifma(
                            left,
                            right,
                            equality,
                            arity,
                            parent_weights,
                        )
                    }
                }
            }
        }
    }

    /// Compute one equality-weighted fp32 polynomial-composition round.
    pub fn compute_weighted_affine_polynomial_round_fp32<const P: u32>(
        self,
        values: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
        equality: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
        polynomial_coefficients: &[FpExt4<Fp32<P>>],
    ) -> [FpExt4<Fp32<P>>; 5] {
        validate_weighted_affine_polynomial_tables(values, equality, polynomial_coefficients.len());
        if multiple_workers_available() {
            return compute_weighted_affine_polynomial_round_scalar(
                values,
                equality,
                polynomial_coefficients,
            );
        }
        match self.fp32_product_round {
            Fp32Kernel::Scalar => compute_weighted_affine_polynomial_round_scalar(
                values,
                equality,
                polynomial_coefficients,
            ),
            #[cfg(target_arch = "aarch64")]
            Fp32Kernel::Neon => {
                if equality.len() < 4 {
                    compute_weighted_affine_polynomial_round_scalar(
                        values,
                        equality,
                        polynomial_coefficients,
                    )
                } else {
                    let (left, right) = coefficient_halves(values);
                    let equality = equality.coefficient_slices::<4>();
                    // SAFETY: production plans select NEON only after runtime
                    // detection, and validation establishes complete chunks.
                    unsafe {
                        akita_field::packed::runtime_neon::compute_weighted_affine_polynomial_round_fp_ext4_fp32_neon(
                            left,
                            right,
                            equality,
                            polynomial_coefficients,
                        )
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx2 => {
                if equality.len() < 8 {
                    compute_weighted_affine_polynomial_round_scalar(
                        values,
                        equality,
                        polynomial_coefficients,
                    )
                } else {
                    let (left, right) = coefficient_halves(values);
                    let equality = equality.coefficient_slices::<4>();
                    // SAFETY: production plans select AVX2 only after runtime
                    // detection, and validation establishes complete chunks.
                    unsafe {
                        akita_field::packed::runtime_x86::compute_weighted_affine_polynomial_round_fp_ext4_fp32_avx2(
                            left,
                            right,
                            equality,
                            polynomial_coefficients,
                        )
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            Fp32Kernel::Avx512Ifma => {
                if equality.len() < 16 {
                    compute_weighted_affine_polynomial_round_scalar(
                        values,
                        equality,
                        polynomial_coefficients,
                    )
                } else {
                    let (left, right) = coefficient_halves(values);
                    let equality = equality.coefficient_slices::<4>();
                    // SAFETY: production plans select AVX-512 only after
                    // runtime detection, and validation establishes chunks.
                    unsafe {
                        akita_field::packed::runtime_x86::compute_weighted_affine_polynomial_round_fp_ext4_fp32_avx512_ifma(
                            left,
                            right,
                            equality,
                            polynomial_coefficients,
                        )
                    }
                }
            }
        }
    }
}

fn validate_weighted_affine_polynomial_tables<const P: u32>(
    values: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    equality: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    coefficient_count: usize,
) {
    assert!(
        values.len().is_power_of_two() && values.len() >= 2,
        "polynomial values must have a nontrivial power-of-two length"
    );
    assert_eq!(equality.len(), values.len() / 2);
    assert!(coefficient_count <= 5);
}

fn tensor_factor_branch_weights<const P: u32>(
    projection: &TensorFactorProjection<Fp32<P>, FpExt4<Fp32<P>>>,
    tail: FpExt4<Fp32<P>>,
) -> [[FpExt4<Fp32<P>>; 4]; 2] {
    let zero_branch = FpExt4::one() - tail;
    std::array::from_fn(|branch| {
        let branch = if branch == 0 { zero_branch } else { tail };
        std::array::from_fn(|coordinate| {
            let basis = FpExt4::new(std::array::from_fn(|index| {
                if index == coordinate {
                    Fp32::one()
                } else {
                    Fp32::zero()
                }
            }));
            projection.project(basis * branch)
        })
    })
}

fn validate_weighted_affine_product_tables<const P: u32, const LANES: usize>(
    lanes: &[EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>; LANES],
    equality: &EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>,
    arity: usize,
    parent_count: usize,
) {
    assert!(matches!(arity, 2 | 4), "product arity must be two or four");
    assert_eq!(LANES, arity * parent_count);
    let table_len = lanes[0].len();
    assert!(
        table_len.is_power_of_two()
            && table_len >= 2
            && lanes.iter().all(|lane| lane.len() == table_len),
        "product lanes must have one nontrivial power-of-two length"
    );
    assert_eq!(equality.len(), table_len / 2);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
type Fp32CoefficientSlices<'a, const P: u32> = [&'a [Fp32<P>]; 4];

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
type AffineProductHalves<'a, const P: u32, const LANES: usize> = (
    [Fp32CoefficientSlices<'a, P>; LANES],
    [Fp32CoefficientSlices<'a, P>; LANES],
);

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn affine_product_halves<const P: u32, const LANES: usize>(
    lanes: &[EvaluationTable<Fp32<P>, FpExt4<Fp32<P>>>; LANES],
) -> AffineProductHalves<'_, P, LANES> {
    let half = lanes[0].len() / 2;
    (
        std::array::from_fn(|lane| {
            std::array::from_fn(|coefficient| &lanes[lane].coefficient_slice(coefficient)[..half])
        }),
        std::array::from_fn(|lane| {
            std::array::from_fn(|coefficient| &lanes[lane].coefficient_slice(coefficient)[half..])
        }),
    )
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
