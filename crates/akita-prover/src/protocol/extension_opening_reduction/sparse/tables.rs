use super::*;

///
/// - [`Dense`](Self::Dense): dense witness paired with a dense factor. The
///   initial shape for non-onehot terms and the steady state of the recursive
///   EOR path.
/// - [`Sparse`](Self::Sparse): sparse witness paired with a (dense or lazy
///   tensor) [`SparseFactor`]. The initial shape for onehot terms.
/// - [`Cylindrical`](Self::Cylindrical): a native-domain term extended over
///   additional high variables without repeating its witness table. The
///   transparent padding factor has Boolean sum one, so native rounds are
///   unchanged and the remaining rounds fold only that factor.
#[derive(Debug, Clone)]
pub(in crate::protocol::extension_opening_reduction) enum ExtensionOpeningTables<
    F: FieldCore,
    E: ExtField<F>,
> {
    Dense {
        witness: EvaluationTable<F, E>,
        factor: EvaluationTable<F, E>,
    },
    Sparse {
        witness: SparseExtensionOpeningWitness<F, E>,
        factor: SparseFactor<E>,
    },
    Cylindrical {
        inner: Box<ExtensionOpeningTables<F, E>>,
        extra_point: Vec<E>,
        extra_round: usize,
        extra_factor_eval: E,
    },
}

impl<F: FieldCore, E: ExtField<F>> ExtensionOpeningTables<F, E> {
    pub(in crate::protocol::extension_opening_reduction) fn len(&self) -> usize {
        match self {
            Self::Dense { witness, .. } => witness.len(),
            Self::Sparse { witness, .. } => witness.table_len(),
            Self::Cylindrical {
                inner,
                extra_point,
                extra_round,
                ..
            } => inner
                .len()
                .checked_shl(
                    u32::try_from(extra_point.len().saturating_sub(*extra_round))
                        .unwrap_or(u32::MAX),
                )
                .unwrap_or(0),
        }
    }

    pub(in crate::protocol::extension_opening_reduction) fn final_witness_and_factor_evals(
        &self,
    ) -> Option<(E, E)> {
        match self {
            Self::Dense { witness, factor } => (factor.len() == 1 && witness.len() == 1)
                .then(|| (witness.evaluation(0), factor.evaluation(0))),
            Self::Sparse { witness, factor } => match factor {
                SparseFactor::Dense(factor_evals) => (factor_evals.len() == 1)
                    .then(|| witness.final_eval())
                    .flatten()
                    .map(|witness| (witness, factor_evals[0])),
                SparseFactor::Tensor(_) => None,
            },
            Self::Cylindrical {
                inner,
                extra_point,
                extra_round,
                extra_factor_eval,
            } => (*extra_round == extra_point.len())
                .then(|| inner.final_witness_and_factor_evals())
                .flatten()
                .map(|(witness, factor)| (witness, factor * *extra_factor_eval)),
        }
    }
}

impl<F, E> ExtensionOpeningTables<F, E>
where
    F: FieldCore,
    E: SumcheckTableOperations<F>,
{
    pub(in crate::protocol::extension_opening_reduction) fn accumulate_round(
        &self,
        plan: SumcheckKernelPlan,
        coeff: E,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        match self {
            Self::Dense { witness, factor } => {
                let (round_constant, round_quadratic) =
                    E::compute_product_round(plan, witness, factor);
                *constant += coeff * round_constant;
                *quadratic += coeff * round_quadratic;
            }
            Self::Sparse { witness, factor } => match factor {
                SparseFactor::Dense(factor_evals) => {
                    witness.accumulate_round(factor_evals, coeff, constant, quadratic);
                }
                SparseFactor::Tensor(factor) => {
                    witness.accumulate_round_with_factor(coeff, constant, quadratic, |pair| {
                        factor.factor_pair(pair)
                    });
                }
            },
            Self::Cylindrical {
                inner,
                extra_point,
                extra_round,
                extra_factor_eval,
            } => {
                if inner.len() > 1 {
                    inner.accumulate_round(plan, coeff, constant, quadratic);
                } else if let (Some((witness, factor)), Some(&point)) = (
                    inner.final_witness_and_factor_evals(),
                    extra_point.get(*extra_round),
                ) {
                    *constant += coeff * witness * factor * *extra_factor_eval * (E::one() - point);
                }
            }
        }
    }
}

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> SparseFactor<E> {
    /// Fold the transparent factor by one sumcheck challenge, materializing the
    /// lazy tensor factor into a dense table once it reaches its split depth.
    pub(in crate::protocol::extension_opening_reduction) fn fold_in_place(&mut self, r_round: E) {
        match self {
            SparseFactor::Dense(factor_evals) => {
                fold_evals_in_place(factor_evals, r_round);
            }
            SparseFactor::Tensor(tensor_factor) => {
                tensor_factor.fold_in_place(r_round);
                if tensor_factor.is_ready_to_materialize() {
                    let dense = tensor_factor.materialize_dense();
                    *self = SparseFactor::Dense(dense);
                }
            }
        }
    }
}

impl<F, E> ExtensionOpeningTables<F, E>
where
    F: FieldCore,
    E: SumcheckTableOperations<F>,
{
    pub(in crate::protocol::extension_opening_reduction) fn fold_in_place(
        &mut self,
        plan: SumcheckKernelPlan,
        r_round: E,
    ) {
        match self {
            Self::Dense { witness, factor } => {
                E::fold_first_variable(plan, witness, r_round);
                E::fold_first_variable(plan, factor, r_round);
            }
            Self::Sparse { witness, factor } => {
                witness.fold_in_place(r_round);
                factor.fold_in_place(r_round);
            }
            Self::Cylindrical {
                inner,
                extra_point,
                extra_round,
                extra_factor_eval,
            } => {
                if inner.len() > 1 {
                    inner.fold_in_place(plan, r_round);
                } else if let Some(&point) = extra_point.get(*extra_round) {
                    *extra_factor_eval *=
                        (E::one() - point) * (E::one() - r_round) + point * r_round;
                    *extra_round += 1;
                }
            }
        }
    }
}

/// Fold a sparse term's factor and witness by one challenge AND compute the next
/// round's `(constant, quadratic)` in a single witness sweep.
///
/// Sparse counterpart of [`fold_and_compute_product_round_scalar`], valid only inside the
/// merge-free plateau. The factor is folded first so the witness sweep reads the
/// next round's factor children while folding each entry. Returns the *unscaled*
/// next-round coefficients; the caller applies the term coefficient.
pub(in crate::protocol::extension_opening_reduction) fn fused_fold_and_accumulate_sparse<F, E>(
    witness: &mut SparseExtensionOpeningWitness<F, E>,
    factor: &mut SparseFactor<E>,
    r_round: E,
) -> (E, E)
where
    F: FieldCore,
    E: ExtField<F> + HasUnreducedOps + HasOptimizedFold,
{
    factor.fold_in_place(r_round);
    match factor {
        SparseFactor::Dense(factor_evals) => witness
            .fused_fold_accumulate_merge_free(r_round, &|pair| {
                (factor_evals[2 * pair], factor_evals[2 * pair + 1])
            }),
        SparseFactor::Tensor(tensor_factor) => witness
            .fused_fold_accumulate_merge_free(r_round, &|pair| tensor_factor.factor_pair(pair)),
    }
}
