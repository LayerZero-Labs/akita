use super::*;

/// Dense factor table of a [`ExtensionOpeningTables::Dense`] term.
///
/// Every dense term of one reduction batch starts from the same transparent
/// tail-point equality table, so the full-size table is built once and
/// shared; the first fold writes a fresh half-size owned table (identical
/// values to an in-place fold) and later rounds fold in place.
#[derive(Debug, Clone)]
pub(in crate::protocol::extension_opening_reduction) enum DenseEorFactor<E> {
    Shared(std::sync::Arc<Vec<E>>),
    Owned(Vec<E>),
}

impl<E: FieldCore> DenseEorFactor<E> {
    pub(in crate::protocol::extension_opening_reduction) fn as_slice(&self) -> &[E] {
        match self {
            Self::Shared(factor) => factor,
            Self::Owned(factor) => factor,
        }
    }
}

impl<E: FieldCore + HasOptimizedFold> DenseEorFactor<E> {
    pub(in crate::protocol::extension_opening_reduction) fn fold_in_place(&mut self, r_round: E) {
        match self {
            Self::Owned(factor) => fold_evals_in_place(factor, r_round),
            Self::Shared(factor) => {
                *self = Self::Owned(fold_evals_shared(factor, r_round));
            }
        }
    }
}

/// Fold a shared evaluation table into a fresh half-size owned table.
///
/// Same per-pair `fold_one` arithmetic as
/// [`fold_evals_in_place`](akita_algebra::fold_evals_in_place), so the
/// folded values are byte-identical; only the destination differs.
pub(in crate::protocol::extension_opening_reduction) fn fold_evals_shared<
    E: FieldCore + HasOptimizedFold,
>(
    src: &[E],
    r: E,
) -> Vec<E> {
    assert!(
        src.len().is_power_of_two(),
        "evals length must be a power of two"
    );
    assert!(src.len() >= 2, "evals must have at least 2 elements");
    let half = src.len() / 2;
    let ctx = E::precompute_fold(r);
    #[cfg(feature = "parallel")]
    {
        const PAR_FOLD_THRESHOLD: usize = 1 << 12;
        if half >= PAR_FOLD_THRESHOLD {
            return (0..half)
                .into_par_iter()
                .map(|i| E::fold_one(&ctx, src[2 * i], src[2 * i + 1]))
                .collect();
        }
    }
    (0..half)
        .map(|i| E::fold_one(&ctx, src[2 * i], src[2 * i + 1]))
        .collect()
}

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
pub(in crate::protocol::extension_opening_reduction) enum ExtensionOpeningTables<E: FieldCore> {
    Dense {
        witness: Vec<E>,
        factor: DenseEorFactor<E>,
    },
    Sparse {
        witness: SparseExtensionOpeningWitness<E>,
        factor: SparseFactor<E>,
    },
    Cylindrical {
        inner: Box<ExtensionOpeningTables<E>>,
        extra_point: Vec<E>,
        extra_round: usize,
        extra_factor_eval: E,
    },
}

impl<E: FieldCore> ExtensionOpeningTables<E> {
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

    pub(in crate::protocol::extension_opening_reduction) fn claim(&self) -> Result<E, AkitaError> {
        match self {
            Self::Dense { witness, factor } => {
                extension_opening_reduction_claim(witness, factor.as_slice())
            }
            Self::Sparse { witness, factor } => match factor {
                SparseFactor::Dense(factor_evals) => witness.claim_with_factor(factor_evals),
                SparseFactor::Tensor(factor) => {
                    if witness.table_len() != factor.len() {
                        return Err(AkitaError::InvalidSize {
                            expected: witness.table_len(),
                            actual: factor.len(),
                        });
                    }
                    Ok(witness.claim_with_factor_fn(|idx| factor.factor_at_index(idx)))
                }
            },
            Self::Cylindrical { inner, .. } => inner.claim(),
        }
    }

    pub(in crate::protocol::extension_opening_reduction) fn final_witness_and_factor_evals(
        &self,
    ) -> Option<(E, E)> {
        match self {
            Self::Dense { witness, factor } => {
                let factor = factor.as_slice();
                (factor.len() == 1 && witness.len() == 1).then(|| (witness[0], factor[0]))
            }
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

impl<E: FieldCore + HasUnreducedOps> ExtensionOpeningTables<E> {
    pub(in crate::protocol::extension_opening_reduction) fn accumulate_round(
        &self,
        coeff: E,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        match self {
            Self::Dense { witness, factor } => {
                let (round_constant, round_quadratic) =
                    accumulate_dense_round(witness, factor.as_slice(), coeff);
                *constant += round_constant;
                *quadratic += round_quadratic;
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
                    inner.accumulate_round(coeff, constant, quadratic);
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

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> ExtensionOpeningTables<E> {
    pub(in crate::protocol::extension_opening_reduction) fn fold_in_place(&mut self, r_round: E) {
        match self {
            Self::Dense { witness, factor } => {
                fold_evals_in_place(witness, r_round);
                factor.fold_in_place(r_round);
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
                    inner.fold_in_place(r_round);
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
/// Sparse counterpart of [`fused_fold_and_accumulate`], valid only inside the
/// merge-free plateau. The factor is folded first so the witness sweep reads the
/// next round's factor children while folding each entry. Returns the *unscaled*
/// next-round coefficients; the caller applies the term coefficient.
pub(in crate::protocol::extension_opening_reduction) fn fused_fold_and_accumulate_sparse<E>(
    witness: &mut SparseExtensionOpeningWitness<E>,
    factor: &mut SparseFactor<E>,
    r_round: E,
) -> (E, E)
where
    E: FieldCore + HasUnreducedOps + HasOptimizedFold,
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
