use super::*;

/// Transparent dense factor table shared by terms at the start of one group.
#[derive(Debug, Clone)]
pub(in crate::protocol::extension_opening_reduction) enum DenseEorFactor<E: FieldCore> {
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

fn fold_evals_shared<E: FieldCore + HasOptimizedFold>(src: &[E], r: E) -> Vec<E> {
    debug_assert!(src.len().is_power_of_two());
    debug_assert!(src.len() >= 2);
    let half = src.len() / 2;
    let ctx = E::precompute_fold(r);
    #[cfg(feature = "parallel")]
    {
        const PARALLEL_FOLD_THRESHOLD: usize = 1 << 12;
        if half >= PARALLEL_FOLD_THRESHOLD {
            return (0..half)
                .into_par_iter()
                .map(|index| E::fold_one(&ctx, src[2 * index], src[2 * index + 1]))
                .collect();
        }
    }
    (0..half)
        .map(|index| E::fold_one(&ctx, src[2 * index], src[2 * index + 1]))
        .collect()
}

/// Dense EOR tables, optionally extended over zero-fixed high variables.
#[derive(Debug, Clone)]
pub(in crate::protocol::extension_opening_reduction) enum ExtensionOpeningTables<E: FieldCore> {
    Dense {
        witness: Vec<E>,
        factor: DenseEorFactor<E>,
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

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> ExtensionOpeningTables<E> {
    pub(in crate::protocol::extension_opening_reduction) fn fold_in_place(&mut self, r_round: E) {
        match self {
            Self::Dense { witness, factor } => {
                fold_evals_in_place(witness, r_round);
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
