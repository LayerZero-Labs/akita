use akita_field::{AkitaError, ExtField, FieldCore};
use akita_sumcheck::EvaluationTable;

pub(super) fn materialize_remaining_equality<F, E>(
    first: &[E],
    second: &[E],
) -> Result<EvaluationTable<F, E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let len = first.len().checked_mul(second.len()).ok_or_else(|| {
        AkitaError::InvalidInput("remaining equality table length overflow".to_string())
    })?;
    EvaluationTable::from_multilinear_evaluation_fn(len, |logical_row| {
        first[logical_row % first.len()] * second[logical_row / first.len()]
    })
}

/// Equality weight of the fully implicit pair suffix in one eq-factored round.
pub(super) struct SplitEqualitySuffixMass<'a, E: FieldCore> {
    first: &'a [E],
    second: &'a [E],
}

impl<'a, E: FieldCore> SplitEqualitySuffixMass<'a, E> {
    pub(super) fn new(first: &'a [E], second: &'a [E]) -> Result<Self, AkitaError> {
        if first.is_empty()
            || second.is_empty()
            || !first.len().is_power_of_two()
            || !second.len().is_power_of_two()
        {
            return Err(AkitaError::InvalidInput(
                "split-equality tables must have nonzero power-of-two lengths".to_string(),
            ));
        }
        Ok(Self { first, second })
    }

    pub(super) fn weight_from(&self, first_implicit_pair: usize) -> Result<E, AkitaError> {
        let pair_count = self
            .first
            .len()
            .checked_mul(self.second.len())
            .ok_or_else(|| {
                AkitaError::InvalidInput("split-equality pair count overflow".to_string())
            })?;
        if first_implicit_pair > pair_count {
            return Err(AkitaError::InvalidSize {
                expected: pair_count,
                actual: first_implicit_pair,
            });
        }
        if first_implicit_pair == pair_count {
            return Ok(E::zero());
        }
        let first_index = first_implicit_pair % self.first.len();
        let second_index = first_implicit_pair / self.first.len();
        let first_tail = self.first[first_index..]
            .iter()
            .copied()
            .fold(E::zero(), |sum, value| sum + value);
        let first_total = self
            .first
            .iter()
            .copied()
            .fold(E::zero(), |sum, value| sum + value);
        let second_tail = self.second[second_index + 1..]
            .iter()
            .copied()
            .fold(E::zero(), |sum, value| sum + value);
        Ok(self.second[second_index] * first_tail + second_tail * first_total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::eq_poly::EqPolynomial;
    use akita_algebra::split_eq::GruenSplitEq;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    #[test]
    fn split_equality_suffix_matches_dense_sum_at_every_boundary() {
        let point = [
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
        ];
        let first = EqPolynomial::evals(&point[..2]).unwrap();
        let second = EqPolynomial::evals(&point[2..]).unwrap();
        let suffix = SplitEqualitySuffixMass::new(&first, &second).unwrap();
        let dense = (0..first.len() * second.len())
            .map(|pair_index| first[pair_index % first.len()] * second[pair_index / first.len()])
            .collect::<Vec<_>>();
        for first_implicit_pair in 0..=dense.len() {
            let expected = dense[first_implicit_pair..]
                .iter()
                .copied()
                .fold(F::zero(), |sum, value| sum + value);
            assert_eq!(suffix.weight_from(first_implicit_pair).unwrap(), expected);
        }
    }

    #[test]
    fn split_equality_suffix_matches_dense_sum_after_every_bind() {
        let point = [
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ];
        let mut split_eq = GruenSplitEq::new(&point).unwrap();
        for round in 0..point.len() {
            let (first, second) = split_eq.remaining_eq_tables();
            let suffix = SplitEqualitySuffixMass::new(first, second).unwrap();
            let dense = (0..first.len() * second.len())
                .map(|pair_index| {
                    first[pair_index % first.len()] * second[pair_index / first.len()]
                })
                .collect::<Vec<_>>();
            for first_implicit_pair in 0..=dense.len() {
                let expected = dense[first_implicit_pair..]
                    .iter()
                    .copied()
                    .fold(F::zero(), |sum, value| sum + value);
                assert_eq!(suffix.weight_from(first_implicit_pair).unwrap(), expected);
            }
            split_eq.bind(F::from_u64(round as u64 + 13));
        }
    }
}
