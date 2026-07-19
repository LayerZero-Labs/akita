use akita_field::{AkitaError, FieldCore};

/// Explicit prefix of a power-of-two table followed by one implicit default value.
pub(super) struct ExactPrefixTable<T: Copy> {
    domain_len: usize,
    explicit: Vec<T>,
    default: T,
}

impl<T: Copy> ExactPrefixTable<T> {
    pub(super) fn new(domain_len: usize, explicit: Vec<T>, default: T) -> Result<Self, AkitaError> {
        if domain_len == 0 || !domain_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(format!(
                "exact-prefix domain length must be a nonzero power of two; got {domain_len}"
            )));
        }
        if explicit.len() > domain_len {
            return Err(AkitaError::InvalidSize {
                expected: domain_len,
                actual: explicit.len(),
            });
        }
        Ok(Self {
            domain_len,
            explicit,
            default,
        })
    }

    pub(super) fn domain_len(&self) -> usize {
        self.domain_len
    }

    pub(super) fn explicit_len(&self) -> usize {
        self.explicit.len()
    }

    pub(super) fn default_value(&self) -> T {
        self.default
    }

    #[inline(always)]
    pub(super) fn value_or_default(&self, index: usize) -> T {
        self.explicit.get(index).copied().unwrap_or(self.default)
    }

    pub(super) fn fold_in_place(
        &mut self,
        mut fold_pair: impl FnMut(T, T) -> T,
    ) -> Result<(), AkitaError> {
        if self.domain_len < 2 {
            return Err(AkitaError::InvalidInput(
                "cannot fold a one-element exact-prefix table".to_string(),
            ));
        }
        let next_explicit_len = self.explicit.len().div_ceil(2);
        for pair_index in 0..next_explicit_len {
            let left = self.value_or_default(2 * pair_index);
            let right = self.value_or_default(2 * pair_index + 1);
            self.explicit[pair_index] = fold_pair(left, right);
        }
        self.explicit.truncate(next_explicit_len);
        self.domain_len /= 2;
        Ok(())
    }

    pub(super) fn final_value(&self) -> Option<T> {
        (self.domain_len == 1).then(|| self.value_or_default(0))
    }
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
    fn exact_prefix_fold_matches_padded_table_for_every_short_prefix() {
        let challenge = F::from_u64(17);
        for domain_len in [2, 4, 8, 16] {
            for explicit_len in 0..=domain_len {
                let default = F::from_u64(91);
                let explicit = (0..explicit_len)
                    .map(|index| F::from_u64(index as u64 + 3))
                    .collect::<Vec<_>>();
                let mut compact =
                    ExactPrefixTable::new(domain_len, explicit.clone(), default).unwrap();
                let mut padded = explicit;
                padded.resize(domain_len, default);
                while padded.len() > 1 {
                    compact
                        .fold_in_place(|left, right| left + challenge * (right - left))
                        .unwrap();
                    for pair_index in 0..padded.len() / 2 {
                        let left = padded[2 * pair_index];
                        let right = padded[2 * pair_index + 1];
                        padded[pair_index] = left + challenge * (right - left);
                    }
                    padded.truncate(padded.len() / 2);
                }
                assert_eq!(compact.final_value(), Some(padded[0]));
            }
        }
    }

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
