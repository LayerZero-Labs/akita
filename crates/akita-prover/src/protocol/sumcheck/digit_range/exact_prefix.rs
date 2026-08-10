use akita_field::{parallel::*, AkitaError, FieldCore};

const EXACT_PREFIX_SEQUENTIAL_LEN: usize = 1 << 12;

/// Explicit prefix of a power-of-two table followed by one implicit default value.
pub(in crate::protocol::sumcheck) struct ExactPrefixTable<T: Copy> {
    domain_len: usize,
    explicit: Vec<T>,
    default: T,
}

impl<T: Copy> ExactPrefixTable<T> {
    pub(in crate::protocol::sumcheck) fn new(
        domain_len: usize,
        explicit: Vec<T>,
        default: T,
    ) -> Result<Self, AkitaError> {
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

    pub(in crate::protocol::sumcheck) fn domain_len(&self) -> usize {
        self.domain_len
    }

    pub(in crate::protocol::sumcheck) fn explicit_len(&self) -> usize {
        self.explicit.len()
    }

    pub(in crate::protocol::sumcheck) fn default_value(&self) -> T {
        self.default
    }

    #[inline(always)]
    pub(in crate::protocol::sumcheck) fn value_or_default(&self, index: usize) -> T {
        self.explicit.get(index).copied().unwrap_or(self.default)
    }

    pub(in crate::protocol::sumcheck) fn fold_in_place(
        &mut self,
        fold_pair: impl Fn(T, T) -> T + Sync,
    ) -> Result<(), AkitaError>
    where
        T: Send + Sync,
    {
        if self.domain_len < 2 {
            return Err(AkitaError::InvalidInput(
                "cannot fold a one-element exact-prefix table".to_string(),
            ));
        }
        let next_explicit_len = self.explicit.len().div_ceil(2);
        let explicit_len = self.explicit.len();
        let default = self.default;
        let sequential_len = next_explicit_len.min(EXACT_PREFIX_SEQUENTIAL_LEN);
        for pair_index in 0..sequential_len {
            let left = self.explicit[2 * pair_index];
            let right = self
                .explicit
                .get(2 * pair_index + 1)
                .copied()
                .unwrap_or(default);
            self.explicit[pair_index] = fold_pair(left, right);
        }

        let mut wave_start = sequential_len;
        while wave_start < next_explicit_len {
            // Pair i writes i and reads 2i..=2i+1. Once [0, a) is folded,
            // [a, 2a) has disjoint inputs and may run in parallel.
            let wave_end = (2 * wave_start).min(next_explicit_len);
            let source_start = 2 * wave_start;
            let (output_prefix, source_suffix) = self.explicit.split_at_mut(source_start);
            let output = &mut output_prefix[wave_start..wave_end];
            let source_len = (2 * output.len()).min(explicit_len - source_start);
            let source = &source_suffix[..source_len];
            cfg_iter_mut!(output)
                .enumerate()
                .for_each(|(pair_index, folded)| {
                    let left = source[2 * pair_index];
                    let right = source.get(2 * pair_index + 1).copied().unwrap_or(default);
                    *folded = fold_pair(left, right);
                });
            wave_start = wave_end;
        }

        self.default = fold_pair(self.default, self.default);
        self.explicit.truncate(next_explicit_len);
        self.domain_len /= 2;
        Ok(())
    }

    pub(in crate::protocol::sumcheck) fn final_value(&self) -> Option<T> {
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
    fn exact_prefix_fold_matches_dense_table_across_parallel_waves() {
        let domain_len = 1 << 16;
        let challenge = F::from_u64(17);
        let default = F::from_u64(91);
        let output_boundaries = [
            EXACT_PREFIX_SEQUENTIAL_LEN,
            2 * EXACT_PREFIX_SEQUENTIAL_LEN,
            4 * EXACT_PREFIX_SEQUENTIAL_LEN,
            8 * EXACT_PREFIX_SEQUENTIAL_LEN,
        ];
        let explicit_lengths = output_boundaries.into_iter().flat_map(|boundary| {
            [
                2 * boundary - 1,
                2 * boundary,
                (2 * boundary + 1).min(domain_len),
            ]
        });

        for explicit_len in explicit_lengths {
            let explicit = (0..explicit_len)
                .map(|index| F::from_u64(index as u64 + 3))
                .collect::<Vec<_>>();
            let mut compact = ExactPrefixTable::new(domain_len, explicit.clone(), default).unwrap();
            let mut expected = explicit;
            expected.resize(domain_len, default);

            compact
                .fold_in_place(|left, right| left + challenge * (right - left))
                .unwrap();
            for pair_index in 0..domain_len / 2 {
                let left = expected[2 * pair_index];
                let right = expected[2 * pair_index + 1];
                expected[pair_index] = left + challenge * (right - left);
            }
            expected.truncate(domain_len / 2);

            assert_eq!(
                compact.explicit,
                expected[..explicit_len.div_ceil(2)],
                "explicit length {explicit_len}"
            );
            assert_eq!(compact.default, default);
            assert_eq!(compact.domain_len, domain_len / 2);
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
