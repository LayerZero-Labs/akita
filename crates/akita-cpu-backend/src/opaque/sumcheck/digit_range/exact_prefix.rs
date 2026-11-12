use akita_error::AkitaError;
use jolt_field::{solinas::parallel::*, Field};

// Keep small folds local: distributing a few thousand field operations costs
// more than executing them, and makes each round wait for another worker.
const EXACT_PREFIX_SEQUENTIAL_LEN: usize = 1 << 15;
#[cfg(feature = "parallel")]
const EXACT_PREFIX_MIN_TASK_LEN: usize = 1 << 12;

/// Explicit prefix of a power-of-two table followed by one implicit default value.
pub(crate) struct ExactPrefixTable<T: Copy> {
    domain_len: usize,
    explicit: Vec<T>,
    scratch: Vec<T>,
    default: T,
}

/// Checked shape for materializing a prefix without validating round-produced values.
#[derive(Clone, Copy)]
pub(super) struct ExactPrefixLayout {
    domain_len: usize,
    explicit_len: usize,
}

impl ExactPrefixLayout {
    pub(super) fn new(domain_len: usize, explicit_len: usize) -> Result<Self, AkitaError> {
        if domain_len == 0 || !domain_len.is_power_of_two() {
            return Err(AkitaError::Internal(format!(
                "exact-prefix domain length must be a nonzero power of two; got {domain_len}"
            )));
        }
        if explicit_len > domain_len {
            return Err(AkitaError::Internal(format!(
                "exact-prefix explicit table length: expected {domain_len}, actual {explicit_len}"
            )));
        }
        Ok(Self {
            domain_len,
            explicit_len,
        })
    }

    /// The checked range determines the length of the generated table.
    pub(super) fn materialize<T: Copy + Send + Sync>(
        self,
        value_at: impl Fn(usize) -> T + Sync + Send,
        default: T,
    ) -> ExactPrefixTable<T> {
        ExactPrefixTable {
            domain_len: self.domain_len,
            explicit: cfg_into_iter!(0..self.explicit_len).map(value_at).collect(),
            scratch: Vec::new(),
            default,
        }
    }
}

impl<T: Copy> ExactPrefixTable<T> {
    pub(crate) fn new(domain_len: usize, explicit: Vec<T>, default: T) -> Result<Self, AkitaError> {
        let layout = ExactPrefixLayout::new(domain_len, explicit.len())?;
        Ok(Self {
            domain_len: layout.domain_len,
            explicit,
            scratch: Vec::new(),
            default,
        })
    }

    pub(crate) fn domain_len(&self) -> usize {
        self.domain_len
    }

    pub(crate) fn explicit_len(&self) -> usize {
        self.explicit.len()
    }

    pub(crate) fn default_value(&self) -> T {
        self.default
    }

    #[inline(always)]
    pub(crate) fn value_or_default(&self, index: usize) -> T {
        self.explicit.get(index).copied().unwrap_or(self.default)
    }

    pub(crate) fn fold_in_place(
        &mut self,
        fold_pair: impl Fn(T, T) -> T + Sync,
    ) -> Result<(), AkitaError>
    where
        T: Send + Sync,
    {
        if self.domain_len < 2 {
            return Err(AkitaError::Internal(
                "cannot fold a one-element exact-prefix table".to_string(),
            ));
        }
        let next_explicit_len = self.explicit.len().div_ceil(2);
        let default = self.default;
        if !cfg!(feature = "parallel") || next_explicit_len < EXACT_PREFIX_SEQUENTIAL_LEN {
            for pair_index in 0..next_explicit_len {
                let left = self.explicit[2 * pair_index];
                let right = self
                    .explicit
                    .get(2 * pair_index + 1)
                    .copied()
                    .unwrap_or(default);
                self.explicit[pair_index] = fold_pair(left, right);
            }
            self.explicit.truncate(next_explicit_len);
        } else {
            // In-place parallel output overlaps later inputs. Doubling waves
            // avoid the overlap but impose a join at every wave. A reusable
            // destination lets all pairs fold in one independent pass instead.
            #[cfg(feature = "parallel")]
            {
                let source = &self.explicit;
                cfg_into_iter!(0..next_explicit_len)
                    .with_min_len(EXACT_PREFIX_MIN_TASK_LEN)
                    .map(|pair_index| {
                        let left = source[2 * pair_index];
                        let right = source.get(2 * pair_index + 1).copied().unwrap_or(default);
                        fold_pair(left, right)
                    })
                    .collect_into_vec(&mut self.scratch);
            }
            std::mem::swap(&mut self.explicit, &mut self.scratch);
        }

        self.default = fold_pair(self.default, self.default);
        self.domain_len /= 2;
        Ok(())
    }

    pub(crate) fn final_value(&self) -> Option<T> {
        (self.domain_len == 1).then(|| self.value_or_default(0))
    }
}

/// Equality weight of the fully implicit pair suffix in one eq-factored round.
pub(super) struct SplitEqualitySuffixMass<'a, E: Field> {
    first: &'a [E],
    second: &'a [E],
}

impl<'a, E: Field> SplitEqualitySuffixMass<'a, E> {
    pub(super) fn new(first: &'a [E], second: &'a [E]) -> Result<Self, AkitaError> {
        if first.is_empty()
            || second.is_empty()
            || !first.len().is_power_of_two()
            || !second.len().is_power_of_two()
        {
            return Err(AkitaError::Internal(
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
                AkitaError::Internal("split-equality pair count overflow".to_string())
            })?;
        if first_implicit_pair > pair_count {
            return Err(AkitaError::Internal(format!(
                "split-equality implicit pair offset: expected {pair_count}, \
                 actual {first_implicit_pair}"
            )));
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
    use jolt_field::{Prime128Offset275, Ring, Zero};

    type F = Prime128Offset275;

    #[test]
    fn generated_prefix_uses_the_checked_layout() {
        for domain_len in [1, 2, 4, 8, 16] {
            for explicit_len in 0..=domain_len {
                let layout = ExactPrefixLayout::new(domain_len, explicit_len).unwrap();
                let table = layout.materialize(|index| index + 3, 91);
                assert_eq!(table.domain_len(), domain_len);
                assert_eq!(table.explicit, (3..explicit_len + 3).collect::<Vec<_>>());
                assert_eq!(table.default_value(), 91);
            }
        }
    }

    #[test]
    fn generated_prefix_rejects_invalid_shapes_before_materialization() {
        for domain_len in [0, 3] {
            assert!(matches!(ExactPrefixLayout::new(domain_len, 0),
            Err(AkitaError::Internal(message)) if message == format!(
                "exact-prefix domain length must be a nonzero power of two; got {domain_len}"
            )));
        }
        assert!(
            matches!(ExactPrefixLayout::new(4, 5), Err(AkitaError::Internal(message))
            if message == "exact-prefix explicit table length: expected 4, actual 5")
        );
    }

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
    fn exact_prefix_fold_matches_dense_table_across_execution_boundaries() {
        let domain_len = 1 << 18;
        let challenge = F::from_u64(17);
        let default = F::from_u64(91);
        let output_boundaries = [
            EXACT_PREFIX_SEQUENTIAL_LEN / 2,
            EXACT_PREFIX_SEQUENTIAL_LEN,
            2 * EXACT_PREFIX_SEQUENTIAL_LEN,
            4 * EXACT_PREFIX_SEQUENTIAL_LEN,
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
    fn exact_prefix_reuses_output_across_rounds_with_an_odd_prefix() {
        let domain_len = 1 << 18;
        let explicit_len = 4 * EXACT_PREFIX_SEQUENTIAL_LEN + 1;
        let mut expected = (0..explicit_len).map(|i| i as u64 + 3).collect::<Vec<_>>();
        let mut compact = ExactPrefixTable::new(domain_len, expected.clone(), 91).unwrap();
        expected.resize(domain_len, 91);
        let fold = |left: u64, right: u64| left.wrapping_add(right.wrapping_mul(17));
        while expected.len() > 1 {
            compact.fold_in_place(fold).unwrap();
            for i in 0..expected.len() / 2 {
                expected[i] = fold(expected[2 * i], expected[2 * i + 1]);
            }
            expected.truncate(expected.len() / 2);
            for (index, value) in expected.iter().enumerate() {
                assert_eq!(compact.value_or_default(index), *value);
            }
        }
        assert_eq!(compact.final_value(), Some(expected[0]));
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
