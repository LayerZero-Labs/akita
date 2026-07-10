use super::utils::{
    accumulate_left_round, accumulate_left_round_compact, accumulate_left_round_compact_eq,
    accumulate_left_round_eq, accumulate_right_round, accumulate_right_round_compact,
    accumulate_second_right_round_compact, fold_compact_left_round, fold_compact_left_round_eq,
    fold_compact_right_round, fold_compact_right_two_rounds, fold_dense_left_round,
    fold_factor_in_place, fold_left_round, fold_right_round, product_claim, product_claim_eq,
    product_claim_from_m_compact, product_claim_from_m_compact_eq,
};
use akita_algebra::uni_poly::UniPoly;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use std::sync::Arc;

/// One factored product term `sum_{l,r} table[l,r] * left[l] * right[r]`.
pub(super) struct FactoredProductTerm<E: FieldCore> {
    table: ProductTable<E>,
    left_factor: LeftFactor<E>,
    right_factor: Vec<E>,
    input_claim: E,
    right_rounds: usize,
    total_rounds: usize,
}

enum LeftFactor<E: FieldCore> {
    Dense(Vec<E>),
    Equality { point: Arc<[E]>, scale: E },
}

impl<E: FieldCore> LeftFactor<E> {
    fn len(&self) -> usize {
        match self {
            Self::Dense(factor) => factor.len(),
            Self::Equality { point, .. } => 1usize << point.len(),
        }
    }
}

enum ProductTable<E: FieldCore> {
    Dense(Vec<E>),
    CompactWitness {
        digits: Arc<[i8]>,
        padded_len: usize,
        pending_right_challenge: Option<E>,
    },
}

impl<E: FieldCore + FromPrimitiveInt> FactoredProductTerm<E> {
    /// Construct a dense factored product-sumcheck term.
    ///
    /// Returns an error if factor lengths are not powers of two, are empty, or if
    /// `table.len() != left_factor.len() * right_factor.len()`.
    pub(super) fn new_dense(
        table: Vec<E>,
        left_factor: Vec<E>,
        right_factor: Vec<E>,
    ) -> Result<Self, AkitaError> {
        Self::new(
            ProductTable::Dense(table),
            LeftFactor::Dense(left_factor),
            right_factor,
        )
    }

    /// Construct the witness-carry term from compact digit storage.
    ///
    /// The witness term shares the same factored product identity as the setup term,
    /// but its source table starts as signed gadget digits. Keeping that distinction
    /// outside `AkitaStage3Prover` makes the term state about sumcheck lifecycle,
    /// while this constructor owns representation choice.
    pub(super) fn new_compact(
        digits: Arc<[i8]>,
        padded_len: usize,
        left_factor: Vec<E>,
        right_factor: Vec<E>,
    ) -> Result<Self, AkitaError> {
        Self::new(
            ProductTable::CompactWitness {
                digits,
                padded_len,
                pending_right_challenge: None,
            },
            LeftFactor::Dense(left_factor),
            right_factor,
        )
    }

    /// Construct a compact witness term whose left factor is an equality
    /// polynomial kept in factored form until it is consumed by the rounds.
    pub(super) fn new_compact_equality(
        digits: Arc<[i8]>,
        padded_len: usize,
        point: Arc<[E]>,
        right_factor: Vec<E>,
    ) -> Result<Self, AkitaError> {
        Self::new(
            ProductTable::CompactWitness {
                digits,
                padded_len,
                pending_right_challenge: None,
            },
            LeftFactor::Equality {
                point,
                scale: E::one(),
            },
            right_factor,
        )
    }

    fn new(
        table: ProductTable<E>,
        left_factor: LeftFactor<E>,
        right_factor: Vec<E>,
    ) -> Result<Self, AkitaError> {
        if left_factor.len() == 0
            || right_factor.is_empty()
            || !left_factor.len().is_power_of_two()
            || !right_factor.len().is_power_of_two()
        {
            return Err(AkitaError::InvalidInput(
                "factored product dimensions must be non-empty powers of two".to_string(),
            ));
        }
        let expected_len = left_factor
            .len()
            .checked_mul(right_factor.len())
            .ok_or_else(|| AkitaError::InvalidInput("factored product size overflow".into()))?;
        if table.len() != expected_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_len,
                actual: table.len(),
            });
        }

        let input_claim = table.product_claim(&left_factor, &right_factor);
        let right_rounds = right_factor.len().trailing_zeros() as usize;
        let total_rounds = right_rounds + left_factor.len().trailing_zeros() as usize;
        Ok(Self {
            table,
            left_factor,
            right_factor,
            input_claim,
            right_rounds,
            total_rounds,
        })
    }

    pub(super) fn num_rounds(&self) -> usize {
        self.total_rounds
    }

    pub(super) fn input_claim(&self) -> E {
        self.input_claim
    }

    pub(super) fn compute_round_univariate(
        &mut self,
        round: usize,
        _previous_claim: E,
    ) -> UniPoly<E> {
        let (constant, linear, quadratic) = if round < self.right_rounds {
            self.table
                .accumulate_right_round(&self.left_factor, &self.right_factor)
        } else {
            self.table
                .accumulate_left_round(&self.left_factor, self.right_factor[0])
        };
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    pub(super) fn ingest_challenge(&mut self, round: usize, r_round: E) {
        if round < self.right_rounds {
            self.table.fold_right_round(&mut self.right_factor, r_round);
        } else {
            self.table.fold_left_round(&mut self.left_factor, r_round);
        }
    }

    pub(super) fn folded_table_value(&self) -> Result<E, AkitaError> {
        self.table.folded_value()
    }
}

impl<E: FieldCore + FromPrimitiveInt> ProductTable<E> {
    fn len(&self) -> usize {
        match self {
            Self::Dense(table) => table.len(),
            Self::CompactWitness { padded_len, .. } => *padded_len,
        }
    }

    fn product_claim(&self, left_factor: &LeftFactor<E>, right_factor: &[E]) -> E {
        match left_factor {
            LeftFactor::Dense(left_factor) => match self {
                Self::Dense(table) => product_claim(table, left_factor, right_factor),
                Self::CompactWitness {
                    digits, padded_len, ..
                } => product_claim_from_m_compact(digits, *padded_len, left_factor, right_factor),
            },
            LeftFactor::Equality { point, scale } => match self {
                Self::Dense(table) => product_claim_eq(table, point, *scale, right_factor),
                Self::CompactWitness {
                    digits, padded_len, ..
                } => product_claim_from_m_compact_eq(
                    digits,
                    *padded_len,
                    point,
                    *scale,
                    right_factor,
                ),
            },
        }
    }

    fn accumulate_right_round(&self, left_factor: &LeftFactor<E>, right_factor: &[E]) -> (E, E, E) {
        let LeftFactor::Dense(left_factor) = left_factor else {
            debug_assert!(
                false,
                "equality left factor requires a zero-width right factor"
            );
            return (E::zero(), E::zero(), E::zero());
        };
        match self {
            Self::Dense(table) => accumulate_right_round(table, left_factor, right_factor),
            Self::CompactWitness {
                digits,
                padded_len,
                pending_right_challenge,
            } => match pending_right_challenge {
                Some(first_challenge) => accumulate_second_right_round_compact(
                    digits,
                    *padded_len,
                    left_factor,
                    right_factor,
                    *first_challenge,
                ),
                None => {
                    accumulate_right_round_compact(digits, *padded_len, left_factor, right_factor)
                }
            },
        }
    }

    fn accumulate_left_round(&self, left_factor: &LeftFactor<E>, right_weight: E) -> (E, E, E) {
        match left_factor {
            LeftFactor::Dense(left_factor) => match self {
                Self::Dense(table) => accumulate_left_round(table, left_factor, right_weight),
                Self::CompactWitness {
                    digits,
                    padded_len,
                    pending_right_challenge,
                } => {
                    debug_assert!(pending_right_challenge.is_none());
                    accumulate_left_round_compact(digits, *padded_len, left_factor, right_weight)
                }
            },
            LeftFactor::Equality { point, scale } => match self {
                Self::Dense(table) => accumulate_left_round_eq(table, point, *scale, right_weight),
                Self::CompactWitness {
                    digits,
                    padded_len,
                    pending_right_challenge,
                } => {
                    debug_assert!(pending_right_challenge.is_none());
                    accumulate_left_round_compact_eq(
                        digits,
                        *padded_len,
                        point,
                        *scale,
                        right_weight,
                    )
                }
            },
        }
    }

    fn fold_right_round(&mut self, right_factor: &mut Vec<E>, r: E) {
        match self {
            Self::Dense(table) => fold_right_round(table, right_factor, r),
            Self::CompactWitness {
                digits,
                padded_len,
                pending_right_challenge,
            } => {
                if let Some(first_challenge) = pending_right_challenge.take() {
                    let folded = fold_compact_right_two_rounds(
                        digits,
                        *padded_len,
                        right_factor,
                        first_challenge,
                        r,
                    );
                    *self = Self::Dense(folded);
                } else if right_factor.len() >= 4 {
                    fold_factor_in_place(right_factor, r);
                    *pending_right_challenge = Some(r);
                } else {
                    let folded = fold_compact_right_round(digits, *padded_len, right_factor, r);
                    *self = Self::Dense(folded);
                }
            }
        }
    }

    fn fold_left_round(&mut self, left_factor: &mut LeftFactor<E>, r: E) {
        match left_factor {
            LeftFactor::Dense(left_factor) => match self {
                Self::Dense(table) => fold_left_round(table, left_factor, r),
                Self::CompactWitness {
                    digits,
                    padded_len,
                    pending_right_challenge,
                } => {
                    debug_assert!(pending_right_challenge.is_none());
                    let folded = fold_compact_left_round(digits, *padded_len, left_factor, r);
                    *self = Self::Dense(folded);
                }
            },
            LeftFactor::Equality { point, scale } => {
                match self {
                    Self::Dense(table) => fold_dense_left_round(table, r),
                    Self::CompactWitness {
                        digits,
                        padded_len,
                        pending_right_challenge,
                    } => {
                        debug_assert!(pending_right_challenge.is_none());
                        let folded = fold_compact_left_round_eq(digits, *padded_len, r);
                        *self = Self::Dense(folded);
                    }
                }
                if let Some((&head, tail)) = point.split_first() {
                    let folded_head = (E::one() - r) * (E::one() - head) + r * head;
                    *scale *= folded_head;
                    *point = Arc::from(tail.to_vec());
                }
            }
        }
    }

    fn folded_value(&self) -> Result<E, AkitaError> {
        match self {
            Self::Dense(table) if table.len() == 1 => Ok(table[0]),
            Self::Dense(table) => Err(AkitaError::InvalidSize {
                expected: 1,
                actual: table.len(),
            }),
            Self::CompactWitness { digits, .. } => {
                Ok(super::utils::compact_value_at::<E>(digits, 0))
            }
        }
    }
}
