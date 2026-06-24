use super::utils::{
    accumulate_left_round, accumulate_left_round_compact, accumulate_right_round,
    accumulate_right_round_compact, accumulate_second_right_round_compact, compact_value_at,
    fold_compact_left_round, fold_compact_right_round, fold_compact_right_two_rounds,
    fold_left_round, fold_right_round, product_claim, product_claim_compact,
};
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};

/// Optional Stage-2 next-witness handoff folded into Stage 3.
pub struct Stage2Handoff<'a, E: FieldCore> {
    /// Next-witness digits laid out over the same `(lambda, y)` domain as setup.
    pub witness_digits: &'a [i8],
    /// Stage-2 point whose witness opening is reduced by Stage 3.
    pub stage2_point: &'a [E],
    /// Claimed `W(stage2_point)` value already used by Stage 2. This is batched
    /// into the Stage-3 sumcheck input while `SetupSumcheckProof::setup_claim`
    /// continues to carry the raw setup contribution for Stage 2.
    pub stage2_claim: E,
}

pub(super) struct Stage2HandoffState<E: FieldCore> {
    table: Stage2HandoffTable<E>,
    left_factor: Vec<E>,
    right_factor: Vec<E>,
    expected_claim: E,
}

enum Stage2HandoffTable<E: FieldCore> {
    Compact {
        digits: Vec<i8>,
        padded_len: usize,
        pending_right_challenge: Option<E>,
    },
    Full(Vec<E>),
}

pub(super) fn prepare_stage2_handoff<E, const D: usize>(
    handoff: Stage2Handoff<'_, E>,
    lambda_len: usize,
    ring_bits: usize,
) -> Result<Stage2HandoffState<E>, AkitaError>
where
    E: FieldCore + FromPrimitiveInt,
{
    let expected_rounds = ring_bits
        .checked_add(lambda_len.trailing_zeros() as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("stage3 witness round count overflow".into()))?;
    if handoff.stage2_point.len() != expected_rounds {
        return Err(AkitaError::InvalidSize {
            expected: expected_rounds,
            actual: handoff.stage2_point.len(),
        });
    }
    let (rho_y, rho_lambda) = handoff.stage2_point.split_at(ring_bits);
    let right_factor = EqPolynomial::evals(rho_y)?;
    let left_factor = EqPolynomial::evals(rho_lambda)?;
    if right_factor.len() != D || left_factor.len() != lambda_len {
        return Err(AkitaError::InvalidProof);
    }
    let table_len = lambda_len
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("stage3 witness table length overflow".into()))?;
    if handoff.witness_digits.len() > table_len {
        return Err(AkitaError::InvalidSize {
            expected: table_len,
            actual: handoff.witness_digits.len(),
        });
    }
    Ok(Stage2HandoffState {
        table: Stage2HandoffTable::Compact {
            digits: handoff.witness_digits.to_vec(),
            padded_len: table_len,
            pending_right_challenge: None,
        },
        left_factor,
        right_factor,
        expected_claim: handoff.stage2_claim,
    })
}

impl<E: FieldCore> Stage2HandoffTable<E> {
    fn len(&self) -> usize {
        match self {
            Self::Compact { padded_len, .. } => *padded_len,
            Self::Full(table) => table.len(),
        }
    }
}

impl<E: FieldCore + FromPrimitiveInt> Stage2HandoffState<E> {
    pub(super) fn matches_shape(
        &self,
        left_len: usize,
        right_len: usize,
        table_len: usize,
    ) -> bool {
        self.left_factor.len() == left_len
            && self.right_factor.len() == right_len
            && self.table.len() == table_len
    }

    pub(super) fn expected_claim(&self) -> E {
        self.expected_claim
    }

    pub(super) fn product_claim(&self) -> E {
        match &self.table {
            Stage2HandoffTable::Compact {
                digits, padded_len, ..
            } => product_claim_compact(digits, *padded_len, &self.left_factor, &self.right_factor),
            Stage2HandoffTable::Full(table) => {
                product_claim(table, &self.left_factor, &self.right_factor)
            }
        }
    }

    pub(super) fn accumulate_right_round(&self) -> (E, E, E) {
        match &self.table {
            Stage2HandoffTable::Compact {
                digits,
                padded_len,
                pending_right_challenge,
            } => match pending_right_challenge {
                Some(first_challenge) => accumulate_second_right_round_compact(
                    digits,
                    *padded_len,
                    &self.left_factor,
                    &self.right_factor,
                    *first_challenge,
                ),
                None => accumulate_right_round_compact(
                    digits,
                    *padded_len,
                    &self.left_factor,
                    &self.right_factor,
                ),
            },
            Stage2HandoffTable::Full(table) => {
                accumulate_right_round(table, &self.left_factor, &self.right_factor)
            }
        }
    }

    pub(super) fn accumulate_left_round(&self) -> (E, E, E) {
        match &self.table {
            Stage2HandoffTable::Compact {
                digits,
                padded_len,
                pending_right_challenge,
            } => {
                debug_assert!(pending_right_challenge.is_none());
                accumulate_left_round_compact(
                    digits,
                    *padded_len,
                    &self.left_factor,
                    self.right_factor[0],
                )
            }
            Stage2HandoffTable::Full(table) => {
                accumulate_left_round(table, &self.left_factor, self.right_factor[0])
            }
        }
    }

    pub(super) fn fold_right_round(&mut self, r: E) {
        match &mut self.table {
            Stage2HandoffTable::Compact {
                digits,
                padded_len,
                pending_right_challenge,
            } => {
                if let Some(first_challenge) = pending_right_challenge.take() {
                    let folded = fold_compact_right_two_rounds(
                        digits,
                        *padded_len,
                        &mut self.right_factor,
                        first_challenge,
                        r,
                    );
                    self.table = Stage2HandoffTable::Full(folded);
                } else if self.right_factor.len() >= 4 {
                    fold_right_factor(&mut self.right_factor, r);
                    *pending_right_challenge = Some(r);
                } else {
                    let folded =
                        fold_compact_right_round(digits, *padded_len, &mut self.right_factor, r);
                    self.table = Stage2HandoffTable::Full(folded);
                }
            }
            Stage2HandoffTable::Full(table) => {
                fold_right_round(table, &mut self.right_factor, r);
            }
        }
    }

    pub(super) fn fold_left_round(&mut self, r: E) {
        match &mut self.table {
            Stage2HandoffTable::Compact {
                digits,
                padded_len,
                pending_right_challenge,
            } => {
                debug_assert!(pending_right_challenge.is_none());
                let folded = fold_compact_left_round(digits, *padded_len, &mut self.left_factor, r);
                self.table = Stage2HandoffTable::Full(folded);
            }
            Stage2HandoffTable::Full(table) => {
                fold_left_round(table, &mut self.left_factor, r);
            }
        }
    }

    pub(super) fn final_value(&self) -> E {
        match &self.table {
            Stage2HandoffTable::Compact { digits, .. } => compact_value_at::<E>(digits, 0),
            Stage2HandoffTable::Full(table) => table[0],
        }
    }
}

fn fold_right_factor<E: FieldCore>(right_factor: &mut Vec<E>, r: E) {
    let half = right_factor.len() / 2;
    let folded = (0..half)
        .map(|idx| {
            let left = right_factor[2 * idx];
            let right = right_factor[2 * idx + 1];
            left + r * (right - left)
        })
        .collect::<Vec<_>>();
    *right_factor = folded;
}
