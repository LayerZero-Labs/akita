//! Prover-side materialized relation-weight evaluations.

use akita_field::{AkitaError, FieldCore};
use akita_sumcheck::fold_evals_in_place;

/// Errors specific to [`RelationWeightPolynomial`] validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationWeightPolynomialError {
    LengthMismatch { expected: usize, actual: usize },
    NonzeroPadding { index: usize },
}

impl std::fmt::Display for RelationWeightPolynomialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "relation weight length mismatch (expected {expected}, got {actual})"
                )
            }
            Self::NonzeroPadding { index } => {
                write!(f, "relation weight padding slot {index} is nonzero")
            }
        }
    }
}

impl std::error::Error for RelationWeightPolynomialError {}

fn relation_weight_error(err: RelationWeightPolynomialError) -> AkitaError {
    match err {
        RelationWeightPolynomialError::LengthMismatch { expected, actual } => {
            AkitaError::InvalidSize { expected, actual }
        }
        RelationWeightPolynomialError::NonzeroPadding { index } => {
            AkitaError::InvalidInput(format!("relation weight padding slot {index} is nonzero"))
        }
    }
}

/// Materialized evaluations of the relation-weight multilinear polynomial.
///
/// Evaluations are stored in the same flat column-major order as the Stage-2
/// witness table: `index = x * y_len + y`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationWeightPolynomial<E: FieldCore> {
    evals: Vec<E>,
    y_len: usize,
    live_x_cols: usize,
}

impl<E: FieldCore> RelationWeightPolynomial<E> {
    /// Construct from materialized evaluations over the padded witness hypercube.
    ///
    /// `witness_len` is the live witness length (`live_x_cols * y_len`). Padding
    /// slots above `witness_len` must be zero.
    pub fn from_evals(
        evals: Vec<E>,
        y_len: usize,
        live_x_cols: usize,
        witness_len: usize,
    ) -> Result<Self, AkitaError> {
        if y_len == 0 || live_x_cols == 0 {
            return Err(relation_weight_error(
                RelationWeightPolynomialError::LengthMismatch {
                    expected: live_x_cols.saturating_mul(y_len),
                    actual: evals.len(),
                },
            ));
        }
        if witness_len > live_x_cols.saturating_mul(y_len) {
            return Err(relation_weight_error(
                RelationWeightPolynomialError::LengthMismatch {
                    expected: witness_len,
                    actual: live_x_cols.saturating_mul(y_len),
                },
            ));
        }
        if evals.len() != live_x_cols.saturating_mul(y_len) {
            return Err(relation_weight_error(
                RelationWeightPolynomialError::LengthMismatch {
                    expected: live_x_cols.saturating_mul(y_len),
                    actual: evals.len(),
                },
            ));
        }
        for (idx, value) in evals.iter().enumerate().skip(witness_len) {
            if !value.is_zero() {
                return Err(relation_weight_error(
                    RelationWeightPolynomialError::NonzeroPadding { index: idx },
                ));
            }
        }
        Ok(Self {
            evals,
            y_len,
            live_x_cols,
        })
    }

    #[must_use]
    pub fn evals(&self) -> &[E] {
        &self.evals
    }

    #[must_use]
    pub fn evals_mut(&mut self) -> &mut [E] {
        &mut self.evals
    }

    #[must_use]
    pub fn y_len(&self) -> usize {
        self.y_len
    }

    #[must_use]
    pub fn live_x_cols(&self) -> usize {
        self.live_x_cols
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.evals.len()
    }

    /// Pair of relation-weight evaluations for one sumcheck fold step.
    pub fn pair_flat(&self, idx0: usize, idx1: usize) -> (E, E) {
        (self.evals[idx0], self.evals[idx1])
    }

    /// Fold relation weights alongside the witness table for one challenge round.
    pub fn fold_in_place(&mut self, challenge: E)
    where
        E: akita_field::unreduced::HasOptimizedFold,
    {
        fold_evals_in_place(&mut self.evals, challenge);
    }
}

/// Bridge helper: collapse legacy split tables into one relation-weight vector.
///
/// Used during cutover and in tests until the ring-switch builder emits the
/// unified polynomial directly.
pub fn bridge_relation_weight_from_split<E: FieldCore>(
    alpha_evals_y: &[E],
    m_evals_x: &[E],
    trace_table: Option<&[E]>,
    y_len: usize,
    live_x_cols: usize,
) -> Result<Vec<E>, AkitaError> {
    if alpha_evals_y.len() != y_len {
        return Err(AkitaError::InvalidSize {
            expected: y_len,
            actual: alpha_evals_y.len(),
        });
    }
    if m_evals_x.len() < live_x_cols {
        return Err(AkitaError::InvalidSize {
            expected: live_x_cols,
            actual: m_evals_x.len(),
        });
    }
    let table_len = live_x_cols
        .checked_mul(y_len)
        .ok_or(AkitaError::InvalidProof)?;
    if let Some(trace) = trace_table {
        if trace.len() != table_len {
            return Err(AkitaError::InvalidSize {
                expected: table_len,
                actual: trace.len(),
            });
        }
    }
    let mut out = Vec::with_capacity(table_len);
    for x in 0..live_x_cols {
        let m_val = m_evals_x[x];
        for y in 0..y_len {
            let idx = x * y_len + y;
            let mut weight = alpha_evals_y[y] * m_val;
            if let Some(trace) = trace_table {
                weight += trace[idx];
            }
            out.push(weight);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{FpExt2, FromPrimitiveInt, NegOneNr, Prime128Offset275, Zero};

    type F = Prime128Offset275;
    type E = FpExt2<F, NegOneNr>;

    #[test]
    fn bridge_matches_pointwise_formula() {
        let y_len = 4;
        let live_x_cols = 3;
        let alpha: Vec<E> = (0..y_len).map(|i| E::from_u64(i as u64 + 1)).collect();
        let m: Vec<E> = (0..live_x_cols)
            .map(|i| E::from_u64((i + 1) as u64 * 10))
            .collect();
        let mut trace = vec![E::zero(); live_x_cols * y_len];
        trace[5] = E::from_u64(7);
        let bridged =
            bridge_relation_weight_from_split(&alpha, &m, Some(&trace), y_len, live_x_cols)
                .unwrap();
        for x in 0..live_x_cols {
            for y in 0..y_len {
                let idx = x * y_len + y;
                let expected = alpha[y] * m[x] + trace[idx];
                assert_eq!(bridged[idx], expected, "mismatch at ({x},{y})");
            }
        }
    }

    #[test]
    fn rejects_nonzero_padding() {
        let y_len = 2;
        let live_x_cols = 2;
        let witness_len = 3;
        let mut evals = vec![E::zero(); 4];
        evals[3] = E::one();
        let err = RelationWeightPolynomial::<E>::from_evals(evals, y_len, live_x_cols, witness_len)
            .expect_err("padding");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }
}
