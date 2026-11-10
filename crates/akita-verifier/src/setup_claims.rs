//! Setup claims and the explicit choice to defer or check them.
use akita_error::AkitaError;
use akita_types::setup_claims::{SetupBatchStatement, SetupClaim, SetupPolynomialId, SetupQuery};
use jolt_field::Field;

pub trait SetupClaimAdvice<E: Field> {
    fn answer(&mut self, query: &SetupQuery<E>) -> Result<E, AkitaError>;
}

/// Setup obligations for one verification. Stage 3 reductions are always checked.
pub enum SetupPolicy<'a, 'b, E: Field> {
    /// Check this proof and any inherited batch. `None` is ordinary verification;
    /// an empty accumulator is also accepted. Pending claims and advice are
    /// rejected at entry; bind pending claims with `from_batch` first.
    Check(Option<&'a mut SetupClaimAccumulator<'b, E>>),
    /// Export direct setup evaluations using the accumulator's untrusted advice.
    Defer(&'a mut SetupClaimAccumulator<'b, E>),
}

impl<'a, 'b, E: Field> SetupPolicy<'a, 'b, E> {
    pub(crate) fn accumulator(&mut self) -> Option<&mut SetupClaimAccumulator<'b, E>> {
        match self {
            Self::Check(accumulator) => accumulator.as_deref_mut(),
            Self::Defer(accumulator) => Some(accumulator),
        }
    }

    pub(crate) fn pending_len(&mut self) -> usize {
        self.accumulator().map_or(0, |a| a.pending.len())
    }

    /// Consume the inherited batch on acceptance; on rejection, drop the
    /// claims collected since `pending_len` and keep the inherited batch.
    pub(crate) fn settle(&mut self, pending_len: usize, accepted: bool) {
        if let Some(accumulator) = self.accumulator() {
            if accepted {
                // The validated target is one of the nonterminal folds replayed.
                accumulator.batch = None;
            } else {
                accumulator.pending.truncate(pending_len);
            }
        }
    }
}
pub use akita_params::SetupMode;

/// Input and output setup obligations for one proof verification.
///
/// Start with `new` to collect inner claims. Use `from_batch` to attach a bound
/// collection to the same nonterminal fold selected by the prover.
#[must_use = "every pending setup claim must be inherited or checked before acceptance"]
pub struct SetupClaimAccumulator<'a, E: Field> {
    pub(crate) polynomial: SetupPolynomialId,
    // Keep the bound input separate from claims collected during this replay.
    pub(crate) batch: Option<(u32, SetupBatchStatement<E>)>,
    pub(crate) pending: Vec<SetupClaim<E>>,
    pub(crate) advice: Option<&'a mut dyn SetupClaimAdvice<E>>,
}

impl<'a, E: Field> SetupClaimAccumulator<'a, E> {
    pub fn new(polynomial: SetupPolynomialId) -> Self {
        Self {
            polynomial,
            batch: None,
            pending: Vec::new(),
            advice: None,
        }
    }

    pub fn from_batch(level: u32, batch: SetupBatchStatement<E>) -> Self {
        Self {
            polynomial: batch.polynomial().clone(),
            batch: (!batch.claims().is_empty()).then_some((level, batch)),
            pending: Vec::new(),
            advice: None,
        }
    }

    /// Supply candidate answers for direct folds of an archived proof.
    /// Every answer becomes a pending obligation; advice never authenticates it.
    pub fn with_advice(mut self, advice: &'a mut dyn SetupClaimAdvice<E>) -> Self {
        self.advice = Some(advice);
        self
    }

    /// Return all unresolved claims, including an input batch not yet replayed.
    pub fn into_pending(mut self) -> Vec<SetupClaim<E>> {
        if let Some((_, batch)) = self.batch {
            self.pending.extend(batch.claims().iter().cloned());
        }
        self.pending
    }

    pub(crate) fn resolve(&mut self, query: SetupQuery<E>) -> Result<E, AkitaError> {
        query.validate(&self.polynomial)?;
        let value = self
            .advice
            .as_deref_mut()
            .ok_or_else(|| {
                AkitaError::InvalidInput("deferred direct setup evaluation requires advice".into())
            })?
            .answer(&query)?;
        self.pending
            .push(SetupClaim::new(self.polynomial.clone(), query, value)?);
        Ok(value)
    }
}
