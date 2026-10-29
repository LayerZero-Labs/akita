//! Setup claims and the explicit choice to defer or check them.
use akita_error::AkitaError;
use akita_types::setup_claims::{SetupBatchStatement, SetupClaim, SetupPolynomialId, SetupQuery};
use jolt_field::Field;

pub trait SetupClaimAdvice<E: Field> {
    fn answer(&mut self, query: &SetupQuery<E>) -> Result<E, AkitaError>;
}

/// Setup verification policy. Only deferral needs external direct-evaluation answers.
pub enum SetupMode<'a, E: Field> {
    Defer(&'a mut dyn SetupClaimAdvice<E>),
    Check,
}

/// Input and output setup obligations for one proof verification.
///
/// Start with `new` to collect inner claims. Use `from_batch` to attach a bound
/// collection to the same nonterminal fold selected by the prover.
#[must_use = "every pending setup claim must be inherited or checked before acceptance"]
pub struct SetupClaimAccumulator<E: Field> {
    pub(crate) polynomial: SetupPolynomialId,
    // Keep the bound input separate from claims collected during this replay.
    pub(crate) batch: Option<(u32, SetupBatchStatement<E>)>,
    pub(crate) pending: Vec<SetupClaim<E>>,
}

impl<E: Field> SetupClaimAccumulator<E> {
    pub fn new(polynomial: SetupPolynomialId) -> Self {
        Self {
            polynomial,
            batch: None,
            pending: Vec::new(),
        }
    }

    pub fn from_batch(level: u32, batch: SetupBatchStatement<E>) -> Self {
        Self {
            polynomial: batch.polynomial().clone(),
            batch: Some((level, batch)),
            pending: Vec::new(),
        }
    }

    /// Return all unresolved claims, including an input batch not yet replayed.
    pub fn into_pending(mut self) -> Vec<SetupClaim<E>> {
        if let Some((_, batch)) = self.batch {
            self.pending.extend(batch.claims().iter().cloned());
        }
        self.pending
    }

    pub(crate) fn resolve(
        &mut self,
        query: SetupQuery<E>,
        advice: &mut dyn SetupClaimAdvice<E>,
    ) -> Result<E, AkitaError> {
        query.validate(&self.polynomial)?;
        let value = advice.answer(&query)?;
        self.pending
            .push(SetupClaim::new(self.polynomial.clone(), query, value)?);
        Ok(value)
    }
}
