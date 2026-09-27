use super::{
    polynomial_identity_loss_factor, GrindingPlanAccumulator, GrindingRun, GrindingSite,
    SumcheckProtocol,
};
use akita_error::AkitaError;

/// Consecutive queries whose protocol shape differs only by the round index.
/// This is not a wire run: replay still has one distinct site for every round.
#[derive(Clone, Copy)]
pub(crate) struct SumcheckRoundBatch {
    pub(crate) capacity: u32,
    pub(crate) protocol: SumcheckProtocol,
    pub(crate) level: u32,
    pub(crate) stage: u32,
    pub(crate) rounds: usize,
    pub(crate) degree: usize,
}

impl SumcheckRoundBatch {
    fn representative(self) -> Result<Option<GrindingRun>, AkitaError> {
        let Some(last_round) = self.rounds.checked_sub(1) else {
            return Ok(None);
        };
        GrindingRun::proof_of_work(
            GrindingSite::SumcheckRound {
                protocol: self.protocol,
                level: self.level,
                stage: self.stage,
                round: crate::narrowing::usize_to_u32(last_round, "sumcheck grinding round")?,
            },
            polynomial_identity_loss_factor(self.degree)?,
            self.capacity,
        )
        .map(Some)
    }

    fn run_at(
        self,
        mut representative: GrindingRun,
        round: usize,
    ) -> Result<GrindingRun, AkitaError> {
        representative.site = GrindingSite::SumcheckRound {
            protocol: self.protocol,
            level: self.level,
            stage: self.stage,
            round: crate::narrowing::usize_to_u32(round, "sumcheck grinding round")?,
        };
        Ok(representative)
    }
}

/// Both replay materialization and planner pricing consume this query order.
/// Every nonempty batch has identical loss, nonce bits, and multiplicity; only
/// the round index differs. Checking the highest index establishes that all
/// earlier indices fit; the accumulator or completed plan validates the site.
/// An empty batch emits no query and deliberately validates no metadata.
/// Closures use the default exact-site expansion; pricing may aggregate.
pub(crate) trait GrindingPlanSink {
    fn push(&mut self, run: GrindingRun) -> Result<(), AkitaError>;

    fn sumcheck_rounds(&mut self, batch: SumcheckRoundBatch) -> Result<(), AkitaError> {
        let Some(representative) = batch.representative()? else {
            return Ok(());
        };
        for round in 0..batch.rounds {
            self.push(batch.run_at(representative, round)?)?;
        }
        Ok(())
    }
}

impl<F: FnMut(GrindingRun) -> Result<(), AkitaError>> GrindingPlanSink for F {
    fn push(&mut self, run: GrindingRun) -> Result<(), AkitaError> {
        self(run)
    }
}

impl GrindingPlanSink for GrindingPlanAccumulator {
    fn push(&mut self, run: GrindingRun) -> Result<(), AkitaError> {
        self.push_repeated(run, 1)
    }

    fn sumcheck_rounds(&mut self, batch: SumcheckRoundBatch) -> Result<(), AkitaError> {
        let Some(run) = batch.representative()? else {
            return Ok(());
        };
        let repetitions = u32::try_from(batch.rounds)
            .map_err(|_| AkitaError::InvalidSetup("grinding plan run count exceeds u32".into()))?;
        // `push_repeated` validates the representative's reserved site fields.
        self.push_repeated(run, repetitions)
    }
}

#[cfg(test)]
#[path = "sink/tests.rs"]
mod tests;
