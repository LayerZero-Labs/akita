use super::*;

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
    fn run(self, round: usize) -> Result<GrindingRun, AkitaError> {
        GrindingRun::proof_of_work(
            GrindingSite::SumcheckRound {
                protocol: self.protocol,
                level: self.level,
                stage: self.stage,
                round: crate::narrowing::usize_to_u32(round, "sumcheck grinding round")?,
            },
            polynomial_identity_loss_factor(self.degree)?,
            self.capacity,
        )
    }
}

/// Both replay materialization and planner pricing consume this query order.
/// The default expands exact sites; pricing can aggregate identical round costs.
pub(crate) trait GrindingPlanSink {
    fn push(&mut self, run: GrindingRun) -> Result<(), AkitaError>;

    fn sumcheck_rounds(&mut self, batch: SumcheckRoundBatch) -> Result<(), AkitaError> {
        for round in 0..batch.rounds {
            self.push(batch.run(round)?)?;
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
        let Some(last_round) = batch.rounds.checked_sub(1) else {
            // Match an empty canonical loop: even invalid degree/capacity/site
            // metadata has no query to validate.
            return Ok(());
        };
        let run = batch.run(last_round)?;
        let repetitions = u32::try_from(batch.rounds)
            .map_err(|_| AkitaError::InvalidSetup("grinding plan run count exceeds u32".into()))?;
        // The canonical run validator checks the last round's reserved sentinel
        // and the shared level/stage. Every preceding index is then valid.
        // Sumcheck runs have multiplicity one; this counts distinct wire runs.
        self.push_repeated(run, repetitions)
    }
}

#[cfg(test)]
#[path = "sink/tests.rs"]
mod tests;
