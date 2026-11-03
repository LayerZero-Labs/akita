use super::{
    polynomial_identity_loss_factor, ChallengeFieldOrder, FoldChallengeCost,
    GrindingPlanAccumulator, GrindingRun, GrindingSite, SumcheckProtocol,
};
use akita_error::{narrowing::usize_to_u32, AkitaError};

/// Consecutive queries whose protocol shape differs only by the round index.
/// This is not a wire run: replay still has one distinct site for every round.
#[derive(Clone, Copy)]
pub(crate) struct SumcheckRoundBatch {
    pub(crate) challenge_order: ChallengeFieldOrder,
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
                round: usize_to_u32(last_round, "sumcheck grinding round")?,
            },
            polynomial_identity_loss_factor(self.degree)?,
            self.challenge_order,
        )
        .map(Some)
    }

    /// The representative run and how many rounds repeat it.
    fn repeated(self) -> Result<Option<(GrindingRun, u32)>, AkitaError> {
        let Some(run) = self.representative()? else {
            return Ok(None);
        };
        let repetitions = u32::try_from(self.rounds)
            .map_err(|_| AkitaError::InvalidSetup("grinding plan run count exceeds u32".into()))?;
        Ok(Some((run, repetitions)))
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
            round: usize_to_u32(round, "sumcheck grinding round")?,
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

    /// The fold proof of work at `level` and each group's challenge message.
    fn fold_queries(
        &mut self,
        level: u32,
        params: &crate::CommittedGroupParams,
        layout: &crate::OpeningClaimsLayout,
    ) -> Result<(), AkitaError> {
        let (work, groups) = crate::fold_challenge_runs(level, params, layout)?;
        for run in std::iter::once(work).chain(groups) {
            self.push(run)?;
        }
        Ok(())
    }

    /// The terminal fold proof of work at `level` and its challenge message.
    fn terminal_fold_queries(
        &mut self,
        level: u32,
        terminal: &crate::TerminalFoldParams,
    ) -> Result<(), AkitaError> {
        let (work, group) =
            crate::transcript_grinding_plan::terminal_fold_challenge_runs(level, terminal)?;
        self.push(work)?;
        self.push(group)
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
        let Some((run, repetitions)) = batch.repeated()? else {
            return Ok(());
        };
        // `push_repeated` validates the representative's reserved site fields.
        self.push_repeated(run, repetitions)
    }
}

/// Planner edge pricing. The plan builders price every run against the
/// edge's own challenge order, so this sink counts them without re-deriving
/// their targets, and it adds the fold challenge costs the planner priced
/// once for the edge's fold and for a terminal successor.
pub(crate) struct PlannerEdgeSink {
    pub(crate) accumulator: GrindingPlanAccumulator,
    pub(crate) fold: FoldChallengeCost,
    pub(crate) terminal_fold: Option<FoldChallengeCost>,
}

impl GrindingPlanSink for PlannerEdgeSink {
    fn push(&mut self, run: GrindingRun) -> Result<(), AkitaError> {
        self.accumulator.count_repeated(run, 1)
    }

    fn sumcheck_rounds(&mut self, batch: SumcheckRoundBatch) -> Result<(), AkitaError> {
        let Some((run, repetitions)) = batch.repeated()? else {
            return Ok(());
        };
        self.accumulator.count_repeated(run, repetitions)
    }

    fn fold_queries(
        &mut self,
        _level: u32,
        _params: &crate::CommittedGroupParams,
        _layout: &crate::OpeningClaimsLayout,
    ) -> Result<(), AkitaError> {
        self.accumulator.add(&self.fold.counts)
    }

    fn terminal_fold_queries(
        &mut self,
        _level: u32,
        _terminal: &crate::TerminalFoldParams,
    ) -> Result<(), AkitaError> {
        let terminal_fold = self.terminal_fold.ok_or_else(|| {
            AkitaError::InvalidSetup("terminal planner edge is missing its fold cost".into())
        })?;
        self.accumulator.add(&terminal_fold.counts)
    }
}

#[cfg(test)]
#[path = "sink/tests.rs"]
mod tests;
