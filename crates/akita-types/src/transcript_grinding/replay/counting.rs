//! Pad-layout replay: the masked verifier run over a counting view of the pad
//! schedule (`specs/whiteout-akita.md` §4.5).

use super::{next_entry, FoldResponseReplay, GrindingPlanCursor, GrindingReplay};
use akita_error::AkitaError;
use akita_params::{GrindingPlan, GrindingQueryKind, GrindingSite};
use akita_transcript::ProofChannel;

/// The third [`GrindingReplay`]: it replays the masked verifier over the
/// counting view's channel (`CommittedPadsView::counting()`), so the pads and
/// product operands the replay takes are the honest schedule's.
///
/// Its proof-of-work queries and fold responses consume their plan entries and
/// read nothing: the replay runs over zero message bytes, which no nonce
/// predicate accepts. Every masked level's fold response is one-shot, so its
/// counter is 0 either way. It produces a pad count and an operand list
/// through its channel, never a verdict on a proof.
///
/// Run it over zero bytes, or a reader that never fails, and never over proof
/// bytes: the layout is a function of the public schedule, and a proof's
/// bytes can end early or fail to decode, which would make the layout depend
/// on the proof.
pub struct CountingGrinding<'a, C> {
    state: &'a mut C,
    cursor: GrindingPlanCursor<'a>,
    invalid: bool,
}

impl<'a, C: ProofChannel> CountingGrinding<'a, C> {
    /// Attach the counting channel to the schedule's public grinding plan.
    #[must_use]
    pub fn new(state: &'a mut C, plan: &'a GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            invalid: false,
        }
    }

    /// Require that the replay consumed exactly the plan of levels
    /// `0..=cut_level`.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] if an earlier step failed or the replay
    /// stopped elsewhere.
    pub fn finish_through(self, cut_level: u32) -> Result<(), AkitaError> {
        if self.invalid || !self.cursor.is_finished_through(cut_level) {
            return Err(AkitaError::InvalidInput(
                "the counting replay did not stop exactly at the cut".into(),
            ));
        }
        Ok(())
    }

    fn checked<T>(&mut self, result: Result<T, AkitaError>) -> Result<T, AkitaError> {
        self.invalid |= result.is_err();
        result
    }
}

impl<C: ProofChannel> GrindingReplay for CountingGrinding<'_, C> {
    type State = C;

    const COUNTS_ONLY: bool = true;

    fn state_mut(&mut self) -> &mut C {
        self.state
    }

    fn poison(&mut self) {
        self.invalid = true;
    }

    /// The counting replay runs over no proof, so a public check gives no
    /// verdict.
    fn check_public(&mut self, _holds: bool) -> Result<(), AkitaError> {
        Ok(())
    }

    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = next_entry(&mut self.cursor, site, GrindingQueryKind::ProofOfWork).map(|_| ());
        self.checked(result)
    }
}

impl<C: ProofChannel> FoldResponseReplay for CountingGrinding<'_, C> {
    fn read_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let entry =
            next_entry(&mut self.cursor, site, GrindingQueryKind::FoldResponse).and_then(|_| {
                next_entry(
                    &mut self.cursor,
                    GrindingSite::FoldChallengeWork {
                        level: site.level(),
                    },
                    GrindingQueryKind::FoldChallengeWork,
                )
            });
        self.checked(entry)?;
        Ok(0)
    }

    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
    ) -> Result<(), AkitaError> {
        let result = coordinate_count
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)
            .and_then(|multiplicity| {
                self.cursor.consume_run(
                    GrindingSite::FoldChallengeGroup { level, group },
                    multiplicity,
                )
            });
        self.checked(result)
    }
}
