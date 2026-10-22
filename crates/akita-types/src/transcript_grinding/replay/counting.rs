//! Pad-layout replay: the masked verifier run over a counting view of the pad
//! schedule (`specs/whiteout-akita.md` §4.5, N8).

use super::{
    fold_response_site, next_entry, FoldResponseReplay, GrindingPlanCursor, GrindingReplay,
};
use crate::{FoldDrawRecord, FoldSite};
use akita_error::AkitaError;
use akita_params::{GrindingPlan, GrindingQueryKind, GrindingSite};
use jolt_transcript::{Channel, Fork, PrivateChannel, FORK_SEED_LEN};

/// The third [`GrindingReplay`]: it replays the masked verifier over the
/// counting view's channel (`CommittedPadsView::counting()`), so the pads and
/// product operands the replay takes are the honest schedule's.
///
/// Its proof-of-work queries consume their plan entry and read no nonce: the
/// replay runs over zero or arbitrary message bytes, which no nonce predicate
/// accepts. Its fold responses draw the fork at counter 0, the one-shot
/// counter of every masked level. It produces a pad count and an operand list
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

impl<'a, C: Channel + PrivateChannel<Error: Into<AkitaError>>> CountingGrinding<'a, C> {
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

impl<C: Channel + PrivateChannel<Error: Into<AkitaError>>> GrindingReplay
    for CountingGrinding<'_, C>
{
    type State = C;

    fn state_mut(&mut self) -> &mut C {
        self.state
    }

    fn poison(&mut self) {
        self.invalid = true;
    }

    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = next_entry(&mut self.cursor, site, GrindingQueryKind::ProofOfWork).map(|_| ());
        self.checked(result)
    }

    fn check_public(&mut self, _holds: bool) -> Result<(), AkitaError> {
        Ok(())
    }
}

impl<C: Channel + PrivateChannel<Error: Into<AkitaError>>> FoldResponseReplay
    for CountingGrinding<'_, C>
{
    type Sponge = C::Sponge;

    fn read_fold_response(&mut self, site: GrindingSite) -> Result<Fork<C::Sponge>, AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::FoldResponse);
        let entry = self.checked(entry)?;
        self.state.site(fold_response_site(site, entry.nonce_bits));
        let seed: [u8; FORK_SEED_LEN] = self.state.challenge_bytes();
        Ok(Fork::new(&seed, 0))
    }

    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
        draws: &FoldDrawRecord,
    ) -> Result<(), AkitaError> {
        let site = FoldSite::FoldChallenge { level, group }.id();
        let site = self.checked(site)?;
        self.state.site(site.into());
        draws.log_into(&mut *self.state);
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
