//! Inline Spongefish nonce replay for transcript grinding.

use akita_error::AkitaError;
use akita_params::{FoldResponseSearch, GrindingPlan, GrindingQueryKind, GrindingSite};
#[cfg(feature = "prover")]
use akita_sumcheck::SumcheckProverChannel;
use akita_sumcheck::{SumcheckRole, SumcheckVerifierChannel};
use akita_transcript::{
    ext_challenges, nonce_max_bytes, receive_nonce, reject, verifier_context, ChallengeCodec,
    ProofChannel, ProtocolContextRecord, ProtocolMessageKind, ProtocolSiteId, VerifierChannel,
    VerifierTranscript, GRINDING_CHECK_BYTES, GRINDING_NONCE_SLACK_BITS, SITE_FAMILY_SUMCHECK,
};
#[cfg(feature = "prover")]
use akita_transcript::{
    nonce_encoded_len, prover_context, send_nonce, FoldPreview, FoldWork, PrivateProofChannel,
    ProverChannel, ProverTranscript,
};
#[cfg(feature = "prover")]
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use std::marker::PhantomData;

#[derive(Clone, Copy)]
struct GrindingPlanEntry {
    site: GrindingSite,
    grind_bits: u8,
    nonce_bits: u8,
    fold_search: Option<FoldResponseSearch>,
    binary_group: Option<u32>,
}

struct GrindingPlanCursor<'a> {
    plan: &'a GrindingPlan,
    run_index: usize,
    run_offset: u64,
}

impl<'a> GrindingPlanCursor<'a> {
    const fn new(plan: &'a GrindingPlan) -> Self {
        Self {
            plan,
            run_index: 0,
            run_offset: 0,
        }
    }

    fn next(&mut self) -> Option<GrindingPlanEntry> {
        let run = *self.plan.runs().get(self.run_index)?;
        let entry = GrindingPlanEntry {
            site: run.site(),
            grind_bits: run.grind_bits(),
            nonce_bits: run.nonce_bits(),
            fold_search: run.fold_response_search(),
            binary_group: run.binary_group(),
        };
        self.run_offset += 1;
        if self.run_offset == run.multiplicity() {
            self.run_index += 1;
            self.run_offset = 0;
        }
        Some(entry)
    }

    #[cfg(feature = "prover")]
    fn peek(&self) -> Option<GrindingPlanEntry> {
        let run = *self.plan.runs().get(self.run_index)?;
        Some(GrindingPlanEntry {
            site: run.site(),
            grind_bits: run.grind_bits(),
            nonce_bits: run.nonce_bits(),
            fold_search: run.fold_response_search(),
            binary_group: run.binary_group(),
        })
    }

    /// Entry of the run after the current one, when the cursor sits at the
    /// start of a single-entry run.
    #[cfg(feature = "prover")]
    fn peek_following(&self) -> Option<GrindingPlanEntry> {
        let current = self.plan.runs().get(self.run_index)?;
        if self.run_offset != 0 || current.multiplicity() != 1 {
            return None;
        }
        let run = *self.plan.runs().get(self.run_index.checked_add(1)?)?;
        Some(GrindingPlanEntry {
            site: run.site(),
            grind_bits: run.grind_bits(),
            nonce_bits: run.nonce_bits(),
            fold_search: run.fold_response_search(),
            binary_group: run.binary_group(),
        })
    }

    fn consume_run(&mut self, site: GrindingSite, multiplicity: usize) -> Result<(), AkitaError> {
        let run = self.plan.runs().get(self.run_index).ok_or_else(|| {
            AkitaError::InvalidInput("fold-challenge replay has no remaining plan run".into())
        })?;
        if self.run_offset != 0
            || run.site() != site
            || run.grind_bits() != 0
            || run.nonce_bits() != 0
            || usize::try_from(run.multiplicity()).ok() != Some(multiplicity)
        {
            return Err(AkitaError::InvalidInput(
                "fold-challenge replay request differs from the next plan run".into(),
            ));
        }
        self.run_index = self.run_index.checked_add(1).ok_or_else(|| {
            AkitaError::Internal("fold-challenge replay run index overflows usize".into())
        })?;
        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.run_index == self.plan.runs().len() && self.run_offset == 0
    }

    /// Whether every unconsumed entry belongs to a level after `cut_level`:
    /// a replay that stops at a zero-knowledge cut consumed exactly the
    /// masked levels' entries (`specs/whiteout-akita.md` §4.4).
    fn is_finished_through(&self, cut_level: u32) -> bool {
        self.run_offset == 0
            && self
                .plan
                .runs()
                .get(self.run_index..)
                .is_some_and(|rest| rest.iter().all(|run| run.site().level() > cut_level))
    }
}

const fn value_fits(value: u32, width: u8) -> bool {
    match width {
        0..=31 => value < (1u32 << width),
        32 => true,
        _ => false,
    }
}

fn next_entry(
    cursor: &mut GrindingPlanCursor<'_>,
    site: GrindingSite,
    kind: GrindingQueryKind,
) -> Result<GrindingPlanEntry, AkitaError> {
    let entry = cursor.next().ok_or_else(|| {
        AkitaError::InvalidInput("grinding query replay has no remaining plan entry".into())
    })?;
    if entry.site != site || site.kind() != kind || entry.binary_group.is_some() {
        return Err(AkitaError::InvalidInput(
            "grinding query replay site or kind differs from the next plan entry".into(),
        ));
    }
    Ok(entry)
}

/// Difficulty of one proof-of-work entry, or `None` when the entry's nonce
/// width is not the `grind_bits + 7` that Jolt's proof-of-work step reads.
fn pow_bits(entry: GrindingPlanEntry) -> Option<u32> {
    let nonce_bits = match entry.grind_bits {
        0 => 0,
        bits => bits.checked_add(GRINDING_NONCE_SLACK_BITS)?,
    };
    (entry.nonce_bits == nonce_bits).then_some(u32::from(entry.grind_bits))
}

fn grinding_records(
    site: GrindingSite,
    grind_bits: u8,
    nonce_bits: u8,
) -> (ProtocolContextRecord, ProtocolContextRecord) {
    let detail = u32::from(grind_bits) | (u32::from(nonce_bits) << 8);
    let site_id = site.site_id(detail).to_bytes();
    (
        ProtocolContextRecord::new(
            site_id,
            ProtocolMessageKind::GrindingNonce as u32,
            1,
            nonce_max_bytes(nonce_bits) as u64,
            0,
        ),
        ProtocolContextRecord::new(
            site_id,
            ProtocolMessageKind::GrindingCheck as u32,
            0,
            0,
            u64::from(GRINDING_CHECK_BYTES),
        ),
    )
}

fn fold_response_record(site: GrindingSite, nonce_bits: u8) -> ProtocolContextRecord {
    ProtocolContextRecord::new(
        site.site_id(u32::from(nonce_bits)).to_bytes(),
        ProtocolMessageKind::FoldResponseNonce as u32,
        1,
        nonce_max_bytes(nonce_bits) as u64,
        0,
    )
}

/// A prover channel Akita runs its grinding plan over: the clear channel, or
/// a masked channel over it (`specs/whiteout-akita.md` §4.4). Proof of work,
/// nonces and context records go to the clear channel underneath, so they
/// take no pad and cost the same bytes in both modes.
#[cfg(feature = "prover")]
pub trait ProverGrindingChannel: ProofChannel {
    /// The clear channel underneath.
    fn clear_state(&mut self) -> &mut ProverChannel;

    /// The clear channel underneath, read-only.
    fn clear_state_ref(&self) -> &ProverChannel;

    /// Whether this channel masks its private messages.
    fn masks(&self) -> bool;
}

#[cfg(feature = "prover")]
impl ProverGrindingChannel for ProverChannel {
    fn clear_state(&mut self) -> &mut ProverChannel {
        self
    }

    fn clear_state_ref(&self) -> &ProverChannel {
        self
    }

    fn masks(&self) -> bool {
        false
    }
}

/// A verifier channel Akita runs its grinding plan over: the clear channel,
/// or a masked channel over it. As [`ProverGrindingChannel`].
pub trait VerifierGrindingChannel<'proof>: ProofChannel {
    /// The clear channel underneath.
    fn clear_state(&mut self) -> &mut VerifierChannel<'proof>;
}

impl<'proof> VerifierGrindingChannel<'proof> for VerifierChannel<'proof> {
    fn clear_state(&mut self) -> &mut VerifierChannel<'proof> {
        self
    }
}

/// One side of grinding-plan replay.
///
/// Scheduled-query code written against this trait runs unchanged for the
/// prover and the verifier. A failed replay call may already have advanced the
/// transcript, so it leaves the replay unusable and `finish` fails even if the
/// caller drops the error.
pub trait GrindingReplay {
    /// The role-specific proof channel state.
    type State: ProofChannel;

    /// Borrow the state for ordinary protocol messages and challenges.
    ///
    /// Callers must propagate errors from operations on this borrow before
    /// `finish`; the verifier state also records them itself.
    fn state_mut(&mut self) -> &mut Self::State;

    /// Mark this replay as failed.
    fn poison(&mut self);

    /// A verifier check on public values read from the proof: rejects when
    /// it fails. The counting replay, which reads no proof, never rejects.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidProof`] when `holds` is false.
    fn check_public(&mut self, holds: bool) -> Result<(), AkitaError> {
        if holds {
            Ok(())
        } else {
            self.poison();
            Err(AkitaError::InvalidProof)
        }
    }

    /// Apply one scheduled proof-of-work query: the prover searches and emits
    /// the nonce, the verifier receives and checks it.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError`] when `site` is not the next plan entry, the
    /// search is exhausted, or the received nonce is invalid.
    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError>;

    /// Apply one scheduled grinding query and draw the `count` challenges it
    /// protects as one challenge message, directly after the proof of work.
    ///
    /// # Errors
    ///
    /// Propagates [`GrindingReplay::grind_query`] failures, and rejects a
    /// count the challenge codec does not support.
    fn grinded_ext_challenges<E: ChallengeCodec>(
        &mut self,
        site: GrindingSite,
        count: usize,
    ) -> Result<Vec<E>, AkitaError> {
        self.grind_query(site)?;
        let result = ext_challenges::<E, _>(self.state_mut(), site.site_id(u32::default()), count);
        if result.is_err() {
            self.poison();
        }
        result
    }

    /// Apply scheduled work and draw the site's single challenge.
    ///
    /// # Errors
    ///
    /// As for [`GrindingReplay::grinded_ext_challenges`].
    fn grinded_ext_challenge<E: ChallengeCodec>(
        &mut self,
        site: GrindingSite,
    ) -> Result<E, AkitaError> {
        let mut challenges = self.grinded_ext_challenges::<E>(site, 1)?;
        challenges.pop().ok_or_else(|| {
            self.poison();
            AkitaError::Internal("challenge codec returned no element".into())
        })
    }
}

#[cfg(feature = "prover")]
impl<C: ProverGrindingChannel> GrindingReplay for ProverGrinding<'_, C> {
    type State = C;

    fn state_mut(&mut self) -> &mut C {
        &mut *self.state
    }

    fn poison(&mut self) {
        self.invalid = true;
    }

    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = self.grind_query_inner(site, GrindingQueryKind::ProofOfWork);
        if result.is_err() {
            self.poison();
        }
        result
    }
}

impl<'proof, C: VerifierGrindingChannel<'proof>> GrindingReplay
    for VerifierGrinding<'proof, '_, C>
{
    type State = C;

    fn state_mut(&mut self) -> &mut C {
        &mut *self.state
    }

    fn poison(&mut self) {
        self.invalid = true;
        let _ = reject(self.state.clear_state());
    }

    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = self.grind_query_inner(site, GrindingQueryKind::ProofOfWork);
        if result.is_err() {
            self.poison();
        }
        result
    }
}

/// Prover-side state paired with exact grinding-plan progress.
///
/// The state is borrowed: it is Akita's own standalone channel or a caller's
/// channel that Akita's messages join, or a masked channel over either.
#[cfg(feature = "prover")]
pub struct ProverGrinding<'plan, C = ProverChannel> {
    state: &'plan mut C,
    cursor: GrindingPlanCursor<'plan>,
    serialized_nonce_bytes: usize,
    invalid: bool,
}

#[cfg(feature = "prover")]
impl<'plan, C: ProverGrindingChannel> ProverGrinding<'plan, C> {
    /// Attach a prover channel to its public grinding plan.
    #[must_use]
    pub fn new(state: &'plan mut C, plan: &'plan GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            serialized_nonce_bytes: 0,
            invalid: false,
        }
    }

    fn grind_query_inner(
        &mut self,
        site: GrindingSite,
        kind: GrindingQueryKind,
    ) -> Result<(), AkitaError> {
        self.proof_of_work(site, kind, |state, bits| {
            state.grind(bits).map_err(|_| {
                AkitaError::Internal("transcript proof of work failed to find a nonce".into())
            })
        })
    }

    /// Consume the next proof-of-work entry and let `work` produce it on the
    /// live channel: a fresh search, or a nonce already found on a preview.
    fn proof_of_work(
        &mut self,
        site: GrindingSite,
        kind: GrindingQueryKind,
        work: impl FnOnce(&mut ProverChannel, u32) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let entry = next_entry(&mut self.cursor, site, kind)?;
        let bits = pow_bits(entry).ok_or_else(|| {
            AkitaError::Internal("grinding entry nonce width differs from grind_bits + 7".into())
        })?;
        if bits == 0 {
            return Ok(());
        }
        let (nonce_record, check_record) =
            grinding_records(site, entry.grind_bits, entry.nonce_bits);
        let state = self.state.clear_state();
        let start = state.narg_string().len();
        prover_context(state, nonce_record);
        work(state, bits)?;
        prover_context(state, check_record);
        let nonce_bytes = state.narg_string().len() - start;
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(nonce_bytes)
            .ok_or_else(|| {
                AkitaError::Internal("proof-of-work serialized nonce byte count overflow".into())
            })?;
        Ok(())
    }

    /// Emit the accepted fold-response nonce as a canonical message, then the
    /// fold's proof of work found on its preview, which protects every
    /// fold-challenge message drawn after it. A one-shot response sends no
    /// nonce: its counter is 0.
    pub fn commit_fold_response(
        &mut self,
        site: GrindingSite,
        work: FoldWork,
    ) -> Result<(), AkitaError> {
        let result = self.commit_fold_response_inner(site, work);
        self.invalid |= result.is_err();
        result
    }

    fn commit_fold_response_inner(
        &mut self,
        site: GrindingSite,
        work: FoldWork,
    ) -> Result<(), AkitaError> {
        let counter = work.nonce();
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::FoldResponse)?;
        if !value_fits(counter, entry.nonce_bits) {
            return Err(AkitaError::InvalidInput(
                "fold-response nonce exceeds its scheduled bit width".into(),
            ));
        }
        if entry.fold_search != Some(FoldResponseSearch::OneShot) {
            let state = self.state.clear_state();
            prover_context(state, fold_response_record(site, entry.nonce_bits));
            send_nonce(state, counter);
            self.serialized_nonce_bytes = self
                .serialized_nonce_bytes
                .checked_add(nonce_encoded_len(counter))
                .ok_or_else(|| {
                    AkitaError::Internal(
                        "fold-response serialized nonce byte count overflow".into(),
                    )
                })?;
            tracing::info!(
                role = "prover",
                level = site.level(),
                accepted_nonce = counter,
                rejected_attempts = counter,
                attempts = u64::from(counter) + 1,
                encoded_bytes = nonce_encoded_len(counter),
                "native fold response nonce"
            );
        }
        self.proof_of_work(
            GrindingSite::FoldChallengeWork {
                level: site.level(),
            },
            GrindingQueryKind::FoldChallengeWork,
            |state, _bits| work.commit(state),
        )
    }

    /// How the plan searches the fold response at `site`: by counter, or one
    /// shot.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] when `site` is not the next plan
    /// entry or not a fold response.
    pub fn fold_response_search(
        &self,
        site: GrindingSite,
    ) -> Result<FoldResponseSearch, AkitaError> {
        let entry = self.cursor.peek().ok_or_else(|| {
            AkitaError::InvalidInput("fold-response search has no remaining plan entry".into())
        })?;
        if entry.site != site || site.kind() != GrindingQueryKind::FoldResponse {
            return Err(AkitaError::InvalidInput(
                "fold-response search site differs from the next plan entry".into(),
            ));
        }
        entry.fold_search.ok_or_else(|| {
            AkitaError::InvalidInput("fold-response plan entry has no search".into())
        })
    }

    /// Preview a fold-response candidate from the current public state
    /// without advancing either the live state or the grinding plan: absorb
    /// the candidate nonce (none for a one-shot response), then run the fold's
    /// proof of work. Pass the accepted preview's [`FoldPreview::take_work`]
    /// to [`commit_fold_response`](Self::commit_fold_response).
    pub fn preview_fold_response(
        &self,
        site: GrindingSite,
        counter: u32,
    ) -> Result<FoldPreview, AkitaError> {
        let entry = self.cursor.peek().ok_or_else(|| {
            AkitaError::InvalidInput(
                "fold-response preview has no remaining grinding plan entry".into(),
            )
        })?;
        if entry.site != site || entry.binary_group.is_some() {
            return Err(AkitaError::InvalidInput(
                "fold-response preview site differs from grinding plan".into(),
            ));
        }
        if site.kind() != GrindingQueryKind::FoldResponse {
            return Err(AkitaError::InvalidInput(
                "fold-response preview query kind is not FoldResponse".into(),
            ));
        }
        if !value_fits(counter, entry.nonce_bits) {
            return Err(AkitaError::InvalidInput(
                "fold-response preview nonce exceeds scheduled bit width".into(),
            ));
        }
        let one_shot = entry.fold_search == Some(FoldResponseSearch::OneShot);
        if one_shot && counter != 0 {
            return Err(AkitaError::InvalidInput(
                "a one-shot fold response has only counter 0".into(),
            ));
        }
        let work_site = GrindingSite::FoldChallengeWork {
            level: site.level(),
        };
        let work = self
            .cursor
            .peek_following()
            .filter(|work| work.site == work_site && work.binary_group.is_none())
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "fold-response preview is not followed by the fold proof of work".into(),
                )
            })?;
        let bits = pow_bits(work).ok_or_else(|| {
            AkitaError::Internal("grinding entry nonce width differs from grind_bits + 7".into())
        })?;
        let state = self.state.clear_state_ref();
        let mut preview = if one_shot {
            FoldPreview::one_shot(state)
        } else {
            FoldPreview::new(state, counter)
        };
        preview.grind(bits)?;
        Ok(preview)
    }

    /// Consume the plan entries for one sparse fold group.
    pub fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
    ) -> Result<(), AkitaError> {
        let result = coordinate_count
            .checked_add(1)
            .ok_or_else(|| {
                AkitaError::InvalidInput("fold-challenge coordinate count overflow".into())
            })
            .and_then(|multiplicity| {
                self.cursor.consume_run(
                    GrindingSite::FoldChallengeGroup { level, group },
                    multiplicity,
                )
            });
        self.invalid |= result.is_err();
        result
    }

    /// Require that replay consumed exactly the grinding plan of levels
    /// `0..=cut_level` without error: a zero-knowledge opening stops at its
    /// cut (`specs/whiteout-akita.md` §4.4).
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] when an earlier step failed or an
    /// entry of a level up to `cut_level` is unconsumed.
    pub fn finish_through(self, cut_level: u32) -> Result<(), AkitaError> {
        if self.invalid || !self.cursor.is_finished_through(cut_level) {
            return Err(AkitaError::InvalidInput(
                "the grinding replay did not stop exactly at the cut".into(),
            ));
        }
        Ok(())
    }

    /// Finish plan replay. Akita's messages are now in the borrowed channel's
    /// argument string.
    pub fn finish(self) -> Result<(), AkitaError> {
        if self.invalid {
            return Err(AkitaError::InvalidInput(
                "an earlier grinding step failed".into(),
            ));
        }
        if !self.cursor.is_finished() {
            return Err(AkitaError::InvalidInput(
                "the grinding plan has unconsumed entries".into(),
            ));
        }
        tracing::info!(
            role = "prover",
            native_nonce_bytes_actual = self.serialized_nonce_bytes,
            "native proof nonce bytes"
        );
        #[cfg(feature = "logging-transcript")]
        akita_transcript::finish_proof_ranges(self.state.clear_state_ref());
        Ok(())
    }
}

/// Verifier-side state paired with exact grinding-plan progress.
///
/// The state is borrowed: it is Akita's own standalone channel or a caller's
/// channel whose argument string continues with Akita's proof, or a masked
/// channel over either.
pub struct VerifierGrinding<'proof, 'plan, C = VerifierChannel<'proof>> {
    state: &'plan mut C,
    cursor: GrindingPlanCursor<'plan>,
    invalid: bool,
    _proof: PhantomData<&'proof ()>,
}

/// Evidence that replay consumed the complete grinding plan and proof.
///
/// The private field makes this value constructible only by
/// [`PendingEof::check_eof`].
#[derive(Debug)]
pub struct ProofAcceptance {
    _private: (),
}

/// Evidence that Akita's verifier accepted its part of the argument string,
/// before the end of that string is checked.
///
/// The verifier state may carry a caller's messages after Akita's, so the
/// owner of the state checks the end once its own last message is read. A
/// poisoned state fails that check, so no rejected proof reaches
/// [`ProofAcceptance`].
#[derive(Debug)]
#[must_use = "Akita's proof is accepted only after `PendingEof::check_eof`"]
pub struct PendingEof {
    _private: (),
}

impl PendingEof {
    /// Consume the verifier state and require that its whole argument string
    /// was read and never rejected.
    ///
    /// `state` must be the state Akita verified on. Nothing ties the token to
    /// that state, so passing another state skips this check.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] on unread bytes or a rejected
    /// state.
    pub fn check_eof(self, state: VerifierChannel<'_>) -> Result<ProofAcceptance, AkitaError> {
        state.check_eof().map_err(|_| AkitaError::InvalidProof)?;
        Ok(ProofAcceptance { _private: () })
    }
}

impl<'proof, 'plan, C: VerifierGrindingChannel<'proof>> VerifierGrinding<'proof, 'plan, C> {
    /// Attach a verifier channel to its public grinding plan.
    #[must_use]
    pub const fn new(state: &'plan mut C, plan: &'plan GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            invalid: false,
            _proof: PhantomData,
        }
    }

    fn grind_query_inner(
        &mut self,
        site: GrindingSite,
        kind: GrindingQueryKind,
    ) -> Result<(), AkitaError> {
        let entry = next_entry(&mut self.cursor, site, kind)?;
        let bits = pow_bits(entry).ok_or(AkitaError::InvalidProof)?;
        if bits == 0 {
            return Ok(());
        }
        let (nonce_record, check_record) =
            grinding_records(site, entry.grind_bits, entry.nonce_bits);
        let state = self.state.clear_state();
        verifier_context(state, nonce_record);
        state
            .check_grind(bits)
            .map_err(|_| AkitaError::InvalidProof)?;
        verifier_context(state, check_record);
        Ok(())
    }

    fn read_fold_response_inner(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::FoldResponse)?;
        // A one-shot response carries no nonce; reading one would consume a
        // byte of the next message.
        let counter = if entry.fold_search == Some(FoldResponseSearch::OneShot) {
            0
        } else {
            let state = self.state.clear_state();
            verifier_context(state, fold_response_record(site, entry.nonce_bits));
            let counter = receive_nonce(state)?;
            if !value_fits(counter, entry.nonce_bits) {
                return Err(AkitaError::InvalidProof);
            }
            counter
        };
        self.grind_query_inner(
            GrindingSite::FoldChallengeWork {
                level: site.level(),
            },
            GrindingQueryKind::FoldChallengeWork,
        )?;
        Ok(counter)
    }

    /// Require grinding-plan completion. The end of the argument string is
    /// checked by the state's owner through the returned [`PendingEof`].
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] and rejects the state when an
    /// earlier step failed or the plan has unconsumed entries.
    pub fn finish(mut self) -> Result<PendingEof, AkitaError> {
        if self.invalid || !self.cursor.is_finished() {
            self.poison();
            return Err(AkitaError::InvalidProof);
        }
        Ok(PendingEof { _private: () })
    }

    /// Require that replay consumed exactly the grinding plan of levels
    /// `0..=cut_level` without error. The end of the argument string is the
    /// zero-knowledge caller's to check: its tail follows the cut.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] and rejects the state otherwise.
    pub fn finish_through(mut self, cut_level: u32) -> Result<(), AkitaError> {
        if self.invalid || !self.cursor.is_finished_through(cut_level) {
            self.poison();
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}

/// The verifier side of grinding replay beyond [`GrindingReplay`]: the
/// fold-response reads and fold-challenge records. [`VerifierGrinding`]
/// checks the proof; the zero-knowledge layout replay
/// (`CountingGrinding`) only counts the pads a masked replay takes.
pub trait FoldResponseReplay: GrindingReplay {
    /// Receive the next scheduled fold-response nonce (0 for a one-shot
    /// response), then check the fold's proof of work.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] on a malformed or out-of-range
    /// nonce, a failed proof of work, or a plan mismatch.
    fn read_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError>;

    /// Consume the plan entries for one sparse fold group.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] on a plan mismatch.
    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
    ) -> Result<(), AkitaError>;
}

impl<'proof, C: VerifierGrindingChannel<'proof>> FoldResponseReplay
    for VerifierGrinding<'proof, '_, C>
{
    fn read_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let result = self.read_fold_response_inner(site);
        if result.is_err() {
            self.poison();
        }
        result
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
        if result.is_err() {
            self.poison();
        }
        result
    }
}

/// Site of one grinding-backed standard-sumcheck atom, shared by both roles.
fn grinding_sumcheck_site(
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    invocation: u32,
    round: u32,
    role: SumcheckRole,
) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_SUMCHECK,
        invocation: protocol.tag(),
        level,
        stage,
        round,
        group: invocation,
        detail: role as u32,
        ..ProtocolSiteId::default()
    }
}

/// Site of round `round`'s body in `invocation` of a grinding-backed
/// sumcheck: where a masked replay reads the round it lowers itself.
#[must_use]
pub fn grinding_sumcheck_body_site(
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    invocation: u32,
    round: u32,
) -> ProtocolSiteId {
    grinding_sumcheck_site(
        protocol,
        level,
        stage,
        invocation,
        round,
        SumcheckRole::RoundBody,
    )
}

/// Whether a sumcheck's rounds are witness-dependent (masked on a masked
/// channel) or public in both modes (`specs/whiteout-akita.md` §4.6).
#[cfg(feature = "prover")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoundPrivacy {
    Private,
    Public,
}

/// Standard-sumcheck channel borrowing a prover grinding context.
#[cfg(feature = "prover")]
pub struct GrindingSumcheckProver<'context, 'plan, F, E, C = ProverChannel> {
    grinding: &'context mut ProverGrinding<'plan, C>,
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    privacy: RoundPrivacy,
    _fields: PhantomData<fn() -> (F, E)>,
}

#[cfg(feature = "prover")]
impl<'context, 'plan, F, E, C> GrindingSumcheckProver<'context, 'plan, F, E, C> {
    /// Bind one sumcheck invocation to its scheduled grinding-site coordinates.
    #[must_use]
    pub fn new(
        grinding: &'context mut ProverGrinding<'plan, C>,
        protocol: akita_params::SumcheckProtocol,
        level: u32,
        stage: u32,
    ) -> Self {
        Self {
            grinding,
            protocol,
            level,
            stage,
            privacy: RoundPrivacy::Private,
            _fields: PhantomData,
        }
    }

    /// As [`new`](Self::new), for a sumcheck whose rounds are public in both
    /// modes: a masked channel sends them unmasked.
    #[must_use]
    pub fn new_public(
        grinding: &'context mut ProverGrinding<'plan, C>,
        protocol: akita_params::SumcheckProtocol,
        level: u32,
        stage: u32,
    ) -> Self {
        Self {
            privacy: RoundPrivacy::Public,
            ..Self::new(grinding, protocol, level, stage)
        }
    }
}

#[cfg(feature = "prover")]
impl<F, E, C> SumcheckProverChannel<E> for GrindingSumcheckProver<'_, '_, F, E, C>
where
    F: Field + CanonicalEncoding,
    E: ChallengeCodec + ExtField<F>,
    C: ProverGrindingChannel + PrivateProofChannel<F>,
{
    fn state_mut(&mut self) -> &mut ProverChannel {
        self.grinding.state.clear_state()
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        grinding_sumcheck_site(
            self.protocol,
            self.level,
            self.stage,
            invocation,
            round,
            role,
        )
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        if invocation != 0 {
            return Err(AkitaError::InvalidInput(
                "grinding sumcheck prover invocation is not zero".into(),
            ));
        }
        self.grinding
            .grinded_ext_challenge::<E>(GrindingSite::SumcheckRound {
                protocol: self.protocol,
                level: self.level,
                stage: self.stage,
                round,
            })
    }

    fn bind_input_claim<B>(&mut self, site: ProtocolSiteId, claim: &E) -> Result<(), AkitaError>
    where
        B: Field + CanonicalEncoding,
        E: ExtField<B>,
    {
        if self.privacy == RoundPrivacy::Private && self.grinding.state.masks() {
            return Ok(());
        }
        akita_transcript::public_extensions::<B, E, _>(
            self.grinding.state.clear_state(),
            site,
            std::slice::from_ref(claim),
        )
    }

    fn exchange_round_body<B>(
        &mut self,
        site: ProtocolSiteId,
        coefficients: &mut [E],
    ) -> Result<(), AkitaError>
    where
        B: Field + CanonicalEncoding,
        E: ExtField<B>,
    {
        match self.privacy {
            RoundPrivacy::Private => self
                .grinding
                .state
                .exchange_private_extension_group(site, coefficients)
                .map(|_| ()),
            RoundPrivacy::Public => akita_transcript::exchange_extension_group::<B, E, _>(
                self.grinding.state.clear_state(),
                site,
                coefficients,
            ),
        }
    }
}

/// Standard-sumcheck channel borrowing a verifier grinding context. Its
/// rounds are clear messages: a clear channel's private rounds, or a public
/// sumcheck's on any channel.
pub struct GrindingSumcheckVerifier<'context, F, E, G> {
    grinding: &'context mut G,
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    _fields: PhantomData<fn() -> (F, E)>,
}

impl<'context, F, E, G> GrindingSumcheckVerifier<'context, F, E, G> {
    /// Bind one sumcheck invocation to its scheduled grinding-site coordinates.
    #[must_use]
    pub fn new(
        grinding: &'context mut G,
        protocol: akita_params::SumcheckProtocol,
        level: u32,
        stage: u32,
    ) -> Self {
        Self {
            grinding,
            protocol,
            level,
            stage,
            _fields: PhantomData,
        }
    }
}

impl<'proof, F, E, G> SumcheckVerifierChannel<'proof, E> for GrindingSumcheckVerifier<'_, F, E, G>
where
    E: ChallengeCodec,
    G: GrindingReplay<State: VerifierGrindingChannel<'proof>>,
{
    fn state_mut(&mut self) -> &mut VerifierChannel<'proof> {
        self.grinding.state_mut().clear_state()
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        grinding_sumcheck_site(
            self.protocol,
            self.level,
            self.stage,
            invocation,
            round,
            role,
        )
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        if invocation != 0 {
            self.grinding.poison();
            return Err(AkitaError::InvalidInput(
                "grinding sumcheck verifier invocation is not zero".into(),
            ));
        }
        self.grinding
            .grinded_ext_challenge::<E>(GrindingSite::SumcheckRound {
                protocol: self.protocol,
                level: self.level,
                stage: self.stage,
                round,
            })
    }
}

#[cfg(feature = "zk")]
#[path = "replay/counting.rs"]
mod counting;
#[cfg(feature = "zk")]
pub use counting::CountingGrinding;

#[cfg(all(test, feature = "prover"))]
#[path = "replay/tests.rs"]
mod tests;

#[cfg(feature = "labinius-trinomial")]
#[path = "replay/binary_columns.rs"]
mod binary_columns;
