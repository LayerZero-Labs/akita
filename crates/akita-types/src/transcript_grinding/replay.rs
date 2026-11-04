//! Inline Spongefish nonce replay for transcript grinding.

use akita_error::AkitaError;
use akita_params::{GrindingPlan, GrindingQueryKind, GrindingSite};
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
    nonce_encoded_len, prover_context, send_nonce, FoldPreview, FoldWork, ProverChannel,
    ProverTranscript,
use crate::FoldSite;
use jolt_transcript::{Channel, PadSlots, PrivateChannel};
};
use std::marker::PhantomData;

#[derive(Clone, Copy)]
struct GrindingPlanEntry {
    site: GrindingSite,
    grind_bits: u8,
    nonce_bits: u8,
    fold_search: Option<FoldResponseSearch>,
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
    /// masked levels' entries.
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
    if entry.site != site || site.kind() != kind {
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

/// One side of grinding-plan replay.

// T0-PORT(private): dev's masked-channel abstraction over the NARG transcript, kept verbatim;
// re-express over akita_transcript::ProofChannel so masked channels can drive the grinding replay.
/// A prover channel Akita runs its grinding plan over: the clear transcript,
/// or a masked channel over it (`specs/whiteout-akita.md` §4.5).
/// Witness-dependent messages go through its [`PrivateChannel`]. Beside it
/// the channel offers only the public proof-of-work messages, which depend on
/// public transcript state alone, and the clear-only byte messages of the
/// terminal.
pub trait ProverGrindingChannel: Channel + PrivateChannel<Error: Into<AkitaError>> {
    /// Finds and sends the proof-of-work nonce for `bits` (`ProverTranscript::grind`).
    ///
    /// # Errors
    ///
    /// As `ProverTranscript::grind`.
    fn grind(&mut self, bits: u8) -> Result<u32, TranscriptError>;

    /// Sends a fold-response counter (`ProverTranscript::send_nonce`).
    fn send_nonce(&mut self, nonce: u32);

    /// The length of the argument string so far.
    fn narg_len(&self) -> usize;

    /// Sends `bytes`, whose length the verifier knows: the terminal's field
    /// digests under `dev-protocol`.
    ///
    /// # Errors
    ///
    /// A masked channel, which never reaches the terminal, returns
    /// [`AkitaError::UnsupportedSchedule`].
    fn send_public_bytes(&mut self, bytes: &[u8]) -> Result<(), AkitaError>;

    /// Sends `bytes` with their length, at most `max_len`: the terminal's
    /// `z` payload.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] past `max_len`; a masked channel returns
    /// [`AkitaError::UnsupportedSchedule`].
    fn send_bounded_public_bytes(&mut self, bytes: &[u8], max_len: usize)
        -> Result<(), AkitaError>;
}

impl<H: Sponge> ProverGrindingChannel for ProverTranscript<H> {
    fn grind(&mut self, bits: u8) -> Result<u32, TranscriptError> {
        Self::grind(self, bits)
    }

    fn send_nonce(&mut self, nonce: u32) {
        Self::send_nonce(self, nonce);
    }

    fn narg_len(&self) -> usize {
        self.narg().len()
    }

    fn send_public_bytes(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.send_bytes(bytes);
        Ok(())
    }

    fn send_bounded_public_bytes(
        &mut self,
        bytes: &[u8],
        max_len: usize,
    ) -> Result<(), AkitaError> {
        self.send_bounded_bytes(bytes, max_len).map_err(|_| {
            AkitaError::InvalidInput("public byte message exceeds its scheduled bound".into())
        })
    }
}

/// The verifier's mirror of [`ProverGrindingChannel`]: the public
/// proof-of-work reads.
pub trait VerifierGrindingChannel: Channel + PrivateChannel<Error: Into<AkitaError>> {
    /// Reads and checks the proof-of-work nonce for `bits`
    /// (`VerifierTranscript::check_grind`).
    ///
    /// # Errors
    ///
    /// As `VerifierTranscript::check_grind`.
    fn check_grind(&mut self, bits: u8) -> Result<u32, TranscriptError>;

    /// Reads a fold-response counter (`VerifierTranscript::receive_nonce`).
    ///
    /// # Errors
    ///
    /// As `VerifierTranscript::receive_nonce`.
    fn receive_nonce(&mut self, nonce_bits: u8) -> Result<u32, TranscriptError>;

    /// The unread proof bytes.
    fn remaining(&self) -> usize;
}

impl<H: Sponge> VerifierGrindingChannel for VerifierTranscript<'_, H> {
    fn check_grind(&mut self, bits: u8) -> Result<u32, TranscriptError> {
        Self::check_grind(self, bits)
    }

    fn receive_nonce(&mut self, nonce_bits: u8) -> Result<u32, TranscriptError> {
        Self::receive_nonce(self, nonce_bits)
    }

    fn remaining(&self) -> usize {
        Self::remaining(self)
    }
}

/// One side of grinding-plan replay over a borrowed transcript.
///
/// Scheduled-query code written against this trait runs unchanged for the
/// prover and the verifier. A failed replay call may already have advanced the
/// transcript, so it leaves the replay unusable and `finish` fails even if the
/// caller drops the error.
pub trait GrindingReplay {
    // T0-PORT(private): dev's State was Channel + PrivateChannel so masked channels plug in; re-express on v8.
    /// The role-specific proof channel state.
    type State: ProofChannel;

    /// Borrow the state for ordinary protocol messages and challenges.
    ///
    /// Callers must propagate errors from operations on this borrow before
    /// `finish`; the verifier state also records them itself.
    fn state_mut(&mut self) -> &mut Self::State;

    /// Mark this replay as failed.
    fn poison(&mut self);

    /// Apply one scheduled proof-of-work query: the prover searches and emits
    /// the nonce, the verifier receives and checks it.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError`] when `site` is not the next plan entry, the
    /// search is exhausted, or the received nonce is invalid.
    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError>;

    /// Check a predicate on public values: values the verifier reads in the
    /// clear in every mode, such as Stage 3's setup-product reduction.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidProof`] if `holds` is false. The counting
    /// replay, which runs over arbitrary bytes and returns no verdict, checks
    /// nothing.
    fn check_public(&mut self, holds: bool) -> Result<(), AkitaError> {
        if holds {
            Ok(())
        } else {
            Err(AkitaError::InvalidProof)
        }
    }

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

// T0-PORT(private): dev made this generic over the channel C (clear or masked, specs/whiteout-akita.md §4.5); re-express on v8.
#[cfg(feature = "prover")]
impl GrindingReplay for ProverGrinding<'_> {
    type State = ProverChannel;

    fn state_mut(&mut self) -> &mut ProverChannel {
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

// T0-PORT(private): dev made this generic over the channel C (clear or masked, specs/whiteout-akita.md §4.5); re-express on v8.
impl<'proof> GrindingReplay for VerifierGrinding<'proof, '_> {
    type State = VerifierChannel<'proof>;

    fn state_mut(&mut self) -> &mut VerifierChannel<'proof> {
        &mut *self.state
    }

    fn poison(&mut self) {
        self.invalid = true;
        let _ = reject(self.state);
    }

    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = self.grind_query_inner(site, GrindingQueryKind::ProofOfWork);
        if result.is_err() {
            self.poison();
        }
        result
    }
}

// T0-PORT(private): dev made this generic over the channel C (clear or masked, specs/whiteout-akita.md §4.5); re-express on v8.
/// Prover-side state paired with exact grinding-plan progress.
///
/// The state is borrowed: it is Akita's own standalone channel or a caller's
/// channel that Akita's messages join.
#[cfg(feature = "prover")]
pub struct ProverGrinding<'plan> {
    state: &'plan mut ProverChannel,
    cursor: GrindingPlanCursor<'plan>,
    serialized_nonce_bytes: usize,
    invalid: bool,
}

#[cfg(feature = "prover")]
impl<'plan> ProverGrinding<'plan> {
    /// Attach a prover channel to its public grinding plan.
    #[must_use]
    pub fn new(state: &'plan mut ProverChannel, plan: &'plan GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            serialized_nonce_bytes: 0,
            invalid: false,
        // T0-PORT(private): dev's Fork-seeded fold-response search (begin_fold_response) and one-shot responses (W2) lived here; re-express on v8.
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
        let start = self.state.narg_string().len();
        prover_context(self.state, nonce_record);
        work(self.state, bits)?;
        prover_context(self.state, check_record);
        let nonce_bytes = self.state.narg_string().len() - start;
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
    /// fold-challenge message drawn after it.
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
        // T0-PORT(private): dev: a one-shot fold response (W2, zk levels) sends no nonce; re-express on v8.
        prover_context(self.state, fold_response_record(site, entry.nonce_bits));
        send_nonce(self.state, counter);
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(nonce_encoded_len(counter))
            .ok_or_else(|| {
                AkitaError::Internal("fold-response serialized nonce byte count overflow".into())
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
        self.proof_of_work(
            GrindingSite::FoldChallengeWork {
                level: site.level(),
            },
            GrindingQueryKind::FoldChallengeWork,
            |state, _bits| work.commit(state),
        )
    }

    /// Preview a fold-response candidate from the current public state
    /// without advancing either the live state or the grinding plan: absorb
    /// the candidate nonce, then run the fold's proof of work. Pass the
    /// accepted preview's [`FoldPreview::take_work`] to
    /// [`commit_fold_response`](Self::commit_fold_response).
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
        if entry.site != site {
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
        let work_site = GrindingSite::FoldChallengeWork {
            level: site.level(),
        };
        let work = self
            .cursor
            .peek_following()
            .filter(|work| work.site == work_site)
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "fold-response preview is not followed by the fold proof of work".into(),
                )
            })?;
        let bits = pow_bits(work).ok_or_else(|| {
            AkitaError::Internal("grinding entry nonce width differs from grind_bits + 7".into())
        })?;
        let mut preview = FoldPreview::new(self.state, counter);
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
    /// cut (`specs/whiteout-akita.md` §4.6).
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
        akita_transcript::finish_proof_ranges(self.state);
        Ok(())
    }
}

// T0-PORT(private): dev made this generic over the channel C (clear or masked); re-express on v8.
/// Verifier-side state paired with exact grinding-plan progress.
///
/// The state is borrowed: it is Akita's own standalone channel or a caller's
/// channel whose argument string continues with Akita's proof.
pub struct VerifierGrinding<'proof, 'plan> {
    state: &'plan mut VerifierChannel<'proof>,
    cursor: GrindingPlanCursor<'plan>,
    invalid: bool,
    /// `C`'s sponge and proof lifetime, which name its [`Fork`]s.
    _transcript: PhantomData<fn() -> &'proof H>,
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

impl<'proof, 'plan> VerifierGrinding<'proof, 'plan> {
    /// Attach a verifier channel to its public grinding plan.
    #[must_use]
    pub const fn new(state: &'plan mut VerifierChannel<'proof>, plan: &'plan GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            invalid: false,
            _transcript: PhantomData,
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
        verifier_context(self.state, nonce_record);
        self.state
            .check_grind(bits)
            .map_err(|_| AkitaError::InvalidProof)?;
        verifier_context(self.state, check_record);
        Ok(())
    }

    /// Receive the next scheduled fold-response nonce, then check the fold's
    /// proof of work.
    pub fn read_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let result = self.read_fold_response_inner(site);
        if result.is_err() {
            self.poison();
        }
        result
    }

    fn read_fold_response_inner(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::FoldResponse)?;
        // T0-PORT(private): dev: a one-shot fold response (W2) carries no nonce on the verifier side; re-express on v8.
        verifier_context(self.state, fold_response_record(site, entry.nonce_bits));
        let counter = receive_nonce(self.state)?;
        if !value_fits(counter, entry.nonce_bits) {
            return Err(AkitaError::InvalidProof);
        }
        self.grind_query_inner(
            GrindingSite::FoldChallengeWork {
                level: site.level(),
            },
            GrindingQueryKind::FoldChallengeWork,
        )?;
        Ok(counter)
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

    /// Require that replay consumed exactly the grinding plan of levels
    /// `0..=cut_level` without error.
    pub fn finish_through(self, cut_level: u32) -> Result<(), AkitaError> {
        if self.invalid || !self.cursor.is_finished_through(cut_level) {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
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
}

/// The verifier side of grinding replay: the fold-response reads and the
/// fold-challenge records the fold verifier needs beyond [`GrindingReplay`].
/// [`VerifierGrinding`] checks the proof; `CountingGrinding` only counts the
/// pads a masked replay takes.
pub trait FoldResponseReplay: GrindingReplay {
    /// The sponge of the forks the fold challenges are drawn from.
    type Sponge: Sponge;

    /// Squeezes the next scheduled fold-response fork seed, reads the
    /// counter, and returns the fork the fold challenges are drawn from.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError`] when `site` is not the next plan entry or the
    /// counter is invalid.
    fn read_fold_response(&mut self, site: GrindingSite) -> Result<Fork<Self::Sponge>, AkitaError>;

    /// Consumes the plan entries of one sparse fold group and logs its fork
    /// draws.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError`] when the group is not the next plan run.
    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
        draws: &FoldDrawRecord,
    ) -> Result<(), AkitaError>;
}

impl<'proof, H: Sponge, C: VerifierGrindingChannel<Sponge = H>> FoldResponseReplay
    for VerifierGrinding<'_, 'proof, H, C>
{
    type Sponge = H;

    fn read_fold_response(&mut self, site: GrindingSite) -> Result<Fork<H>, AkitaError> {
        Self::read_fold_response(self, site)
    }

    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
        draws: &FoldDrawRecord,
    ) -> Result<(), AkitaError> {
        Self::record_fold_challenges(self, level, group, coordinate_count, draws)
    }
}

/// Site of one grinding-backed sumcheck atom, shared by both roles.
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

/// Standard-sumcheck channel borrowing a prover grinding context.
#[cfg(feature = "prover")]
pub struct GrindingSumcheckProver<'context, 'plan, F, E> {
    grinding: &'context mut ProverGrinding<'plan>,
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    _fields: PhantomData<fn() -> (F, E)>,
}

#[cfg(feature = "prover")]
impl<'context, 'plan, F, E> GrindingSumcheckProver<'context, 'plan, F, E> {
    /// Bind one sumcheck invocation to its scheduled grinding-site coordinates.
    #[must_use]
    pub fn new(
        grinding: &'context mut ProverGrinding<'plan>,
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

#[cfg(feature = "prover")]
impl<F, E> SumcheckProverChannel<E> for GrindingSumcheckProver<'_, '_, F, E>
where
    E: ChallengeCodec,
{
    fn state_mut(&mut self) -> &mut ProverChannel {
        self.grinding.state_mut()
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        grinding_sumcheck_site(
            self.protocol,
            self.level,
            self.stage,
            site,
        ))
    }

    fn challenge(&mut self, site: SumcheckSite) -> Result<E, AkitaError> {
        if site.invocation != 0 {
            return Err(AkitaError::InvalidInput(
                "grinding sumcheck prover invocation is not zero".into(),
            ));
        }
        self.grinding
            .grinded_ext_challenge::<E>(GrindingSite::SumcheckRound {
                protocol: self.protocol,
                level: self.level,
                stage: self.stage,
                round: site.round,
            })
    }
}

// T0-PORT(private): dev's verifier sumcheck context was generic over any GrindingReplay (masked channel, CountingGrinding); re-express on v8.
/// Standard-sumcheck channel borrowing a verifier grinding context.
pub struct GrindingSumcheckVerifier<'context, 'proof, 'plan, F, E> {
    grinding: &'context mut VerifierGrinding<'proof, 'plan>,
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    _fields: PhantomData<fn() -> (F, E)>,
}

impl<'context, 'proof, 'plan, F, E> GrindingSumcheckVerifier<'context, 'proof, 'plan, F, E> {
    /// Bind one sumcheck invocation to its scheduled grinding-site coordinates.
    #[must_use]
    pub fn new(
        grinding: &'context mut VerifierGrinding<'proof, 'plan>,
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

// T0-PORT(private): dev used PrivateChannel here; re-express on v8.
impl<'proof, F, E> SumcheckVerifierChannel<'proof, E>
    for GrindingSumcheckVerifier<'_, 'proof, '_, F, E>
where
    E: ChallengeCodec,
{
    fn state_mut(&mut self) -> &mut VerifierChannel<'proof> {
        self.grinding.state_mut()
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        grinding_sumcheck_site(
            self.protocol,
            self.level,
            self.stage,
            site,
        ))
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
                round: site.round,
            })
    }
}

/// Sumcheck round context whose round messages are public in both modes:
/// every private exchange goes out as an ordinary public message, with no
/// pads, and the input claim is bound as a public value, so the verifier
/// folds and checks the rounds in the clear even over a masked channel.
///
/// For a sumcheck whose round polynomials are functions of public data alone,
/// such as Stage 3's setup product (`specs/whiteout-akita.md` §4.3). Over the
/// clear transcript, whose private exchanges already are public messages, it
/// sends exactly the bytes [`GrindingSumcheckProver`] and
/// [`GrindingSumcheckVerifier`] do.
pub struct PublicGrindingSumcheck<'context, F, E, G> {
    grinding: &'context mut G,
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    _fields: PhantomData<fn() -> (F, E)>,
}

impl<'context, F, E, G> PublicGrindingSumcheck<'context, F, E, G> {
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

    fn sumcheck_site(&self, site: SumcheckSite) -> SiteId {
        grinding_sumcheck_site(self.protocol, self.level, self.stage, site)
    }
}

impl<F, E, G> PublicGrindingSumcheck<'_, F, E, G>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    G: GrindingReplay,
{
    fn draw(&mut self, site: SumcheckSite) -> Result<E, AkitaError> {
        if site.invocation != 0 {
            return Err(AkitaError::InvalidInput(
                "public grinding sumcheck invocation is not zero".into(),
            ));
        }
        self.grinding
            .grinded_ext_challenge::<F, E>(GrindingSite::SumcheckRound {
                protocol: self.protocol,
                level: self.level,
                stage: self.stage,
                round: site.round,
            })
    }
}

// T0-PORT(private): dev's Channel/PrivateChannel/round-context impls for PublicGrindingSumcheck
// (A42: rounds public in both modes) were dropped with the old driver; re-express on v8.
#[cfg(feature = "zk")]
#[path = "replay/counting.rs"]
mod counting;
#[cfg(feature = "zk")]
pub use counting::CountingGrinding;

#[cfg(all(test, feature = "prover"))]
#[path = "replay/tests.rs"]
mod tests;
