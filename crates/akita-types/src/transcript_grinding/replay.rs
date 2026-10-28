//! Inline Spongefish nonce replay for transcript grinding.

use akita_error::AkitaError;
use akita_params::{GrindingPlan, GrindingQueryKind, GrindingSite};
#[cfg(feature = "prover")]
use akita_sumcheck::SumcheckProverChannel;
use akita_sumcheck::{SumcheckRole, SumcheckVerifierChannel};
use akita_transcript::{
    ext_challenges, grinding_check_bytes, nonce_encoded_len, nonce_max_bytes, reject,
    verifier_context, ChallengeCodec, NonceAtom, ProofChannel, ProtocolContextRecord,
    ProtocolMessageKind, ProtocolSiteId, VerifierChannel, VerifierTranscript,
    GRINDING_NONCE_SLACK_BITS, SITE_FAMILY_SUMCHECK,
};
#[cfg(feature = "prover")]
use akita_transcript::{prover_context, FoldPreview, ProverChannel, ProverTranscript};
use std::marker::PhantomData;

#[derive(Clone, Copy)]
struct GrindingPlanEntry {
    site: GrindingSite,
    grind_bits: u8,
    nonce_bits: u8,
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
            u64::from(grinding_check_bytes(grind_bits)),
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
impl GrindingReplay for ProverGrinding<'_> {
    type State = ProverChannel;

    fn state_mut(&mut self) -> &mut ProverChannel {
        &mut self.state
    }

    fn poison(&mut self) {
        self.invalid = true;
    }

    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = self.grind_query_inner(site);
        if result.is_err() {
            self.poison();
        }
        result
    }
}

impl<'proof> GrindingReplay for VerifierGrinding<'proof, '_> {
    type State = VerifierChannel<'proof>;

    fn state_mut(&mut self) -> &mut VerifierChannel<'proof> {
        &mut self.state
    }

    fn poison(&mut self) {
        self.invalid = true;
        let _ = reject(&mut self.state);
    }

    fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = self.grind_query_inner(site);
        if result.is_err() {
            self.poison();
        }
        result
    }
}

/// Prover-side state paired with exact grinding-plan progress.
#[cfg(feature = "prover")]
pub struct ProverGrinding<'plan> {
    state: ProverChannel,
    cursor: GrindingPlanCursor<'plan>,
    serialized_nonce_bytes: usize,
    invalid: bool,
}

#[cfg(feature = "prover")]
impl<'plan> ProverGrinding<'plan> {
    /// Attach a prover channel to its public grinding plan.
    #[must_use]
    pub fn new(state: ProverChannel, plan: &'plan GrindingPlan) -> Self {
        #[cfg(feature = "logging-transcript")]
        akita_transcript::clear_thread_events();
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            serialized_nonce_bytes: 0,
            invalid: false,
        }
    }

    fn grind_query_inner(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::ProofOfWork)?;
        let bits = pow_bits(entry).ok_or_else(|| {
            AkitaError::Internal("grinding entry nonce width differs from grind_bits + 7".into())
        })?;
        if bits == 0 {
            return Ok(());
        }
        let (nonce_record, check_record) =
            grinding_records(site, entry.grind_bits, entry.nonce_bits);
        let start = self.state.narg_string().len();
        prover_context(&mut self.state, nonce_record);
        self.state.grind(bits).map_err(|_| {
            AkitaError::Internal("transcript proof of work failed to find a nonce".into())
        })?;
        prover_context(&mut self.state, check_record);
        let nonce_bytes = self.state.narg_string().len() - start;
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(nonce_bytes)
            .ok_or_else(|| {
                AkitaError::Internal("proof-of-work serialized nonce byte count overflow".into())
            })?;
        Ok(())
    }

    /// Emit the next scheduled fold-response nonce as a canonical message.
    pub fn commit_fold_response(
        &mut self,
        site: GrindingSite,
        counter: u32,
    ) -> Result<(), AkitaError> {
        let result = self.commit_fold_response_inner(site, counter);
        self.invalid |= result.is_err();
        result
    }

    fn commit_fold_response_inner(
        &mut self,
        site: GrindingSite,
        counter: u32,
    ) -> Result<(), AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::FoldResponse)?;
        if !value_fits(counter, entry.nonce_bits) {
            return Err(AkitaError::InvalidInput(
                "fold-response nonce exceeds its scheduled bit width".into(),
            ));
        }
        prover_context(
            &mut self.state,
            fold_response_record(site, entry.nonce_bits),
        );
        self.state.prover_message(&NonceAtom::new(counter));
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
        Ok(())
    }

    /// Preview a fold-response candidate from the current public state
    /// without advancing either the live state or the grinding plan.
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
        Ok(FoldPreview::new(&self.state, counter))
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

    /// Finish plan replay and return the authoritative Spongefish argument.
    pub fn finish(self) -> Result<Vec<u8>, AkitaError> {
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
        akita_transcript::finish_proof_ranges(&self.state);
        Ok(self.state.narg_string().to_vec())
    }
}

/// Verifier-side state paired with exact grinding-plan progress.
pub struct VerifierGrinding<'proof, 'plan> {
    state: VerifierChannel<'proof>,
    cursor: GrindingPlanCursor<'plan>,
    serialized_nonce_bytes: usize,
    invalid: bool,
}

/// Evidence that replay consumed the complete grinding plan and proof.
///
/// The private field makes this value constructible only by the consuming
/// verifier finish boundary.
#[derive(Debug)]
pub struct ProofAcceptance {
    _private: (),
}

impl<'proof, 'plan> VerifierGrinding<'proof, 'plan> {
    /// Attach a verifier channel to its public grinding plan.
    #[must_use]
    pub const fn new(state: VerifierChannel<'proof>, plan: &'plan GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            serialized_nonce_bytes: 0,
            invalid: false,
        }
    }

    fn grind_query_inner(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::ProofOfWork)?;
        let bits = pow_bits(entry).ok_or(AkitaError::InvalidProof)?;
        if bits == 0 {
            return Ok(());
        }
        let (nonce_record, check_record) =
            grinding_records(site, entry.grind_bits, entry.nonce_bits);
        verifier_context(&mut self.state, nonce_record);
        self.state
            .check_grind(bits)
            .map_err(|_| AkitaError::InvalidProof)?;
        verifier_context(&mut self.state, check_record);
        Ok(())
    }

    /// Receive the next scheduled fold-response nonce.
    pub fn read_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let result = self.read_fold_response_inner(site);
        if result.is_err() {
            self.poison();
        }
        result
    }

    fn read_fold_response_inner(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::FoldResponse)?;
        verifier_context(
            &mut self.state,
            fold_response_record(site, entry.nonce_bits),
        );
        let counter = self
            .state
            .prover_message::<NonceAtom>()
            .map(NonceAtom::into_inner)
            .map_err(|_| AkitaError::InvalidProof)?;
        if !value_fits(counter, entry.nonce_bits) {
            return Err(AkitaError::InvalidProof);
        }
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(nonce_encoded_len(counter))
            .ok_or(AkitaError::InvalidProof)?;
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

    /// Require grinding-plan completion and consume the verifier for EOF.
    pub fn finish(self) -> Result<ProofAcceptance, AkitaError> {
        if self.invalid || !self.cursor.is_finished() {
            return Err(AkitaError::InvalidProof);
        }
        tracing::info!(
            role = "verifier",
            native_nonce_bytes_actual = self.serialized_nonce_bytes,
            "native proof nonce bytes"
        );
        self.state
            .check_eof()
            .map_err(|_| AkitaError::InvalidProof)?;
        Ok(ProofAcceptance { _private: () })
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
}

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

#[cfg(all(test, feature = "prover"))]
#[path = "replay/tests.rs"]
mod tests;
