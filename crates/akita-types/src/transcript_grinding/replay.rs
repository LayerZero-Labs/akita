//! Inline Spongefish nonce replay for transcript grinding.

use akita_error::AkitaError;
use akita_params::{GrindingPlan, GrindingQueryKind, GrindingSite};
use akita_sumcheck::{SumcheckProverChannel, SumcheckRole, SumcheckVerifierChannel};
use akita_transcript::{
    commit_grinding_nonce, ext_challenge, grinding_predicate_accepts, nonce_encoded_len,
    nonce_max_bytes, prover_context, receive_grinding_nonce, search_grinding_nonce,
    verifier_context, FoldPreview, NonceAtom, ProofChannel, ProtocolContextRecord,
    ProtocolMessageKind, ProtocolSiteId, ProverChannel, VerifierChannel, GRINDING_PREDICATE_LEN,
    SITE_FAMILY_SUMCHECK,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::marker::PhantomData;
use std::num::NonZeroU8;

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

    fn peek(&self) -> Option<GrindingPlanEntry> {
        let run = *self.plan.runs().get(self.run_index)?;
        Some(GrindingPlanEntry {
            site: run.site(),
            grind_bits: run.grind_bits(),
            nonce_bits: run.nonce_bits(),
        })
    }

    fn consume_run(&mut self, site: GrindingSite, multiplicity: usize) -> Result<(), AkitaError> {
        let run = self
            .plan
            .runs()
            .get(self.run_index)
            .ok_or(AkitaError::InvalidProof)?;
        if self.run_offset != 0
            || run.site() != site
            || run.grind_bits() != 0
            || run.nonce_bits() != 0
            || usize::try_from(run.multiplicity()).ok() != Some(multiplicity)
        {
            return Err(AkitaError::InvalidProof);
        }
        self.run_index = self
            .run_index
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)?;
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
    let entry = cursor.next().ok_or(AkitaError::InvalidProof)?;
    if entry.site != site || site.kind() != kind {
        return Err(AkitaError::InvalidProof);
    }
    Ok(entry)
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
            ProtocolMessageKind::GrindingPredicate as u32,
            0,
            0,
            GRINDING_PREDICATE_LEN as u64,
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
/// prover and the verifier. Every failure poisons the owner so its `finish`
/// boundary rejects even if a caller drops the error.
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

    /// Draw a context-bound extension challenge after its governing grinding
    /// query has already been consumed.
    ///
    /// Some Akita query sites protect a vector of independent field draws with
    /// one proof-of-work nonce. Callers must supply a distinct, public
    /// schedule-derived `site` for every draw.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] when challenge sampling fails.
    fn ext_challenge_at<F, E>(&mut self, site: ProtocolSiteId) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let result = ext_challenge::<F, E, _>(self.state_mut(), site);
        if result.is_err() {
            self.poison();
        }
        result
    }

    /// Apply scheduled work and draw the site's extension-field challenge.
    ///
    /// # Errors
    ///
    /// Propagates [`GrindingReplay::grind_query`] and
    /// [`GrindingReplay::ext_challenge_at`] failures.
    fn grinded_ext_challenge<F, E>(&mut self, site: GrindingSite) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        self.grind_query(site)?;
        self.ext_challenge_at::<F, E>(site.site_id(u32::default()))
    }

    /// Apply one scheduled grinding query and draw `count` independently
    /// context-bound extension challenges protected by that query.
    ///
    /// # Errors
    ///
    /// Propagates [`GrindingReplay::grind_query`] and
    /// [`GrindingReplay::ext_challenge_at`] failures and rejects counts that
    /// cannot be allocated or indexed.
    fn grinded_ext_challenges<F, E>(
        &mut self,
        site: GrindingSite,
        count: usize,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        self.grind_query(site)?;
        let mut challenges = Vec::new();
        if challenges.try_reserve_exact(count).is_err() {
            self.poison();
            return Err(AkitaError::InvalidProof);
        }
        for index in 0..count {
            let mut challenge_site = site.site_id(u32::default());
            let Ok(group) = u32::try_from(index) else {
                self.poison();
                return Err(AkitaError::InvalidProof);
            };
            challenge_site.group = group;
            challenges.push(self.ext_challenge_at::<F, E>(challenge_site)?);
        }
        Ok(challenges)
    }
}

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
        self.state.invalidate();
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
pub struct ProverGrinding<'plan> {
    state: ProverChannel,
    cursor: GrindingPlanCursor<'plan>,
    serialized_nonce_bytes: usize,
    invalid: bool,
}

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
        let Some(bits) = NonZeroU8::new(entry.grind_bits) else {
            return (entry.nonce_bits == 0).then_some(()).ok_or_else(|| {
                AkitaError::Internal("zero-bit grinding entry has nonzero nonce bits".into())
            });
        };
        let (nonce_record, predicate_record) =
            grinding_records(site, entry.grind_bits, entry.nonce_bits);
        let (nonce, winning_predicate) =
            search_grinding_nonce(&self.state, entry.grind_bits, entry.nonce_bits)
                .ok_or_else(|| AkitaError::InvalidInput("transcript grinding exhausted".into()))?;
        let predicate =
            commit_grinding_nonce(&mut self.state, nonce_record, nonce, predicate_record);
        if winning_predicate != predicate {
            return Err(AkitaError::Internal(
                "committed grinding predicate differs from winning predicate".into(),
            ));
        }
        if !grinding_predicate_accepts(&predicate, bits) {
            return Err(AkitaError::Internal(
                "committed grinding predicate fails its bound".into(),
            ));
        }
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(nonce_encoded_len(nonce))
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
            AkitaError::Internal(
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
            return Err(AkitaError::Internal(
                "an earlier grinding step failed".into(),
            ));
        }
        if !self.cursor.is_finished() {
            return Err(AkitaError::Internal(
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
        let Some(bits) = NonZeroU8::new(entry.grind_bits) else {
            return (entry.nonce_bits == 0)
                .then_some(())
                .ok_or(AkitaError::InvalidProof);
        };
        let (nonce_record, predicate_record) =
            grinding_records(site, entry.grind_bits, entry.nonce_bits);
        let (nonce, predicate) =
            receive_grinding_nonce(&mut self.state, nonce_record, predicate_record)?;
        if !value_fits(nonce, entry.nonce_bits) || !grinding_predicate_accepts(&predicate, bits) {
            return Err(AkitaError::InvalidProof);
        }
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(nonce_encoded_len(nonce))
            .ok_or(AkitaError::InvalidProof)?;
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
            .map(NonceAtom::into_inner)?;
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
        self.state.check_eof()?;
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
pub struct GrindingSumcheckProver<'context, 'plan, F, E> {
    grinding: &'context mut ProverGrinding<'plan>,
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    _fields: PhantomData<fn() -> (F, E)>,
}

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

impl<F, E> SumcheckProverChannel<E> for GrindingSumcheckProver<'_, '_, F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
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
            .grinded_ext_challenge::<F, E>(GrindingSite::SumcheckRound {
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
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
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
            return Err(AkitaError::InvalidProof);
        }
        self.grinding
            .grinded_ext_challenge::<F, E>(GrindingSite::SumcheckRound {
                protocol: self.protocol,
                level: self.level,
                stage: self.stage,
                round,
            })
    }
}

#[cfg(test)]
#[path = "replay/tests.rs"]
mod tests;
