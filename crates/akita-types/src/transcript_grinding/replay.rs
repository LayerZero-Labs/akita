//! Inline nonce replay for transcript grinding.

use akita_error::AkitaError;
use akita_params::transcript_site::{ProtocolSiteId, SITE_FAMILY_SUMCHECK};
use akita_params::{GrindingPlan, GrindingQueryKind, GrindingSite};
use akita_sumcheck::{SumcheckProverChannel, SumcheckRole, SumcheckVerifierChannel};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use jolt_transcript::{
    Channel, Fork, ProverTranscript, SiteId, Sponge, TranscriptError, VerifierTranscript,
    FORK_SEED_LEN,
};
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

/// Diagnostic site of a proof-of-work nonce and its predicate.
fn grinding_site(site: GrindingSite, grind_bits: u8, nonce_bits: u8) -> SiteId {
    site.site_id(u32::from(grind_bits) | (u32::from(nonce_bits) << 8))
        .into()
}

/// Diagnostic site of a fold-response nonce.
fn fold_response_site(site: GrindingSite, nonce_bits: u8) -> SiteId {
    site.site_id(u32::from(nonce_bits)).into()
}

/// One side of grinding-plan replay over a borrowed transcript.
///
/// Scheduled-query code written against this trait runs unchanged for the
/// prover and the verifier. A failed replay call may already have advanced the
/// transcript, so it leaves the replay unusable and `finish` fails even if the
/// caller drops the error.
pub trait GrindingReplay {
    /// The role-specific transcript.
    type State: Channel;

    /// Borrow the transcript for ordinary protocol messages and challenges.
    ///
    /// Callers must propagate errors from operations on this borrow before
    /// `finish`; the verifier transcript also records them itself.
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

    /// Apply scheduled work and draw the site's extension-field challenge.
    ///
    /// # Errors
    ///
    /// Propagates [`GrindingReplay::grind_query`] failures.
    fn grinded_ext_challenge<F, E>(&mut self, site: GrindingSite) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        self.grind_query(site)?;
        let state = self.state_mut();
        state.site(site.site_id(u32::default()).into());
        Ok(state.challenge())
    }

    /// Apply one scheduled grinding query and draw `count` independent
    /// extension challenges protected by that query.
    ///
    /// Some Akita query sites protect a vector of independent field draws with
    /// one proof-of-work nonce; each draw gets a distinct diagnostic site.
    ///
    /// # Errors
    ///
    /// Propagates [`GrindingReplay::grind_query`] failures and rejects counts
    /// that cannot be allocated or indexed.
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
            let Ok(group) = u32::try_from(index) else {
                self.poison();
                return Err(AkitaError::InvalidProof);
            };
            let challenge_site = ProtocolSiteId {
                group,
                ..site.site_id(u32::default())
            };
            let state = self.state_mut();
            state.site(challenge_site.into());
            challenges.push(state.challenge());
        }
        Ok(challenges)
    }
}

impl<H: Sponge> GrindingReplay for ProverGrinding<'_, H> {
    type State = ProverTranscript<H>;

    fn state_mut(&mut self) -> &mut ProverTranscript<H> {
        self.state
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

impl<'proof, H: Sponge> GrindingReplay for VerifierGrinding<'_, 'proof, H> {
    type State = VerifierTranscript<'proof, H>;

    fn state_mut(&mut self) -> &mut VerifierTranscript<'proof, H> {
        self.state
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

/// Prover transcript paired with exact grinding-plan progress.
pub struct ProverGrinding<'a, H: Sponge> {
    state: &'a mut ProverTranscript<H>,
    cursor: GrindingPlanCursor<'a>,
    serialized_nonce_bytes: usize,
    invalid: bool,
    /// The fork seed of the fold-response search in progress, squeezed by
    /// [`begin_fold_response`](Self::begin_fold_response).
    fold_seed: Option<(GrindingSite, [u8; FORK_SEED_LEN])>,
}

impl<'a, H: Sponge> ProverGrinding<'a, H> {
    /// Attach the caller's prover transcript to its public grinding plan.
    #[must_use]
    pub fn new(state: &'a mut ProverTranscript<H>, plan: &'a GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            serialized_nonce_bytes: 0,
            invalid: false,
            fold_seed: None,
        }
    }

    fn grind_query_inner(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::ProofOfWork)?;
        if entry.grind_bits == 0 {
            return (entry.nonce_bits == 0).then_some(()).ok_or_else(|| {
                AkitaError::Internal("zero-bit grinding entry has nonzero nonce bits".into())
            });
        }
        self.state
            .site(grinding_site(site, entry.grind_bits, entry.nonce_bits));
        let start = self.state.narg().len();
        self.state
            .grind(entry.grind_bits)
            .map_err(|error| match error {
                TranscriptError::GrindingExhausted => {
                    AkitaError::InvalidInput("transcript grinding exhausted".into())
                }
                TranscriptError::UnsupportedGrinding => AkitaError::InvalidSetup(
                    "scheduled grinding bits exceed the transcript maximum".into(),
                ),
                other => AkitaError::Internal(format!("prover grinding failed: {other}")),
            })?;
        self.record_nonce_bytes(start)?;
        Ok(())
    }

    fn record_nonce_bytes(&mut self, start: usize) -> Result<usize, AkitaError> {
        let len = self.state.narg().len() - start;
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(len)
            .ok_or_else(|| AkitaError::Internal("serialized nonce byte count overflow".into()))?;
        Ok(len)
    }

    /// Starts the scheduled fold-response search at `site`: squeezes the
    /// fork seed every candidate counter is tried under.
    pub fn begin_fold_response(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = self.begin_fold_response_inner(site);
        self.invalid |= result.is_err();
        result
    }

    fn begin_fold_response_inner(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let entry = self.cursor.peek().ok_or_else(|| {
            AkitaError::InvalidInput("fold-response search has no remaining plan entry".into())
        })?;
        if entry.site != site || site.kind() != GrindingQueryKind::FoldResponse {
            return Err(AkitaError::InvalidInput(
                "fold-response search site differs from the next plan entry".into(),
            ));
        }
        if self.fold_seed.is_some() {
            return Err(AkitaError::InvalidInput(
                "a fold-response search is already in progress".into(),
            ));
        }
        self.state.site(fold_response_site(site, entry.nonce_bits));
        self.fold_seed = Some((site, self.state.challenge_bytes()));
        Ok(())
    }

    /// The fork a candidate `counter`'s fold challenges are drawn from, for the
    /// search [`begin_fold_response`](Self::begin_fold_response) started.
    pub fn fold_response_fork(
        &self,
        site: GrindingSite,
        counter: u32,
    ) -> Result<Fork<H>, AkitaError> {
        let entry = self.cursor.peek().ok_or_else(|| {
            AkitaError::InvalidInput("fold-response candidate has no remaining plan entry".into())
        })?;
        match self.fold_seed {
            Some((seed_site, seed)) if seed_site == site && entry.site == site => {
                if value_fits(counter, entry.nonce_bits) {
                    Ok(Fork::new(&seed, counter))
                } else {
                    Err(AkitaError::InvalidInput(
                        "fold-response candidate exceeds its scheduled bit width".into(),
                    ))
                }
            }
            Some(_) | None => Err(AkitaError::InvalidInput(
                "fold-response candidate has no search in progress at its site".into(),
            )),
        }
    }

    /// Sends the accepted fold-response `counter` as a nonce message and
    /// returns its fork, which the accepted fold challenges come from.
    pub fn commit_fold_response(
        &mut self,
        site: GrindingSite,
        counter: u32,
    ) -> Result<Fork<H>, AkitaError> {
        let result = self.commit_fold_response_inner(site, counter);
        self.invalid |= result.is_err();
        result
    }

    fn commit_fold_response_inner(
        &mut self,
        site: GrindingSite,
        counter: u32,
    ) -> Result<Fork<H>, AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::FoldResponse)?;
        let Some((seed_site, seed)) = self.fold_seed.take() else {
            return Err(AkitaError::InvalidInput(
                "fold-response commit has no search in progress".into(),
            ));
        };
        if seed_site != site {
            return Err(AkitaError::InvalidInput(
                "fold-response commit site differs from its search".into(),
            ));
        }
        if !value_fits(counter, entry.nonce_bits) {
            return Err(AkitaError::InvalidInput(
                "fold-response nonce exceeds its scheduled bit width".into(),
            ));
        }
        self.state.site(fold_response_site(site, entry.nonce_bits));
        let start = self.state.narg().len();
        self.state.send_nonce(counter);
        let encoded_bytes = self.record_nonce_bytes(start)?;
        tracing::info!(
            role = "prover",
            level = site.level(),
            accepted_nonce = counter,
            rejected_attempts = counter,
            attempts = u64::from(counter) + 1,
            encoded_bytes,
            "native fold response nonce"
        );
        Ok(Fork::new(&seed, counter))
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

    /// Require that replay consumed the complete grinding plan without error.
    ///
    /// The proof bytes stay in the caller's transcript.
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
        Ok(())
    }
}

/// Verifier transcript paired with exact grinding-plan progress.
pub struct VerifierGrinding<'a, 'proof, H: Sponge> {
    state: &'a mut VerifierTranscript<'proof, H>,
    cursor: GrindingPlanCursor<'a>,
    serialized_nonce_bytes: usize,
    invalid: bool,
}

impl<'a, 'proof, H: Sponge> VerifierGrinding<'a, 'proof, H> {
    /// Attach the caller's verifier transcript to its public grinding plan.
    #[must_use]
    pub fn new(state: &'a mut VerifierTranscript<'proof, H>, plan: &'a GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            serialized_nonce_bytes: 0,
            invalid: false,
        }
    }

    fn grind_query_inner(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::ProofOfWork)?;
        if entry.grind_bits == 0 {
            return (entry.nonce_bits == 0)
                .then_some(())
                .ok_or(AkitaError::InvalidProof);
        }
        self.state
            .site(grinding_site(site, entry.grind_bits, entry.nonce_bits));
        let remaining = self.state.remaining();
        self.state.check_grind(entry.grind_bits)?;
        self.record_nonce_bytes(remaining)
    }

    fn record_nonce_bytes(&mut self, remaining_before: usize) -> Result<(), AkitaError> {
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(
                remaining_before
                    .checked_sub(self.state.remaining())
                    .ok_or(AkitaError::InvalidProof)?,
            )
            .ok_or(AkitaError::InvalidProof)?;
        Ok(())
    }

    /// Squeezes the next scheduled fold-response fork seed, receives the
    /// counter, and returns the fork the fold challenges are drawn from.
    pub fn read_fold_response(&mut self, site: GrindingSite) -> Result<Fork<H>, AkitaError> {
        let result = self.read_fold_response_inner(site);
        if result.is_err() {
            self.poison();
        }
        result
    }

    fn read_fold_response_inner(&mut self, site: GrindingSite) -> Result<Fork<H>, AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::FoldResponse)?;
        self.state.site(fold_response_site(site, entry.nonce_bits));
        let seed: [u8; FORK_SEED_LEN] = self.state.challenge_bytes();
        let remaining = self.state.remaining();
        let counter = self.state.receive_nonce(entry.nonce_bits)?;
        self.record_nonce_bytes(remaining)?;
        Ok(Fork::new(&seed, counter))
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

    /// Require that replay consumed the complete grinding plan without error.
    ///
    /// Proof exhaustion is the caller's check: the outermost verifier finishes
    /// the transcript.
    pub fn finish(self) -> Result<(), AkitaError> {
        if self.invalid || !self.cursor.is_finished() {
            return Err(AkitaError::InvalidProof);
        }
        tracing::info!(
            role = "verifier",
            native_nonce_bytes_actual = self.serialized_nonce_bytes,
            "native proof nonce bytes"
        );
        Ok(())
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
) -> SiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_SUMCHECK,
        invocation: protocol.tag(),
        level,
        stage,
        round,
        group: invocation,
        detail: role as u32,
    }
    .into()
}

/// Standard-sumcheck channel borrowing a prover grinding context.
pub struct GrindingSumcheckProver<'context, 'a, F, E, H: Sponge> {
    grinding: &'context mut ProverGrinding<'a, H>,
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    _fields: PhantomData<fn() -> (F, E)>,
}

impl<'context, 'a, F, E, H: Sponge> GrindingSumcheckProver<'context, 'a, F, E, H> {
    /// Bind one sumcheck invocation to its scheduled grinding-site coordinates.
    #[must_use]
    pub fn new(
        grinding: &'context mut ProverGrinding<'a, H>,
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

impl<F, E, H> SumcheckProverChannel<E> for GrindingSumcheckProver<'_, '_, F, E, H>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    H: Sponge,
{
    type Sponge = H;

    fn state_mut(&mut self) -> &mut ProverTranscript<H> {
        self.grinding.state_mut()
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> SiteId {
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
pub struct GrindingSumcheckVerifier<'context, 'a, 'proof, F, E, H: Sponge> {
    grinding: &'context mut VerifierGrinding<'a, 'proof, H>,
    protocol: akita_params::SumcheckProtocol,
    level: u32,
    stage: u32,
    _fields: PhantomData<fn() -> (F, E)>,
}

impl<'context, 'a, 'proof, F, E, H: Sponge>
    GrindingSumcheckVerifier<'context, 'a, 'proof, F, E, H>
{
    /// Bind one sumcheck invocation to its scheduled grinding-site coordinates.
    #[must_use]
    pub fn new(
        grinding: &'context mut VerifierGrinding<'a, 'proof, H>,
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

impl<'proof, F, E, H> SumcheckVerifierChannel<'proof, E>
    for GrindingSumcheckVerifier<'_, '_, 'proof, F, E, H>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    H: Sponge,
{
    type Sponge = H;

    fn state_mut(&mut self) -> &mut VerifierTranscript<'proof, H> {
        self.grinding.state_mut()
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> SiteId {
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
                "grinding sumcheck verifier invocation is not zero".into(),
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

#[cfg(test)]
#[path = "replay/tests.rs"]
mod tests;
