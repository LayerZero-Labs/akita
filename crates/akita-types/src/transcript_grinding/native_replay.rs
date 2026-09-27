//! Inline native Spongefish nonce replay for transcript grinding.

use super::{GrindingPlan, GrindingQueryKind, GrindingSite};
use akita_error::AkitaError;
use akita_sumcheck::{
    NativeSumcheckProverChannel, NativeSumcheckRole, NativeSumcheckVerifierChannel,
};
use akita_transcript::{
    commit_native_grinding_nonce, grinding_predicate_accepts, native_nonce_encoded_len,
    native_nonce_max_bytes, native_prover_ext_challenge, native_verifier_ext_challenge,
    prover_context, receive_native_grinding_nonce, search_native_grinding_nonce, verifier_context,
    NativeFoldPreview, NativeNonce, NativeProverState, NativeVerifierState, ProtocolContextRecord,
    ProtocolMessageKind, ProtocolSiteId, GRINDING_PREDICATE_LEN, SITE_FAMILY_SUMCHECK,
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
        let run = *self.plan.runs.get(self.run_index)?;
        let entry = GrindingPlanEntry {
            site: run.site,
            grind_bits: run.grind_bits,
            nonce_bits: run.nonce_bits,
        };
        self.run_offset += 1;
        if self.run_offset == run.multiplicity {
            self.run_index += 1;
            self.run_offset = 0;
        }
        Some(entry)
    }

    fn peek(&self) -> Option<GrindingPlanEntry> {
        let run = *self.plan.runs.get(self.run_index)?;
        Some(GrindingPlanEntry {
            site: run.site,
            grind_bits: run.grind_bits,
            nonce_bits: run.nonce_bits,
        })
    }

    fn consume_run(&mut self, site: GrindingSite, multiplicity: usize) -> Result<(), AkitaError> {
        let run = self
            .plan
            .runs
            .get(self.run_index)
            .ok_or(AkitaError::InvalidProof)?;
        if self.run_offset != 0
            || run.site != site
            || run.grind_bits != 0
            || run.nonce_bits != 0
            || usize::try_from(run.multiplicity).ok() != Some(multiplicity)
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
        self.run_index == self.plan.runs.len() && self.run_offset == 0
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
    let site_id = site.native_site_id(detail).to_bytes();
    (
        ProtocolContextRecord::new(
            site_id,
            ProtocolMessageKind::GrindingNonce as u32,
            1,
            native_nonce_max_bytes(nonce_bits) as u64,
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
        site.native_site_id(u32::from(nonce_bits)).to_bytes(),
        ProtocolMessageKind::FoldResponseNonce as u32,
        1,
        native_nonce_max_bytes(nonce_bits) as u64,
        0,
    )
}

/// Prover-side native state paired with exact grinding-plan progress.
pub struct NativeProverGrinding<'plan> {
    state: NativeProverState,
    cursor: GrindingPlanCursor<'plan>,
    serialized_nonce_bytes: usize,
    invalid: bool,
}

impl<'plan> NativeProverGrinding<'plan> {
    /// Attach a native prover state to its public grinding plan.
    #[must_use]
    pub fn new(state: NativeProverState, plan: &'plan GrindingPlan) -> Self {
        #[cfg(feature = "logging-transcript")]
        akita_transcript::clear_thread_events();
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            serialized_nonce_bytes: 0,
            invalid: false,
        }
    }

    /// Borrow the native state for ordinary protocol messages and challenges.
    ///
    /// Operations performed through this raw role-specific state are outside
    /// replay poisoning. Callers must propagate their errors before `finish`;
    /// the completion boundary certifies only plan exhaustion and proof EOF.
    pub fn state_mut(&mut self) -> &mut NativeProverState {
        &mut self.state
    }

    /// Search, emit, and verify one scheduled proof-of-work nonce.
    pub fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = self.grind_query_inner(site);
        self.invalid |= result.is_err();
        result
    }

    /// Apply scheduled work and draw the site's extension-field challenge.
    pub fn grinded_ext_challenge<F, E>(&mut self, site: GrindingSite) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        self.grind_query(site)?;
        let result =
            native_prover_ext_challenge(&mut self.state, site.native_site_id(u32::default()))
                .map_err(|_| AkitaError::InvalidProof);
        self.invalid |= result.is_err();
        result
    }

    /// Apply one scheduled grinding query and draw `count` independently
    /// context-bound extension challenges protected by that query.
    pub fn grinded_ext_challenges<F, E>(
        &mut self,
        site: GrindingSite,
        count: usize,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let result = self.grinded_ext_challenges_inner::<F, E>(site, count);
        self.invalid |= result.is_err();
        result
    }

    fn grinded_ext_challenges_inner<F, E>(
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
        challenges
            .try_reserve_exact(count)
            .map_err(|_| AkitaError::InvalidProof)?;
        for index in 0..count {
            let mut challenge_site = site.native_site_id(u32::default());
            challenge_site.group = u32::try_from(index).map_err(|_| AkitaError::InvalidProof)?;
            challenges.push(self.ext_challenge_at::<F, E>(challenge_site)?);
        }
        Ok(challenges)
    }

    /// Draw a context-bound extension challenge after its governing grinding
    /// query has already been consumed.
    ///
    /// Some Akita query sites protect a vector of independent field draws with
    /// one proof-of-work nonce. Callers must supply a distinct, public
    /// schedule-derived `site` for every draw.
    pub fn ext_challenge_at<F, E>(
        &mut self,
        site: akita_transcript::ProtocolSiteId,
    ) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let result = native_prover_ext_challenge(&mut self.state, site)
            .map_err(|_| AkitaError::InvalidProof);
        self.invalid |= result.is_err();
        result
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
        let (nonce, winning_predicate) =
            search_native_grinding_nonce(&self.state, entry.grind_bits, entry.nonce_bits)
                .ok_or_else(|| AkitaError::InvalidInput("transcript grinding exhausted".into()))?;
        let predicate =
            commit_native_grinding_nonce(&mut self.state, nonce_record, nonce, predicate_record);
        if winning_predicate != predicate || !grinding_predicate_accepts(&predicate, bits) {
            return Err(AkitaError::InvalidProof);
        }
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(native_nonce_encoded_len(nonce))
            .ok_or(AkitaError::InvalidProof)?;
        Ok(())
    }

    /// Emit the next scheduled fold-response nonce as a canonical native message.
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
            return Err(AkitaError::InvalidProof);
        }
        prover_context(
            &mut self.state,
            fold_response_record(site, entry.nonce_bits),
        );
        self.state.prover_message(&NativeNonce::new(counter));
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(native_nonce_encoded_len(counter))
            .ok_or(AkitaError::InvalidProof)?;
        tracing::info!(
            role = "prover",
            level = site.level(),
            accepted_nonce = counter,
            rejected_attempts = counter,
            attempts = u64::from(counter) + 1,
            encoded_bytes = native_nonce_encoded_len(counter),
            "native fold response nonce"
        );
        Ok(())
    }

    /// Preview a fold-response candidate from the current native public state
    /// without advancing either the live state or the grinding plan.
    pub fn preview_fold_response(
        &self,
        site: GrindingSite,
        counter: u32,
    ) -> Result<NativeFoldPreview, AkitaError> {
        let entry = self.cursor.peek().ok_or(AkitaError::InvalidProof)?;
        if entry.site != site
            || site.kind() != GrindingQueryKind::FoldResponse
            || !value_fits(counter, entry.nonce_bits)
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(NativeFoldPreview::new(&self.state, counter))
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
        self.invalid |= result.is_err();
        result
    }

    /// Finish plan replay and return the authoritative Spongefish argument.
    pub fn finish(self) -> Result<Vec<u8>, AkitaError> {
        if self.invalid || !self.cursor.is_finished() {
            return Err(AkitaError::InvalidProof);
        }
        tracing::info!(
            role = "prover",
            native_nonce_bytes_actual = self.serialized_nonce_bytes,
            "native proof nonce bytes"
        );
        #[cfg(feature = "logging-transcript")]
        akita_transcript::finish_native_proof_ranges(&self.state);
        Ok(self.state.narg_string().to_vec())
    }
}

/// Verifier-side native state paired with exact grinding-plan progress.
pub struct NativeVerifierGrinding<'proof, 'plan> {
    state: NativeVerifierState<'proof>,
    cursor: GrindingPlanCursor<'plan>,
    serialized_nonce_bytes: usize,
    invalid: bool,
}

/// Evidence that native replay consumed the complete grinding plan and proof.
///
/// The private field makes this value constructible only by the consuming
/// verifier finish boundary.
#[derive(Debug)]
pub struct NativeProofAcceptance {
    _private: (),
}

impl<'proof, 'plan> NativeVerifierGrinding<'proof, 'plan> {
    fn invalidate(&mut self) {
        self.invalid = true;
        self.state.invalidate();
    }

    /// Attach a native verifier state to its public grinding plan.
    #[must_use]
    pub const fn new(state: NativeVerifierState<'proof>, plan: &'plan GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            serialized_nonce_bytes: 0,
            invalid: false,
        }
    }

    /// Borrow the native state for ordinary protocol receipt and challenges.
    ///
    /// The Akita state owner records every decoding or bounded-receipt failure,
    /// including errors returned through this borrow. Callers must still
    /// propagate errors so algebraic verification stops at the failing step.
    pub fn state_mut(&mut self) -> &mut NativeVerifierState<'proof> {
        &mut self.state
    }

    /// Receive and validate one scheduled proof-of-work nonce.
    pub fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = self.grind_query_inner(site);
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    /// Verify scheduled work and draw the site's extension-field challenge.
    pub fn grinded_ext_challenge<F, E>(&mut self, site: GrindingSite) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        self.grind_query(site)?;
        let result =
            native_verifier_ext_challenge(&mut self.state, site.native_site_id(u32::default()))
                .map_err(|_| AkitaError::InvalidProof);
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    /// Verify one scheduled grinding query and draw `count` independently
    /// context-bound extension challenges protected by that query.
    pub fn grinded_ext_challenges<F, E>(
        &mut self,
        site: GrindingSite,
        count: usize,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let result = self.grinded_ext_challenges_inner::<F, E>(site, count);
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    fn grinded_ext_challenges_inner<F, E>(
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
        challenges
            .try_reserve_exact(count)
            .map_err(|_| AkitaError::InvalidProof)?;
        for index in 0..count {
            let mut challenge_site = site.native_site_id(u32::default());
            challenge_site.group = u32::try_from(index).map_err(|_| AkitaError::InvalidProof)?;
            challenges.push(self.ext_challenge_at::<F, E>(challenge_site)?);
        }
        Ok(challenges)
    }

    /// Draw a context-bound extension challenge after its governing grinding
    /// query has already been consumed.
    pub fn ext_challenge_at<F, E>(
        &mut self,
        site: akita_transcript::ProtocolSiteId,
    ) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let result = native_verifier_ext_challenge(&mut self.state, site)
            .map_err(|_| AkitaError::InvalidProof);
        if result.is_err() {
            self.invalidate();
        }
        result
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
            receive_native_grinding_nonce(&mut self.state, nonce_record, predicate_record)
                .map_err(|_| AkitaError::InvalidProof)?;
        if !value_fits(nonce, entry.nonce_bits) || !grinding_predicate_accepts(&predicate, bits) {
            return Err(AkitaError::InvalidProof);
        }
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(native_nonce_encoded_len(nonce))
            .ok_or(AkitaError::InvalidProof)?;
        Ok(())
    }

    /// Receive the next scheduled fold-response nonce.
    pub fn read_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let result = self.read_fold_response_inner(site);
        if result.is_err() {
            self.invalidate();
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
            .prover_message::<NativeNonce>()
            .map(NativeNonce::into_inner)
            .map_err(|_| AkitaError::InvalidProof)?;
        if !value_fits(counter, entry.nonce_bits) {
            return Err(AkitaError::InvalidProof);
        }
        self.serialized_nonce_bytes = self
            .serialized_nonce_bytes
            .checked_add(native_nonce_encoded_len(counter))
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
            self.invalidate();
        }
        result
    }

    /// Require grinding-plan completion and consume the verifier for EOF.
    pub fn finish(self) -> Result<NativeProofAcceptance, AkitaError> {
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
        Ok(NativeProofAcceptance { _private: () })
    }
}

/// Standard-sumcheck channel borrowing a native prover grinding context.
pub struct NativeGrindingSumcheckProver<'context, 'plan, F, E> {
    grinding: &'context mut NativeProverGrinding<'plan>,
    protocol: super::SumcheckProtocol,
    level: u32,
    stage: u32,
    _fields: PhantomData<fn() -> (F, E)>,
}

impl<'context, 'plan, F, E> NativeGrindingSumcheckProver<'context, 'plan, F, E> {
    /// Bind one sumcheck invocation to its scheduled grinding-site coordinates.
    #[must_use]
    pub fn new(
        grinding: &'context mut NativeProverGrinding<'plan>,
        protocol: super::SumcheckProtocol,
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

impl<F, E> NativeSumcheckProverChannel<E> for NativeGrindingSumcheckProver<'_, '_, F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn state_mut(&mut self) -> &mut NativeProverState {
        self.grinding.state_mut()
    }

    fn sumcheck_site(
        &self,
        invocation: u32,
        round: u32,
        role: NativeSumcheckRole,
    ) -> ProtocolSiteId {
        ProtocolSiteId {
            family: SITE_FAMILY_SUMCHECK,
            invocation: self.protocol.tag(),
            level: self.level,
            stage: self.stage,
            round,
            group: invocation,
            detail: role as u32,
            ..ProtocolSiteId::default()
        }
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

/// Standard-sumcheck channel borrowing a native verifier grinding context.
pub struct NativeGrindingSumcheckVerifier<'context, 'proof, 'plan, F, E> {
    grinding: &'context mut NativeVerifierGrinding<'proof, 'plan>,
    protocol: super::SumcheckProtocol,
    level: u32,
    stage: u32,
    _fields: PhantomData<fn() -> (F, E)>,
}

impl<'context, 'proof, 'plan, F, E> NativeGrindingSumcheckVerifier<'context, 'proof, 'plan, F, E> {
    /// Bind one sumcheck invocation to its scheduled grinding-site coordinates.
    #[must_use]
    pub fn new(
        grinding: &'context mut NativeVerifierGrinding<'proof, 'plan>,
        protocol: super::SumcheckProtocol,
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

impl<'proof, F, E> NativeSumcheckVerifierChannel<'proof, E>
    for NativeGrindingSumcheckVerifier<'_, 'proof, '_, F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn state_mut(&mut self) -> &mut NativeVerifierState<'proof> {
        self.grinding.state_mut()
    }

    fn sumcheck_site(
        &self,
        invocation: u32,
        round: u32,
        role: NativeSumcheckRole,
    ) -> ProtocolSiteId {
        ProtocolSiteId {
            family: SITE_FAMILY_SUMCHECK,
            invocation: self.protocol.tag(),
            level: self.level,
            stage: self.stage,
            round,
            group: invocation,
            detail: role as u32,
            ..ProtocolSiteId::default()
        }
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
mod tests {
    use super::*;
    use crate::GrindingRun;
    use akita_challenges::{
        FoldChallengeDrawDomain, FoldDraw, NativePreviewFoldDraw, NativeProverFoldDraw,
        NativeVerifierFoldDraw, SparseChallengeConfig,
    };
    use akita_transcript::{
        new_native_prover, new_native_verifier, preview_native_grinding_predicate,
    };
    use jolt_field::Prime128Offset275 as F;

    fn plan() -> GrindingPlan {
        GrindingPlan::new(
            vec![
                GrindingRun::proof_of_work(GrindingSite::EvaluationBatch { level: 0 }, 3, 128)
                    .unwrap(),
                GrindingRun::fold_response(0),
                GrindingRun::fold_challenge_group(0, 0, 2).unwrap(),
            ],
            128,
        )
        .unwrap()
    }

    #[test]
    fn native_grinding_roundtrip_uses_inline_canonical_nonces_and_eof() {
        let plan = plan();
        let state = new_native_prover(b"native-grinding", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        let prover_challenge = prover
            .grinded_ext_challenge::<F, F>(GrindingSite::EvaluationBatch { level: 0 })
            .unwrap();
        prover
            .commit_fold_response(GrindingSite::FoldResponse { level: 0 }, 7)
            .unwrap();
        prover.record_fold_challenges(0, 0, 2).unwrap();
        let proof = prover.finish().unwrap();

        // One PoW nonce and one response nonce; public context records occupy
        // no argument bytes.
        assert!(proof.len() <= plan.native_nonce_max_bytes());
        let state = new_native_verifier(b"native-grinding", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        let verifier_challenge = verifier
            .grinded_ext_challenge::<F, F>(GrindingSite::EvaluationBatch { level: 0 })
            .unwrap();
        assert_eq!(verifier_challenge, prover_challenge);
        assert_eq!(
            verifier
                .read_fold_response(GrindingSite::FoldResponse { level: 0 })
                .unwrap(),
            7
        );
        verifier.record_fold_challenges(0, 0, 2).unwrap();
        verifier.finish().unwrap();
    }

    #[test]
    fn verifier_accepts_a_valid_nonminimal_two_byte_pow_nonce() {
        let site = GrindingSite::EvaluationBatch { level: 0 };
        let plan = GrindingPlan::new(vec![GrindingRun::proof_of_work(site, 2, 128).unwrap()], 128)
            .unwrap();
        let mut state = new_native_prover(b"native-long-nonce", b"fixture").unwrap();
        let nonce = (128..=u8::MAX as u32)
            .find(|&candidate| {
                grinding_predicate_accepts(
                    &preview_native_grinding_predicate(&state, candidate),
                    NonZeroU8::new(1).unwrap(),
                )
            })
            .expect("the two-byte half of an 8-bit nonce domain must contain a winner");
        let (nonce_record, predicate_record) = grinding_records(site, 1, 8);
        let _ = commit_native_grinding_nonce(&mut state, nonce_record, nonce, predicate_record);
        let proof = state.narg_string().to_vec();
        assert_eq!(proof.len(), 2);

        let state = new_native_verifier(b"native-long-nonce", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        verifier.grinded_ext_challenge::<F, F>(site).unwrap();
        verifier.finish().unwrap();
    }

    #[test]
    fn one_native_grinding_query_can_protect_multiple_draws() {
        let plan = GrindingPlan::new(
            vec![GrindingRun::proof_of_work(
                GrindingSite::ExtensionOpeningPoint { level: 4 },
                2,
                128,
            )
            .unwrap()],
            128,
        )
        .unwrap();
        let state = new_native_prover(b"native-vector-grinding", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        let prover_challenges = prover
            .grinded_ext_challenges::<F, F>(GrindingSite::ExtensionOpeningPoint { level: 4 }, 3)
            .unwrap();
        let proof = prover.finish().unwrap();
        assert!(proof.len() <= plan.native_nonce_max_bytes());

        let state = new_native_verifier(b"native-vector-grinding", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        let verifier_challenges = verifier
            .grinded_ext_challenges::<F, F>(GrindingSite::ExtensionOpeningPoint { level: 4 }, 3)
            .unwrap();
        assert_eq!(verifier_challenges, prover_challenges);
        verifier.finish().unwrap();
    }

    #[test]
    fn native_grinding_rejects_out_of_range_response_and_incomplete_plan() {
        let plan = GrindingPlan::new(vec![GrindingRun::fold_response(0)], 128).unwrap();
        let state = new_native_prover(b"native-grinding", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        assert_eq!(
            prover.commit_fold_response(
                GrindingSite::FoldResponse { level: 0 },
                super::super::FOLD_RESPONSE_ATTEMPTS,
            ),
            Err(AkitaError::InvalidProof)
        );
        assert_eq!(prover.finish(), Err(AkitaError::InvalidProof));

        for (proof, expected) in [
            (&[0xff, 0x1f][..], Ok(4095)),
            (&[0x80, 0x20][..], Err(AkitaError::InvalidProof)),
        ] {
            let should_accept = expected.is_ok();
            let state = new_native_verifier(b"native-grinding", b"fixture", proof).unwrap();
            let mut verifier = NativeVerifierGrinding::new(state, &plan);
            assert_eq!(
                verifier.read_fold_response(GrindingSite::FoldResponse { level: 0 }),
                expected
            );
            if should_accept {
                verifier.finish().unwrap();
            } else {
                assert!(verifier.state_mut().verifier_message::<[u8; 32]>().is_err());
                assert!(matches!(verifier.finish(), Err(AkitaError::InvalidProof)));
            }
        }
    }

    #[test]
    fn replay_poison_survives_post_advance_allocation_failure() {
        let site = GrindingSite::ExtensionOpeningPoint { level: 9 };
        let plan = GrindingPlan::new(vec![GrindingRun::proof_of_work(site, 1, 128).unwrap()], 128)
            .unwrap();

        let state = new_native_prover(b"native-poison", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        assert_eq!(
            prover.grinded_ext_challenges::<F, F>(site, usize::MAX),
            Err(AkitaError::InvalidProof)
        );
        assert_eq!(prover.finish(), Err(AkitaError::InvalidProof));

        let state = new_native_verifier(b"native-poison", b"fixture", &[]).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        assert_eq!(
            verifier.grinded_ext_challenges::<F, F>(site, usize::MAX),
            Err(AkitaError::InvalidProof)
        );
        assert!(matches!(verifier.finish(), Err(AkitaError::InvalidProof)));
    }

    #[test]
    fn nonce_width_membership_is_total() {
        assert!(value_fits(0, 0));
        assert!(!value_fits(1, 0));
        assert!(value_fits((1 << 31) - 1, 31));
        assert!(!value_fits(1 << 31, 31));
        assert!(value_fits(u32::MAX, 32));
        assert!(!value_fits(0, 33));
        assert!(!value_fits(u32::MAX, u8::MAX));
    }

    #[test]
    fn native_fold_candidate_replays_all_groups_as_one_transaction() {
        let plan = GrindingPlan::new(
            vec![
                GrindingRun::fold_response(3),
                GrindingRun::fold_challenge_group(3, 0, 2).unwrap(),
                GrindingRun::fold_challenge_group(3, 1, 2).unwrap(),
            ],
            128,
        )
        .unwrap();
        let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let site = GrindingSite::FoldResponse { level: 3 };
        let nonce = 7;
        let state = new_native_prover(b"native-fold-transaction", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        let (preview_first, preview_second) = {
            let mut preview_state = prover.preview_fold_response(site, nonce).unwrap();
            let first = NativePreviewFoldDraw::new(&mut preview_state)
                .draw_folding_challenges_with_rejection(
                    FoldChallengeDrawDomain::EvaluationTrace,
                    64,
                    0,
                    2,
                    1,
                    &config,
                    None,
                )
                .unwrap();
            let second = NativePreviewFoldDraw::new(&mut preview_state)
                .draw_folding_challenges_with_rejection(
                    FoldChallengeDrawDomain::EvaluationTrace,
                    64,
                    1,
                    1,
                    2,
                    &config,
                    None,
                )
                .unwrap();
            (first, second)
        };
        prover.commit_fold_response(site, nonce).unwrap();
        let live_first = NativeProverFoldDraw::new(prover.state_mut(), 3, 0)
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                0,
                2,
                1,
                &config,
                None,
            )
            .unwrap();
        prover.record_fold_challenges(3, 0, 2).unwrap();
        let live_second = NativeProverFoldDraw::new(prover.state_mut(), 3, 1)
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                1,
                1,
                2,
                &config,
                None,
            )
            .unwrap();
        prover.record_fold_challenges(3, 1, 2).unwrap();
        assert_eq!(
            (preview_first, preview_second),
            (live_first.clone(), live_second.clone())
        );
        let proof = prover.finish().unwrap();

        let state = new_native_verifier(b"native-fold-transaction", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        assert_eq!(verifier.read_fold_response(site).unwrap(), nonce);
        let verified_first = NativeVerifierFoldDraw::new(verifier.state_mut(), 3, 0)
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                0,
                2,
                1,
                &config,
                None,
            )
            .unwrap();
        verifier.record_fold_challenges(3, 0, 2).unwrap();
        let verified_second = NativeVerifierFoldDraw::new(verifier.state_mut(), 3, 1)
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                1,
                1,
                2,
                &config,
                None,
            )
            .unwrap();
        verifier.record_fold_challenges(3, 1, 2).unwrap();
        assert_eq!((verified_first, verified_second), (live_first, live_second));
        verifier.finish().unwrap();
    }
}
