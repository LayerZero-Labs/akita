//! Inline native Spongefish nonce replay for transcript grinding.

use super::replay::{value_fits, GrindingPlanCursor, GrindingPlanEntry};
use super::{GrindingPlan, GrindingQueryKind, GrindingSite};
use akita_error::AkitaError;
use akita_sumcheck::{NativeSumcheckProverChannel, NativeSumcheckVerifierChannel};
use akita_transcript::{
    commit_native_grinding_nonce, grinding_predicate_accepts, native_prover_ext_challenge,
    native_verifier_ext_challenge, preview_native_grinding_predicate, prover_context,
    receive_native_grinding_nonce, search_native_grinding_nonce, verifier_context,
    NativeProverState, NativeVerifierState, ProtocolContextRecord, ProtocolMessageKind,
    GRINDING_PREDICATE_LEN,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::marker::PhantomData;
use std::num::NonZeroU8;

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
        ProtocolContextRecord::new(site_id, ProtocolMessageKind::GrindingNonce as u32, 1, 4, 0),
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
        4,
        0,
    )
}

/// Prover-side native state paired with exact grinding-plan progress.
pub struct NativeProverGrinding<'plan> {
    state: NativeProverState,
    cursor: GrindingPlanCursor<'plan>,
    invalid: bool,
}

impl<'plan> NativeProverGrinding<'plan> {
    /// Attach a native prover state to its public grinding plan.
    #[must_use]
    pub const fn new(state: NativeProverState, plan: &'plan GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            invalid: false,
        }
    }

    /// Borrow the native state for ordinary protocol messages and challenges.
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

    fn grind_query_inner(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let entry = next_entry(&mut self.cursor, site, GrindingQueryKind::ProofOfWork)?;
        let Some(bits) = NonZeroU8::new(entry.grind_bits) else {
            return (entry.nonce_bits == 0)
                .then_some(())
                .ok_or(AkitaError::InvalidProof);
        };
        let (nonce_record, predicate_record) =
            grinding_records(site, entry.grind_bits, entry.nonce_bits);
        let nonce = search_native_grinding_nonce(
            &self.state,
            entry.grind_bits,
            entry.nonce_bits,
            nonce_record,
            predicate_record,
        )
        .ok_or_else(|| AkitaError::InvalidInput("transcript grinding exhausted".into()))?;
        let preview =
            preview_native_grinding_predicate(&self.state, nonce_record, nonce, predicate_record);
        let predicate =
            commit_native_grinding_nonce(&mut self.state, nonce_record, nonce, predicate_record);
        if preview != predicate || !grinding_predicate_accepts(&predicate, bits) {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    /// Emit the next scheduled fold-response nonce as a native `u32` message.
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
        self.state.prover_message(&counter);
        Ok(())
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
        Ok(self.state.narg_string().to_vec())
    }
}

/// Verifier-side native state paired with exact grinding-plan progress.
pub struct NativeVerifierGrinding<'proof, 'plan> {
    state: NativeVerifierState<'proof>,
    cursor: GrindingPlanCursor<'plan>,
    invalid: bool,
}

impl<'proof, 'plan> NativeVerifierGrinding<'proof, 'plan> {
    /// Attach a native verifier state to its public grinding plan.
    #[must_use]
    pub const fn new(state: NativeVerifierState<'proof>, plan: &'plan GrindingPlan) -> Self {
        Self {
            state,
            cursor: GrindingPlanCursor::new(plan),
            invalid: false,
        }
    }

    /// Borrow the native state for ordinary protocol receipt and challenges.
    pub fn state_mut(&mut self) -> &mut NativeVerifierState<'proof> {
        &mut self.state
    }

    /// Receive and validate one scheduled proof-of-work nonce.
    pub fn grind_query(&mut self, site: GrindingSite) -> Result<(), AkitaError> {
        let result = self.grind_query_inner(site);
        self.invalid |= result.is_err();
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
        let (nonce, predicate) =
            receive_native_grinding_nonce(&mut self.state, nonce_record, predicate_record)
                .map_err(|_| AkitaError::InvalidProof)?;
        if !value_fits(nonce, entry.nonce_bits) || !grinding_predicate_accepts(&predicate, bits) {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    /// Receive the next scheduled fold-response nonce.
    pub fn read_fold_response(&mut self, site: GrindingSite) -> Result<u32, AkitaError> {
        let result = self.read_fold_response_inner(site);
        self.invalid |= result.is_err();
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
            .prover_message::<u32>()
            .map_err(|_| AkitaError::InvalidProof)?;
        value_fits(counter, entry.nonce_bits)
            .then_some(counter)
            .ok_or(AkitaError::InvalidProof)
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

    /// Require grinding-plan completion and consume the verifier for EOF.
    pub fn finish(self) -> Result<(), AkitaError> {
        if self.invalid || !self.cursor.is_finished() {
            return Err(AkitaError::InvalidProof);
        }
        self.state.check_eof().map_err(|_| AkitaError::InvalidProof)
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

    fn round_challenge(&mut self, round: u32) -> Result<E, AkitaError> {
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

    fn round_challenge(&mut self, round: u32) -> Result<E, AkitaError> {
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
    use akita_transcript::{new_native_prover, new_native_verifier};
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
    fn native_grinding_roundtrip_uses_inline_u32_nonces_and_eof() {
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
        assert_eq!(proof.len(), 8);
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
    }
}
