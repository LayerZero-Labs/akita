//! The sole dependency boundary to Akita's current transcript transport.
#[cfg(feature = "prover")]
use akita_challenges::PreviewFoldDraw;
use akita_challenges::ProverFoldDraw;
use akita_challenges::VerifierFoldDraw;
use akita_challenges::{BinaryChallenge, BinaryChallengeSampler};
use akita_error::AkitaError;
use akita_params::FOLD_RESPONSE_ATTEMPTS;
use akita_sumcheck::SumcheckRole;
use akita_transcript::{
    ext_challenge, nonce_max_bytes, receive_byte_group, receive_nonce, send_byte_group, send_nonce,
    verifier_context, ChallengeCodec, ProofChannel, ProtocolContextRecord, ProtocolMessageKind,
    ProtocolSiteId, ProverChannel, VerifierChannel, VerifierTranscript, GRINDING_CHECK_BYTES,
};
#[cfg(feature = "prover")]
use akita_transcript::{prover_context, FoldPreview, ProverTranscript};
use core::marker::PhantomData;
use jolt_algebra::{CanonicalEncoding, ExtField, Field};

use crate::grinding::{RootGrindingCursor, RootGrindingPlan, RootGrindingSite};

/// Minimal role-generic channel for the standalone clear protocol.
/// Site labels are diagnostic; protocol domains are explicitly absorbed bytes.
pub trait ClearChannel {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError>;
    /// Emit the fixed-length buffer, or overwrite it with received bytes.
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError>;
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError>;
    /// Run a native opening on this same prover channel. Custom observational
    /// channels may reject native openings instead of exposing transport state.
    fn with_prover_session<R>(
        &mut self,
        _prove: impl FnOnce(&mut ProverChannel) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        Err(AkitaError::InvalidProof)
    }

    /// Replay a native opening on this same verifier channel. The closure only
    /// borrows the owner and cannot finish or replace the surrounding channel.
    fn with_verifier_session<R>(
        &mut self,
        _verify: impl FnOnce(&mut VerifierChannel<'_>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        Err(AkitaError::InvalidProof)
    }

    /// Exchange the fold-response nonce, then draw the fold challenges it
    /// selects.
    ///
    /// The nonce is one canonical unsigned LEB128 proof message placed
    /// directly before the fold-challenge draw, as in Akita's fold grinding
    /// (`specs/fold-linf-rejection.md`). A prover emits `*nonce`; a verifier
    /// overwrites it with the received value. Either role refuses a value
    /// outside the `FOLD_RESPONSE_ATTEMPTS` search domain.
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError>;
}

/// A prover channel that can evaluate a candidate fold-response nonce without
/// writing it to the proof.
#[cfg(feature = "prover")]
pub trait FoldSearchChannel: ClearChannel {
    /// The challenges [`ClearChannel::fold_challenges`] returns for `nonce`
    /// from the current transcript state, which is left unchanged.
    fn preview_fold_challenges(
        &self,
        sampler: &mut BinaryChallengeSampler,
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError>;
}

/// Diagnostic site of the clear protocol's fixed-length proof messages.
fn clear_message_site(len: usize) -> ProtocolSiteId {
    ProtocolSiteId {
        // "LCMS": the clear protocol's own proof messages.
        family: 0x4c43_4d53,
        detail: u32::try_from(len).unwrap_or(u32::MAX),
        ..ProtocolSiteId::default()
    }
}

fn challenge_block<S: ProofChannel>(state: &mut S) -> Result<[u8; 32], AkitaError> {
    state.challenge_message(32, |bytes| {
        bytes
            .try_into()
            .map_err(|_| AkitaError::Internal("challenge block width".into()))
    })
}

impl ClearChannel for ProverChannel {
    fn with_prover_session<R>(
        &mut self,
        prove: impl FnOnce(&mut ProverChannel) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        prove(self)
    }
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        ProofChannel::public(self, bytes);
        Ok(())
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        send_byte_group(self, clear_message_site(bytes.len()), bytes)
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        challenge_block(self)
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        if *nonce >= FOLD_RESPONSE_ATTEMPTS {
            return Err(AkitaError::InvalidInput(
                "fold-response nonce exceeds its search domain".into(),
            ));
        }
        send_nonce(self, *nonce);
        sampler.sample_challenges(&mut ProverFoldDraw::new(self, 0, 0), count)
    }
}

#[cfg(feature = "prover")]
impl FoldSearchChannel for ProverChannel {
    fn preview_fold_challenges(
        &self,
        sampler: &mut BinaryChallengeSampler,
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        let mut preview = FoldPreview::new(self, nonce);
        sampler.sample_challenges(&mut PreviewFoldDraw::new(&mut preview), count)
    }
}

impl ClearChannel for VerifierChannel<'_> {
    fn with_verifier_session<R>(
        &mut self,
        verify: impl FnOnce(&mut VerifierChannel<'_>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        verify(self)
    }
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        ProofChannel::public(self, bytes);
        Ok(())
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        let received = receive_byte_group(self, clear_message_site(bytes.len()), bytes.len())?;
        bytes.copy_from_slice(&received);
        Ok(())
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        challenge_block(self)
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        *nonce = receive_nonce(self)?;
        if *nonce >= FOLD_RESPONSE_ATTEMPTS {
            return Err(AkitaError::InvalidProof);
        }
        sampler.sample_challenges(&mut VerifierFoldDraw::new(self, 0, 0), count)
    }
}

/// Bind a versioned geometry header and little-endian `u32` residues.
/// The header fixes the residue count, so fixed-size absorption is injective.
pub fn matrix_digest(tagged_bytes: &[u8], residues: &[u32]) -> Result<[u8; 32], AkitaError> {
    const CHUNK: usize = 1 << 12;
    let mut channel = akita_transcript::new_prover_channel(tagged_bytes);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(CHUNK * size_of::<u32>())
        .map_err(|_| AkitaError::InvalidSetup("matrix digest allocation failed".into()))?;
    for chunk in residues.chunks(CHUNK) {
        bytes.clear();
        for residue in chunk {
            bytes.extend_from_slice(&residue.to_le_bytes());
        }
        ProofChannel::public(&mut channel, &bytes);
    }
    challenge_block(&mut channel)
}

pub fn new_prover() -> ProverChannel {
    akita_transcript::new_prover_channel(b"akita/labinius/clear-binary-opening/v1")
}

pub fn finish_prover(channel: ProverChannel) -> Vec<u8> {
    channel.narg_string().to_vec()
}

pub fn new_verifier(proof: &[u8]) -> VerifierChannel<'_> {
    akita_transcript::new_verifier_channel(b"akita/labinius/clear-binary-opening/v1", proof)
}

pub fn finish_verifier(channel: VerifierChannel<'_>) -> Result<(), AkitaError> {
    channel.check_eof().map_err(|_| AkitaError::InvalidProof)
}

/// Borrow the existing prover channel for the root reduction over the base
/// field `F`.
///
/// Every challenge field `E: ExtField<F>` is served by the same channel: a
/// challenge is one codec draw of `E` and a sumcheck message is its
/// canonical base coordinates, as in Akita's extension-field transcript.
///
/// Every field challenge and round challenge must be the next planned site;
/// an adapter without a scheduled plan rejects them. The root statement
/// binding schedules the reduction's plan ([`RootChallengeChannel::schedule`]);
/// standalone sumchecks schedule [`RootGrindingPlan::sumcheck_rounds`]. The
/// adapter runs the site's transcript proof of work before drawing.
#[cfg(feature = "prover")]
pub struct RootSumcheckProverChannel<'state, F> {
    state: &'state mut ProverChannel,
    cursor: RootGrindingCursor,
    base: PhantomData<fn() -> F>,
}

#[cfg(feature = "prover")]
impl<'state, F> RootSumcheckProverChannel<'state, F> {
    pub fn new(state: &'state mut ProverChannel) -> Self {
        Self {
            state,
            cursor: RootGrindingCursor::default(),
            base: PhantomData,
        }
    }

    /// Pay the next planned site, which must be `site`, with the transcript's
    /// proof of work.
    fn grind(&mut self, site: RootGrindingSite, id: ProtocolSiteId) -> Result<(), AkitaError> {
        let run = self.cursor.next(site)?;
        if run.grind_bits() == 0 {
            return Ok(());
        }
        let (nonce_record, check_record) = grinding_records(id, run.nonce_bits());
        prover_context(self.state, nonce_record);
        self.state
            .grind(u32::from(run.grind_bits()))
            .map_err(|_| AkitaError::InvalidInput("transcript grinding exhausted".into()))?;
        prover_context(self.state, check_record);
        Ok(())
    }
}

/// Borrow the existing verifier channel for the root reduction over the base
/// field `F`. Scheduling is as for the prover adapter; a scheduled site's
/// proof of work is checked before its challenge is drawn.
pub struct RootSumcheckVerifierChannel<'state, 'proof, F> {
    state: &'state mut VerifierChannel<'proof>,
    cursor: RootGrindingCursor,
    base: PhantomData<fn() -> F>,
}

impl<'state, 'proof, F> RootSumcheckVerifierChannel<'state, 'proof, F> {
    pub fn new(state: &'state mut VerifierChannel<'proof>) -> Self {
        Self {
            state,
            cursor: RootGrindingCursor::default(),
            base: PhantomData,
        }
    }

    /// Check the next planned site, which must be `site`: the transcript's
    /// proof-of-work nonce must be canonical, inside the search range, and
    /// clear the target's check bits.
    fn grind(&mut self, site: RootGrindingSite, id: ProtocolSiteId) -> Result<(), AkitaError> {
        let run = self.cursor.next(site)?;
        if run.grind_bits() == 0 {
            return Ok(());
        }
        let (nonce_record, check_record) = grinding_records(id, run.nonce_bits());
        verifier_context(self.state, nonce_record);
        self.state
            .check_grind(u32::from(run.grind_bits()))
            .map_err(|_| AkitaError::InvalidProof)?;
        verifier_context(self.state, check_record);
        Ok(())
    }
}

/// Diagnostic records of one proof-of-work nonce and its check draw.
fn grinding_records(
    site: ProtocolSiteId,
    nonce_bits: u8,
) -> (ProtocolContextRecord, ProtocolContextRecord) {
    let site_id = site.to_bytes();
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

fn root_sumcheck_site(invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
    ProtocolSiteId {
        // "LRSC": a diagnostic family separate from the Akita stage sumchecks.
        family: 0x4c52_5343,
        invocation,
        round,
        detail: role as u32,
        ..ProtocolSiteId::default()
    }
}

#[cfg(feature = "prover")]
impl<F, E> akita_sumcheck::SumcheckProverChannel<E> for RootSumcheckProverChannel<'_, F>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + ChallengeCodec,
{
    fn state_mut(&mut self) -> &mut ProverChannel {
        self.state
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        root_sumcheck_site(invocation, round, role)
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        let id = root_sumcheck_site(invocation, round, SumcheckRole::Challenge);
        self.grind(RootGrindingSite::Round { invocation }, id)?;
        ext_challenge::<E, _>(self.state, id)
    }
}

impl<'proof, F, E> akita_sumcheck::SumcheckVerifierChannel<'proof, E>
    for RootSumcheckVerifierChannel<'_, 'proof, F>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + ChallengeCodec,
{
    fn state_mut(&mut self) -> &mut VerifierChannel<'proof> {
        self.state
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        root_sumcheck_site(invocation, round, role)
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        let id = root_sumcheck_site(invocation, round, SumcheckRole::Challenge);
        self.grind(RootGrindingSite::Round { invocation }, id)?;
        ext_challenge::<E, _>(self.state, id)
    }
}

/// A root channel replays a grinding plan and draws exact
/// uniform challenges of the challenge field `E`, each after the proof of
/// work its site is planned for. Site identifiers are diagnostics; the root
/// grammar and the absorbed plan supply cryptographic binding.
pub trait RootChallengeChannel<E>: ClearChannel {
    /// Start replaying `plan`. The root statement binding calls this once,
    /// after absorbing the plan's canonical bytes. Standalone sumchecks
    /// schedule their round-site plan before the first round challenge.
    fn schedule(&mut self, plan: RootGrindingPlan) -> Result<(), AkitaError>;

    /// Pay `site`, which must be the next planned site, then draw
    /// `coordinates` independent challenges under that one proof of work.
    fn field_point(
        &mut self,
        site: RootGrindingSite,
        coordinates: usize,
    ) -> Result<Vec<E>, AkitaError>;

    /// Pay `site`, then draw its single challenge.
    fn field_challenge(&mut self, site: RootGrindingSite) -> Result<E, AkitaError> {
        self.field_point(site, 1)?
            .pop()
            .ok_or(AkitaError::InvalidProof)
    }

    /// Require that every planned site was visited.
    fn finish_schedule(&mut self) -> Result<(), AkitaError>;
}

/// Draw the `coordinates` challenges of a site whose work is already paid.
fn field_point<E, C>(
    state: &mut C,
    site: RootGrindingSite,
    coordinates: usize,
) -> Result<Vec<E>, AkitaError>
where
    E: ChallengeCodec,
    C: ProofChannel,
{
    let mut point = Vec::new();
    point
        .try_reserve_exact(coordinates)
        .map_err(|_| AkitaError::InvalidInput("root challenge allocation failed".into()))?;
    for coordinate in 0..coordinates {
        let coordinate = u32::try_from(coordinate).map_err(|_| AkitaError::InvalidProof)?;
        point.push(ext_challenge::<E, _>(
            state,
            root_field_site(site, coordinate),
        )?);
    }
    Ok(point)
}

/// Diagnostic site of coordinate `group` of a challenge-field site.
fn root_field_site(site: RootGrindingSite, group: u32) -> ProtocolSiteId {
    let (tag, invocation) = site.key();
    ProtocolSiteId {
        // "LRRD": the root reduction's own field challenges.
        family: 0x4c52_5244,
        invocation,
        group,
        detail: u32::from(tag),
        ..ProtocolSiteId::default()
    }
}

/// Absorb the root reduction's public setup identity on a caller's channel,
/// before its first challenge: the fixed root-reduction tag, then the identity
/// as a `u64` little-endian length word and its bytes.
pub fn bind_root_context<S: ProofChannel>(
    state: &mut S,
    identity: &[u8],
) -> Result<(), AkitaError> {
    let len = u64::try_from(identity.len())
        .map_err(|_| AkitaError::InvalidInput("root identity length overflow".into()));
    let len = state.guard(len)?;
    let mut bytes = ROOT_SESSION.to_vec();
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(identity);
    akita_transcript::public_bytes(
        state,
        ProtocolSiteId {
            // "LRCX": the root reduction's public context.
            family: 0x4c52_4358,
            ..ProtocolSiteId::default()
        },
        &bytes,
    )
}

const ROOT_SESSION: &[u8] = b"akita/labinius/root-reduction/v1";

/// Standalone root session; separate from the clear-opening session domain.
pub fn new_root_prover() -> ProverChannel {
    akita_transcript::new_prover_channel(ROOT_SESSION)
}

/// Standalone root replay session, bounded by the caller's proof slice.
pub fn new_root_verifier(proof: &[u8]) -> VerifierChannel<'_> {
    akita_transcript::new_verifier_channel(ROOT_SESSION, proof)
}

#[cfg(feature = "prover")]
impl<F> ClearChannel for RootSumcheckProverChannel<'_, F> {
    fn with_prover_session<R>(
        &mut self,
        prove: impl FnOnce(&mut ProverChannel) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        prove(self.state)
    }
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        ClearChannel::public(self.state, bytes)
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        ClearChannel::message(self.state, bytes)
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        ClearChannel::challenge_block(self.state)
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        ClearChannel::fold_challenges(self.state, sampler, count, nonce)
    }
}
#[cfg(feature = "prover")]
impl<F> FoldSearchChannel for RootSumcheckProverChannel<'_, F> {
    fn preview_fold_challenges(
        &self,
        sampler: &mut BinaryChallengeSampler,
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        FoldSearchChannel::preview_fold_challenges(&*self.state, sampler, count, nonce)
    }
}
impl<F> ClearChannel for RootSumcheckVerifierChannel<'_, '_, F> {
    fn with_verifier_session<R>(
        &mut self,
        verify: impl FnOnce(&mut VerifierChannel<'_>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        verify(self.state)
    }
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        ClearChannel::public(self.state, bytes)
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        ClearChannel::message(self.state, bytes)
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        ClearChannel::challenge_block(self.state)
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        ClearChannel::fold_challenges(self.state, sampler, count, nonce)
    }
}
#[cfg(feature = "prover")]
impl<F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec> RootChallengeChannel<E>
    for RootSumcheckProverChannel<'_, F>
{
    fn schedule(&mut self, plan: RootGrindingPlan) -> Result<(), AkitaError> {
        self.cursor.schedule(plan)
    }
    fn field_point(
        &mut self,
        site: RootGrindingSite,
        coordinates: usize,
    ) -> Result<Vec<E>, AkitaError> {
        self.grind(site, root_field_site(site, 0))?;
        field_point::<E, _>(self.state, site, coordinates)
    }
    fn finish_schedule(&mut self) -> Result<(), AkitaError> {
        self.cursor.finish()
    }
}
impl<F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec> RootChallengeChannel<E>
    for RootSumcheckVerifierChannel<'_, '_, F>
{
    fn schedule(&mut self, plan: RootGrindingPlan) -> Result<(), AkitaError> {
        self.cursor.schedule(plan)
    }
    fn field_point(
        &mut self,
        site: RootGrindingSite,
        coordinates: usize,
    ) -> Result<Vec<E>, AkitaError> {
        self.grind(site, root_field_site(site, 0))?;
        field_point::<E, _>(self.state, site, coordinates)
    }
    fn finish_schedule(&mut self) -> Result<(), AkitaError> {
        self.cursor.finish()
    }
}
