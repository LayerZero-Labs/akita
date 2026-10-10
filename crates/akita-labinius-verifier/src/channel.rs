//! The sole dependency boundary to Akita's current transcript transport.
use akita_challenges::{
    BinaryChallenge, BinaryChallengeSampler, PreviewFoldDraw, ProverFoldDraw, VerifierFoldDraw,
};
use akita_error::AkitaError;
use akita_params::FOLD_RESPONSE_ATTEMPTS;
use akita_sumcheck::SumcheckRole;
use akita_transcript::{
    commit_grinding_nonce, ext_challenge, grinding_predicate_accepts, nonce_max_bytes,
    receive_grinding_nonce, search_grinding_nonce, FoldPreview, NonceAtom, ProofChannel,
    ProtocolContextRecord, ProtocolMessageKind, ProtocolSiteId, GRINDING_PREDICATE_LEN,
};
use core::{marker::PhantomData, num::NonZeroU8};
use jolt_field::{CanonicalEncoding, ExtField, Field};

use crate::grinding::{RootGrindingCursor, RootGrindingPlan, RootGrindingSite};

/// Minimal role-generic channel for the standalone clear protocol.
/// Site labels are diagnostic; protocol domains are explicitly absorbed bytes.
pub trait ClearChannel {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError>;
    /// Emit the fixed-length buffer, or overwrite it with received bytes.
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError>;
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError>;
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
        label: &[u8],
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError>;
}

/// A prover channel that can evaluate a candidate fold-response nonce without
/// writing it to the proof.
pub trait FoldSearchChannel: ClearChannel {
    /// The challenges [`ClearChannel::fold_challenges`] returns for `nonce`
    /// from the current transcript state, which is left unchanged.
    fn preview_fold_challenges(
        &self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError>;
}

impl ClearChannel for akita_transcript::ProverChannel {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.public_message(bytes);
        Ok(())
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        akita_transcript::send_bytes(self, bytes);
        Ok(())
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        Ok(self.verifier_message())
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        if *nonce >= FOLD_RESPONSE_ATTEMPTS {
            return Err(AkitaError::InvalidInput(
                "fold-response nonce exceeds its search domain".into(),
            ));
        }
        self.prover_message(&NonceAtom::new(*nonce));
        sampler.sample_challenges(&mut ProverFoldDraw::new(self, 0, 0), label, count)
    }
}

impl FoldSearchChannel for akita_transcript::ProverChannel {
    fn preview_fold_challenges(
        &self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        let mut preview = FoldPreview::new(self, nonce);
        sampler.sample_challenges(&mut PreviewFoldDraw::new(&mut preview), label, count)
    }
}

impl ClearChannel for akita_transcript::VerifierChannel<'_> {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.public_message(bytes);
        Ok(())
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        let received = akita_transcript::receive_bytes(self, bytes.len())?;
        bytes.copy_from_slice(&received);
        Ok(())
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        self.verifier_message()
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        *nonce = self.prover_message::<NonceAtom>()?.into_inner();
        if *nonce >= FOLD_RESPONSE_ATTEMPTS {
            return Err(AkitaError::InvalidProof);
        }
        sampler.sample_challenges(&mut VerifierFoldDraw::new(self, 0, 0), label, count)
    }
}

/// Bind a versioned geometry header and little-endian `u32` residues.
/// The header fixes the residue count, so fixed-size absorption is injective.
pub fn matrix_digest(tagged_bytes: &[u8], residues: &[u32]) -> Result<[u8; 32], AkitaError> {
    const CHUNK: usize = 1 << 12;
    let mut channel = akita_transcript::new_prover_channel(tagged_bytes, b"")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(CHUNK * size_of::<u32>())
        .map_err(|_| AkitaError::InvalidSetup("matrix digest allocation failed".into()))?;
    for chunk in residues.chunks(CHUNK) {
        bytes.clear();
        for residue in chunk {
            bytes.extend_from_slice(&residue.to_le_bytes());
        }
        ClearChannel::public(&mut channel, &bytes)?;
    }
    ClearChannel::challenge_block(&mut channel)
}

pub fn new_prover() -> Result<akita_transcript::ProverChannel, AkitaError> {
    akita_transcript::new_prover_channel(b"akita/labinius/clear-binary-opening/v1", b"")
}

pub fn finish_prover(channel: akita_transcript::ProverChannel) -> Vec<u8> {
    channel.narg_string().to_vec()
}

pub fn new_verifier(proof: &[u8]) -> Result<akita_transcript::VerifierChannel<'_>, AkitaError> {
    akita_transcript::new_verifier_channel(b"akita/labinius/clear-binary-opening/v1", b"", proof)
}

pub fn finish_verifier(channel: akita_transcript::VerifierChannel<'_>) -> Result<(), AkitaError> {
    channel.check_eof()
}

/// Borrow the existing prover channel for the root reduction over the base
/// field `F`.
///
/// Every challenge field `E: ExtField<F>` is served by the same channel: a
/// challenge is `E::DEGREE` base-field draws and a sumcheck message is its
/// canonical base coordinates, as in Akita's extension-field transcript.
///
/// Every field challenge and round challenge must be the next planned site;
/// an adapter without a scheduled plan rejects them. The root statement
/// binding schedules the reduction's plan ([`RootChallengeChannel::schedule`]);
/// standalone sumchecks schedule [`RootGrindingPlan::sumcheck_rounds`]. The
/// adapter searches and emits the site's proof-of-work nonce before drawing.
pub struct RootSumcheckProverChannel<'state, F> {
    state: &'state mut akita_transcript::ProverChannel,
    cursor: RootGrindingCursor,
    base: PhantomData<fn() -> F>,
}

impl<'state, F> RootSumcheckProverChannel<'state, F> {
    pub fn new(state: &'state mut akita_transcript::ProverChannel) -> Self {
        Self {
            state,
            cursor: RootGrindingCursor::default(),
            base: PhantomData,
        }
    }

    /// Pay the next planned site, which must be `site`: search the bounded
    /// nonce range on previews of the public state and commit the winner.
    fn grind(&mut self, site: RootGrindingSite, id: ProtocolSiteId) -> Result<(), AkitaError> {
        let run = self.cursor.next(site)?;
        let Some(target) = NonZeroU8::new(run.grind_bits()) else {
            return Ok(());
        };
        let (nonce, preview) =
            search_grinding_nonce(self.state, run.grind_bits(), run.nonce_bits())
                .ok_or_else(|| AkitaError::InvalidInput("transcript grinding exhausted".into()))?;
        let (nonce_record, predicate_record) = grinding_records(id, run.nonce_bits());
        let predicate = commit_grinding_nonce(self.state, nonce_record, nonce, predicate_record);
        if predicate != preview || !grinding_predicate_accepts(&predicate, target) {
            return Err(AkitaError::Internal(
                "committed grinding predicate differs from its preview".into(),
            ));
        }
        Ok(())
    }
}

/// Borrow the existing verifier channel for the root reduction over the base
/// field `F`. Scheduling is as for the prover adapter; a scheduled site's
/// nonce is received, range-checked and tested before its challenge is drawn.
pub struct RootSumcheckVerifierChannel<'state, 'proof, F> {
    state: &'state mut akita_transcript::VerifierChannel<'proof>,
    cursor: RootGrindingCursor,
    base: PhantomData<fn() -> F>,
}

impl<'state, 'proof, F> RootSumcheckVerifierChannel<'state, 'proof, F> {
    pub fn new(state: &'state mut akita_transcript::VerifierChannel<'proof>) -> Self {
        Self {
            state,
            cursor: RootGrindingCursor::default(),
            base: PhantomData,
        }
    }

    /// Check the next planned site, which must be `site`: its nonce must be
    /// canonical, inside the search range, and clear the target's predicate
    /// bits.
    fn grind(&mut self, site: RootGrindingSite, id: ProtocolSiteId) -> Result<(), AkitaError> {
        let run = self.cursor.next(site)?;
        let Some(target) = NonZeroU8::new(run.grind_bits()) else {
            return Ok(());
        };
        let (nonce_record, predicate_record) = grinding_records(id, run.nonce_bits());
        let (nonce, predicate) =
            receive_grinding_nonce(self.state, nonce_record, predicate_record)?;
        if u64::from(nonce) >> run.nonce_bits() != 0
            || !grinding_predicate_accepts(&predicate, target)
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }
}

/// Diagnostic records of one proof-of-work nonce and its predicate draw.
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
            ProtocolMessageKind::GrindingPredicate as u32,
            0,
            0,
            GRINDING_PREDICATE_LEN as u64,
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

impl<F, E> akita_sumcheck::SumcheckProverChannel<E> for RootSumcheckProverChannel<'_, F>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn state_mut(&mut self) -> &mut akita_transcript::ProverChannel {
        self.state
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        root_sumcheck_site(invocation, round, role)
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        let id = root_sumcheck_site(invocation, round, SumcheckRole::Challenge);
        self.grind(RootGrindingSite::Round { invocation }, id)?;
        ext_challenge::<F, E, _>(self.state, id)
    }
}

impl<'proof, F, E> akita_sumcheck::SumcheckVerifierChannel<'proof, E>
    for RootSumcheckVerifierChannel<'_, 'proof, F>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn state_mut(&mut self) -> &mut akita_transcript::VerifierChannel<'proof> {
        self.state
    }

    fn sumcheck_site(&self, invocation: u32, round: u32, role: SumcheckRole) -> ProtocolSiteId {
        root_sumcheck_site(invocation, round, role)
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        let id = root_sumcheck_site(invocation, round, SumcheckRole::Challenge);
        self.grind(RootGrindingSite::Round { invocation }, id)?;
        ext_challenge::<F, E, _>(self.state, id)
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
    /// `coordinates` independent challenges under that one nonce.
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
fn field_point<F, E, C>(
    state: &mut C,
    site: RootGrindingSite,
    coordinates: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: ProofChannel,
{
    let mut point = Vec::new();
    point
        .try_reserve_exact(coordinates)
        .map_err(|_| AkitaError::InvalidInput("root challenge allocation failed".into()))?;
    for coordinate in 0..coordinates {
        let coordinate = u32::try_from(coordinate).map_err(|_| AkitaError::InvalidProof)?;
        point.push(ext_challenge::<F, E, _>(
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

/// Standalone root session; separate from the clear-opening session domain.
pub fn new_root_prover() -> Result<akita_transcript::ProverChannel, AkitaError> {
    akita_transcript::new_prover_channel(b"akita/labinius/root-reduction/v1", b"")
}

/// Standalone root replay session, bounded by the caller's proof slice.
pub fn new_root_verifier(
    proof: &[u8],
) -> Result<akita_transcript::VerifierChannel<'_>, AkitaError> {
    akita_transcript::new_verifier_channel(b"akita/labinius/root-reduction/v1", b"", proof)
}

impl<F> ClearChannel for RootSumcheckProverChannel<'_, F> {
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
        label: &[u8],
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        ClearChannel::fold_challenges(self.state, sampler, label, count, nonce)
    }
}
impl<F> FoldSearchChannel for RootSumcheckProverChannel<'_, F> {
    fn preview_fold_challenges(
        &self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        FoldSearchChannel::preview_fold_challenges(&*self.state, sampler, label, count, nonce)
    }
}
impl<F> ClearChannel for RootSumcheckVerifierChannel<'_, '_, F> {
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
        label: &[u8],
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        ClearChannel::fold_challenges(self.state, sampler, label, count, nonce)
    }
}
impl<F: Field + CanonicalEncoding, E: ExtField<F>> RootChallengeChannel<E>
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
        field_point::<F, E, _>(self.state, site, coordinates)
    }
    fn finish_schedule(&mut self) -> Result<(), AkitaError> {
        self.cursor.finish()
    }
}
impl<F: Field + CanonicalEncoding, E: ExtField<F>> RootChallengeChannel<E>
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
        field_point::<F, E, _>(self.state, site, coordinates)
    }
    fn finish_schedule(&mut self) -> Result<(), AkitaError> {
        self.cursor.finish()
    }
}
