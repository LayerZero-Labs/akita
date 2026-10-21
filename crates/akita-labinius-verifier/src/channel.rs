//! The sole dependency boundary to the shared `jolt-transcript` channel.
//!
//! Both roles run on a [`ProverTranscript`] or [`VerifierTranscript`] over any
//! [`Sponge`]. Fold challenges and proof of work use seeded forks, as Akita's
//! fold grinding does: a seed squeezed from the live transcript keys an
//! off-transcript fork per candidate counter, and only the accepted counter is
//! a proof message.
use akita_challenges::{BinaryChallenge, BinaryChallengeDraw, BinaryChallengeSampler};
use akita_error::AkitaError;
use akita_params::{ProtocolSiteId, FOLD_RESPONSE_ATTEMPTS, FOLD_RESPONSE_NONCE_BITS};
use core::marker::PhantomData;
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use jolt_sumcheck::{ProverRoundContext, SumcheckRole, SumcheckSite, VerifierRoundContext};
use jolt_transcript::{
    Blake2b512, Channel, DuplexSpongeInterface, Fork, ProtocolId, ProverTranscript, SiteId, Sponge,
    VerifierTranscript, FORK_SEED_LEN,
};

use crate::grinding::{RootGrindingCursor, RootGrindingPlan, RootGrindingSite};

/// Protocol name of a standalone clear binary opening.
const CLEAR_PROTOCOL: &str = "akita/labinius/clear-binary-opening/v1";
/// Protocol name of a standalone root reduction; separate from the clear
/// opening's session domain.
const ROOT_PROTOCOL: &str = "akita/labinius/root-reduction/v1";

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
    /// As in Akita's fold grinding (`specs/fold-linf-rejection.md`), a fork
    /// seed is squeezed from the live transcript, the nonce follows as one
    /// canonical unsigned LEB128 proof message, and the fold challenges are
    /// drawn from the seed's fork for that nonce. A prover emits `*nonce`; a
    /// verifier overwrites it with the received value. Either role refuses a
    /// value outside the `FOLD_RESPONSE_ATTEMPTS` search domain.
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError>;
}

/// The fork seed of one fold-response search, squeezed from the live
/// transcript before any candidate is tried.
pub struct FoldSeed([u8; FORK_SEED_LEN]);

/// A prover channel that searches fold-response nonces off the transcript.
///
/// [`ClearChannel::fold_challenges`] is [`Self::begin_fold_search`] followed
/// by [`Self::commit_fold_response`].
pub trait FoldSearchChannel: ClearChannel {
    /// Squeeze the seed of the next fold-challenge draw.
    fn begin_fold_search(&mut self) -> FoldSeed;

    /// The challenges `nonce` selects under `seed`; the transcript is
    /// unchanged.
    fn preview_fold_challenges(
        &self,
        seed: &FoldSeed,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError>;

    /// Send the accepted `nonce` and draw the challenges it selects under
    /// `seed`.
    fn commit_fold_response(
        &mut self,
        seed: &FoldSeed,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError>;
}

/// Seed draws for a binary challenge batch from one fold-response fork.
struct ForkDraw<'a, Sp>(&'a mut Fork<Sp>);

impl<Sp: Sponge> BinaryChallengeDraw for ForkDraw<'_, Sp> {
    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        self.0.absorb(payload);
        Ok(self.0.squeeze())
    }
}

/// The fold challenges of `nonce` under `seed`.
fn fork_challenges<Sp: Sponge>(
    seed: &FoldSeed,
    nonce: u32,
    sampler: &mut BinaryChallengeSampler,
    label: &[u8],
    count: usize,
) -> Result<Vec<BinaryChallenge>, AkitaError> {
    let mut fork = Fork::<Sp>::new(&seed.0, nonce);
    sampler.sample_challenges(&mut ForkDraw(&mut fork), label, count)
}

impl<Sp: Sponge> ClearChannel for ProverTranscript<Sp> {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.public_bytes(bytes);
        Ok(())
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        self.send_bytes(bytes);
        Ok(())
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        Ok(self.challenge_bytes())
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        let seed = self.begin_fold_search();
        self.commit_fold_response(&seed, sampler, label, count, *nonce)
    }
}

impl<Sp: Sponge> FoldSearchChannel for ProverTranscript<Sp> {
    fn begin_fold_search(&mut self) -> FoldSeed {
        FoldSeed(self.challenge_bytes())
    }

    fn preview_fold_challenges(
        &self,
        seed: &FoldSeed,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        fork_challenges::<Sp>(seed, nonce, sampler, label, count)
    }

    fn commit_fold_response(
        &mut self,
        seed: &FoldSeed,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        if nonce >= FOLD_RESPONSE_ATTEMPTS {
            return Err(AkitaError::InvalidInput(
                "fold-response nonce exceeds its search domain".into(),
            ));
        }
        self.send_nonce(nonce);
        fork_challenges::<Sp>(seed, nonce, sampler, label, count)
    }
}

impl<Sp: Sponge> ClearChannel for VerifierTranscript<'_, Sp> {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.public_bytes(bytes);
        Ok(())
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        bytes.copy_from_slice(self.receive_bytes(bytes.len())?);
        Ok(())
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        Ok(self.challenge_bytes())
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        let seed = FoldSeed(self.challenge_bytes());
        *nonce = self.receive_nonce(FOLD_RESPONSE_NONCE_BITS)?;
        if *nonce >= FOLD_RESPONSE_ATTEMPTS {
            return Err(AkitaError::InvalidProof);
        }
        fork_challenges::<Sp>(&seed, *nonce, sampler, label, count)
    }
}

/// Bind a versioned geometry header and little-endian `u32` residues.
/// The header fixes the residue count, so fixed-size absorption is injective.
///
/// The digest is a setup identity, independent of the sponge a proof runs on,
/// so it always uses Blake2b.
pub fn matrix_digest(tagged_bytes: &[u8], residues: &[u32]) -> Result<[u8; 32], AkitaError> {
    const CHUNK: usize = 1 << 12;
    let header_len = u64::try_from(tagged_bytes.len())
        .map_err(|_| AkitaError::InvalidSetup("matrix digest header exceeds u64".into()))?;
    let mut sponge = Blake2b512::default();
    sponge.absorb(&header_len.to_le_bytes());
    sponge.absorb(tagged_bytes);
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(CHUNK * size_of::<u32>())
        .map_err(|_| AkitaError::InvalidSetup("matrix digest allocation failed".into()))?;
    for chunk in residues.chunks(CHUNK) {
        bytes.clear();
        for residue in chunk {
            bytes.extend_from_slice(&residue.to_le_bytes());
        }
        sponge.absorb(&bytes);
    }
    let mut digest = [0u8; 32];
    sponge.squeeze(&mut digest);
    Ok(digest)
}

fn standalone_protocol<Sp: Sponge>(name: &str) -> ProtocolId {
    ProtocolId::new::<Sp>(name)
}

pub fn new_prover<Sp: Sponge>() -> Result<ProverTranscript<Sp>, AkitaError> {
    Ok(ProverTranscript::new(
        &standalone_protocol::<Sp>(CLEAR_PROTOCOL),
        b"",
    ))
}

pub fn finish_prover<Sp: Sponge>(channel: ProverTranscript<Sp>) -> Vec<u8> {
    channel.finish()
}

pub fn new_verifier<Sp: Sponge>(proof: &[u8]) -> Result<VerifierTranscript<'_, Sp>, AkitaError> {
    Ok(VerifierTranscript::new(
        &standalone_protocol::<Sp>(CLEAR_PROTOCOL),
        b"",
        proof,
    ))
}

pub fn finish_verifier<Sp: Sponge>(channel: VerifierTranscript<'_, Sp>) -> Result<(), AkitaError> {
    Ok(channel.finish()?)
}

/// Borrow the existing prover transcript for the root reduction over the base
/// field `F`.
///
/// Every challenge field `E: ExtField<F>` is served by the same transcript;
/// sumcheck messages are canonical base coordinates of `E`.
///
/// Every field challenge and round challenge must be the next planned site;
/// an adapter without a scheduled plan rejects them. The root statement
/// binding schedules the reduction's plan ([`RootChallengeChannel::schedule`]);
/// standalone sumchecks schedule [`RootGrindingPlan::sumcheck_rounds`]. The
/// adapter grinds the site's proof of work before drawing.
pub struct RootSumcheckProverChannel<'state, F, Sp: Sponge> {
    state: &'state mut ProverTranscript<Sp>,
    cursor: RootGrindingCursor,
    base: PhantomData<fn() -> F>,
}

impl<'state, F, Sp: Sponge> RootSumcheckProverChannel<'state, F, Sp> {
    pub fn new(state: &'state mut ProverTranscript<Sp>) -> Self {
        Self {
            state,
            cursor: RootGrindingCursor::default(),
            base: PhantomData,
        }
    }

    /// Pay the next planned site, which must be `site`: squeeze a seed, search
    /// nonces on its forks, and send the first that clears the target.
    fn grind(&mut self, site: RootGrindingSite, id: ProtocolSiteId) -> Result<(), AkitaError> {
        let run = self.cursor.next(site)?;
        self.state.site(id.into());
        self.state
            .grind(run.grind_bits())
            .map_err(|_| AkitaError::InvalidInput("transcript grinding exhausted".into()))?;
        Ok(())
    }
}

/// Borrow the existing verifier transcript for the root reduction over the
/// base field `F`. Scheduling is as for the prover adapter; a scheduled site's
/// nonce is received, range-checked and tested before its challenge is drawn.
pub struct RootSumcheckVerifierChannel<'state, 'proof, F, Sp: Sponge> {
    state: &'state mut VerifierTranscript<'proof, Sp>,
    cursor: RootGrindingCursor,
    base: PhantomData<fn() -> F>,
}

impl<'state, 'proof, F, Sp: Sponge> RootSumcheckVerifierChannel<'state, 'proof, F, Sp> {
    pub fn new(state: &'state mut VerifierTranscript<'proof, Sp>) -> Self {
        Self {
            state,
            cursor: RootGrindingCursor::default(),
            base: PhantomData,
        }
    }

    /// Check the next planned site, which must be `site`: its nonce must be
    /// canonical, inside the search range, and clear the target.
    fn grind(&mut self, site: RootGrindingSite, id: ProtocolSiteId) -> Result<(), AkitaError> {
        let run = self.cursor.next(site)?;
        self.state.site(id.into());
        self.state.check_grind(run.grind_bits())?;
        Ok(())
    }
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

impl<F, E, Sp> ProverRoundContext<E> for RootSumcheckProverChannel<'_, F, Sp>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    Sp: Sponge,
{
    type Channel = ProverTranscript<Sp>;
    type Error = AkitaError;

    fn channel_mut(&mut self) -> &mut ProverTranscript<Sp> {
        self.state
    }

    fn site(&self, site: SumcheckSite) -> Option<SiteId> {
        Some(root_sumcheck_site(site.invocation, site.round, site.role).into())
    }

    fn challenge(&mut self, site: SumcheckSite) -> Result<E, AkitaError> {
        let id = root_sumcheck_site(site.invocation, site.round, SumcheckRole::Challenge);
        self.grind(
            RootGrindingSite::Round {
                invocation: site.invocation,
            },
            id,
        )?;
        self.state.site(id.into());
        Ok(self.state.challenge())
    }
}

impl<'proof, F, E, Sp> VerifierRoundContext<E> for RootSumcheckVerifierChannel<'_, 'proof, F, Sp>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    Sp: Sponge,
{
    type Channel = VerifierTranscript<'proof, Sp>;
    type Error = AkitaError;

    fn channel_mut(&mut self) -> &mut VerifierTranscript<'proof, Sp> {
        self.state
    }

    fn site(&self, site: SumcheckSite) -> Option<SiteId> {
        Some(root_sumcheck_site(site.invocation, site.round, site.role).into())
    }

    fn challenge(&mut self, site: SumcheckSite) -> Result<E, AkitaError> {
        let id = root_sumcheck_site(site.invocation, site.round, SumcheckRole::Challenge);
        self.grind(
            RootGrindingSite::Round {
                invocation: site.invocation,
            },
            id,
        )?;
        self.state.site(id.into());
        Ok(self.state.challenge())
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
fn field_point<E, C>(
    state: &mut C,
    site: RootGrindingSite,
    coordinates: usize,
) -> Result<Vec<E>, AkitaError>
where
    E: Field,
    C: Channel,
{
    let mut point = Vec::new();
    point
        .try_reserve_exact(coordinates)
        .map_err(|_| AkitaError::InvalidInput("root challenge allocation failed".into()))?;
    for coordinate in 0..coordinates {
        let coordinate = u32::try_from(coordinate).map_err(|_| AkitaError::InvalidProof)?;
        state.site(root_field_site(site, coordinate).into());
        point.push(state.challenge());
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
pub fn new_root_prover<Sp: Sponge>() -> Result<ProverTranscript<Sp>, AkitaError> {
    Ok(ProverTranscript::new(
        &standalone_protocol::<Sp>(ROOT_PROTOCOL),
        b"",
    ))
}

/// Standalone root replay session, bounded by the caller's proof slice.
pub fn new_root_verifier<Sp: Sponge>(
    proof: &[u8],
) -> Result<VerifierTranscript<'_, Sp>, AkitaError> {
    Ok(VerifierTranscript::new(
        &standalone_protocol::<Sp>(ROOT_PROTOCOL),
        b"",
        proof,
    ))
}

impl<F, Sp: Sponge> ClearChannel for RootSumcheckProverChannel<'_, F, Sp> {
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
impl<F, Sp: Sponge> FoldSearchChannel for RootSumcheckProverChannel<'_, F, Sp> {
    fn begin_fold_search(&mut self) -> FoldSeed {
        self.state.begin_fold_search()
    }
    fn preview_fold_challenges(
        &self,
        seed: &FoldSeed,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        self.state
            .preview_fold_challenges(seed, sampler, label, count, nonce)
    }
    fn commit_fold_response(
        &mut self,
        seed: &FoldSeed,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        self.state
            .commit_fold_response(seed, sampler, label, count, nonce)
    }
}
impl<F, Sp: Sponge> ClearChannel for RootSumcheckVerifierChannel<'_, '_, F, Sp> {
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
impl<F: Field + CanonicalEncoding, E: ExtField<F>, Sp: Sponge> RootChallengeChannel<E>
    for RootSumcheckProverChannel<'_, F, Sp>
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
impl<F: Field + CanonicalEncoding, E: ExtField<F>, Sp: Sponge> RootChallengeChannel<E>
    for RootSumcheckVerifierChannel<'_, '_, F, Sp>
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
