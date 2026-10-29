//! Session construction, canonical Akita proof atoms and diagnostic site records.

use akita_error::{narrowing::usize_to_u64, AkitaError};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use jolt_transcript::{BytesMsg, ChallengeCodec, Shake128};
use spongefish::{
    Encoding, FromNarg, NargReader, ProverState, SessionId, Transcript, VerificationError,
    VerifierState,
};
use std::marker::PhantomData;

use crate::{ProverChannel, VerifierChannel};

mod channel;
mod nonce;
mod site;
pub use channel::ProofChannel;
pub use nonce::{nonce_encoded_len, nonce_max_bytes, receive_nonce, send_nonce, NonceAtom};

/// Proof channel and proof-stream format version, named by [`SESSION_TAG`].
///
/// Version 8 fixes:
///
/// - **The challenge codec.** Jolt's pooled rejection codec, within total
///   variation `2^-161` of uniform per message; pools of 4-byte fields round
///   their candidate count up to even.
/// - **The proof-of-work layout.** The least nonce below `2^(bits + 7)`, sent
///   as canonical LEB128 and absorbed as one `u64` little-endian word, then
///   [`GRINDING_CHECK_BYTES`](crate::GRINDING_CHECK_BYTES) check bytes. The
///   fold-response nonce is sent and absorbed the same way.
/// - **The absorb layout.** Positional and word aligned. No message carries
///   a label; its place in the schedule identifies it. Every absorb and every
///   squeeze covers whole 8-byte words. A message whose length bound data
///   fixes is absorbed as its wire bytes, zero-padded to a word; a length
///   only the prover knows is absorbed as one `u64` word before the bytes.
/// - **The sparse-message codec.** One fold-challenge message per commitment
///   group, squeezed directly after the fold's proof of work with nothing
///   absorbed in between. Its length is the candidate bits rounded up to
///   whole words, and candidate `c` occupies bits `[c·K, (c + 1)·K)` of the
///   message read as little-endian `u64` words.
///
/// The fold draw absorbs no payload, so these values are bound only by this
/// version and change it when they change: the `OperatorNormRejection`
/// constants, the pool rates that set a sparse message's length, and the
/// meaning of the coefficient-packing draw domain.
pub const PROTOCOL_VERSION: u32 = 8;

/// Spongefish session tag: the protocol name, [`PROTOCOL_VERSION`] and the
/// digest hash ([`jolt_transcript::DIGEST_HASH`]), zero-padded to 32 bytes:
/// `akita-pcs/transcript/v8/sha3`, or `akita-pcs/transcript/v8/keccak` under
/// `keccak256-digests`.
///
/// Naming the digest hash keeps a proof made under one selection from
/// verifying under the other. The tag does not name the sponge: a different
/// sponge is a different hash function and already changes every output.
pub const SESSION_TAG: [u8; 32] =
    jolt_transcript::protocol_tag(match jolt_transcript::DIGEST_HASH {
        jolt_transcript::DigestHash::Sha3_256 => b"akita-pcs/transcript/v8/sha3".as_slice(),
        jolt_transcript::DigestHash::Keccak256 => b"akita-pcs/transcript/v8/keccak".as_slice(),
    });

/// Domain tag stored in every diagnostic context record.
pub const CONTEXT_DOMAIN: [u8; 32] = *b"akita-pcs/native-context/v8\0\0\0\0\0";

/// Bytes in one transcript word. Every absorb and every squeeze covers whole
/// words, so a sponge over `u64` lanes reproduces the byte sponge exactly.
const WORD_BYTES: usize = 8;

/// Stable family identifier for standard and batched sumcheck sites.
pub const SITE_FAMILY_SUMCHECK: u32 = 1;
/// Stable family identifier for extension-opening reduction messages.
pub const SITE_FAMILY_EXTENSION_OPENING_REDUCTION: u32 = 2;
/// Stable family identifier for stage-2 terminal claims.
pub const SITE_FAMILY_STAGE2: u32 = 3;
/// Stable family identifier for stage-1 late oracle claims.
pub const SITE_FAMILY_STAGE1: u32 = 4;
/// Stable family identifier for physical-L2 proof values.
pub const SITE_FAMILY_PHYSICAL_L2: u32 = 5;
/// Stable family identifier for recursive setup-product stage 3.
pub const SITE_FAMILY_STAGE3: u32 = 6;
/// Stable family identifier for indexed sparse fold-challenge roots.
pub const SITE_FAMILY_FOLD_CHALLENGE: u32 = 7;
/// Stable family identifier for ring-relation opening payloads.
pub const SITE_FAMILY_OPENING_PAYLOAD: u32 = 8;
/// Stable family identifier for derived fold-opening values.
pub const SITE_FAMILY_FOLD_BINDING: u32 = 9;
/// Stable family identifier for the successor witness binding.
pub const SITE_FAMILY_NEXT_WITNESS: u32 = 10;
/// Stable family identifier for root public commitments and opening points.
pub const SITE_FAMILY_ROOT_STATEMENT: u32 = 11;
/// Stable family identifier for terminal response messages.
pub const SITE_FAMILY_TERMINAL: u32 = 12;

/// Proof-stream operation kind recorded by diagnostic context metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ProtocolMessageKind {
    /// A public or derived value supplied outside the argument string.
    PublicValue = 1,
    /// A bounded variable payload's length atom.
    ProofLength = 2,
    /// One or more canonical proof atoms.
    ProofAtoms = 3,
    /// A verifier challenge group.
    Challenge = 4,
    /// A proof-of-work nonce.
    GrindingNonce = 5,
    /// The check bytes squeezed by a proof-of-work step.
    GrindingCheck = 6,
    /// A fold-response search nonce atom.
    FoldResponseNonce = 7,
}

/// Canonical 32-byte identity of one protocol site.
///
/// Unused coordinates are zero. Callers derive every coordinate from validated
/// public schedule state; proof input never selects a site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProtocolSiteId {
    /// Protocol family, such as sumcheck or folding.
    pub family: u32,
    /// Invocation within the enclosing public schedule.
    pub invocation: u32,
    /// Recursive fold level.
    pub level: u32,
    /// Stage within a level.
    pub stage: u32,
    /// Round within a stage.
    pub round: u32,
    /// Commitment group within a round.
    pub group: u32,
    /// Element index within a challenge group.
    pub limb: u32,
    /// Family-specific public discriminator, such as grinding widths.
    pub detail: u32,
}

impl ProtocolSiteId {
    /// Encode this site identity as eight little-endian `u32` words.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 32] {
        let words = [
            self.family,
            self.invocation,
            self.level,
            self.stage,
            self.round,
            self.group,
            self.limb,
            self.detail,
        ];
        let mut encoded = [0u8; 32];
        let mut index = 0;
        while index < 32 {
            encoded[index] = words[index / 4].to_le_bytes()[index % 4];
            index += 1;
        }
        encoded
    }
}

/// Fixed-width diagnostic record identifying logical message and challenge groups.
///
/// Records are never absorbed. They describe the proof stream for logging and
/// structural tests only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProtocolContextRecord {
    /// Fixed domain for all Akita context records.
    pub domain: [u8; 32],
    /// Context-record format version.
    pub version: u32,
    /// Canonical protocol-site identity.
    pub site_id: [u8; 32],
    /// Message or challenge kind.
    pub kind: u32,
    /// Number of fixed-width atoms in the group.
    pub atom_count: u64,
    /// Total canonical encoded bytes in the group.
    pub encoded_bytes: u64,
    /// Number of challenge bytes squeezed after the record.
    pub challenge_bytes: u64,
}

impl ProtocolContextRecord {
    /// Construct a versioned context record.
    #[must_use]
    pub const fn new(
        site_id: [u8; 32],
        kind: u32,
        atom_count: u64,
        encoded_bytes: u64,
        challenge_bytes: u64,
    ) -> Self {
        Self {
            domain: CONTEXT_DOMAIN,
            version: PROTOCOL_VERSION,
            site_id,
            kind,
            atom_count,
            encoded_bytes,
            challenge_bytes,
        }
    }
}

fn session_id() -> SessionId {
    spongefish::derive_session_id::<Shake128>(&SESSION_TAG)
}

/// Spongefish instance: the caller's session bytes, then the public statement
/// bytes, each absorbed as a `u64` little-endian length word, the bytes, then
/// zero padding to a word.
fn instance(session: &[u8], statement: &[u8]) -> (BytesMsg, BytesMsg) {
    (BytesMsg(session.to_vec()), BytesMsg(statement.to_vec()))
}

/// Construct a prover channel bound to [`SESSION_TAG`], `session` and `instance`.
#[must_use]
pub fn new_prover_channel(session: &[u8], statement: &[u8]) -> ProverChannel {
    ProverState::new(&session_id(), &instance(session, statement))
}

/// Construct a verifier channel over one proof byte string.
#[must_use]
pub fn new_verifier_channel<'proof>(
    session: &[u8],
    statement: &[u8],
    proof: &'proof [u8],
) -> VerifierChannel<'proof> {
    VerifierState::new(&session_id(), &instance(session, statement), proof)
}

/// Permanently reject a verifier transcript after a semantic protocol failure.
///
/// Every later read fails and `check_eof` fails, even if the caller drops the
/// returned error.
pub fn reject(state: &mut VerifierChannel<'_>) -> AkitaError {
    let _ = Transcript::check(state, || false);
    AkitaError::InvalidProof
}

const fn invalid_proof(_: VerificationError) -> AkitaError {
    AkitaError::InvalidProof
}

/// A fixed-width, canonical field atom for proof transport.
///
/// Runtime proof shapes are enforced by schedule-derived receive loops in the
/// protocol crates; this decoder only accepts canonical representatives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldAtom<F: CanonicalEncoding>(F);

impl<F: CanonicalEncoding> FieldAtom<F> {
    /// Wrap one field element for proof transport.
    #[must_use]
    pub const fn new(value: F) -> Self {
        Self(value)
    }

    /// Return the wrapped field element.
    #[must_use]
    pub const fn into_inner(self) -> F {
        self.0
    }
}

impl<F: CanonicalEncoding> Encoding for FieldAtom<F> {
    fn encode(&self) -> impl AsRef<[u8]> {
        AtomBytes::with_len(F::NUM_BYTES, |out| self.0.to_bytes_le(out))
    }
}

/// Encoded bytes of one proof atom, inline when they fit in 64 bytes.
///
/// Every received and public atom is encoded before it is absorbed;
/// writing the encoding into an inline, word-aligned buffer avoids
/// allocating it on the heap.
enum AtomBytes {
    Inline { bytes: InlineAtomBytes, len: usize },
    Heap(Vec<u8>),
}

#[repr(C, align(8))]
struct InlineAtomBytes([u8; 64]);

impl AtomBytes {
    fn with_len(len: usize, fill: impl FnOnce(&mut [u8])) -> Self {
        let mut bytes = InlineAtomBytes([0; 64]);
        if let Some(out) = bytes.0.get_mut(..len) {
            fill(out);
            Self::Inline { bytes, len }
        } else {
            let mut out = vec![0; len];
            fill(&mut out);
            Self::Heap(out)
        }
    }
}

impl AsRef<[u8]> for AtomBytes {
    fn as_ref(&self) -> &[u8] {
        match self {
            // `with_len` builds `Inline` only when `len` fits the buffer.
            Self::Inline { bytes, len } => bytes.0.get(..*len).unwrap_or_default(),
            Self::Heap(bytes) => bytes,
        }
    }
}

impl<F: CanonicalEncoding> FromNarg for FieldAtom<F> {
    fn from_narg(reader: &mut NargReader<'_>) -> Result<Self, VerificationError> {
        let encoded = reader.take(F::NUM_BYTES)?;
        F::from_bytes_le_checked(encoded)
            .map(Self)
            .ok_or(VerificationError)
    }
}

/// Canonical extension-field proof atom: its base coordinates in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtensionAtom<F, E> {
    value: E,
    _base: PhantomData<F>,
}

impl<F, E> ExtensionAtom<F, E> {
    /// Wrap one extension element for proof transport.
    #[must_use]
    pub const fn new(value: E) -> Self {
        Self {
            value,
            _base: PhantomData,
        }
    }

    /// Return the wrapped extension element.
    #[must_use]
    pub fn into_inner(self) -> E {
        self.value
    }
}

impl<F, E> Encoding for ExtensionAtom<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn encode(&self) -> impl AsRef<[u8]> {
        let coefficients = self.value.to_base_vec();
        AtomBytes::with_len(coefficients.len() * F::NUM_BYTES, |out| {
            for (coefficient, chunk) in coefficients.iter().zip(out.chunks_exact_mut(F::NUM_BYTES))
            {
                coefficient.to_bytes_le(chunk);
            }
        })
    }
}

impl<F, E> FromNarg for ExtensionAtom<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn from_narg(reader: &mut NargReader<'_>) -> Result<Self, VerificationError> {
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(E::DEGREE)
            .map_err(|_| VerificationError)?;
        for _ in 0..E::DEGREE {
            coefficients.push(reader.read::<FieldAtom<F>>()?.into_inner());
        }
        Ok(Self::new(E::from_base_slice(&coefficients)))
    }
}

/// Absorbed form of a message whose length bound data fixes: its wire bytes,
/// then zero padding to the next word.
fn word_padded(bytes: &[u8]) -> Vec<u8> {
    let mut padded = bytes.to_vec();
    padded.resize(bytes.len().next_multiple_of(WORD_BYTES), 0);
    padded
}

/// Absorbed form of a length only the prover knows: one `u64` little-endian
/// word.
fn length_word(len: u32) -> [u8; WORD_BYTES] {
    u64::from(len).to_le_bytes()
}

/// Concatenated wire encodings of `atoms`.
fn encode_atoms<T: Encoding>(atoms: impl IntoIterator<Item = T>) -> Vec<u8> {
    let mut wire = Vec::new();
    for atom in atoms {
        wire.extend_from_slice(atom.encode().as_ref());
    }
    wire
}

/// Emit `wire` as one proof message, absorbed zero-padded to a word.
fn send_padded(state: &mut ProverChannel, wire: &[u8]) {
    state.prover_message_with(wire, word_padded, |wire, out| out.extend_from_slice(wire));
}

/// Emit `atoms` as one proof message, absorbed zero-padded to a word.
fn send_atoms<T: Encoding>(state: &mut ProverChannel, atoms: impl IntoIterator<Item = T>) {
    send_padded(state, &encode_atoms(atoms));
}

/// Receive `count` canonical atoms sent by [`send_atoms`], rejecting a
/// noncanonical one. The count must come from bound data.
fn receive_atoms<T: Encoding + FromNarg + Copy>(
    state: &mut VerifierChannel<'_>,
    count: usize,
) -> Result<Vec<T>, AkitaError> {
    let mut atoms = Vec::new();
    if atoms.try_reserve_exact(count).is_err() {
        return Err(reject(state));
    }
    state
        .prover_message_with(
            move |reader| {
                for _ in 0..count {
                    atoms.push(T::from_narg(reader)?);
                }
                Ok(atoms)
            },
            |atoms| word_padded(&encode_atoms(atoms.iter().copied())),
        )
        .map_err(invalid_proof)
}

fn group_record(
    site: ProtocolSiteId,
    kind: ProtocolMessageKind,
    atom_count: Option<usize>,
    atom_bytes: usize,
) -> Result<ProtocolContextRecord, AkitaError> {
    let overflow = || AkitaError::InvalidInput("proof group record size overflows usize".into());
    let atom_count = atom_count.ok_or_else(overflow)?;
    let encoded_bytes = atom_count.checked_mul(atom_bytes).ok_or_else(overflow)?;
    Ok(ProtocolContextRecord::new(
        site.to_bytes(),
        kind as u32,
        usize_to_u64(atom_count, "proof group atom count")?,
        usize_to_u64(encoded_bytes, "proof group encoded byte count")?,
        0,
    ))
}

/// Emit a schedule-bounded byte sequence as one labelled proof group.
pub fn send_byte_group(
    state: &mut ProverChannel,
    site: ProtocolSiteId,
    bytes: &[u8],
) -> Result<(), AkitaError> {
    prover_context(
        state,
        group_record(site, ProtocolMessageKind::ProofAtoms, Some(bytes.len()), 1)?,
    );
    send_padded(state, bytes);
    Ok(())
}

/// Receive an exact schedule-bounded labelled proof group.
pub fn receive_byte_group(
    state: &mut VerifierChannel<'_>,
    site: ProtocolSiteId,
    len: usize,
) -> Result<Vec<u8>, AkitaError> {
    let record = group_record(site, ProtocolMessageKind::ProofAtoms, Some(len), 1);
    let record = state.guard(record)?;
    verifier_context(state, record);
    state
        .prover_message_with(
            |reader| reader.take(len).map(<[u8]>::to_vec),
            |bytes| word_padded(bytes),
        )
        .map_err(invalid_proof)
}

/// Length-atom record and payload site of one bounded byte payload at `site`.
fn bounded_bytes_sites(site: ProtocolSiteId) -> (ProtocolContextRecord, ProtocolSiteId) {
    let length_site = ProtocolSiteId { stage: 0, ..site };
    let length_record = ProtocolContextRecord::new(
        length_site.to_bytes(),
        ProtocolMessageKind::ProofLength as u32,
        1,
        4,
        0,
    );
    (length_record, ProtocolSiteId { stage: 1, ..site })
}

/// Emit a bounded variable byte payload: a `u32` length on the wire, absorbed
/// as one `u64` word, then the payload as one byte group.
pub fn send_bounded_bytes(
    state: &mut ProverChannel,
    site: ProtocolSiteId,
    bytes: &[u8],
    max_len: usize,
) -> Result<(), AkitaError> {
    if bytes.len() > max_len {
        return Err(AkitaError::InvalidInput(
            "bounded proof payload length exceeds its scheduled maximum".into(),
        ));
    }
    let len = u32::try_from(bytes.len()).map_err(|_| {
        AkitaError::InvalidInput("bounded proof payload length does not fit u32".into())
    })?;
    let (length_record, payload_site) = bounded_bytes_sites(site);
    prover_context(state, length_record);
    state.prover_message_with(
        &len,
        |len| length_word(*len),
        |len, out| out.extend_from_slice(&len.to_le_bytes()),
    );
    send_byte_group(state, payload_site, bytes)
}

/// Receive a bounded variable byte payload, validating its length atom
/// before allocating its body.
pub fn receive_bounded_bytes(
    state: &mut VerifierChannel<'_>,
    site: ProtocolSiteId,
    max_len: usize,
) -> Result<Vec<u8>, AkitaError> {
    let (length_record, payload_site) = bounded_bytes_sites(site);
    verifier_context(state, length_record);
    let len = state
        .prover_message_with(u32::from_narg, |len| length_word(*len))
        .map_err(invalid_proof)?;
    match usize::try_from(len) {
        Ok(len) if len <= max_len => receive_byte_group(state, payload_site, len),
        _ => Err(reject(state)),
    }
}

/// Absorb one public byte string and record its diagnostic site.
pub fn public_bytes<S: ProofChannel>(
    state: &mut S,
    site: ProtocolSiteId,
    bytes: &[u8],
) -> Result<(), AkitaError> {
    let record = group_record(site, ProtocolMessageKind::PublicValue, Some(bytes.len()), 1);
    let record = state.guard(record)?;
    state.context(record);
    state.public(bytes);
    Ok(())
}

/// Absorb a fixed-count group of public base-field values.
pub fn public_fields<F, S>(
    state: &mut S,
    site: ProtocolSiteId,
    values: &[F],
) -> Result<(), AkitaError>
where
    F: CanonicalEncoding,
    S: ProofChannel,
{
    let record = group_record(
        site,
        ProtocolMessageKind::PublicValue,
        Some(values.len()),
        F::NUM_BYTES,
    );
    let record = state.guard(record)?;
    state.context(record);
    state.public(&encode_atoms(
        values.iter().map(|&value| FieldAtom::new(value)),
    ));
    Ok(())
}

/// Absorb a fixed-count group of public extension-field values.
pub fn public_extensions<F, E, S>(
    state: &mut S,
    site: ProtocolSiteId,
    values: &[E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ProofChannel,
{
    let record = group_record(
        site,
        ProtocolMessageKind::PublicValue,
        values.len().checked_mul(E::DEGREE),
        F::NUM_BYTES,
    );
    let record = state.guard(record)?;
    state.context(record);
    state.public(&encode_atoms(
        values
            .iter()
            .map(|&value| ExtensionAtom::<F, E>::new(value)),
    ));
    Ok(())
}

/// Emit a fixed-count group of canonical base-field proof atoms.
pub fn send_field_group<F: CanonicalEncoding>(
    state: &mut ProverChannel,
    site: ProtocolSiteId,
    values: &[F],
) -> Result<(), AkitaError> {
    prover_context(
        state,
        group_record(
            site,
            ProtocolMessageKind::ProofAtoms,
            Some(values.len()),
            F::NUM_BYTES,
        )?,
    );
    send_atoms(state, values.iter().map(|&value| FieldAtom::new(value)));
    Ok(())
}

/// Receive a schedule-fixed group of canonical base-field proof atoms.
pub fn receive_field_group<F: CanonicalEncoding>(
    state: &mut VerifierChannel<'_>,
    site: ProtocolSiteId,
    value_count: usize,
) -> Result<Vec<F>, AkitaError> {
    let record = group_record(
        site,
        ProtocolMessageKind::ProofAtoms,
        Some(value_count),
        F::NUM_BYTES,
    );
    let record = state.guard(record)?;
    verifier_context(state, record);
    let atoms = receive_atoms::<FieldAtom<F>>(state, value_count)?;
    Ok(atoms.into_iter().map(FieldAtom::into_inner).collect())
}

/// Allocate `count` zeroed extension slots for an in-place exchange.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] when the allocation cannot be
/// reserved; a verifier transcript is rejected.
pub fn extension_slots<E: Field, S: ProofChannel>(
    state: &mut S,
    count: usize,
) -> Result<Vec<E>, AkitaError> {
    let mut values = Vec::new();
    let reserved = values
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidProof);
    state.guard(reserved)?;
    values.resize(count, E::zero());
    Ok(values)
}

/// Exchange a schedule-fixed group of extension-field proof values in place.
///
/// The prover emits `values`; the verifier overwrites them with the received
/// atoms. The group length is public and comes from the schedule.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidInput`] when the group record size overflows,
/// or [`AkitaError::InvalidProof`] when the verifier cannot decode an atom.
pub fn exchange_extension_group<F, E, S>(
    state: &mut S,
    site: ProtocolSiteId,
    values: &mut [E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    S: ProofChannel,
{
    let record = group_record(
        site,
        ProtocolMessageKind::ProofAtoms,
        values.len().checked_mul(E::DEGREE),
        F::NUM_BYTES,
    );
    let record = state.guard(record)?;
    state.context(record);
    let mut atoms: Vec<ExtensionAtom<F, E>> = values
        .iter()
        .map(|&value| ExtensionAtom::new(value))
        .collect();
    state.exchange(&mut atoms)?;
    for (value, atom) in values.iter_mut().zip(atoms) {
        *value = atom.into_inner();
    }
    Ok(())
}

/// Draw one challenge message of `count` field elements at `site`.
///
/// The whole message shares one bias budget: Jolt's codec for `E` squeezes a
/// fixed number of bytes for `count` and returns a joint draw within total
/// variation `2^-161` of uniform.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on the prover, or
/// [`AkitaError::InvalidProof`] on the verifier, when the codec does not
/// support `count` or its rejection pool is exhausted.
pub fn ext_challenges<E, S>(
    state: &mut S,
    site: ProtocolSiteId,
    count: usize,
) -> Result<Vec<E>, AkitaError>
where
    E: ChallengeCodec,
    S: ProofChannel,
{
    let challenge_bytes = E::challenge_bytes(count).unwrap_or(0);
    let challenge_bytes = usize_to_u64(challenge_bytes, "challenge message byte count");
    let challenge_bytes = state.guard(challenge_bytes)?;
    state.context(ProtocolContextRecord::new(
        site.to_bytes(),
        ProtocolMessageKind::Challenge as u32,
        0,
        0,
        challenge_bytes,
    ));
    state.challenges(count)
}

/// Draw one field element as its own challenge message at `site`.
///
/// # Errors
///
/// As for [`ext_challenges`].
pub fn ext_challenge<E, S>(state: &mut S, site: ProtocolSiteId) -> Result<E, AkitaError>
where
    E: ChallengeCodec,
    S: ProofChannel,
{
    let value = ext_challenges(state, site, 1)?
        .pop()
        .ok_or_else(|| AkitaError::Internal("challenge codec returned no element".into()));
    state.guard(value)
}

/// Record prover-side protocol metadata when transcript diagnostics are enabled.
#[inline(always)]
pub fn prover_context(state: &mut ProverChannel, record: ProtocolContextRecord) {
    #[cfg(feature = "logging-transcript")]
    {
        crate::logging::record_context(record);
        crate::logging::record_proof_boundary(record, state.narg_string().len());
    }
    #[cfg(not(feature = "logging-transcript"))]
    let _ = (state, record);
}

/// Close the final diagnostic proof range at the authoritative argument end.
#[cfg(feature = "logging-transcript")]
pub fn finish_proof_ranges(state: &ProverChannel) {
    crate::logging::finish_proof_ranges(state.narg_string().len());
}

/// Record verifier-side protocol metadata when transcript diagnostics are enabled.
#[inline(always)]
pub fn verifier_context(_state: &mut VerifierChannel<'_>, record: ProtocolContextRecord) {
    #[cfg(feature = "logging-transcript")]
    crate::logging::record_context(record);
    #[cfg(not(feature = "logging-transcript"))]
    let _ = record;
}

/// A prover-side copy of the public sponge for previewing a fold-response
/// candidate: the nonce, the fold's proof of work, then one challenge message
/// per commitment group.
///
/// Jolt's `verifier_message_oracle` evaluates only "absorb a candidate, then
/// squeeze", so it cannot express the work between the nonce and the
/// messages. The preview instead runs the same calls on a detached prover
/// state built from a copy of the sponge, so it cannot mutate the live
/// argument string or the prover's private randomness. The accepted
/// candidate's proof of work is carried to the live channel as a
/// [`FoldWork`], so the live prover does not search again. [`FoldWork::commit`]
/// first checks that the live sponge matches the preview's sponge from before
/// the work, so a preview that diverged from the channel fails loudly instead
/// of producing an invalid proof.
#[cfg(feature = "prover")]
pub struct FoldPreview {
    state: ProverChannel,
    nonce: u32,
    work: Option<FoldWork>,
}

/// The fold's proof of work as found on a [`FoldPreview`]: a fingerprint of
/// the sponge right before the work, the sponge right after it, and the
/// argument bytes the work sent.
#[cfg(feature = "prover")]
pub struct FoldWork {
    nonce: u32,
    start: [u8; 32],
    sponge: crate::TranscriptSponge,
    argument: Vec<u8>,
}

/// 32 bytes squeezed from a copy of `sponge`: equal sponges give equal
/// fingerprints, and different ones differ except with probability `2^-256`.
#[cfg(feature = "prover")]
fn sponge_fingerprint(sponge: &crate::TranscriptSponge) -> [u8; 32] {
    use spongefish::DuplexSpongeInterface;
    sponge.clone().squeeze_array()
}

#[cfg(feature = "prover")]
impl FoldWork {
    /// The fold-response nonce this work follows.
    #[must_use]
    pub const fn nonce(&self) -> u32 {
        self.nonce
    }

    /// Carry the work to the live channel, which has just sent the same
    /// fold-response nonce: check that the live sponge is the one the preview
    /// ground from, write the work's argument bytes without absorbing them,
    /// then take the preview's sponge, which already holds the work's nonce
    /// word and check.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::Internal`], leaving the channel unchanged, when
    /// the live sponge differs from the preview's.
    pub fn commit(self, state: &mut ProverChannel) -> Result<(), AkitaError> {
        if sponge_fingerprint(&state.duplex_sponge_state) != self.start {
            return Err(AkitaError::Internal(
                "live transcript diverged from the fold preview".into(),
            ));
        }
        state.prover_message_with(
            &self.argument,
            |_| [0u8; 0],
            |argument, out| out.extend_from_slice(argument),
        );
        state.duplex_sponge_state = self.sponge;
        Ok(())
    }
}

#[cfg(feature = "prover")]
impl FoldPreview {
    /// Copy the public state and absorb one candidate nonce's word, as the
    /// live channel will when it sends the nonce.
    #[must_use]
    pub fn new(state: &ProverChannel, nonce: u32) -> Self {
        // A fresh channel has an empty argument string; its sponge is then
        // replaced by the live one.
        let mut detached = new_prover_channel(&[], &[]);
        detached.duplex_sponge_state = state.duplex_sponge_state.clone();
        let mut state = detached;
        state.public_message_as(&NonceAtom::new(nonce), |nonce| nonce.word());
        Self {
            state,
            nonce,
            work: None,
        }
    }

    /// Run Jolt's `grind(bits)` on the detached state, exactly as the live
    /// prover will.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `bits` exceeds
    /// [`MAX_GRINDING_BITS`](crate::MAX_GRINDING_BITS) or the search finds no
    /// nonce (probability at most `e^-128` for an ideal sponge).
    pub fn grind(&mut self, bits: u32) -> Result<(), AkitaError> {
        if self.work.is_some() {
            return Err(AkitaError::Internal(
                "fold preview ran its proof of work twice".into(),
            ));
        }
        let start = sponge_fingerprint(&self.state.duplex_sponge_state);
        jolt_transcript::ProverTranscript::grind(&mut self.state, bits)
            .map_err(|_| AkitaError::InvalidSetup("fold preview proof of work failed".into()))?;
        self.work = Some(FoldWork {
            nonce: self.nonce,
            start,
            sponge: self.state.duplex_sponge_state.clone(),
            argument: self.state.narg_string().to_vec(),
        });
        Ok(())
    }

    /// Take the proof of work found by [`grind`](Self::grind), for
    /// [`FoldWork::commit`] on the live channel.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::Internal`] when `grind` has not run or the work
    /// was already taken.
    pub fn take_work(&mut self) -> Result<FoldWork, AkitaError> {
        self.work
            .take()
            .ok_or_else(|| AkitaError::Internal("fold preview has no proof of work".into()))
    }

    /// Squeeze a challenge message of `len` bytes, continuing the squeeze run,
    /// and decode it.
    ///
    /// # Errors
    ///
    /// Returns the error of `decode`.
    pub fn challenge_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<T, AkitaError>,
    ) -> Result<T, AkitaError> {
        self.state.verifier_message_as(len, decode)
    }
}

#[cfg(test)]
mod atom_encoding_tests;
#[cfg(test)]
mod tests;
