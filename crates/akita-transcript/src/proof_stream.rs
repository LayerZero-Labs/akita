//! Session construction, canonical Akita proof atoms and diagnostic site records.

use akita_error::{narrowing::usize_to_u64, AkitaError};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use jolt_transcript::{BytesMsg, ChallengeCodec, Shake128};
use spongefish::{
    Encoding, FromNarg, NargReader, ProverState, SessionId, Transcript, VerificationError,
    VerifierState,
};
use std::marker::PhantomData;

use crate::{ProverChannel, VerifierChannel, FOLD_CHALLENGE_SEED_LEN};

mod channel;
mod nonce;
mod site;
pub use channel::ProofChannel;
pub use nonce::{nonce_encoded_len, nonce_max_bytes, NonceAtom};

/// Proof channel and proof-stream format version.
pub const PROTOCOL_VERSION: u32 = 8;

/// Spongefish session tag.
///
/// It names the proof-stream version, the sponge, the field-challenge codec
/// and the proof-of-work layout, so a change to any of them changes every
/// challenge.
#[cfg(feature = "transcript-blake2b")]
pub const SESSION_TAG: &[u8] =
    b"akita-pcs/native-proof-stream/v8/blake2b512/jolt-pooled-rejection-tv161/jolt-sponge-pow-leb128";

/// Spongefish session tag.
///
/// It names the proof-stream version, the sponge, the field-challenge codec
/// and the proof-of-work layout, so a change to any of them changes every
/// challenge.
#[cfg(feature = "transcript-keccak")]
pub const SESSION_TAG: &[u8] =
    b"akita-pcs/native-proof-stream/v8/keccak-f1600/jolt-pooled-rejection-tv161/jolt-sponge-pow-leb128";

/// Domain tag stored in every diagnostic context record.
pub const CONTEXT_DOMAIN: [u8; 32] = *b"akita-pcs/native-context/v8\0\0\0\0\0";

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
    spongefish::derive_session_id::<Shake128>(SESSION_TAG)
}

/// Spongefish instance: the caller's session bytes, then the public statement
/// bytes, each with a `u64` little-endian length prefix.
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

/// Emit one canonical field atom and absorb the same bytes.
pub fn send_field<F: CanonicalEncoding>(state: &mut ProverChannel, value: F) {
    state.prover_message(&FieldAtom::new(value));
}

/// Receive one canonical field atom, rejecting noncanonical representatives.
pub fn receive_field<F: CanonicalEncoding>(
    state: &mut VerifierChannel<'_>,
) -> Result<F, AkitaError> {
    state
        .prover_message::<FieldAtom<F>>()
        .map(FieldAtom::into_inner)
        .map_err(invalid_proof)
}

/// Emit one extension-field proof atom as ordered canonical base coordinates.
pub fn send_extension<F, E>(state: &mut ProverChannel, value: E)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    state.prover_message(&ExtensionAtom::<F, E>::new(value));
}

/// Receive one extension-field proof atom from canonical base coordinates.
pub fn receive_extension<F, E>(state: &mut VerifierChannel<'_>) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    state
        .prover_message::<ExtensionAtom<F, E>>()
        .map(ExtensionAtom::into_inner)
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
    state.prover_message_as(bytes, |bytes| bytes);
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
        .prover_message_as(|reader| reader.take(len).map(<[u8]>::to_vec))
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

/// Emit a bounded variable byte payload with a `u32` length atom.
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
    state.prover_message(&len);
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
    let len = state.prover_message::<u32>().map_err(invalid_proof)?;
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
    for &value in values {
        state.public(&value.to_bytes_le_vec());
    }
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
    for value in values {
        state.public(ExtensionAtom::<F, E>::new(*value).encode().as_ref());
    }
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
    for &value in values {
        send_field(state, value);
    }
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
    let mut values = Vec::new();
    if values.try_reserve_exact(value_count).is_err() {
        return Err(reject(state));
    }
    for _ in 0..value_count {
        values.push(receive_field(state)?);
    }
    Ok(values)
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
    for value in values {
        let mut atom = ExtensionAtom::<F, E>::new(*value);
        state.exchange(&mut atom)?;
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
/// candidate: the nonce, then for each commitment group a public payload and
/// its squeezed root.
///
/// Jolt's `MessageSearch` probes only "absorb a candidate, then squeeze", so
/// it cannot express the payload absorbed between the nonce and each root.
/// The copy cannot mutate the live argument string or the prover's private
/// randomness.
#[cfg(feature = "prover")]
pub struct FoldPreview {
    sponge: crate::TranscriptSponge,
}

#[cfg(feature = "prover")]
impl FoldPreview {
    /// Copy the public state and absorb one candidate nonce message.
    #[must_use]
    pub fn new(state: &ProverChannel, nonce: u32) -> Self {
        use spongefish::DuplexSpongeInterface;
        let mut sponge = state.duplex_sponge_state.clone();
        let _ = sponge.absorb(NonceAtom::new(nonce).encode().as_ref());
        Self { sponge }
    }

    /// Absorb one public fold payload and squeeze its root.
    #[must_use]
    pub fn fold_root(&mut self, payload: &[u8]) -> [u8; FOLD_CHALLENGE_SEED_LEN] {
        use spongefish::DuplexSpongeInterface;
        let _ = self.sponge.absorb(payload);
        let mut root = [0u8; FOLD_CHALLENGE_SEED_LEN];
        let _ = self.sponge.squeeze(&mut root);
        root
    }
}

/// Absorb one public fold payload and draw its root.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] when the verifier transcript is
/// already rejected.
pub fn fold_root<S: ProofChannel>(
    state: &mut S,
    record: ProtocolContextRecord,
    payload: &[u8],
) -> Result<[u8; FOLD_CHALLENGE_SEED_LEN], AkitaError> {
    state.context(record);
    state.public(payload);
    state.seed()
}

#[cfg(test)]
mod atom_encoding_tests;
#[cfg(test)]
mod tests;
