//! Native Spongefish state construction and canonical Akita message codecs.

use jolt_field::{CanonicalEncoding, ExtField, Field};
use spongefish::{
    protocol_id, DomainSeparator, DuplexSpongeInterface, Encoding, NargDeserialize, ProverState,
    VerificationError, WithoutInstance,
};
use std::marker::PhantomData;
use std::{error::Error, fmt};

use crate::TranscriptSponge;

mod nonce;
pub use nonce::{native_nonce_encoded_len, native_nonce_max_bytes, NativeNonce};
mod sampling;
pub use sampling::{
    native_field_challenge_bytes, native_field_sampling_is_certified,
    native_prover_field_challenge, native_verifier_field_challenge, NATIVE_FIELD_CHALLENGE_BYTES,
    NATIVE_FIELD_SAMPLING_QUERY_LIMIT,
};
mod verifier;
pub use verifier::NativeVerifierState;
mod site;

/// Native transcript and proof-stream format version.
pub const NATIVE_PROTOCOL_VERSION: u32 = 7;

/// Domain tag stored in every native diagnostic context record.
pub const NATIVE_CONTEXT_DOMAIN: [u8; 32] = *b"akita-pcs/native-context/v7\0\0\0\0\0";

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

/// Native proof-stream operation kind recorded by diagnostic context metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ProtocolMessageKind {
    /// A public or derived value supplied outside the argument string.
    PublicValue = 1,
    /// A bounded variable payload's native length atom.
    ProofLength = 2,
    /// One or more canonical proof atoms.
    ProofAtoms = 3,
    /// A verifier challenge group.
    Challenge = 4,
    /// A proof-of-work nonce atom.
    GrindingNonce = 5,
    /// A proof-of-work predicate challenge.
    GrindingPredicate = 6,
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
    /// Base-field limb within an extension challenge.
    pub limb: u32,
    /// Family-specific public discriminator, such as grinding widths.
    pub detail: u32,
}

impl ProtocolSiteId {
    /// Encode this site identity in its fixed canonical layout.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 32] {
        let mut encoded = [0u8; 32];
        let family = self.family.to_le_bytes();
        let invocation = self.invocation.to_le_bytes();
        let level = self.level.to_le_bytes();
        let stage = self.stage.to_le_bytes();
        let round = self.round.to_le_bytes();
        let group = self.group.to_le_bytes();
        let limb = self.limb.to_le_bytes();
        let detail = self.detail.to_le_bytes();
        encoded[0] = family[0];
        encoded[1] = family[1];
        encoded[2] = family[2];
        encoded[3] = family[3];
        encoded[4] = invocation[0];
        encoded[5] = invocation[1];
        encoded[6] = invocation[2];
        encoded[7] = invocation[3];
        encoded[8] = level[0];
        encoded[9] = level[1];
        encoded[10] = level[2];
        encoded[11] = level[3];
        encoded[12] = stage[0];
        encoded[13] = stage[1];
        encoded[14] = stage[2];
        encoded[15] = stage[3];
        encoded[16] = round[0];
        encoded[17] = round[1];
        encoded[18] = round[2];
        encoded[19] = round[3];
        encoded[20] = group[0];
        encoded[21] = group[1];
        encoded[22] = group[2];
        encoded[23] = group[3];
        encoded[24] = limb[0];
        encoded[25] = limb[1];
        encoded[26] = limb[2];
        encoded[27] = limb[3];
        encoded[28] = detail[0];
        encoded[29] = detail[1];
        encoded[30] = detail[2];
        encoded[31] = detail[3];
        encoded
    }
}

/// Native Spongefish prover state used by Akita.
pub type NativeProverState = ProverState<TranscriptSponge>;

/// Failure to construct a native transcript from an unrepresentable public input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeInitializationError;

impl fmt::Display for NativeInitializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("native transcript input length exceeds u64")
    }
}

impl Error for NativeInitializationError {}

/// A public native context cannot be represented by the fixed site grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeContextError;

impl fmt::Display for NativeContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("native protocol value is unsupported or cannot be represented safely")
    }
}

impl Error for NativeContextError {}

#[derive(Clone, Copy)]
struct FramedBytes<'a> {
    bytes: &'a [u8],
    len: u64,
}

impl<'a> FramedBytes<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, NativeInitializationError> {
        Ok(Self {
            bytes,
            len: u64::try_from(bytes.len()).map_err(|_| NativeInitializationError)?,
        })
    }
}

impl Encoding<[u8]> for FramedBytes<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = Vec::with_capacity(8 + self.bytes.len());
        out.extend_from_slice(&self.len.to_le_bytes());
        out.extend_from_slice(self.bytes);
        out
    }
}

fn native_protocol_id() -> [u8; 64] {
    #[cfg(feature = "transcript-blake2b")]
    let name = "akita-pcs/native-proof-stream/v7/blake2b";
    #[cfg(feature = "transcript-keccak")]
    let name = "akita-pcs/native-proof-stream/v7/keccak";
    protocol_id(format_args!("{name}"))
}

fn native_domain<'a>(
    session: &'a [u8],
    instance: &'a [u8],
) -> Result<
    DomainSeparator<
        spongefish::WithInstance<FramedBytes<'a>>,
        spongefish::WithSession<FramedBytes<'a>>,
    >,
    NativeInitializationError,
> {
    Ok(
        DomainSeparator::<WithoutInstance>::new(native_protocol_id())
            .session(FramedBytes::new(session)?)
            .instance(FramedBytes::new(instance)?),
    )
}

/// Construct a bound native prover state.
pub fn new_native_prover(
    session: &[u8],
    instance: &[u8],
) -> Result<NativeProverState, NativeInitializationError> {
    Ok(native_domain(session, instance)?.to_prover(TranscriptSponge::default()))
}

/// Construct a bound native verifier state over one proof byte string.
pub fn new_native_verifier<'proof>(
    session: &[u8],
    instance: &[u8],
    proof: &'proof [u8],
) -> Result<NativeVerifierState<'proof>, NativeInitializationError> {
    Ok(NativeVerifierState::new(
        native_domain(session, instance)?.to_verifier(TranscriptSponge::default(), proof),
    ))
}

/// A fixed-width, canonical field atom for native proof transport.
///
/// This is deliberately an atom rather than a shape-aware container. Runtime
/// proof shapes are enforced by schedule-derived receive loops in the protocol
/// crates, while this decoder only accepts canonical field representatives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeField<F: CanonicalEncoding>(F);

impl<F: CanonicalEncoding> NativeField<F> {
    /// Wrap one field element for native proof transport.
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

/// Canonical extension-field proof atom with transactional decoding.
///
/// All base coordinates are decoded against a local cursor. The caller's
/// cursor advances only after every coordinate is present and canonical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeExtension<F, E> {
    value: E,
    _base: PhantomData<F>,
}

impl<F, E> NativeExtension<F, E> {
    /// Wrap one extension element for native proof transport.
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

impl<F, E> Encoding<[u8]> for NativeExtension<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = Vec::new();
        for coefficient in self.value.to_base_vec() {
            out.extend_from_slice(NativeField::new(coefficient).encode().as_ref());
        }
        out
    }
}

impl<F, E> NargDeserialize for NativeExtension<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    fn deserialize_from_narg(buf: &mut &[u8]) -> Result<Self, VerificationError> {
        let mut remaining = *buf;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(E::DEGREE)
            .map_err(|_| VerificationError)?;
        for _ in 0..E::DEGREE {
            coefficients
                .push(NativeField::<F>::deserialize_from_narg(&mut remaining)?.into_inner());
        }
        let value = E::from_base_slice(&coefficients);
        *buf = remaining;
        Ok(Self::new(value))
    }
}

impl<F: CanonicalEncoding> Encoding<[u8]> for NativeField<F> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0.to_bytes_le_vec()
    }
}

impl<F: CanonicalEncoding> NargDeserialize for NativeField<F> {
    fn deserialize_from_narg(buf: &mut &[u8]) -> Result<Self, VerificationError> {
        let (encoded, remaining) = buf
            .split_at_checked(F::NUM_BYTES)
            .ok_or(VerificationError)?;
        let value = F::from_bytes_le_checked(encoded).ok_or(VerificationError)?;
        *buf = remaining;
        Ok(Self(value))
    }
}

/// Fixed-width little-endian `u128` proof atom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeU128(u128);

impl NativeU128 {
    /// Wrap an integer for native proof transport.
    #[must_use]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Return the wrapped integer.
    #[must_use]
    pub const fn into_inner(self) -> u128 {
        self.0
    }
}

impl Encoding<[u8]> for NativeU128 {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0.to_le_bytes()
    }
}

impl NargDeserialize for NativeU128 {
    fn deserialize_from_narg(buf: &mut &[u8]) -> Result<Self, VerificationError> {
        let (encoded, remaining) = buf.split_at_checked(16).ok_or(VerificationError)?;
        let bytes: [u8; 16] = encoded.try_into().map_err(|_| VerificationError)?;
        *buf = remaining;
        Ok(Self(u128::from_le_bytes(bytes)))
    }
}

/// Emit one canonical field atom and absorb the same bytes.
pub fn send_native_field<F: CanonicalEncoding>(state: &mut NativeProverState, value: F) {
    state.prover_message(&NativeField::new(value));
}

/// Receive one canonical field atom, rejecting noncanonical representatives.
pub fn receive_native_field<F: CanonicalEncoding>(
    state: &mut NativeVerifierState<'_>,
) -> Result<F, VerificationError> {
    state
        .prover_message::<NativeField<F>>()
        .map(NativeField::into_inner)
}

/// Emit one extension-field proof atom as ordered canonical base coordinates.
pub fn send_native_extension<F, E>(state: &mut NativeProverState, value: E)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    state.prover_message(&NativeExtension::<F, E>::new(value));
}

/// Receive one extension-field proof atom from canonical base coordinates.
pub fn receive_native_extension<F, E>(
    state: &mut NativeVerifierState<'_>,
) -> Result<E, VerificationError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    state
        .prover_message::<NativeExtension<F, E>>()
        .map(NativeExtension::into_inner)
}

const NATIVE_BYTE_CHUNK_BYTES: usize = 1024;
const NATIVE_BYTE_TAIL_CHUNK_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NativeByteChunk<const N: usize>([u8; N]);

impl<const N: usize> Encoding<[u8]> for NativeByteChunk<N> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0.as_slice()
    }
}

impl<const N: usize> NargDeserialize for NativeByteChunk<N> {
    fn deserialize_from_narg(buf: &mut &[u8]) -> Result<Self, VerificationError> {
        let (encoded, remaining) = buf.split_at_checked(N).ok_or(VerificationError)?;
        let bytes = encoded.try_into().map_err(|_| VerificationError)?;
        *buf = remaining;
        Ok(Self(bytes))
    }
}

fn send_native_byte_chunks<'a, const N: usize>(
    state: &mut NativeProverState,
    mut bytes: &'a [u8],
) -> &'a [u8] {
    while bytes.len() >= N {
        let (chunk, remaining) = bytes.split_at(N);
        let chunk: NativeByteChunk<N> =
            NativeByteChunk(chunk.try_into().expect("chunk length is fixed"));
        state.prover_message(&chunk);
        bytes = remaining;
    }
    bytes
}

fn receive_native_byte_chunks<const N: usize>(
    state: &mut NativeVerifierState<'_>,
    len: usize,
    bytes: &mut Vec<u8>,
) -> Result<usize, VerificationError> {
    for _ in 0..len / N {
        let chunk = state.prover_message::<NativeByteChunk<N>>()?;
        bytes.extend_from_slice(&chunk.0);
    }
    Ok(len % N)
}

/// Emit a schedule-bounded byte sequence in fixed-size native chunks.
///
/// Spongefish absorption is associative, so this emits and absorbs exactly the
/// same byte string as one-byte messages while avoiding one call per byte.
pub fn send_native_bytes(state: &mut NativeProverState, bytes: &[u8]) {
    let bytes = send_native_byte_chunks::<NATIVE_BYTE_CHUNK_BYTES>(state, bytes);
    let bytes = send_native_byte_chunks::<NATIVE_BYTE_TAIL_CHUNK_BYTES>(state, bytes);
    for &byte in bytes {
        state.prover_message(&[byte]);
    }
}

/// Receive an exact schedule-bounded number of bytes in fixed-size native chunks.
pub fn receive_native_bytes(
    state: &mut NativeVerifierState<'_>,
    len: usize,
) -> Result<Vec<u8>, VerificationError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| VerificationError)?;
    let remaining = receive_native_byte_chunks::<NATIVE_BYTE_CHUNK_BYTES>(state, len, &mut bytes)?;
    let remaining =
        receive_native_byte_chunks::<NATIVE_BYTE_TAIL_CHUNK_BYTES>(state, remaining, &mut bytes)?;
    for _ in 0..remaining {
        bytes.push(state.prover_message::<[u8; 1]>()?[0]);
    }
    Ok(bytes)
}

/// Emit a schedule-bounded byte sequence as one diagnostically labelled proof group.
pub fn send_native_byte_group(
    state: &mut NativeProverState,
    site: ProtocolSiteId,
    bytes: &[u8],
) -> Result<(), NativeContextError> {
    let len = u64::try_from(bytes.len()).map_err(|_| NativeContextError)?;
    prover_context(
        state,
        ProtocolContextRecord::new(
            site.to_bytes(),
            ProtocolMessageKind::ProofAtoms as u32,
            len,
            len,
            0,
        ),
    );
    send_native_bytes(state, bytes);
    Ok(())
}

/// Receive an exact schedule-bounded diagnostically labelled proof group.
pub fn receive_native_byte_group(
    state: &mut NativeVerifierState<'_>,
    site: ProtocolSiteId,
    len: usize,
) -> Result<Vec<u8>, VerificationError> {
    let len_u64 = u64::try_from(len).map_err(|_| VerificationError)?;
    verifier_context(
        state,
        ProtocolContextRecord::new(
            site.to_bytes(),
            ProtocolMessageKind::ProofAtoms as u32,
            len_u64,
            len_u64,
            0,
        ),
    );
    receive_native_bytes(state, len)
}

/// Emit a bounded variable byte payload with a native `u32` length atom.
pub fn send_native_bounded_bytes(
    state: &mut NativeProverState,
    site: ProtocolSiteId,
    bytes: &[u8],
    max_len: usize,
) -> Result<(), NativeContextError> {
    if bytes.len() > max_len {
        return Err(NativeContextError);
    }
    let len = u32::try_from(bytes.len()).map_err(|_| NativeContextError)?;
    let mut length_site = site;
    length_site.stage = 0;
    prover_context(
        state,
        ProtocolContextRecord::new(
            length_site.to_bytes(),
            ProtocolMessageKind::ProofLength as u32,
            1,
            4,
            0,
        ),
    );
    state.prover_message(&len);
    let mut payload_site = site;
    payload_site.stage = 1;
    send_native_byte_group(state, payload_site, bytes)
}

/// Receive a bounded variable byte payload after validating its native length
/// atom and before allocating its body.
pub fn receive_native_bounded_bytes(
    state: &mut NativeVerifierState<'_>,
    site: ProtocolSiteId,
    max_len: usize,
) -> Result<Vec<u8>, VerificationError> {
    let mut length_site = site;
    length_site.stage = 0;
    verifier_context(
        state,
        ProtocolContextRecord::new(
            length_site.to_bytes(),
            ProtocolMessageKind::ProofLength as u32,
            1,
            4,
            0,
        ),
    );
    let len = state.prover_message::<u32>()?;
    let len = usize::try_from(len).map_err(|_| {
        state.invalidate();
        VerificationError
    })?;
    if len > max_len {
        state.invalidate();
        return Err(VerificationError);
    }
    let mut payload_site = site;
    payload_site.stage = 1;
    receive_native_byte_group(state, payload_site, len)
}

fn public_bytes_record(
    site: ProtocolSiteId,
    len: usize,
) -> Result<ProtocolContextRecord, NativeContextError> {
    let len = u64::try_from(len).map_err(|_| NativeContextError)?;
    Ok(ProtocolContextRecord::new(
        site.to_bytes(),
        ProtocolMessageKind::PublicValue as u32,
        len,
        len,
        0,
    ))
}

/// Absorb one public byte string and record its diagnostic site on the prover side.
pub fn public_native_bytes_prover(
    state: &mut NativeProverState,
    site: ProtocolSiteId,
    bytes: &[u8],
) -> Result<(), NativeContextError> {
    prover_context(state, public_bytes_record(site, bytes.len())?);
    state.public_message(bytes);
    Ok(())
}

/// Absorb one public byte string and record its diagnostic site on the verifier side.
pub fn public_native_bytes_verifier(
    state: &mut NativeVerifierState<'_>,
    site: ProtocolSiteId,
    bytes: &[u8],
) -> Result<(), NativeContextError> {
    verifier_context(state, public_bytes_record(site, bytes.len())?);
    state.public_message(bytes);
    Ok(())
}

/// Draw a context-bound extension-field challenge on the prover side.
pub fn native_prover_ext_challenge<F, E>(
    state: &mut NativeProverState,
    site: ProtocolSiteId,
) -> Result<E, NativeContextError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(E::DEGREE)
        .map_err(|_| NativeContextError)?;
    for limb in 0..E::DEGREE {
        let mut limb_site = site;
        limb_site.limb = u32::try_from(limb).map_err(|_| NativeContextError)?;
        prover_context(
            state,
            ProtocolContextRecord::new(
                limb_site.to_bytes(),
                ProtocolMessageKind::Challenge as u32,
                0,
                0,
                native_field_challenge_bytes::<F>(),
            ),
        );
        coefficients.push(native_prover_field_challenge(state)?);
    }
    Ok(E::from_base_slice(&coefficients))
}

/// Draw a context-bound extension-field challenge on the verifier side.
pub fn native_verifier_ext_challenge<F, E>(
    state: &mut NativeVerifierState<'_>,
    site: ProtocolSiteId,
) -> Result<E, NativeContextError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(E::DEGREE)
        .map_err(|_| NativeContextError)?;
    for limb in 0..E::DEGREE {
        let mut limb_site = site;
        limb_site.limb = u32::try_from(limb).map_err(|_| NativeContextError)?;
        verifier_context(
            state,
            ProtocolContextRecord::new(
                limb_site.to_bytes(),
                ProtocolMessageKind::Challenge as u32,
                0,
                0,
                native_field_challenge_bytes::<F>(),
            ),
        );
        coefficients.push(native_verifier_field_challenge(state)?);
    }
    Ok(E::from_base_slice(&coefficients))
}

/// Fixed-width diagnostic record identifying logical message and challenge groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProtocolContextRecord {
    /// Fixed domain for all native Akita context records.
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
    /// Construct a versioned native context record.
    #[must_use]
    pub const fn new(
        site_id: [u8; 32],
        kind: u32,
        atom_count: u64,
        encoded_bytes: u64,
        challenge_bytes: u64,
    ) -> Self {
        Self {
            domain: NATIVE_CONTEXT_DOMAIN,
            version: NATIVE_PROTOCOL_VERSION,
            site_id,
            kind,
            atom_count,
            encoded_bytes,
            challenge_bytes,
        }
    }
}

/// Record prover-side protocol metadata when transcript diagnostics are enabled.
#[inline(always)]
pub fn prover_context(state: &mut NativeProverState, record: ProtocolContextRecord) {
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
pub fn finish_native_proof_ranges(state: &NativeProverState) {
    crate::logging::finish_proof_ranges(state.narg_string().len());
}

/// Record verifier-side protocol metadata when transcript diagnostics are enabled.
#[inline(always)]
pub fn verifier_context(_state: &mut NativeVerifierState<'_>, record: ProtocolContextRecord) {
    #[cfg(feature = "logging-transcript")]
    crate::logging::record_context(record);
    #[cfg(not(feature = "logging-transcript"))]
    let _ = record;
}

/// A prover-side clone of only the public duplex state used to preview a
/// fold-response candidate.
///
/// The type deliberately exposes only the fold transition needed by Akita. It
/// cannot mutate the live argument string, the prover's private RNG, or a
/// grinding-plan cursor.
pub struct NativeFoldPreview {
    sponge: TranscriptSponge,
}

impl NativeFoldPreview {
    /// Clone the public state and absorb one candidate native nonce message.
    #[must_use]
    pub fn new(state: &NativeProverState, nonce: u32) -> Self {
        let mut sponge = state.duplex_sponge_state.clone();
        sponge.absorb(NativeNonce::new(nonce).encode().as_ref());
        Self { sponge }
    }

    /// Absorb one public fold payload and squeeze its root.
    #[must_use]
    pub fn fold_root(&mut self, payload: &[u8]) -> [u8; crate::FOLD_CHALLENGE_SEED_LEN] {
        self.sponge.absorb(payload);
        let mut root = [0u8; crate::FOLD_CHALLENGE_SEED_LEN];
        self.sponge.squeeze(&mut root);
        root
    }
}

/// Absorb one public fold payload and draw its root live on the
/// prover side.
#[must_use]
pub fn native_prover_fold_root(
    state: &mut NativeProverState,
    record: ProtocolContextRecord,
    payload: &[u8],
) -> [u8; crate::FOLD_CHALLENGE_SEED_LEN] {
    prover_context(state, record);
    state.public_message(payload);
    state.verifier_message()
}

/// Absorb one public fold payload and draw its root live on the
/// verifier side.
pub fn native_verifier_fold_root(
    state: &mut NativeVerifierState<'_>,
    record: ProtocolContextRecord,
    payload: &[u8],
) -> Result<[u8; crate::FOLD_CHALLENGE_SEED_LEN], VerificationError> {
    verifier_context(state, record);
    state.public_message(payload);
    state.verifier_message()
}

fn extension_group_record<F, E>(
    site: ProtocolSiteId,
    kind: ProtocolMessageKind,
    value_count: usize,
) -> Result<ProtocolContextRecord, NativeContextError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let atom_count = value_count
        .checked_mul(E::DEGREE)
        .ok_or(NativeContextError)?;
    let encoded_bytes = atom_count
        .checked_mul(F::NUM_BYTES)
        .ok_or(NativeContextError)?;
    Ok(ProtocolContextRecord::new(
        site.to_bytes(),
        kind as u32,
        u64::try_from(atom_count).map_err(|_| NativeContextError)?,
        u64::try_from(encoded_bytes).map_err(|_| NativeContextError)?,
        0,
    ))
}

fn field_group_record<F>(
    site: ProtocolSiteId,
    kind: ProtocolMessageKind,
    value_count: usize,
) -> Result<ProtocolContextRecord, NativeContextError>
where
    F: CanonicalEncoding,
{
    let encoded_bytes = value_count
        .checked_mul(F::NUM_BYTES)
        .ok_or(NativeContextError)?;
    Ok(ProtocolContextRecord::new(
        site.to_bytes(),
        kind as u32,
        u64::try_from(value_count).map_err(|_| NativeContextError)?,
        u64::try_from(encoded_bytes).map_err(|_| NativeContextError)?,
        0,
    ))
}

/// Absorb a fixed-count group of public base-field values.
pub fn public_native_fields_prover<F>(
    state: &mut NativeProverState,
    site: ProtocolSiteId,
    values: &[F],
) -> Result<(), NativeContextError>
where
    F: CanonicalEncoding,
{
    prover_context(
        state,
        field_group_record::<F>(site, ProtocolMessageKind::PublicValue, values.len())?,
    );
    for &value in values {
        state.public_message(&NativeField::new(value));
    }
    Ok(())
}

/// Absorb a fixed-count group of public base-field values.
pub fn public_native_fields_verifier<F>(
    state: &mut NativeVerifierState<'_>,
    site: ProtocolSiteId,
    values: &[F],
) -> Result<(), NativeContextError>
where
    F: CanonicalEncoding,
{
    verifier_context(
        state,
        field_group_record::<F>(site, ProtocolMessageKind::PublicValue, values.len())?,
    );
    for &value in values {
        state.public_message(&NativeField::new(value));
    }
    Ok(())
}

/// Emit a fixed-count group of canonical base-field proof atoms.
pub fn send_native_field_group<F>(
    state: &mut NativeProverState,
    site: ProtocolSiteId,
    values: &[F],
) -> Result<(), NativeContextError>
where
    F: CanonicalEncoding,
{
    prover_context(
        state,
        field_group_record::<F>(site, ProtocolMessageKind::ProofAtoms, values.len())?,
    );
    for &value in values {
        send_native_field(state, value);
    }
    Ok(())
}

/// Receive a schedule-fixed group of canonical base-field proof atoms.
pub fn receive_native_field_group<F>(
    state: &mut NativeVerifierState<'_>,
    site: ProtocolSiteId,
    value_count: usize,
) -> Result<Vec<F>, VerificationError>
where
    F: CanonicalEncoding,
{
    let record = field_group_record::<F>(site, ProtocolMessageKind::ProofAtoms, value_count)
        .map_err(|_| VerificationError)?;
    verifier_context(state, record);
    let mut values = Vec::new();
    values
        .try_reserve_exact(value_count)
        .map_err(|_| VerificationError)?;
    for _ in 0..value_count {
        values.push(receive_native_field(state)?);
    }
    Ok(values)
}

/// Absorb a fixed-count group of public extension-field values.
pub fn public_native_extensions_prover<F, E>(
    state: &mut NativeProverState,
    site: ProtocolSiteId,
    values: &[E],
) -> Result<(), NativeContextError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    prover_context(
        state,
        extension_group_record::<F, E>(site, ProtocolMessageKind::PublicValue, values.len())?,
    );
    for value in values {
        for coefficient in value.to_base_vec() {
            state.public_message(&NativeField::new(coefficient));
        }
    }
    Ok(())
}

/// Absorb a fixed-count group of public extension-field values.
pub fn public_native_extensions_verifier<F, E>(
    state: &mut NativeVerifierState<'_>,
    site: ProtocolSiteId,
    values: &[E],
) -> Result<(), NativeContextError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    verifier_context(
        state,
        extension_group_record::<F, E>(site, ProtocolMessageKind::PublicValue, values.len())?,
    );
    for value in values {
        for coefficient in value.to_base_vec() {
            state.public_message(&NativeField::new(coefficient));
        }
    }
    Ok(())
}

/// Emit a fixed-count group of extension-field proof values.
pub fn send_native_extension_group<F, E>(
    state: &mut NativeProverState,
    site: ProtocolSiteId,
    values: &[E],
) -> Result<(), NativeContextError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    prover_context(
        state,
        extension_group_record::<F, E>(site, ProtocolMessageKind::ProofAtoms, values.len())?,
    );
    for &value in values {
        send_native_extension::<F, E>(state, value);
    }
    Ok(())
}

/// Receive a schedule-fixed group of extension-field proof values.
pub fn receive_native_extension_group<F, E>(
    state: &mut NativeVerifierState<'_>,
    site: ProtocolSiteId,
    value_count: usize,
) -> Result<Vec<E>, VerificationError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let record = extension_group_record::<F, E>(site, ProtocolMessageKind::ProofAtoms, value_count)
        .map_err(|_| VerificationError)?;
    verifier_context(state, record);
    let mut values = Vec::new();
    values
        .try_reserve_exact(value_count)
        .map_err(|_| VerificationError)?;
    for _ in 0..value_count {
        values.push(receive_native_extension::<F, E>(state)?);
    }
    Ok(values)
}

/// Preview the native predicate produced by a candidate grinding nonce.
///
/// Only the public duplex state is cloned. The live state, private prover RNG,
/// and argument string are not mutated.
#[must_use]
pub fn preview_native_grinding_predicate(
    state: &NativeProverState,
    nonce: u32,
) -> [u8; crate::GRINDING_PREDICATE_LEN] {
    let mut sponge = state.duplex_sponge_state.clone();
    sponge.absorb(NativeNonce::new(nonce).encode().as_ref());
    let mut predicate = [0u8; crate::GRINDING_PREDICATE_LEN];
    sponge.squeeze(&mut predicate);
    predicate
}

/// Search the canonical bounded nonce range against native Spongefish previews.
///
/// A zero-bit target is a no-op and does not absorb either record.
#[must_use]
pub fn search_native_grinding_nonce(
    state: &NativeProverState,
    grind_bits: u8,
    nonce_bits: u8,
) -> Option<(u32, [u8; crate::GRINDING_PREDICATE_LEN])> {
    crate::grinding::search_grinding_nonce_with(grind_bits, nonce_bits, |nonce| {
        Some(preview_native_grinding_predicate(state, nonce))
    })
}

/// Commit a winning grinding nonce and draw its native predicate.
#[must_use]
pub fn commit_native_grinding_nonce(
    state: &mut NativeProverState,
    nonce_record: ProtocolContextRecord,
    nonce: u32,
    predicate_record: ProtocolContextRecord,
) -> [u8; crate::GRINDING_PREDICATE_LEN] {
    prover_context(state, nonce_record);
    state.prover_message(&NativeNonce::new(nonce));
    prover_context(state, predicate_record);
    state.verifier_message()
}

/// Receive a grinding nonce and draw its native predicate.
pub fn receive_native_grinding_nonce(
    state: &mut NativeVerifierState<'_>,
    nonce_record: ProtocolContextRecord,
    predicate_record: ProtocolContextRecord,
) -> Result<(u32, [u8; crate::GRINDING_PREDICATE_LEN]), VerificationError> {
    verifier_context(state, nonce_record);
    let nonce = state.prover_message::<NativeNonce>()?.into_inner();
    verifier_context(state, predicate_record);
    Ok((nonce, state.verifier_message()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{
        CanonicalBytes, Prime128OffsetA7F7, Prime32Offset99 as F, Prime48Offset59, Prime64Offset59,
        Ring,
    };

    #[test]
    fn public_absorption_never_extends_the_argument_string() {
        let mut prover = new_native_prover(b"public-size", b"fixture").unwrap();
        prover.prover_message(&[7u8; 3]);
        let before = prover.narg_string().to_vec();
        prover.public_message(b"public fold payload");
        assert_eq!(prover.narg_string(), before);
    }

    #[cfg(feature = "transcript-blake2b")]
    #[test]
    fn blake2b_transcript_has_a_cross_width_known_answer() {
        let mut prover =
            new_native_prover(b"cross-width/session", b"cross-width/instance").unwrap();
        prover.public_message(b"public-message");
        let challenge = prover.verifier_message::<[u8; 32]>();
        assert_eq!(
            challenge,
            [
                174, 120, 165, 116, 108, 247, 146, 33, 182, 143, 40, 94, 150, 167, 244, 49, 71, 31,
                91, 26, 214, 49, 240, 75, 87, 142, 44, 99, 169, 37, 81, 104,
            ]
        );
    }

    #[test]
    fn exact_rejection_sampling_covers_every_production_field() {
        assert!(native_field_sampling_is_certified(
            F::NUM_BYTES,
            F::MODULUS_BITS,
            NATIVE_FIELD_SAMPLING_QUERY_LIMIT,
        ));
        assert!(native_field_sampling_is_certified(
            Prime64Offset59::NUM_BYTES,
            Prime64Offset59::MODULUS_BITS,
            NATIVE_FIELD_SAMPLING_QUERY_LIMIT,
        ));
        assert!(native_field_sampling_is_certified(
            Prime128OffsetA7F7::NUM_BYTES,
            Prime128OffsetA7F7::MODULUS_BITS,
            NATIVE_FIELD_SAMPLING_QUERY_LIMIT,
        ));
        assert_eq!(native_field_challenge_bytes::<F>(), 4);
        assert_eq!(native_field_challenge_bytes::<Prime64Offset59>(), 8);
        assert_eq!(native_field_challenge_bytes::<Prime128OffsetA7F7>(), 16);
    }

    #[test]
    fn exact_rejection_sampling_rejects_unsupported_metadata() {
        assert!(!native_field_sampling_is_certified(
            0,
            1,
            NATIVE_FIELD_SAMPLING_QUERY_LIMIT,
        ));
        assert!(!native_field_sampling_is_certified(
            NATIVE_FIELD_CHALLENGE_BYTES as usize + 1,
            1,
            NATIVE_FIELD_SAMPLING_QUERY_LIMIT,
        ));
        assert!(!native_field_sampling_is_certified(
            8,
            48,
            NATIVE_FIELD_SAMPLING_QUERY_LIMIT,
        ));
        assert!(!native_field_sampling_is_certified(
            8,
            65,
            NATIVE_FIELD_SAMPLING_QUERY_LIMIT,
        ));
        assert!(!native_field_sampling_is_certified(
            8,
            64,
            NATIVE_FIELD_SAMPLING_QUERY_LIMIT + 1,
        ));

        let mut prover = new_native_prover(b"unsupported-field", b"fixture").unwrap();
        assert!(native_prover_field_challenge::<Prime48Offset59>(&mut prover).is_err());
    }

    #[test]
    fn native_field_roundtrip_and_eof() {
        let mut prover = new_native_prover(b"session", b"instance").unwrap();
        send_native_field(&mut prover, F::from_u64(42));
        let proof = prover.narg_string().to_vec();

        let mut verifier = new_native_verifier(b"session", b"instance", &proof).unwrap();
        assert_eq!(
            receive_native_field::<F>(&mut verifier).unwrap(),
            F::from_u64(42)
        );
        assert!(verifier.check_eof().is_ok());
    }

    #[test]
    fn chunked_bytes_match_bytewise_proof_and_transcript() {
        for len in [0, 1, 63, 64, 65, 1023, 1024, 1025, 21_066] {
            let bytes = (0..len)
                .map(|index| (index as u8).wrapping_mul(29).wrapping_add(7))
                .collect::<Vec<_>>();
            let site = ProtocolSiteId {
                family: SITE_FAMILY_TERMINAL,
                level: 6,
                round: 3,
                ..ProtocolSiteId::default()
            };

            let mut chunked = new_native_prover(b"chunked-bytes", b"instance").unwrap();
            send_native_bytes(&mut chunked, &bytes);
            public_native_bytes_prover(&mut chunked, site, &bytes).unwrap();
            let chunked_challenge = chunked.verifier_message::<[u8; 32]>();

            let mut bytewise = new_native_prover(b"chunked-bytes", b"instance").unwrap();
            for &byte in &bytes {
                bytewise.prover_message(&[byte]);
            }
            prover_context(
                &mut bytewise,
                public_bytes_record(site, bytes.len()).unwrap(),
            );
            for &byte in &bytes {
                bytewise.public_message(&[byte]);
            }
            let bytewise_challenge = bytewise.verifier_message::<[u8; 32]>();

            assert_eq!(chunked.narg_string(), bytes);
            assert_eq!(chunked.narg_string(), bytewise.narg_string());
            assert_eq!(chunked_challenge, bytewise_challenge);

            let proof = chunked.narg_string().to_vec();
            let mut verifier = new_native_verifier(b"chunked-bytes", b"instance", &proof).unwrap();
            assert_eq!(receive_native_bytes(&mut verifier, len).unwrap(), bytes);
            public_native_bytes_verifier(&mut verifier, site, &bytes).unwrap();
            assert_eq!(
                verifier.verifier_message::<[u8; 32]>().unwrap(),
                chunked_challenge
            );
            verifier.check_eof().unwrap();
        }
    }

    #[test]
    fn native_extension_groups_share_context_and_fixed_shape() {
        type E = jolt_field::FpExt4<F>;

        let public = [E::from_u64(3), E::from_u64(5)];
        let private = [E::from_u64(8), E::from_u64(13)];
        let public_site = ProtocolSiteId {
            family: 27,
            stage: 1,
            ..ProtocolSiteId::default()
        };
        let private_site = ProtocolSiteId {
            family: 27,
            stage: 2,
            ..ProtocolSiteId::default()
        };
        let mut prover = new_native_prover(b"groups", b"fixture").unwrap();
        public_native_extensions_prover::<F, E>(&mut prover, public_site, &public).unwrap();
        send_native_extension_group::<F, E>(&mut prover, private_site, &private).unwrap();
        let prover_challenge = native_prover_ext_challenge::<F, E>(
            &mut prover,
            ProtocolSiteId {
                family: 27,
                stage: 3,
                ..ProtocolSiteId::default()
            },
        )
        .unwrap();
        let proof = prover.narg_string().to_vec();
        assert_eq!(proof.len(), private.len() * E::DEGREE * F::NUM_BYTES);

        let mut verifier = new_native_verifier(b"groups", b"fixture", &proof).unwrap();
        public_native_extensions_verifier::<F, E>(&mut verifier, public_site, &public).unwrap();
        assert_eq!(
            receive_native_extension_group::<F, E>(&mut verifier, private_site, private.len())
                .unwrap(),
            private
        );
        let verifier_challenge = native_verifier_ext_challenge::<F, E>(
            &mut verifier,
            ProtocolSiteId {
                family: 27,
                stage: 3,
                ..ProtocolSiteId::default()
            },
        )
        .unwrap();
        assert_eq!(verifier_challenge, prover_challenge);
        assert!(verifier.check_eof().is_ok());
    }

    #[test]
    fn native_fold_preview_matches_live_prover_and_verifier() {
        let nonce_site = ProtocolSiteId {
            family: SITE_FAMILY_FOLD_CHALLENGE,
            level: 3,
            detail: 12,
            ..ProtocolSiteId::default()
        };
        let nonce_record = ProtocolContextRecord::new(
            nonce_site.to_bytes(),
            ProtocolMessageKind::FoldResponseNonce as u32,
            1,
            native_nonce_max_bytes(12) as u64,
            0,
        );
        let root_record = ProtocolContextRecord::new(
            ProtocolSiteId {
                family: SITE_FAMILY_FOLD_CHALLENGE,
                level: 3,
                group: 2,
                ..ProtocolSiteId::default()
            }
            .to_bytes(),
            ProtocolMessageKind::Challenge as u32,
            3,
            3,
            crate::FOLD_CHALLENGE_SEED_LEN as u64,
        );
        let payload = [5, 8, 13];
        let nonce = 7u32;
        let mut prover = new_native_prover(b"fold-preview", b"fixture").unwrap();
        let preview = NativeFoldPreview::new(&prover, nonce).fold_root(&payload);
        prover_context(&mut prover, nonce_record);
        prover.prover_message(&NativeNonce::new(nonce));
        let live = native_prover_fold_root(&mut prover, root_record, &payload);
        assert_eq!(preview, live);
        let proof = prover.narg_string().to_vec();
        assert_eq!(proof.len(), 1);

        let mut verifier = new_native_verifier(b"fold-preview", b"fixture", &proof).unwrap();
        verifier_context(&mut verifier, nonce_record);
        assert_eq!(
            verifier
                .prover_message::<NativeNonce>()
                .unwrap()
                .into_inner(),
            nonce
        );
        assert_eq!(
            native_verifier_fold_root(&mut verifier, root_record, &payload).unwrap(),
            live
        );
        verifier.check_eof().unwrap();
    }

    #[test]
    fn native_field_failure_does_not_consume_cursor() {
        let noncanonical = vec![u8::MAX; F::NUM_BYTES];
        let mut bytes = noncanonical.as_slice();
        let original = bytes;
        assert!(NativeField::<F>::deserialize_from_narg(&mut bytes).is_err());
        assert_eq!(bytes, original);
    }

    #[test]
    fn native_extension_failure_does_not_consume_cursor() {
        type E = jolt_field::FpExt4<F>;

        let encoded = NativeExtension::<F, E>::new(E::from_u64(9))
            .encode()
            .as_ref()
            .to_vec();
        let truncated = &encoded[..encoded.len() - 1];
        let mut cursor = truncated;
        let original = cursor;
        assert!(NativeExtension::<F, E>::deserialize_from_narg(&mut cursor).is_err());
        assert_eq!(cursor, original);

        let mut noncanonical = encoded;
        noncanonical[F::NUM_BYTES..2 * F::NUM_BYTES].fill(u8::MAX);
        let mut cursor = noncanonical.as_slice();
        let original = cursor;
        assert!(NativeExtension::<F, E>::deserialize_from_narg(&mut cursor).is_err());
        assert_eq!(cursor, original);
    }

    #[test]
    fn ignored_bounded_receipt_failure_poisons_eof() {
        let site = ProtocolSiteId {
            family: SITE_FAMILY_TERMINAL,
            stage: 7,
            ..ProtocolSiteId::default()
        };
        let mut prover = new_native_prover(b"bounded", b"fixture").unwrap();
        prover.prover_message(&9u32);
        let proof = prover.narg_string().to_vec();

        let mut verifier = new_native_verifier(b"bounded", b"fixture", &proof).unwrap();
        assert!(receive_native_bounded_bytes(&mut verifier, site, 8).is_err());
        assert!(verifier.verifier_message::<[u8; 32]>().is_err());
        assert!(verifier.prover_message::<u32>().is_err());
        assert!(verifier.check_eof().is_err());
    }

    #[test]
    fn session_and_instance_are_independent_domains() {
        let mut left = new_native_prover(b"session-a", b"instance").unwrap();
        let mut right = new_native_prover(b"session-b", b"instance").unwrap();
        assert_ne!(
            left.verifier_message::<[u8; 32]>(),
            right.verifier_message::<[u8; 32]>()
        );
    }

    #[cfg(feature = "transcript-keccak")]
    #[test]
    fn rejection_attempt_markers_prevent_keccak_reconvergence() {
        let mut short = new_native_prover(b"width", b"substrate").unwrap();
        let mut long = new_native_prover(b"width", b"substrate").unwrap();
        let _: [u8; 1] = short.verifier_message();
        let _: [u8; 2] = long.verifier_message();
        short.public_message(&[17u8]);
        long.public_message(&[17u8]);
        assert_eq!(
            short.verifier_message::<[u8; 32]>(),
            long.verifier_message::<[u8; 32]>(),
            "the Keccak substrate forgets distinct nonzero squeeze widths after absorb",
        );

        let mut framed_short = new_native_prover(b"width", b"framed").unwrap();
        let mut framed_long = new_native_prover(b"width", b"framed").unwrap();
        let _: [u8; 1] = framed_short.verifier_message();
        let _: [u8; 2] = framed_long.verifier_message();
        framed_short.public_message(&1u32);
        framed_long.public_message(&2u32);
        framed_short.public_message(&[17u8]);
        framed_long.public_message(&[17u8]);
        assert_ne!(
            framed_short.verifier_message::<[u8; 32]>(),
            framed_long.verifier_message::<[u8; 32]>(),
            "exact-sampling retry markers must bind the consumed squeeze path",
        );
    }

    #[cfg(feature = "transcript-blake2b")]
    #[test]
    fn blake2b_retains_distinct_squeeze_histories_across_absorb() {
        let mut short = new_native_prover(b"width", b"substrate").unwrap();
        let mut long = new_native_prover(b"width", b"substrate").unwrap();
        let _: [u8; 1] = short.verifier_message();
        let _: [u8; 2] = long.verifier_message();
        short.public_message(&[17u8]);
        long.public_message(&[17u8]);
        assert_ne!(
            short.verifier_message::<[u8; 32]>(),
            long.verifier_message::<[u8; 32]>(),
            "Blake2b must retain distinct rejection histories without a marker",
        );
    }

    #[test]
    fn grinding_preview_matches_live_replay_without_mutation() {
        let nonce_record = ProtocolContextRecord::new(
            ProtocolSiteId {
                family: 9,
                invocation: 2,
                ..ProtocolSiteId::default()
            }
            .to_bytes(),
            ProtocolMessageKind::GrindingNonce as u32,
            1,
            native_nonce_max_bytes(12) as u64,
            0,
        );
        let predicate_record = ProtocolContextRecord::new(
            ProtocolSiteId {
                family: 9,
                invocation: 2,
                stage: 1,
                ..ProtocolSiteId::default()
            }
            .to_bytes(),
            ProtocolMessageKind::GrindingPredicate as u32,
            0,
            0,
            crate::GRINDING_PREDICATE_LEN as u64,
        );
        let mut prover = new_native_prover(b"grinding", b"fixture").unwrap();
        let before = prover.narg_string().to_vec();
        let preview = preview_native_grinding_predicate(&prover, 17);
        assert_eq!(prover.narg_string(), before);
        let live = commit_native_grinding_nonce(&mut prover, nonce_record, 17, predicate_record);
        assert_eq!(preview, live);

        let proof = prover.narg_string().to_vec();
        let mut verifier = new_native_verifier(b"grinding", b"fixture", &proof).unwrap();
        let (nonce, replay) =
            receive_native_grinding_nonce(&mut verifier, nonce_record, predicate_record).unwrap();
        assert_eq!(nonce, 17);
        assert_eq!(replay, live);
        assert!(verifier.check_eof().is_ok());
    }
}
