//! Native Spongefish state construction and canonical Akita message codecs.

use jolt_field::{CanonicalEncoding, ExtField, Field};
use spongefish::{
    protocol_id, DomainSeparator, DuplexSpongeInterface, Encoding, NargDeserialize, ProverState,
    VerificationError, VerifierState, WithoutInstance,
};
use std::{error::Error, fmt};

use crate::TranscriptSponge;

/// Native transcript and proof-stream format version.
pub const NATIVE_PROTOCOL_VERSION: u32 = 2;

/// Domain tag included in every native protocol context record.
pub const NATIVE_CONTEXT_DOMAIN: [u8; 32] = *b"akita-pcs/native-context/v2\0\0\0\0\0";

/// Number of random-oracle bytes used for each base-field coordinate challenge.
pub const NATIVE_FIELD_CHALLENGE_BYTES: u64 = 64;

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

/// Native proof-stream operation kind committed by a context record.
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

/// Native Spongefish verifier state used by Akita.
pub type NativeVerifierState<'proof> = VerifierState<'proof, TranscriptSponge>;

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
        formatter.write_str("native protocol context coordinate exceeds u32")
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
    let name = "akita-pcs/native-proof-stream/v2/blake2b";
    #[cfg(feature = "transcript-keccak")]
    let name = "akita-pcs/native-proof-stream/v2/keccak";
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
    Ok(native_domain(session, instance)?.to_verifier(TranscriptSponge::default(), proof))
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
    for coefficient in value.to_base_vec() {
        send_native_field(state, coefficient);
    }
}

/// Receive one extension-field proof atom from canonical base coordinates.
pub fn receive_native_extension<F, E>(
    state: &mut NativeVerifierState<'_>,
) -> Result<E, VerificationError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(E::DEGREE)
        .map_err(|_| VerificationError)?;
    for _ in 0..E::DEGREE {
        coefficients.push(receive_native_field(state)?);
    }
    Ok(E::from_base_slice(&coefficients))
}

/// Emit a schedule-bounded byte sequence as native one-byte proof atoms.
pub fn send_native_bytes(state: &mut NativeProverState, bytes: &[u8]) {
    for &byte in bytes {
        state.prover_message(&[byte]);
    }
}

/// Receive an exact schedule-bounded number of native one-byte proof atoms.
pub fn receive_native_bytes(
    state: &mut NativeVerifierState<'_>,
    len: usize,
) -> Result<Vec<u8>, VerificationError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| VerificationError)?;
    for _ in 0..len {
        bytes.push(state.prover_message::<[u8; 1]>()?[0]);
    }
    Ok(bytes)
}

/// Draw one base-field challenge from 512 random-oracle bits.
#[must_use]
pub fn native_prover_field_challenge<F: CanonicalEncoding>(state: &mut NativeProverState) -> F {
    let bytes = state.verifier_message::<[u8; NATIVE_FIELD_CHALLENGE_BYTES as usize]>();
    F::from_challenge_bytes(&bytes)
}

/// Draw one base-field challenge from 512 random-oracle bits.
#[must_use]
pub fn native_verifier_field_challenge<F: CanonicalEncoding>(
    state: &mut NativeVerifierState<'_>,
) -> F {
    let bytes = state.verifier_message::<[u8; NATIVE_FIELD_CHALLENGE_BYTES as usize]>();
    F::from_challenge_bytes(&bytes)
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
                NATIVE_FIELD_CHALLENGE_BYTES,
            ),
        );
        coefficients.push(native_prover_field_challenge(state));
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
                NATIVE_FIELD_CHALLENGE_BYTES,
            ),
        );
        coefficients.push(native_verifier_field_challenge(state));
    }
    Ok(E::from_base_slice(&coefficients))
}

/// Fixed-width public record separating logical message and challenge groups.
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

/// Absorb a public context record on the prover side.
pub fn prover_context(state: &mut NativeProverState, record: ProtocolContextRecord) {
    state.public_message(&record);
}

/// Absorb a public context record on the verifier side.
pub fn verifier_context(state: &mut NativeVerifierState<'_>, record: ProtocolContextRecord) {
    state.public_message(&record);
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
    nonce_record: ProtocolContextRecord,
    nonce: u32,
    predicate_record: ProtocolContextRecord,
) -> [u8; crate::GRINDING_PREDICATE_LEN] {
    let mut sponge = state.duplex_sponge_state.clone();
    sponge.absorb(nonce_record.encode().as_ref());
    sponge.absorb(nonce.encode().as_ref());
    sponge.absorb(predicate_record.encode().as_ref());
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
    nonce_record: ProtocolContextRecord,
    predicate_record: ProtocolContextRecord,
) -> Option<u32> {
    crate::grinding::search_grinding_nonce_with(grind_bits, nonce_bits, |nonce| {
        Some(preview_native_grinding_predicate(
            state,
            nonce_record,
            nonce,
            predicate_record,
        ))
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
    state.prover_message(&nonce);
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
    let nonce = state.prover_message::<u32>()?;
    verifier_context(state, predicate_record);
    Ok((nonce, state.verifier_message()))
}

impl Encoding<[u8]> for ProtocolContextRecord {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = [0u8; 96];
        out[..32].copy_from_slice(&self.domain);
        out[32..36].copy_from_slice(&self.version.to_le_bytes());
        out[36..68].copy_from_slice(&self.site_id);
        out[68..72].copy_from_slice(&self.kind.to_le_bytes());
        out[72..80].copy_from_slice(&self.atom_count.to_le_bytes());
        out[80..88].copy_from_slice(&self.encoded_bytes.to_le_bytes());
        out[88..96].copy_from_slice(&self.challenge_bytes.to_le_bytes());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{CanonicalBytes, Prime32Offset99 as F, Ring};

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
    fn native_field_failure_does_not_consume_cursor() {
        let noncanonical = vec![u8::MAX; F::NUM_BYTES];
        let mut bytes = noncanonical.as_slice();
        let original = bytes;
        assert!(NativeField::<F>::deserialize_from_narg(&mut bytes).is_err());
        assert_eq!(bytes, original);
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

    #[test]
    fn context_record_encoding_is_fixed_width() {
        let record = ProtocolContextRecord {
            domain: [1; 32],
            version: NATIVE_PROTOCOL_VERSION,
            site_id: [2; 32],
            kind: 3,
            atom_count: 4,
            encoded_bytes: 5,
            challenge_bytes: 6,
        };
        assert_eq!(record.encode().as_ref().len(), 96);
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
            4,
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
        let preview =
            preview_native_grinding_predicate(&prover, nonce_record, 17, predicate_record);
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
