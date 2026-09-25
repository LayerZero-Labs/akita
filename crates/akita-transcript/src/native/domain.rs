//! Domain separation for native proof streams.
//!
//! Every native stream starts by absorbing three public values, in this order:
//!
//! 1. a 64-byte protocol identifier, zero-padded;
//! 2. the caller's `session` bytes, framed as `LE64(len) || bytes`;
//! 3. the caller's `instance` bytes, framed as `LE64(len) || bytes`.
//!
//! Akita's polynomial commitment scheme uses its own protocol identifier. A
//! protocol that drives its own rounds with the native stream types uses a
//! [`NativeProtocolId`] instead.

use spongefish::{DomainSeparator, Encoding, WithInstance, WithSession, WithoutInstance};
use std::{error::Error, fmt};

use super::{NativeProverState, NativeVerifierState, NATIVE_PROTOCOL_VERSION};
use crate::TranscriptSponge;

#[cfg(feature = "transcript-blake2b")]
const NATIVE_HASH_SUITE: &str = "blake2b";
#[cfg(feature = "transcript-keccak")]
const NATIVE_HASH_SUITE: &str = "keccak";

/// Longest application name accepted by [`NativeProtocolId::for_application`].
///
/// The identifier appends `/akita-native/v{NATIVE_PROTOCOL_VERSION}/{suite}`
/// to the application name and must fit in 64 bytes. The bound is computed for
/// the longest hash-suite name, so an application name accepted under one
/// transcript backend is accepted under the other.
pub const NATIVE_APPLICATION_NAME_MAX_LEN: usize = 40;

/// Failure to construct a native transcript from an unrepresentable public input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeInitializationError;

impl fmt::Display for NativeInitializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("native transcript input length exceeds u64")
    }
}

impl Error for NativeInitializationError {}

/// An application name cannot be turned into a [`NativeProtocolId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeProtocolIdError {
    /// The application name is empty.
    Empty,
    /// The application name is longer than [`NATIVE_APPLICATION_NAME_MAX_LEN`].
    TooLong {
        /// Length of the rejected name in bytes.
        len: usize,
    },
    /// The application name contains a byte outside visible ASCII
    /// (`0x21..=0x7e`).
    InvalidByte {
        /// Offset of the first rejected byte.
        index: usize,
        /// The rejected byte.
        byte: u8,
    },
}

impl fmt::Display for NativeProtocolIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("native application name is empty"),
            Self::TooLong { len } => write!(
                formatter,
                "native application name has {len} bytes; the limit is \
                 {NATIVE_APPLICATION_NAME_MAX_LEN}"
            ),
            Self::InvalidByte { index, byte } => write!(
                formatter,
                "native application name has byte {byte:#04x} at offset {index}; \
                 only visible ASCII is allowed"
            ),
        }
    }
}

impl Error for NativeProtocolIdError {}

/// A caller-owned 64-byte protocol identifier for a native proof stream.
///
/// Use this when a protocol other than Akita's polynomial commitment scheme
/// drives its own rounds with the native stream types (field atoms, samplers,
/// sumcheck channels). Such a protocol should not claim Akita's identifier.
///
/// The identifier is the zero-padded ASCII string
///
/// ```text
/// {application}/akita-native/v{NATIVE_PROTOCOL_VERSION}/{suite}
/// ```
///
/// where `suite` is `blake2b` or `keccak`, following the selected transcript
/// backend. Akita appends the version and suite, so a caller cannot omit
/// them. The map from application name to identifier is injective, and no
/// caller identifier equals the identifier used by
/// [`new_native_prover`] and [`new_native_verifier`].
///
/// The identifier separates the caller's streams from Akita's. It does not
/// separate two callers that choose the same application name; include an
/// owner and a version in the name, for example `"example-org/my-protocol/v1"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NativeProtocolId([u8; 64]);

impl NativeProtocolId {
    /// Derive the identifier for `application`.
    ///
    /// # Errors
    ///
    /// Returns [`NativeProtocolIdError`] when `application` is empty, longer
    /// than [`NATIVE_APPLICATION_NAME_MAX_LEN`] bytes, or contains a byte
    /// outside visible ASCII (`0x21..=0x7e`). Rejecting NUL keeps the
    /// zero padding unambiguous.
    pub fn for_application(application: &str) -> Result<Self, NativeProtocolIdError> {
        let bytes = application.as_bytes();
        if bytes.is_empty() {
            return Err(NativeProtocolIdError::Empty);
        }
        if bytes.len() > NATIVE_APPLICATION_NAME_MAX_LEN {
            return Err(NativeProtocolIdError::TooLong { len: bytes.len() });
        }
        if let Some((index, &byte)) = bytes
            .iter()
            .enumerate()
            .find(|(_, byte)| !byte.is_ascii_graphic())
        {
            return Err(NativeProtocolIdError::InvalidByte { index, byte });
        }
        // Distinctness from Akita's own identifier. Both identifiers are
        // NUL-free ASCII strings zero-padded to 64 bytes, so they are equal
        // only if the unpadded strings are equal. Every caller string ends in
        // `/akita-native/v{V}/{suite}`. Akita's string ends in
        // `/native-proof-stream/v{V}/{suite}` with the same `V` and `suite`, and
        // the 13 bytes before its `/v{V}/{suite}` tail are `-proof-stream`,
        // not `/akita-native`. Hence no application name reproduces Akita's
        // identifier. `caller_ids_never_equal_akita_id` checks this.
        let name =
            format!("{application}/akita-native/v{NATIVE_PROTOCOL_VERSION}/{NATIVE_HASH_SUITE}");
        let mut id = [0u8; 64];
        id.get_mut(..name.len())
            .ok_or(NativeProtocolIdError::TooLong { len: bytes.len() })?
            .copy_from_slice(name.as_bytes());
        Ok(Self(id))
    }

    /// Return the 64 identifier bytes absorbed at the start of the stream.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }
}

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

/// Akita's own protocol identifier. Its streams must stay byte-identical, so
/// the string and padding are pinned by `akita_protocol_id_is_pinned`.
fn native_protocol_id() -> [u8; 64] {
    spongefish::protocol_id(format_args!(
        "akita-pcs/native-proof-stream/v{NATIVE_PROTOCOL_VERSION}/{NATIVE_HASH_SUITE}"
    ))
}

type NativeDomain<'a> =
    DomainSeparator<WithInstance<FramedBytes<'a>>, WithSession<FramedBytes<'a>>>;

fn native_domain<'a>(
    protocol: [u8; 64],
    session: &'a [u8],
    instance: &'a [u8],
) -> Result<NativeDomain<'a>, NativeInitializationError> {
    Ok(DomainSeparator::<WithoutInstance>::new(protocol)
        .session(FramedBytes::new(session)?)
        .instance(FramedBytes::new(instance)?))
}

/// Construct a native prover state for Akita's polynomial commitment scheme.
///
/// # Domain separation
///
/// The state first absorbs Akita's 64-byte protocol identifier, then
/// `LE64(session.len()) || session`, then `LE64(instance.len()) || instance`.
/// Both inputs are length-framed, so distinct `(session, instance)` pairs
/// never produce the same absorbed bytes.
///
/// - `session` identifies the application context. The caller chooses it and
///   the verifier must use the same bytes.
/// - `instance` is the canonical public statement. The commitment scheme
///   passes the canonical `AkitaInstanceDescriptor` bytes here.
///
/// # Composition inside an outer Fiat-Shamir protocol
///
/// Each call starts a fresh sponge. State from an enclosing transcript is not
/// carried over implicitly, so a protocol that embeds this stream in a larger
/// Fiat-Shamir transcript must pass that state through `session`. A supported
/// form is
///
/// ```text
/// session = application_label || outer_digest
/// ```
///
/// where `outer_digest` is squeezed from the outer transcript after it has
/// absorbed every outer message that precedes this stream. Use a full-width
/// hash output of at least 256 bits. The session frame fixes the total length,
/// so with a fixed digest width the split between label and digest is unique.
///
/// The returned proof bytes are not absorbed into the outer transcript
/// automatically. An outer protocol that continues after this stream must
/// absorb the bytes and public results it depends on.
///
/// # Errors
///
/// Returns [`NativeInitializationError`] if an input length does not fit in
/// `u64`.
pub fn new_native_prover(
    session: &[u8],
    instance: &[u8],
) -> Result<NativeProverState, NativeInitializationError> {
    Ok(native_domain(native_protocol_id(), session, instance)?
        .to_prover(TranscriptSponge::default()))
}

/// Construct a native verifier state for Akita's polynomial commitment scheme
/// over one proof byte string.
///
/// `session` and `instance` are bound exactly as in [`new_native_prover`],
/// including the composition rules for an outer transcript digest.
///
/// # Errors
///
/// Returns [`NativeInitializationError`] if an input length does not fit in
/// `u64`.
pub fn new_native_verifier<'proof>(
    session: &[u8],
    instance: &[u8],
    proof: &'proof [u8],
) -> Result<NativeVerifierState<'proof>, NativeInitializationError> {
    Ok(NativeVerifierState::new(
        native_domain(native_protocol_id(), session, instance)?
            .to_verifier(TranscriptSponge::default(), proof),
    ))
}

/// Construct a native prover state under a caller-owned protocol identifier.
///
/// The stream uses the same framing, codecs, and samplers as
/// [`new_native_prover`]; only the first absorbed value differs. `session` and
/// `instance` are bound as documented there.
///
/// # Errors
///
/// Returns [`NativeInitializationError`] if an input length does not fit in
/// `u64`.
pub fn new_native_prover_for(
    protocol: &NativeProtocolId,
    session: &[u8],
    instance: &[u8],
) -> Result<NativeProverState, NativeInitializationError> {
    Ok(native_domain(protocol.0, session, instance)?.to_prover(TranscriptSponge::default()))
}

/// Construct a native verifier state under a caller-owned protocol identifier
/// over one proof byte string.
///
/// The verifier state has the same fail-closed behavior as
/// [`new_native_verifier`].
///
/// # Errors
///
/// Returns [`NativeInitializationError`] if an input length does not fit in
/// `u64`.
pub fn new_native_verifier_for<'proof>(
    protocol: &NativeProtocolId,
    session: &[u8],
    instance: &[u8],
    proof: &'proof [u8],
) -> Result<NativeVerifierState<'proof>, NativeInitializationError> {
    Ok(NativeVerifierState::new(
        native_domain(protocol.0, session, instance)?
            .to_verifier(TranscriptSponge::default(), proof),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{
        native_prover_field_challenge, native_verifier_field_challenge, receive_native_field,
        send_native_field,
    };
    use jolt_field::{Prime32Offset99 as F, Ring};

    fn padded(name: &[u8]) -> [u8; 64] {
        let mut id = [0u8; 64];
        id[..name.len()].copy_from_slice(name);
        id
    }

    fn unpadded(id: &[u8; 64]) -> &[u8] {
        let end = id.iter().rposition(|&byte| byte != 0).map_or(0, |i| i + 1);
        &id[..end]
    }

    /// A small stream whose proof bytes depend on the transcript state: the
    /// second field atom is a squeezed challenge.
    fn prove_small_stream(mut prover: NativeProverState) -> Vec<u8> {
        send_native_field(&mut prover, F::from_u64(42));
        let challenge = native_prover_field_challenge::<F>(&mut prover).unwrap();
        send_native_field(&mut prover, challenge);
        prover.narg_string().to_vec()
    }

    fn verify_small_stream(mut verifier: NativeVerifierState<'_>) -> bool {
        let Ok(first) = receive_native_field::<F>(&mut verifier) else {
            return false;
        };
        let Ok(challenge) = native_verifier_field_challenge::<F>(&mut verifier) else {
            return false;
        };
        let Ok(echoed) = receive_native_field::<F>(&mut verifier) else {
            return false;
        };
        first == F::from_u64(42) && echoed == challenge && verifier.check_eof().is_ok()
    }

    fn known_answer_stream(prover: &mut NativeProverState) -> ([u8; 32], F, [u8; 32]) {
        send_native_field(prover, F::from_u64(42));
        prover.public_message(b"public-message");
        let first = prover.verifier_message::<[u8; 32]>();
        let field = native_prover_field_challenge::<F>(prover).unwrap();
        send_native_field(prover, field);
        let second = prover.verifier_message::<[u8; 32]>();
        (first, field, second)
    }

    #[test]
    fn akita_protocol_id_is_pinned() {
        #[cfg(feature = "transcript-blake2b")]
        let expected = padded(b"akita-pcs/native-proof-stream/v7/blake2b");
        #[cfg(feature = "transcript-keccak")]
        let expected = padded(b"akita-pcs/native-proof-stream/v7/keccak");
        assert_eq!(native_protocol_id(), expected);
    }

    // Values captured from `new_native_prover` at LayerZero-Labs/akita
    // cad9f221 (before the protocol-identifier refactor). They pin the
    // identifier, framing, codecs, and sampler for Akita's own streams.
    // Replace them only with an intentional native stream change.
    #[cfg(feature = "transcript-blake2b")]
    #[test]
    fn akita_stream_has_a_known_answer() {
        let mut prover = new_native_prover(b"kat/session", b"kat/instance").unwrap();
        let (first, field, second) = known_answer_stream(&mut prover);
        assert_eq!(
            first,
            [
                188, 149, 150, 122, 177, 244, 175, 32, 192, 29, 231, 91, 33, 10, 108, 241, 122,
                199, 87, 178, 234, 157, 167, 193, 236, 244, 143, 240, 138, 26, 187, 105,
            ]
        );
        assert_eq!(field, F::from_u64(3_095_314_968));
        assert_eq!(
            second,
            [
                62, 88, 137, 203, 213, 120, 151, 70, 94, 154, 130, 243, 4, 161, 196, 7, 97, 57,
                223, 246, 177, 174, 210, 184, 184, 172, 125, 111, 103, 90, 194, 90,
            ]
        );
        assert_eq!(prover.narg_string(), [42, 0, 0, 0, 24, 194, 126, 184]);
    }

    #[cfg(feature = "transcript-keccak")]
    #[test]
    fn akita_stream_has_a_known_answer() {
        let mut prover = new_native_prover(b"kat/session", b"kat/instance").unwrap();
        let (first, field, second) = known_answer_stream(&mut prover);
        assert_eq!(
            first,
            [
                2, 252, 253, 175, 134, 52, 69, 26, 253, 142, 217, 200, 138, 91, 190, 141, 45, 222,
                15, 79, 153, 186, 229, 248, 176, 80, 18, 179, 206, 25, 73, 149,
            ]
        );
        assert_eq!(field, F::from_u64(2_687_546_360));
        assert_eq!(
            second,
            [
                61, 205, 212, 236, 234, 181, 190, 27, 21, 16, 178, 223, 22, 138, 154, 45, 249, 111,
                163, 43, 141, 249, 12, 253, 171, 90, 134, 151, 215, 108, 153, 35,
            ]
        );
        assert_eq!(prover.narg_string(), [42, 0, 0, 0, 248, 179, 48, 160]);
    }

    #[test]
    fn caller_id_appends_version_and_suite() {
        let id = NativeProtocolId::for_application("example-org/sumcheck/v1").unwrap();
        let expected = format!(
            "example-org/sumcheck/v1/akita-native/v{NATIVE_PROTOCOL_VERSION}/{NATIVE_HASH_SUITE}"
        );
        assert_eq!(id.as_bytes(), &padded(expected.as_bytes()));
    }

    #[test]
    fn application_name_bound_is_tight_for_every_suite() {
        let longest_suffix = ["blake2b", "keccak"]
            .iter()
            .map(|suite| format!("/akita-native/v{NATIVE_PROTOCOL_VERSION}/{suite}").len())
            .max()
            .unwrap();
        assert_eq!(NATIVE_APPLICATION_NAME_MAX_LEN + longest_suffix, 64);

        let longest = "a".repeat(NATIVE_APPLICATION_NAME_MAX_LEN);
        assert!(NativeProtocolId::for_application(&longest).is_ok());
        let too_long = "a".repeat(NATIVE_APPLICATION_NAME_MAX_LEN + 1);
        assert_eq!(
            NativeProtocolId::for_application(&too_long),
            Err(NativeProtocolIdError::TooLong {
                len: NATIVE_APPLICATION_NAME_MAX_LEN + 1
            })
        );
    }

    #[test]
    fn caller_id_rejects_empty_and_non_visible_names() {
        assert_eq!(
            NativeProtocolId::for_application(""),
            Err(NativeProtocolIdError::Empty)
        );
        for (name, index, byte) in [
            ("\0", 0, 0x00),
            ("app\0", 3, 0x00),
            ("my app", 2, b' '),
            ("app\n", 3, b'\n'),
            ("app\x7f", 3, 0x7f),
            ("app\u{e9}", 3, 0xc3),
        ] {
            assert_eq!(
                NativeProtocolId::for_application(name),
                Err(NativeProtocolIdError::InvalidByte { index, byte }),
                "{name:?}"
            );
        }
    }

    #[test]
    fn caller_id_is_injective_and_invertible() {
        let suffix = format!("/akita-native/v{NATIVE_PROTOCOL_VERSION}/{NATIVE_HASH_SUITE}");
        let names = [
            "a",
            "a/",
            "a/akita-native",
            "a/akita-native/v7",
            "ab",
            "akita-pcs",
            "akita-pcs/native-proof-stream",
            "akita-pcs/native-proof-stream/v7",
        ];
        let ids = names
            .iter()
            .map(|name| NativeProtocolId::for_application(name).unwrap())
            .collect::<Vec<_>>();
        for (name, id) in names.iter().zip(&ids) {
            let recovered = unpadded(id.as_bytes())
                .strip_suffix(suffix.as_bytes())
                .unwrap();
            assert_eq!(recovered, name.as_bytes());
        }
        for (i, left) in ids.iter().enumerate() {
            for right in &ids[i + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn caller_ids_never_equal_akita_id() {
        let akita = native_protocol_id();
        let suffix = format!("/akita-native/v{NATIVE_PROTOCOL_VERSION}/{NATIVE_HASH_SUITE}");
        // Structural form of the argument in `for_application`: every caller
        // identifier ends in `suffix`, and Akita's does not.
        assert!(!unpadded(&akita).ends_with(suffix.as_bytes()));
        for name in [
            "akita-pcs",
            "akita-pcs/native-proof-stream",
            "akita-pcs/native-proof-stream/v7",
            "akita-pcs/native-proof-stream/v7/blake2b",
            "akita-pcs/native-proof-stream/v7/keccak",
        ] {
            let id = NativeProtocolId::for_application(name).unwrap();
            assert_ne!(id.as_bytes(), &akita, "{name}");
        }
    }

    #[test]
    fn caller_streams_roundtrip_and_are_separated_from_akita() {
        let id = NativeProtocolId::for_application("example-org/sumcheck/v1").unwrap();
        let other = NativeProtocolId::for_application("example-org/sumcheck/v2").unwrap();
        let proof = prove_small_stream(new_native_prover_for(&id, b"session", b"x").unwrap());

        assert!(verify_small_stream(
            new_native_verifier_for(&id, b"session", b"x", &proof).unwrap()
        ));
        assert!(!verify_small_stream(
            new_native_verifier_for(&other, b"session", b"x", &proof).unwrap()
        ));
        assert!(!verify_small_stream(
            new_native_verifier(b"session", b"x", &proof).unwrap()
        ));
        assert_ne!(
            proof,
            prove_small_stream(new_native_prover(b"session", b"x").unwrap())
        );
    }

    #[test]
    fn session_and_instance_frames_do_not_shift() {
        let mut left = new_native_prover(b"ab", b"c").unwrap();
        let mut right = new_native_prover(b"a", b"bc").unwrap();
        assert_ne!(
            left.verifier_message::<[u8; 32]>(),
            right.verifier_message::<[u8; 32]>()
        );
    }

    #[test]
    fn outer_digest_suffix_binds_the_stream() {
        let label = b"example-org/outer-protocol/v1";
        let session_a = [&label[..], &[0xa5; 32]].concat();
        let session_b = [&label[..], &[0x5a; 32]].concat();

        let proof_a = prove_small_stream(new_native_prover(&session_a, b"instance").unwrap());
        let proof_b = prove_small_stream(new_native_prover(&session_b, b"instance").unwrap());
        assert_ne!(proof_a, proof_b);

        assert!(verify_small_stream(
            new_native_verifier(&session_a, b"instance", &proof_a).unwrap()
        ));
        assert!(!verify_small_stream(
            new_native_verifier(&session_b, b"instance", &proof_a).unwrap()
        ));
        assert!(!verify_small_stream(
            new_native_verifier(label, b"instance", &proof_a).unwrap()
        ));
    }
}
