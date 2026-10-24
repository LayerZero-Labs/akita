//! Akita's Fiat-Shamir policy over Jolt's spongefish transcript.
//!
//! `jolt-transcript` supplies the sponge, the field-challenge codecs and the
//! proof-of-work step. This crate adds what is specific to Akita: its session
//! identifier, canonical proof atoms, diagnostic site records, and the
//! role-generic [`ProofChannel`].

mod grinding;
#[cfg(feature = "logging-transcript")]
mod logging;
mod proof_stream;

#[cfg(not(any(feature = "transcript-blake2b", feature = "transcript-keccak")))]
compile_error!("enable exactly one transcript backend: transcript-blake2b or transcript-keccak");

#[cfg(all(feature = "transcript-blake2b", feature = "transcript-keccak"))]
compile_error!("enable exactly one transcript backend: transcript-blake2b or transcript-keccak");

pub use grinding::{
    GRINDING_CHECK_BYTES, GRINDING_LITTLE_ENDIAN_BIT_ORDER, GRINDING_NONCE_SLACK_BITS,
    MAX_GRINDING_BITS,
};
pub use jolt_transcript::{ChallengeCodec, ProverTranscript, VerifierTranscript};
#[cfg(feature = "logging-transcript")]
pub use logging::{
    clear_thread_events, thread_events, thread_proof_ranges, ProofMessageRange, TranscriptEvent,
};
#[cfg(feature = "logging-transcript")]
pub use proof_stream::finish_proof_ranges;
pub use proof_stream::{
    exchange_extension_group, ext_challenge, ext_challenges, extension_slots, new_prover_channel,
    new_verifier_channel, nonce_encoded_len, nonce_max_bytes, prover_context, public_bytes,
    public_extensions, public_fields, receive_bounded_bytes, receive_byte_group,
    receive_field_group, receive_nonce, reject, send_bounded_bytes, send_byte_group,
    send_field_group, send_nonce, verifier_context, ExtensionAtom, FieldAtom, NonceAtom,
    ProofChannel, ProtocolContextRecord, ProtocolMessageKind, ProtocolSiteId, CONTEXT_DOMAIN,
    PROTOCOL_VERSION, SESSION_TAG, SITE_FAMILY_EXTENSION_OPENING_REDUCTION,
    SITE_FAMILY_FOLD_BINDING, SITE_FAMILY_FOLD_CHALLENGE, SITE_FAMILY_NEXT_WITNESS,
    SITE_FAMILY_OPENING_PAYLOAD, SITE_FAMILY_PHYSICAL_L2, SITE_FAMILY_ROOT_STATEMENT,
    SITE_FAMILY_STAGE1, SITE_FAMILY_STAGE2, SITE_FAMILY_STAGE3, SITE_FAMILY_SUMCHECK,
    SITE_FAMILY_TERMINAL,
};
#[cfg(feature = "prover")]
pub use proof_stream::{FoldPreview, FoldWork};

/// Sponge backend selected by the active transcript feature.
#[cfg(feature = "transcript-blake2b")]
pub type TranscriptSponge = jolt_transcript::Blake2b512;

/// Sponge backend selected by the active transcript feature.
#[cfg(feature = "transcript-keccak")]
pub type TranscriptSponge = spongefish::instantiations::Keccak;

/// Prover end of the proof channel.
pub type ProverChannel = spongefish::ProverState<TranscriptSponge>;

/// Verifier end of the proof channel.
///
/// A failed read, a failed challenge draw, a failed proof-of-work check and
/// [`reject`] all poison the state, so `check_eof` fails afterwards even when
/// the caller drops the error.
pub type VerifierChannel<'proof> = spongefish::VerifierState<'proof, TranscriptSponge>;
