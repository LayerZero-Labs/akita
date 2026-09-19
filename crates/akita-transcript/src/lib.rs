//! Native Spongefish proof streams and Akita-specific message codecs.

mod grinding;
pub mod labels;
#[cfg(feature = "logging-transcript")]
mod logging;
mod native;
mod sponge;

#[cfg(not(any(feature = "transcript-blake2b", feature = "transcript-keccak")))]
compile_error!("enable exactly one transcript backend: transcript-blake2b or transcript-keccak");

#[cfg(all(feature = "transcript-blake2b", feature = "transcript-keccak"))]
compile_error!("enable exactly one transcript backend: transcript-blake2b or transcript-keccak");

pub use grinding::{
    grinding_predicate_accepts, GRINDING_LITTLE_ENDIAN_BIT_ORDER, GRINDING_NONCE_SLACK_BITS,
    GRINDING_PREDICATE_BYTES, GRINDING_PREDICATE_LEN, MAX_GRINDING_BITS,
};
#[cfg(feature = "logging-transcript")]
pub use logging::{clear_thread_events, thread_events, TranscriptEvent};
pub use native::{
    commit_native_grinding_nonce, native_field_sampling_budget_is_certified,
    native_prover_ext_challenge, native_prover_field_challenge, native_prover_fold_root,
    native_verifier_ext_challenge, native_verifier_field_challenge, native_verifier_fold_root,
    new_native_prover, new_native_verifier, preview_native_grinding_predicate, prover_context,
    public_native_bytes_prover, public_native_bytes_verifier, public_native_extensions_prover,
    public_native_extensions_verifier, public_native_fields_prover, public_native_fields_verifier,
    receive_native_bounded_bytes, receive_native_byte_group, receive_native_bytes,
    receive_native_extension, receive_native_extension_group, receive_native_field,
    receive_native_field_group, receive_native_grinding_nonce, search_native_grinding_nonce,
    send_native_bounded_bytes, send_native_byte_group, send_native_bytes, send_native_extension,
    send_native_extension_group, send_native_field, send_native_field_group, verifier_context,
    NativeContextError, NativeExtension, NativeField, NativeFoldPreview, NativeInitializationError,
    NativeProverState, NativeU128, NativeVerifierState, ProtocolContextRecord, ProtocolMessageKind,
    ProtocolSiteId, NATIVE_CONTEXT_DOMAIN, NATIVE_FIELD_CHALLENGE_BYTES,
    NATIVE_FIELD_SAMPLING_DRAW_LIMIT, NATIVE_FIELD_SAMPLING_SECURITY_BITS, NATIVE_PROTOCOL_VERSION,
    SITE_FAMILY_EXTENSION_OPENING_REDUCTION, SITE_FAMILY_FOLD_BINDING, SITE_FAMILY_FOLD_CHALLENGE,
    SITE_FAMILY_NEXT_WITNESS, SITE_FAMILY_OPENING_PAYLOAD, SITE_FAMILY_PHYSICAL_L2,
    SITE_FAMILY_ROOT_STATEMENT, SITE_FAMILY_STAGE1, SITE_FAMILY_STAGE2, SITE_FAMILY_STAGE3,
    SITE_FAMILY_SUMCHECK, SITE_FAMILY_TERMINAL,
};
pub use sponge::TranscriptSponge;

/// Byte length of every native transcript challenge block.
pub const TRANSCRIPT_CHALLENGE_BLOCK_LEN: usize = 32;
/// Byte length of every fold-challenge seed.
pub const FOLD_CHALLENGE_SEED_LEN: usize = TRANSCRIPT_CHALLENGE_BLOCK_LEN;
