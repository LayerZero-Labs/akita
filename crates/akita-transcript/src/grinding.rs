//! Akita's proof-of-work policy over Jolt's transcript proof of work.
//!
//! `jolt_transcript::ProverTranscript::grind` and
//! `VerifierTranscript::check_grind` define the format: the least nonce below
//! `2^(bits + 7)`, sent as canonical LEB128 and absorbed as one `u64`
//! little-endian word, then [`GRINDING_CHECK_BYTES`] squeezed check bytes
//! whose low `bits` bits must be zero. The constants here are the parts of
//! that format Akita binds into its grinding-plan digest.

/// Extra nonce bits that make honest proof-of-work exhaustion negligible.
pub const GRINDING_NONCE_SLACK_BITS: u8 = 7;
/// Largest proof-of-work target.
pub const MAX_GRINDING_BITS: u8 = 25;
/// Check bytes squeezed by every proof-of-work step: one `u64` word, so the
/// squeeze stays word aligned whatever the target.
pub const GRINDING_CHECK_BYTES: u8 = 8;
/// Low-bit-first check order.
pub const GRINDING_LITTLE_ENDIAN_BIT_ORDER: u8 = 0;

const _: () = assert!(MAX_GRINDING_BITS as u32 == jolt_transcript::MAX_POW_BITS);
const _: () = assert!((MAX_GRINDING_BITS as u32) < u8::BITS * GRINDING_CHECK_BYTES as u32);
