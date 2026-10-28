//! Akita's proof-of-work policy over Jolt's transcript proof of work.
//!
//! `jolt_transcript::ProverTranscript::grind` and
//! `VerifierTranscript::check_grind` define the wire format: the least nonce
//! below `2^(bits + 7)` as canonical LEB128, one byte per prover message, then
//! `ceil(bits / 8)` squeezed check bytes whose low `bits` bits must be zero.
//! The constants here are the parts of that format Akita binds into its
//! grinding-plan digest.

/// Extra nonce bits that make honest proof-of-work exhaustion negligible.
pub const GRINDING_NONCE_SLACK_BITS: u8 = 7;
/// Largest proof-of-work target.
pub const MAX_GRINDING_BITS: u8 = 25;
/// Largest number of check bytes squeezed by one proof-of-work step.
pub const GRINDING_CHECK_MAX_BYTES: u8 = grinding_check_bytes(MAX_GRINDING_BITS);
/// Low-bit-first check order.
pub const GRINDING_LITTLE_ENDIAN_BIT_ORDER: u8 = 0;

const _: () = assert!(MAX_GRINDING_BITS as u32 == jolt_transcript::MAX_POW_BITS);
const _: () = assert!(GRINDING_CHECK_MAX_BYTES == 4);

/// Check bytes squeezed by a proof-of-work step of `grind_bits` bits.
#[must_use]
pub const fn grinding_check_bytes(grind_bits: u8) -> u8 {
    grind_bits.div_ceil(u8::BITS as u8)
}
