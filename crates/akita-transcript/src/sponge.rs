//! Spongefish backend selected by Akita's transcript feature.

/// Sponge backend selected by the active transcript feature.
///
/// Exactly one transcript backend feature must be active in the complete PCS graph.
#[cfg(all(
    feature = "transcript-blake2b",
    not(all(feature = "blake2-inline", target_arch = "riscv64"))
))]
pub type TranscriptSponge = spongefish::instantiations::Blake2b512;

/// Sponge backend selected by the active transcript feature: spongefish's
/// Blake2b512 sponge over the inline hasher's `digest` adapter, which
/// produces the `blake2` crate's bytes. Host builds keep the `blake2` crate;
/// only a RISC-V guest has the inline.
#[cfg(all(
    feature = "transcript-blake2b",
    feature = "blake2-inline",
    target_arch = "riscv64"
))]
pub type TranscriptSponge = spongefish::instantiations::Hash<
    jolt_inlines_blake2::digest_adapter::Blake2b<jolt_inlines_blake2::digest_adapter::U64>,
>;

/// Sponge backend selected by the active transcript feature.
#[cfg(feature = "transcript-keccak")]
pub type TranscriptSponge = spongefish::instantiations::Keccak;
