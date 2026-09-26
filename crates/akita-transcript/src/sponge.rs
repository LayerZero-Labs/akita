//! Spongefish backend selected by Akita's transcript feature.

/// Sponge backend selected by the active transcript feature.
#[cfg(feature = "transcript-blake2b")]
pub type TranscriptSponge = spongefish::instantiations::Blake2b512;

/// Sponge backend selected by the active transcript feature.
#[cfg(feature = "transcript-keccak")]
pub type TranscriptSponge = spongefish::instantiations::Keccak;
