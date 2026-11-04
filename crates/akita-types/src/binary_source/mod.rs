//! Opt-in binary source wire contracts; no protocol flow is selected here.
use akita_error::AkitaError;
use akita_transcript::ProofChannel;

mod a_product;
mod agreement;
mod authentication;
mod commitment_manifest;
mod descriptor;
mod evaluation_bounds;
mod evaluation_weights;
mod field_switch;
mod grinding;
mod index;
mod ordinary_batch;
mod ordinary_root;
mod prime;
mod prime_plan;
mod prime_weights;
mod projection;
mod projection_rows;
mod root_plan;
mod root_public;
mod root_request;
mod root_transcript;
mod terminal;
mod word_claim;
pub use a_product::*;
pub use agreement::*;
pub use authentication::*;
pub use commitment_manifest::*;
pub use descriptor::*;
pub use evaluation_bounds::*;
pub use evaluation_weights::*;
pub use field_switch::*;
pub use grinding::*;
pub use index::*;
pub use ordinary_batch::*;
pub use ordinary_root::*;
pub use prime::*;
pub use prime_weights::*;
pub use projection::*;
pub use projection_rows::*;
pub use root_plan::*;
pub use root_public::*;
pub use root_request::*;
pub use root_transcript::*;
pub use terminal::*;
pub use word_claim::*;

/// Squeeze one 32-byte challenge block for a binary-source rejection sampler.
///
/// The number of blocks a sampler draws depends on earlier squeezed values.
/// Such variable-length draws are confined to the experimental binary source.
pub fn challenge_block(channel: &mut impl ProofChannel) -> Result<[u8; 32], AkitaError> {
    channel.challenge_message(32, |bytes| {
        bytes
            .try_into()
            .map_err(|_| AkitaError::Internal("challenge block width".into()))
    })
}
#[cfg(test)]
mod tests;

#[cfg(feature = "test-support")]
mod matrix_access;
#[cfg(feature = "test-support")]
pub use matrix_access::BinaryAReadGuard;
