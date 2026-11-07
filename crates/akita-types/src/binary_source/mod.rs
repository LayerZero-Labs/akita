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
mod research_batch;
mod research_relation;
mod research_terminals;
mod research_transcript;
mod root_plan;
mod root_public;
mod root_request;
mod root_transcript;
mod terminal;
mod typed_prefix;
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
pub use research_batch::*;
pub use research_relation::*;
pub use research_terminals::*;
pub use research_transcript::*;
pub use root_plan::*;
pub use root_public::*;
pub use root_request::*;
pub use root_transcript::*;
pub use terminal::*;
pub use typed_prefix::*;
pub use word_claim::*;

/// Squeeze one 32-byte challenge block for a binary-source rejection sampler.
///
/// The number of blocks a sampler draws depends on earlier squeezed values.
/// Such variable-length draws are confined to the experimental binary source.
pub fn challenge_block(channel: &mut impl ProofChannel) -> Result<[u8; 32], AkitaError> {
    channel.challenge_array()
}
#[cfg(test)]
mod tests;

#[cfg(feature = "test-support")]
mod matrix_access;
#[cfg(any(test, feature = "test-support"))]
mod root_weight_fixtures;
#[cfg(feature = "test-support")]
pub use matrix_access::BinaryAReadGuard;
#[cfg(any(test, feature = "test-support"))]
pub use root_weight_fixtures::{
    binary_ordinary_profile_fixture, binary_word_manifest_fixture, root_weight_fixtures,
    RootFixtureField, RootWeightFixture, Stage3Fixture,
};
