//! JL consistency-sumcheck layout and verifier-wire shapes.
//!
//! Matrix sampling, integer projection, and joint-matrix MLE eval live in
//! `akita-challenges::jl`. Prove/verify sumcheck instances live in
//! `akita-prover::protocol::jl` and `akita-verifier::protocol::jl`.

mod claim;
mod layout;
mod transcript;
mod wire;

pub use claim::jl_image_claim;
pub use layout::{
    padded_live_table, validate_layout_for_matrix_mle, JlWitnessLayout, JL_CONSISTENCY_DEGREE,
};
pub use transcript::{absorb_jl_image, sample_jl_row_point};
pub use wire::{embed_jl_image_coords, embed_signed_i32, field_modulus};

#[cfg(feature = "jl-test-fixtures")]
pub mod fixtures;
