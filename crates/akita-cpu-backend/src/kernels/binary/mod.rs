//! Packed binary arithmetic owned by the CPU backend.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

pub mod fold;
pub mod limb_commit;
pub mod weighted_integer_convolution;
pub mod weighted_source;
