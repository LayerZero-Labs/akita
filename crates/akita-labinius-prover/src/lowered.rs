//! Clear witnesses for the admitted lowered root relation.
//!
//! All quotients and integer carries are fixed before the caller chooses the
//! explicit lowered challenges. These helpers introduce no transcript or
//! proof encoding.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

mod parity;
mod witness;

pub use parity::parity_quotient_and_carry;
pub use witness::{encode_witness, flatten_image};
