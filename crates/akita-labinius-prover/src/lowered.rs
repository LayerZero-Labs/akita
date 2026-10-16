//! Clear witnesses for the admitted lowered root relation.
//!
//! All quotients and integer carries, and the prime left opening, are fixed
//! before the caller chooses the explicit lowered challenges. These helpers
//! introduce no transcript or proof encoding.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

mod a_carry;
mod a_remainders;
mod parity;
mod prime;
mod witness;

pub use a_carry::a_relation_carry;
pub use a_remainders::matrix_remainders_limb;
pub use parity::parity_quotient_and_carry;
pub use prime::prime_left_opening;
pub use witness::{encode_image, encode_witness};
