//! Per-polynomial recursive opening claim carried across fold levels.
//!
//! Each entry of a recursive verifier's state vector represents one
//! polynomial opening that the next fold level must discharge: an
//! `opening_point`, the claimed `opening` value, the borrowed commitment
//! to the underlying witness, the `basis` the opening point lives in,
//! the witness length `w_len`, and the current digit basis `log_basis`.
//!
//! The single-poly recursive path is the `Vec.len() == 1` special case;
//! Phase D-full slice F adds an additional claim to the vector to open
//! the shared setup polynomial `S` alongside the folded witness via
//! multi-claim batched Hachi at the next level.

use crate::{BasisMode, FlatRingVec};
use akita_field::FieldCore;

/// One recursive opening claim carried into the next fold level.
///
/// The verifier holds a `Vec<RecursiveOpeningClaim>` describing every
/// polynomial that must be opened jointly at the next level. Fields are
/// public so consumers can construct claims directly.
#[derive(Debug)]
pub struct RecursiveOpeningClaim<'a, F: FieldCore> {
    /// Opening point for this claim.
    pub opening_point: Vec<F>,
    /// Claimed evaluation at `opening_point`.
    pub opening: F,
    /// Commitment to the witness being opened.
    pub commitment: &'a FlatRingVec<F>,
    /// Basis used to interpret `opening_point`.
    pub basis: BasisMode,
    /// Length of the committed witness, in field elements.
    pub w_len: usize,
    /// Digit basis of the committed witness, as `log2(b)`.
    pub log_basis: u32,
}
