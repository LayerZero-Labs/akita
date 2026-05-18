//! Per-polynomial recursive opening claim carried across fold levels.
//!
//! Each entry of a recursive verifier's state vector represents one
//! polynomial opening that the next fold level must discharge: an
//! `opening_point`, the claimed `opening` value, the commitment
//! to the underlying witness, the `basis` the opening point lives in,
//! the witness length `w_len`, and the current digit basis `log_basis`.
//!
//! The single-poly recursive path is the `Vec.len() == 1` special case.
//! For the joint `(w, S)` recursive open at level `L+1` (book §5.3
//! lines 627–660), the verifier pushes an additional claim into the
//! vector. Each claim may carry a per-claim
//! [`LevelParams`](crate::LevelParams) override so the multi-group
//! batched Hachi commit at L+1 can use per-group `(m, r, B,
//! digit_count)` for the `w` and `S` groups under shared outer
//! `(D, A)`. When `per_claim_lp == None` the claim inherits the
//! level's shared LP.

use crate::{BasisMode, FlatRingVec, LevelParams, TieredSetupParams};
use akita_field::FieldCore;

/// One recursive opening claim carried into the next fold level.
///
/// The verifier holds a `Vec<RecursiveOpeningClaim>` describing every
/// polynomial that must be opened jointly at the next level. Fields are
/// public so consumers can construct claims directly.
#[derive(Debug)]
pub struct RecursiveOpeningClaim<F: FieldCore> {
    /// Opening point for this claim.
    pub opening_point: Vec<F>,
    /// Claimed evaluation at `opening_point`.
    pub opening: F,
    /// Commitment to the witness being opened.
    pub commitment: FlatRingVec<F>,
    /// Basis used to interpret `opening_point`.
    pub basis: BasisMode,
    /// Length of the committed witness, in field elements.
    pub w_len: usize,
    /// Digit basis of the committed witness, as `log2(b)`.
    pub log_basis: u32,
    /// Optional per-claim [`LevelParams`] override.
    ///
    /// `None` inherits the level's shared LP. `Some(lp)` carries this
    /// claim's per-commitment-group `(m, r, B, digit_count)` for the
    /// multi-group batched Hachi commit at the next level. Heterogeneous
    /// per-claim LPs are grouped via [`LevelParams::groups`](crate::LevelParams)
    /// and dispatched through the multi-group commit kernel.
    pub per_claim_lp: Option<LevelParams>,
    /// Tiered routing marker (book §5.4): `Some(t)` on every chunk
    /// claim of a routed tiered S handle so consecutive chunk claims
    /// merge into one commitment group with `claim_count = k` carrying
    /// `tier = Some(t)`. `None` for ordinary claims and the meta claim.
    pub tier_marker: Option<TieredSetupParams>,
}
