//! Akita-specific sumcheck verifier stages.

pub(crate) mod opening_claims;
pub(crate) mod physical_l2_norm;
pub(crate) mod ring_switch;
pub(crate) mod stage1;
pub(crate) mod stage2;
pub(crate) mod stage3;

pub(crate) use stage3::SetupSumcheckVerifier;
