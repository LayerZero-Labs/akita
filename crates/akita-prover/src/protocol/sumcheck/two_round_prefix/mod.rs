//! Prover-internal two-round-prefix kernels for Akita stages 1 and 2.
//!
//! This directory keeps the transient two-round prefix optimization split by
//! shared lookup machinery and stage-specific state machines. The public crate
//! surface remains the `two_round_prefix` module itself; submodules are private.

mod common;
mod stage1;
mod stage2;

#[cfg(test)]
mod tests;

pub(crate) use common::{
    stage1_b4_s_digit_from_compact_s, stage1_b8_s_digit_from_compact_s, stage2_b4_w_digit,
    stage2_b8_w_digit,
};
pub(crate) use stage1::{
    build_stage1_bivariate_skip_proof_from_s_compact, can_use_stage1_two_round_prefix,
    Stage1BivariateSkipState,
};
pub(crate) use stage2::{
    build_stage2_bivariate_skip_proof_from_compact, can_use_stage2_two_round_prefix,
    Stage2BivariateSkipState,
};
