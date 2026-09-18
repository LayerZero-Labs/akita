//! Verifier-side transcript driver for the non-zk extension-opening reduction.
//!
//! The EOR sumcheck rounds are public-transcript checks. Their final claim is
//! linked to explicit terminal claims and then enforced through fused stage-2
//! `trace_eval_target` and per-claim scales.
