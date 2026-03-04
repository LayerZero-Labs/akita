//! Greyhound evaluation proof types.

use crate::algebra::ring::CyclotomicRing;
use crate::protocol::labrador::types::LabradorReductionConfig;
use crate::FieldCore;

/// Shape metadata for reshaped witness matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GreyhoundDimensions {
    /// Number of matrix rows (`2^{k_inner}`).
    pub m_rows: usize,
    /// Number of matrix columns (`2^{k_outer}`).
    pub n_cols: usize,
    /// Number of inner variables (`k_inner`).
    pub inner_vars: usize,
}

/// Greyhound evaluation proof payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreyhoundEvalProof<F: FieldCore, const D: usize> {
    /// Outer commitment to decomposed partial evaluations.
    pub u2: Vec<CyclotomicRing<F, D>>,
    /// Matrix row count.
    pub m_rows: usize,
    /// Matrix column count.
    pub n_cols: usize,
    /// Split point for `r = (r_outer, r_inner)`.
    pub inner_vars: usize,
    /// Labrador config agreed between prover and verifier.
    pub config: LabradorReductionConfig,
}

impl<F: FieldCore, const D: usize> GreyhoundEvalProof<F, D> {
    /// Construct an empty proof (used when Greyhound is disabled).
    pub fn empty() -> Self {
        Self {
            u2: Vec::new(),
            m_rows: 0,
            n_cols: 0,
            inner_vars: 0,
            config: LabradorReductionConfig {
                f: 1,
                b: 1,
                fu: 1,
                bu: 1,
                kappa: 1,
                kappa1: 0,
                tail: false,
            },
        }
    }
}
