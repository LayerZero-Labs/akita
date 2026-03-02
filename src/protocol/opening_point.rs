//! Ring-native opening point for the Hachi protocol.

use crate::FieldCore;

/// Polynomial basis mode for the evaluation relation.
///
/// Determines how the polynomial's values are interpreted during an opening
/// proof. The commitment itself is basis-agnostic; the basis only affects
/// the tensor-product weights used in `prove` and `verify`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisMode {
    /// Evaluations over the boolean hypercube.
    ///
    /// The weight vector is `⊗ᵢ (1 − xᵢ, xᵢ)` (multilinear Lagrange basis).
    /// Use when the committed values are `f(b)` for `b ∈ {0,1}^n`.
    Lagrange,

    /// Coefficients of multilinear monomials.
    ///
    /// The weight vector is `⊗ᵢ (1, xᵢ)`.
    /// Use when the committed values are the coefficients `c_S` such that
    /// `f(x) = Σ_S c_S · ∏_{i ∈ S} x_i`.
    Monomial,
}

/// Ring-native opening point storing field scalars.
///
/// Contains the two vectors used by the §4.2 prover:
/// - `a`: evaluation vector of length `2^m` (inner-block coordinates).
/// - `b`: block-select vector of length `2^r` (outer coordinates).
///
/// These are raw field scalars, not ring elements — they originate from
/// basis weight evaluations (Lagrange or monomial) and are always constant
/// (scalar) ring elements when embedded into the ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingOpeningPoint<F: FieldCore> {
    /// Evaluation vector of length `2^m` (field scalars).
    pub a: Vec<F>,
    /// Block-select vector of length `2^r` (field scalars).
    pub b: Vec<F>,
}
