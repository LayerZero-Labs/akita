//! Ring-native opening point for the Hachi protocol.

use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::poly::multilinear_lagrange_basis;
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

/// Multilinear Lagrange weights: `⊗ᵢ (1 − xᵢ, xᵢ)`.
pub fn lagrange_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    multilinear_lagrange_basis(&mut weights, point);
    weights
}

/// Multilinear monomial weights: `⊗ᵢ (1, xᵢ)`.
///
/// The j-th entry is `∏_{i ∈ bits(j)} point[i]`.
pub fn monomial_weights<F: FieldCore>(point: &[F]) -> Vec<F> {
    let len = 1usize << point.len();
    let mut weights = vec![F::zero(); len];
    weights[0] = F::one();
    for (level, &p) in point.iter().enumerate() {
        let k = 1usize << level;
        for i in (0..k).rev() {
            weights[i + k] = weights[i] * p;
        }
    }
    weights
}

/// Return tensor-product weights for one opening point under the chosen basis.
pub fn basis_weights<F: FieldCore>(point: &[F], basis: BasisMode) -> Vec<F> {
    match basis {
        BasisMode::Lagrange => lagrange_weights(point),
        BasisMode::Monomial => monomial_weights(point),
    }
}

/// Convert the outer portion of a field opening point into ring-native vectors.
///
/// The first `m_vars` coordinates select the position within each block; the
/// remaining `r_vars` coordinates select which block is opened.
///
/// # Errors
///
/// Returns an error if `m_vars + r_vars` overflows or if `opening_point` has
/// the wrong length.
pub fn ring_opening_point_from_field<F: FieldCore>(
    opening_point: &[F],
    r_vars: usize,
    m_vars: usize,
    basis: BasisMode,
) -> Result<RingOpeningPoint<F>, HachiError> {
    let expected_len = r_vars
        .checked_add(m_vars)
        .ok_or_else(|| HachiError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(HachiError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let a = basis_weights(&opening_point[..m_vars], basis);
    let b = basis_weights(&opening_point[m_vars..], basis);
    Ok(RingOpeningPoint { a, b })
}

/// Reduce the inner `alpha = log2(D)` opening coordinates to one ring element.
///
/// # Errors
///
/// Returns an error if the number of basis weights implied by `inner_point`
/// does not match `D`.
pub fn reduce_inner_opening_to_ring_element<F: FieldCore, const D: usize>(
    inner_point: &[F],
    basis: BasisMode,
) -> Result<CyclotomicRing<F, D>, HachiError> {
    let weights = basis_weights(inner_point, basis);
    if weights.len() != D {
        return Err(HachiError::InvalidInput(format!(
            "inner basis length {} does not match D={D}",
            weights.len()
        )));
    }
    Ok(CyclotomicRing::from_slice(&weights))
}
