//! Ring-native opening point for the Akita protocol.

use akita_algebra::CyclotomicRing;
use akita_field::AkitaError;
use akita_field::FieldCore;
use akita_field::FromPrimitiveInt;
use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;

use crate::field_reduction::{embed_ring_subfield_scalar, FpExtEncoding};
use crate::OpeningBlockLayout;

const BLOCK_EMBED_ERROR: &str = "block opening weight does not embed in the ring-subfield basis";

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
/// Contains the two exact factors of the virtual block opening:
/// - `a`: position weights of length `position_stride`.
/// - `b`: block weights of length `live_fold_count`.
///
/// These are raw field scalars, not ring elements — they originate from
/// basis weight evaluations (Lagrange or monomial) and are always constant
/// (scalar) ring elements when embedded into the ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingOpeningPoint<F: FieldCore> {
    /// Virtual-position weights.
    pub a: Vec<F>,
    /// Block-select weights.
    pub b: Vec<F>,
}

/// Multilinear Lagrange weights: `⊗ᵢ (1 − xᵢ, xᵢ)`.
///
/// # Errors
///
/// Returns an error if the implied weight table would overflow or exceed the
/// verifier sequence bound.
pub fn lagrange_weights<F: FieldCore>(point: &[F]) -> Result<Vec<F>, AkitaError> {
    let len = basis_weight_len(point.len())?;
    let mut weights = vec![F::zero(); len];
    if weights.is_empty() {
        return Ok(weights);
    }
    weights[0] = F::one();
    for (level, &p) in point.iter().enumerate() {
        let k = 1usize << level;
        let one_minus_p = F::one() - p;
        for i in (0..k).rev() {
            let value = weights[i];
            weights[i] = value * one_minus_p;
            weights[i + k] = value * p;
        }
    }
    Ok(weights)
}

/// Multilinear monomial weights: `⊗ᵢ (1, xᵢ)`.
///
/// The j-th entry is `∏_{i ∈ bits(j)} point[i]`.
///
/// # Errors
///
/// Returns an error if the implied weight table would overflow or exceed the
/// verifier sequence bound.
pub fn monomial_weights<F: FieldCore>(point: &[F]) -> Result<Vec<F>, AkitaError> {
    let len = basis_weight_len(point.len())?;
    let mut weights = vec![F::zero(); len];
    weights[0] = F::one();
    for (level, &p) in point.iter().enumerate() {
        let k = 1usize << level;
        for i in (0..k).rev() {
            weights[i + k] = weights[i] * p;
        }
    }
    Ok(weights)
}

/// Return tensor-product weights for one opening point under the chosen basis.
pub fn basis_weights<F: FieldCore>(point: &[F], basis: BasisMode) -> Result<Vec<F>, AkitaError> {
    match basis {
        BasisMode::Lagrange => lagrange_weights(point),
        BasisMode::Monomial => monomial_weights(point),
    }
}

fn basis_weight_len(num_vars: usize) -> Result<usize, AkitaError> {
    let shift = u32::try_from(num_vars).map_err(|_| AkitaError::InvalidSize {
        expected: usize::BITS as usize,
        actual: num_vars,
    })?;
    let len = 1usize
        .checked_shl(shift)
        .ok_or_else(|| AkitaError::InvalidInput("basis weight dimension overflow".to_string()))?;
    if len > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSize {
            expected: DEFAULT_MAX_SEQUENCE_LEN,
            actual: len,
        });
    }
    Ok(len)
}

/// Convert the outer portion of a field opening point into ring-native vectors.
///
/// The first `log2(position_stride)` coordinates select a virtual position,
/// and the remaining `log2(live_fold_count)` coordinates select a block. Physical
/// coefficients remain compact while the opening MLE inserts structural zeros
/// after each live block.
///
/// # Errors
///
/// Returns an error if the point dimension does not match `layout`.
pub fn ring_opening_point_from_field<F: FieldCore>(
    opening_point: &[F],
    layout: OpeningBlockLayout,
    basis: BasisMode,
) -> Result<RingOpeningPoint<F>, AkitaError> {
    let position_bits = layout.position_stride().trailing_zeros() as usize;
    let block_bits = layout.live_fold_count().trailing_zeros() as usize;
    let expected_len = position_bits
        .checked_add(block_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(AkitaError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let a = basis_weights(&opening_point[..position_bits], basis)?;
    let b = basis_weights(&opening_point[position_bits..], basis)?;
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
) -> Result<CyclotomicRing<F, D>, AkitaError> {
    let weights = basis_weights(inner_point, basis)?;
    if weights.len() != D {
        return Err(AkitaError::InvalidInput(format!(
            "inner basis length {} does not match D={D}",
            weights.len()
        )));
    }
    Ok(CyclotomicRing::from_slice(&weights))
}

/// Embed `eq(b_open, j)` as a ring element for each block index `j`.
pub fn block_rings_at_opening<F, E, const D: usize>(
    b_open: &[E],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + FieldCore,
{
    lagrange_weights(b_open)?
        .into_iter()
        .map(|weight| {
            embed_ring_subfield_scalar::<F, E, D>(
                weight,
                AkitaError::InvalidInput(BLOCK_EMBED_ERROR.to_string()),
            )
        })
        .collect()
}
