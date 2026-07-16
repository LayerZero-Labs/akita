//! Ring-native opening point for the Akita protocol.

use akita_algebra::{eq_poly::EqPolynomial, CyclotomicRing};
use akita_field::AkitaError;
use akita_field::FieldCore;
use akita_field::FromPrimitiveInt;
use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;

use crate::field_reduction::{embed_ring_subfield_scalar, FpExtEncoding};
const BLOCK_EMBED_ERROR: &str = "fold opening weight does not embed in the ring-subfield basis";

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
/// Contains the two exact factors of the physical source opening:
/// - `position_weights`: position weights of length `L`.
/// - `block_weights`: the live prefix of `F` block weights.
///
/// These are raw field scalars, not ring elements — they originate from
/// basis weight evaluations (Lagrange or monomial) and are always constant
/// (scalar) ring elements when embedded into the ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingOpeningPoint<F: FieldCore> {
    /// Position weights, with exactly one entry per block position.
    pub position_weights: Vec<F>,
    /// Block-select weights, truncated to the exact live block prefix.
    pub block_weights: Vec<F>,
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

/// Return the first `live_len` tensor-product weights without retaining the
/// padded Boolean suffix.
pub fn basis_weights_prefix<F: FieldCore>(
    point: &[F],
    basis: BasisMode,
    live_len: usize,
) -> Result<Vec<F>, AkitaError> {
    let capacity = basis_weight_len(point.len())?;
    if live_len == 0 || live_len > capacity {
        return Err(AkitaError::InvalidSize {
            expected: capacity,
            actual: live_len,
        });
    }
    match basis {
        BasisMode::Lagrange => EqPolynomial::evals_prefix(point, live_len),
        BasisMode::Monomial => (0..live_len)
            .map(|index| {
                Ok(point
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| index & (1usize << bit) != 0)
                    .fold(F::one(), |acc, (_, &coordinate)| acc * coordinate))
            })
            .collect(),
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

/// Boolean opening-domain capacity for an exact compact physical source.
pub fn opening_domain_len(source_len: usize) -> Result<usize, AkitaError> {
    if source_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "opening source length must be positive".to_string(),
        ));
    }
    source_len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("opening domain length overflow".to_string()))
}

/// Validate and return an index in the exact physical opening source.
pub fn checked_opening_source_index(
    source_len: usize,
    physical_index: usize,
) -> Result<usize, AkitaError> {
    if physical_index >= source_len {
        return Err(AkitaError::InvalidInput(
            "physical opening index out of range".to_string(),
        ));
    }
    Ok(physical_index)
}

/// Convert the outer portion of a field opening point into ring-native vectors.
///
/// The first `log2(block_len)` coordinates select a position and the
/// remaining `log2(next_power_of_two(num_blocks))` coordinates select a
/// block. Only the exact live block prefix is retained.
///
/// # Errors
///
/// Returns an error if the point dimension does not match `layout`.
pub fn ring_opening_point_from_field<F: FieldCore>(
    opening_point: &[F],
    block_len: usize,
    num_blocks: usize,
    basis: BasisMode,
) -> Result<RingOpeningPoint<F>, AkitaError> {
    if !block_len.is_power_of_two() || num_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "opening geometry requires power-of-two L and positive F".to_string(),
        ));
    }
    let position_bits = block_len.trailing_zeros() as usize;
    let block_capacity = num_blocks
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("block capacity overflow".to_string()))?;
    let block_bits = block_capacity.trailing_zeros() as usize;
    let expected_len = position_bits
        .checked_add(block_bits)
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(AkitaError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let position_weights = basis_weights(&opening_point[..position_bits], basis)?;
    let block_weights = basis_weights_prefix(&opening_point[position_bits..], basis, num_blocks)?;
    Ok(RingOpeningPoint {
        position_weights,
        block_weights,
    })
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

/// Embed `eq(block_open, j)` as a ring element for each live block index `j`.
pub fn block_rings_at_opening<F, E, const D: usize>(
    fold_open: &[E],
    num_blocks: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F> + FieldCore,
{
    basis_weights_prefix(fold_open, BasisMode::Lagrange, num_blocks)?
        .into_iter()
        .map(|weight| {
            embed_ring_subfield_scalar::<F, E, D>(
                weight,
                AkitaError::InvalidInput(BLOCK_EMBED_ERROR.to_string()),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    #[test]
    fn opening_point_keeps_exact_live_block_prefix() {
        let point = [
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
        ];
        let opening = ring_opening_point_from_field(&point, 4, 3, BasisMode::Lagrange).unwrap();
        assert_eq!(
            opening.position_weights,
            lagrange_weights(&point[..2]).unwrap()
        );
        assert_eq!(
            opening.block_weights,
            lagrange_weights(&point[2..]).unwrap()[..3]
        );
    }

    #[test]
    fn monomial_prefix_omits_boolean_capacity_tail() {
        let point = [F::from_u64(2), F::from_u64(3)];
        let prefix = basis_weights_prefix(&point, BasisMode::Monomial, 3).unwrap();
        assert_eq!(prefix, monomial_weights(&point).unwrap()[..3]);
    }
}
