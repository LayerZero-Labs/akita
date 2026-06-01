//! Ring-native opening point for the Akita protocol.

use akita_algebra::CyclotomicRing;
use akita_field::AkitaError;
use akita_field::FieldCore;
use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;

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

impl BasisMode {
    /// Stable wire identifier for proof serialization and transcript binding.
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Lagrange => 0,
            Self::Monomial => 1,
        }
    }

    /// Decode a stable wire identifier.
    pub fn from_u8(tag: u8) -> Result<Self, AkitaError> {
        match tag {
            0 => Ok(Self::Lagrange),
            1 => Ok(Self::Monomial),
            _ => Err(AkitaError::InvalidProof),
        }
    }
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

/// Block-order convention used when splitting outer opening coordinates into
/// in-block weights `a` and block weights `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockOrder {
    /// Level-0 polynomial layout: the first `m_vars` coordinates select the
    /// position within a block, and the remaining `r_vars` select the block.
    RowMajor,

    /// Recursive witness layout: the first `r_vars` coordinates select the
    /// block, and the remaining `m_vars` coordinates select the position
    /// within that block.
    ColumnMajor,
}

/// Convert the outer portion of a field opening point into ring-native vectors.
///
/// **Row-major (level 0):** the first `m_vars` coordinates select the
/// position within each block (`a`), the remaining `r_vars` select the
/// block (`b`).
///
/// **Column-major (recursive levels):** the first `r_vars` coordinates
/// select the block (`b`), the remaining `m_vars` select the position (`a`).
/// This corresponds to the column-major block interpretation where the
/// sequential polynomial index decomposes as
/// `i = position * 2^r + block`.
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
    block_order: BlockOrder,
) -> Result<RingOpeningPoint<F>, AkitaError> {
    let expected_len = r_vars
        .checked_add(m_vars)
        .ok_or_else(|| AkitaError::InvalidSetup("opening point length overflow".to_string()))?;
    if opening_point.len() != expected_len {
        return Err(AkitaError::InvalidPointDimension {
            expected: expected_len,
            actual: opening_point.len(),
        });
    }

    let (a, b) = match block_order {
        BlockOrder::ColumnMajor => {
            let b = basis_weights(&opening_point[..r_vars], basis)?;
            let a = basis_weights(&opening_point[r_vars..], basis)?;
            (a, b)
        }
        BlockOrder::RowMajor => {
            let a = basis_weights(&opening_point[..m_vars], basis)?;
            let b = basis_weights(&opening_point[m_vars..], basis)?;
            (a, b)
        }
    };
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
