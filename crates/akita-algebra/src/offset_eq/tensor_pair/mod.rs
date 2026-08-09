mod evaluate;
mod materialize;

pub use evaluate::eval_boolean_pair_tensor_families;
pub use materialize::materialize_eq_tensor_left;

use super::MAX_COMPACT_STRIDE_TERMS;
use crate::{AkitaError, FieldCore};
use std::sync::Arc;

/// Weights carried by one affine tensor axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EqPairTensorWeights<F: FieldCore> {
    /// Every coordinate has coefficient one.
    Unit,
    /// Coordinate weights in increasing axis order.
    Dense(Arc<[F]>),
}

/// One axis in a tensor product of paired equality addresses.
///
/// Coordinate `i` adds `left_stride * i` and `right_stride * i` to the
/// equality addresses. A zero stride is permitted on either side because
/// setup row and fold axes act on only one equality domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EqPairTensorAxis<F: FieldCore> {
    /// Number of coordinates on the axis.
    pub len: usize,
    /// Address increment per coordinate in the left equality domain.
    pub left_stride: usize,
    /// Address increment per coordinate in the right equality domain.
    pub right_stride: usize,
    /// Coordinate coefficients.
    pub weights: EqPairTensorWeights<F>,
}

impl<F: FieldCore> EqPairTensorAxis<F> {
    /// Construct an axis whose coordinate coefficients are all one.
    #[must_use]
    pub const fn unit(len: usize, left_stride: usize, right_stride: usize) -> Self {
        Self {
            len,
            left_stride,
            right_stride,
            weights: EqPairTensorWeights::Unit,
        }
    }

    /// Construct an axis with explicit coordinate coefficients.
    #[must_use]
    pub fn dense(left_stride: usize, right_stride: usize, weights: impl Into<Arc<[F]>>) -> Self {
        let weights = weights.into();
        Self {
            len: weights.len(),
            left_stride,
            right_stride,
            weights: EqPairTensorWeights::Dense(weights),
        }
    }
}

/// A direct tensor description of paired equality-address geometry.
///
/// The represented value is
///
/// ```text
/// scalar * sum_{i_0, ..., i_k}
///     product_j axis_weight_j[i_j]
///   * eq(left,  left_offset  + sum_j left_stride_j  * i_j)
///   * eq(right, right_offset + sum_j right_stride_j * i_j).
/// ```
///
/// Axes are supplied from innermost to outermost. Construction merges adjacent
/// unit-weight axes whenever both address maps are contiguous. This is what
/// turns the uniform ring-dimension case into the same long affine streams as
/// the former specialized evaluator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EqPairTensorFamily<F: FieldCore> {
    /// Base address in the left equality domain.
    pub left_offset: usize,
    /// Base address in the right equality domain.
    pub right_offset: usize,
    /// Coefficient shared by the whole tensor family.
    pub scalar: F,
    /// Tensor axes from innermost to outermost.
    pub axes: Vec<EqPairTensorAxis<F>>,
}

impl<F: FieldCore> EqPairTensorFamily<F> {
    /// Validate and normalize a tensor family.
    ///
    /// # Errors
    ///
    /// Returns an error for empty axes, mismatched dense weights, or address
    /// arithmetic overflow.
    pub fn new(
        left_offset: usize,
        right_offset: usize,
        mut scalar: F,
        axes: Vec<EqPairTensorAxis<F>>,
    ) -> Result<Self, AkitaError> {
        let mut normalized = Vec::<EqPairTensorAxis<F>>::new();
        for axis in axes {
            match &axis.weights {
                EqPairTensorWeights::Unit => {}
                EqPairTensorWeights::Dense(weights) if weights.len() == axis.len => {}
                EqPairTensorWeights::Dense(_) => {
                    return Err(AkitaError::InvalidInput(
                        "paired tensor axis weight length mismatch".into(),
                    ));
                }
            }
            if axis.len == 0 {
                return Err(AkitaError::InvalidInput(
                    "paired tensor axes must be non-empty".into(),
                ));
            }
            checked_axis_offset(0, axis.left_stride, axis.len - 1, "left")?;
            checked_axis_offset(0, axis.right_stride, axis.len - 1, "right")?;

            if axis.len == 1 {
                if let EqPairTensorWeights::Dense(weights) = axis.weights {
                    let weight = *weights.first().ok_or(AkitaError::InvalidProof)?;
                    if weight.is_zero() {
                        return Ok(Self {
                            left_offset,
                            right_offset,
                            scalar: F::zero(),
                            axes: Vec::new(),
                        });
                    }
                    if weight != F::one() {
                        scalar *= weight;
                    }
                }
                continue;
            }

            let merged = normalized.last_mut().is_some_and(|inner| {
                matches!(inner.weights, EqPairTensorWeights::Unit)
                    && matches!(axis.weights, EqPairTensorWeights::Unit)
                    && inner
                        .left_stride
                        .checked_mul(inner.len)
                        .is_some_and(|stride| stride == axis.left_stride)
                    && inner
                        .right_stride
                        .checked_mul(inner.len)
                        .is_some_and(|stride| stride == axis.right_stride)
            });
            if merged {
                let inner = normalized.last_mut().ok_or(AkitaError::InvalidProof)?;
                inner.len = inner.len.checked_mul(axis.len).ok_or_else(|| {
                    AkitaError::InvalidInput("paired tensor merged length overflow".into())
                })?;
            } else {
                normalized.push(axis);
            }
        }

        Ok(Self {
            left_offset,
            right_offset,
            scalar,
            axes: normalized,
        })
    }
}

pub(super) fn checked_axis_offset(
    base: usize,
    stride: usize,
    coordinate: usize,
    side: &'static str,
) -> Result<usize, AkitaError> {
    stride
        .checked_mul(coordinate)
        .and_then(|offset| base.checked_add(offset))
        .ok_or_else(|| AkitaError::InvalidInput(format!("paired tensor {side} address overflow")))
}

pub(super) fn charge_work(work: &mut usize, additional: usize) -> Result<(), AkitaError> {
    *work = work
        .checked_add(additional)
        .ok_or_else(|| AkitaError::InvalidInput("paired tensor work overflow".into()))?;
    if *work > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: *work,
        });
    }
    Ok(())
}
