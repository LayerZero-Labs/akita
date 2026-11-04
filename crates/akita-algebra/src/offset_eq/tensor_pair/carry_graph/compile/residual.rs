//! Value-independent geometry corresponding to aligned-axis contraction.
use super::{Axis, Shape, WeightKind};
use akita_error::AkitaError;

pub(super) struct ResidualShape {
    bits: [u32; 2],
    offsets: [u32; 2],
    axes: Vec<Axis>,
}

impl ResidualShape {
    pub(super) fn new(shape: Shape<'_>) -> Result<Self, AkitaError> {
        let mut result = Self {
            bits: shape.bits,
            offsets: shape.offsets,
            axes: shape.axes.to_vec(),
        };
        let mut maximum = shape.offsets;
        for axis in shape.axes {
            for (side, value) in maximum.iter_mut().enumerate() {
                *value = axis
                    .len
                    .checked_sub(1)
                    .and_then(|n| n.checked_mul(axis.strides[side]))
                    .and_then(|n| n.checked_add(*value))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("carry shape address overflow".into())
                    })?;
            }
        }
        if maximum
            .iter()
            .zip(shape.bits)
            .any(|(&v, bits)| bits >= 32 || v >= (1u32 << bits))
        {
            return Ok(result);
        }
        loop {
            let candidate = result.axes.iter().enumerate().position(|(index, axis)| {
                if axis.len < 2
                    || !axis.len.is_power_of_two()
                    || axis.strides.iter().any(|&s| s > 1)
                    || axis.strides == [0, 0]
                    || axis.kind == WeightKind::Dense
                {
                    return false;
                }
                let bits = axis.len.trailing_zeros();
                (0..2).all(|side| {
                    axis.strides[side] == 0
                        || (bits <= result.bits[side] && result.offsets[side] & (axis.len - 1) == 0)
                }) && result.axes.iter().enumerate().all(|(j, other)| {
                    j == index
                        || other.len <= 1
                        || (0..2).all(|side| {
                            axis.strides[side] == 0 || other.strides[side].is_multiple_of(axis.len)
                        })
                })
            });
            let Some(index) = candidate else { break };
            let axis = result.axes.remove(index);
            let bits = axis.len.trailing_zeros();
            for side in 0..2 {
                if axis.strides[side] == 1 {
                    result.bits[side] -= bits;
                    result.offsets[side] >>= bits;
                    for other in &mut result.axes {
                        other.strides[side] = if other.len <= 1 {
                            0
                        } else {
                            other.strides[side] >> bits
                        };
                    }
                }
            }
        }
        Ok(result)
    }

    pub(super) fn view(&self) -> Shape<'_> {
        Shape {
            bits: self.bits,
            offsets: self.offsets,
            axes: &self.axes,
        }
    }
}
