use super::{PrefixSource, Shape};
use akita_algebra::{ring::cyclotomic::BalancedDecomposePow2Params, CyclotomicRing};
use akita_error::{checked, AkitaError};
use akita_params::FlatMatrix;
use akita_types::{prepare_ntt_cache, NttCacheMode};
#[allow(unused_imports)]
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::ops::Range;

/// A bounded matrix tile and one source tile per worker. Cached leading
/// contributions skip both source generation and decomposition.
pub(super) fn products<F: Field + CanonicalEncoding, const D: usize>(
    source_setup: PrefixSource<'_, F>,
    shape: Shape,
    output: &mut [F],
    reused: &[usize],
    source: impl Fn(usize, Range<usize>) -> Result<Vec<F>, AkitaError> + Sync,
) -> Result<(), AkitaError> {
    let stride = checked::product([shape.rows, D]).ok_or(AkitaError::InvalidProof)?;
    let digit_width = checked::product([shape.digits, D]).ok_or(AkitaError::InvalidProof)?;
    let params = BalancedDecomposePow2Params::<F>::new(shape.digits, shape.basis);
    let tile_positions = ((1 << 16) / digit_width).max(1);
    let bound = akita_params::balanced_signed_digit_abs_bound(shape.basis)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid prefix digit basis".into()))?;
    for row in 0..shape.rows {
        let mut begin = 0;
        while begin < shape.positions {
            let mut end = (begin + tile_positions).min(shape.positions);
            // A source range is entirely cached or entirely missing for each
            // output row, even when rows reuse different prefix widths.
            for &prefix in reused {
                if prefix > begin {
                    end = end.min(prefix);
                }
            }
            let active =
                |block: usize| reused.get(block * shape.rows + row).copied().unwrap_or(0) < end;
            if (0..output.len() / stride).any(active) {
                let columns = (end - begin) * shape.digits;
                let matrix_begin = (row * shape.positions + begin) * digit_width;
                let matrix = FlatMatrix::from_flat_data(
                    source_setup.read(matrix_begin..matrix_begin + columns * D)?,
                );
                let ntt = prepare_ntt_cache(
                    matrix.ring_view::<D>(1, columns)?,
                    NttCacheMode::ExactNegacyclic {
                        width: columns,
                        rhs_abs_bound: bound,
                    },
                )?;
                drop(matrix);
                cfg_chunks_mut!(output, stride).enumerate().try_for_each(
                    |(block, image)| -> Result<(), AkitaError> {
                        if !active(block) {
                            return Ok(());
                        }
                        let coefficients = source(block, begin..end)?;
                        if coefficients.len() != (end - begin) * D {
                            return Err(AkitaError::InvalidSetup(
                                "streamed prefix tile has the wrong size".into(),
                            ));
                        }
                        let mut digits = vec![[0i16; D]; columns];
                        for (coefficients, planes) in coefficients
                            .chunks_exact(D)
                            .zip(digits.chunks_exact_mut(shape.digits))
                        {
                            CyclotomicRing::<F, D>::from_slice(coefficients)
                                .balanced_decompose_pow2_i16_into(planes, &params);
                        }
                        drop(coefficients);
                        let product = ntt.mat_vec_i16::<F>(shape.basis, 1, &digits)?;
                        let value = product.first().ok_or(AkitaError::InvalidProof)?;
                        let target = image
                            .get_mut(row * D..(row + 1) * D)
                            .ok_or(AkitaError::InvalidProof)?;
                        for (target, value) in target.iter_mut().zip(value.coefficients()) {
                            *target += *value;
                        }
                        Ok(())
                    },
                )?;
            }
            begin = end;
        }
    }
    Ok(())
}
