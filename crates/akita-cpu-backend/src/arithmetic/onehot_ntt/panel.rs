use super::decode;
use crate::opaque::{CommitInnerPlan, OneHotSource};
use crate::OneHotIndex;
use akita_algebra::ntt::{tables::q128_primes, I32_LAZY_DOT_BATCH};
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut};
use akita_error::{checked, AkitaError};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};

/// Bounded block/column panels reuse the existing matrix's CRT prime prefix.
pub(super) fn commit<
    F: Field + CanonicalEncoding,
    I: OneHotIndex,
    const K: usize,
    const D: usize,
>(
    matrix: &[CyclotomicCrtNtt<i32, 6, D>],
    source: OneHotSource<'_, I>,
    plan: CommitInnerPlan,
    group: usize,
    binary: bool,
    chunk_limit: usize,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let width = checked::product([plan.num_positions_per_block, plan.num_digits_inner])
        .ok_or_else(|| AkitaError::InvalidSetup("panel width overflow".into()))?;
    let params = CrtNttParamSet::<i32, K, D>::new(std::array::from_fn(|i| q128_primes()[i]));
    // validate_ring_dimension proves chunk_size divides D or vice versa.
    // With at most one hot coefficient per chunk this bounds each ring's L1.
    let l1 = u64::try_from(D.div_ceil(source.chunk_size))
        .map_err(|_| AkitaError::InvalidInput("one-hot norm overflow".into()))?;
    let chunk_width = params
        .crt_capacity()
        .max_safe_width_for_ring_l1::<F>(l1)
        .filter(|&w| w > 0)
        .ok_or_else(|| AkitaError::InvalidSetup("one-hot CRT capacity too small".into()))?
        .min(width)
        .min(chunk_limit);
    if chunk_width == 0 {
        return Err(AkitaError::InvalidInput(
            "NTT chunk limit must be positive".into(),
        ));
    }
    let expected = checked::product([plan.n_a, width])
        .ok_or_else(|| AkitaError::InvalidSetup("matrix extent overflow".into()))?;
    if matrix.len() < expected {
        return Err(AkitaError::InvalidSetup(
            "one-hot NTT matrix too short".into(),
        ));
    }
    let tile = std::env::var("AKITA_ONEHOT_NTT_COLUMNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(96);
    if !(6..=768).contains(&tile) {
        return Err(AkitaError::InvalidInput(
            "NTT column tile must be in 6..=768".into(),
        ));
    }
    let lut = DigitMontLut::new_with_digit_bound(&params, 2);
    let prefetch = std::env::var("AKITA_ONEHOT_NTT_PREFETCH").as_deref() == Ok("1");
    eprintln!("[onehot_ntt] primes={K} blocks={group} tile={tile} binary={binary} safe_chunk={chunk_width} plan={:?}", params.kernel_plan());
    let panels = cfg_into_iter!(0..plan.num_live_blocks.div_ceil(group))
        .map(|panel| {
            let first = panel * group;
            let live = group.min(plan.num_live_blocks - first);
            let mut out = vec![vec![CyclotomicRing::<F, D>::zero(); plan.n_a]; live];
            let mut digits = vec![vec![[0i8; D]; tile]; live];
            let mut rhs = vec![CyclotomicCrtNtt::<i32, K, D>::zero(); I32_LAZY_DOT_BATCH];
            for chunk_start in (0..width).step_by(chunk_width) {
                let chunk_end = (chunk_start + chunk_width).min(width);
                let mut accs = vec![vec![CyclotomicCrtNtt::<i32, K, D>::zero(); plan.n_a]; live];
                for start in (chunk_start..chunk_end).step_by(tile) {
                    let count = tile.min(chunk_end - start);
                    for (block, values) in digits.iter_mut().enumerate() {
                        decode::fill(source, plan, first + block, start, &mut values[..count])?;
                    }
                    for batch_start in (0..count).step_by(I32_LAZY_DOT_BATCH) {
                        let batch = I32_LAZY_DOT_BATCH.min(count - batch_start);
                        let column = start + batch_start;
                        if prefetch {
                            for row in 0..plan.n_a {
                                for entry in
                                    &matrix[row * width + column..row * width + column + batch]
                                {
                                    entry.prefetch();
                                }
                            }
                        }
                        for (values, rows) in digits.iter().zip(accs.iter_mut()) {
                            for (dst, digit) in rhs
                                .iter_mut()
                                .zip(&values[batch_start..batch_start + batch])
                            {
                                if binary {
                                    dst.assign_binary_with_params(digit, &params)?;
                                } else {
                                    dst.assign_i8_with_lut(digit, &params, &lut);
                                }
                            }
                            for (row, acc) in rows.iter_mut().enumerate() {
                                acc.add_assign_pointwise_dot(
                                    &matrix[row * width + column..row * width + column + batch],
                                    &rhs[..batch],
                                    &params,
                                );
                            }
                        }
                    }
                }
                for (dst, rows) in out.iter_mut().zip(accs) {
                    for (dst, row) in dst.iter_mut().zip(rows) {
                        *dst += row.to_ring(&params);
                    }
                }
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    Ok(panels.into_iter().flatten().collect())
}
