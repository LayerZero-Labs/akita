use super::{RingValueSource, RingValues};
use crate::sources::poly_helpers::{
    fused_evaluate_and_fold_base, fused_evaluate_and_fold_materialized,
};
use akita_error::AkitaError;
use jolt_algebra::checked;
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::CyclotomicRing;
use jolt_algebra::{CanonicalEncoding, Field};

pub(crate) fn evaluate_and_fold_base<F, S, const D: usize>(
    source: &S,
    live_block_weights: &[F],
    position_weights: &[F],
    num_positions_per_block: usize,
    num_live_blocks: usize,
) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: Field + CanonicalEncoding,
    S: RingValueSource<F, D>,
{
    let folded = cfg_into_iter!(0..num_live_blocks)
        .map(|block_idx| {
            let mut acc = [F::zero(); D];
            for (col_idx, &scalar) in position_weights
                .iter()
                .take(num_positions_per_block)
                .enumerate()
            {
                let Some(index) = checked::product([block_idx, num_positions_per_block])
                    .and_then(|base| checked::sum([base, col_idx]))
                    .filter(|&index| index < source.num_rings())
                else {
                    break;
                };
                let ring = source.values(index)?;
                for (k, coefficient) in acc.iter_mut().enumerate() {
                    let value = ring.coeff(k);
                    if !<S::Values<'_> as RingValues<F, D>>::SKIPS_ZEROS || value != F::zero() {
                        *coefficient += scalar * value;
                    }
                }
            }
            Ok(CyclotomicRing::from_coefficients(acc))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    Ok(fused_evaluate_and_fold_base(folded, live_block_weights))
}

pub(crate) fn evaluate_and_fold_ring<F, S, const D: usize>(
    source: &S,
    live_block_weights: &[CyclotomicRing<F, D>],
    position_weights: &[CyclotomicRing<F, D>],
    num_positions_per_block: usize,
    num_live_blocks: usize,
) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError>
where
    F: Field + CanonicalEncoding,
    S: RingValueSource<F, D>,
{
    let folded = cfg_into_iter!(0..num_live_blocks)
        .map(|block_idx| {
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (col_idx, scalar) in position_weights
                .iter()
                .take(num_positions_per_block)
                .enumerate()
            {
                let Some(index) = checked::product([block_idx, num_positions_per_block])
                    .and_then(|base| checked::sum([base, col_idx]))
                    .filter(|&index| index < source.num_rings())
                else {
                    break;
                };
                let values = source.values(index)?;
                let ring = CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|k| {
                    values.coeff(k)
                }));
                ring.mul_accumulate_sparse_rhs_into(scalar, &mut acc);
            }
            Ok(acc)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    Ok(fused_evaluate_and_fold_materialized(
        folded,
        live_block_weights,
    ))
}
