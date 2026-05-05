//! Prover-side sampling for commitment masking.

use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_types::{FlatDigitBlocks, Mode};
use rand_core::OsRng;

/// Sample and decompose the fresh B-blinding vector for one commitment.
///
/// # Errors
///
/// Returns an error if digit block sizing overflows.
pub(crate) fn sample_masking_factor<M, F, const D: usize>(
    output_ring_len: usize,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError>
where
    M: Mode,
    F: FieldCore + CanonicalField + RandomSampling,
{
    let blind_rings = M::blind_ring_count::<F>(output_ring_len, D);
    if blind_rings == 0 {
        return Ok(FlatDigitBlocks::empty());
    }

    let block_sizes = vec![num_digits_open; blind_rings];
    let mut out = FlatDigitBlocks::zeroed(block_sizes)?;
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits_open, log_basis, q);
    for block in out.split_blocks_mut() {
        let mask = CyclotomicRing::<F, D>::random(&mut OsRng);
        mask.balanced_decompose_pow2_i8_into_with_params(block, &decompose_params);
    }
    Ok(out)
}
