//! Prover-side sampling for commitment masking.

use akita_field::{AkitaError, CanonicalField};
use akita_types::{zk, FlatDigitBlocks};
use rand_core::{OsRng, RngCore};

fn sample_balanced_pow2_digit<R: RngCore>(rng: &mut R, log_basis: u32) -> i8 {
    // The alphabet size is a power of two, so masking low bits is uniform.
    let raw = (rng.next_u32() & ((1u32 << log_basis) - 1)) as i16;
    let half_basis = 1i16 << (log_basis - 1);
    let basis = half_basis << 1;
    let balanced = if raw >= half_basis { raw - basis } else { raw };
    balanced as i8
}

/// Sample a fresh digit-source LHL blinding vector.
///
/// # Errors
///
/// Returns an error if digit block sizing overflows.
pub(crate) fn sample_blinding_digits<F, const D: usize>(
    output_ring_len: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError>
where
    F: CanonicalField,
{
    if !(1..=8).contains(&log_basis) {
        return Err(AkitaError::InvalidInput(
            "ZK digit blinding log_basis must be in 1..=8".to_string(),
        ));
    }

    let blinding_planes = zk::blinding_digit_plane_count::<F>(output_ring_len, D, log_basis);
    if blinding_planes == 0 {
        return Ok(FlatDigitBlocks::empty());
    }

    let block_sizes = vec![blinding_planes];
    let mut out = FlatDigitBlocks::zeroed(block_sizes)?;
    let mut rng = OsRng;
    for plane in out.flat_digits_mut() {
        for coeff in plane {
            *coeff = sample_balanced_pow2_digit(&mut rng, log_basis);
        }
    }
    Ok(out)
}
