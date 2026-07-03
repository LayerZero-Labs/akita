use akita_challenges::{SparseChallenge, TensorChallenges as TensorChallengeSet};
use akita_field::AkitaError;

pub(crate) fn validate_tensor_blocks<const D: usize>(
    tensor: &TensorChallengeSet,
    expected_blocks: usize,
) -> Result<usize, AkitaError> {
    tensor.validate::<D>()?;
    let blocks_per_claim = tensor.blocks_per_claim()?;
    let actual_blocks = tensor.total_blocks()?;
    if actual_blocks != expected_blocks {
        return Err(AkitaError::InvalidSize {
            expected: expected_blocks,
            actual: actual_blocks,
        });
    }
    Ok(blocks_per_claim)
}

pub(crate) fn sparse_i8_mul_acc_i64<const D: usize>(
    digit_plane: &[i8; D],
    challenge: &SparseChallenge,
    acc: &mut [i64; D],
) {
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let p = pos as usize;
        let split = D - p;
        let coeff = i64::from(coeff);
        for i in 0..split {
            acc[i + p] += coeff * i64::from(digit_plane[i]);
        }
        for i in split..D {
            acc[i - split] -= coeff * i64::from(digit_plane[i]);
        }
    }
}

pub(crate) fn sparse_i64_mul_acc_i64<const D: usize>(
    input: &[i64; D],
    challenge: &SparseChallenge,
    acc: &mut [i64; D],
) {
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let p = pos as usize;
        let split = D - p;
        let coeff = i64::from(coeff);
        for i in 0..split {
            acc[i + p] += coeff * input[i];
        }
        for i in split..D {
            acc[i - split] -= coeff * input[i];
        }
    }
}

pub(crate) fn fill_rotated_sparse_challenge_i64<const D: usize>(
    table: &mut [[i64; D]],
    challenge: &SparseChallenge,
) {
    debug_assert!(D.is_power_of_two());
    debug_assert!(table.len() >= D);

    let mut dense = [0i64; D];
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        dense[pos as usize] = i64::from(coeff);
    }

    for (shift, row) in table.iter_mut().enumerate().take(D) {
        row[shift..D].copy_from_slice(&dense[..D - shift]);
        for (dst, src) in row[..shift].iter_mut().zip(dense[D - shift..].iter()) {
            *dst = -*src;
        }
    }
}

pub(crate) fn narrow_tensor_accum_to_i32<const D: usize>(
    accum_i64: Vec<[i64; D]>,
) -> Result<Vec<[i32; D]>, AkitaError> {
    let mut out = Vec::with_capacity(accum_i64.len());
    for row in accum_i64 {
        let mut narrowed = [0i32; D];
        for (dst, src) in narrowed.iter_mut().zip(row.iter()) {
            *dst = i32::try_from(*src).map_err(|_| {
                AkitaError::InvalidSetup(format!(
                    "tensor fold accumulator overflowed i32 envelope (value = {src})"
                ))
            })?;
        }
        out.push(narrowed);
    }
    Ok(out)
}
