//! Exact negacyclic NTT kernels for terminal verifier matrix relations.

use akita_algebra::ntt::prime::PrimeWidth;
use akita_algebra::{
    CenteredMontLut, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut,
};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_types::{max_safe_crt_accumulation_width, AkitaVerifierSetup, VerifierNttSlot};

const MAX_I8_LOG_BASIS: u32 = 6;
const CENTERED_LUT_MAX_ABS: u64 = (1 << 16) - 1;

fn checked_matrix_prefix<T>(
    flat: &[T],
    num_rows: usize,
    num_cols: usize,
) -> Result<&[T], AkitaError> {
    let required = num_rows
        .checked_mul(num_cols)
        .ok_or(AkitaError::InvalidProof)?;
    let prefix = flat.get(..required).ok_or_else(|| {
        AkitaError::InvalidSetup("prepared verifier matrix prefix is undersized".into())
    })?;
    Ok(prefix)
}

fn safe_chunk_width<F, W, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
    full_width: usize,
    rhs_abs_bound: u64,
) -> Option<usize>
where
    F: CanonicalField,
    W: PrimeWidth,
{
    if full_width == 0 {
        return Some(0);
    }
    max_safe_crt_accumulation_width::<F, W, K, D>(params, rhs_abs_bound)
        .map(|width| width.min(full_width))
        .filter(|&width| width != 0)
}

fn accumulate_i8<F, W, const K: usize, const D: usize>(
    flat: &[CyclotomicCrtNtt<W, K, D>],
    num_rows: usize,
    num_cols: usize,
    rhs: &[[i8; D]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    if !(1..=MAX_I8_LOG_BASIS).contains(&log_basis) || rhs.len() != num_cols {
        return Err(AkitaError::InvalidProof);
    }
    let digit_bound = 1i16 << (log_basis - 1);
    if rhs
        .iter()
        .flatten()
        .any(|&digit| !(-digit_bound..digit_bound).contains(&i16::from(digit)))
    {
        return Err(AkitaError::InvalidProof);
    }
    if num_rows == 0 || num_cols == 0 {
        return Ok(vec![CyclotomicRing::zero(); num_rows]);
    }
    let matrix = checked_matrix_prefix(flat, num_rows, num_cols)?;
    let rhs_bound = (1u64 << (log_basis - 1)).max(1);
    let chunk_width = safe_chunk_width::<F, W, K, D>(params, num_cols, rhs_bound)
        .ok_or_else(|| AkitaError::InvalidSetup("CRT profile cannot fit one i8 product".into()))?;
    let lut = DigitMontLut::new_with_digit_bound(params, rhs_bound);
    let mut out = vec![CyclotomicRing::<F, D>::zero(); num_rows];
    for start in (0..num_cols).step_by(chunk_width) {
        let end = (start + chunk_width).min(num_cols);
        let mut accumulators = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
        for column in start..end {
            let digit = &rhs[column];
            if digit.iter().all(|&coefficient| coefficient == 0) {
                continue;
            }
            let transformed = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
            for (accumulator, row) in accumulators.iter_mut().zip(matrix.chunks_exact(num_cols)) {
                accumulator.add_assign_pointwise_mul_with_params(
                    &row[column],
                    &transformed,
                    params,
                );
            }
        }
        for (dst, accumulator) in out.iter_mut().zip(accumulators) {
            *dst += accumulator.to_ring_with_params(params);
        }
    }
    Ok(out)
}

fn centered_i32<F, W, const K: usize, const D: usize>(
    flat: &[CyclotomicCrtNtt<W, K, D>],
    num_rows: usize,
    num_cols: usize,
    rhs: &[[i32; D]],
    rhs_abs_bound: u64,
    chunk_width: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
{
    if rhs.len() != num_cols {
        return Err(AkitaError::InvalidProof);
    }
    if num_rows == 0 || num_cols == 0 {
        return Ok(vec![CyclotomicRing::zero(); num_rows]);
    }
    let matrix = checked_matrix_prefix(flat, num_rows, num_cols)?;
    let lut = (rhs_abs_bound <= CENTERED_LUT_MAX_ABS)
        .then(|| CenteredMontLut::new(params, rhs_abs_bound as i32));
    let mut out = vec![CyclotomicRing::<F, D>::zero(); num_rows];
    for start in (0..num_cols).step_by(chunk_width) {
        let end = (start + chunk_width).min(num_cols);
        let mut accumulators = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
        for column in start..end {
            let value = &rhs[column];
            if value.iter().all(|&coefficient| coefficient == 0) {
                continue;
            }
            let transformed = if let Some(ref lut) = lut {
                CyclotomicCrtNtt::from_centered_i32_with_lut(value, params, lut)
            } else {
                CyclotomicCrtNtt::from_centered_i32_with_params(value, params)
            };
            for (accumulator, row) in accumulators.iter_mut().zip(matrix.chunks_exact(num_cols)) {
                accumulator.add_assign_pointwise_mul_with_params(
                    &row[column],
                    &transformed,
                    params,
                );
            }
        }
        for (dst, accumulator) in out.iter_mut().zip(accumulators) {
            *dst += accumulator.to_ring_with_params(params);
        }
    }
    Ok(out)
}

fn balanced_i64_planes<const D: usize>(rhs: &[[i64; D]]) -> Vec<Vec<[i8; D]>> {
    let mut remaining = rhs
        .iter()
        .map(|ring| ring.map(i128::from))
        .collect::<Vec<_>>();
    let mut planes = Vec::new();
    while remaining
        .iter()
        .flatten()
        .any(|&coefficient| coefficient != 0)
    {
        let mut plane = vec![[0i8; D]; rhs.len()];
        for (source, digits) in remaining.iter_mut().zip(&mut plane) {
            for (coefficient, digit) in source.iter_mut().zip(digits) {
                let residue = *coefficient & 63;
                let balanced = if residue >= 32 { residue - 64 } else { residue };
                *coefficient = (*coefficient - balanced) >> 6;
                *digit = balanced as i8;
            }
        }
        planes.push(plane);
    }
    planes
}

fn balanced_i64_plane_count<const D: usize>(rhs: &[[i64; D]]) -> usize {
    rhs.iter()
        .flatten()
        .map(|&value| {
            let mut remaining = i128::from(value);
            let mut planes = 0;
            while remaining != 0 {
                let residue = remaining & 63;
                let balanced = if residue >= 32 { residue - 64 } else { residue };
                remaining = (remaining - balanced) >> 6;
                planes += 1;
            }
            planes
        })
        .max()
        .unwrap_or(0)
}

fn try_centered_i32<const D: usize>(rhs: &[[i64; D]]) -> Option<Vec<[i32; D]>> {
    let mut centered = Vec::with_capacity(rhs.len());
    for ring in rhs {
        let mut converted = [0i32; D];
        for (dst, &source) in converted.iter_mut().zip(ring) {
            *dst = i32::try_from(source).ok()?;
        }
        centered.push(converted);
    }
    Some(centered)
}

fn accumulate_centered_i64<F, W, const K: usize, const D: usize>(
    flat: &[CyclotomicCrtNtt<W, K, D>],
    num_rows: usize,
    num_cols: usize,
    rhs: &[[i64; D]],
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    W: PrimeWidth,
{
    if rhs.len() != num_cols {
        return Err(AkitaError::InvalidProof);
    }
    let actual_bound = rhs
        .iter()
        .flatten()
        .map(|&value| value.unsigned_abs())
        .max()
        .unwrap_or(0);
    if actual_bound == 0 {
        return Ok(vec![CyclotomicRing::zero(); num_rows]);
    }
    let centered_safe = safe_chunk_width::<F, W, K, D>(params, num_cols, actual_bound);
    let balanced_planes = balanced_i64_plane_count(rhs);
    let centered_chunks = centered_safe.map_or(usize::MAX, |width| num_cols.div_ceil(width));
    tracing::debug!(
        target: "akita_verifier::terminal_ntt",
        ring_d = D,
        num_rows,
        num_cols,
        rhs_abs_bound = actual_bound,
        centered_safe_width = centered_safe.unwrap_or(0),
        centered_chunks,
        balanced_planes,
        "selected exact terminal A matrix-product strategy"
    );
    if centered_chunks <= balanced_planes.max(1) {
        if let Some(chunk_width) = centered_safe {
            if let Some(centered) = try_centered_i32(rhs) {
                return centered_i32::<F, W, K, D>(
                    flat,
                    num_rows,
                    num_cols,
                    &centered,
                    actual_bound,
                    chunk_width,
                    params,
                );
            }
        }
    }

    let planes = balanced_i64_planes(rhs);
    let mut out = vec![CyclotomicRing::<F, D>::zero(); num_rows];
    let mut scale = F::one();
    let radix = F::from_i64(64);
    for plane in planes {
        let rows = accumulate_i8::<F, W, K, D>(flat, num_rows, num_cols, &plane, 6, params)?;
        for (dst, row) in out.iter_mut().zip(rows) {
            *dst += row.scale(&scale);
        }
        scale *= radix;
    }
    Ok(out)
}

/// Compute a prepared negacyclic matrix product with balanced i8 digit rings.
pub(super) fn digit_rows<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    num_rows: usize,
    digits: &[[i8; D]],
    log_basis: u32,
    prepared_prefix_len: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let required = num_rows
        .checked_mul(digits.len())
        .ok_or(AkitaError::InvalidProof)?;
    if prepared_prefix_len < required {
        return Err(AkitaError::InvalidSetup(
            "verifier B cache prefix is undersized".into(),
        ));
    }
    let slot = setup.prepared_verifier_ntt_prefix::<D>(prepared_prefix_len)?;
    match slot.as_d::<D>()? {
        VerifierNttSlot::Q32 { neg, params } => {
            accumulate_i8(neg, num_rows, digits.len(), digits, log_basis, params)
        }
        VerifierNttSlot::Q64 { neg, params } => {
            accumulate_i8(neg, num_rows, digits.len(), digits, log_basis, params)
        }
        VerifierNttSlot::Q128 { neg, params } => {
            accumulate_i8(neg, num_rows, digits.len(), digits, log_basis, params)
        }
    }
}

/// Compute a prepared negacyclic matrix product with arbitrary centered i64 rings.
pub(super) fn centered_rows<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    num_rows: usize,
    rhs: &[[i64; D]],
    prepared_prefix_len: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    let required = num_rows
        .checked_mul(rhs.len())
        .ok_or(AkitaError::InvalidProof)?;
    if prepared_prefix_len < required {
        return Err(AkitaError::InvalidSetup(
            "verifier A cache prefix is undersized".into(),
        ));
    }
    let slot = setup.prepared_verifier_ntt_prefix::<D>(prepared_prefix_len)?;
    match slot.as_d::<D>()? {
        VerifierNttSlot::Q32 { neg, params } => {
            accumulate_centered_i64(neg, num_rows, rhs.len(), rhs, params)
        }
        VerifierNttSlot::Q64 { neg, params } => {
            accumulate_centered_i64(neg, num_rows, rhs.len(), rhs, params)
        }
        VerifierNttSlot::Q128 { neg, params } => {
            accumulate_centered_i64(neg, num_rows, rhs.len(), rhs, params)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ntt::tables::{q128_primes, Q128_NUM_PRIMES};
    use akita_field::Prime128Offset275 as F;

    const D: usize = 64;

    fn matrix() -> Vec<CyclotomicRing<F, D>> {
        (0..10)
            .map(|entry| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_i64(((entry * 17 + coefficient * 5) % 31) as i64 - 15)
                }))
            })
            .collect()
    }

    fn expected(
        matrix: &[CyclotomicRing<F, D>],
        rhs: &[CyclotomicRing<F, D>],
    ) -> Vec<CyclotomicRing<F, D>> {
        matrix
            .chunks_exact(rhs.len())
            .map(|row| {
                row.iter()
                    .zip(rhs)
                    .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
                        sum + (*lhs * *rhs)
                    })
            })
            .collect()
    }

    #[test]
    fn negacyclic_digit_and_centered_kernels_match_schoolbook() {
        let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
        let matrix = matrix();
        let prepared = matrix
            .iter()
            .map(|ring| CyclotomicCrtNtt::from_ring_with_params(ring, &params))
            .collect::<Vec<_>>();

        let digits = (0..5)
            .map(|column| {
                std::array::from_fn(|coefficient| ((column + coefficient) % 17) as i8 - 8)
            })
            .collect::<Vec<_>>();
        let digit_rings = digits
            .iter()
            .map(|ring| {
                CyclotomicRing::from_coefficients(ring.map(|value| F::from_i64(i64::from(value))))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            accumulate_i8::<F, _, Q128_NUM_PRIMES, D>(&prepared, 2, 5, &digits, 6, &params,)
                .expect("digit matvec"),
            expected(&matrix, &digit_rings)
        );

        let centered = (0..5)
            .map(|column| {
                std::array::from_fn(|coefficient| {
                    ((column * 911 + coefficient * 37) % 4031) as i64 - 2015
                })
            })
            .collect::<Vec<_>>();
        let centered_rings = centered
            .iter()
            .map(|ring| CyclotomicRing::from_coefficients(ring.map(F::from_i64)))
            .collect::<Vec<_>>();
        assert_eq!(
            accumulate_centered_i64::<F, _, Q128_NUM_PRIMES, D>(
                &prepared, 2, 5, &centered, &params,
            )
            .expect("centered matvec"),
            expected(&matrix, &centered_rings)
        );
    }

    #[test]
    fn centered_kernel_covers_full_i64_via_exact_digit_fallback() {
        let params = CrtNttParamSet::<i32, Q128_NUM_PRIMES, D>::new(q128_primes());
        let matrix = matrix();
        let prepared = matrix
            .iter()
            .map(|ring| CyclotomicCrtNtt::from_ring_with_params(ring, &params))
            .collect::<Vec<_>>();
        let mut centered = vec![[0i64; D]; 5];
        centered[0][0] = i64::MAX;
        centered[1][1] = -i64::MAX;
        centered[2][2] = i64::MIN;
        let centered_rings = centered
            .iter()
            .map(|ring| CyclotomicRing::from_coefficients(ring.map(F::from_i64)))
            .collect::<Vec<_>>();
        assert_eq!(
            accumulate_centered_i64::<F, _, Q128_NUM_PRIMES, D>(
                &prepared, 2, 5, &centered, &params,
            )
            .expect("full i64 matvec"),
            expected(&matrix, &centered_rings)
        );
    }
}
