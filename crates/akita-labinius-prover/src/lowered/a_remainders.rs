use akita_algebra::{TrinomialLimbAccumulator, TrinomialLimbDomain, TrinomialModulus};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::profile::BinaryClearSetup;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const DEGREE: usize = 648;

/// Exact integer `rem_Phi(sum_j A_ij * packed_j)` for every matrix row,
/// ordered by row then coefficient, using limb transforms and signed CRT.
///
/// Each unreduced coefficient contains at most `m * D` products, each bounded
/// by `(q - 1) * max|packed|`, since matrix integers are the residues in `[0,q)`.
/// For `Phi = Y^648 - Y^324 + 1`, reduction uses at most three unreduced
/// coefficients per remainder coefficient: `Y^648 = Y^324 - 1` and
/// `Y^972 = -1`. Thus `B = 3 * m * D * (q - 1) * max|packed|` bounds its
/// magnitude. Distinct admitted primes, largest first, give a product `P > 2B`
/// so that centering the CRT result recovers the same integer as the reference.
///
/// Return `None` for another degree or middle coefficient, an overflowing bound,
/// insufficient admitted primes, or a vector exceeding the reference's checked
/// `D * q * max|packed| <= i64::MAX` product limit. The caller then uses
/// `matrix_row_remainder`, including its original errors. A mismatched packed
/// length returns its `InvalidInput`; allocation failures return `InvalidInput`,
/// and invalid extents or transform failures return `InvalidProof`.
pub fn matrix_remainders_limb<const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<D, M>,
    packed: &[[i64; D]],
) -> Result<Option<Vec<i128>>, AkitaError> {
    if D != DEGREE || M::MIDDLE_COEFFICIENT != -1 {
        return Ok(None);
    }
    if packed.len() != setup.m() {
        return Err(AkitaError::InvalidInput(
            "matrix vector length mismatch".into(),
        ));
    }
    let magnitude = packed
        .iter()
        .flatten()
        .fold(0, |max, value| max.max(value.unsigned_abs()));
    let degree = D as u128;
    let narrow = degree
        .checked_mul(u128::from(setup.modulus()))
        .and_then(|bound| bound.checked_mul(u128::from(magnitude)));
    if narrow.is_none_or(|bound| bound > i64::MAX as u128) {
        return Ok(None);
    }
    let twice_bound = (setup.m() as u128)
        .checked_mul(degree)
        .and_then(|bound| bound.checked_mul(u128::from(setup.modulus() - 1)))
        .and_then(|bound| bound.checked_mul(u128::from(magnitude)))
        .and_then(|bound| bound.checked_mul(6));
    let Some(twice_bound) = twice_bound else {
        return Ok(None);
    };
    let mut primes = TrinomialLimbDomain::ADMITTED_PRIMES;
    primes.sort_unstable_by(|left, right| right.cmp(left));
    let mut product = 1u128;
    let mut count = 0;
    for (index, &prime) in primes.iter().enumerate() {
        let Some(next) = product.checked_mul(u128::from(prime)) else {
            return Ok(None);
        };
        product = next;
        count = checked::sum([index, 1]).ok_or(AkitaError::InvalidProof)?;
        if product > twice_bound {
            break;
        }
    }
    if product <= twice_bound {
        return Ok(None);
    }
    let row_len = checked::product([setup.m(), D]).ok_or(AkitaError::InvalidProof)?;
    let len = checked::product([setup.n_a(), D]).ok_or(AkitaError::InvalidProof)?;
    let mut remainders = Vec::new();
    remainders
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidInput("matrix product allocation failed".into()))?;
    remainders.resize(len, 0i128);
    let mut modulus = 1i128;
    for &prime in primes.iter().take(count) {
        let domain = TrinomialLimbDomain::new(prime).map_err(|_| AkitaError::InvalidProof)?;
        #[cfg(feature = "parallel")]
        let residues = packed
            .par_chunks(128)
            .enumerate()
            .map(|(chunk, elements)| {
                let start = checked::product([chunk, 128]).ok_or(AkitaError::InvalidProof)?;
                remainders_mod_prime(setup, &domain, row_len, start, elements)
            })
            .try_reduce(Vec::new, |mut left, right| {
                if left.is_empty() {
                    return Ok(right);
                }
                for (out, value) in left.iter_mut().zip(right) {
                    *out = centered_mod(i64::from(*out) + i64::from(value), prime);
                }
                Ok(left)
            })?;
        #[cfg(not(feature = "parallel"))]
        let residues = remainders_mod_prime(setup, &domain, row_len, 0, packed)?;
        let prime = i128::from(prime);
        let inverse = inverse_mod_prime(modulus, prime);
        for (out, residue) in remainders.iter_mut().zip(residues) {
            // out is canonical modulo modulus; the multiplier is in [0, prime).
            let multiplier =
                ((i128::from(residue) - out.rem_euclid(prime)) * inverse).rem_euclid(prime);
            *out = modulus
                .checked_mul(multiplier)
                .and_then(|term| out.checked_add(term))
                .ok_or(AkitaError::InvalidProof)?;
        }
        modulus = modulus.checked_mul(prime).ok_or(AkitaError::InvalidProof)?;
    }
    for value in &mut remainders {
        if *value > modulus / 2 {
            *value -= modulus;
        }
    }
    Ok(Some(remainders))
}

// All four primes are below 2^28, so the centered residue fits in i32.
fn centered_mod(value: i64, prime: u32) -> i32 {
    let prime = i64::from(prime);
    let residue = if value <= -prime || value >= prime {
        value % prime
    } else {
        value
    };
    (if residue > prime / 2 {
        residue - prime
    } else if residue < -prime / 2 {
        residue + prime
    } else {
        residue
    }) as i32
}

// Fermat inversion for distinct admitted primes. Products are below 2^56.
fn inverse_mod_prime(value: i128, prime: i128) -> i128 {
    let mut base = value.rem_euclid(prime);
    let mut exponent = prime - 2;
    let mut inverse = 1;
    while exponent > 0 {
        if exponent % 2 == 1 {
            inverse = inverse * base % prime;
        }
        base = base * base % prime;
        exponent /= 2;
    }
    inverse
}

fn remainders_mod_prime<const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<D, M>,
    domain: &TrinomialLimbDomain,
    row_len: usize,
    start: usize,
    packed: &[[i64; D]],
) -> Result<Vec<i32>, AkitaError> {
    let mut accumulators = Vec::new();
    accumulators
        .try_reserve_exact(setup.n_a())
        .map_err(|_| AkitaError::InvalidInput("matrix product allocation failed".into()))?;
    accumulators.resize(setup.n_a(), TrinomialLimbAccumulator::new(domain));
    let mut coefficients = [0i32; DEGREE];
    let mut response = domain.zero_slots();
    let mut matrix = domain.zero_slots();
    for (element, values) in packed.iter().enumerate() {
        for (out, &value) in coefficients.iter_mut().zip(values) {
            *out = centered_mod(value, domain.prime());
        }
        domain
            .forward_centered(&coefficients, &mut response)
            .map_err(|_| AkitaError::InvalidProof)?;
        let offset = checked::sum([start, element])
            .and_then(|element| checked::product([element, D]))
            .ok_or(AkitaError::InvalidProof)?;
        let range = checked::range(offset, D).ok_or(AkitaError::InvalidProof)?;
        for (accumulator, row) in accumulators
            .iter_mut()
            .zip(setup.matrix().chunks_exact(row_len))
        {
            let element = row.get(range.clone()).ok_or(AkitaError::InvalidProof)?;
            for (out, &value) in coefficients.iter_mut().zip(element) {
                // Preserve the integer in [0,q), reducing only modulo this prime.
                *out = centered_mod(i64::from(value), domain.prime());
            }
            domain
                .forward_centered(&coefficients, &mut matrix)
                .map_err(|_| AkitaError::InvalidProof)?;
            accumulator
                .add_product(&matrix, &response)
                .map_err(|_| AkitaError::InvalidProof)?;
        }
    }
    let len = checked::product([setup.n_a(), D]).ok_or(AkitaError::InvalidProof)?;
    let mut residues = Vec::new();
    residues
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidInput("matrix product allocation failed".into()))?;
    residues.resize(len, 0);
    for (accumulator, row) in accumulators.iter_mut().zip(residues.chunks_exact_mut(D)) {
        accumulator
            .finish(&mut matrix)
            .map_err(|_| AkitaError::InvalidProof)?;
        domain
            .inverse_centered(&matrix, row)
            .map_err(|_| AkitaError::InvalidProof)?;
    }
    Ok(residues)
}
