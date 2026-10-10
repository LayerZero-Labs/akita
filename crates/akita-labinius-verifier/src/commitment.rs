//! Commitment data and matrix arithmetic modulo the commitment prime.
//!
//! Products modulo `q` use the 32-bit limb transform where it exists (degree
//! 648, minus trinomial, an admitted limb prime) and exact integers otherwise.
//! Nothing here touches the proof field.

use akita_algebra::ring::TrinomialModulus;
use akita_algebra::{MinusTrinomial, TrinomialLimbAccumulator, TrinomialLimbDomain};
use akita_challenges::BinaryChallenge;
use akita_error::{checked, AkitaError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::profile::BinaryClearSetup;

/// Public images as canonical residues below the commitment prime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryClearCommitment {
    /// Coefficient `t` of the image of column `col` under matrix row `i` is at
    /// `(col * n_a + i) * D + t`.
    pub images: Vec<u32>,
}

fn zero_vec<T: Clone + Default>(len: usize) -> Result<Vec<T>, AkitaError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidInput("matrix product allocation failed".into()))?;
    values.resize(len, T::default());
    Ok(values)
}

/// Reduce an unreduced integer product modulo `Y^D + c*Y^(D/2) + 1` in place,
/// `c = M::MIDDLE_COEFFICIENT`. The low `degree` entries hold the remainder;
/// the rest are zero.
pub(crate) fn reduce_trinomial<M: TrinomialModulus>(
    product: &mut [i128],
    degree: usize,
) -> Result<(), AkitaError> {
    let overflow = || AkitaError::InvalidInput("integer trinomial reduction overflow".into());
    let half = degree / 2;
    let middle = i128::from(M::MIDDLE_COEFFICIENT);
    for n in (degree..product.len()).rev() {
        let leading = core::mem::take(product.get_mut(n).ok_or_else(overflow)?);
        // Y^n = -c * Y^(n - D/2) - Y^(n - D).
        for (index, factor) in [(n - half, middle), (n - degree, 1)] {
            let destination = product.get_mut(index).ok_or_else(overflow)?;
            *destination = factor
                .checked_mul(leading)
                .and_then(|term| destination.checked_sub(term))
                .ok_or_else(overflow)?;
        }
    }
    Ok(())
}

/// Exact `rem_Phi(sum_j A_ij * v_j)` over the integers for matrix row `row`.
///
/// `A` holds canonical residues in `[0, q)`. Every element product is
/// accumulated in `i64` under the checked bound `D * (q - 1) * max|v|`, then
/// summed and reduced in `i128`.
pub fn matrix_row_remainder<const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<D, M>,
    row: usize,
    vector: &[[i64; D]],
) -> Result<Vec<i128>, AkitaError> {
    if vector.len() != setup.m() || row >= setup.n_a() {
        return Err(AkitaError::InvalidInput(
            "matrix vector length mismatch".into(),
        ));
    }
    let magnitude = vector
        .iter()
        .flatten()
        .map(|value| value.unsigned_abs())
        .max()
        .unwrap_or(0);
    let narrow = u128::try_from(D)
        .ok()
        .and_then(|degree| degree.checked_mul(u128::from(setup.modulus())))
        .and_then(|bound| bound.checked_mul(u128::from(magnitude)))
        .is_some_and(|bound| bound <= i64::MAX as u128);
    if !narrow {
        return Err(AkitaError::InvalidInput(
            "matrix row product exceeds the integer accumulator".into(),
        ));
    }
    let row_len = checked::product([setup.m(), D]).ok_or(AkitaError::InvalidProof)?;
    let start = checked::product([row, row_len]).ok_or(AkitaError::InvalidProof)?;
    let matrix_row = setup
        .matrix()
        .get(checked::range(start, row_len).ok_or(AkitaError::InvalidProof)?)
        .ok_or(AkitaError::InvalidProof)?;
    let unreduced = checked::product([2, D])
        .and_then(|len| len.checked_sub(1))
        .ok_or(AkitaError::InvalidProof)?;
    let add_element = |mut total: Vec<i128>,
                       (element, value): (&[u32], &[i64; D])|
     -> Result<Vec<i128>, AkitaError> {
        if total.is_empty() {
            total = zero_vec(unreduced)?;
        }
        let mut product = zero_vec::<i64>(unreduced)?;
        for (s, &coefficient) in element.iter().enumerate() {
            let coefficient = i64::from(coefficient);
            let end = checked::sum([s, D]).ok_or(AkitaError::InvalidProof)?;
            for (destination, &factor) in product
                .get_mut(s..end)
                .ok_or(AkitaError::InvalidProof)?
                .iter_mut()
                .zip(value)
            {
                *destination += coefficient * factor;
            }
        }
        for (destination, value) in total.iter_mut().zip(product) {
            *destination += i128::from(value);
        }
        Ok(total)
    };
    #[cfg(feature = "parallel")]
    let mut total = matrix_row
        .par_chunks_exact(D)
        .zip(vector.par_iter())
        .try_fold(Vec::new, add_element)
        .try_reduce(Vec::new, |mut left, right| {
            if left.is_empty() {
                return Ok(right);
            }
            for (destination, value) in left.iter_mut().zip(right) {
                *destination += value;
            }
            Ok(left)
        })?;
    #[cfg(not(feature = "parallel"))]
    let mut total = matrix_row
        .chunks_exact(D)
        .zip(vector)
        .try_fold(Vec::new(), add_element)?;
    if total.is_empty() {
        total = zero_vec(unreduced)?;
    }
    reduce_trinomial::<M>(&mut total, D)?;
    total.truncate(D);
    Ok(total)
}

/// Exact `rem_Phi(sum_col iota(c_col) * T_(col,row))` over the integers.
///
/// `iota` places scalar coefficient `s` at `s * k` with the packing sign.
/// Images are read as canonical residues; the caller checks they are below `q`.
pub fn folded_image_remainder<const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<D, M>,
    commitment: &BinaryClearCommitment,
    challenges: &[BinaryChallenge],
    row: usize,
) -> Result<Vec<i128>, AkitaError> {
    let image_len =
        checked::product([setup.columns(), setup.n_a(), D]).ok_or(AkitaError::InvalidProof)?;
    if commitment.images.len() != image_len
        || challenges.len() != setup.columns()
        || row >= setup.n_a()
    {
        return Err(AkitaError::InvalidProof);
    }
    let k = setup.k();
    let unreduced = checked::product([2, D])
        .and_then(|len| len.checked_sub(1))
        .ok_or(AkitaError::InvalidProof)?;
    let mut product = zero_vec::<i128>(unreduced)?;
    for (column, challenge) in challenges.iter().enumerate() {
        let entry = checked::mul_add(column, setup.n_a(), row).ok_or(AkitaError::InvalidProof)?;
        let start = checked::product([entry, D]).ok_or(AkitaError::InvalidProof)?;
        let image = commitment
            .images
            .get(checked::range(start, D).ok_or(AkitaError::InvalidProof)?)
            .ok_or(AkitaError::InvalidProof)?;
        for term in challenge.terms() {
            let scalar = usize::from(term.position);
            let sign = if k == 1 || scalar.is_multiple_of(2) {
                i128::from(term.coefficient)
            } else {
                -i128::from(term.coefficient)
            };
            let shift = checked::product([scalar, k]).ok_or(AkitaError::InvalidProof)?;
            let end = checked::sum([shift, D]).ok_or(AkitaError::InvalidProof)?;
            for (destination, &value) in product
                .get_mut(shift..end)
                .ok_or(AkitaError::InvalidProof)?
                .iter_mut()
                .zip(image)
            {
                *destination += sign * i128::from(value);
            }
        }
    }
    reduce_trinomial::<M>(&mut product, D)?;
    product.truncate(D);
    Ok(product)
}

/// `A * vector mod (q, Phi)`: canonical residues ordered by row, then
/// coefficient.
pub fn apply_matrix<const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<D, M>,
    vector: &[[i64; D]],
) -> Result<Vec<u32>, AkitaError> {
    if vector.len() != setup.m() {
        return Err(AkitaError::InvalidInput(
            "matrix vector length mismatch".into(),
        ));
    }
    if D == 648 && M::MIDDLE_COEFFICIENT == MinusTrinomial::MIDDLE_COEFFICIENT {
        if let Ok(domain) = TrinomialLimbDomain::new(setup.modulus()) {
            return apply_matrix_limb(setup, &domain, vector);
        }
    }
    let q = i128::from(setup.modulus());
    let mut result =
        zero_vec::<u32>(checked::product([setup.n_a(), D]).ok_or(AkitaError::InvalidProof)?)?;
    for (row, image) in result.chunks_exact_mut(D).enumerate() {
        for (destination, value) in image
            .iter_mut()
            .zip(matrix_row_remainder(setup, row, vector)?)
        {
            *destination =
                u32::try_from(value.rem_euclid(q)).map_err(|_| AkitaError::InvalidProof)?;
        }
    }
    Ok(result)
}

/// The same product through the limb transform of `Z_q[Y]/(Y^648 - Y^324 + 1)`.
fn apply_matrix_limb<const D: usize, M: TrinomialModulus>(
    setup: &BinaryClearSetup<D, M>,
    domain: &TrinomialLimbDomain,
    vector: &[[i64; D]],
) -> Result<Vec<u32>, AkitaError> {
    let limb = |error: akita_algebra::TrinomialError| AkitaError::InvalidInput(error.to_string());
    let q = i64::from(domain.prime());
    let center = |value: i64| -> Result<i32, AkitaError> {
        let residue = value.rem_euclid(q);
        i32::try_from(if residue > q / 2 {
            residue - q
        } else {
            residue
        })
        .map_err(|_| AkitaError::InvalidProof)
    };
    let mut accumulators = Vec::new();
    accumulators
        .try_reserve_exact(setup.n_a())
        .map_err(|_| AkitaError::InvalidInput("matrix product allocation failed".into()))?;
    accumulators.resize(setup.n_a(), TrinomialLimbAccumulator::new(domain));
    let mut centered = zero_vec::<i32>(D)?;
    let mut matrix_slots = domain.zero_slots();
    let mut vector_slots = domain.zero_slots();
    let row_len = checked::product([setup.m(), D]).ok_or(AkitaError::InvalidProof)?;
    for (j, element) in vector.iter().enumerate() {
        for (destination, &value) in centered.iter_mut().zip(element) {
            *destination = center(value)?;
        }
        domain
            .forward_centered(&centered, &mut vector_slots)
            .map_err(limb)?;
        for (accumulator, row) in accumulators
            .iter_mut()
            .zip(setup.matrix().chunks_exact(row_len))
        {
            let start = checked::product([j, D]).ok_or(AkitaError::InvalidProof)?;
            let entry = row
                .get(checked::range(start, D).ok_or(AkitaError::InvalidProof)?)
                .ok_or(AkitaError::InvalidProof)?;
            for (destination, &value) in centered.iter_mut().zip(entry) {
                *destination = center(i64::from(value))?;
            }
            domain
                .forward_centered(&centered, &mut matrix_slots)
                .map_err(limb)?;
            accumulator
                .add_product(&matrix_slots, &vector_slots)
                .map_err(limb)?;
        }
    }
    let mut result =
        zero_vec::<u32>(checked::product([setup.n_a(), D]).ok_or(AkitaError::InvalidProof)?)?;
    for (accumulator, image) in accumulators.iter_mut().zip(result.chunks_exact_mut(D)) {
        accumulator.finish(&mut matrix_slots).map_err(limb)?;
        domain
            .inverse_centered(&matrix_slots, &mut centered)
            .map_err(limb)?;
        for (destination, &value) in image.iter_mut().zip(&centered) {
            *destination = u32::try_from(i64::from(value).rem_euclid(q))
                .map_err(|_| AkitaError::InvalidProof)?;
        }
    }
    Ok(result)
}
