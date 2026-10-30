//! Typed A product weights and coefficient order for the ordinary Stage 3.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::trinomial::adjoint::TrinomialAdjoint;
use akita_algebra::ring::trinomial::MinusTrinomial;
use akita_error::{checked, AkitaError};
use akita_params::SetupPrefixSlotId;
use akita_serialization::AkitaSerialize;
use jolt_field::Field;

use super::{binary_root_matrix, BinaryCommittedGroupManifest, BinaryRootWitnessPlan};

const DEGREE: usize = 648;
const STRIDE: usize = 1024;

/// Coefficient functional for the committed-quotient relation.
///
/// The offload itself accepts arbitrary per-row `rho`; replacing the quotient
/// with other linear tests changes only the caller constructing this input.
pub fn binary_a_rho<E: Field>(alpha: E, batches: &[E]) -> Vec<[E; DEGREE]> {
    batches
        .iter()
        .map(|&batch| {
            let mut power = batch;
            core::array::from_fn(|_| {
                let value = power;
                power *= alpha;
                value
            })
        })
        .collect()
}

/// Canonical typed prefix: source row, padded rank row, padded coefficient.
/// All padding is zero, including any extra capacity of the selected prefix.
/// This function is for setup preparation and proving, never online verification.
pub fn binary_a_setup_coefficients<F: Field>(
    seed: &[u8; 32],
    manifest: &BinaryCommittedGroupManifest,
    prefix_len: usize,
) -> Result<Vec<F>, AkitaError> {
    manifest.validate()?;
    let rows = checked::product([4, manifest.descriptor().index_map().columns()])
        .and_then(|width| {
            checked::exact_div(manifest.descriptor().index_map().padded_carriers(), width)
        })
        .ok_or(AkitaError::InvalidProof)?;
    let row_stride = coefficient_len(manifest.a_rank())?;
    validate_prefix(rows, row_stride, prefix_len)?;
    let matrix = binary_root_matrix(seed, manifest)?;
    let mut coefficients = field_buffer(prefix_len)?;
    for (i, row) in matrix.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            let offset = checked::product([i, STRIDE])
                .and_then(|offset| checked::mul_add(j, row_stride, offset))
                .ok_or(AkitaError::InvalidProof)?;
            let range = checked::range(offset, DEGREE).ok_or(AkitaError::InvalidProof)?;
            for (slot, &value) in coefficients
                .get_mut(range)
                .ok_or(AkitaError::InvalidProof)?
                .iter_mut()
                .zip(entry)
            {
                *slot = F::from_i128(value);
            }
        }
    }
    Ok(coefficients)
}

/// Bind the stream label, complete manifest, layout version and scheduled slot.
/// The prepared prefix commitment is bound again by ordinary grouped recursion.
pub fn binary_a_prefix_identity(
    manifest: &BinaryCommittedGroupManifest,
    slot: &SetupPrefixSlotId,
) -> Result<Vec<u8>, AkitaError> {
    manifest.validate()?;
    let mut bytes = b"akita/binary-source/typed-a-prefix/j-rank-coefficient/v1".to_vec();
    bytes.extend(manifest.canonical_bytes());
    slot.serialize_compressed(&mut bytes)
        .map_err(|_| AkitaError::InvalidProof)?;
    Ok(bytes)
}

/// Factored setup weights at the root Stage 2 point, with no A coefficients.
///
/// The response occupies the first segment, with four digits and coefficient
/// stride 1024. Its digit contraction is included in `h`; the remaining point
/// addresses source rows. The signed reduction adjoint uses two exact length
/// 1295 convolutions per rank row, independently of the matrix width.
#[derive(Clone, Debug)]
pub struct BinaryAProduct<E: Field> {
    coefficient_factor: Vec<E>,
    row_point: Vec<E>,
    rows: usize,
    prefix_rows: usize,
}

impl<E: Field> BinaryAProduct<E> {
    pub fn new(
        plan: &BinaryRootWitnessPlan,
        rho: &[[E; DEGREE]],
        stage2_point: &[E],
        prefix_len: usize,
    ) -> Result<Self, AkitaError> {
        let response = plan
            .repacking()
            .segments()
            .first()
            .ok_or(AkitaError::InvalidProof)?;
        if rho.len() != plan.rank()
            || plan.descriptor().polynomials().len() != 1
            || response.digits != 4
            || response.geometry.physical_coefficient_width() != STRIDE
            || stage2_point.len() != plan.repacking().domain_len().trailing_zeros() as usize
            || plan.repacking().segment_range(0)?.start != 0
        {
            return Err(AkitaError::InvalidInput(
                "invalid typed A product geometry".into(),
            ));
        }
        let row_stride = coefficient_len(plan.rank())?;
        validate_prefix(response.elements, row_stride, prefix_len)?;
        let coefficient_point = stage2_point.get(..10).ok_or(AkitaError::InvalidProof)?;
        let digit_point = stage2_point.get(10..12).ok_or(AkitaError::InvalidProof)?;
        let row_point = stage2_point
            .get(12..)
            .ok_or(AkitaError::InvalidProof)?
            .to_vec();
        let v = EqPolynomial::evals_prefix(coefficient_point, DEGREE)?;
        let digits = EqPolynomial::evals(digit_point)?;
        let mut power = E::one();
        let mut digit_weight = E::zero();
        for value in digits {
            digit_weight += value * power;
            power *= E::from_u64(8);
        }
        let adjoint = TrinomialAdjoint::<DEGREE, MinusTrinomial>::new()?;
        let mut coefficient_factor = field_buffer(row_stride)?;
        for (i, row) in rho.iter().enumerate() {
            let offset = checked::product([i, STRIDE]).ok_or(AkitaError::InvalidProof)?;
            let range = checked::range(offset, DEGREE).ok_or(AkitaError::InvalidProof)?;
            for (slot, weight) in coefficient_factor
                .get_mut(range)
                .ok_or(AkitaError::InvalidProof)?
                .iter_mut()
                .zip(adjoint.apply(row, &v)?)
            {
                *slot = digit_weight * weight;
            }
        }
        Ok(Self {
            coefficient_factor,
            row_point,
            rows: response.elements,
            prefix_rows: prefix_len / row_stride,
        })
    }

    /// Prover factors for the existing rectangular Stage 3 kernel.
    pub fn factors(&self) -> Result<(Vec<E>, Vec<E>), AkitaError> {
        let mut index = field_buffer(self.prefix_rows)?;
        for (j, value) in index.iter_mut().take(self.rows).enumerate() {
            *value = eq_eval_at_index(&self.row_point, j);
        }
        Ok((index, self.coefficient_factor.clone()))
    }

    /// Online terminal weight: O(rank * 1024 + log(prefix_len)), without A.
    pub fn evaluate(&self, point: &[E]) -> Result<E, AkitaError> {
        let bits = self.coefficient_factor.len().trailing_zeros() as usize;
        let row_bits = self.prefix_rows.trailing_zeros() as usize;
        if point.len() != checked::sum([bits, row_bits]).ok_or(AkitaError::InvalidProof)? {
            return Err(AkitaError::InvalidProof);
        }
        let coefficient_point = point.get(..bits).ok_or(AkitaError::InvalidProof)?;
        let row_point = point.get(bits..).ok_or(AkitaError::InvalidProof)?;
        let eq = EqPolynomial::evals(coefficient_point)?;
        let coefficient = self
            .coefficient_factor
            .iter()
            .zip(eq)
            .fold(E::zero(), |acc, (&a, b)| acc + a * b);
        Ok(coefficient * truncated_equality_product(&self.row_point, row_point, self.rows)?)
    }
}

fn coefficient_len(rank: usize) -> Result<usize, AkitaError> {
    checked::ceil_log2(rank)
        .and_then(checked::pow2)
        .and_then(|rank| checked::product([rank, STRIDE]))
        .filter(|_| rank > 0)
        .ok_or_else(|| AkitaError::InvalidInput("typed A rank overflow".into()))
}

fn validate_prefix(rows: usize, stride: usize, prefix_len: usize) -> Result<(), AkitaError> {
    let required = checked::product([rows, stride]).ok_or(AkitaError::InvalidProof)?;
    if rows == 0 || !prefix_len.is_power_of_two() || prefix_len < required || prefix_len < stride {
        return Err(AkitaError::InvalidInput(
            "typed A exceeds the selected prefix domain".into(),
        ));
    }
    Ok(())
}

fn field_buffer<F: Field>(len: usize) -> Result<Vec<F>, AkitaError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidInput("typed A allocation failed".into()))?;
    values.resize(len, F::zero());
    Ok(values)
}

// Sum eq(a,j) eq(b,j) over the real rows, without expansion or division.
fn truncated_equality_product<E: Field>(a: &[E], b: &[E], rows: usize) -> Result<E, AkitaError> {
    let bits = a.len().max(b.len());
    let capacity = checked::pow2(bits).ok_or(AkitaError::InvalidProof)?;
    if rows > checked::pow2(a.len()).ok_or(AkitaError::InvalidProof)?
        || rows > checked::pow2(b.len()).ok_or(AkitaError::InvalidProof)?
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut lower = vec![E::one()];
    let mut branches = Vec::new();
    for bit in 0..bits {
        let x = a.get(bit).copied().unwrap_or(E::zero());
        let y = b.get(bit).copied().unwrap_or(E::zero());
        let left = (E::one() - x) * (E::one() - y);
        let right = x * y;
        lower.push(*lower.last().ok_or(AkitaError::InvalidProof)? * (left + right));
        branches.push((left, right));
    }
    if rows == capacity {
        return lower.last().copied().ok_or(AkitaError::InvalidProof);
    }
    let mut result = E::zero();
    let mut equal = E::one();
    for bit in (0..bits).rev() {
        let &(left, right) = branches.get(bit).ok_or(AkitaError::InvalidProof)?;
        if (rows >> bit) & 1 != 0 {
            result += equal * left * lower.get(bit).copied().ok_or(AkitaError::InvalidProof)?;
            equal *= right;
        } else {
            equal *= left;
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "a_product_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "a_product_measurements.rs"]
mod measurements;
