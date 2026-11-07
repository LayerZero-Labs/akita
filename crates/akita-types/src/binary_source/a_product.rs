//! Typed A product weights and coefficient order for the ordinary Stage 3.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::trinomial::adjoint::TrinomialAdjoint;
use akita_algebra::ring::trinomial::MinusTrinomial;
use akita_error::{checked, AkitaError};
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::Field;

use super::{BinaryCommittedGroupManifest, BinaryIndexMap, BinaryRootWitnessPlan};

pub(super) const DEGREE: usize = 648;
pub(super) const STRIDE: usize = 1024;

/// Size of the canonical typed A prefix: one row per source ring row, each
/// holding the rank axis padded to a power of two times 1024 coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryAPrefixLen {
    /// Source ring rows, `padded_carriers / (4 * columns)`.
    pub rows: usize,
    /// Cells per source row: the padded rank times the coefficient stride.
    pub row_stride: usize,
    /// Natural cells of the padded rank/coefficient tensor.
    pub natural_len: usize,
    /// Smallest power-of-two prefix domain that holds the natural cells.
    pub domain_len: usize,
}

impl BinaryAPrefixLen {
    /// Size the typed A prefix of an index map and A rank.
    pub fn new(index: &BinaryIndexMap, rank: usize) -> Result<Self, AkitaError> {
        let rows = checked::product([4, index.columns()])
            .and_then(|width| checked::exact_div(index.padded_carriers(), width))
            .filter(|&rows| rows > 0)
            .ok_or(AkitaError::InvalidProof)?;
        let row_stride = checked::ceil_log2(rank)
            .and_then(checked::pow2)
            .and_then(|rank| checked::product([rank, STRIDE]))
            .filter(|_| rank > 0)
            .ok_or_else(|| AkitaError::InvalidInput("typed A rank overflow".into()))?;
        let natural_len = checked::product([rows, row_stride]).ok_or(AkitaError::InvalidProof)?;
        let domain_len = checked::ceil_log2(natural_len)
            .and_then(checked::pow2)
            .ok_or(AkitaError::InvalidProof)?;
        Ok(Self {
            rows,
            row_stride,
            natural_len,
            domain_len,
        })
    }

    /// Accept a selected prefix domain that holds the typed A tensor.
    pub fn validate_prefix(&self, prefix_len: usize) -> Result<(), AkitaError> {
        if !prefix_len.is_power_of_two() || prefix_len < self.domain_len {
            return Err(AkitaError::InvalidInput(
                "typed A exceeds the selected prefix domain".into(),
            ));
        }
        Ok(())
    }
}

/// Canonical A part of the typed prefix: source row, padded rank row, padded
/// coefficient. Every other cell is zero; [`super::BinaryTypedPrefix`] adds
/// the B/D setup. `matrix` is [`binary_root_matrix`](super::binary_root_matrix)
/// for the same manifest; a caller that already holds it does not derive it
/// again. This function is for setup preparation and proving, never online
/// verification.
pub fn binary_a_setup_coefficients<F: Field>(
    matrix: &[Vec<[i128; DEGREE]>],
    manifest: &BinaryCommittedGroupManifest,
    prefix_len: usize,
) -> Result<Vec<F>, AkitaError> {
    manifest.validate()?;
    let size = BinaryAPrefixLen::new(manifest.descriptor().index_map(), manifest.a_rank())?;
    size.validate_prefix(prefix_len)?;
    if matrix.len() != manifest.a_rank() || matrix.iter().any(|row| row.len() != size.rows) {
        return Err(AkitaError::InvalidInput(
            "typed A matrix does not match its manifest".into(),
        ));
    }
    let row_stride = size.row_stride;
    // Every cell is a pure function of its offset, so the buffer fills in
    // parallel without a serial zeroing pass.
    let cell = |offset: usize| {
        let (j, within) = (offset / row_stride, offset % row_stride);
        let (i, k) = (within / STRIDE, within % STRIDE);
        matrix
            .get(i)
            .and_then(|row| row.get(j))
            .and_then(|entry| entry.get(k))
            .map_or_else(F::zero, |&value| F::from_i128(value))
    };
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(prefix_len)
        .map_err(|_| AkitaError::InvalidInput("typed A allocation failed".into()))?;
    #[cfg(feature = "parallel")]
    coefficients.par_extend((0..prefix_len).into_par_iter().map(cell));
    #[cfg(not(feature = "parallel"))]
    coefficients.extend((0..prefix_len).map(cell));
    Ok(coefficients)
}

/// Factored setup weights at the root Stage 2 point, with no A coefficients.
///
/// The response occupies the first segment, with four digits and coefficient
/// stride 1024. Its digit contraction is included in `h`; the remaining point
/// addresses source rows and owners. The owner equality is contracted into one
/// coefficient factor, so every polynomial in the group shares one product.
/// Owner rows are combined before the adjoint, which reads the coefficient
/// equality table in closed form once per rank row, independently of the
/// matrix width.
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
        let owners = plan.descriptor().polynomials().len();
        let size = BinaryAPrefixLen::new(plan.descriptor().index_map(), plan.rank())?;
        let rows = size.rows;
        if E::from_u128(plan.field_profile().modulus()) != E::zero()
            || E::from_u128(plan.field_profile().modulus() - 1) != -E::one()
            || rho.len()
                != checked::product([owners, plan.rank()]).ok_or(AkitaError::InvalidProof)?
            || checked::product([owners, rows]) != Some(response.elements)
            || !rows.is_power_of_two()
            || response.digits != 4
            || response.geometry.physical_coefficient_width() != STRIDE
            || stage2_point.len() != plan.repacking().domain_len().trailing_zeros() as usize
            || plan.repacking().segment_range(0)?.start != 0
        {
            return Err(AkitaError::InvalidInput(
                "invalid typed A product geometry".into(),
            ));
        }
        size.validate_prefix(prefix_len)?;
        let row_stride = size.row_stride;
        let coefficient_point = stage2_point.get(..10).ok_or(AkitaError::InvalidProof)?;
        let digit_point = stage2_point.get(10..12).ok_or(AkitaError::InvalidProof)?;
        let row_end =
            checked::sum([12, rows.trailing_zeros() as usize]).ok_or(AkitaError::InvalidProof)?;
        let row_point = stage2_point
            .get(12..row_end)
            .ok_or(AkitaError::InvalidProof)?
            .to_vec();
        let owner_point = stage2_point
            .get(row_end..)
            .ok_or(AkitaError::InvalidProof)?;
        if owners > checked::pow2(owner_point.len()).ok_or(AkitaError::InvalidProof)? {
            return Err(AkitaError::InvalidProof);
        }
        let digits = EqPolynomial::evals(digit_point)?;
        let mut power = E::one();
        let mut digit_weight = E::zero();
        for value in digits {
            digit_weight += value * power;
            power *= E::from_u64(8);
        }
        let adjoint = TrinomialAdjoint::<DEGREE, MinusTrinomial>::new()?;
        let mut coefficient_factor = field_buffer(row_stride)?;
        // All owners share A. Fold owner equality into rho first, so the
        // one rectangular product needs only one adjoint per matrix rank row.
        let combined_len =
            checked::product([plan.rank(), DEGREE]).ok_or(AkitaError::InvalidProof)?;
        let mut combined = field_buffer(combined_len)?;
        for (owner, owner_rho) in rho.chunks_exact(plan.rank()).enumerate() {
            let owner_weight = eq_eval_at_index(owner_point, owner);
            for (target, row) in combined.chunks_exact_mut(DEGREE).zip(owner_rho) {
                for (target, &weight) in target.iter_mut().zip(row) {
                    *target += owner_weight * weight;
                }
            }
        }
        for (target, row) in coefficient_factor
            .chunks_exact_mut(STRIDE)
            .zip(combined.chunks_exact(DEGREE))
        {
            for (slot, weight) in target
                .iter_mut()
                .zip(adjoint.apply_equality(row, coefficient_point)?)
            {
                *slot = digit_weight * weight;
            }
        }
        Ok(Self {
            coefficient_factor,
            row_point,
            rows,
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
        Ok(coefficient * truncated_equality_product(&self.row_point, row_point)?)
    }
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
fn truncated_equality_product<E: Field>(a: &[E], b: &[E]) -> Result<E, AkitaError> {
    let (shared, tail) = b
        .split_at_checked(a.len())
        .ok_or(AkitaError::InvalidProof)?;
    Ok(EqPolynomial::mle(a, shared)?
        * tail.iter().fold(E::one(), |product, &coordinate| {
            product * (E::one() - coordinate)
        }))
}

#[cfg(test)]
#[path = "a_product_tests.rs"]
mod tests;
