//! Typed A product weights and coefficient order for the ordinary Stage 3.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::trinomial::adjoint::TrinomialAdjoint;
use akita_algebra::trinomial::MinusTrinomial;
use akita_error::AkitaError;
use jolt_algebra::checked;
#[cfg(feature = "parallel")]
use jolt_algebra::solinas::parallel::*;
use jolt_algebra::Field;

use super::{BinaryCommittedGroupManifest, BinaryIndexMap, BinaryRootWitnessPlan};

pub(super) const DEGREE: usize = 648;
pub(super) const STRIDE: usize = 1024;

/// A rectangular A weight, factored into source-row and coefficient weights.
pub type BinaryAProductFactors<E> = (Vec<E>, Vec<E>);

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
    translated_terms: Option<Vec<TranslatedAProductTerm<E>>>,
}

/// One carry class of the low digit/coefficient lane and source-row lane.
#[derive(Clone, Debug)]
struct TranslatedAProductTerm<E: Field> {
    coefficient_factor: Vec<E>,
    row_start: usize,
    output_start: usize,
    count: usize,
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
            translated_terms: None,
        })
    }

    /// Translate the response's natural addresses into a common root table.
    ///
    /// Addition is split at the 4096-cell response-element boundary and the
    /// source-row boundary. Each boundary has at most two carry states, giving
    /// at most four rectangular terms without reading the setup matrix or
    /// expanding a source-row equality table during terminal evaluation.
    pub fn translated(
        plan: &BinaryRootWitnessPlan,
        rho: &[[E; DEGREE]],
        global_stage2_point: &[E],
        prefix_len: usize,
        response_start: usize,
    ) -> Result<Self, AkitaError> {
        let response = plan
            .repacking()
            .segments()
            .first()
            .ok_or(AkitaError::InvalidProof)?;
        let owners = plan.descriptor().polynomials().len();
        let size = BinaryAPrefixLen::new(plan.descriptor().index_map(), plan.rank())?;
        size.validate_prefix(prefix_len)?;
        let root_len = checked::pow2(global_stage2_point.len())
            .filter(|_| global_stage2_point.len() <= 32)
            .ok_or(AkitaError::InvalidProof)?;
        let element_stride = checked::product([4, STRIDE]).ok_or(AkitaError::InvalidProof)?;
        let response_len = checked::product([response.elements, element_stride])
            .ok_or(AkitaError::InvalidProof)?;
        let response_end =
            checked::sum([response_start, response_len]).ok_or(AkitaError::InvalidProof)?;
        let row_end = checked::sum([12, size.rows.trailing_zeros() as usize])
            .ok_or(AkitaError::InvalidProof)?;
        if E::from_u128(plan.field_profile().modulus()) != E::zero()
            || E::from_u128(plan.field_profile().modulus() - 1) != -E::one()
            || rho.len()
                != checked::product([owners, plan.rank()]).ok_or(AkitaError::InvalidProof)?
            || checked::product([owners, size.rows]) != Some(response.elements)
            || !size.rows.is_power_of_two()
            || response.digits != 4
            || response.geometry.physical_coefficient_width() != STRIDE
            || response_end > root_len
            || prefix_len.trailing_zeros() > 32
            || row_end > global_stage2_point.len()
        {
            return Err(AkitaError::InvalidInput(
                "invalid translated typed A product geometry".into(),
            ));
        }
        // Keep the existing specialization, including its exact factors.
        if response_start == 0
            && global_stage2_point.len() == plan.repacking().domain_len().trailing_zeros() as usize
        {
            return Self::new(plan, rho, global_stage2_point, prefix_len);
        }
        let low_point = global_stage2_point
            .get(..12)
            .ok_or(AkitaError::InvalidProof)?;
        let row_point = global_stage2_point
            .get(12..row_end)
            .ok_or(AkitaError::InvalidProof)?
            .to_vec();
        let owner_point = global_stage2_point
            .get(row_end..)
            .ok_or(AkitaError::InvalidProof)?;
        let owner_capacity = checked::pow2(owner_point.len()).ok_or(AkitaError::InvalidProof)?;
        let owner_stride =
            checked::product([element_stride, size.rows]).ok_or(AkitaError::InvalidProof)?;
        let owner_base = response_start / owner_stride;
        let row_base = (response_start / element_stride) % size.rows;
        let low_base = response_start % element_stride;
        let combined_len =
            checked::product([plan.rank(), DEGREE]).ok_or(AkitaError::InvalidProof)?;
        let adjoint = TrinomialAdjoint::<DEGREE, MinusTrinomial>::new()?;
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(4)
            .map_err(|_| AkitaError::InvalidInput("typed A allocation failed".into()))?;
        for low_carry in 0..2 {
            let mut lane = field_buffer(DEGREE)?;
            let mut active = false;
            let mut power = E::one();
            for digit in 0..4 {
                let digit_base = checked::product([digit, STRIDE])
                    .and_then(|value| checked::sum([low_base, value]))
                    .ok_or(AkitaError::InvalidProof)?;
                for (coefficient, value) in lane.iter_mut().enumerate() {
                    let address =
                        checked::sum([digit_base, coefficient]).ok_or(AkitaError::InvalidProof)?;
                    if address / element_stride == low_carry {
                        *value += power * eq_eval_at_index(low_point, address % element_stride);
                        active = true;
                    }
                }
                power *= E::from_u64(8);
            }
            if !active {
                continue;
            }
            let base = checked::sum([row_base, low_carry]).ok_or(AkitaError::InvalidProof)?;
            let first_count = size
                .rows
                .checked_sub(base)
                .ok_or(AkitaError::InvalidProof)?;
            for (row_carry, row_start, output_start, count) in
                [(0, 0, base, first_count), (1, first_count, 0, base)]
            {
                if count == 0 {
                    continue;
                }
                let mut combined = field_buffer(combined_len)?;
                for (owner, owner_rho) in rho.chunks_exact(plan.rank()).enumerate() {
                    let output = checked::sum([owner_base, owner, row_carry])
                        .filter(|&output| output < owner_capacity)
                        .ok_or(AkitaError::InvalidProof)?;
                    let weight = eq_eval_at_index(owner_point, output);
                    for (target, row) in combined.chunks_exact_mut(DEGREE).zip(owner_rho) {
                        for (target, &value) in target.iter_mut().zip(row) {
                            *target += weight * value;
                        }
                    }
                }
                let mut coefficient_factor = field_buffer(size.row_stride)?;
                for (target, row) in coefficient_factor
                    .chunks_exact_mut(STRIDE)
                    .zip(combined.chunks_exact(DEGREE))
                {
                    for (target, value) in target.iter_mut().zip(adjoint.apply(row, &lane)?) {
                        *target = value;
                    }
                }
                terms.push(TranslatedAProductTerm {
                    coefficient_factor,
                    row_start,
                    output_start,
                    count,
                });
            }
        }
        Ok(Self {
            coefficient_factor: Vec::new(),
            row_point,
            rows: size.rows,
            prefix_rows: prefix_len / size.row_stride,
            translated_terms: Some(terms),
        })
    }

    /// Prover rectangles, including every carry class of a translated response.
    pub fn factor_terms(&self) -> Result<Vec<BinaryAProductFactors<E>>, AkitaError> {
        let Some(terms) = &self.translated_terms else {
            return Ok(vec![self.factors()?]);
        };
        let mut factors = Vec::new();
        factors
            .try_reserve_exact(terms.len())
            .map_err(|_| AkitaError::InvalidInput("typed A allocation failed".into()))?;
        for term in terms {
            let mut index = field_buffer(self.prefix_rows)?;
            let end = checked::sum([term.row_start, term.count]).ok_or(AkitaError::InvalidProof)?;
            for (delta, value) in index
                .get_mut(term.row_start..end)
                .ok_or(AkitaError::InvalidProof)?
                .iter_mut()
                .enumerate()
            {
                let output =
                    checked::sum([term.output_start, delta]).ok_or(AkitaError::InvalidProof)?;
                *value = eq_eval_at_index(&self.row_point, output);
            }
            factors.push((index, term.coefficient_factor.clone()));
        }
        Ok(factors)
    }

    /// Prover factors for the existing rectangular Stage 3 kernel.
    pub fn factors(&self) -> Result<BinaryAProductFactors<E>, AkitaError> {
        if let Some(terms) = &self.translated_terms {
            if terms.len() != 1 {
                return Err(AkitaError::InvalidInput(
                    "translated typed A product has multiple rectangles".into(),
                ));
            }
            return self.factor_terms()?.pop().ok_or(AkitaError::InvalidProof);
        }
        let mut index = field_buffer(self.prefix_rows)?;
        for (j, value) in index.iter_mut().take(self.rows).enumerate() {
            *value = eq_eval_at_index(&self.row_point, j);
        }
        Ok((index, self.coefficient_factor.clone()))
    }

    /// Online terminal weight: O(rank * 1024 + log(prefix_len)), without A.
    pub fn evaluate(&self, point: &[E]) -> Result<E, AkitaError> {
        let coefficient_len = match &self.translated_terms {
            Some(terms) => terms
                .first()
                .ok_or(AkitaError::InvalidProof)?
                .coefficient_factor
                .len(),
            None => self.coefficient_factor.len(),
        };
        let bits = coefficient_len.trailing_zeros() as usize;
        let row_bits = self.prefix_rows.trailing_zeros() as usize;
        if point.len() != checked::sum([bits, row_bits]).ok_or(AkitaError::InvalidProof)? {
            return Err(AkitaError::InvalidProof);
        }
        let coefficient_point = point.get(..bits).ok_or(AkitaError::InvalidProof)?;
        let row_point = point.get(bits..).ok_or(AkitaError::InvalidProof)?;
        let eq = EqPolynomial::evals(coefficient_point)?;
        if let Some(terms) = &self.translated_terms {
            let mut value = E::zero();
            for term in terms {
                let coefficient = term
                    .coefficient_factor
                    .iter()
                    .zip(&eq)
                    .fold(E::zero(), |acc, (&a, &b)| acc + a * b);
                value += coefficient
                    * super::typed_prefix::paired_offset_eq_sum(
                        row_point,
                        term.row_start,
                        &self.row_point,
                        term.output_start,
                        term.count,
                    )?;
            }
            return Ok(value);
        }
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

#[cfg(test)]
#[path = "a_product_offset_tests.rs"]
mod offset_tests;
