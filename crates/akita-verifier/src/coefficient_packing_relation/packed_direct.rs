//! Contract direct-opening digits once per point, then stream physical blocks.
//! Claim and witness-unit offsets stay in the authenticated physical geometry.
//! The left address factors as `coefficient | role | block`, and the right
//! address as `coefficient | row`, with the coefficient bits least significant.

use akita_algebra::offset_eq::{
    eval_boolean_pair_tensor_families, EqPairTensorAxis, EqPairTensorFamily, PreparedDigits,
    MAX_COMPACT_STRIDE_TERMS, MAX_PREPARED_STRIDE,
};
use akita_error::AkitaError;
use akita_params::{basis_weights, BasisMode};
use akita_types::ValidatedCoefficientPackingGroup;
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::sync::Arc;

/// Bound the materialized block-weight table to at most 2^18 field elements.
const MAX_BLOCK_BITS: usize = 18;

/// Direct-opening geometry selected once, before any evaluation points arrive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PackedDirect<E: Field> {
    /// Shared low coefficient coordinates on the left and right.
    coefficient_bits: usize,
    /// Remaining source-plane coordinates spanning D-role subcolumns.
    role_bits: usize,
    /// Physical D-role coefficients per shared coefficient block.
    digit_stride: usize,
    /// Adjacent digit addresses in one prepared row.
    width: usize,
    /// Opening gadget shared by every claim and witness unit.
    digits: Arc<[E]>,
    /// Extension-plane weights, distinct from the source polynomial's basis mode.
    plane_basis: Vec<E>,
    /// Group-wide direct-opening claim weight.
    scalar: E,
    units: Vec<Unit<E>>,
}

/// Live physical block window with a weighted address base for each claim.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Unit<E: Field> {
    /// First global block, indexing the source block weights.
    first: usize,
    /// Number of live physical blocks in this unit.
    rows: usize,
    /// Physical addresses after removing coefficient bits, with claim weights.
    bases: Vec<(usize, E)>,
}

fn invalid() -> AkitaError {
    AkitaError::InvalidInput("invalid packed direct-opening geometry".into())
}

impl<E: Field> PackedDirect<E> {
    /// Select packed evaluation only when all fixed point dimensions and row
    /// widths fit its bounded tables. Unsupported shapes retain tensor families.
    pub(super) fn new<F: Field + CanonicalEncoding>(
        group: &ValidatedCoefficientPackingGroup<'_, '_, F, E>,
        digits: Arc<[E]>,
        claims: &[E],
    ) -> Result<Option<Self>, AkitaError>
    where
        E: ExtField<F>,
    {
        let challenge_subring_dim = group.geometry().challenge_subring_dimension();
        let role_ring_dim = group.d_d();
        let coefficient_count = challenge_subring_dim.min(role_ring_dim);
        if !challenge_subring_dim.is_power_of_two() || !role_ring_dim.is_power_of_two() {
            return Ok(None);
        }
        let width = group
            .geometry()
            .partial_base_field_width()
            .checked_div(coefficient_count)
            .and_then(|n| n.checked_mul(digits.len()))
            .ok_or_else(invalid)?;
        if width == 0 || width > MAX_PREPARED_STRIDE {
            return Ok(None);
        }
        let coefficient_bits = coefficient_count.trailing_zeros() as usize;
        let role_bits = (challenge_subring_dim / coefficient_count).trailing_zeros() as usize;
        let left_bits = group
            .prepared_point()
            .tail_point()
            .len()
            .checked_add(group.prepared_point().block_point().len())
            .ok_or_else(invalid)?;
        let block_bits = left_bits
            .checked_sub(coefficient_bits + role_bits)
            .ok_or_else(invalid)?;
        let right_bits = group
            .physical_field_len()
            .checked_next_power_of_two()
            .ok_or_else(invalid)?
            .trailing_zeros() as usize;
        let row_bits = right_bits
            .checked_sub(coefficient_bits)
            .ok_or_else(invalid)?;
        if block_bits > MAX_BLOCK_BITS
            || row_bits >= usize::BITS as usize
            || (width.next_power_of_two().trailing_zeros() as usize) > row_bits
        {
            return Ok(None);
        }
        let mut units = Vec::new();
        for unit in group
            .witness_layout()
            .units_for_group(group.group_index())?
        {
            if unit.num_live_blocks() == 0 {
                continue;
            }
            let mut bases = Vec::new();
            for (claim, &weight) in claims.iter().enumerate() {
                let base = unit.e_coefficient_index(
                    role_ring_dim,
                    group.num_claims(),
                    digits.len(),
                    claim,
                    unit.global_block_start(),
                    0,
                    0,
                    0,
                )?;
                if !base.is_multiple_of(coefficient_count) {
                    return Ok(None);
                }
                bases.push((base / coefficient_count, weight));
            }
            units.push(Unit {
                first: unit.global_block_start(),
                rows: unit.num_live_blocks(),
                bases,
            });
        }
        let odd_stride = width >> width.trailing_zeros();
        if group.num_claims() == 1
            && (!matches!(odd_stride, 1 | 3 | 5 | 7) || !units.iter().any(|unit| unit.rows >= 128))
        {
            return Ok(None);
        }
        Ok(Some(Self {
            coefficient_bits,
            role_bits,
            digit_stride: role_ring_dim / coefficient_count,
            width,
            digits,
            plane_basis: group.basis().to_vec(),
            scalar: group.scalar_claim_weight(),
            units,
        }))
    }

    /// Evaluate the selected packed geometry. Failure to construct its prepared
    /// window is an invariant error; evaluation never switches representations.
    pub(super) fn evaluate(
        &self,
        basis: BasisMode,
        left: &[E],
        right: &[E],
    ) -> Result<E, AkitaError> {
        let _search = akita_error::experiment_scope("search/packed_direct");
        let (left_coefficient, left_tail) = left
            .split_at_checked(self.coefficient_bits)
            .ok_or_else(invalid)?;
        let (role_point, block_point) = left_tail
            .split_at_checked(self.role_bits)
            .ok_or_else(invalid)?;
        let (right_coefficient, row_point) = right
            .split_at_checked(self.coefficient_bits)
            .ok_or_else(invalid)?;
        let roles = basis_weights(role_point, basis)?;
        let mut digits = vec![E::zero(); self.width];
        for (plane, &plane_weight) in self.plane_basis.iter().enumerate() {
            for (role, &role_weight) in roles.iter().enumerate() {
                let flat = plane
                    .checked_mul(roles.len())
                    .and_then(|n| n.checked_add(role))
                    .ok_or_else(invalid)?;
                let column = flat / self.digit_stride;
                let lane = flat % self.digit_stride;
                for (digit, &weight) in self.digits.iter().enumerate() {
                    let at = column
                        .checked_mul(self.digits.len())
                        .and_then(|n| n.checked_add(digit))
                        .and_then(|n| n.checked_mul(self.digit_stride))
                        .and_then(|n| n.checked_add(lane))
                        .ok_or_else(invalid)?;
                    *digits.get_mut(at).ok_or_else(invalid)? += plane_weight * role_weight * weight;
                }
            }
        }
        let mut coefficient = E::one();
        for (&left_value, &right_value) in left_coefficient.iter().zip(right_coefficient) {
            coefficient *= if basis == BasisMode::Monomial {
                (E::one() - right_value) + left_value * right_value
            } else {
                (E::one() - left_value) * (E::one() - right_value) + left_value * right_value
            };
        }
        let odd_stride = self
            .width
            .checked_shr(self.width.trailing_zeros())
            .ok_or_else(invalid)?;
        let seed_count = self.units.iter().try_fold(0usize, |count, unit| {
            count.checked_add(unit.bases.len()).ok_or_else(invalid)
        })?;
        let family_count = seed_count.checked_mul(self.width).ok_or_else(invalid)?;
        let contract = matches!(odd_stride, 1 | 3 | 5 | 7)
            && family_count <= MAX_COMPACT_STRIDE_TERMS
            && self.units.iter().any(|unit| unit.rows >= 128);
        let value = if contract {
            // Contract eq(block, first + i) * eq(row, base + width*i + digit)
            // over each live interval instead of materializing and walking blocks.
            let mut families = Vec::new();
            families
                .try_reserve_exact(family_count)
                .map_err(|_| invalid())?;
            let block_domain = akita_error::checked::pow2(block_point.len()).ok_or_else(invalid)?;
            let row_domain = akita_error::checked::pow2(row_point.len()).ok_or_else(invalid)?;
            for unit in &self.units {
                if unit
                    .first
                    .checked_add(unit.rows)
                    .is_none_or(|end| end > block_domain)
                {
                    return Err(invalid());
                }
                if unit.rows == 0 {
                    continue;
                }
                let span = unit
                    .rows
                    .checked_mul(self.width)
                    .and_then(|n| n.checked_sub(1))
                    .ok_or_else(invalid)?;
                for &(base, weight) in &unit.bases {
                    if base.checked_add(span).is_none_or(|last| last >= row_domain) {
                        return Err(invalid());
                    }
                    for (digit, &digit_weight) in digits.iter().enumerate() {
                        if digit_weight.is_zero() {
                            continue;
                        }
                        families.push(EqPairTensorFamily::new(
                            unit.first,
                            base.checked_add(digit).ok_or_else(invalid)?,
                            weight * digit_weight,
                            vec![EqPairTensorAxis::unit(unit.rows, 1, self.width)],
                        )?);
                    }
                }
            }
            match basis {
                BasisMode::Lagrange => eval_boolean_pair_tensor_families::<_, false, false>(
                    block_point,
                    row_point,
                    &families,
                )?,
                BasisMode::Monomial => eval_boolean_pair_tensor_families::<_, true, false>(
                    block_point,
                    row_point,
                    &families,
                )?,
            }
        } else {
            let blocks = basis_weights(block_point, basis)?;
            let prepared =
                PreparedDigits::<E, E>::new(row_point, &digits, &[])?.ok_or_else(invalid)?;
            let mut value = E::zero();
            for unit in &self.units {
                for &(base, weight) in &unit.bases {
                    let mut rows = E::zero();
                    prepared.accumulate(&mut rows, base, unit.first, unit.rows, &blocks)?;
                    value += weight * rows;
                }
            }
            value
        };
        Ok(self.scalar * coefficient * value)
    }
}

#[cfg(test)]
mod tests;
