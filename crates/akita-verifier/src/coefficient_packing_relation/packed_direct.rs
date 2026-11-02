//! Contract direct-opening digits once per point, then stream physical blocks.
//! Claim and witness-unit offsets stay in the authenticated physical geometry.
use akita_algebra::offset_eq::PreparedDigits;
use akita_error::AkitaError;
use akita_params::{basis_weights, BasisMode};
use akita_types::ValidatedCoefficientPackingGroup;
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::sync::Arc;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PackedDirect<E: Field> {
    coefficient_bits: usize,
    role_bits: usize,
    digit_stride: usize,
    width: usize,
    digits: Arc<[E]>,
    basis: Vec<E>,
    scalar: E,
    units: Vec<Unit<E>>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Unit<E: Field> {
    first: usize,
    rows: usize,
    bases: Vec<(usize, E)>,
}
fn invalid() -> AkitaError {
    AkitaError::InvalidInput("invalid packed direct-opening geometry".into())
}
impl<E: Field> PackedDirect<E> {
    pub(super) fn new<F: Field + CanonicalEncoding>(
        group: &ValidatedCoefficientPackingGroup<'_, '_, F, E>,
        digits: Arc<[E]>,
        claims: &[E],
    ) -> Result<Option<Self>, AkitaError>
    where
        E: ExtField<F>,
    {
        if group.num_claims() == 1 {
            return Ok(None);
        }
        let s = group.geometry().challenge_subring_dimension();
        let d = group.d_d();
        let c = s.min(d);
        if !s.is_power_of_two() || !d.is_power_of_two() {
            return Ok(None);
        }
        let width = group
            .geometry()
            .partial_base_field_width()
            .checked_div(c)
            .and_then(|n| n.checked_mul(digits.len()))
            .ok_or_else(invalid)?;
        if width == 0 || width > 512 {
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
                    d,
                    group.num_claims(),
                    digits.len(),
                    claim,
                    unit.global_block_start(),
                    0,
                    0,
                    0,
                )?;
                if !base.is_multiple_of(c) {
                    return Ok(None);
                }
                bases.push((base / c, weight));
            }
            units.push(Unit {
                first: unit.global_block_start(),
                rows: unit.num_live_blocks(),
                bases,
            });
        }
        Ok(Some(Self {
            coefficient_bits: c.trailing_zeros() as usize,
            role_bits: (s / c).trailing_zeros() as usize,
            digit_stride: d / c,
            width,
            digits,
            basis: group.basis().to_vec(),
            scalar: group.scalar_claim_weight(),
            units,
        }))
    }
    pub(super) fn evaluate(
        &self,
        basis: BasisMode,
        left: &[E],
        right: &[E],
    ) -> Result<Option<E>, AkitaError> {
        let cb = self.coefficient_bits;
        let rb = self.role_bits;
        if left.len() < cb + rb || right.len() < cb {
            return Err(invalid());
        }
        if left.len() - cb - rb > 18 {
            return Ok(None);
        }
        let roles = basis_weights(&left[cb..cb + rb], basis)?;
        let blocks = basis_weights(&left[cb + rb..], basis)?;
        let mut digits = vec![E::zero(); self.width];
        for (plane, &plane_weight) in self.basis.iter().enumerate() {
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
        let Some(prepared) = PreparedDigits::<E, E>::new(&right[cb..], &digits, &[])? else {
            return Ok(None);
        };
        let mut coefficient = E::one();
        for (&l, &r) in left[..cb].iter().zip(&right[..cb]) {
            coefficient *= if basis == BasisMode::Monomial {
                (E::one() - r) + l * r
            } else {
                (E::one() - l) * (E::one() - r) + l * r
            };
        }
        let mut value = E::zero();
        for unit in &self.units {
            for &(base, weight) in &unit.bases {
                let mut rows = E::zero();
                prepared.accumulate(&mut rows, base, unit.first, unit.rows, &blocks)?;
                value += weight * rows;
            }
        }
        Ok(Some(self.scalar * coefficient * value))
    }
}
