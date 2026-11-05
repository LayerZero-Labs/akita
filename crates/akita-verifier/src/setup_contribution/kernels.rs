use super::{DirectScanWeights, PhysicalBWeightTerm};
use akita_error::AkitaError;
#[cfg(test)]
use jolt_field::{ExtField, MulBaseUnreduced};
use jolt_field::{Field, Unreduced, Zero};

pub(crate) struct GroupSetupSegment<E> {
    pub(super) lo: usize,
    pub(super) hi: usize,
    pub(super) has_d: bool,
    pub(super) d_start_abs: usize,
    pub(super) d_weight: E,
    pub(super) has_b: bool,
    pub(super) b_start_abs: usize,
    pub(super) b_terms: std::sync::Arc<[PhysicalBWeightTerm<E>]>,
    pub(super) has_a: bool,
    pub(super) a_start_abs: usize,
    pub(super) a_row_weight: E,
}

pub(super) enum RoleProjection<E> {
    Identity,
    Projected {
        scales: Vec<E>,
        shift: usize,
        mask: usize,
    },
}

pub(super) fn role_projection<E: Field>(
    alpha_pows: &[E],
    base_pows: &[E],
    expected_ratio: usize,
) -> Option<RoleProjection<E>> {
    let base_d = base_pows.len();
    if base_d == 0 || !alpha_pows.len().is_multiple_of(base_d) {
        return None;
    }
    let ratio = alpha_pows.len() / base_d;
    if ratio != expected_ratio {
        return None;
    }
    if ratio == 1 {
        return (alpha_pows == base_pows).then_some(RoleProjection::Identity);
    }
    let mut scales = Vec::with_capacity(ratio);
    for chunk in alpha_pows.chunks_exact(base_d) {
        let scale = *chunk.first()?;
        for (&power, &base_power) in chunk.iter().zip(base_pows) {
            if power != scale * base_power {
                return None;
            }
        }
        scales.push(scale);
    }
    Some(RoleProjection::Projected {
        scales,
        shift: ratio.trailing_zeros() as usize,
        mask: ratio.checked_sub(1)?,
    })
}

/// One setup-ring weight assembled from role products.
///
/// [`DelayedProductSum`] defers reduction for fields whose product sums are
/// exact; [`CanonicalProductSum`] reduces every product. Both finish to the
/// same field element.
pub(super) trait ProductSum<E: Unreduced> {
    fn zero() -> Self;
    fn add_product(&mut self, lhs: E, rhs: E);
    fn finish(self) -> E;
}

pub(super) struct DelayedProductSum<E: Unreduced>(E::Product);

impl<E: Unreduced> ProductSum<E> for DelayedProductSum<E> {
    #[inline(always)]
    fn zero() -> Self {
        Self(E::Product::zero())
    }

    #[inline(always)]
    fn add_product(&mut self, lhs: E, rhs: E) {
        self.0 += lhs.mul_unreduced(rhs);
    }

    #[inline(always)]
    fn finish(self) -> E {
        E::reduce_product(self.0)
    }
}

pub(super) struct CanonicalProductSum<E>(E);

impl<E: Unreduced> ProductSum<E> for CanonicalProductSum<E> {
    #[inline(always)]
    fn zero() -> Self {
        Self(E::zero())
    }

    #[inline(always)]
    fn add_product(&mut self, lhs: E, rhs: E) {
        self.0 += lhs * rhs;
    }

    #[inline(always)]
    fn finish(self) -> E {
        self.0
    }
}

/// Add the D/B/A weights `segment` assigns to the base rings of `overlap`.
///
/// `slots[0]` is base ring `overlap.start`. Each role adds
/// `row_weight · eq[role(b) − start] · scale(b)` to every base ring `b`, one
/// contiguous pass per role, so the hot loop carries no per-ring dispatch.
pub(super) fn add_segment_weights<E, A>(
    slots: &mut [A],
    overlap: std::ops::Range<usize>,
    segment: &GroupSetupSegment<E>,
    weights: &DirectScanWeights<E>,
    [a_projection, b_projection, d_projection]: [&RoleProjection<E>; 3],
) -> Result<(), AkitaError>
where
    E: Field + Unreduced,
    A: ProductSum<E>,
{
    if segment.has_d {
        add_role_weights(
            slots,
            overlap.clone(),
            segment.d_start_abs,
            segment.d_weight,
            &weights.e,
            d_projection,
        )?;
    }
    if segment.has_b {
        for term in segment.b_terms.iter() {
            let t = weights
                .t
                .get(term.logical_start..)
                .ok_or(AkitaError::InvalidProof)?;
            add_role_weights(
                slots,
                overlap.clone(),
                segment.b_start_abs,
                term.row_weight,
                t,
                b_projection,
            )?;
        }
    }
    if segment.has_a {
        add_role_weights(
            slots,
            overlap,
            segment.a_start_abs,
            segment.a_row_weight,
            &weights.z,
            a_projection,
        )?;
    }
    Ok(())
}

#[inline(always)]
fn add_role_weights<E, A>(
    slots: &mut [A],
    range: std::ops::Range<usize>,
    start_abs: usize,
    row_weight: E,
    eq: &[E],
    projection: &RoleProjection<E>,
) -> Result<(), AkitaError>
where
    E: Field + Unreduced,
    A: ProductSum<E>,
{
    if slots.len() != range.len() {
        return Err(AkitaError::InvalidProof);
    }
    match projection {
        RoleProjection::Identity => {
            let first = range
                .start
                .checked_sub(start_abs)
                .ok_or(AkitaError::InvalidProof)?;
            let end = first
                .checked_add(slots.len())
                .ok_or(AkitaError::InvalidProof)?;
            let eq = eq.get(first..end).ok_or(AkitaError::InvalidProof)?;
            for (slot, &eq) in slots.iter_mut().zip(eq) {
                slot.add_product(row_weight, eq);
            }
        }
        RoleProjection::Projected {
            scales,
            shift,
            mask,
        } => {
            // One role ring covers `scales.len()` consecutive base rings: its
            // weighted equality value is formed once, then scaled per lane.
            let mut slots = slots.iter_mut();
            let mut base = range.start;
            while base < range.end {
                let role = base >> shift;
                let role_end = role
                    .checked_add(1)
                    .and_then(|next| next.checked_mul(scales.len()))
                    .ok_or(AkitaError::InvalidProof)?
                    .min(range.end);
                let eq_index = role
                    .checked_sub(start_abs)
                    .ok_or(AkitaError::InvalidProof)?;
                let value = row_weight * *eq.get(eq_index).ok_or(AkitaError::InvalidProof)?;
                for base_idx in base..role_end {
                    let scale = *scales
                        .get(base_idx & mask)
                        .ok_or(AkitaError::InvalidProof)?;
                    slots
                        .next()
                        .ok_or(AkitaError::InvalidProof)?
                        .add_product(value, scale);
                }
                base = role_end;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn evaluate_weighted_setup_row<Base, E>(
    row: &[Base],
    col_offset: usize,
    col_weights: &[E],
    row_weight: E,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    Base: Field,
    E: ExtField<Base> + MulBaseUnreduced<Base>,
{
    use super::checked_slice;

    let ring_d = alpha_pows.len();
    let mut acc = E::zero();
    for (col, &col_weight) in col_weights.iter().enumerate() {
        if col_weight.is_zero() {
            continue;
        }
        let setup_col = col_offset
            .checked_add(col)
            .ok_or_else(|| AkitaError::InvalidSetup("weighted setup column overflow".into()))?;
        let coeff_start = setup_col.checked_mul(ring_d).ok_or_else(|| {
            AkitaError::InvalidSetup("weighted setup coeff start overflow".into())
        })?;
        let coeffs = checked_slice(row, coeff_start, ring_d, "weighted setup coeffs")?;
        acc += row_weight * col_weight * MulBaseUnreduced::dot_base(alpha_pows, coeffs);
    }
    Ok(acc)
}
