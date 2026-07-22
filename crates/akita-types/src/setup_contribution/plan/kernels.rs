#[cfg(test)]
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_field::FieldCore;
#[cfg(test)]
use akita_field::{AkitaError, ExtField, MulBaseUnreduced};

pub(super) struct RoleProjection<E> {
    pub(super) scales: Vec<E>,
    pub(super) shift: usize,
    pub(super) mask: usize,
}

impl<E: FieldCore> RoleProjection<E> {
    #[inline(always)]
    pub(super) fn identity() -> Self {
        Self {
            scales: vec![E::one()],
            shift: 0,
            mask: 0,
        }
    }

    #[inline(always)]
    pub(super) fn is_identity(&self) -> bool {
        self.scales.len() == 1
    }
}

pub(super) fn role_projection<E: FieldCore>(
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
        return (alpha_pows == base_pows).then(RoleProjection::identity);
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
    Some(RoleProjection {
        scales,
        shift: ratio.trailing_zeros() as usize,
        mask: ratio - 1,
    })
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
    Base: FieldCore,
    E: ExtField<Base> + MulBaseUnreduced<Base>,
{
    use super::super::checked_slice;

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
        acc += row_weight * col_weight * eval_flat_ring_at_pows_fast::<Base, E>(coeffs, alpha_pows);
    }
    Ok(acc)
}
