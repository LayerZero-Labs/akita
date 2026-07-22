#[cfg(test)]
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_algebra::ring::eval_ring_at_pows_fast;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, ExtField, FieldCore, MulBaseUnreduced};

#[derive(Clone)]
pub(crate) struct GroupSetupSegment<E> {
    pub(super) lo: usize,
    pub(super) hi: usize,
    pub(super) has_d: bool,
    pub(super) d_start_abs: usize,
    pub(super) d_weight: E,
    pub(super) has_b: bool,
    pub(super) b_start_abs: usize,
    pub(super) b_weight: E,
    pub(super) has_a: bool,
    pub(super) a_start_abs: usize,
    pub(super) a_row_weight: E,
}

macro_rules! dispatch_segment_roles {
    ($segment:expr, $none:expr, |$has_d:ident, $has_b:ident, $has_a:ident| $body:block) => {{
        match ($segment.has_d, $segment.has_b, $segment.has_a) {
            (true, true, true) => {
                const $has_d: bool = true;
                const $has_b: bool = true;
                const $has_a: bool = true;
                $body
            }
            (true, true, false) => {
                const $has_d: bool = true;
                const $has_b: bool = true;
                const $has_a: bool = false;
                $body
            }
            (true, false, true) => {
                const $has_d: bool = true;
                const $has_b: bool = false;
                const $has_a: bool = true;
                $body
            }
            (false, true, true) => {
                const $has_d: bool = false;
                const $has_b: bool = true;
                const $has_a: bool = true;
                $body
            }
            (true, false, false) => {
                const $has_d: bool = true;
                const $has_b: bool = false;
                const $has_a: bool = false;
                $body
            }
            (false, true, false) => {
                const $has_d: bool = false;
                const $has_b: bool = true;
                const $has_a: bool = false;
                $body
            }
            (false, false, true) => {
                const $has_d: bool = false;
                const $has_b: bool = false;
                const $has_a: bool = true;
                $body
            }
            (false, false, false) => $none,
        }
    }};
}

pub(super) use dispatch_segment_roles;

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

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn base_ring_segment_inner_sum_typed<
    F,
    E,
    const D: usize,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    setup_flat: &[CyclotomicRing<F, D>],
    base_pows: &[E],
    segment: &GroupSetupSegment<E>,
    e_eq: &[E],
    t_eq: &[E],
    z_eq: &[E],
    d_projection: &RoleProjection<E>,
    b_projection: &RoleProjection<E>,
    a_projection: &RoleProjection<E>,
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let setup = setup_flat
        .get(range.clone())
        .ok_or(AkitaError::InvalidProof)?;
    let mut acc = E::zero();
    for_each_base_ring_segment_weight_typed::<E, HAS_D, HAS_B, HAS_A>(
        range,
        segment,
        e_eq,
        t_eq,
        z_eq,
        d_projection,
        b_projection,
        a_projection,
        |offset, weight| {
            if !weight.is_zero() {
                let ring = setup.get(offset).ok_or(AkitaError::InvalidProof)?;
                acc += eval_ring_at_pows_fast(ring, base_pows) * weight;
            }
            Ok(())
        },
    )?;
    Ok(acc)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn for_each_base_ring_segment_weight_typed<
    E,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    segment: &GroupSetupSegment<E>,
    e_eq: &[E],
    t_eq: &[E],
    z_eq: &[E],
    d_projection: &RoleProjection<E>,
    b_projection: &RoleProjection<E>,
    a_projection: &RoleProjection<E>,
    mut visit: impl FnMut(usize, E) -> Result<(), AkitaError>,
) -> Result<(), AkitaError>
where
    E: FieldCore,
{
    let len = range
        .end
        .checked_sub(range.start)
        .ok_or(AkitaError::InvalidProof)?;
    let identity =
        d_projection.is_identity() && b_projection.is_identity() && a_projection.is_identity();
    if identity {
        let d_eq = checked_role_eq_slice::<E, HAS_D>(e_eq, range.start, len, segment.d_start_abs)?;
        let b_eq = checked_role_eq_slice::<E, HAS_B>(t_eq, range.start, len, segment.b_start_abs)?;
        let a_eq = checked_role_eq_slice::<E, HAS_A>(z_eq, range.start, len, segment.a_start_abs)?;
        let mut d_eq = d_eq.iter();
        let mut b_eq = b_eq.iter();
        let mut a_eq = a_eq.iter();
        for offset in 0..len {
            let mut weight = E::zero();
            if HAS_D {
                weight += segment.d_weight * *d_eq.next().ok_or(AkitaError::InvalidProof)?;
            }
            if HAS_B {
                weight += segment.b_weight * *b_eq.next().ok_or(AkitaError::InvalidProof)?;
            }
            if HAS_A {
                weight += segment.a_row_weight * *a_eq.next().ok_or(AkitaError::InvalidProof)?;
            }
            visit(offset, weight)?;
        }
        return Ok(());
    }

    for offset in 0..len {
        let base_idx = range
            .start
            .checked_add(offset)
            .ok_or(AkitaError::InvalidProof)?;
        let weight = base_ring_segment_weight_at::<E, HAS_D, HAS_B, HAS_A>(
            base_idx,
            segment,
            e_eq,
            t_eq,
            z_eq,
            d_projection,
            b_projection,
            a_projection,
        )?;
        visit(offset, weight)?;
    }
    Ok(())
}

fn checked_role_eq_slice<E, const ACTIVE: bool>(
    eq: &[E],
    base_start: usize,
    len: usize,
    role_start: usize,
) -> Result<&[E], AkitaError> {
    if !ACTIVE {
        return Ok(&[]);
    }
    let start = base_start
        .checked_sub(role_start)
        .ok_or(AkitaError::InvalidProof)?;
    let end = start.checked_add(len).ok_or(AkitaError::InvalidProof)?;
    eq.get(start..end).ok_or(AkitaError::InvalidProof)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn base_ring_segment_weight_at<E, const HAS_D: bool, const HAS_B: bool, const HAS_A: bool>(
    base_idx: usize,
    segment: &GroupSetupSegment<E>,
    e_eq: &[E],
    t_eq: &[E],
    z_eq: &[E],
    d_projection: &RoleProjection<E>,
    b_projection: &RoleProjection<E>,
    a_projection: &RoleProjection<E>,
) -> Result<E, AkitaError>
where
    E: FieldCore,
{
    let mut weight = E::zero();
    if HAS_D {
        weight += projected_role_weight_at(
            base_idx,
            segment.d_start_abs,
            segment.d_weight,
            e_eq,
            d_projection,
        )?;
    }
    if HAS_B {
        weight += projected_role_weight_at(
            base_idx,
            segment.b_start_abs,
            segment.b_weight,
            t_eq,
            b_projection,
        )?;
    }
    if HAS_A {
        weight += projected_role_weight_at(
            base_idx,
            segment.a_start_abs,
            segment.a_row_weight,
            z_eq,
            a_projection,
        )?;
    }
    Ok(weight)
}

#[inline(always)]
fn projected_role_weight_at<E: FieldCore>(
    base_idx: usize,
    start_abs: usize,
    row_weight: E,
    eq_slice: &[E],
    projection: &RoleProjection<E>,
) -> Result<E, AkitaError> {
    let identity = projection.is_identity();
    let role_idx = if identity {
        base_idx
    } else {
        base_idx >> projection.shift
    };
    let scale = if identity {
        E::one()
    } else {
        *projection
            .scales
            .get(base_idx & projection.mask)
            .ok_or(AkitaError::InvalidProof)?
    };
    let eq_idx = role_idx
        .checked_sub(start_abs)
        .ok_or(AkitaError::InvalidProof)?;
    Ok(row_weight * scale * *eq_slice.get(eq_idx).ok_or(AkitaError::InvalidProof)?)
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
