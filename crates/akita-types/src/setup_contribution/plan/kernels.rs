#[cfg(test)]
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_algebra::ring::eval_ring_at_pows_fast;
use akita_algebra::CyclotomicRing;
#[cfg(test)]
use akita_error::AkitaError;
use jolt_field::parallel::*;
use jolt_field::{ExtField, FieldCore, MulBaseUnreduced};

#[derive(Clone)]
pub(crate) struct GroupSetupSegment<E> {
    pub(super) lo: usize,
    pub(super) hi: usize,
    pub(super) has_d: bool,
    pub(super) d_row: usize,
    pub(super) d_start_abs: usize,
    pub(super) d_weight: E,
    pub(super) has_b: bool,
    pub(super) b_row: usize,
    pub(super) b_start_abs: usize,
    pub(super) b_weight: E,
    pub(super) has_a: bool,
    pub(super) a_row: usize,
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

impl<E: FieldCore> GroupSetupSegment<E> {
    #[inline(always)]
    pub(super) fn weight_at(&self, setup_idx: usize, e_eq: &[E], t_eq: &[E], z_eq: &[E]) -> E {
        let mut weight = E::zero();
        if self.has_d {
            weight += self.d_weight_at(setup_idx, e_eq);
        }
        if self.has_b {
            weight += self.b_weight_at(setup_idx, t_eq);
        }
        if self.has_a {
            weight += self.a_row_weight_at(setup_idx, z_eq);
        }
        weight
    }

    #[inline(always)]
    pub(super) fn typed_weight_at<const HAS_D: bool, const HAS_B: bool, const HAS_A: bool>(
        &self,
        setup_idx: usize,
        e_eq: &[E],
        t_eq: &[E],
        z_eq: &[E],
    ) -> E {
        let mut weight = E::zero();
        if HAS_D {
            weight += self.d_weight_at(setup_idx, e_eq);
        }
        if HAS_B {
            weight += self.b_weight_at(setup_idx, t_eq);
        }
        if HAS_A {
            weight += self.a_row_weight_at(setup_idx, z_eq);
        }
        weight
    }

    #[inline(always)]
    pub(super) fn d_weight_at(&self, setup_idx: usize, e_eq: &[E]) -> E {
        self.d_weight * e_eq[setup_idx - self.d_start_abs]
    }

    #[inline(always)]
    pub(super) fn b_weight_at(&self, setup_idx: usize, t_eq: &[E]) -> E {
        self.b_weight * t_eq[setup_idx - self.b_start_abs]
    }

    #[inline(always)]
    pub(super) fn a_row_weight_at(&self, setup_idx: usize, z_eq: &[E]) -> E {
        self.a_row_weight * z_eq[setup_idx - self.a_start_abs]
    }
}

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
    pub(super) fn ratio(&self) -> usize {
        self.scales.len()
    }

    #[inline(always)]
    pub(super) fn is_identity(&self) -> bool {
        self.scales.len() == 1
    }
}

pub(super) fn role_projection<E: FieldCore>(
    alpha_pows: &[E],
    base_pows: &[E],
) -> Option<RoleProjection<E>> {
    let base_d = base_pows.len();
    if base_d == 0 || !alpha_pows.len().is_multiple_of(base_d) {
        return None;
    }
    let ratio = alpha_pows.len() / base_d;
    if ratio == 0 || !ratio.is_power_of_two() {
        return None;
    }
    if ratio == 1 {
        return (alpha_pows == base_pows).then(RoleProjection::identity);
    }
    let mut scales = Vec::with_capacity(ratio);
    for chunk in 0..ratio {
        let offset = chunk * base_d;
        let scale = alpha_pows[offset];
        for idx in 0..base_d {
            if alpha_pows[offset + idx] != scale * base_pows[idx] {
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

pub(super) struct ProjectedRoleWeights<E> {
    scaled: Vec<E>,
    ratio: usize,
}

impl<E: FieldCore> ProjectedRoleWeights<E> {
    pub(super) fn new(row_weights: &[E], projection: &RoleProjection<E>) -> Self {
        if projection.is_identity() {
            return Self {
                scaled: Vec::new(),
                ratio: 1,
            };
        }

        let mut scaled = Vec::with_capacity(row_weights.len() * projection.ratio());
        for &row_weight in row_weights {
            scaled.extend(projection.scales.iter().map(|&scale| row_weight * scale));
        }
        Self {
            scaled,
            ratio: projection.ratio(),
        }
    }

    #[inline(always)]
    pub(super) fn get(&self, row: usize, base_idx: usize, projection: &RoleProjection<E>) -> E {
        self.scaled[row * self.ratio + (base_idx & projection.mask)]
    }
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
    d_weights: &ProjectedRoleWeights<E>,
    b_weights: &ProjectedRoleWeights<E>,
    a_row_weights: &ProjectedRoleWeights<E>,
) -> E
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, base_idx| {
            let weight = base_ring_segment_weight_at::<E, HAS_D, HAS_B, HAS_A>(
                base_idx,
                segment,
                e_eq,
                t_eq,
                z_eq,
                d_projection,
                b_projection,
                a_projection,
                d_weights,
                b_weights,
                a_row_weights,
            );
            if !weight.is_zero() {
                acc += eval_ring_at_pows_fast(&setup_flat[base_idx], base_pows) * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn identity_base_ring_segment_inner_sum_typed<
    F,
    E,
    const D: usize,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    setup_flat: &[CyclotomicRing<F, D>],
    alpha_pows: &[E],
    segment: &GroupSetupSegment<E>,
    e_eq: &[E],
    t_eq: &[E],
    z_eq: &[E],
) -> E
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, setup_idx| {
            let weight =
                segment.typed_weight_at::<HAS_D, HAS_B, HAS_A>(setup_idx, e_eq, t_eq, z_eq);
            if !weight.is_zero() {
                acc += eval_ring_at_pows_fast(&setup_flat[setup_idx], alpha_pows) * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
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
    d_weights: &ProjectedRoleWeights<E>,
    b_weights: &ProjectedRoleWeights<E>,
    a_row_weights: &ProjectedRoleWeights<E>,
) -> E
where
    E: FieldCore,
{
    let mut weight = E::zero();
    if HAS_D {
        weight += projected_role_weight_at(
            base_idx,
            segment.d_row,
            segment.d_start_abs,
            segment.d_weight,
            e_eq,
            d_projection,
            d_weights,
        );
    }
    if HAS_B {
        weight += projected_role_weight_at(
            base_idx,
            segment.b_row,
            segment.b_start_abs,
            segment.b_weight,
            t_eq,
            b_projection,
            b_weights,
        );
    }
    if HAS_A {
        weight += projected_role_weight_at(
            base_idx,
            segment.a_row,
            segment.a_start_abs,
            segment.a_row_weight,
            z_eq,
            a_projection,
            a_row_weights,
        );
    }
    weight
}

#[inline(always)]
fn projected_role_weight_at<E: FieldCore>(
    base_idx: usize,
    row: usize,
    start_abs: usize,
    row_weight: E,
    eq_slice: &[E],
    projection: &RoleProjection<E>,
    weights: &ProjectedRoleWeights<E>,
) -> E {
    let identity = projection.is_identity();
    let role_idx = if identity {
        base_idx
    } else {
        base_idx >> projection.shift
    };
    let weight = if identity {
        row_weight
    } else {
        weights.get(row, base_idx, projection)
    };
    weight * eq_slice[role_idx - start_abs]
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
