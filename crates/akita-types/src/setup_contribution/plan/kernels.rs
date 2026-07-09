#[cfg(test)]
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_algebra::ring::eval_ring_at_pows_fast;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
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
    pub(super) a_weight: E,
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
            weight += self.a_weight_at(setup_idx, z_eq);
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
            weight += self.a_weight_at(setup_idx, z_eq);
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
    pub(super) fn a_weight_at(&self, setup_idx: usize, z_eq: &[E]) -> E {
        self.a_weight * z_eq[setup_idx - self.a_start_abs]
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_typed_packed_scan_access<E>(
    d_rows: usize,
    d_physical_cols: usize,
    has_d_view: bool,
    d_len: usize,
    n_b: usize,
    t_cols: usize,
    b_len: usize,
    n_a: usize,
    z_cols: usize,
    a_len: usize,
    segments: &[GroupSetupSegment<E>],
) -> Result<(), AkitaError>
where
    E: FieldCore,
{
    for segment in segments {
        if segment.has_d && !has_d_view {
            return Err(AkitaError::InvalidSetup(
                "setup packed D scan missing D view".into(),
            ));
        }
    }
    let d_required = d_rows
        .checked_mul(d_physical_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
    if d_required > d_len {
        return Err(AkitaError::InvalidSetup(
            "shared D matrix is too small for selected verifier layout".into(),
        ));
    }
    let b_required = n_b
        .checked_mul(t_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?;
    if b_required > b_len {
        return Err(AkitaError::InvalidSetup(
            "shared B matrix is too small for selected verifier layout".into(),
        ));
    }
    let a_required = n_a
        .checked_mul(z_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
    if a_required > a_len {
        return Err(AkitaError::InvalidSetup(
            "shared A matrix is too small for selected verifier layout".into(),
        ));
    }
    Ok(())
}

pub(super) struct AlphaChunkScales<E> {
    pub(super) scales: Vec<E>,
    pub(super) shift: usize,
    pub(super) mask: usize,
}

pub(super) fn alpha_chunk_scales<E: FieldCore>(
    alpha_pows: &[E],
    base_pows: &[E],
) -> Option<AlphaChunkScales<E>> {
    let base_d = base_pows.len();
    if base_d == 0 || !alpha_pows.len().is_multiple_of(base_d) {
        return None;
    }
    let ratio = alpha_pows.len() / base_d;
    if ratio == 0 || !ratio.is_power_of_two() {
        return None;
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
    Some(AlphaChunkScales {
        scales,
        shift: ratio.trailing_zeros() as usize,
        mask: ratio - 1,
    })
}

pub(super) fn scaled_row_weights<E: FieldCore>(row_weights: &[E], scales: &[E]) -> Vec<E> {
    let mut scaled = Vec::with_capacity(row_weights.len() * scales.len());
    for &row_weight in row_weights {
        scaled.extend(scales.iter().map(|&scale| row_weight * scale));
    }
    scaled
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn group_bar_omega_segment_eval<
    E,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    eq_setup_idx: &[E],
    segment: &GroupSetupSegment<E>,
    e_eq: &[E],
    t_eq: &[E],
    z_eq: &[E],
) -> E
where
    E: FieldCore,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, setup_idx| {
            let weight =
                segment.typed_weight_at::<HAS_D, HAS_B, HAS_A>(setup_idx, e_eq, t_eq, z_eq);
            if !weight.is_zero() {
                acc += eq_setup_idx[setup_idx] * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_uniform_group_slice_inner_sum_typed<
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
pub(super) fn packed_group_slice_inner_sum_typed<
    F,
    E,
    const D_A: usize,
    const D_B: usize,
    const D_D: usize,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    d_flat: Option<&[CyclotomicRing<F, D_D>]>,
    d_physical_cols: usize,
    b_flat: &[CyclotomicRing<F, D_B>],
    t_cols: usize,
    a_flat: &[CyclotomicRing<F, D_A>],
    z_cols: usize,
    alpha_pows_a: &[E],
    alpha_pows_b: &[E],
    alpha_pows_d: &[E],
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
            if HAS_D {
                let eq_w = segment.d_weight_at(setup_idx, e_eq);
                if !eq_w.is_zero() {
                    if let Some(d_flat) = d_flat {
                        let d_row = setup_idx / d_physical_cols;
                        let d_col = setup_idx % d_physical_cols;
                        let setup_idx = d_row * d_physical_cols + d_col;
                        acc += eval_ring_at_pows_fast(&d_flat[setup_idx], alpha_pows_d) * eq_w;
                    }
                }
            }
            if HAS_B {
                let eq_w = segment.b_weight_at(setup_idx, t_eq);
                if !eq_w.is_zero() {
                    let b_row = setup_idx / t_cols;
                    let b_col = setup_idx % t_cols;
                    let role_idx = b_row * t_cols + b_col;
                    acc += eval_ring_at_pows_fast(&b_flat[role_idx], alpha_pows_b) * eq_w;
                }
            }
            if HAS_A {
                let eq_w = segment.a_weight_at(setup_idx, z_eq);
                if !eq_w.is_zero() {
                    let a_row = setup_idx / z_cols;
                    let a_col = setup_idx % z_cols;
                    let role_idx = a_row * z_cols + a_col;
                    acc += eval_ring_at_pows_fast(&a_flat[role_idx], alpha_pows_a) * eq_w;
                }
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
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
