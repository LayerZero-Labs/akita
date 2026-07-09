#[cfg(test)]
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_algebra::ring::eval_ring_at_pows_fast;
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
#[cfg(test)]
use akita_field::AkitaError;
use akita_field::{ExtField, FieldCore, MulBaseUnreduced};

const PARALLEL_BASE_RING_SEGMENT_MIN_LEN: usize = 8192;

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

macro_rules! dispatch_role_projections {
    ($d_projection:expr, $b_projection:expr, $a_projection:expr, |$d_identity:ident, $b_identity:ident, $a_identity:ident| $body:block) => {{
        match (
            $d_projection.is_identity(),
            $b_projection.is_identity(),
            $a_projection.is_identity(),
        ) {
            (true, true, true) => {
                const $d_identity: bool = true;
                const $b_identity: bool = true;
                const $a_identity: bool = true;
                $body
            }
            (true, true, false) => {
                const $d_identity: bool = true;
                const $b_identity: bool = true;
                const $a_identity: bool = false;
                $body
            }
            (true, false, true) => {
                const $d_identity: bool = true;
                const $b_identity: bool = false;
                const $a_identity: bool = true;
                $body
            }
            (false, true, true) => {
                const $d_identity: bool = false;
                const $b_identity: bool = true;
                const $a_identity: bool = true;
                $body
            }
            (true, false, false) => {
                const $d_identity: bool = true;
                const $b_identity: bool = false;
                const $a_identity: bool = false;
                $body
            }
            (false, true, false) => {
                const $d_identity: bool = false;
                const $b_identity: bool = true;
                const $a_identity: bool = false;
                $body
            }
            (false, false, true) => {
                const $d_identity: bool = false;
                const $b_identity: bool = false;
                const $a_identity: bool = true;
                $body
            }
            (false, false, false) => {
                const $d_identity: bool = false;
                const $b_identity: bool = false;
                const $a_identity: bool = false;
                $body
            }
        }
    }};
}

pub(super) use dispatch_role_projections;

macro_rules! dispatch_projected_ratio_a {
    ($a_ratio:expr, |$a_ratio_const:ident| $body:block) => {{
        match $a_ratio {
            1 => {
                const $a_ratio_const: usize = 1;
                Some($body)
            }
            2 => {
                const $a_ratio_const: usize = 2;
                Some($body)
            }
            4 => {
                const $a_ratio_const: usize = 4;
                Some($body)
            }
            8 => {
                const $a_ratio_const: usize = 8;
                Some($body)
            }
            16 => {
                const $a_ratio_const: usize = 16;
                Some($body)
            }
            _ => None,
        }
    }};
}

macro_rules! dispatch_projected_ratio_b {
    ($b_ratio:expr, $a_ratio:expr, |$b_ratio_const:ident, $a_ratio_const:ident| $body:block) => {{
        match $b_ratio {
            1 => {
                const $b_ratio_const: usize = 1;
                dispatch_projected_ratio_a!($a_ratio, |$a_ratio_const| $body)
            }
            2 => {
                const $b_ratio_const: usize = 2;
                dispatch_projected_ratio_a!($a_ratio, |$a_ratio_const| $body)
            }
            4 => {
                const $b_ratio_const: usize = 4;
                dispatch_projected_ratio_a!($a_ratio, |$a_ratio_const| $body)
            }
            8 => {
                const $b_ratio_const: usize = 8;
                dispatch_projected_ratio_a!($a_ratio, |$a_ratio_const| $body)
            }
            16 => {
                const $b_ratio_const: usize = 16;
                dispatch_projected_ratio_a!($a_ratio, |$a_ratio_const| $body)
            }
            _ => None,
        }
    }};
}

macro_rules! dispatch_projected_ratios {
    ($d_ratio:expr, $b_ratio:expr, $a_ratio:expr, |$d_ratio_const:ident, $b_ratio_const:ident, $a_ratio_const:ident| $body:block) => {{
        match $d_ratio {
            1 => {
                const $d_ratio_const: usize = 1;
                dispatch_projected_ratio_b!($b_ratio, $a_ratio, |$b_ratio_const, $a_ratio_const| {
                    $body
                })
            }
            2 => {
                const $d_ratio_const: usize = 2;
                dispatch_projected_ratio_b!($b_ratio, $a_ratio, |$b_ratio_const, $a_ratio_const| {
                    $body
                })
            }
            4 => {
                const $d_ratio_const: usize = 4;
                dispatch_projected_ratio_b!($b_ratio, $a_ratio, |$b_ratio_const, $a_ratio_const| {
                    $body
                })
            }
            8 => {
                const $d_ratio_const: usize = 8;
                dispatch_projected_ratio_b!($b_ratio, $a_ratio, |$b_ratio_const, $a_ratio_const| {
                    $body
                })
            }
            16 => {
                const $d_ratio_const: usize = 16;
                dispatch_projected_ratio_b!($b_ratio, $a_ratio, |$b_ratio_const, $a_ratio_const| {
                    $body
                })
            }
            _ => None,
        }
    }};
}

pub(super) use dispatch_projected_ratio_a;
pub(super) use dispatch_projected_ratio_b;
pub(super) use dispatch_projected_ratios;

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
pub(super) fn base_ring_segment_inner_sum_typed<
    F,
    E,
    const D: usize,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
    const D_IDENTITY: bool,
    const B_IDENTITY: bool,
    const A_IDENTITY: bool,
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
    if range.len() >= PARALLEL_BASE_RING_SEGMENT_MIN_LEN {
        return cfg_fold_reduce!(
            range,
            E::zero,
            |mut acc, base_idx| {
                let weight = base_ring_segment_weight_at::<
                    E,
                    HAS_D,
                    HAS_B,
                    HAS_A,
                    D_IDENTITY,
                    B_IDENTITY,
                    A_IDENTITY,
                >(
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
        );
    }

    // Keep the small-segment path inline. Routing each index through a shared
    // helper costs the ordinary verifier scan, while the wide path above keeps
    // the parallel shape needed by distributed profiles.
    let mut acc = E::zero();
    for base_idx in range {
        let mut weight = E::zero();
        if HAS_D {
            weight += projected_role_weight_at::<E, D_IDENTITY>(
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
            weight += projected_role_weight_at::<E, B_IDENTITY>(
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
            weight += projected_role_weight_at::<E, A_IDENTITY>(
                base_idx,
                segment.a_row,
                segment.a_start_abs,
                segment.a_row_weight,
                z_eq,
                a_projection,
                a_row_weights,
            );
        }
        if !weight.is_zero() {
            acc += eval_ring_at_pows_fast(&setup_flat[base_idx], base_pows) * weight;
        }
    }
    acc
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
    if range.len() >= PARALLEL_BASE_RING_SEGMENT_MIN_LEN {
        return cfg_fold_reduce!(
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
        );
    }

    let mut acc = E::zero();
    for setup_idx in range {
        let weight = segment.typed_weight_at::<HAS_D, HAS_B, HAS_A>(setup_idx, e_eq, t_eq, z_eq);
        if !weight.is_zero() {
            acc += eval_ring_at_pows_fast(&setup_flat[setup_idx], alpha_pows) * weight;
        }
    }
    acc
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn projected_base_ring_segment_inner_sum_typed<
    F,
    E,
    const D: usize,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
    const D_RATIO: usize,
    const B_RATIO: usize,
    const A_RATIO: usize,
>(
    range: std::ops::Range<usize>,
    setup_flat: &[CyclotomicRing<F, D>],
    base_pows: &[E],
    segment: &GroupSetupSegment<E>,
    e_eq: &[E],
    t_eq: &[E],
    z_eq: &[E],
    d_scales: &[E],
    b_scales: &[E],
    a_scales: &[E],
) -> E
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    debug_assert_eq!(d_scales.len(), D_RATIO);
    debug_assert_eq!(b_scales.len(), B_RATIO);
    debug_assert_eq!(a_scales.len(), A_RATIO);
    let period = D_RATIO.max(B_RATIO).max(A_RATIO);
    debug_assert_ne!(period, 0);
    let mut prefix_end = range.start;
    while prefix_end < range.end && !prefix_end.is_multiple_of(period) {
        prefix_end += 1;
    }
    let mut aligned_end = range.end - (range.end % period);
    if aligned_end < prefix_end {
        aligned_end = prefix_end;
    }
    let block_range = (prefix_end / period)..(aligned_end / period);

    let mut edge_acc = E::zero();
    for setup_idx in range.start..prefix_end {
        edge_acc += projected_base_ring_segment_index_sum::<
            F,
            E,
            D,
            HAS_D,
            HAS_B,
            HAS_A,
            D_RATIO,
            B_RATIO,
            A_RATIO,
        >(
            setup_idx, setup_flat, base_pows, segment, e_eq, t_eq, z_eq, d_scales, b_scales,
            a_scales,
        );
    }
    for setup_idx in aligned_end..range.end {
        edge_acc += projected_base_ring_segment_index_sum::<
            F,
            E,
            D,
            HAS_D,
            HAS_B,
            HAS_A,
            D_RATIO,
            B_RATIO,
            A_RATIO,
        >(
            setup_idx, setup_flat, base_pows, segment, e_eq, t_eq, z_eq, d_scales, b_scales,
            a_scales,
        );
    }
    if range.len() >= PARALLEL_BASE_RING_SEGMENT_MIN_LEN {
        return edge_acc
            + cfg_fold_reduce!(
                block_range,
                E::zero,
                |mut acc, block| {
                    acc += projected_base_ring_segment_block_sum::<
                        F,
                        E,
                        D,
                        HAS_D,
                        HAS_B,
                        HAS_A,
                        D_RATIO,
                        B_RATIO,
                        A_RATIO,
                    >(
                        block * period,
                        period,
                        setup_flat,
                        base_pows,
                        segment,
                        e_eq,
                        t_eq,
                        z_eq,
                        d_scales,
                        b_scales,
                        a_scales,
                    );
                    acc
                },
                |lhs, rhs| lhs + rhs
            );
    }

    let mut acc = edge_acc;
    for block in block_range {
        acc += projected_base_ring_segment_block_sum::<
            F,
            E,
            D,
            HAS_D,
            HAS_B,
            HAS_A,
            D_RATIO,
            B_RATIO,
            A_RATIO,
        >(
            block * period,
            period,
            setup_flat,
            base_pows,
            segment,
            e_eq,
            t_eq,
            z_eq,
            d_scales,
            b_scales,
            a_scales,
        );
    }
    acc
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn projected_base_ring_segment_index_sum<
    F,
    E,
    const D: usize,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
    const D_RATIO: usize,
    const B_RATIO: usize,
    const A_RATIO: usize,
>(
    setup_idx: usize,
    setup_flat: &[CyclotomicRing<F, D>],
    base_pows: &[E],
    segment: &GroupSetupSegment<E>,
    e_eq: &[E],
    t_eq: &[E],
    z_eq: &[E],
    d_scales: &[E],
    b_scales: &[E],
    a_scales: &[E],
) -> E
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let mut weight = E::zero();
    if HAS_D {
        let role_idx = setup_idx / D_RATIO;
        weight +=
            segment.d_weight * d_scales[setup_idx % D_RATIO] * e_eq[role_idx - segment.d_start_abs];
    }
    if HAS_B {
        let role_idx = setup_idx / B_RATIO;
        weight +=
            segment.b_weight * b_scales[setup_idx % B_RATIO] * t_eq[role_idx - segment.b_start_abs];
    }
    if HAS_A {
        let role_idx = setup_idx / A_RATIO;
        weight += segment.a_row_weight
            * a_scales[setup_idx % A_RATIO]
            * z_eq[role_idx - segment.a_start_abs];
    }
    if weight.is_zero() {
        E::zero()
    } else {
        eval_ring_at_pows_fast(&setup_flat[setup_idx], base_pows) * weight
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn projected_base_ring_segment_block_sum<
    F,
    E,
    const D: usize,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
    const D_RATIO: usize,
    const B_RATIO: usize,
    const A_RATIO: usize,
>(
    block_start: usize,
    period: usize,
    setup_flat: &[CyclotomicRing<F, D>],
    base_pows: &[E],
    segment: &GroupSetupSegment<E>,
    e_eq: &[E],
    t_eq: &[E],
    z_eq: &[E],
    d_scales: &[E],
    b_scales: &[E],
    a_scales: &[E],
) -> E
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let d_role_base = block_start / D_RATIO;
    let b_role_base = block_start / B_RATIO;
    let a_role_base = block_start / A_RATIO;
    let mut acc = E::zero();
    for lane in 0..period {
        let setup_idx = block_start + lane;
        let mut weight = E::zero();
        if HAS_D {
            let role_idx = d_role_base + lane / D_RATIO;
            weight +=
                segment.d_weight * d_scales[lane % D_RATIO] * e_eq[role_idx - segment.d_start_abs];
        }
        if HAS_B {
            let role_idx = b_role_base + lane / B_RATIO;
            weight +=
                segment.b_weight * b_scales[lane % B_RATIO] * t_eq[role_idx - segment.b_start_abs];
        }
        if HAS_A {
            let role_idx = a_role_base + lane / A_RATIO;
            weight += segment.a_row_weight
                * a_scales[lane % A_RATIO]
                * z_eq[role_idx - segment.a_start_abs];
        }
        if !weight.is_zero() {
            acc += eval_ring_at_pows_fast(&setup_flat[setup_idx], base_pows) * weight;
        }
    }
    acc
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn base_ring_segment_weight_at<
    E,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
    const D_IDENTITY: bool,
    const B_IDENTITY: bool,
    const A_IDENTITY: bool,
>(
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
        weight += projected_role_weight_at::<E, D_IDENTITY>(
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
        weight += projected_role_weight_at::<E, B_IDENTITY>(
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
        weight += projected_role_weight_at::<E, A_IDENTITY>(
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
fn projected_role_weight_at<E: FieldCore, const IDENTITY: bool>(
    base_idx: usize,
    row: usize,
    start_abs: usize,
    row_weight: E,
    eq_slice: &[E],
    projection: &RoleProjection<E>,
    weights: &ProjectedRoleWeights<E>,
) -> E {
    let role_idx = if IDENTITY {
        base_idx
    } else {
        base_idx >> projection.shift
    };
    let weight = if IDENTITY {
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
