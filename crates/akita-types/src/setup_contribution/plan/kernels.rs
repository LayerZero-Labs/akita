use super::{PhysicalBWeightTerm, ReducedRoleCoefficientState};
#[cfg(test)]
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_algebra::ring::eval_ring_at_pows_fast;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use jolt_field::{ExtField, Field, MulBaseUnreduced};

#[derive(Clone)]
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

pub(super) enum RoleProjection<E> {
    Identity,
    Projected {
        scales: Vec<E>,
        shift: usize,
        mask: usize,
    },
}

pub(super) struct ReducedScanGroupWeights<'a, E: Field> {
    pub(super) e: &'a [E],
    pub(super) t: &'a [E],
    pub(super) z: &'a [E],
    pub(super) role_ratios: [usize; 3],
    pub(super) roles: &'a [ReducedRoleCoefficientState<E>; 3],
}

impl<E: Field> RoleProjection<E> {
    #[inline(always)]
    pub(super) fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }
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
        mask: ratio - 1,
    })
}

/// Add one setup ring's reduced A/B/D coefficient weights.
///
/// The three role contributions are combined coefficient-by-coefficient before
/// the setup ring is evaluated, so even overlapping mixed-dimension views read
/// each public setup coefficient exactly once.
pub(super) fn add_reduced_base_ring_weights<E: Field, const D: usize>(
    base_idx: usize,
    segment: &GroupSetupSegment<E>,
    group: &ReducedScanGroupWeights<'_, E>,
    output: &mut [E; D],
) -> Result<(), AkitaError> {
    let [a_ratio, b_ratio, d_ratio] = group.role_ratios;
    if segment.has_d {
        let role_idx = projected_role_index(base_idx, d_ratio)?;
        let eq_idx = role_idx
            .checked_sub(segment.d_start_abs)
            .ok_or(AkitaError::InvalidProof)?;
        let weight = segment.d_weight * *group.e.get(eq_idx).ok_or(AkitaError::InvalidProof)?;
        add_functional_chunk(
            base_idx,
            d_ratio,
            &group.roles[2].functional,
            weight,
            output,
        )?;
    }
    if segment.has_b {
        let role_idx = projected_role_index(base_idx, b_ratio)?;
        let local = role_idx
            .checked_sub(segment.b_start_abs)
            .ok_or(AkitaError::InvalidProof)?;
        let mut weight = E::zero();
        for term in segment.b_terms.iter() {
            let logical = term
                .logical_start
                .checked_add(local)
                .and_then(|index| group.t.get(index))
                .copied()
                .ok_or(AkitaError::InvalidProof)?;
            weight += term.row_weight * logical;
        }
        add_functional_chunk(
            base_idx,
            b_ratio,
            &group.roles[1].functional,
            weight,
            output,
        )?;
    }
    if segment.has_a {
        let role_idx = projected_role_index(base_idx, a_ratio)?;
        let eq_idx = role_idx
            .checked_sub(segment.a_start_abs)
            .ok_or(AkitaError::InvalidProof)?;
        let weight = segment.a_row_weight * *group.z.get(eq_idx).ok_or(AkitaError::InvalidProof)?;
        add_functional_chunk(
            base_idx,
            a_ratio,
            &group.roles[0].functional,
            weight,
            output,
        )?;
    }
    Ok(())
}

fn projected_role_index(base_idx: usize, ratio: usize) -> Result<usize, AkitaError> {
    if !ratio.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup role projection ratio must be a power of two".into(),
        ));
    }
    Ok(base_idx / ratio)
}

fn add_functional_chunk<E: Field, const D: usize>(
    base_idx: usize,
    ratio: usize,
    functional: &[E],
    scalar: E,
    output: &mut [E; D],
) -> Result<(), AkitaError> {
    if scalar.is_zero() {
        return Ok(());
    }
    let chunk = base_idx % ratio;
    let start = chunk
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("coefficient functional offset overflow".into()))?;
    let end = start
        .checked_add(D)
        .ok_or_else(|| AkitaError::InvalidSetup("coefficient functional extent overflow".into()))?;
    let weights = functional.get(start..end).ok_or(AkitaError::InvalidProof)?;
    for (target, &weight) in output.iter_mut().zip(weights) {
        *target += scalar * weight;
    }
    Ok(())
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
    F: Field,
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
    E: Field,
{
    let len = range
        .end
        .checked_sub(range.start)
        .ok_or(AkitaError::InvalidProof)?;
    let identity =
        d_projection.is_identity() && b_projection.is_identity() && a_projection.is_identity();
    if identity {
        let d_eq = checked_role_eq_slice::<E, HAS_D>(e_eq, range.start, len, segment.d_start_abs)?;
        let a_eq = checked_role_eq_slice::<E, HAS_A>(z_eq, range.start, len, segment.a_start_abs)?;
        let mut d_eq = d_eq.iter();
        let mut a_eq = a_eq.iter();
        for offset in 0..len {
            let mut weight = E::zero();
            if HAS_D {
                weight += segment.d_weight * *d_eq.next().ok_or(AkitaError::InvalidProof)?;
            }
            if HAS_B {
                weight +=
                    projected_b_role_weight_at(range.start + offset, segment, t_eq, b_projection)?;
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
    E: Field,
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
        weight += projected_b_role_weight_at(base_idx, segment, t_eq, b_projection)?;
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
fn projected_b_role_weight_at<E: Field>(
    base_idx: usize,
    segment: &GroupSetupSegment<E>,
    weights: &[E],
    projection: &RoleProjection<E>,
) -> Result<E, AkitaError> {
    let (role_idx, scale) = projected_role_index_and_scale(base_idx, projection)?;
    let local = role_idx
        .checked_sub(segment.b_start_abs)
        .ok_or(AkitaError::InvalidProof)?;
    let mut weight = E::zero();
    for term in segment.b_terms.iter() {
        let logical = term
            .logical_start
            .checked_add(local)
            .and_then(|index| weights.get(index))
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        weight += term.row_weight * logical;
    }
    Ok(scale.map_or(weight, |scale| weight * scale))
}

#[inline(always)]
fn projected_role_weight_at<E: Field>(
    base_idx: usize,
    start_abs: usize,
    row_weight: E,
    eq_slice: &[E],
    projection: &RoleProjection<E>,
) -> Result<E, AkitaError> {
    let (role_idx, scale) = projected_role_index_and_scale(base_idx, projection)?;
    let eq_idx = role_idx
        .checked_sub(start_abs)
        .ok_or(AkitaError::InvalidProof)?;
    let weight = row_weight * *eq_slice.get(eq_idx).ok_or(AkitaError::InvalidProof)?;
    Ok(scale.map_or(weight, |scale| weight * scale))
}

#[inline(always)]
fn projected_role_index_and_scale<E: Field>(
    base_idx: usize,
    projection: &RoleProjection<E>,
) -> Result<(usize, Option<E>), AkitaError> {
    Ok(match projection {
        RoleProjection::Identity => (base_idx, None),
        RoleProjection::Projected {
            scales,
            shift,
            mask,
        } => (
            base_idx >> shift,
            Some(
                *scales
                    .get(base_idx & mask)
                    .ok_or(AkitaError::InvalidProof)?,
            ),
        ),
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
    Base: Field,
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
