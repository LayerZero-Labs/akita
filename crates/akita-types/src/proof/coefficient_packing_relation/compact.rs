use akita_algebra::offset_eq::{
    eq_eval_at_index, eval_affine_digit_intervals, eval_boolean_pair_tensor_families,
    EqPairTensorAxis, EqPairTensorFamily, MAX_COMPACT_STRIDE_TERMS,
};
use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;
use jolt_field::Field;
use std::ops::Range;
use std::sync::Arc;

use crate::{
    PreparedSubringCoefficientPackingPoint, SubringCoefficientPackingGeometry, WitnessLayout,
};

/// One verifier group's compact coefficient-packing semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingVerifierGroupSemantics<E: Field> {
    pub(super) group_index: usize,
    pub(super) geometry: SubringCoefficientPackingGeometry,
    pub(super) group_claim_range: Range<usize>,
    pub(super) scalar_claim_weight: E,
    pub(super) compact_factors: CoefficientPackingCompactFactors<E>,
}

/// Compact tensor factors used by the verifier at the Stage 2 final point.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingCompactFactors<E: Field> {
    pub(super) basis: crate::BasisMode,
    pub(super) physical_field_len: usize,
    pub(super) direct_opening_point: Arc<[E]>,
    pub(super) packing_z_point: Arc<[E]>,
    pub(super) affine_relation_families: Vec<CoefficientPackingAffineRelationFamily<E>>,
    pub(super) quotient_families: Vec<EqPairTensorFamily<E>>,
    pub(super) direct_opening_families: Vec<EqPairTensorFamily<E>>,
    pub(super) packing_z_families: Vec<EqPairTensorFamily<E>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CoefficientPackingAffineRelationFamily<E: Field> {
    pub(super) scalar: E,
    pub(super) coefficient_weights: Arc<[E]>,
    pub(super) coefficient_len: usize,
    pub(super) base_offset: usize,
    pub(super) outer_len: usize,
    pub(super) outer_stride: usize,
    pub(super) digit_stride: usize,
    pub(super) digit_weights: Arc<[E]>,
    pub(super) outer_weights: Arc<[E]>,
}

/// Checked compact packing semantics for the Stage 2 verifier.
///
/// Unlike the prover batch, this carrier never builds the expanded event and
/// segment representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingVerifierBatchSemantics<E: Field> {
    pub(super) groups: Vec<CoefficientPackingVerifierGroupSemantics<E>>,
}

impl<E: Field> CoefficientPackingVerifierBatchSemantics<E> {
    #[must_use]
    pub fn groups(&self) -> &[CoefficientPackingVerifierGroupSemantics<E>] {
        &self.groups
    }
}

impl<E: Field> CoefficientPackingVerifierGroupSemantics<E> {
    #[must_use]
    pub const fn group_index(&self) -> usize {
        self.group_index
    }

    #[must_use]
    pub const fn geometry(&self) -> SubringCoefficientPackingGeometry {
        self.geometry
    }

    #[must_use]
    pub fn group_claim_range(&self) -> Range<usize> {
        self.group_claim_range.clone()
    }

    #[must_use]
    pub const fn scalar_claim_weight(&self) -> E {
        self.scalar_claim_weight
    }

    #[must_use]
    pub const fn compact_factors(&self) -> &CoefficientPackingCompactFactors<E> {
        &self.compact_factors
    }
}

impl<E: Field> CoefficientPackingAffineRelationFamily<E> {
    fn shares_contraction_geometry(&self, other: &Self) -> bool {
        self.coefficient_len == other.coefficient_len
            && self.outer_len == other.outer_len
            && self.outer_stride == other.outer_stride
            && self.digit_stride == other.digit_stride
            && Arc::ptr_eq(&self.digit_weights, &other.digit_weights)
            && Arc::ptr_eq(&self.outer_weights, &other.outer_weights)
    }

    fn coefficient_evaluation_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        let coefficient_len = self.coefficient_len;
        if coefficient_len == 0 || !coefficient_len.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "packing affine coefficient axis is malformed".into(),
            ));
        }
        let coefficient_bits = coefficient_len.trailing_zeros() as usize;
        let (coefficient_point, _) = point
            .split_at_checked(coefficient_bits)
            .ok_or(AkitaError::InvalidProof)?;
        Ok(self
            .coefficient_weights
            .get(..coefficient_len)
            .ok_or(AkitaError::InvalidProof)?
            .iter()
            .enumerate()
            .fold(E::zero(), |sum, (coefficient, &weight)| {
                sum + weight * eq_eval_at_index(coefficient_point, coefficient)
            }))
    }
}

impl<E: Field> CoefficientPackingCompactFactors<E> {
    fn validate_point(&self, point: &[E]) -> Result<(), AkitaError> {
        let point_variables = u32::try_from(point.len())
            .map_err(|_| AkitaError::InvalidSetup("packing point domain overflow".into()))?;
        let expected = 1usize
            .checked_shl(point_variables)
            .ok_or_else(|| AkitaError::InvalidSetup("packing point domain overflow".into()))?;
        let padded = self
            .physical_field_len
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("packing field domain overflow".into()))?;
        if expected != padded {
            return Err(AkitaError::InvalidSize {
                expected: padded.trailing_zeros() as usize,
                actual: point.len(),
            });
        }
        Ok(())
    }

    /// Evaluate packed E and Q relation weights without expanding their
    /// claim/block/digit/plane support.
    pub fn evaluate_relation_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        self.validate_point(point)?;
        let evaluate_affine = || -> Result<E, AkitaError> {
            let mut coefficient_evaluations = [None; usize::BITS as usize];
            let mut base_offsets = Vec::new();
            let mut base_scales = Vec::new();
            let mut affine = E::zero();
            let mut family_index = 0usize;
            while let Some(family) = self.affine_relation_families.get(family_index) {
                let coefficient_bits = family.coefficient_len.trailing_zeros() as usize;
                let coefficient_evaluation = if let Some(evaluation) = coefficient_evaluations
                    .get(coefficient_bits)
                    .copied()
                    .flatten()
                {
                    evaluation
                } else {
                    let evaluation = family.coefficient_evaluation_at_point(point)?;
                    let slot = coefficient_evaluations
                        .get_mut(coefficient_bits)
                        .ok_or(AkitaError::InvalidProof)?;
                    *slot = Some(evaluation);
                    evaluation
                };
                let (_, outer_point) = point
                    .split_at_checked(coefficient_bits)
                    .ok_or(AkitaError::InvalidProof)?;
                let next_family_index = family_index.checked_add(1).ok_or_else(|| {
                    AkitaError::InvalidSetup("packing affine family index overflow".into())
                })?;
                let remaining_families = self
                    .affine_relation_families
                    .get(next_family_index..)
                    .ok_or(AkitaError::InvalidProof)?;
                let incompatible_offset = remaining_families
                    .iter()
                    .position(|candidate| !family.shares_contraction_geometry(candidate));
                let group_end = if let Some(offset) = incompatible_offset {
                    next_family_index.checked_add(offset).ok_or_else(|| {
                        AkitaError::InvalidSetup("packing affine family index overflow".into())
                    })?
                } else {
                    self.affine_relation_families.len()
                };
                let group_len = group_end
                    .checked_sub(family_index)
                    .ok_or(AkitaError::InvalidProof)?;
                base_offsets.clear();
                base_scales.clear();
                base_offsets.try_reserve(group_len).map_err(|_| {
                    AkitaError::InvalidInput("packing affine base allocation failed".into())
                })?;
                base_scales.try_reserve(group_len).map_err(|_| {
                    AkitaError::InvalidInput("packing affine scale allocation failed".into())
                })?;
                let family_group = self
                    .affine_relation_families
                    .get(family_index..group_end)
                    .ok_or(AkitaError::InvalidProof)?;
                for candidate in family_group {
                    base_offsets.push(candidate.base_offset);
                    base_scales.push(candidate.scalar * coefficient_evaluation);
                }
                affine += eval_affine_digit_intervals(
                    outer_point,
                    &base_offsets,
                    0,
                    family.outer_len,
                    family.outer_stride,
                    family.digit_stride,
                    family.digit_weights.as_ref(),
                    family.outer_weights.as_ref(),
                    &[],
                    &base_scales,
                )?;
                family_index = group_end;
            }
            Ok(affine)
        };
        let evaluate_quotient = || {
            eval_boolean_pair_tensor_families::<_, false, false>(
                &[],
                point,
                &self.quotient_families,
            )
        };
        let (affine, quotient) = cfg_join!(evaluate_affine, evaluate_quotient);
        Ok(affine? + quotient?)
    }

    /// Evaluate the direct-opening and packing-Z structured terms from their
    /// retained tensor factors.
    pub fn evaluate_stage2_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        self.validate_point(point)?;
        let evaluate = |left: &[E], families: &[EqPairTensorFamily<E>]| match self.basis {
            crate::BasisMode::Lagrange => {
                eval_boolean_pair_tensor_families::<_, false, false>(left, point, families)
            }
            crate::BasisMode::Monomial => {
                eval_boolean_pair_tensor_families::<_, true, false>(left, point, families)
            }
        };
        Ok(
            evaluate(&self.direct_opening_point, &self.direct_opening_families)?
                + evaluate(&self.packing_z_point, &self.packing_z_families)?,
        )
    }
}

pub(super) struct CompactFactorInputs<'a, E: Field> {
    pub geometry: SubringCoefficientPackingGeometry,
    pub prepared_point: &'a PreparedSubringCoefficientPackingPoint<E>,
    pub witness_layout: &'a WitnessLayout,
    pub group_index: usize,
    pub num_claims: usize,
    pub num_live_blocks: usize,
    pub d_d: usize,
    pub consistency_row: usize,
    pub physical_field_len: usize,
    pub consistency_weight: E,
    pub scalar_claim_weight: E,
    pub denominator: E,
    pub claim_coefficients: &'a [E],
    pub challenge_alpha: &'a [E],
    pub alpha_powers: &'a [E],
    pub basis_elements: &'a [E],
    pub opening_gadget: &'a [E],
    pub quotient_gadget: &'a [E],
    pub witness_gadget: &'a [E],
    pub fold_gadget: &'a [E],
}

fn dyadic_segments(range: Range<usize>) -> Result<Vec<Range<usize>>, AkitaError> {
    let mut segments = Vec::new();
    segments
        .try_reserve_exact(usize::BITS as usize)
        .map_err(|_| {
            AkitaError::InvalidInput("coefficient-packing segment allocation failed".into())
        })?;
    let mut start = range.start;
    while start < range.end {
        let remaining = range.end - start;
        let len = 1usize << (usize::BITS - remaining.leading_zeros() - 1);
        let end = start.checked_add(len).ok_or_else(|| {
            AkitaError::InvalidSetup("coefficient-packing tensor segment overflow".into())
        })?;
        segments.push(start..end);
        start = end;
    }
    Ok(segments)
}

fn geometric_axis<E: Field>(
    left_stride: usize,
    right_stride: usize,
    weights: &[E],
    len: usize,
) -> Result<EqPairTensorAxis<E>, AkitaError> {
    if len == 0 || !len.is_power_of_two() || len > weights.len() {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing geometric tensor axis is malformed".into(),
        ));
    }
    let mut factors = Vec::new();
    factors
        .try_reserve_exact(len.trailing_zeros() as usize)
        .map_err(|_| {
            AkitaError::InvalidInput("coefficient-packing axis allocation failed".into())
        })?;
    for bit in 0..len.trailing_zeros() as usize {
        let coordinate = 1usize << bit;
        factors.push([
            E::one(),
            *weights.get(coordinate).ok_or(AkitaError::InvalidProof)?,
        ]);
    }
    let axis = EqPairTensorAxis::bit_product(left_stride, right_stride, factors)?;
    for coordinate in 0..len {
        if axis.coordinate_weight(coordinate) != weights.get(coordinate).copied() {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing tensor weights are not geometric".into(),
            ));
        }
    }
    Ok(axis)
}

fn shared_slice<E: Field>(values: &[E], label: &'static str) -> Result<Arc<[E]>, AkitaError> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(values.len())
        .map_err(|_| AkitaError::InvalidInput(format!("{label} allocation failed")))?;
    owned.extend_from_slice(values);
    Ok(owned.into())
}

fn geometric_axes<E: Field>(
    left_stride: usize,
    right_stride: usize,
    weights: &[E],
    segments: &[Range<usize>],
) -> Result<Vec<EqPairTensorAxis<E>>, AkitaError> {
    let mut axes = Vec::new();
    axes.try_reserve_exact(segments.len()).map_err(|_| {
        AkitaError::InvalidInput("coefficient-packing axis table allocation failed".into())
    })?;
    for segment in segments {
        axes.push(geometric_axis(
            left_stride,
            right_stride,
            weights,
            segment.len(),
        )?);
    }
    Ok(axes)
}

fn reserve_families<T>(families: &mut Vec<T>, additional: usize) -> Result<(), AkitaError> {
    let total = families
        .len()
        .checked_add(additional)
        .ok_or_else(|| AkitaError::InvalidInput("packing tensor family count overflow".into()))?;
    if total > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: total,
        });
    }
    families
        .try_reserve_exact(additional)
        .map_err(|_| AkitaError::InvalidInput("packing tensor family allocation failed".into()))
}

fn extend_point<E: Field>(target: &mut Vec<E>, point: &[E]) -> Result<(), AkitaError> {
    target
        .try_reserve(point.len())
        .map_err(|_| AkitaError::InvalidInput("packing point allocation failed".into()))?;
    target.extend_from_slice(point);
    Ok(())
}

pub(super) fn prepare_compact_factors<E: Field>(
    inputs: CompactFactorInputs<'_, E>,
) -> Result<CoefficientPackingCompactFactors<E>, AkitaError> {
    let s = inputs.geometry.challenge_subring_dimension();
    let k = inputs.geometry.extension_degree();
    let kh = inputs.geometry.subring_embedding_stride();
    let d_a = inputs.geometry.a_ring_dimension();
    let partial_width = inputs.geometry.partial_base_field_width();
    if inputs.prepared_point.geometry() != inputs.geometry
        || inputs.claim_coefficients.len() != inputs.num_claims
        || inputs.challenge_alpha.len()
            != inputs
                .num_claims
                .checked_mul(inputs.num_live_blocks)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("packing challenge count overflow".into())
                })?
        || inputs.alpha_powers.len() != s
        || inputs.basis_elements.len() != k
        || inputs.d_d == 0
        || !partial_width.is_multiple_of(inputs.d_d)
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing compact factors disagree with their geometry".into(),
        ));
    }

    let semantic_stride = partial_width
        .checked_mul(inputs.opening_gadget.len())
        .ok_or_else(|| AkitaError::InvalidSetup("packing E stride overflow".into()))?;
    let digit_segments = dyadic_segments(0..inputs.opening_gadget.len())?;
    let opening_digit_axes = geometric_axes(0, inputs.d_d, inputs.opening_gadget, &digit_segments)?;
    let claim_coefficients = shared_slice(inputs.claim_coefficients, "packing claim axis")?;
    let coefficient_weights = shared_slice(inputs.alpha_powers, "packing coefficient axis")?;
    let opening_gadget = shared_slice(inputs.opening_gadget, "packing digit axis")?;
    let mut affine_relation_families = Vec::new();
    let mut quotient_families = Vec::new();
    let mut direct_opening_families = Vec::new();
    for unit in inputs.witness_layout.units_for_group(inputs.group_index)? {
        if unit.num_live_blocks() == 0 {
            continue;
        }
        let semantic_count = inputs
            .num_claims
            .checked_mul(unit.num_live_blocks())
            .ok_or_else(|| AkitaError::InvalidSetup("packing E semantic count overflow".into()))?;
        let mut challenge_weights = Vec::new();
        challenge_weights
            .try_reserve_exact(semantic_count)
            .map_err(|_| {
                AkitaError::InvalidInput("packing challenge axis allocation failed".into())
            })?;
        for claim in 0..inputs.num_claims {
            for global_block in unit.global_block_range() {
                let challenge = claim
                    .checked_mul(inputs.num_live_blocks)
                    .and_then(|base| base.checked_add(global_block))
                    .and_then(|index| inputs.challenge_alpha.get(index).copied())
                    .ok_or(AkitaError::InvalidProof)?;
                challenge_weights.push(challenge);
            }
        }
        let challenge_weights: Arc<[E]> = challenge_weights.into();
        let block_segments = dyadic_segments(unit.global_block_range())?;
        let mut plane_segments = 0usize;
        for plane in 0..k {
            let mut plane_offset = 0usize;
            while plane_offset < s {
                let flat = plane
                    .checked_mul(s)
                    .and_then(|base| base.checked_add(plane_offset))
                    .ok_or_else(|| AkitaError::InvalidSetup("packing plane overflow".into()))?;
                let role_coefficient = flat % inputs.d_d;
                let coefficient_count = (inputs.d_d - role_coefficient).min(s - plane_offset);
                plane_segments = plane_segments.checked_add(1).ok_or_else(|| {
                    AkitaError::InvalidInput("packing plane segment count overflow".into())
                })?;
                plane_offset += coefficient_count;
            }
        }
        reserve_families(&mut affine_relation_families, plane_segments)?;
        // A full direct-opening plane can cross several physical E role
        // subcolumns. When those boundaries are coefficient-aligned, retain
        // the role as another unit tensor axis instead of emitting one family
        // per subcolumn. Unlike the affine relation above, this source term
        // has no alpha weight along the role axis.
        let compact_role_axis = if s > inputs.d_d && s.is_multiple_of(inputs.d_d) {
            let role_subcolumns = s / inputs.d_d;
            let role_stride = inputs
                .opening_gadget
                .len()
                .checked_mul(inputs.d_d)
                .ok_or_else(|| AkitaError::InvalidSetup("packing role stride overflow".into()))?;
            Some(EqPairTensorAxis::unit(
                role_subcolumns,
                inputs.d_d,
                role_stride,
            ))
        } else {
            None
        };
        let direct_plane_segments = if compact_role_axis.is_some() {
            k
        } else {
            plane_segments
        };
        let direct_count = direct_plane_segments
            .checked_mul(digit_segments.len())
            .and_then(|count| count.checked_mul(block_segments.len()))
            .ok_or_else(|| AkitaError::InvalidInput("packing family count overflow".into()))?;
        reserve_families(&mut direct_opening_families, direct_count)?;
        for (plane, &basis_element) in inputs.basis_elements.iter().enumerate() {
            let mut plane_offset = 0usize;
            while plane_offset < s {
                let flat = plane
                    .checked_mul(s)
                    .and_then(|base| base.checked_add(plane_offset))
                    .ok_or_else(|| AkitaError::InvalidSetup("packing plane overflow".into()))?;
                let role_subcolumn = flat / inputs.d_d;
                let role_coefficient = flat % inputs.d_d;
                let coefficient_count = (inputs.d_d - role_coefficient).min(s - plane_offset);
                let alpha_offset = *inputs
                    .alpha_powers
                    .get(plane_offset)
                    .ok_or(AkitaError::InvalidProof)?;
                let physical_start = unit.e_coefficient_index(
                    inputs.d_d,
                    inputs.num_claims,
                    inputs.opening_gadget.len(),
                    0,
                    unit.global_block_start(),
                    role_subcolumn,
                    0,
                    role_coefficient,
                )?;
                if !physical_start.is_multiple_of(coefficient_count)
                    || !semantic_stride.is_multiple_of(coefficient_count)
                    || !inputs.d_d.is_multiple_of(coefficient_count)
                {
                    return Err(AkitaError::InvalidSetup(
                        "packing E affine geometry is not coefficient aligned".into(),
                    ));
                }
                affine_relation_families.push(CoefficientPackingAffineRelationFamily {
                    scalar: inputs.consistency_weight * basis_element * alpha_offset,
                    coefficient_weights: Arc::clone(&coefficient_weights),
                    coefficient_len: coefficient_count,
                    base_offset: physical_start / coefficient_count,
                    outer_len: semantic_count,
                    outer_stride: semantic_stride / coefficient_count,
                    digit_stride: inputs.d_d / coefficient_count,
                    digit_weights: Arc::clone(&opening_gadget),
                    outer_weights: Arc::clone(&challenge_weights),
                });
                plane_offset += coefficient_count;
            }
        }

        let claim_stride = unit
            .num_live_blocks()
            .checked_mul(semantic_stride)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("direct-opening claim stride overflow".into())
            })?;
        for (plane, &basis_element) in inputs.basis_elements.iter().enumerate() {
            let mut plane_offset = 0usize;
            while plane_offset < s {
                let flat = plane
                    .checked_mul(s)
                    .and_then(|base| base.checked_add(plane_offset))
                    .ok_or_else(|| AkitaError::InvalidSetup("packing plane overflow".into()))?;
                let role_subcolumn = flat / inputs.d_d;
                let role_coefficient = flat % inputs.d_d;
                let coefficient_count = if compact_role_axis.is_some() {
                    if role_coefficient != 0 {
                        return Err(AkitaError::InvalidSetup(
                            "compact packing role axis is not coefficient aligned".into(),
                        ));
                    }
                    inputs.d_d
                } else {
                    (inputs.d_d - role_coefficient).min(s - plane_offset)
                };
                if compact_role_axis.is_some() && coefficient_count > s - plane_offset {
                    return Err(AkitaError::InvalidSetup(
                        "compact packing role axis exceeds its plane".into(),
                    ));
                }
                for (digit_segment, digit_axis) in digit_segments.iter().zip(&opening_digit_axes) {
                    for block_segment in &block_segments {
                        let physical_start = unit.e_coefficient_index(
                            inputs.d_d,
                            inputs.num_claims,
                            inputs.opening_gadget.len(),
                            0,
                            block_segment.start,
                            role_subcolumn,
                            digit_segment.start,
                            role_coefficient,
                        )?;
                        let left_offset = block_segment
                            .start
                            .checked_mul(s)
                            .and_then(|base| base.checked_add(plane_offset))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("direct-opening offset overflow".into())
                            })?;
                        let opening_weight = inputs
                            .opening_gadget
                            .get(digit_segment.start)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)?;
                        let mut axes = vec![EqPairTensorAxis::unit(coefficient_count, 1, 1)];
                        if let Some(role_axis) = &compact_role_axis {
                            axes.push(role_axis.clone());
                        }
                        axes.extend([
                            digit_axis.clone(),
                            EqPairTensorAxis::unit(block_segment.len(), s, semantic_stride),
                            EqPairTensorAxis::dense(
                                0,
                                claim_stride,
                                Arc::clone(&claim_coefficients),
                            ),
                        ]);
                        direct_opening_families.push(EqPairTensorFamily::new(
                            left_offset,
                            physical_start,
                            inputs.scalar_claim_weight * basis_element * opening_weight,
                            axes,
                        )?);
                    }
                }
                if compact_role_axis.is_some() {
                    plane_offset = s;
                } else {
                    plane_offset += coefficient_count;
                }
            }
        }
    }

    let quotient_digit_segments = dyadic_segments(0..inputs.quotient_gadget.len())?;
    let quotient_digit_axes = geometric_axes(
        0,
        partial_width,
        inputs.quotient_gadget,
        &quotient_digit_segments,
    )?;
    let alpha_axis = geometric_axis(0, 1, inputs.alpha_powers, s)?;
    reserve_families(
        &mut quotient_families,
        k.checked_mul(quotient_digit_segments.len())
            .ok_or_else(|| AkitaError::InvalidInput("packing family count overflow".into()))?,
    )?;
    for (plane, &basis_element) in inputs.basis_elements.iter().enumerate() {
        for (digit_segment, digit_axis) in quotient_digit_segments.iter().zip(&quotient_digit_axes)
        {
            let physical_start = inputs.witness_layout.r_coefficient_index(
                inputs.consistency_row,
                digit_segment.start,
                plane,
                0,
            )?;
            let quotient_weight = inputs
                .quotient_gadget
                .get(digit_segment.start)
                .copied()
                .ok_or(AkitaError::InvalidProof)?;
            quotient_families.push(EqPairTensorFamily::new(
                0,
                physical_start,
                -(inputs.consistency_weight * basis_element * inputs.denominator * quotient_weight),
                vec![alpha_axis.clone(), digit_axis.clone()],
            )?);
        }
    }

    let witness_digit_segments = dyadic_segments(0..inputs.witness_gadget.len())?;
    let fold_digit_segments = dyadic_segments(0..inputs.fold_gadget.len())?;
    let mut packing_z_families = Vec::new();
    let position_stride = inputs
        .witness_gadget
        .len()
        .checked_mul(inputs.fold_gadget.len())
        .and_then(|count| count.checked_mul(d_a))
        .ok_or_else(|| AkitaError::InvalidSetup("packing-Z position stride overflow".into()))?;
    let witness_stride = inputs
        .fold_gadget
        .len()
        .checked_mul(d_a)
        .ok_or_else(|| AkitaError::InvalidSetup("packing-Z witness stride overflow".into()))?;
    let witness_digit_axes = geometric_axes(
        0,
        witness_stride,
        inputs.witness_gadget,
        &witness_digit_segments,
    )?;
    let fold_digit_axes = geometric_axes(0, d_a, inputs.fold_gadget, &fold_digit_segments)?;
    let packing_alpha_axis = geometric_axis(0, kh, inputs.alpha_powers, s)?;
    let unit_count = inputs
        .witness_layout
        .units_for_group(inputs.group_index)?
        .count();
    reserve_families(
        &mut packing_z_families,
        unit_count
            .checked_mul(witness_digit_segments.len())
            .and_then(|count| count.checked_mul(fold_digit_segments.len()))
            .ok_or_else(|| AkitaError::InvalidInput("packing-Z family count overflow".into()))?,
    )?;
    for unit in inputs.witness_layout.units_for_group(inputs.group_index)? {
        for (witness_segment, witness_axis) in
            witness_digit_segments.iter().zip(&witness_digit_axes)
        {
            for (fold_segment, fold_axis) in fold_digit_segments.iter().zip(&fold_digit_axes) {
                let physical_start = unit.z_coefficient_index(
                    d_a,
                    inputs.prepared_point.num_positions_per_block(),
                    inputs.witness_gadget.len(),
                    inputs.fold_gadget.len(),
                    0,
                    witness_segment.start,
                    fold_segment.start,
                    0,
                )?;
                let witness_weight = inputs
                    .witness_gadget
                    .get(witness_segment.start)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                let fold_weight = inputs
                    .fold_gadget
                    .get(fold_segment.start)
                    .copied()
                    .ok_or(AkitaError::InvalidProof)?;
                packing_z_families.push(EqPairTensorFamily::new(
                    0,
                    physical_start,
                    -(inputs.consistency_weight * witness_weight * fold_weight),
                    vec![
                        EqPairTensorAxis::unit(kh, 1, 1),
                        packing_alpha_axis.clone(),
                        fold_axis.clone(),
                        witness_axis.clone(),
                        EqPairTensorAxis::unit(
                            inputs.prepared_point.num_positions_per_block(),
                            kh,
                            position_stride,
                        ),
                    ],
                )?);
            }
        }
    }

    let mut direct_opening_point = Vec::new();
    extend_point(
        &mut direct_opening_point,
        inputs.prepared_point.tail_point(),
    )?;
    extend_point(
        &mut direct_opening_point,
        inputs.prepared_point.block_point(),
    )?;
    let mut packing_z_point = Vec::new();
    extend_point(&mut packing_z_point, inputs.prepared_point.packing_point())?;
    extend_point(&mut packing_z_point, inputs.prepared_point.position_point())?;
    Ok(CoefficientPackingCompactFactors {
        basis: inputs.prepared_point.basis(),
        physical_field_len: inputs.physical_field_len,
        direct_opening_point: direct_opening_point.into(),
        packing_z_point: packing_z_point.into(),
        affine_relation_families,
        quotient_families,
        direct_opening_families,
        packing_z_families,
    })
}
