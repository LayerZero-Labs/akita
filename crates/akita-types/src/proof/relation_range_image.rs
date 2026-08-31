//! Checked geometry shared by the direct relation/range-image sum-check.

use std::ops::Range;

use akita_algebra::offset_eq::{EqPairTensorAxis, EqPairTensorFamily};
use akita_error::AkitaError;
use jolt_field::{Field, Ring};

use crate::{
    CommitmentRingDims, CommittedGroupParams, DigitRangePlan, FlatBooleanDomain,
    InnerCommitSecurityRoute, OpeningClaimsLayout, PhysicalL2NormProofShape,
    RelationAddressGeometry, RelationRowFamily, RelationWitnessGeometry, WitnessLayout,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct PhysicalResponseSegment {
    physical_start: usize,
    physical_len: usize,
    witness_start: usize,
}

/// Checked map from the canonical physical folded response to the Z digit
/// addresses in the Stage-1/Stage-2 witness table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalResponsePlan {
    domain: FlatBooleanDomain,
    shape: PhysicalL2NormProofShape,
    segments: Vec<PhysicalResponseSegment>,
    ring_dimension: usize,
    fold_digit_count: usize,
    fold_basis: usize,
}

impl PhysicalResponsePlan {
    /// Derive the complete physical response map from the same
    /// [`WitnessLayout`] used by ring switching.
    pub fn new(
        params: &CommittedGroupParams,
        plan: &RelationRangeImagePlan,
    ) -> Result<Option<Self>, AkitaError> {
        let InnerCommitSecurityRoute::L2 {
            norm_proof_shape: shape,
            ..
        } = params.inner().matrix.security_route()
        else {
            return Ok(None);
        };
        shape.validate()?;
        if !params.precommitted_groups().is_empty() || plan.groups.len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "L2 response proofs are restricted to scalar recursive folds".into(),
            ));
        }
        let ring_dimension = params.d_a();
        let fold_digit_count = params.num_digits_fold();
        let fold_basis = 1usize
            .checked_shl(params.open().digits.log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("fold basis overflow".into()))?;
        if ring_dimension == 0 || fold_digit_count == 0 {
            return Err(AkitaError::InvalidSetup(
                "L2 response proof has empty ring or digit geometry".into(),
            ));
        }
        let digit_stride = fold_digit_count
            .checked_mul(ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("L2 response stride overflow".into()))?;
        let mut physical_cursor = 0usize;
        let mut segments = Vec::with_capacity(plan.witness_layout.units().len());
        for unit in plan.witness_layout.units() {
            if unit.group_index() != 0 || !unit.z_range().len().is_multiple_of(digit_stride) {
                return Err(AkitaError::InvalidSetup(
                    "L2 response Z layout is not a scalar digit grid".into(),
                ));
            }
            let physical_len = unit.z_range().len() / fold_digit_count;
            segments.push(PhysicalResponseSegment {
                physical_start: physical_cursor,
                physical_len,
                witness_start: unit.z_range().start,
            });
            physical_cursor = physical_cursor
                .checked_add(physical_len)
                .ok_or_else(|| AkitaError::InvalidSetup("L2 response length overflow".into()))?;
        }
        if physical_cursor != shape.physical_response_len() {
            return Err(AkitaError::InvalidSetup(format!(
                "L2 response shape length {} disagrees with WitnessLayout length {physical_cursor}",
                shape.physical_response_len()
            )));
        }
        if matches!(
            shape,
            PhysicalL2NormProofShape::LimbGram { limb_count, .. }
                if limb_count != fold_digit_count
        ) {
            return Err(AkitaError::InvalidSetup(
                "L2 limb count disagrees with folded-response digit depth".into(),
            ));
        }
        if physical_cursor > plan.digit_witness_domain().domain_len() {
            return Err(AkitaError::InvalidSetup(
                "L2 response table exceeds the Stage-1 domain".into(),
            ));
        }
        Ok(Some(Self {
            domain: plan.digit_witness_domain(),
            shape,
            segments,
            ring_dimension,
            fold_digit_count,
            fold_basis,
        }))
    }

    /// Scheduled integer norm proof shape.
    #[must_use]
    pub const fn shape(&self) -> PhysicalL2NormProofShape {
        self.shape
    }

    /// Shared padded Stage-1 witness domain.
    #[must_use]
    pub const fn domain(&self) -> FlatBooleanDomain {
        self.domain
    }

    /// Number of virtual tables bound at the final Stage-1 point.
    #[must_use]
    pub const fn virtual_table_count(&self) -> usize {
        match self.shape {
            PhysicalL2NormProofShape::Direct { .. } => 1,
            PhysicalL2NormProofShape::LimbGram { limb_count, .. } => limb_count,
        }
    }

    /// Integer folded-response digit basis.
    #[must_use]
    pub const fn fold_basis(&self) -> usize {
        self.fold_basis
    }

    /// Number of balanced fold digits recomposed into one physical response.
    #[must_use]
    pub const fn fold_digit_count(&self) -> usize {
        self.fold_digit_count
    }

    /// Materialize the unpadded centered integer virtual tables used for exact
    /// norm and limb-Gram claim construction.
    pub fn materialize_virtual_integers(
        &self,
        compact_witness_len: usize,
        mut decode_digits: impl FnMut(usize, &mut [i8]) -> Result<(), AkitaError>,
    ) -> Result<Vec<Vec<i128>>, AkitaError> {
        if compact_witness_len != self.domain.live_len() {
            return Err(AkitaError::InvalidSize {
                expected: self.domain.live_len(),
                actual: compact_witness_len,
            });
        }
        let mut tables =
            vec![vec![0i128; self.shape.physical_response_len()]; self.virtual_table_count()];
        let digit_stride = self
            .fold_digit_count
            .checked_mul(self.ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("L2 response stride overflow".into()))?;
        let mut digits = vec![0i8; self.ring_dimension];
        for segment in &self.segments {
            for row in 0..segment.physical_len / self.ring_dimension {
                let witness_row = segment
                    .witness_start
                    .checked_add(row.checked_mul(digit_stride).ok_or_else(|| {
                        AkitaError::InvalidSetup("L2 witness row overflow".into())
                    })?)
                    .ok_or_else(|| AkitaError::InvalidSetup("L2 witness row overflow".into()))?;
                let physical_row = segment
                    .physical_start
                    .checked_add(row.checked_mul(self.ring_dimension).ok_or_else(|| {
                        AkitaError::InvalidSetup("L2 physical row overflow".into())
                    })?)
                    .ok_or_else(|| AkitaError::InvalidSetup("L2 physical row overflow".into()))?;
                match self.shape {
                    PhysicalL2NormProofShape::Direct { .. } => {
                        let table = tables.first_mut().ok_or(AkitaError::InvalidProof)?;
                        let output = table
                            .get_mut(physical_row..physical_row + self.ring_dimension)
                            .ok_or(AkitaError::InvalidProof)?;
                        let mut basis_power = 1i128;
                        for limb in 0..self.fold_digit_count {
                            let index = witness_row + limb * self.ring_dimension;
                            decode_digits(index, &mut digits)?;
                            for (value, &digit) in output.iter_mut().zip(&digits) {
                                *value = value
                                    .checked_add(
                                        i128::from(digit).checked_mul(basis_power).ok_or_else(
                                            || {
                                                AkitaError::InvalidSetup(
                                                    "L2 basis power overflow".into(),
                                                )
                                            },
                                        )?,
                                    )
                                    .ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "L2 response recomposition overflow".into(),
                                        )
                                    })?;
                            }
                            basis_power = basis_power
                                .checked_mul(self.fold_basis as i128)
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("L2 basis power overflow".into())
                                })?;
                        }
                    }
                    PhysicalL2NormProofShape::LimbGram { limb_count, .. } => {
                        for (limb, table) in tables.iter_mut().enumerate().take(limb_count) {
                            let index = witness_row + limb * self.ring_dimension;
                            decode_digits(index, &mut digits)?;
                            let output = table
                                .get_mut(physical_row..physical_row + self.ring_dimension)
                                .ok_or(AkitaError::InvalidProof)?;
                            for (value, &digit) in output.iter_mut().zip(&digits) {
                                *value = i128::from(digit);
                            }
                        }
                    }
                }
            }
        }
        Ok(tables)
    }

    /// Canonical paired-equality geometry that batches all virtual response
    /// evaluations into the existing Stage-2 witness relation.
    pub fn virtualization_families<E: Field + Ring>(
        &self,
        batching: &[E],
    ) -> Result<Vec<EqPairTensorFamily<E>>, AkitaError> {
        if batching.len() != self.virtual_table_count() {
            return Err(AkitaError::InvalidSize {
                expected: self.virtual_table_count(),
                actual: batching.len(),
            });
        }
        let limb_weights = match self.shape {
            PhysicalL2NormProofShape::Direct { .. } => {
                let mut weights = Vec::with_capacity(self.fold_digit_count);
                let mut power = E::one();
                let basis = E::from_u64(self.fold_basis as u64);
                for _ in 0..self.fold_digit_count {
                    weights.push(batching[0] * power);
                    power *= basis;
                }
                weights
            }
            PhysicalL2NormProofShape::LimbGram { limb_count, .. } => {
                batching.iter().copied().take(limb_count).collect()
            }
        };
        let digit_stride = self
            .fold_digit_count
            .checked_mul(self.ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("L2 response stride overflow".into()))?;
        let mut families = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            let row_count = segment.physical_len / self.ring_dimension;
            families.push(EqPairTensorFamily::new(
                segment.witness_start,
                segment.physical_start,
                E::one(),
                vec![
                    EqPairTensorAxis::unit(self.ring_dimension, 1, 1),
                    EqPairTensorAxis::dense(self.ring_dimension, 0, limb_weights.clone()),
                    EqPairTensorAxis::unit(row_count, digit_stride, self.ring_dimension),
                ],
            )?);
        }
        Ok(families)
    }
}

/// Reconstruct the exact nonnegative square sum from canonical block-major,
/// upper-triangular limb Gram claims.
pub fn reconstruct_l2_sq_from_gram(
    shape: PhysicalL2NormProofShape,
    fold_basis: usize,
    claims: &[i128],
) -> Result<u128, AkitaError> {
    let layout = shape.limb_gram_layout()?.ok_or_else(|| {
        AkitaError::InvalidInput("direct L2 shape has no limb-Gram reconstruction".into())
    })?;
    let expected = layout.subclaim_count();
    if claims.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: claims.len(),
        });
    }
    let basis = i128::try_from(fold_basis)
        .map_err(|_| AkitaError::InvalidSetup("L2 fold basis exceeds i128".into()))?;
    let power_count = layout
        .limb_count()
        .checked_mul(2)
        .ok_or_else(|| AkitaError::InvalidSetup("L2 power count overflow".into()))?;
    let mut powers = Vec::with_capacity(power_count);
    let mut power = 1i128;
    for _ in 0..power_count {
        powers.push(power);
        power = power
            .checked_mul(basis)
            .ok_or_else(|| AkitaError::InvalidSetup("L2 basis power overflow".into()))?;
    }
    let mut total = 0i128;
    for block_index in 0..layout.block_count() {
        for (left, right) in layout.limb_pairs() {
            let claim_index = layout
                .subclaim_index(block_index, left, right)
                .ok_or(AkitaError::InvalidProof)?;
            let claim = claims
                .get(claim_index)
                .copied()
                .ok_or(AkitaError::InvalidProof)?;
            let exponent = left
                .checked_add(right)
                .ok_or_else(|| AkitaError::InvalidSetup("L2 exponent overflow".into()))?;
            let scale = powers
                .get(exponent)
                .copied()
                .ok_or(AkitaError::InvalidProof)?
                .checked_mul(if left == right { 1 } else { 2 })
                .ok_or_else(|| AkitaError::InvalidSetup("L2 Gram scale overflow".into()))?;
            total =
                total
                    .checked_add(claim.checked_mul(scale).ok_or_else(|| {
                        AkitaError::InvalidSetup("L2 Gram product overflow".into())
                    })?)
                    .ok_or_else(|| AkitaError::InvalidSetup("L2 Gram sum overflow".into()))?;
        }
    }
    u128::try_from(total).map_err(|_| AkitaError::InvalidProof)
}

/// One commitment group's claims and witness units in physical processing order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationRangeImageGroupPlan {
    group_index: usize,
    claim_range: Range<usize>,
    unit_indices: Vec<usize>,
}

impl RelationRangeImageGroupPlan {
    /// Index of the commitment group in [`OpeningClaimsLayout`].
    #[must_use]
    pub fn group_index(&self) -> usize {
        self.group_index
    }

    /// Global claim indices owned by this group.
    #[must_use]
    pub fn claim_range(&self) -> Range<usize> {
        self.claim_range.clone()
    }

    /// Chunk-ordered indices into [`WitnessLayout::units`] owned by this group.
    #[must_use]
    pub fn unit_indices(&self) -> &[usize] {
        &self.unit_indices
    }
}

/// Checked semantic plan for the direct relation/evaluation-trace/range-image sum-check.
///
/// This plan joins the existing flat coefficient domain, Stage 1 range basis,
/// semantic witness layout, opening-claim order, and per-role dimensions. Mutable
/// compact/folded tables remain prover state and are intentionally not represented here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationRangeImagePlan {
    relation_witness_geometry: RelationWitnessGeometry,
    relation_address_geometry: RelationAddressGeometry,
    digit_range_plan: DigitRangePlan,
    witness_layout: WitnessLayout,
    groups: Vec<RelationRangeImageGroupPlan>,
}

impl RelationRangeImagePlan {
    /// Join and validate every layout authority used by the direct fused sum-check.
    ///
    /// # Errors
    ///
    /// Returns an error if role dimensions are unsupported, the flat live prefix does
    /// not exactly encode the compact semantic witness layout, or witness groups/chunks
    /// do not follow the authenticated opening order.
    pub fn new(
        relation_witness_geometry: RelationWitnessGeometry,
        relation_address_geometry: RelationAddressGeometry,
        digit_range_plan: DigitRangePlan,
        witness_layout: WitnessLayout,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<Self, AkitaError> {
        opening_batch.check()?;
        witness_layout.validate_internal_ranges()?;

        let expected_address_geometry = RelationAddressGeometry::for_relation(
            &relation_witness_geometry,
            relation_address_geometry.outgoing_witness_ring_dimension(),
            witness_layout.live_coeff_len(),
        )?;
        if relation_address_geometry != expected_address_geometry {
            return Err(AkitaError::InvalidSetup(
                "relation address geometry disagrees with relation rows".into(),
            ));
        }

        let digit_witness_domain = relation_address_geometry.digit_witness_domain();
        let expected_live_len = witness_layout.live_coeff_len();
        if digit_witness_domain.live_len() != expected_live_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_live_len,
                actual: digit_witness_domain.live_len(),
            });
        }

        let coeff_count = relation_address_geometry.relation_coefficient_block_len();
        if !digit_witness_domain.live_len().is_multiple_of(coeff_count) {
            return Err(AkitaError::InvalidSetup(
                "digit witness is not aligned to the current relation coefficient block".into(),
            ));
        }
        if relation_address_geometry.live_relation_lane_count() == 0 {
            return Err(AkitaError::InvalidSetup(
                "relation/range-image plan requires a non-empty lane domain".into(),
            ));
        }

        let order = opening_batch.root_group_order()?;
        let num_groups = order.len();
        if num_groups == 0 || !witness_layout.units().len().is_multiple_of(num_groups) {
            return Err(AkitaError::InvalidSetup(
                "witness units do not form a chunk/group grid".into(),
            ));
        }
        let num_chunks = witness_layout.units().len() / num_groups;
        let groups = order
            .iter()
            .enumerate()
            .map(|(group_position, &group_index)| {
                Ok(RelationRangeImageGroupPlan {
                    group_index,
                    claim_range: opening_batch.root_group_claim_range(group_index)?,
                    unit_indices: (0..num_chunks)
                        .map(|chunk_index| chunk_index * num_groups + group_position)
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let mut expected_global_block_starts = vec![0usize; num_groups];
        let mut witness_cursor = 0usize;
        for chunk_index in 0..num_chunks {
            for (group_position, &group_index) in order.iter().enumerate() {
                let unit_index = chunk_index
                    .checked_mul(num_groups)
                    .and_then(|base| base.checked_add(group_position))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("witness unit index overflow".into())
                    })?;
                let unit = witness_layout.units().get(unit_index).ok_or_else(|| {
                    AkitaError::InvalidSetup("witness chunk/group unit is missing".into())
                })?;
                if unit.group_index() != group_index
                    || unit.chunk_index() != chunk_index
                    || unit.global_block_start() != expected_global_block_starts[group_position]
                {
                    return Err(AkitaError::InvalidSetup(
                        "witness chunks do not form one ordered global block partition".into(),
                    ));
                }
                let z_range = unit.z_range();
                let e_range = unit.e_range();
                let t_range = unit.t_range();
                if unit.e_geometry()
                    != relation_witness_geometry.group_opening_geometry(group_index)?
                {
                    return Err(AkitaError::InvalidSetup(
                        "witness E geometry disagrees with its relation group".into(),
                    ));
                }
                if z_range.start != witness_cursor
                    || z_range.end != e_range.start
                    || e_range.end != t_range.start
                    || z_range.is_empty()
                    || e_range.is_empty() != (unit.num_live_blocks() == 0)
                    || t_range.is_empty() != (unit.num_live_blocks() == 0)
                {
                    return Err(AkitaError::InvalidSetup(
                        "witness unit ranges disagree with the chunk geometry".into(),
                    ));
                }

                witness_cursor = t_range.end;
                expected_global_block_starts[group_position] = expected_global_block_starts
                    [group_position]
                    .checked_add(unit.num_live_blocks())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("witness block coverage overflow".into())
                    })?;
            }
        }

        if witness_layout.tail_range().start != witness_cursor
            || witness_layout.tail_range().end != witness_layout.live_coeff_len()
        {
            return Err(AkitaError::InvalidSetup(
                "witness layout tail disagrees with its chunk-major body".into(),
            ));
        }
        let row_geometries = relation_witness_geometry.rhs_layout().row_geometries()?;
        match witness_layout.relation_quotient_layout() {
            crate::RelationQuotientLayout::QuotientLift { rows, .. } => {
                if rows.len() != row_geometries.len()
                    || rows
                        .iter()
                        .zip(row_geometries)
                        .any(|(row, geometry)| row.geometry() != geometry)
                {
                    return Err(AkitaError::InvalidSetup(
                        "witness quotient rows disagree with relation geometry".into(),
                    ));
                }
            }
            crate::RelationQuotientLayout::ReducedEvaluation => {}
        }

        Ok(Self {
            relation_witness_geometry,
            relation_address_geometry,
            digit_range_plan,
            witness_layout,
            groups,
        })
    }

    /// Canonical relation-witness address geometry.
    #[must_use]
    pub fn relation_address_geometry(&self) -> RelationAddressGeometry {
        self.relation_address_geometry
    }

    /// Canonical row and opening-coefficient geometry.
    #[must_use]
    pub const fn relation_witness_geometry(&self) -> &RelationWitnessGeometry {
        &self.relation_witness_geometry
    }

    /// Complete coefficient-domain authority shared with Stage 1.
    #[must_use]
    pub fn digit_witness_domain(&self) -> FlatBooleanDomain {
        self.relation_address_geometry.digit_witness_domain()
    }

    /// Global range-basis authority shared with Stage 1.
    #[must_use]
    pub fn digit_range_plan(&self) -> DigitRangePlan {
        self.digit_range_plan
    }

    /// Canonical semantic witness layout.
    #[must_use]
    pub fn witness_layout(&self) -> &WitnessLayout {
        &self.witness_layout
    }

    /// Nested inner/outer/opening ring dimensions.
    #[must_use]
    pub fn role_dims(&self) -> CommitmentRingDims {
        self.relation_address_geometry.role_dims()
    }

    /// Groups in authenticated root processing order.
    #[must_use]
    pub fn groups(&self) -> &[RelationRangeImageGroupPlan] {
        &self.groups
    }

    /// Canonical consistency-row index for one authenticated opening group.
    pub fn consistency_row_index(&self, group_index: usize) -> Result<usize, AkitaError> {
        self.relation_witness_geometry
            .rhs_layout()
            .row_families()?
            .iter()
            .position(|row| {
                matches!(
                    row,
                    RelationRowFamily::Consistency {
                        group_index: row_group_index,
                        ..
                    } if *row_group_index == group_index
                )
            })
            .ok_or(AkitaError::InvalidProof)
    }

    /// Canonical trailing scalar-opening row after every physical relation row.
    pub fn scalar_opening_row_index(&self) -> Result<usize, AkitaError> {
        Ok(self
            .relation_witness_geometry
            .rhs_layout()
            .row_families()?
            .len())
    }

    /// Exact Boolean width of the authenticated relation-row point.
    pub fn relation_row_index_num_vars(&self) -> Result<usize, AkitaError> {
        let row_count = self
            .scalar_opening_row_index()?
            .checked_add(1)
            .ok_or_else(|| AkitaError::InvalidSetup("relation-row count overflow".into()))?;
        let padded = row_count
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("relation-row index width overflow".into()))?;
        Ok(padded.trailing_zeros() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dyadic_block_ranges, CommitmentSliceCount, PolynomialGroupLayout, RelationGroupRows,
        RelationRhsLayout, RelationRowGeometry, WitnessQuotientRowLayout, WitnessUnitLayout,
    };

    fn test_relation_geometry(
        opening_batch: &OpeningClaimsLayout,
        role_dims: CommitmentRingDims,
    ) -> RelationWitnessGeometry {
        let opening_geometry = RelationRowGeometry::native(role_dims.d_a()).unwrap();
        let groups = opening_batch
            .root_group_order()
            .unwrap()
            .into_iter()
            .map(|group_index| RelationGroupRows {
                group_index,
                role_dims,
                opening_geometry,
                opening_method: crate::OpeningMethod::EvaluationTrace,
                n_a: 1,
                physical_b_rows: 1,
                outer_slice_count: CommitmentSliceCount::ONE,
            })
            .collect();
        RelationWitnessGeometry::from_parts(
            1,
            RelationRhsLayout::new_for_test(role_dims.d_d(), 1, groups),
        )
    }

    fn test_layout(
        opening_batch: &OpeningClaimsLayout,
        relation_geometry: &RelationWitnessGeometry,
        chunks_per_group: usize,
        source_ring_dimension: usize,
    ) -> WitnessLayout {
        let mut units = Vec::new();
        let mut cursor = 0usize;
        let order = opening_batch.root_group_order().unwrap();
        let block_ranges = (0..opening_batch.num_groups())
            .map(|group_index| dyadic_block_ranges(group_index + 1, chunks_per_group).unwrap())
            .collect::<Vec<_>>();
        for chunk_index in 0..chunks_per_group {
            for &group_index in &order {
                let num_claims = opening_batch
                    .group_layout(group_index)
                    .unwrap()
                    .num_polynomials();
                let blocks = block_ranges
                    .get(group_index)
                    .and_then(|ranges| ranges.get(chunk_index))
                    .unwrap()
                    .clone();
                let z_range = cursor..cursor + 2 * source_ring_dimension;
                let e_range =
                    z_range.end..z_range.end + blocks.len() * num_claims * source_ring_dimension;
                let t_range = e_range.end
                    ..e_range.end + 2 * blocks.len() * num_claims * source_ring_dimension;
                cursor = t_range.end;
                units.push(WitnessUnitLayout::new_for_test(
                    group_index,
                    chunk_index,
                    blocks.start,
                    blocks.len(),
                    z_range,
                    e_range,
                    relation_geometry
                        .group_opening_geometry(group_index)
                        .unwrap(),
                    t_range,
                ));
            }
        }
        let quotient_rows = relation_geometry
            .rhs_layout()
            .row_geometries()
            .unwrap()
            .into_iter()
            .map(|geometry| {
                let range = cursor..cursor + geometry.physical_coefficient_width();
                cursor = range.end;
                WitnessQuotientRowLayout::new_for_test(geometry, range)
            })
            .collect();
        WitnessLayout::new_for_test(units, quotient_rows, 1)
    }

    fn plan_for(
        group_sizes: &[usize],
        chunks_per_group: usize,
        role_dims: CommitmentRingDims,
        opening_ring_dimension: usize,
        basis: usize,
    ) -> RelationRangeImagePlan {
        let opening_batch = OpeningClaimsLayout::from_groups(
            group_sizes
                .iter()
                .enumerate()
                .map(|(group_index, &size)| PolynomialGroupLayout::new(group_index + 2, size))
                .collect(),
        )
        .unwrap();
        let relation_geometry = test_relation_geometry(&opening_batch, role_dims);
        let witness_layout = test_layout(
            &opening_batch,
            &relation_geometry,
            chunks_per_group,
            role_dims.d_a(),
        );
        let geometry = RelationAddressGeometry::for_relation(
            &relation_geometry,
            opening_ring_dimension,
            witness_layout.live_coeff_len(),
        )
        .unwrap();
        RelationRangeImagePlan::new(
            relation_geometry,
            geometry,
            DigitRangePlan::new(basis).unwrap(),
            witness_layout,
            &opening_batch,
        )
        .unwrap()
    }

    #[test]
    fn plan_covers_group_chunk_dimension_and_basis_cross_product() {
        for group_sizes in [&[2][..], &[1, 2][..]] {
            for chunks_per_group in [1, 2] {
                for role_dims in [
                    CommitmentRingDims::uniform(64),
                    CommitmentRingDims {
                        inner: 128,
                        outer: 64,
                        opening: 64,
                    },
                ] {
                    for basis in [4, 8, 16, 32, 64] {
                        let plan = plan_for(
                            group_sizes,
                            chunks_per_group,
                            role_dims,
                            role_dims.d_d(),
                            basis,
                        );
                        assert_eq!(plan.digit_range_plan().basis(), basis);
                        assert_eq!(plan.role_dims(), role_dims);
                        let geometry = plan.relation_address_geometry();
                        assert_eq!(geometry.relation_coefficient_block_len(), role_dims.d_d());
                        assert_eq!(
                            geometry.live_relation_lane_count()
                                * geometry.relation_coefficient_block_len(),
                            plan.digit_witness_domain().live_len()
                        );
                        assert_eq!(plan.groups().len(), group_sizes.len());
                        assert_eq!(
                            plan.groups()[0].group_index(),
                            group_sizes.len().saturating_sub(1)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn plan_preserves_global_claim_ranges_in_physical_group_order() {
        let plan = plan_for(&[2, 3], 2, CommitmentRingDims::uniform(64), 64, 8);
        assert_eq!(plan.groups()[0].group_index(), 1);
        assert_eq!(plan.groups()[0].claim_range(), 2..5);
        assert_eq!(plan.groups()[0].unit_indices(), &[0, 2]);
        assert_eq!(plan.groups()[1].group_index(), 0);
        assert_eq!(plan.groups()[1].claim_range(), 0..2);
        assert_eq!(plan.groups()[1].unit_indices(), &[1, 3]);
    }

    #[test]
    fn plan_common_block_ignores_outgoing_repacking() {
        let plan = plan_for(&[1], 1, CommitmentRingDims::uniform(128), 64, 8);
        let geometry = plan.relation_address_geometry();
        assert_eq!(geometry.relation_coefficient_block_len(), 128);
        assert_eq!(
            geometry.live_relation_lane_count() * geometry.relation_coefficient_block_len(),
            plan.digit_witness_domain().live_len()
        );
    }

    #[test]
    fn plan_rejects_domain_and_physical_order_disagreement() {
        let opening_batch = OpeningClaimsLayout::from_group_sizes(3, &[1, 1]).unwrap();
        let relation_geometry =
            test_relation_geometry(&opening_batch, CommitmentRingDims::uniform(64));
        let witness_layout = test_layout(&opening_batch, &relation_geometry, 1, 64);
        let live_len = witness_layout.live_coeff_len();
        let short_geometry =
            RelationAddressGeometry::new(CommitmentRingDims::uniform(64), 64, live_len - 64)
                .unwrap();
        assert!(RelationRangeImagePlan::new(
            relation_geometry.clone(),
            short_geometry,
            DigitRangePlan::new(8).unwrap(),
            witness_layout.clone(),
            &opening_batch,
        )
        .is_err());

        let mut reversed_units = witness_layout.units().to_vec();
        reversed_units.reverse();
        let malformed = WitnessLayout::new_for_test(
            reversed_units,
            witness_layout.r_rows().to_vec(),
            witness_layout
                .quotient_depth()
                .expect("quotient-lift fixture"),
        );
        let geometry =
            RelationAddressGeometry::new(CommitmentRingDims::uniform(64), 64, live_len).unwrap();
        assert!(RelationRangeImagePlan::new(
            relation_geometry,
            geometry,
            DigitRangePlan::new(8).unwrap(),
            malformed,
            &opening_batch,
        )
        .is_err());
    }

    #[test]
    fn plan_rejects_equal_length_geometry_permutations() {
        let opening_batch = OpeningClaimsLayout::from_group_sizes(3, &[1]).unwrap();
        let relation_geometry =
            test_relation_geometry(&opening_batch, CommitmentRingDims::uniform(64));
        let witness_layout = test_layout(&opening_batch, &relation_geometry, 1, 64);
        let address_geometry = RelationAddressGeometry::for_relation(
            &relation_geometry,
            64,
            witness_layout.live_coeff_len(),
        )
        .unwrap();

        let mut wrong_rows = witness_layout.r_rows().to_vec();
        wrong_rows[0] = WitnessQuotientRowLayout::new_for_test(
            RelationRowGeometry::new(32, 2).unwrap(),
            wrong_rows[0].range(),
        );
        let malformed = WitnessLayout::new_for_test(
            witness_layout.units().to_vec(),
            wrong_rows,
            witness_layout
                .quotient_depth()
                .expect("quotient-lift fixture"),
        );
        assert!(RelationRangeImagePlan::new(
            relation_geometry.clone(),
            address_geometry,
            DigitRangePlan::new(8).unwrap(),
            malformed,
            &opening_batch,
        )
        .is_err());

        let mut duplicated_rows = witness_layout.r_rows().to_vec();
        duplicated_rows[1] = WitnessQuotientRowLayout::new_for_test(
            duplicated_rows[1].geometry(),
            duplicated_rows[0].range(),
        );
        let malformed = WitnessLayout::new_for_test(
            witness_layout.units().to_vec(),
            duplicated_rows,
            witness_layout
                .quotient_depth()
                .expect("quotient-lift fixture"),
        );
        assert!(RelationRangeImagePlan::new(
            relation_geometry.clone(),
            address_geometry,
            DigitRangePlan::new(8).unwrap(),
            malformed,
            &opening_batch,
        )
        .is_err());

        let mut wrong_units = witness_layout.units().to_vec();
        let unit = &wrong_units[0];
        wrong_units[0] = WitnessUnitLayout::new_for_test(
            unit.group_index(),
            unit.chunk_index(),
            unit.global_block_start(),
            unit.num_live_blocks(),
            unit.z_range(),
            unit.e_range(),
            RelationRowGeometry::new(32, 2).unwrap(),
            unit.t_range(),
        );
        let malformed = WitnessLayout::new_for_test(
            wrong_units,
            witness_layout.r_rows().to_vec(),
            witness_layout
                .quotient_depth()
                .expect("quotient-lift fixture"),
        );
        assert!(RelationRangeImagePlan::new(
            relation_geometry,
            address_geometry,
            DigitRangePlan::new(8).unwrap(),
            malformed,
            &opening_batch,
        )
        .is_err());
    }

    #[test]
    fn limb_gram_reconstruction_handles_signs_and_block_boundaries() {
        let shape = PhysicalL2NormProofShape::LimbGram {
            physical_response_len: 5,
            block_len: 2,
            limb_count: 2,
        };
        // Block-major upper triangles for limbs
        // l0 = [-2, -1, 0, 1, 2], l1 = [1, -2, 2, 0, -1].
        let claims = [5, 0, 5, 1, 0, 4, 4, -2, 1];
        assert_eq!(
            reconstruct_l2_sq_from_gram(shape, 4, &claims).expect("signed block Gram norm"),
            154
        );
        assert!(reconstruct_l2_sq_from_gram(shape, 4, &claims[..8]).is_err());
    }

    #[test]
    fn limb_gram_reconstruction_rejects_integer_overflow() {
        let shape = PhysicalL2NormProofShape::LimbGram {
            physical_response_len: 1,
            block_len: 1,
            limb_count: 2,
        };
        assert!(
            reconstruct_l2_sq_from_gram(shape, usize::MAX, &[i128::MAX, 0, i128::MAX]).is_err()
        );
    }
}
