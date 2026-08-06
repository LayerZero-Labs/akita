//! Checked geometry shared by the direct relation/range-image sum-check.

use std::ops::Range;

use akita_field::AkitaError;

use crate::{
    CommitmentRingDims, DigitRangePlan, FlatBooleanDomain, OpeningClaimsLayout,
    RelationAddressGeometry, WitnessLayout,
};

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
        relation_address_geometry: RelationAddressGeometry,
        digit_range_plan: DigitRangePlan,
        witness_layout: WitnessLayout,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<Self, AkitaError> {
        opening_batch.check()?;

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
                    || unit.num_live_blocks() == 0
                {
                    return Err(AkitaError::InvalidSetup(
                        "witness chunks do not form one ordered global block partition".into(),
                    ));
                }
                let z_range = unit.z_range();
                let e_range = unit.e_range();
                let t_range = unit.t_range();
                if z_range.start != witness_cursor
                    || z_range.end != e_range.start
                    || e_range.end != t_range.start
                    || z_range.is_empty()
                    || e_range.is_empty()
                    || t_range.is_empty()
                {
                    return Err(AkitaError::InvalidSetup(
                        "witness unit ranges are not non-empty and contiguous".into(),
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

        if witness_layout.r_range().start != witness_cursor
            || witness_layout.r_range().end != witness_layout.live_coeff_len()
        {
            return Err(AkitaError::InvalidSetup(
                "witness layout does not end in one shared quotient range".into(),
            ));
        }

        Ok(Self {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dyadic_block_ranges, PolynomialGroupLayout, WitnessQuotientRowLayout, WitnessUnitLayout,
    };

    fn test_layout(
        opening_batch: &OpeningClaimsLayout,
        chunks_per_group: usize,
        source_ring_dimension: usize,
    ) -> WitnessLayout {
        let mut units = Vec::new();
        let mut cursor = 0usize;
        let order = opening_batch.root_group_order().unwrap();
        let block_ranges = (0..opening_batch.num_groups())
            .map(|group_index| {
                dyadic_block_ranges(2 * chunks_per_group + group_index + 1, chunks_per_group)
                    .unwrap()
            })
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
                    t_range,
                ));
            }
        }
        WitnessLayout::new_for_test(
            units,
            vec![WitnessQuotientRowLayout::new_for_test(
                source_ring_dimension,
                cursor..cursor + source_ring_dimension,
            )],
            1,
        )
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
        let witness_layout = test_layout(&opening_batch, chunks_per_group, role_dims.d_a());
        let geometry = RelationAddressGeometry::new(
            role_dims,
            opening_ring_dimension,
            witness_layout.live_coeff_len(),
        )
        .unwrap();
        RelationRangeImagePlan::new(
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
        let witness_layout = test_layout(&opening_batch, 1, 64);
        let live_len = witness_layout.live_coeff_len();
        let short_geometry =
            RelationAddressGeometry::new(CommitmentRingDims::uniform(64), 64, live_len - 64)
                .unwrap();
        assert!(RelationRangeImagePlan::new(
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
            witness_layout.quotient_depth(),
        );
        let geometry =
            RelationAddressGeometry::new(CommitmentRingDims::uniform(64), 64, live_len).unwrap();
        assert!(RelationRangeImagePlan::new(
            geometry,
            DigitRangePlan::new(8).unwrap(),
            malformed,
            &opening_batch,
        )
        .is_err());
    }
}
