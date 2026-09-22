//! Allocation-free common relation modulus sizing.

use super::*;

impl RelationWitnessGeometry {
    /// Common Stage-2 coefficient block derived from row polynomial moduli.
    pub fn relation_coefficient_block_len(&self) -> Result<usize, AkitaError> {
        let layout = self.rhs_layout();
        layout.validate()?;
        // Retain the materialized row API's checked count, even though sizing
        // only needs one geometry per nonempty family.
        layout.checked_uncompressed_row_count()?;
        let mut coefficient_block = usize::MAX;
        for group in &layout.groups {
            coefficient_block =
                coefficient_block.min(group.opening_geometry.polynomial_modulus_dimension());
            let inner = RelationRowGeometry::native(group.role_dims.d_a())?;
            let outer = RelationRowGeometry::native(group.role_dims.d_b())?;
            if group.n_a != 0 {
                coefficient_block = coefficient_block.min(inner.polynomial_modulus_dimension());
            }
            if group.logical_b_rows()? != 0 {
                coefficient_block = coefficient_block.min(outer.polynomial_modulus_dimension());
            }
        }
        let opening = RelationRowGeometry::native(layout.d_ring_dimension)?;
        if layout.n_d != 0 {
            coefficient_block = coefficient_block.min(opening.polynomial_modulus_dimension());
        }
        // Compression rows are excluded from the common block, but the old
        // materialization validated their geometry before filtering them.
        if let Some(compression) = &layout.compression {
            for map_index in 0..crate::COMPRESSION_MAP_COUNT {
                for plan in &compression.group_plans {
                    RelationRowGeometry::native(plan.maps()[map_index].ring_dimension())?;
                }
                RelationRowGeometry::native(
                    compression.opening_plan.maps()[map_index].ring_dimension(),
                )?;
            }
        }
        // Every checked geometry has power-of-two modulus and coordinate
        // width, so the smallest present modulus divides every present width.
        // layout.validate() guarantees a nonempty consistency-row family.
        Ok(coefficient_block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn materialized_block(layout: &RelationRhsLayout) -> Result<usize, AkitaError> {
        let rows: Vec<_> = layout
            .row_families()?
            .into_iter()
            .filter(|row| {
                !matches!(
                    row,
                    RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
                )
            })
            .map(RelationRowFamily::geometry)
            .collect();
        let block = rows
            .iter()
            .map(|row| row.polynomial_modulus_dimension())
            .min()
            .ok_or_else(|| AkitaError::InvalidSetup("relation rows are empty".into()))?;
        if rows
            .iter()
            .any(|row| !row.physical_coefficient_width().is_multiple_of(block))
        {
            return Err(AkitaError::InvalidSetup("misaligned rows".into()));
        }
        Ok(block)
    }

    #[test]
    fn sizing_matches_expanded_rows_with_empty_families_and_extension_planes() {
        for inner in [32, 64, 128] {
            for outer in [32, 64, 128] {
                if outer > inner {
                    continue;
                }
                for n_a in [0, 1, 3] {
                    for n_b in [0, 1, 3] {
                        for n_d in [0, 1, 3] {
                            for planes in [1, 2, 4] {
                                let dims = CommitmentRingDims {
                                    inner,
                                    outer,
                                    opening: 32,
                                };
                                let mut layout = RelationRhsLayout::uniform(
                                    dims,
                                    n_d,
                                    n_a,
                                    n_b,
                                    CommitmentSliceCount::ONE,
                                    2,
                                )
                                .unwrap();
                                layout.groups[0].opening_geometry =
                                    RelationRowGeometry::new(inner / planes, planes).unwrap();
                                let expected =
                                    materialized_block(&layout).map_err(|e| e.to_string());
                                let geometry = RelationWitnessGeometry::from_parts(planes, layout);
                                assert_eq!(
                                    geometry
                                        .relation_coefficient_block_len()
                                        .map_err(|e| e.to_string()),
                                    expected
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sizing_matches_heterogeneous_sliced_rows_and_count_overflow_errors() {
        let mut layout = RelationRhsLayout::uniform(
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 32,
            },
            2,
            3,
            2,
            CommitmentSliceCount::TWO,
            2,
        )
        .unwrap();
        layout.groups[1].role_dims = CommitmentRingDims {
            inner: 64,
            outer: 32,
            opening: 32,
        };
        layout.groups[1].opening_geometry = RelationRowGeometry::new(16, 2).unwrap();
        layout.groups[1].n_a = 1;
        layout.groups[1].physical_b_rows = 3;
        layout.groups[1].outer_slice_count = CommitmentSliceCount::FOUR;
        // The second consistency row has width 32 but modulus 16; using its
        // physical width or only the first group's geometry would be wrong.
        let expected = materialized_block(&layout).unwrap();
        assert_eq!(expected, 16);
        assert_eq!(
            RelationWitnessGeometry::from_parts(2, layout.clone())
                .relation_coefficient_block_len()
                .unwrap(),
            expected
        );

        let mut b_overflow = layout.clone();
        b_overflow.groups[0].physical_b_rows = usize::MAX;
        let mut d_overflow = layout;
        d_overflow.n_d = usize::MAX;
        // Both failures must be detected before the oracle allocates its rows:
        // B fails its slice multiplication, D fails the final count addition.
        for malformed in [b_overflow, d_overflow] {
            let expected = materialized_block(&malformed).unwrap_err();
            let actual = RelationWitnessGeometry::from_parts(2, malformed)
                .relation_coefficient_block_len()
                .unwrap_err();
            assert!(matches!(expected, AkitaError::InvalidSetup(_)));
            assert!(matches!(actual, AkitaError::InvalidSetup(_)));
            assert_eq!(actual.to_string(), expected.to_string());
        }
    }

    #[test]
    fn sizing_excludes_compression_rows_but_validates_compression_shape() {
        let mut layout = RelationRhsLayout::uniform(
            CommitmentRingDims::uniform(128),
            2,
            3,
            4,
            CommitmentSliceCount::ONE,
            2,
        )
        .unwrap();
        let group_plan = compression_plan(SisModulusProfileId::Q64Offset59, 4, 128).unwrap();
        layout.compression = Some(RelationCompressionLayout {
            group_indices: vec![0, 1],
            group_plans: vec![group_plan.clone(), group_plan],
            opening_plan: compression_plan(SisModulusProfileId::Q64Offset59, 2, 128).unwrap(),
        });
        let expected = materialized_block(&layout).unwrap();
        assert_eq!(
            RelationWitnessGeometry::from_parts(1, layout.clone())
                .relation_coefficient_block_len()
                .unwrap(),
            expected
        );
        layout.n_d += 1;
        assert!(materialized_block(&layout).is_err());
        assert!(RelationWitnessGeometry::from_parts(1, layout)
            .relation_coefficient_block_len()
            .is_err());
    }

    #[test]
    fn sizing_rejects_row_count_overflow_and_invalid_layout() {
        let mut layout = RelationRhsLayout::uniform(
            CommitmentRingDims::uniform(64),
            1,
            1,
            1,
            CommitmentSliceCount::ONE,
            1,
        )
        .unwrap();
        layout.groups[0].n_a = usize::MAX;
        assert!(layout.row_families().is_err());
        assert!(RelationWitnessGeometry::from_parts(1, layout.clone())
            .relation_coefficient_block_len()
            .is_err());
        layout.groups.clear();
        assert!(RelationWitnessGeometry::from_parts(1, layout)
            .relation_coefficient_block_len()
            .is_err());
    }
}
