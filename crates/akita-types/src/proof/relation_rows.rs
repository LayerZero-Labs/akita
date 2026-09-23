//! Canonical relation row runs, materialization, and common-block sizing.

use super::{RelationRhsLayout, RelationRowFamily, RelationRowGeometry};
use crate::OpeningMethod;
use akita_error::AkitaError;

/// One canonical run of rows. Sizing visits runs; only the materialized API
/// expands their per-row identities.
enum RelationRowRun {
    Consistency {
        group_index: usize,
        opening_method: OpeningMethod,
        geometry: RelationRowGeometry,
    },
    Inner {
        group_index: usize,
        count: usize,
        geometry: RelationRowGeometry,
    },
    Outer {
        group_index: usize,
        slice_index: usize,
        count: usize,
        geometry: RelationRowGeometry,
    },
    Opening {
        count: usize,
        geometry: RelationRowGeometry,
    },
    CompressionF {
        group_index: usize,
        map_index: usize,
        geometry: RelationRowGeometry,
    },
    CompressionH {
        map_index: usize,
        geometry: RelationRowGeometry,
    },
}

impl RelationRowRun {
    fn common_block_modulus(&self) -> Option<usize> {
        let geometry = match self {
            Self::Consistency { geometry, .. } => geometry,
            Self::Inner {
                count: 1..,
                geometry,
                ..
            }
            | Self::Outer {
                count: 1..,
                geometry,
                ..
            }
            | Self::Opening {
                count: 1..,
                geometry,
            } => geometry,
            Self::Inner { count: 0, .. }
            | Self::Outer { count: 0, .. }
            | Self::Opening { count: 0, .. }
            | Self::CompressionF { .. }
            | Self::CompressionH { .. } => return None,
        };
        Some(geometry.polynomial_modulus_dimension())
    }

    fn append_rows(self, rows: &mut Vec<RelationRowFamily>) {
        match self {
            Self::Consistency {
                group_index,
                opening_method,
                geometry,
            } => rows.push(RelationRowFamily::Consistency {
                group_index,
                opening_method,
                geometry,
            }),
            Self::Inner {
                group_index,
                count,
                geometry,
            } => rows.extend((0..count).map(|row| RelationRowFamily::Inner {
                group_index,
                row,
                geometry,
            })),
            Self::Outer {
                group_index,
                slice_index,
                count,
                geometry,
            } => rows.extend((0..count).map(|physical_row| RelationRowFamily::Outer {
                group_index,
                slice_index,
                physical_row,
                geometry,
            })),
            Self::Opening { count, geometry } => {
                rows.extend((0..count).map(|row| RelationRowFamily::Opening { row, geometry }));
            }
            Self::CompressionF {
                group_index,
                map_index,
                geometry,
            } => rows.push(RelationRowFamily::CompressionF {
                group_index,
                map_index,
                geometry,
            }),
            Self::CompressionH {
                map_index,
                geometry,
            } => rows.push(RelationRowFamily::CompressionH {
                map_index,
                geometry,
            }),
        }
    }
}

impl RelationRhsLayout {
    fn checked_uncompressed_row_count(&self) -> Result<usize, AkitaError> {
        let row_count = self.groups.iter().try_fold(0usize, |rows, group| {
            rows.checked_add(1)
                .and_then(|rows| rows.checked_add(group.n_a))
                .and_then(|rows| rows.checked_add(group.logical_b_rows().ok()?))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation quotient row count overflow".into())
                })
        })?;
        row_count
            .checked_add(self.n_d)
            .ok_or_else(|| AkitaError::InvalidSetup("relation quotient row count overflow".into()))
    }

    /// Visit checked row geometries in canonical relation order without
    /// expanding logical A, B, or D rows.
    fn visit_row_runs(&self, mut visit: impl FnMut(RelationRowRun)) -> Result<(), AkitaError> {
        for group in &self.groups {
            let group_index = group.group_index;
            visit(RelationRowRun::Consistency {
                group_index,
                opening_method: group.opening_method,
                geometry: group.opening_geometry,
            });
            let geometry = RelationRowGeometry::native(group.role_dims.d_a())?;
            visit(RelationRowRun::Inner {
                group_index,
                count: group.n_a,
                geometry,
            });
            let geometry = RelationRowGeometry::native(group.role_dims.d_b())?;
            for slice_index in 0..group.outer_slice_count.get() {
                visit(RelationRowRun::Outer {
                    group_index,
                    slice_index,
                    count: group.physical_b_rows,
                    geometry,
                });
            }
        }
        let geometry = RelationRowGeometry::native(self.d_ring_dimension)?;
        visit(RelationRowRun::Opening {
            count: self.n_d,
            geometry,
        });
        if let Some(compression) = &self.compression {
            for map_index in 0..crate::COMPRESSION_MAP_COUNT {
                for (&group_index, plan) in compression
                    .group_indices
                    .iter()
                    .zip(&compression.group_plans)
                {
                    let geometry =
                        RelationRowGeometry::native(plan.maps()[map_index].ring_dimension())?;
                    visit(RelationRowRun::CompressionF {
                        group_index,
                        map_index,
                        geometry,
                    });
                }
                let geometry = RelationRowGeometry::native(
                    compression.opening_plan.maps()[map_index].ring_dimension(),
                )?;
                visit(RelationRowRun::CompressionH {
                    map_index,
                    geometry,
                });
            }
        }
        Ok(())
    }

    /// Semantic row families in canonical relation and quotient order.
    pub fn row_families(&self) -> Result<Vec<RelationRowFamily>, AkitaError> {
        self.validate()?;
        let row_count = self.checked_uncompressed_row_count()?;
        let mut rows = Vec::with_capacity(row_count);
        self.visit_row_runs(|run| run.append_rows(&mut rows))?;
        Ok(rows)
    }

    /// Minimum modulus among present non-compression relation rows.
    ///
    /// Unlike `CommitmentRingDims::common_relation_coeff_count`, absent A or D
    /// row families do not constrain this block. Dimension-only schedule
    /// validation uses that conservative role bound without constructing a
    /// relation layout; witness sizing uses this exact present-row bound.
    pub fn relation_coefficient_block_len(&self) -> Result<usize, AkitaError> {
        self.validate()?;
        // Preserve row-count overflow rejection and error precedence before
        // visiting geometries, even though sizing never materializes rows.
        self.checked_uncompressed_row_count()?;
        let mut block = None;
        self.visit_row_runs(|run| {
            if let Some(modulus) = run.common_block_modulus() {
                block = Some(block.map_or(modulus, |current: usize| current.min(modulus)));
            }
        })?;
        block.ok_or_else(|| AkitaError::InvalidSetup("relation rows are empty".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::{RelationRhsLayout, RelationRowFamily, RelationRowGeometry};
    use crate::layout::CommitmentRingDims;
    use crate::proof::relation::compression_plan;
    use crate::proof::relation::layout_types::RelationCompressionLayout;
    use crate::{CommitmentSliceCount, SisModulusProfileId};
    use akita_error::AkitaError;

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
                                assert_eq!(
                                    layout
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
        assert_eq!(layout.relation_coefficient_block_len().unwrap(), expected);

        let mut b_overflow = layout.clone();
        b_overflow.groups[0].physical_b_rows = usize::MAX;
        let mut d_overflow = layout;
        d_overflow.n_d = usize::MAX;
        // Both failures must be detected before the oracle allocates its rows:
        // B fails its slice multiplication, D fails the final count addition.
        for malformed in [b_overflow, d_overflow] {
            let expected = materialized_block(&malformed).unwrap_err();
            let actual = malformed.relation_coefficient_block_len().unwrap_err();
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
        assert_eq!(layout.relation_coefficient_block_len().unwrap(), expected);
        layout.n_d += 1;
        assert!(materialized_block(&layout).is_err());
        assert!(layout.relation_coefficient_block_len().is_err());
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
        assert!(layout.relation_coefficient_block_len().is_err());
        layout.groups.clear();
        assert!(layout.relation_coefficient_block_len().is_err());
    }

    #[test]
    fn row_runs_preserve_sliced_and_compression_order() {
        let mut layout = RelationRhsLayout::uniform(
            CommitmentRingDims::uniform(64),
            1,
            1,
            1,
            CommitmentSliceCount::TWO,
            2,
        )
        .unwrap();
        let group_plan = compression_plan(SisModulusProfileId::Q64Offset59, 2, 64).unwrap();
        layout.compression = Some(RelationCompressionLayout {
            group_indices: vec![0, 1],
            group_plans: vec![group_plan.clone(), group_plan],
            opening_plan: compression_plan(SisModulusProfileId::Q64Offset59, 1, 64).unwrap(),
        });
        let actual: Vec<_> = layout
            .row_families()
            .unwrap()
            .into_iter()
            .map(|row| match row {
                RelationRowFamily::Consistency { group_index, .. } => format!("C{group_index}"),
                RelationRowFamily::Inner {
                    group_index, row, ..
                } => format!("A{group_index}:{row}"),
                RelationRowFamily::Outer {
                    group_index,
                    slice_index,
                    physical_row,
                    ..
                } => format!("B{group_index}:{slice_index}:{physical_row}"),
                RelationRowFamily::Opening { row, .. } => format!("D{row}"),
                RelationRowFamily::CompressionF {
                    group_index,
                    map_index,
                    ..
                } => format!("F{map_index}:{group_index}"),
                RelationRowFamily::CompressionH { map_index, .. } => format!("H{map_index}"),
            })
            .collect();
        let mut expected = vec![
            "C0", "A0:0", "B0:0:0", "B0:1:0", "C1", "A1:0", "B1:0:0", "B1:1:0", "D0",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        for map_index in 0..crate::COMPRESSION_MAP_COUNT {
            expected.extend([
                format!("F{map_index}:0"),
                format!("F{map_index}:1"),
                format!("H{map_index}"),
            ]);
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn sizing_handles_large_row_counts_without_expanding_them() {
        let layout = RelationRhsLayout::uniform(
            CommitmentRingDims::uniform(64),
            1,
            1 << 28,
            1,
            CommitmentSliceCount::ONE,
            1,
        )
        .unwrap();
        assert_eq!(layout.relation_coefficient_block_len().unwrap(), 64);
    }

    #[test]
    fn sizing_preserves_zero_b_error() {
        let mut layout = RelationRhsLayout::uniform(
            CommitmentRingDims::uniform(64),
            1,
            1,
            1,
            CommitmentSliceCount::TWO,
            1,
        )
        .unwrap();
        layout.groups[0].physical_b_rows = 0;
        let expected = layout.row_families().unwrap_err();
        let actual = layout.relation_coefficient_block_len().unwrap_err();
        assert_eq!(actual.to_string(), expected.to_string());
        assert!(actual
            .to_string()
            .contains("relation quotient row count overflow"));
    }
}
