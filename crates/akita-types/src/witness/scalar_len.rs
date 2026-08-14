use akita_field::AkitaError;

use super::{
    checked_align_up, dyadic_block_ranges, witness_unit_lengths, WitnessLayout, MAX_WITNESS_CHUNKS,
};
use crate::{
    CommittedGroupParams, CompressionChainPlan, OpeningClaimsLayout, RelationRowFamily,
    RelationWitnessGeometry, COMPRESSION_MAP_COUNT,
};

impl WitnessLayout {
    /// Compute the exact live length of a scalar witness without materializing
    /// its address ranges.
    ///
    /// This is the candidate-aware counterpart of [`Self::new`] for planner
    /// hot paths. `None` means that a compressed B or D source exceeds the
    /// protocol compression cap. All other malformed geometry remains an
    /// error.
    pub fn try_scalar_live_coeff_len(
        lp: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        relation_geometry: &RelationWitnessGeometry,
        num_chunks: usize,
        quotient_depth: usize,
    ) -> Result<Option<usize>, AkitaError> {
        if opening_batch.num_groups() != 1 {
            return Err(AkitaError::InvalidSetup(
                "scalar witness sizing requires exactly one opening group".into(),
            ));
        }
        if lp.has_precommitted_groups() {
            return Err(AkitaError::InvalidSetup(
                "scalar witness sizing does not accept precommitted groups".into(),
            ));
        }
        if num_chunks == 0 || quotient_depth == 0 {
            return Err(AkitaError::InvalidSetup(
                "witness layout requires non-empty groups, chunks, and quotient depth".into(),
            ));
        }
        if num_chunks > MAX_WITNESS_CHUNKS {
            return Err(AkitaError::InvalidSetup(
                "witness chunk count exceeds verifier cap".into(),
            ));
        }
        let expected_relation_geometry = RelationWitnessGeometry::for_level(
            lp,
            opening_batch,
            relation_geometry.extension_degree(),
        )?;
        if &expected_relation_geometry != relation_geometry {
            return Err(AkitaError::InvalidSetup(
                "scalar witness sizing received relation geometry for different level parameters"
                    .into(),
            ));
        }
        let relation_group_order = opening_batch.root_group_order()?;
        let group_index = *relation_group_order.first().ok_or_else(|| {
            AkitaError::InvalidSetup("scalar witness relation group is missing".into())
        })?;
        let params = lp.group_params_geometry(opening_batch, group_index)?;
        let group = opening_batch.group_layout(group_index)?;
        let role_dims = lp.group_role_dims_geometry(opening_batch, group_index)?;
        let opening_geometry = relation_geometry.group_opening_geometry(group_index)?;
        let num_claims = group.num_polynomials();
        if num_claims == 0
            || params.num_live_blocks() == 0
            || params.num_positions_per_block() == 0
            || params.num_digits_open() == 0
            || params.num_digits_inner() == 0
            || params.num_digits_outer() == 0
            || params.num_digits_fold() == 0
            || params.a_rows_len() == 0
        {
            return Err(AkitaError::InvalidSetup(
                "witness group has malformed dimensions".into(),
            ));
        }

        let mut cursor = 0usize;
        for block_range in dyadic_block_ranges(params.num_live_blocks(), num_chunks)? {
            let (z_len, e_len, t_len) = witness_unit_lengths(
                params,
                role_dims,
                opening_geometry,
                num_claims,
                block_range.len(),
            )?;
            cursor = cursor
                .checked_add(z_len)
                .and_then(|n| n.checked_add(e_len))
                .and_then(|n| n.checked_add(t_len))
                .ok_or_else(|| AkitaError::InvalidSetup("witness unit range overflow".into()))?;
        }

        let relation_layout = relation_geometry.rhs_layout();
        let row_families = relation_layout.row_families()?;
        let first_compression_row = row_families
            .iter()
            .position(|row| {
                matches!(
                    row,
                    RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
                )
            })
            .unwrap_or(row_families.len());
        for row in &row_families[..first_compression_row] {
            let len = quotient_depth
                .checked_mul(row.geometry().physical_coefficient_width())
                .ok_or_else(|| AkitaError::InvalidSetup("witness R width overflow".into()))?;
            cursor = cursor
                .checked_add(len)
                .ok_or_else(|| AkitaError::InvalidSetup("witness R range overflow".into()))?;
        }
        if !lp.payload_mode.is_compressed() {
            return Ok(Some(cursor));
        }

        let b_source_coefficients = params
            .logical_b_rows_len()?
            .checked_mul(role_dims.d_b())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation B compression shape overflow".into())
            })?;
        let Some(b_plan) = CompressionChainPlan::try_for_complete_source(
            lp.outer_commit_matrix.sis_modulus_profile(),
            b_source_coefficients,
        )?
        else {
            return Ok(None);
        };
        let d_source_coefficients = lp
            .open_commit_matrix
            .output_rank()
            .checked_mul(role_dims.d_d())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation D compression shape overflow".into())
            })?;
        let Some(d_plan) = CompressionChainPlan::try_for_complete_source(
            lp.open_commit_matrix.sis_modulus_profile(),
            d_source_coefficients,
        )?
        else {
            return Ok(None);
        };

        let relation_coefficient_block = relation_geometry.relation_coefficient_block_len()?;
        cursor = checked_align_up(
            cursor,
            relation_coefficient_block,
            "compression witness alignment overflow",
        )?;
        for map_index in 0..COMPRESSION_MAP_COUNT {
            let b_map = *b_plan
                .maps()
                .get(map_index)
                .ok_or_else(|| AkitaError::InvalidSetup("compression B map is missing".into()))?;
            let d_map = *d_plan
                .maps()
                .get(map_index)
                .ok_or_else(|| AkitaError::InvalidSetup("compression D map is missing".into()))?;
            cursor = checked_align_up(
                cursor,
                b_map.ring_dimension().max(d_map.ring_dimension()),
                "compression layer alignment overflow",
            )?;
            cursor = cursor
                .checked_add(b_map.padded_digit_count())
                .and_then(|n| n.checked_add(d_map.padded_digit_count()))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("compression witness range overflow".into())
                })?;
            for ring_dim in [b_map.ring_dimension(), d_map.ring_dimension()] {
                let len = quotient_depth.checked_mul(ring_dim).ok_or_else(|| {
                    AkitaError::InvalidSetup("compression quotient width overflow".into())
                })?;
                cursor = cursor.checked_add(len).ok_or_else(|| {
                    AkitaError::InvalidSetup("compression quotient range overflow".into())
                })?;
            }
        }
        Ok(Some(checked_align_up(
            cursor,
            relation_coefficient_block,
            "compression witness suffix alignment overflow",
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OpeningMethod, RelationRowGeometry, SisModulusProfileId};

    fn base_params(mixed_dimensions: bool) -> CommittedGroupParams {
        let profile = if mixed_dimensions {
            SisModulusProfileId::Q128OffsetA7F7
        } else {
            SisModulusProfileId::Q32Offset99
        };
        let ring_dimension = if mixed_dimensions { 64 } else { 32 };
        let mut params = CommittedGroupParams::params_only(
            profile,
            ring_dimension,
            2,
            3,
            2,
            3,
            akita_challenges::SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 32, 2, 2, 2)
        .expect("scalar test params");
        if mixed_dimensions {
            let outer = params.outer_commit_matrix;
            params.outer_commit_matrix = crate::OuterCommitMatrixParams::new_unchecked(
                outer.security_policy(),
                outer.sis_table_key().table_digest,
                outer.sis_modulus_profile(),
                outer.output_rank(),
                outer.input_width() * 2,
                outer.coeff_linf_bound(),
                32,
            );
            let open = params.open_commit_matrix;
            params.open_commit_matrix = crate::OpenCommitMatrixParams::new_unchecked(
                open.security_policy(),
                open.sis_table_key().table_digest,
                open.sis_modulus_profile(),
                open.output_rank(),
                open.input_width() * 4,
                open.coeff_linf_bound(),
                16,
            );
        }
        params
    }

    #[test]
    fn scalar_live_length_matches_materialized_layout() {
        for base in [base_params(false), base_params(true)] {
            for payload_mode in [
                crate::CommitmentPayloadMode::Raw,
                crate::CommitmentPayloadMode::Compressed,
            ] {
                for num_polynomials in [1, 2, 5] {
                    for num_chunks in [1, 2, 4] {
                        let mut params = base.clone();
                        params.payload_mode = payload_mode;
                        let opening_batch = OpeningClaimsLayout::new(0, num_polynomials)
                            .expect("scalar opening batch");
                        let relation_geometry =
                            RelationWitnessGeometry::for_evaluation_trace_execution(
                                &params,
                                &opening_batch,
                            )
                            .expect("relation geometry");
                        let materialized = WitnessLayout::new(
                            &params,
                            &opening_batch,
                            &relation_geometry,
                            num_chunks,
                            2,
                        )
                        .expect("materialized witness layout")
                        .live_coeff_len();
                        let scalar = WitnessLayout::try_scalar_live_coeff_len(
                            &params,
                            &opening_batch,
                            &relation_geometry,
                            num_chunks,
                            2,
                        )
                        .expect("scalar witness sizing")
                        .expect("compression source is supported");
                        assert_eq!(scalar, materialized);
                    }
                }
            }
        }
    }

    fn coefficient_packing_params(
        payload_mode: crate::CommitmentPayloadMode,
    ) -> CommittedGroupParams {
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            256,
            2,
            2,
            2,
            2,
            akita_challenges::SparseChallengeConfig::production_for_ring_dim(64)
                .expect("D64 challenge"),
        )
        .with_decomp(4, 8, 2, 2, 2)
        .expect("packing test params");
        params.payload_mode = payload_mode;
        params.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
        let opening = params.open_commit_matrix;
        params.open_commit_matrix = crate::OpenCommitMatrixParams::new_unchecked(
            opening.security_policy(),
            opening.sis_table_key().table_digest,
            opening.sis_modulus_profile(),
            opening.output_rank(),
            opening.input_width(),
            opening.coeff_linf_bound(),
            128,
        );
        params
    }

    #[test]
    fn coefficient_packing_sizes_one_multiplane_row_and_smaller_e() {
        let opening_batch = OpeningClaimsLayout::new(0, 2).expect("opening batch");
        for payload_mode in [
            crate::CommitmentPayloadMode::Raw,
            crate::CommitmentPayloadMode::Compressed,
        ] {
            let params = coefficient_packing_params(payload_mode);
            let relation_geometry = RelationWitnessGeometry::for_level(&params, &opening_batch, 2)
                .expect("packing relation geometry");
            let opening_geometry = relation_geometry
                .group_opening_geometry(0)
                .expect("group geometry");
            assert_eq!(opening_geometry.polynomial_modulus_dimension(), 64);
            assert_eq!(opening_geometry.coordinate_plane_count(), 2);
            assert_eq!(opening_geometry.physical_coefficient_width(), 128);
            assert_eq!(
                relation_geometry.relation_coefficient_block_len().unwrap(),
                64
            );
            assert_eq!(params.role_dims().common_relation_coeff_count(), 128);
            assert!(!64usize.is_multiple_of(128));
            assert!(opening_geometry
                .physical_coefficient_width()
                .is_multiple_of(128));
            let setup_geometry =
                crate::SetupProjectionGeometry::from_role_footprints(params.role_dims(), 1, 1, 1)
                    .expect("Stage-3 setup geometry");
            assert_eq!(setup_geometry.base_ring_dim(), 128);

            let rhs = relation_geometry.rhs_layout();
            assert_eq!(
                crate::relation_rhs_row_count(rhs),
                1 + params.inner_commit_matrix.output_rank()
                    + params.outer_commit_matrix.output_rank()
                    + params.open_commit_matrix.output_rank()
                    + if payload_mode.is_compressed() {
                        crate::COMPRESSION_MAP_COUNT * 2
                    } else {
                        0
                    }
            );
            assert_eq!(
                rhs.row_geometries().unwrap()[0],
                RelationRowGeometry::new(64, 2).unwrap()
            );
            assert_eq!(
                rhs.row_geometries().unwrap()[1],
                RelationRowGeometry::native(256).unwrap()
            );
            assert_eq!(
                crate::relation_rhs_coeff_len(rhs).unwrap(),
                rhs.row_geometries()
                    .unwrap()
                    .into_iter()
                    .map(RelationRowGeometry::physical_coefficient_width)
                    .sum::<usize>()
            );
            let v = crate::RingVec::<akita_field::Prime128OffsetA7F7>::from_coeffs(vec![
                Default::default();
                params.open_commit_matrix.output_rank() * 128
            ]);
            let u = crate::RingVec::<akita_field::Prime128OffsetA7F7>::from_coeffs(vec![
                Default::default();
                params.outer_commit_matrix.output_rank() * 256
            ]);
            assert!(crate::assemble_relation_rhs(rhs, &v, &u).is_err());

            for num_chunks in [1, 2] {
                let layout =
                    WitnessLayout::new(&params, &opening_batch, &relation_geometry, num_chunks, 2)
                        .expect("packing witness layout");
                for unit in layout.units() {
                    assert_eq!(
                        unit.e_range().len(),
                        2 * unit.num_live_blocks() * params.num_digits_open * 128
                    );
                    assert_eq!(
                        unit.z_range().len(),
                        params.num_positions_per_block
                            * params.num_digits_inner
                            * params.num_digits_fold
                            * 256
                    );
                    assert_eq!(
                        unit.t_range().len(),
                        2 * unit.num_live_blocks()
                            * params.inner_commit_matrix.output_rank()
                            * params.num_digits_outer
                            * 256
                    );
                }
                assert_eq!(layout.r_rows()[0].geometry(), opening_geometry);
                assert_eq!(layout.r_rows()[0].range().len(), 2 * 128);
                let scalar = WitnessLayout::try_scalar_live_coeff_len(
                    &params,
                    &opening_batch,
                    &relation_geometry,
                    num_chunks,
                    2,
                )
                .expect("scalar packing size")
                .expect("supported compression source");
                assert_eq!(scalar, layout.live_coeff_len());
                if num_chunks == params.witness_chunk.num_chunks {
                    let field_quotient_depth =
                        crate::sis::compute_num_digits_field_width(128, params.log_basis_open);
                    let field_layout = WitnessLayout::new(
                        &params,
                        &opening_batch,
                        &relation_geometry,
                        num_chunks,
                        field_quotient_depth,
                    )
                    .expect("field-bit witness layout");
                    assert_eq!(
                        params
                            .output_witness_len_for_field_bits(128, 2, &opening_batch)
                            .expect("field-bit witness size"),
                        field_layout.live_coeff_len()
                    );
                }
                let address_geometry = crate::RelationAddressGeometry::for_relation(
                    &relation_geometry,
                    128,
                    layout.live_coeff_len(),
                )
                .expect("Stage-2 address geometry");
                assert_eq!(address_geometry.relation_coefficient_block_len(), 64);
            }
        }
    }

    #[test]
    fn relation_geometry_distinguishes_equal_width_opening_methods() {
        let opening_batch = OpeningClaimsLayout::new(0, 1).expect("opening batch");
        let mut packing = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            128,
            2,
            2,
            2,
            2,
            akita_challenges::SparseChallengeConfig::production_for_ring_dim(64)
                .expect("D64 challenge"),
        )
        .with_decomp(4, 8, 2, 2, 2)
        .expect("equal-width params");
        packing.payload_mode = crate::CommitmentPayloadMode::Raw;
        packing.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
        let packing_geometry = RelationWitnessGeometry::for_level(&packing, &opening_batch, 2)
            .expect("packing geometry")
            .group_opening_geometry(0)
            .unwrap();

        let mut evaluation_trace = packing;
        evaluation_trace.opening_method = OpeningMethod::EvaluationTrace;
        let trace_geometry =
            RelationWitnessGeometry::for_level(&evaluation_trace, &opening_batch, 2)
                .expect("trace geometry")
                .group_opening_geometry(0)
                .unwrap();
        assert_eq!(packing_geometry.physical_coefficient_width(), 128);
        assert_eq!(trace_geometry.physical_coefficient_width(), 128);
        assert_eq!(packing_geometry.polynomial_modulus_dimension(), 64);
        assert_eq!(trace_geometry.polynomial_modulus_dimension(), 128);
        assert_ne!(packing_geometry, trace_geometry);

        let mut overlap = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            128,
            2,
            2,
            2,
            2,
            akita_challenges::SparseChallengeConfig::production_for_ring_dim(128)
                .expect("D128 challenge"),
        )
        .with_decomp(4, 8, 2, 2, 2)
        .expect("overlap params");
        overlap.payload_mode = crate::CommitmentPayloadMode::Raw;
        overlap.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 128,
        };
        let overlap_geometry = RelationWitnessGeometry::for_level(&overlap, &opening_batch, 1)
            .expect("degree-one overlap")
            .group_opening_geometry(0)
            .unwrap();
        assert_eq!(overlap_geometry, RelationRowGeometry::native(128).unwrap());
        assert!(matches!(
            overlap.opening_method,
            OpeningMethod::SubringCoefficientPacking { .. }
        ));
        let overlap_relation = RelationWitnessGeometry::for_level(&overlap, &opening_batch, 1)
            .expect("degree-one overlap relation");
        assert!(matches!(
            overlap_relation.rhs_layout().row_families().unwrap()[0],
            RelationRowFamily::Consistency {
                opening_method: OpeningMethod::SubringCoefficientPacking { .. },
                ..
            }
        ));
        let v = crate::RingVec::<akita_field::Prime128OffsetA7F7>::from_coeffs(vec![
            Default::default();
            overlap.open_commit_matrix.output_rank() * 128
        ]);
        let u = crate::RingVec::<akita_field::Prime128OffsetA7F7>::from_coeffs(vec![
            Default::default();
            overlap.outer_commit_matrix.output_rank() * 128
        ]);
        assert!(crate::assemble_relation_rhs(overlap_relation.rhs_layout(), &v, &u).is_err());
    }

    #[test]
    fn relation_geometry_rejects_invalid_extension_or_subring_shapes() {
        let opening_batch = OpeningClaimsLayout::new(0, 1).expect("opening batch");
        let params = coefficient_packing_params(crate::CommitmentPayloadMode::Raw);
        assert!(RelationWitnessGeometry::for_level(&params, &opening_batch, 0).is_err());
        assert!(RelationWitnessGeometry::for_level(&params, &opening_batch, 3).is_err());

        let mut invalid = params;
        invalid.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 32,
        };
        assert!(RelationWitnessGeometry::for_level(&invalid, &opening_batch, 2).is_err());
        assert!(
            RelationWitnessGeometry::for_evaluation_trace_execution(&invalid, &opening_batch)
                .is_err()
        );
    }
}
