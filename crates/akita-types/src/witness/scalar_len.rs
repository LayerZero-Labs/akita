use akita_field::AkitaError;

use super::{
    checked_align_up, dyadic_block_ranges, witness_unit_lengths, WitnessLayout, MAX_WITNESS_CHUNKS,
};
use crate::{
    CommittedGroupParams, CompressionChainPlan, OpeningClaimsLayout, COMPRESSION_MAP_COUNT,
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
        lp.validate_opening_batch(opening_batch)?;
        let relation_group_order = opening_batch.root_group_order()?;
        let group_index = *relation_group_order.first().ok_or_else(|| {
            AkitaError::InvalidSetup("scalar witness relation group is missing".into())
        })?;
        let params = lp.group_params(opening_batch, group_index)?;
        let group = opening_batch.group_layout(group_index)?;
        let role_dims = lp.group_role_dims(opening_batch, group_index)?;
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
            let (z_len, e_len, t_len) =
                witness_unit_lengths(params, role_dims, num_claims, block_range.len())?;
            cursor = cursor
                .checked_add(z_len)
                .and_then(|n| n.checked_add(e_len))
                .and_then(|n| n.checked_add(t_len))
                .ok_or_else(|| AkitaError::InvalidSetup("witness unit range overflow".into()))?;
        }

        let a_rows = params
            .a_rows_len()
            .checked_add(1)
            .ok_or_else(|| AkitaError::InvalidSetup("relation A row count overflow".into()))?;
        for (rows, ring_dim) in [
            (a_rows, role_dims.d_a()),
            (params.b_rows_len(), role_dims.d_b()),
            (lp.open_commit_matrix.output_rank(), role_dims.d_d()),
        ] {
            let len = rows
                .checked_mul(quotient_depth)
                .and_then(|n| n.checked_mul(ring_dim))
                .ok_or_else(|| AkitaError::InvalidSetup("witness R width overflow".into()))?;
            cursor = cursor
                .checked_add(len)
                .ok_or_else(|| AkitaError::InvalidSetup("witness R range overflow".into()))?;
        }
        if !lp.payload_mode.is_compressed() {
            return Ok(Some(cursor));
        }

        let b_source_coefficients = params
            .b_rows_len()
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

        let relation_coefficient_block = role_dims.common_relation_coeff_count();
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
    use crate::SisModulusProfileId;

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
                        let materialized =
                            WitnessLayout::new(&params, &opening_batch, num_chunks, 2)
                                .expect("materialized witness layout")
                                .live_coeff_len();
                        let scalar = WitnessLayout::try_scalar_live_coeff_len(
                            &params,
                            &opening_batch,
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
}
