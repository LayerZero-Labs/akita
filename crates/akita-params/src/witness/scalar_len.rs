use akita_error::AkitaError;

use super::{witness_unit_lengths, RelationQuotientPlan, WitnessLayout, MAX_WITNESS_CHUNKS};
use crate::{CommittedGroupParams, OpeningClaimsLayout, RelationWitnessGeometry};

impl WitnessLayout {
    /// Compute the exact live length of a scalar witness without materializing
    /// its address ranges.
    ///
    /// This is the candidate-aware counterpart of [`Self::new`] for planner
    /// hot paths. The caller supplies the extension degree, not a prebuilt
    /// geometry: this function owns its one construction and validation of
    /// the candidate's relation layout.
    pub fn scalar_live_coeff_len(
        lp: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        extension_degree: usize,
        num_chunks: usize,
        quotient_plan: RelationQuotientPlan,
    ) -> Result<usize, AkitaError> {
        if opening_batch.num_groups() != 1 {
            return Err(AkitaError::InvalidSetup(
                "scalar witness sizing requires exactly one opening group".into(),
            ));
        }
        if lp.has_preceding_groups() {
            return Err(AkitaError::InvalidSetup(
                "scalar witness sizing does not accept precommitted groups".into(),
            ));
        }
        if num_chunks == 0 {
            return Err(AkitaError::InvalidSetup(
                "witness layout requires non-empty groups and chunks".into(),
            ));
        }
        if num_chunks > MAX_WITNESS_CHUNKS {
            return Err(AkitaError::InvalidSetup(
                "witness chunk count exceeds verifier cap".into(),
            ));
        }
        let relation_geometry =
            RelationWitnessGeometry::for_level(lp, opening_batch, extension_degree)?;
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
        for block_range in lp.witness_block_ranges(group_index, num_chunks)? {
            let (z_len, e_len, t_len) = witness_unit_lengths(
                &params,
                role_dims,
                opening_geometry,
                num_claims,
                block_range.len(),
            )?;
            cursor = cursor
                .checked_add(z_len)
                .and_then(|n| n.checked_add(e_len))
                .and_then(|n| n.checked_add(t_len))
                .and_then(|len| {
                    akita_error::checked::align_up(
                        len,
                        if num_chunks > 1 {
                            lp.successor_block_len.unwrap_or(1)
                        } else {
                            1
                        },
                    )
                })
                .ok_or_else(|| AkitaError::InvalidSetup("witness unit range overflow".into()))?;
        }

        let successor_a_alignment = relation_geometry
            .rhs_layout()
            .relation_coefficient_block_len()?;
        super::tail::measure(
            lp,
            &relation_geometry,
            1,
            successor_a_alignment,
            cursor,
            quotient_plan,
        )
    }
}
