use super::compiler::RelationWeightGroupPlan;
use super::schedule::{EtBlockRange, ZPositionRange};
use akita_error::AkitaError;
use jolt_field::Field;

pub(super) trait EtWeightSink<E> {
    fn add_e(
        &mut self,
        physical_start: usize,
        challenge_index: usize,
        role_subcolumn: usize,
        setup_column: usize,
        constraint_scale: E,
    ) -> Result<(), AkitaError>;

    fn add_t(
        &mut self,
        physical_start: usize,
        challenge_index: usize,
        role_subcolumn: usize,
        slice_index: usize,
        setup_column: usize,
        constraint_scale: E,
    ) -> Result<(), AkitaError>;
}

pub(super) trait ZWeightSink<E> {
    fn add_z(
        &mut self,
        physical_start: usize,
        position: usize,
        setup_column: usize,
        constraint_scale: E,
        setup_scale: E,
    ) -> Result<(), AkitaError>;
}

pub(super) fn compile_et_block_range<E: Field>(
    plan: &RelationWeightGroupPlan<E>,
    range: &EtBlockRange<'_>,
    sink: &mut impl EtWeightSink<E>,
) -> Result<(), AkitaError> {
    let (unit, claim) = (range.unit, range.claim);
    for block in range.blocks.clone() {
        let challenge_index = claim
            .checked_mul(plan.witness.num_live_blocks)
            .and_then(|base| base.checked_add(block))
            .ok_or(AkitaError::InvalidProof)?;
        let (slice_index, slice_block) = plan.witness.slice_geometry.block_coordinates(block)?;
        for (digit, &gadget) in plan.gadgets.opening_gadget.iter().enumerate() {
            let constraint_scale = plan.rows.consistency_weight * gadget;
            for role_subcolumn in 0..plan.roles.d_subcolumns {
                let physical_start = unit.e_coefficient_index(
                    plan.roles.d_d,
                    plan.witness.num_claims,
                    plan.witness.depth_open,
                    claim,
                    block,
                    role_subcolumn,
                    digit,
                    0,
                )?;
                let setup_column = challenge_index
                    .checked_mul(plan.roles.d_subcolumns)
                    .and_then(|base| base.checked_add(role_subcolumn))
                    .and_then(|base| base.checked_mul(plan.witness.depth_open))
                    .and_then(|base| base.checked_add(digit))
                    .ok_or(AkitaError::InvalidProof)?;
                sink.add_e(
                    physical_start,
                    challenge_index,
                    role_subcolumn,
                    setup_column,
                    constraint_scale,
                )?;
            }
        }
        let block_claim = plan
            .witness
            .slice_geometry
            .max_blocks_per_slice()
            .checked_mul(claim)
            .and_then(|base| base.checked_add(slice_block))
            .ok_or(AkitaError::InvalidProof)?;
        for (a_row, &a_row_weight) in plan.rows.a_row_weights.iter().enumerate() {
            let row_block_claim = plan
                .witness
                .n_a
                .checked_mul(block_claim)
                .and_then(|base| base.checked_add(a_row))
                .ok_or(AkitaError::InvalidProof)?;
            for (digit, &gadget) in plan.gadgets.commitment_gadget.iter().enumerate() {
                let constraint_scale = a_row_weight * gadget;
                for role_subcolumn in 0..plan.roles.b_subcolumns {
                    let setup_column = row_block_claim
                        .checked_mul(plan.roles.b_subcolumns)
                        .and_then(|base| base.checked_add(role_subcolumn))
                        .and_then(|base| base.checked_mul(plan.witness.depth_commit))
                        .and_then(|base| base.checked_add(digit))
                        .ok_or(AkitaError::InvalidProof)?;
                    let physical_start = unit.t_coefficient_index(
                        plan.roles.d_a,
                        plan.roles.d_b,
                        plan.witness.num_claims,
                        plan.witness.n_a,
                        plan.witness.depth_commit,
                        claim,
                        block,
                        a_row,
                        role_subcolumn,
                        digit,
                        0,
                    )?;
                    sink.add_t(
                        physical_start,
                        challenge_index,
                        role_subcolumn,
                        slice_index,
                        setup_column,
                        constraint_scale,
                    )?;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn compile_z_position_range<E: Field>(
    plan: &RelationWeightGroupPlan<E>,
    range: &ZPositionRange<'_>,
    sink: &mut impl ZWeightSink<E>,
) -> Result<(), AkitaError> {
    for position in range.positions.clone() {
        for (witness_digit, &witness_scale) in plan.gadgets.witness_gadget.iter().enumerate() {
            let setup_column = position
                .checked_mul(plan.witness.depth_witness)
                .and_then(|base| base.checked_add(witness_digit))
                .ok_or(AkitaError::InvalidProof)?;
            for (fold_digit, &fold_scale) in plan.gadgets.fold_gadget.iter().enumerate() {
                sink.add_z(
                    range.unit.z_coefficient_index(
                        plan.roles.d_a,
                        plan.witness.num_positions,
                        plan.witness.depth_witness,
                        plan.witness.depth_fold,
                        position,
                        witness_digit,
                        fold_digit,
                        0,
                    )?,
                    position,
                    setup_column,
                    -(plan.rows.consistency_weight * witness_scale * fold_scale),
                    -fold_scale,
                )?;
            }
        }
    }
    Ok(())
}
