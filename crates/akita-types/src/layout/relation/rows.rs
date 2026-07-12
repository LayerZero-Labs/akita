use super::*;
use akita_serialization::DEFAULT_MAX_SEQUENCE_LEN;
impl RelationRowPlan {
    /// Compile the checked base relation-row schedule from authenticated level context.
    pub fn compile_base(
        lp: &LevelParams,
        opening: &OpeningClaimsLayout,
        row_layout: RelationMatrixRowLayout,
    ) -> Result<Self, AkitaError> {
        lp.validate_root_opening_batch(opening)?;
        let mut group_order = Vec::with_capacity(opening.num_groups());
        group_order.push(RelationGroupId::Current);
        for index in 0..lp.precommitted_groups.len() {
            group_order.push(RelationGroupId::Precommitted { index });
        }

        let mut families = Vec::new();
        let mut row_cursor = 0usize;
        let mut push = |id, rows, native_ring_dim, inputs, rhs| -> Result<(), AkitaError> {
            if rows == 0 || native_ring_dim == 0 {
                return Err(AkitaError::InvalidSetup(
                    "relation row family has zero rows or ring dimension".into(),
                ));
            }
            let start = row_cursor;
            row_cursor = checked_add(row_cursor, rows, "row count")?;
            families.push(RelationRowFamily {
                id,
                rows: RowSpan { start, len: rows },
                native_ring_dim,
                quotient: RelationSegmentId::BaseQuotient { row: id },
                inputs,
                rhs,
            });
            Ok(())
        };
        let a_dim = lp.a_key.sis_table_key().ring_dimension as usize;
        push(
            RelationRowId::Consistency,
            1,
            a_dim,
            RelationRowInputs::Consistency {
                z: group_order
                    .iter()
                    .map(|&group| RelationSegmentId::Z { group })
                    .collect(),
                e: group_order
                    .iter()
                    .map(|&group| RelationSegmentId::E { group })
                    .collect(),
            },
            RelationRowRhs::Zero,
        )?;
        for &group in &group_order {
            let (a_rows, a_dim, b_rows, b_dim) = match group {
                RelationGroupId::Current => (
                    lp.a_key.row_len(),
                    lp.a_key.sis_table_key().ring_dimension as usize,
                    lp.b_key.row_len(),
                    lp.b_key.sis_table_key().ring_dimension as usize,
                ),
                RelationGroupId::Precommitted { index } => {
                    let pre = lp.precommitted_groups.get(index).ok_or_else(|| {
                        AkitaError::InvalidSetup("relation precommitted group is missing".into())
                    })?;
                    (
                        pre.a_key.row_len(),
                        pre.a_key.sis_table_key().ring_dimension as usize,
                        pre.b_key.row_len(),
                        pre.b_key.sis_table_key().ring_dimension as usize,
                    )
                }
            };
            push(
                RelationRowId::A { group },
                a_rows,
                a_dim,
                RelationRowInputs::A {
                    z: RelationSegmentId::Z { group },
                },
                RelationRowRhs::Zero,
            )?;
            push(
                RelationRowId::B { group },
                b_rows,
                b_dim,
                RelationRowInputs::B {
                    t: RelationSegmentId::T { group },
                    compression_input: None,
                },
                RelationRowRhs::Commitment { group },
            )?;
        }
        if row_layout == RelationMatrixRowLayout::WithDBlock {
            push(
                RelationRowId::D,
                lp.d_key.row_len(),
                lp.d_key.sis_table_key().ring_dimension as usize,
                RelationRowInputs::D {
                    e: group_order
                        .iter()
                        .map(|&group| RelationSegmentId::E { group })
                        .collect(),
                    compression_input: None,
                },
                RelationRowRhs::Opening,
            )?;
        }
        let trace_row = row_cursor;
        Ok(Self {
            group_order,
            families,
            trace_row,
            padded_row_count: checked_padded_row_count(trace_row)?,
        })
    }

    pub fn group_order(&self) -> &[RelationGroupId] {
        &self.group_order
    }

    pub fn families(&self) -> &[RelationRowFamily] {
        &self.families
    }

    /// Validate that this plan can be consumed by a fused, uniform-dimension kernel.
    ///
    /// Compression rows require per-family execution and are therefore rejected at
    /// this boundary even when their native dimension happens to equal `dimension`.
    pub fn validate_uniform_execution(&self, dimension: usize) -> Result<(), AkitaError> {
        for family in &self.families {
            if matches!(family.id, RelationRowId::Compression { .. }) {
                return Err(AkitaError::InvalidInput(
                    "compressed relation rows require per-family execution".into(),
                ));
            }
            if family.native_ring_dim != dimension {
                return Err(AkitaError::InvalidInput(format!(
                    "relation row family {:?} has native dimension {}, not uniform execution dimension {dimension}",
                    family.id, family.native_ring_dim
                )));
            }
        }
        Ok(())
    }

    pub fn trace_row(&self) -> usize {
        self.trace_row
    }

    pub fn padded_row_count(&self) -> usize {
        self.padded_row_count
    }

    pub fn family(&self, id: RelationRowId) -> Result<&RelationRowFamily, AkitaError> {
        self.families
            .iter()
            .find(|family| family.id == id)
            .ok_or_else(|| AkitaError::InvalidSetup("relation row family is absent".into()))
    }

    pub fn rhs_coeff_len(&self) -> Result<usize, AkitaError> {
        self.families.iter().try_fold(0usize, |total, family| {
            let len = family
                .rows
                .len
                .checked_mul(family.native_ring_dim)
                .ok_or_else(|| AkitaError::InvalidSetup("relation RHS length overflow".into()))?;
            let next = total
                .checked_add(len)
                .ok_or_else(|| AkitaError::InvalidSetup("relation RHS length overflow".into()))?;
            if next > DEFAULT_MAX_SEQUENCE_LEN {
                return Err(AkitaError::InvalidSetup(
                    "relation RHS length exceeds sequence cap".into(),
                ));
            }
            Ok(next)
        })
    }
}

pub(super) fn checked_add(current: usize, add: usize, label: &str) -> Result<usize, AkitaError> {
    let next = current
        .checked_add(add)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("logical {label} overflow")))?;
    if next > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(format!(
            "logical {label} {next} exceeds cap {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    Ok(next)
}

pub(super) fn checked_padded_row_count(trace_row: usize) -> Result<usize, AkitaError> {
    let padded = trace_row
        .checked_add(1)
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(|| AkitaError::InvalidSetup("relation padded row count overflow".into()))?;
    if padded > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(AkitaError::InvalidSetup(format!(
            "relation padded row count {padded} exceeds cap {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    Ok(padded)
}
