use super::*;
use akita_types::{CommitmentSliceGeometry, RelationRangeImageGroupPlan, WitnessLayout};

pub(super) struct RelationWeightCompilationPlan<E> {
    pub(super) groups: Vec<RelationWeightGroupPlan<E>>,
    pub(super) d_row_weights: Vec<(usize, Vec<E>)>,
    pub(super) d_column_count: usize,
}

pub(super) struct RelationWeightGroupPlan<E> {
    pub(super) group_index: usize,
    pub(super) opening_method: OpeningMethod,
    pub(super) group_d_a: usize,
    pub(super) group_d_b: usize,
    pub(super) group_d_d: usize,
    pub(super) b_ratio: usize,
    pub(super) d_ratio: usize,
    pub(super) num_claims: usize,
    pub(super) num_live_blocks: usize,
    pub(super) num_positions: usize,
    pub(super) depth_witness: usize,
    pub(super) depth_commit: usize,
    pub(super) depth_open: usize,
    pub(super) depth_fold: usize,
    pub(super) n_a: usize,
    pub(super) inner_width: usize,
    pub(super) physical_n_b: usize,
    pub(super) b_width: usize,
    pub(super) slice_count: usize,
    pub(super) d_setup_range: std::ops::Range<usize>,
    pub(super) slice_geometry: CommitmentSliceGeometry,
    pub(super) consistency_weight: E,
    pub(super) a_row_weights: Vec<E>,
    pub(super) b_setup_row_weights: Vec<(usize, Vec<E>)>,
    pub(super) a_setup_row_weights: Vec<(usize, Vec<E>)>,
    pub(super) opening_gadget: Vec<E>,
    pub(super) commitment_gadget: Vec<E>,
    pub(super) witness_gadget: Vec<E>,
    pub(super) fold_gadget: Vec<E>,
}

struct RelationWeightCompilationInputs<'a, E> {
    lp: &'a CommittedGroupParams,
    opening_batch: &'a OpeningClaimsLayout,
    relation_geometry: &'a RelationWitnessGeometry,
    witness_layout: &'a WitnessLayout,
    row_families: &'a [RelationRowFamily],
    row_weights: &'a [E],
}

impl<E: FieldCore> RelationWeightCompilationPlan<E> {
    pub(super) fn new<F>(
        lp: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        relation_plan: &RelationRangeImagePlan,
        row_families: &[RelationRowFamily],
        row_weights: &[E],
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: LiftBase<F>,
    {
        let relation_geometry = relation_plan.relation_witness_geometry();
        let d_column_ranges = relation_d_column_ranges(lp, opening_batch, relation_geometry)?;
        let d_column_count = d_column_ranges
            .iter()
            .map(|range| range.end)
            .max()
            .unwrap_or(0);
        let d_start = row_families
            .iter()
            .position(|row| matches!(row, RelationRowFamily::Opening { .. }))
            .ok_or(AkitaError::InvalidProof)?;
        let n_d_active = lp.open().matrix.output_rank();
        let d_row_weights = (0..n_d_active)
            .filter_map(|row| {
                let weight = row_weights.get(d_start + row).copied();
                match weight {
                    Some(weight) if !weight.is_zero() => Some(Ok((row, vec![weight]))),
                    Some(_) => None,
                    None => Some(Err(AkitaError::InvalidProof)),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inputs = RelationWeightCompilationInputs {
            lp,
            opening_batch,
            relation_geometry,
            witness_layout: relation_plan.witness_layout(),
            row_families,
            row_weights,
        };
        let groups = relation_plan
            .groups()
            .iter()
            .map(|canonical_group| {
                let group_index = canonical_group.group_index();
                Self::build_group::<F>(
                    &inputs,
                    canonical_group,
                    d_column_ranges
                        .get(group_index)
                        .ok_or(AkitaError::InvalidProof)?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            groups,
            d_row_weights,
            d_column_count,
        })
    }

    fn build_group<F>(
        inputs: &RelationWeightCompilationInputs<'_, E>,
        canonical_group: &RelationRangeImageGroupPlan,
        d_columns: &std::ops::Range<usize>,
    ) -> Result<RelationWeightGroupPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: LiftBase<F>,
    {
        let RelationWeightCompilationInputs {
            lp,
            opening_batch,
            relation_geometry,
            witness_layout,
            row_families,
            row_weights,
        } = *inputs;
        let group_index = canonical_group.group_index();
        let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
        let group_dims = lp.group_role_dims_geometry(opening_batch, group_index)?;
        let group_d_a = group_dims.d_a();
        let group_d_b = group_dims.d_b();
        let group_d_d = group_dims.d_d();
        let (b_ratio, _) = SetupProjectionGeometry::native_role_subcolumn_counts(group_dims)?;
        let opening_width = relation_geometry
            .group_opening_geometry(group_index)?
            .physical_coefficient_width();
        let d_ratio = opening_width
            .checked_div(group_d_d)
            .filter(|count| *count > 0 && opening_width.is_multiple_of(group_d_d))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("opening width does not factor the D role".into())
            })?;
        let num_claims = opening_batch.group_layout(group_index)?.num_polynomials();
        if canonical_group.claim_range().len() != num_claims
            || canonical_group.unit_indices().iter().any(|&unit_index| {
                witness_layout
                    .units()
                    .get(unit_index)
                    .is_none_or(|unit| unit.group_index() != group_index)
            })
        {
            return Err(AkitaError::InvalidSetup(
                "canonical relation group disagrees with its witness layout".into(),
            ));
        }
        let num_live_blocks = group_lp.num_live_blocks();
        let depth_witness = group_lp.num_digits_inner();
        let depth_commit = group_lp.num_digits_outer();
        let depth_open = group_lp.num_digits_open();
        let depth_fold = group_lp.num_digits_fold();
        let n_a = group_lp.a_rows_len();
        let num_positions = group_lp.num_positions_per_block();
        let slice_geometry = CommitmentSliceGeometry::try_new(
            group_lp.outer_slice_count(),
            num_live_blocks,
            num_claims,
            n_a,
            depth_commit,
            group_d_a,
            group_d_b,
        )?;
        let physical_n_b = group_lp.b_rows_len();
        let b_width = slice_geometry.physical_input_width();
        let slice_count = group_lp.outer_slice_count().get();
        let a_range = matching_row_range(
            row_families,
            |family| matches!(family, RelationRowFamily::Inner { group_index: group, .. } if *group == group_index),
        )?;
        let b_range = matching_row_range(
            row_families,
            |family| matches!(family, RelationRowFamily::Outer { group_index: group, .. } if *group == group_index),
        )?;
        let expected_b_rows = group_lp.logical_b_rows_len()?;
        if a_range.end > row_weights.len()
            || b_range.end > row_weights.len()
            || b_range.len() != expected_b_rows
        {
            return Err(AkitaError::InvalidProof);
        }
        let consistency_row = relation_plan_consistency_row(row_families, group_index)?;
        let consistency_weight = *row_weights
            .get(consistency_row)
            .ok_or(AkitaError::InvalidProof)?;
        let lift_gadget = |depth, log_basis| {
            gadget_row_scalars::<F>(depth, log_basis)
                .into_iter()
                .map(E::lift_base)
                .collect::<Vec<_>>()
        };
        let b_row_weights = row_weights[b_range].to_vec();
        let b_setup_row_weights = (0..physical_n_b)
            .filter_map(|row| {
                let weights = (0..slice_count)
                    .map(|slice| {
                        let logical = slice_geometry.logical_row_index(slice, row, physical_n_b)?;
                        b_row_weights
                            .get(logical)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)
                    })
                    .collect::<Result<Vec<_>, _>>();
                match weights {
                    Ok(weights) if weights.iter().all(|weight| weight.is_zero()) => None,
                    other => Some(other.map(|weights| (row, weights))),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let a_row_weights = row_weights[a_range].to_vec();
        let a_setup_row_weights = a_row_weights
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(row, weight)| (!weight.is_zero()).then_some((row, vec![weight])))
            .collect();
        let d_setup_len = num_claims
            .checked_mul(num_live_blocks)
            .and_then(|len| len.checked_mul(d_ratio))
            .and_then(|len| len.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
        let d_setup_end = d_columns
            .start
            .checked_add(d_setup_len)
            .filter(|end| *end <= d_columns.end)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D extent overflow".into()))?;
        Ok(RelationWeightGroupPlan {
            group_index,
            opening_method: relation_geometry.group_opening_method(group_index)?,
            group_d_a,
            group_d_b,
            group_d_d,
            b_ratio,
            d_ratio,
            num_claims,
            num_live_blocks,
            num_positions,
            depth_witness,
            depth_commit,
            depth_open,
            depth_fold,
            n_a,
            inner_width: group_lp.a_col_len(),
            physical_n_b,
            b_width,
            slice_count,
            d_setup_range: d_columns.start..d_setup_end,
            slice_geometry,
            consistency_weight,
            a_row_weights,
            b_setup_row_weights,
            a_setup_row_weights,
            opening_gadget: lift_gadget(depth_open, group_lp.log_basis_open()),
            commitment_gadget: lift_gadget(depth_commit, group_lp.log_basis_outer()),
            witness_gadget: lift_gadget(depth_witness, group_lp.log_basis_inner()),
            fold_gadget: lift_gadget(depth_fold, group_lp.log_basis_open()),
        })
    }
}

pub(super) struct RelationWeightSetupSources<'a, F: FieldCore> {
    pub(super) d: SetupRows<'a, F>,
    groups: Vec<RelationWeightGroupSetupSources<'a, F>>,
}

pub(super) struct RelationWeightGroupSetupSources<'a, F: FieldCore> {
    group_index: usize,
    pub(super) a: SetupRows<'a, F>,
    pub(super) b: SetupRows<'a, F>,
}

impl<'a, F: FieldCore> RelationWeightSetupSources<'a, F> {
    pub(super) fn new<E: FieldCore>(
        setup: &'a AkitaExpandedSetup<F>,
        lp: &CommittedGroupParams,
        compilation: &RelationWeightCompilationPlan<E>,
    ) -> Result<Self, AkitaError> {
        let d_d = lp.role_dims().d_d();
        let n_d_active = lp.open().matrix.output_rank();
        let d_view =
            setup
                .shared_matrix()
                .ring_view_dyn(n_d_active, compilation.d_column_count, d_d)?;
        let d = SetupRows {
            rows: (0..n_d_active)
                .map(|row| d_view.row_flat(row))
                .collect::<Result<Vec<_>, _>>()?,
            ring_d: d_d,
        };
        let groups = compilation
            .groups
            .iter()
            .map(|group| {
                let a_view = setup.shared_matrix().ring_view_dyn(
                    group.n_a,
                    group.inner_width,
                    group.group_d_a,
                )?;
                let a = SetupRows {
                    rows: (0..group.n_a)
                        .map(|row| a_view.row_flat(row))
                        .collect::<Result<Vec<_>, _>>()?,
                    ring_d: group.group_d_a,
                };
                let b_view = setup.shared_matrix().ring_view_dyn(
                    group.physical_n_b,
                    group.b_width,
                    group.group_d_b,
                )?;
                let b = SetupRows {
                    rows: (0..group.physical_n_b)
                        .map(|row| b_view.row_flat(row))
                        .collect::<Result<Vec<_>, _>>()?,
                    ring_d: group.group_d_b,
                };
                Ok(RelationWeightGroupSetupSources {
                    group_index: group.group_index,
                    a,
                    b,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        Ok(Self { d, groups })
    }

    pub(super) fn group(
        &self,
        group_index: usize,
    ) -> Result<&RelationWeightGroupSetupSources<'a, F>, AkitaError> {
        self.groups
            .iter()
            .find(|group| group.group_index == group_index)
            .ok_or(AkitaError::InvalidProof)
    }
}

fn relation_plan_consistency_row(
    row_families: &[RelationRowFamily],
    group_index: usize,
) -> Result<usize, AkitaError> {
    row_families
        .iter()
        .position(|family| {
            matches!(family, RelationRowFamily::Consistency { group_index: group, .. } if *group == group_index)
        })
        .ok_or(AkitaError::InvalidProof)
}

pub(super) struct EAddress<E> {
    pub(super) physical_start: usize,
    pub(super) challenge_index: usize,
    pub(super) role_subcolumn: usize,
    pub(super) setup_column: usize,
    pub(super) constraint_scale: E,
}

pub(super) struct TAddress<E> {
    pub(super) physical_start: usize,
    pub(super) challenge_index: usize,
    pub(super) role_subcolumn: usize,
    pub(super) slice_index: usize,
    pub(super) setup_column: usize,
    pub(super) constraint_scale: E,
}

pub(super) struct ZAddress<E> {
    pub(super) physical_start: usize,
    pub(super) position: usize,
    pub(super) setup_column: usize,
    pub(super) constraint_scale: E,
    pub(super) setup_scale: E,
}

pub(super) trait EtWeightSink<E> {
    fn add_e(&mut self, address: EAddress<E>) -> Result<(), AkitaError>;
    fn add_t(&mut self, address: TAddress<E>) -> Result<(), AkitaError>;
}

pub(super) trait ZWeightSink<E> {
    fn add_z(&mut self, address: ZAddress<E>) -> Result<(), AkitaError>;
}

pub(super) fn compile_group_et_addresses<E: FieldCore>(
    plan: &RelationWeightGroupPlan<E>,
    witness_layout: &WitnessLayout,
    sink: &mut impl EtWeightSink<E>,
) -> Result<(), AkitaError> {
    for claim in 0..plan.num_claims {
        for block in 0..plan.num_live_blocks {
            let unit = witness_layout.unit_for_block(plan.group_index, block)?;
            let challenge_index = claim
                .checked_mul(plan.num_live_blocks)
                .and_then(|base| base.checked_add(block))
                .ok_or(AkitaError::InvalidProof)?;
            let (slice_index, slice_block) = plan.slice_geometry.block_coordinates(block)?;
            for (digit, &gadget) in plan.opening_gadget.iter().enumerate() {
                for role_subcolumn in 0..plan.d_ratio {
                    let physical_start = unit.e_coefficient_index(
                        plan.group_d_d,
                        plan.num_claims,
                        plan.depth_open,
                        claim,
                        block,
                        role_subcolumn,
                        digit,
                        0,
                    )?;
                    let logical_block = claim * plan.num_live_blocks + block;
                    let setup_column = logical_block
                        .checked_mul(plan.d_ratio)
                        .and_then(|base| base.checked_add(role_subcolumn))
                        .and_then(|base| base.checked_mul(plan.depth_open))
                        .and_then(|base| base.checked_add(digit))
                        .ok_or(AkitaError::InvalidProof)?;
                    sink.add_e(EAddress {
                        physical_start,
                        challenge_index,
                        role_subcolumn,
                        setup_column,
                        constraint_scale: plan.consistency_weight * gadget,
                    })?;
                }
            }
            for a_row in 0..plan.n_a {
                for (digit, &gadget) in plan.commitment_gadget.iter().enumerate() {
                    let block_claim = plan
                        .slice_geometry
                        .max_blocks_per_slice()
                        .checked_mul(claim)
                        .and_then(|base| base.checked_add(slice_block))
                        .ok_or(AkitaError::InvalidProof)?;
                    let row_block_claim = plan
                        .n_a
                        .checked_mul(block_claim)
                        .and_then(|base| base.checked_add(a_row))
                        .ok_or(AkitaError::InvalidProof)?;
                    for role_subcolumn in 0..plan.b_ratio {
                        let setup_column = row_block_claim
                            .checked_mul(plan.b_ratio)
                            .and_then(|base| base.checked_add(role_subcolumn))
                            .and_then(|base| base.checked_mul(plan.depth_commit))
                            .and_then(|base| base.checked_add(digit))
                            .ok_or(AkitaError::InvalidProof)?;
                        let physical_start = unit.t_coefficient_index(
                            plan.group_d_a,
                            plan.group_d_b,
                            plan.num_claims,
                            plan.n_a,
                            plan.depth_commit,
                            claim,
                            block,
                            a_row,
                            role_subcolumn,
                            digit,
                            0,
                        )?;
                        sink.add_t(TAddress {
                            physical_start,
                            challenge_index,
                            role_subcolumn,
                            slice_index,
                            setup_column,
                            constraint_scale: plan.a_row_weights[a_row] * gadget,
                        })?;
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn compile_group_z_addresses<E: FieldCore>(
    plan: &RelationWeightGroupPlan<E>,
    witness_layout: &WitnessLayout,
    sink: &mut impl ZWeightSink<E>,
) -> Result<(), AkitaError> {
    for unit in witness_layout.units_for_group(plan.group_index)? {
        for position in 0..plan.num_positions {
            for (witness_digit, &witness_scale) in plan.witness_gadget.iter().enumerate() {
                let setup_column = position
                    .checked_mul(plan.depth_witness)
                    .and_then(|base| base.checked_add(witness_digit))
                    .ok_or(AkitaError::InvalidProof)?;
                for (fold_digit, &fold_scale) in plan.fold_gadget.iter().enumerate() {
                    sink.add_z(ZAddress {
                        physical_start: unit.z_coefficient_index(
                            plan.group_d_a,
                            plan.num_positions,
                            plan.depth_witness,
                            plan.depth_fold,
                            position,
                            witness_digit,
                            fold_digit,
                            0,
                        )?,
                        position,
                        setup_column,
                        constraint_scale: -(plan.consistency_weight * witness_scale * fold_scale),
                        setup_scale: -fold_scale,
                    })?;
                }
            }
        }
    }
    Ok(())
}
