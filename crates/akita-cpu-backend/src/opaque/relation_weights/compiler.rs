use super::*;
use akita_types::{
    CommitmentSliceGeometry, RelationQuotientLayout, RelationRangeImageGroupPlan,
    RingMultiplierOpeningPoint, RingRelationGroupOpeningView, RingRelationMode, WitnessLayout,
};

pub(super) struct RelationWeightCompilation<'a, F: Field, E: Field> {
    pub(super) plan: RelationWeightCompilationPlan<E>,
    pub(super) setup_sources: Option<RelationWeightSetupSources<'a, F>>,
    pub(super) group_sources: Vec<RelationWeightGroupSources<'a, F>>,
    pub(super) witness_layout: WitnessLayout,
    pub(super) relation_geometry: RelationWitnessGeometry,
    pub(super) row_families: Vec<RelationRowFamily>,
    pub(super) row_weights: Vec<E>,
    pub(super) physical_field_len: usize,
    pub(super) relation_coefficient_block_len: usize,
}

pub(super) struct RelationWeightGroupSources<'a, F: Field> {
    pub(super) group_index: usize,
    pub(super) challenges: &'a akita_challenges::Challenges,
    pub(super) opening: OpeningFamily<&'a RingMultiplierOpeningPoint<F>, ()>,
}

impl<'a, F, E> RelationWeightCompilation<'a, F, E>
where
    F: Field + CanonicalEncoding,
    E: Field + ExtField<F>,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        setup: Option<&'a AkitaExpandedSetup<F>>,
        instance: &'a RingRelationInstance<F>,
        lp: &CommittedGroupParams,
        tau1: &[E],
        opening_source_len: usize,
        opening_ring_dim: usize,
        relation_plan: &RelationRangeImagePlan,
    ) -> Result<Self, AkitaError> {
        lp.witness_chunk.validate()?;
        if instance.role_dims() != lp.role_dims() {
            return Err(AkitaError::InvalidSetup(
                "relation instance and level role dimensions disagree".into(),
            ));
        }
        let opening_batch = instance.opening_batch();
        let relation_geometry =
            RelationWitnessGeometry::for_level(lp, opening_batch, instance.extension_degree())?;
        let row_families = relation_geometry.rhs_layout().row_families()?;
        if row_families.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        let witness_layout = instance.segment_layout(lp, None)?;
        match lp.ring_relation_mode {
            RingRelationMode::QuotientLift => {
                let levels = r_decomp_levels::<F>(lp.open().digits.log_basis);
                if witness_layout.r_rows().len() != row_families.len()
                    || witness_layout.quotient_depth() != Some(levels)
                    || witness_layout
                        .r_rows()
                        .iter()
                        .zip(&row_families)
                        .any(|(row, family)| row.geometry() != family.geometry())
                {
                    return Err(AkitaError::InvalidSetup(
                        "relation quotient layout disagrees with canonical row geometry".into(),
                    ));
                }
            }
            RingRelationMode::ReducedEvaluation => {
                if !matches!(
                    witness_layout.relation_quotient_layout(),
                    RelationQuotientLayout::ReducedEvaluation
                ) || !witness_layout.r_rows().is_empty()
                    || (0..opening_batch.num_groups()).any(|group| {
                        !matches!(
                            relation_geometry.group_opening_method(group),
                            Ok(OpeningMethod::EvaluationTrace)
                        )
                    })
                {
                    return Err(AkitaError::InvalidSetup(
                        "reduced relation requires a quotient-free evaluation-trace layout".into(),
                    ));
                }
            }
        }
        let physical_field_len = opening_source_len
            .checked_mul(opening_ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("opening field length overflow".into()))?;
        let domain = relation_plan.digit_witness_domain();
        if domain.domain_len() != physical_field_len
            || domain.live_len() != witness_layout.live_coeff_len()
            || relation_plan.witness_layout() != &witness_layout
            || relation_plan.relation_witness_geometry() != &relation_geometry
        {
            return Err(AkitaError::InvalidSetup(
                "relation plan disagrees with the current ring switch".into(),
            ));
        }
        let relation_coefficient_block_len = relation_plan
            .relation_address_geometry()
            .relation_coefficient_block_len();
        let expected_block_len = RelationAddressGeometry::for_relation(
            &relation_geometry,
            opening_ring_dim,
            witness_layout.live_coeff_len(),
        )?
        .relation_coefficient_block_len();
        if relation_coefficient_block_len != expected_block_len {
            return Err(AkitaError::InvalidSetup(
                "relation address geometry disagrees with the current ring switch".into(),
            ));
        }
        let eq_tau1 = SplitEqEvals::new(tau1)?;
        if eq_tau1.len() < row_families.len() {
            return Err(AkitaError::InvalidSize {
                expected: row_families.len(),
                actual: eq_tau1.len(),
            });
        }
        let row_weights = (0..row_families.len())
            .map(|row| eq_tau1.eval_at(row))
            .collect::<Result<Vec<_>, _>>()?;
        let plan = RelationWeightCompilationPlan::new::<F>(
            lp,
            opening_batch,
            relation_plan,
            &row_families,
            &row_weights,
        )?;
        let setup_sources = setup
            .map(|setup| RelationWeightSetupSources::new(setup, lp, &plan))
            .transpose()?;
        let group_sources = plan
            .groups
            .iter()
            .map(|group| {
                let (challenges, opening) = match instance.group_opening_view(group.group_index)? {
                    RingRelationGroupOpeningView::EvaluationTrace {
                        challenges,
                        ring_multiplier_point,
                    } if matches!(group.opening_method, OpeningMethod::EvaluationTrace) => (
                        challenges,
                        OpeningFamily::EvaluationTrace(ring_multiplier_point),
                    ),
                    RingRelationGroupOpeningView::SubringCoefficientPacking {
                        ambient_a_challenges,
                        ..
                    } if matches!(
                        group.opening_method,
                        OpeningMethod::SubringCoefficientPacking { .. }
                    ) =>
                    {
                        (
                            ambient_a_challenges,
                            OpeningFamily::SubringCoefficientPacking(()),
                        )
                    }
                    _ => {
                        return Err(AkitaError::InvalidSetup(
                            "relation opening source disagrees with the scheduled method".into(),
                        ));
                    }
                };
                let expected_challenges = group
                    .witness
                    .num_claims
                    .checked_mul(group.witness.num_live_blocks)
                    .ok_or(AkitaError::InvalidProof)?;
                if challenges.len() != expected_challenges {
                    return Err(AkitaError::InvalidSize {
                        expected: expected_challenges,
                        actual: challenges.len(),
                    });
                }
                if let OpeningFamily::EvaluationTrace(point) = opening {
                    if point.position_len() != group.witness.num_positions
                        || point.fold_len() != group.witness.num_live_blocks
                    {
                        return Err(AkitaError::InvalidProof);
                    }
                }
                Ok(RelationWeightGroupSources {
                    group_index: group.group_index,
                    challenges,
                    opening,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            plan,
            setup_sources,
            group_sources,
            witness_layout,
            relation_geometry,
            row_families,
            row_weights,
            physical_field_len,
            relation_coefficient_block_len,
        })
    }

    pub(super) fn group_source(
        &self,
        group_index: usize,
    ) -> Result<&RelationWeightGroupSources<'a, F>, AkitaError> {
        self.group_sources
            .iter()
            .find(|group| group.group_index == group_index)
            .ok_or(AkitaError::InvalidProof)
    }
}

pub(super) struct RelationWeightCompilationPlan<E> {
    pub(super) groups: Vec<RelationWeightGroupPlan<E>>,
    pub(super) d_row_weights: Vec<(usize, Vec<E>)>,
    pub(super) d_column_count: usize,
}

pub(super) struct RelationWeightGroupPlan<E> {
    pub(super) group_index: usize,
    pub(super) opening_method: OpeningMethod,
    pub(super) roles: RelationWeightRoleGeometry,
    pub(super) witness: RelationWeightWitnessGeometry,
    pub(super) rows: RelationWeightRowBatches<E>,
    pub(super) gadgets: RelationWeightGadgets<E>,
}

pub(super) struct RelationWeightRoleGeometry {
    pub(super) d_a: usize,
    pub(super) d_b: usize,
    pub(super) d_d: usize,
    pub(super) b_subcolumns: usize,
    pub(super) d_subcolumns: usize,
}

pub(super) struct RelationWeightWitnessGeometry {
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
    pub(super) slice_geometry: CommitmentSliceGeometry,
}

pub(super) struct RelationWeightRowBatches<E> {
    pub(super) d_setup_range: std::ops::Range<usize>,
    pub(super) consistency_weight: E,
    pub(super) a_row_weights: Vec<E>,
    pub(super) b_setup_row_weights: Vec<(usize, Vec<E>)>,
    pub(super) a_setup_row_weights: Vec<(usize, Vec<E>)>,
}

pub(super) struct RelationWeightGadgets<E> {
    pub(super) opening_gadget: Vec<E>,
    pub(super) commitment_gadget: Vec<E>,
    pub(super) witness_gadget: Vec<E>,
    pub(super) fold_gadget: Vec<E>,
}

struct RelationWeightCompilationInputs<'a, E> {
    lp: &'a CommittedGroupParams,
    opening_batch: &'a OpeningClaimsLayout,
    relation_plan: &'a RelationRangeImagePlan,
    relation_geometry: &'a RelationWitnessGeometry,
    witness_layout: &'a WitnessLayout,
    row_families: &'a [RelationRowFamily],
    row_weights: &'a [E],
}

impl<E: Field> RelationWeightCompilationPlan<E> {
    pub(super) fn new<F>(
        lp: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        relation_plan: &RelationRangeImagePlan,
        row_families: &[RelationRowFamily],
        row_weights: &[E],
    ) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
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
            relation_plan,
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
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let RelationWeightCompilationInputs {
            lp,
            opening_batch,
            relation_plan,
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
        let consistency_row = relation_plan.consistency_row_index(group_index)?;
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
            roles: RelationWeightRoleGeometry {
                d_a: group_d_a,
                d_b: group_d_b,
                d_d: group_d_d,
                b_subcolumns: b_ratio,
                d_subcolumns: d_ratio,
            },
            witness: RelationWeightWitnessGeometry {
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
                slice_geometry,
            },
            rows: RelationWeightRowBatches {
                d_setup_range: d_columns.start..d_setup_end,
                consistency_weight,
                a_row_weights,
                b_setup_row_weights,
                a_setup_row_weights,
            },
            gadgets: RelationWeightGadgets {
                opening_gadget: lift_gadget(depth_open, group_lp.log_basis_open()),
                commitment_gadget: lift_gadget(depth_commit, group_lp.log_basis_outer()),
                witness_gadget: lift_gadget(depth_witness, group_lp.log_basis_inner()),
                fold_gadget: lift_gadget(depth_fold, group_lp.log_basis_open()),
            },
        })
    }
}

pub(super) struct RelationWeightSetupSources<'a, F: Field> {
    pub(super) d: SetupRows<'a, F>,
    groups: Vec<RelationWeightGroupSetupSources<'a, F>>,
}

pub(super) struct RelationWeightGroupSetupSources<'a, F: Field> {
    group_index: usize,
    pub(super) a: SetupRows<'a, F>,
    pub(super) b: SetupRows<'a, F>,
}

impl<'a, F: Field> RelationWeightSetupSources<'a, F> {
    pub(super) fn new<E: Field>(
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
                    group.witness.n_a,
                    group.witness.inner_width,
                    group.roles.d_a,
                )?;
                let a = SetupRows {
                    rows: (0..group.witness.n_a)
                        .map(|row| a_view.row_flat(row))
                        .collect::<Result<Vec<_>, _>>()?,
                    ring_d: group.roles.d_a,
                };
                let b_view = setup.shared_matrix().ring_view_dyn(
                    group.witness.physical_n_b,
                    group.witness.b_width,
                    group.roles.d_b,
                )?;
                let b = SetupRows {
                    rows: (0..group.witness.physical_n_b)
                        .map(|row| b_view.row_flat(row))
                        .collect::<Result<Vec<_>, _>>()?,
                    ring_d: group.roles.d_b,
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

pub(super) fn compile_group_et_addresses<E: Field>(
    plan: &RelationWeightGroupPlan<E>,
    witness_layout: &WitnessLayout,
    sink: &mut impl EtWeightSink<E>,
) -> Result<(), AkitaError> {
    for claim in 0..plan.witness.num_claims {
        for block in 0..plan.witness.num_live_blocks {
            let unit = witness_layout.unit_for_block(plan.group_index, block)?;
            let challenge_index = claim
                .checked_mul(plan.witness.num_live_blocks)
                .and_then(|base| base.checked_add(block))
                .ok_or(AkitaError::InvalidProof)?;
            let (slice_index, slice_block) =
                plan.witness.slice_geometry.block_coordinates(block)?;
            for (digit, &gadget) in plan.gadgets.opening_gadget.iter().enumerate() {
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
                    let logical_block = claim * plan.witness.num_live_blocks + block;
                    let setup_column = logical_block
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
                        plan.rows.consistency_weight * gadget,
                    )?;
                }
            }
            for a_row in 0..plan.witness.n_a {
                for (digit, &gadget) in plan.gadgets.commitment_gadget.iter().enumerate() {
                    let block_claim = plan
                        .witness
                        .slice_geometry
                        .max_blocks_per_slice()
                        .checked_mul(claim)
                        .and_then(|base| base.checked_add(slice_block))
                        .ok_or(AkitaError::InvalidProof)?;
                    let row_block_claim = plan
                        .witness
                        .n_a
                        .checked_mul(block_claim)
                        .and_then(|base| base.checked_add(a_row))
                        .ok_or(AkitaError::InvalidProof)?;
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
                            plan.rows.a_row_weights[a_row] * gadget,
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn compile_group_z_addresses<E: Field>(
    plan: &RelationWeightGroupPlan<E>,
    witness_layout: &WitnessLayout,
    sink: &mut impl ZWeightSink<E>,
) -> Result<(), AkitaError> {
    for unit in witness_layout.units_for_group(plan.group_index)? {
        for position in 0..plan.witness.num_positions {
            for (witness_digit, &witness_scale) in plan.gadgets.witness_gadget.iter().enumerate() {
                let setup_column = position
                    .checked_mul(plan.witness.depth_witness)
                    .and_then(|base| base.checked_add(witness_digit))
                    .ok_or(AkitaError::InvalidProof)?;
                for (fold_digit, &fold_scale) in plan.gadgets.fold_gadget.iter().enumerate() {
                    sink.add_z(
                        unit.z_coefficient_index(
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
    }
    Ok(())
}
