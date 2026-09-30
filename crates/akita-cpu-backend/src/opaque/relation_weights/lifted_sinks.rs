use super::*;

#[derive(Clone, Copy)]
enum LiftedEtSetup<'a, E: Field> {
    Matrix {
        d: &'a SetupColumnValues<E>,
        b: &'a SetupColumnValues<E>,
    },
    Deferred,
}

/// Disjoint mutable subslices of `values` at `ranges`, in the order of
/// `ranges`. An empty range has an empty subslice.
pub(super) fn split_disjoint_mut<'a, T>(
    values: &'a mut [T],
    ranges: &[Range<usize>],
) -> Result<Vec<&'a mut [T]>, AkitaError> {
    let mut order = (0..ranges.len())
        .filter(|&index| !ranges[index].is_empty())
        .collect::<Vec<_>>();
    order.sort_unstable_by_key(|&index| ranges[index].start);
    let mut windows = ranges.iter().map(|_| None).collect::<Vec<_>>();
    let mut rest = values;
    let mut consumed = 0;
    for index in order {
        let range = &ranges[index];
        let window = range
            .start
            .checked_sub(consumed)
            .and_then(|gap| std::mem::take(&mut rest).split_at_mut_checked(gap))
            .and_then(|(_, tail)| tail.split_at_mut_checked(range.len()))
            .map(|(window, tail)| {
                rest = tail;
                window
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation windows overlap or exceed their domain".into())
            })?;
        windows[index] = Some(window);
        consumed = range.end;
    }
    Ok(windows.into_iter().map(Option::unwrap_or_default).collect())
}

/// Add `scale * lane_values[j] + native * alpha^(j L)` to lane `j` of one
/// address, where `lane_alpha_powers[j] = alpha^(j L)`.
fn add_address_lanes<E: Field>(
    lanes: &mut [E],
    lane_values: Option<(&[E], E)>,
    native: E,
    lane_alpha_powers: &[E],
) -> Result<(), AkitaError> {
    if let Some((values, scale)) = lane_values {
        if values.len() != lanes.len() {
            return Err(AkitaError::InvalidProof);
        }
        for (lane, &value) in lanes.iter_mut().zip(values) {
            *lane += value * scale;
        }
    }
    let powers = lane_alpha_powers
        .get(1..lanes.len())
        .ok_or(AkitaError::InvalidProof)?;
    let (first, rest) = lanes.split_first_mut().ok_or(AkitaError::InvalidProof)?;
    *first += native;
    for (lane, &power) in rest.iter_mut().zip(powers) {
        *lane += native * power;
    }
    Ok(())
}

struct LiftedEtSink<'w, 'a, E: Field> {
    e_lanes: LaneWindow<'w, E>,
    t_lanes: LaneWindow<'w, E>,
    plan: &'a compiler::RelationWeightGroupPlan<E>,
    relation_coefficient_block_len: usize,
    /// `c_k(alpha) alpha^(m L)` for challenge `k` and every lane-aligned
    /// exponent `m L` of the relation alpha powers, challenge-major.
    challenge_lanes: &'a [E],
    lane_alpha_powers: &'a [E],
    setup: LiftedEtSetup<'a, E>,
}

impl<'a, E: Field> LiftedEtSink<'_, 'a, E> {
    /// Challenge lanes under subcolumn `role_subcolumn` of a `role_ring_dim`
    /// address.
    fn challenge_lanes(
        &self,
        challenge_index: usize,
        role_subcolumn: usize,
        role_ring_dim: usize,
    ) -> Result<&'a [E], AkitaError> {
        let width = self.lane_alpha_powers.len();
        let lanes = role_ring_dim / self.relation_coefficient_block_len;
        let offset = role_subcolumn
            .checked_mul(lanes)
            .filter(|offset| offset.checked_add(lanes).is_some_and(|end| end <= width))
            .ok_or(AkitaError::InvalidProof)?;
        let start = challenge_index
            .checked_mul(width)
            .and_then(|base| base.checked_add(offset))
            .ok_or(AkitaError::InvalidProof)?;
        self.challenge_lanes
            .get(start..start + lanes)
            .ok_or(AkitaError::InvalidProof)
    }
}

impl<E: Field> EtWeightSink<E> for LiftedEtSink<'_, '_, E> {
    fn add_e(
        &mut self,
        physical_start: usize,
        challenge_index: usize,
        role_subcolumn: usize,
        setup_column: usize,
        constraint_scale: E,
    ) -> Result<(), AkitaError> {
        let d_d = self.plan.roles.d_d;
        let setup = match self.setup {
            LiftedEtSetup::Matrix { d, .. } => d.get_scalar(0, setup_column)?,
            LiftedEtSetup::Deferred => E::zero(),
        };
        let challenge = match self.plan.opening_method {
            OpeningMethod::EvaluationTrace => Some((
                self.challenge_lanes(challenge_index, role_subcolumn, d_d)?,
                constraint_scale,
            )),
            OpeningMethod::SubringCoefficientPacking { .. } => None,
        };
        add_address_lanes(
            self.e_lanes.lanes_mut(physical_start, d_d)?,
            challenge,
            setup,
            self.lane_alpha_powers,
        )
    }

    fn add_t(
        &mut self,
        physical_start: usize,
        challenge_index: usize,
        role_subcolumn: usize,
        slice_index: usize,
        setup_column: usize,
        constraint_scale: E,
    ) -> Result<(), AkitaError> {
        let d_b = self.plan.roles.d_b;
        let setup = match self.setup {
            LiftedEtSetup::Matrix { b, .. } => b.get_scalar(slice_index, setup_column)?,
            LiftedEtSetup::Deferred => E::zero(),
        };
        let challenge = self.challenge_lanes(challenge_index, role_subcolumn, d_b)?;
        add_address_lanes(
            self.t_lanes.lanes_mut(physical_start, d_b)?,
            Some((challenge, constraint_scale)),
            setup,
            self.lane_alpha_powers,
        )
    }
}

#[derive(Clone, Copy)]
enum LiftedZSetup<'a, E: Field> {
    Matrix(&'a SetupColumnValues<E>),
    Deferred,
}

struct LiftedZSink<'w, 'a, E: Field> {
    lanes: LaneWindow<'w, E>,
    plan: &'a compiler::RelationWeightGroupPlan<E>,
    opening_evaluations: &'a [E],
    lane_alpha_powers: &'a [E],
    setup: LiftedZSetup<'a, E>,
}

impl<E: Field> ZWeightSink<E> for LiftedZSink<'_, '_, E> {
    fn add_z(
        &mut self,
        physical_start: usize,
        position: usize,
        setup_column: usize,
        constraint_scale: E,
        setup_scale: E,
    ) -> Result<(), AkitaError> {
        let mut native = match self.setup {
            LiftedZSetup::Matrix(setup) => setup.get_scalar(0, setup_column)? * setup_scale,
            LiftedZSetup::Deferred => E::zero(),
        };
        if matches!(self.plan.opening_method, OpeningMethod::EvaluationTrace) {
            native += self
                .opening_evaluations
                .get(position)
                .copied()
                .ok_or(AkitaError::InvalidProof)?
                * constraint_scale;
        }
        let lanes = self.lanes.lanes_mut(physical_start, self.plan.roles.d_a)?;
        add_address_lanes(lanes, None, native, self.lane_alpha_powers)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_et<E: Field>(
    group_plan: &compiler::RelationWeightGroupPlan<E>,
    witness_layout: &akita_types::WitnessLayout,
    lanes: &mut [E],
    relation_coefficient_block_len: usize,
    challenge_lanes: &[E],
    lane_alpha_powers: &[E],
    d_setup_accs: Option<&SetupColumnValues<E>>,
    b_setup_accs: Option<&SetupColumnValues<E>>,
) -> Result<(), AkitaError> {
    let setup = match (d_setup_accs, b_setup_accs) {
        (Some(d), Some(b)) => LiftedEtSetup::Matrix { d, b },
        (None, None) => LiftedEtSetup::Deferred,
        _ => {
            return Err(AkitaError::InvalidSetup(
                "lifted E/T setup phases disagree".into(),
            ));
        }
    };
    let tasks =
        group_plan.et_scatter_tasks(witness_layout, lanes, relation_coefficient_block_len)?;
    cfg_into_iter!(tasks).try_for_each(
        |schedule::EtScatterTask {
             range,
             e: e_lanes,
             t: t_lanes,
         }| {
            let mut sink = LiftedEtSink {
                e_lanes,
                t_lanes,
                plan: group_plan,
                relation_coefficient_block_len,
                challenge_lanes,
                lane_alpha_powers,
                setup,
            };
            compile_et_block_range(group_plan, &range, &mut sink)
        },
    )?;
    Ok(())
}

pub(super) fn scatter_z<E: Field>(
    group_plan: &compiler::RelationWeightGroupPlan<E>,
    witness_layout: &akita_types::WitnessLayout,
    lanes: &mut [E],
    relation_coefficient_block_len: usize,
    opening_evaluations: &[E],
    lane_alpha_powers: &[E],
    a_setup: Option<&SetupColumnValues<E>>,
) -> Result<(), AkitaError> {
    let setup = match a_setup {
        Some(values) => LiftedZSetup::Matrix(values),
        None => LiftedZSetup::Deferred,
    };
    let tasks =
        group_plan.z_scatter_tasks(witness_layout, lanes, relation_coefficient_block_len)?;
    cfg_into_iter!(tasks).try_for_each(|schedule::ZScatterTask { range, z: lanes }| {
        let mut sink = LiftedZSink {
            lanes,
            plan: group_plan,
            opening_evaluations,
            lane_alpha_powers,
            setup,
        };
        compile_z_position_range(group_plan, &range, &mut sink)
    })?;
    Ok(())
}
