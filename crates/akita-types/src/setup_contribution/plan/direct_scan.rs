use super::*;

/// Challenge-dependent state of one direct setup scan.
///
/// A scan is built from one prepared [`SetupContributionPlan`] and one
/// coefficient functional. It owns the per-group E/T/Z column weights and the
/// packed D/B/A segment partition that the scan kernels walk. The plan itself
/// stays functional-free, so the deferred (closed-form) verifier path never
/// pays for, or can observe, a partially prepared scan.
pub struct DirectScan<E: Field> {
    pub(crate) mode: DirectScanMode<E>,
    pub(crate) partitions: Vec<GroupScanPartition<E>>,
}

pub(crate) enum DirectScanMode<E: Field> {
    Lifted {
        alpha: E,
        groups: Vec<DirectScanWeights<E>>,
    },
    Reduced {
        alpha: E,
        groups: Vec<ReducedDirectScanWeights<E>>,
    },
}

/// Packed D/B/A partition of one group's projected setup footprint.
pub(crate) struct GroupScanPartition<E> {
    /// Logical base-ring footprint of the group.
    pub(crate) required: usize,
    /// Scan jobs in increasing base-ring order.
    pub(crate) segments: Vec<GroupSetupSegment<E>>,
}

impl<E: Field> DirectScanMode<E> {
    pub(crate) fn weights(&self, group_index: usize) -> Option<&DirectScanWeights<E>> {
        match self {
            Self::Lifted { groups, .. } => groups.get(group_index),
            Self::Reduced { groups, .. } => groups.get(group_index).map(|group| &group.weights),
        }
    }

    fn group_count(&self) -> usize {
        match self {
            Self::Lifted { groups, .. } => groups.len(),
            Self::Reduced { groups, .. } => groups.len(),
        }
    }
}

impl<E: Field> DirectScan<E> {
    /// Materialize the column weights and packed segments for `functional`.
    ///
    /// # Errors
    ///
    /// Returns an error if the plan's tensors or projection geometry are
    /// malformed for the requested functional.
    pub fn new(
        plan: &SetupContributionPlan<E>,
        functional: PreparedCoefficientFunctional<E>,
    ) -> Result<Self, AkitaError> {
        let mode = match functional {
            PreparedCoefficientFunctional::LiftedPower { alpha } => {
                let groups = plan
                    .groups
                    .iter()
                    .map(|group| plan.materialize_lifted_direct_scan_weights(group, alpha))
                    .collect::<Result<Vec<_>, _>>()?;
                DirectScanMode::Lifted { alpha, groups }
            }
            PreparedCoefficientFunctional::ReducedEvaluation {
                alpha,
                coefficient_point,
            } => {
                let mut cache = Vec::new();
                let maximum_functionals =
                    checked::product([plan.groups.len(), 3]).ok_or_else(|| {
                        AkitaError::InvalidSetup("direct setup functional count overflow".into())
                    })?;
                cache.try_reserve_exact(maximum_functionals).map_err(|_| {
                    AkitaError::InvalidSetup("too many direct setup functionals".into())
                })?;
                let groups = plan
                    .groups
                    .iter()
                    .map(|group| {
                        plan.materialize_reduced_direct_scan_weights(
                            group,
                            alpha,
                            &coefficient_point,
                            &mut cache,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                DirectScanMode::Reduced { alpha, groups }
            }
        };
        Self::with_mode(plan, mode)
    }

    /// Partition `plan`'s packed setup for already materialized weights.
    pub(crate) fn with_mode(
        plan: &SetupContributionPlan<E>,
        mode: DirectScanMode<E>,
    ) -> Result<Self, AkitaError> {
        if mode.group_count() != plan.groups.len() {
            return Err(AkitaError::InvalidSetup(
                "direct setup scan group count disagrees with its plan".into(),
            ));
        }
        let _span = tracing::info_span!("setup_materialize_scan_segments").entered();
        let partitions = plan
            .groups
            .iter()
            .enumerate()
            .map(|(group_index, group)| {
                let weights = mode.weights(group_index).ok_or_else(|| {
                    AkitaError::InvalidSetup("direct setup group is missing".into())
                })?;
                group.scan_partition(
                    weights.e.len(),
                    &plan.d_weights,
                    plan.d_rows,
                    plan.d_physical_cols,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { mode, partitions })
    }

    /// Verify that this scan was partitioned for a plan with `plan`'s groups.
    pub(crate) fn check_plan(&self, plan: &SetupContributionPlan<E>) -> Result<(), AkitaError> {
        if self.partitions.len() != plan.groups.len()
            || self.mode.group_count() != plan.groups.len()
        {
            return Err(AkitaError::InvalidSetup(
                "direct setup scan was prepared for a different plan".into(),
            ));
        }
        Ok(())
    }

    /// Prepared D/B/A column equality slices for `group_id`.
    ///
    /// The D-role slice is laid out
    /// `(claim, block, opening_subcolumn, opening_digit)`, the B-role slice
    /// `(claim, block, A_row, outer_subcolumn, commit_digit)`, and the A-role
    /// slice `(position, witness_digit)` after contraction over units and fold
    /// digits. Subcolumn axes have length one for uniform roles.
    /// Tests compare these slices against independent address oracles.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn group_column_eq_slices(
        &self,
        plan: &SetupContributionPlan<E>,
        group_id: usize,
    ) -> Option<(&[E], &[E], &[E])> {
        let group_index = plan
            .groups
            .iter()
            .position(|group| group.group_id == group_id)?;
        self.mode
            .weights(group_index)
            .map(DirectScanWeights::slices)
    }
}

fn intern_reduced_functional<E: Field>(
    cache: &mut Vec<(usize, ReducedRoleCoefficientState<E>)>,
    dimension: usize,
    prepare: impl FnOnce() -> Result<ReducedRoleCoefficientState<E>, AkitaError>,
) -> Result<ReducedRoleCoefficientState<E>, AkitaError> {
    if let Some((_, prepared)) = cache.iter().find(|(cached, _)| *cached == dimension) {
        return Ok(prepared.clone());
    }
    let candidate = prepare()?;
    cache.push((dimension, candidate.clone()));
    Ok(candidate)
}

impl<E: Field> SetupContributionPlan<E> {
    fn materialize_reduced_direct_scan_weights(
        &self,
        group: &SetupContributionGroupPlan<E>,
        alpha: E,
        coefficient_point: &[E],
        cache: &mut Vec<(usize, ReducedRoleCoefficientState<E>)>,
    ) -> Result<ReducedDirectScanWeights<E>, AkitaError> {
        let a_role = intern_reduced_functional(cache, group.role_dims.d_a(), || {
            self.prepare_reduced_role_coefficient_state(
                group.role_dims.d_a(),
                alpha,
                coefficient_point,
            )
        })?;
        let b_role = intern_reduced_functional(cache, group.role_dims.d_b(), || {
            self.prepare_reduced_role_coefficient_state(
                group.role_dims.d_b(),
                alpha,
                coefficient_point,
            )
        })?;
        let d_role = intern_reduced_functional(cache, group.role_dims.d_d(), || {
            self.prepare_reduced_role_coefficient_state(
                group.role_dims.d_d(),
                alpha,
                coefficient_point,
            )
        })?;
        let evaluate_e = || {
            self.materialize_reduced_role_tensor_weights(
                group.d_relation_ratio,
                group.role_dims.d_d(),
                &group.d_tensors,
                group.d_col_range.len(),
            )
        };
        let evaluate_t = || {
            self.materialize_reduced_role_tensor_weights(
                group.b_relation_ratio,
                group.role_dims.d_b(),
                &group.physical_b.relation_tensors,
                group.physical_b.logical_input_width(),
            )
        };
        let evaluate_z = || {
            self.materialize_reduced_role_tensor_weights(
                group.a_relation_ratio,
                group.role_dims.d_a(),
                &group.a_tensors,
                group.z_cols,
            )
        };
        let (e, t, z) = materialize_three_roles(
            group.d_col_range.len(),
            group.physical_b.logical_input_width(),
            group.z_cols,
            evaluate_e,
            evaluate_t,
            evaluate_z,
        )?;
        Ok(ReducedDirectScanWeights {
            weights: DirectScanWeights { e, t, z },
            roles: [a_role, b_role, d_role],
        })
    }

    fn materialize_lifted_direct_scan_weights(
        &self,
        group: &SetupContributionGroupPlan<E>,
        alpha: E,
    ) -> Result<DirectScanWeights<E>, AkitaError> {
        let evaluate_e = || {
            let _span = tracing::info_span!("setup_materialize_e_weights").entered();
            self.materialize_role_tensor_weights(
                group.d_relation_ratio,
                &group.d_tensors,
                group.d_col_range.len(),
                alpha,
            )
        };
        let evaluate_t = || {
            let _span = tracing::info_span!("setup_materialize_t_weights").entered();
            self.materialize_role_tensor_weights(
                group.b_relation_ratio,
                &group.physical_b.relation_tensors,
                group.physical_b.logical_input_width(),
                alpha,
            )
        };
        let evaluate_z = || {
            let _span = tracing::info_span!("setup_materialize_z_weights").entered();
            self.materialize_role_tensor_weights(
                group.a_relation_ratio,
                &group.a_tensors,
                group.z_cols,
                alpha,
            )
        };
        let (e, t, z) = materialize_three_roles(
            group.d_col_range.len(),
            group.physical_b.logical_input_width(),
            group.z_cols,
            evaluate_e,
            evaluate_t,
            evaluate_z,
        )?;
        Ok(DirectScanWeights { e, t, z })
    }
}

fn materialize_three_roles<A, B, C>(
    e_len: usize,
    t_len: usize,
    z_len: usize,
    evaluate_e: impl FnOnce() -> Result<A, AkitaError> + Send,
    evaluate_t: impl FnOnce() -> Result<B, AkitaError> + Send,
    evaluate_z: impl FnOnce() -> Result<C, AkitaError> + Send,
) -> Result<(A, B, C), AkitaError>
where
    A: Send,
    B: Send,
    C: Send,
{
    // Each tensor materializer owns its internal parallel threshold. Fork the
    // independent role preparations only once their largest output is large.
    const PARALLEL_THRESHOLD: usize = 1 << 14;
    if e_len.max(t_len).max(z_len) >= PARALLEL_THRESHOLD {
        let (e, (t, z)) = cfg_join!(evaluate_e, || cfg_join!(evaluate_t, evaluate_z));
        Ok((e?, t?, z?))
    } else {
        Ok((evaluate_e()?, evaluate_t()?, evaluate_z()?))
    }
}
