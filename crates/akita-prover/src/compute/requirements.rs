//! Declarative NTT requirements for one resolved prover execution.

use akita_field::AkitaError;
use akita_types::{
    CommittedGroupParams, FoldSchedule, NttCacheKey, NttTransformDomain, PrecommittedLevelParams,
    SetupPrefixSlotId, TerminalCommittedGroupParams,
};

/// Compute cluster that owns one public-matrix transform request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NttOperationCluster {
    /// Root, recursive, terminal, or setup-prefix commitments.
    Commit,
    /// Opening kernels that consume public-matrix rows.
    Opening,
    /// Tensor kernels that consume public-matrix rows.
    Tensor,
    /// Ring-switch relation and quotient construction.
    RingSwitch,
}

/// One exact cache request routed to a fold-level operation cluster.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedNttRequirement {
    /// Fold level whose compute stack owns this work.
    pub fold_level: usize,
    /// Operation cluster within that stack.
    pub cluster: NttOperationCluster,
    /// Exact transform prefix used when this operation is retained.
    pub key: NttCacheKey,
    /// Full operation extent used by the backend's cached-versus-streamed route.
    ///
    /// The production relation flow invokes A, B, and opening/D work as
    /// separate single-role operations. The A operation emits both transform
    /// domains with one shared extent; each B or D operation emits its own
    /// cyclic request. This keeps prewarm routing identical to runtime routing
    /// without joining independent operations.
    pub routing_extent: usize,
}

/// Canonical NTT requirement plan for one resolved schedule and call layout.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NttExecutionRequirements {
    entries: Vec<RoutedNttRequirement>,
}

impl NttExecutionRequirements {
    /// Compile matrix work performed by one resolved prover execution.
    ///
    /// The root commitment is completed before `batched_prove` and remains
    /// excluded. Setup-prefix commitments are part of the execution plan:
    /// their slots are prepared before the recursive fold consumes them, so
    /// their commit-cluster requirements must be included here.
    pub fn from_prove_schedule(schedule: &FoldSchedule) -> Result<Self, AkitaError> {
        schedule.validate_structure()?;
        let mut requirements = Self::default();
        let root = &schedule.root.params;
        requirements.add_group_relation(0, &root.final_group.commitment)?;
        for precommitted in &root.precommitted_groups {
            requirements.add_precommitted_relation(0, &precommitted.commitment)?;
        }
        let root_open_extent = matrix_extent(
            root.open_commit_matrix.output_rank(),
            root.open_commit_matrix.input_width(),
        )?;
        requirements.add_matrix(
            0,
            NttOperationCluster::RingSwitch,
            NttCacheKey::from_matrix_shape(
                root.open_commit_matrix.ring_dimension(),
                root.open_commit_matrix.output_rank(),
                root.open_commit_matrix.input_width(),
                NttTransformDomain::Negacyclic,
            )?,
            root_open_extent,
        )?;
        requirements.add_matrix(
            0,
            NttOperationCluster::RingSwitch,
            NttCacheKey::from_matrix_shape(
                root.open_commit_matrix.ring_dimension(),
                root.open_commit_matrix.output_rank(),
                root.open_commit_matrix.input_width(),
                NttTransformDomain::Cyclic,
            )?,
            root_open_extent,
        )?;

        for (index, step) in schedule.recursive_folds.iter().enumerate() {
            let predecessor_level = index;
            let level = index + 1;
            requirements.add_group_commit(predecessor_level, &step.params.witness)?;
            requirements.add_group_relation(level, &step.params.witness)?;
            if let Some(prefix) = &step.params.incoming_setup_prefix {
                requirements.add_setup_prefix_commitment(level, prefix)?;
                requirements.add_precommitted_relation(level, &prefix.commitment_params)?;
            }
            let open_extent = matrix_extent(
                step.params.open_commit_matrix.output_rank(),
                step.params.open_commit_matrix.input_width(),
            )?;
            requirements.add_matrix(
                level,
                NttOperationCluster::RingSwitch,
                NttCacheKey::from_matrix_shape(
                    step.params.open_commit_matrix.ring_dimension(),
                    step.params.open_commit_matrix.output_rank(),
                    step.params.open_commit_matrix.input_width(),
                    NttTransformDomain::Negacyclic,
                )?,
                open_extent,
            )?;
            requirements.add_matrix(
                level,
                NttOperationCluster::RingSwitch,
                NttCacheKey::from_matrix_shape(
                    step.params.open_commit_matrix.ring_dimension(),
                    step.params.open_commit_matrix.output_rank(),
                    step.params.open_commit_matrix.input_width(),
                    NttTransformDomain::Cyclic,
                )?,
                open_extent,
            )?;
        }

        requirements.add_terminal(
            schedule.recursive_folds.len(),
            &schedule.terminal.params.witness,
        )?;
        Ok(requirements)
    }

    /// Exact operation requests in deterministic routing order.
    pub fn entries(&self) -> &[RoutedNttRequirement] {
        &self.entries
    }

    /// Add the A/B work needed to materialize one setup-prefix commitment slot.
    pub fn add_setup_prefix_commitment(
        &mut self,
        fold_level: usize,
        slot: &SetupPrefixSlotId,
    ) -> Result<(), AkitaError> {
        let params = &slot.commitment_params.layout;
        self.add_matrix(
            fold_level,
            NttOperationCluster::Commit,
            NttCacheKey::from_matrix_shape(
                params.inner_commit_matrix.ring_dimension(),
                params.inner_commit_matrix.output_rank(),
                params.inner_commit_matrix.input_width(),
                NttTransformDomain::Negacyclic,
            )?,
            matrix_extent(
                params.inner_commit_matrix.output_rank(),
                params.inner_commit_matrix.input_width(),
            )?,
        )?;
        self.add_matrix(
            fold_level,
            NttOperationCluster::Commit,
            NttCacheKey::from_matrix_shape(
                params.outer_commit_matrix.ring_dimension(),
                params.outer_commit_matrix.output_rank(),
                params.outer_commit_matrix.input_width(),
                NttTransformDomain::Negacyclic,
            )?,
            matrix_extent(
                params.outer_commit_matrix.output_rank(),
                params.outer_commit_matrix.input_width(),
            )?,
        )
    }

    /// Add one exact matrix request with its operation-level routing extent.
    pub fn add_matrix(
        &mut self,
        fold_level: usize,
        cluster: NttOperationCluster,
        key: NttCacheKey,
        routing_extent: usize,
    ) -> Result<(), AkitaError> {
        if routing_extent < key.num_ring_elements {
            return Err(AkitaError::InvalidSetup(
                "NTT routing extent is smaller than its cache prefix".into(),
            ));
        }
        self.entries.push(RoutedNttRequirement {
            fold_level,
            cluster,
            key,
            routing_extent,
        });
        self.entries.sort_by_key(|entry| {
            (
                entry.fold_level,
                cluster_order(entry.cluster),
                entry.key.ring_d,
                domain_order(entry.key.domain),
                entry.routing_extent,
                std::cmp::Reverse(entry.key.num_ring_elements),
            )
        });
        Ok(())
    }

    fn add_group_commit(
        &mut self,
        level: usize,
        params: &CommittedGroupParams,
    ) -> Result<(), AkitaError> {
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            NttCacheKey::from_matrix_shape(
                params.inner_commit_matrix.ring_dimension(),
                params.inner_commit_matrix.output_rank(),
                params.inner_commit_matrix.input_width(),
                NttTransformDomain::Negacyclic,
            )?,
            matrix_extent(
                params.inner_commit_matrix.output_rank(),
                params.inner_commit_matrix.input_width(),
            )?,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            NttCacheKey::from_matrix_shape(
                params.outer_commit_matrix.ring_dimension(),
                params.outer_commit_matrix.output_rank(),
                params.outer_commit_matrix.input_width(),
                NttTransformDomain::Negacyclic,
            )?,
            matrix_extent(
                params.outer_commit_matrix.output_rank(),
                params.outer_commit_matrix.input_width(),
            )?,
        )
    }

    fn add_group_relation(
        &mut self,
        level: usize,
        params: &CommittedGroupParams,
    ) -> Result<(), AkitaError> {
        self.add_relation_ab(
            level,
            params.inner_commit_matrix.ring_dimension(),
            params.inner_commit_matrix.output_rank(),
            params.inner_commit_matrix.input_width(),
            params.outer_commit_matrix.ring_dimension(),
            params.outer_commit_matrix.output_rank(),
            params.outer_commit_matrix.input_width(),
        )?;
        for precommitted in &params.precommitted_groups {
            self.add_precommitted_relation(level, precommitted)?;
        }
        Ok(())
    }

    fn add_precommitted_relation(
        &mut self,
        level: usize,
        params: &PrecommittedLevelParams,
    ) -> Result<(), AkitaError> {
        self.add_relation_ab(
            level,
            params.layout.inner_commit_matrix.ring_dimension(),
            params.layout.inner_commit_matrix.output_rank(),
            params.layout.inner_commit_matrix.input_width(),
            params.layout.outer_commit_matrix.ring_dimension(),
            params.layout.outer_commit_matrix.output_rank(),
            params.layout.outer_commit_matrix.input_width(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn add_relation_ab(
        &mut self,
        level: usize,
        d_a: usize,
        n_a: usize,
        width_a: usize,
        d_b: usize,
        n_b: usize,
        width_b: usize,
    ) -> Result<(), AkitaError> {
        let a_extent = matrix_extent(n_a, width_a)?;
        for domain in [NttTransformDomain::Negacyclic, NttTransformDomain::Cyclic] {
            self.add_matrix(
                level,
                NttOperationCluster::RingSwitch,
                NttCacheKey::from_matrix_shape(d_a, n_a, width_a, domain)?,
                a_extent,
            )?;
        }
        self.add_matrix(
            level,
            NttOperationCluster::RingSwitch,
            NttCacheKey::from_matrix_shape(d_b, n_b, width_b, NttTransformDomain::Cyclic)?,
            matrix_extent(n_b, width_b)?,
        )
    }

    fn add_terminal(
        &mut self,
        level: usize,
        params: &TerminalCommittedGroupParams,
    ) -> Result<(), AkitaError> {
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            NttCacheKey::from_matrix_shape(
                params.inner_commit_matrix.ring_dimension(),
                params.inner_commit_matrix.output_rank(),
                params.inner_commit_matrix.input_width(),
                NttTransformDomain::Negacyclic,
            )?,
            matrix_extent(
                params.inner_commit_matrix.output_rank(),
                params.inner_commit_matrix.input_width(),
            )?,
        )
    }
}

fn matrix_extent(num_rows: usize, active_width: usize) -> Result<usize, AkitaError> {
    num_rows
        .checked_mul(active_width)
        .ok_or_else(|| AkitaError::InvalidSetup("NTT matrix extent overflow".into()))
}

const fn cluster_order(cluster: NttOperationCluster) -> u8 {
    match cluster {
        NttOperationCluster::Commit => 0,
        NttOperationCluster::Opening => 1,
        NttOperationCluster::Tensor => 2,
        NttOperationCluster::RingSwitch => 3,
    }
}

const fn domain_order(domain: NttTransformDomain) -> u8 {
    match domain {
        NttTransformDomain::Negacyclic => 0,
        NttTransformDomain::Cyclic => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "schedules-default")]
    use akita_config::proof_optimized::fp128;
    #[cfg(feature = "schedules-default")]
    use akita_config::CommitmentConfig;
    #[cfg(feature = "schedules-default")]
    use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

    #[test]
    fn equal_routing_coordinates_remain_exact_before_backend_routing() {
        let mut requirements = NttExecutionRequirements::default();
        for width in [7, 3, 11, 5] {
            requirements
                .add_matrix(
                    2,
                    NttOperationCluster::Commit,
                    NttCacheKey::from_matrix_shape(64, 3, width, NttTransformDomain::Negacyclic)
                        .unwrap(),
                    33,
                )
                .unwrap();
        }
        assert_eq!(requirements.entries.len(), 4);
        assert_eq!(requirements.entries[0].key.num_ring_elements, 33);
        assert_eq!(requirements.entries[1].key.num_ring_elements, 21);
        assert_eq!(requirements.entries[2].key.num_ring_elements, 15);
        assert_eq!(requirements.entries[3].key.num_ring_elements, 9);
    }

    #[test]
    fn distinct_operation_extents_are_not_joined_before_routing() {
        let mut requirements = NttExecutionRequirements::default();
        requirements
            .add_matrix(
                0,
                NttOperationCluster::RingSwitch,
                NttCacheKey::from_matrix_shape(64, 1, 5, NttTransformDomain::Cyclic).unwrap(),
                5,
            )
            .unwrap();
        requirements
            .add_matrix(
                0,
                NttOperationCluster::RingSwitch,
                NttCacheKey::from_matrix_shape(64, 1, 7, NttTransformDomain::Cyclic).unwrap(),
                11,
            )
            .unwrap();

        assert_eq!(requirements.entries.len(), 2);
        assert_eq!(requirements.entries[0].routing_extent, 5);
        assert_eq!(requirements.entries[1].routing_extent, 11);
    }

    #[test]
    fn relation_requirements_preserve_single_role_runtime_extents() {
        let mut requirements = NttExecutionRequirements::default();
        requirements
            .add_relation_ab(0, 64, 2, 3, 128, 5, 7)
            .unwrap();

        assert_eq!(requirements.entries.len(), 3);
        assert_eq!(requirements.entries[0].routing_extent, 6);
        assert_eq!(requirements.entries[1].routing_extent, 6);
        assert_eq!(requirements.entries[2].routing_extent, 35);
        assert_eq!(
            requirements.entries[0].key.domain,
            NttTransformDomain::Negacyclic
        );
        assert_eq!(
            requirements.entries[1].key.domain,
            NttTransformDomain::Cyclic
        );
        assert_eq!(
            requirements.entries[2].key.domain,
            NttTransformDomain::Cyclic
        );
    }

    #[test]
    fn domains_clusters_and_levels_remain_independent() {
        let mut requirements = NttExecutionRequirements::default();
        for (level, cluster, domain) in [
            (
                0,
                NttOperationCluster::Commit,
                NttTransformDomain::Negacyclic,
            ),
            (
                1,
                NttOperationCluster::Commit,
                NttTransformDomain::Negacyclic,
            ),
            (
                0,
                NttOperationCluster::RingSwitch,
                NttTransformDomain::Negacyclic,
            ),
            (
                0,
                NttOperationCluster::RingSwitch,
                NttTransformDomain::Cyclic,
            ),
        ] {
            requirements
                .add_matrix(
                    level,
                    cluster,
                    NttCacheKey::from_matrix_shape(64, 2, 9, domain).unwrap(),
                    18,
                )
                .unwrap();
        }
        assert_eq!(requirements.entries.len(), 4);
    }

    #[test]
    #[cfg(feature = "schedules-default")]
    fn generated_schedule_excludes_prior_root_commitment() {
        let schedule = fp128::OneHot::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(32),
        ))
        .expect("generated schedule");
        let requirements =
            NttExecutionRequirements::from_prove_schedule(&schedule).expect("compile requirements");
        let mut expected_root_level_commits = NttExecutionRequirements::default();
        if let Some(first_recursive) = schedule.recursive_folds.first() {
            expected_root_level_commits
                .add_group_commit(0, &first_recursive.params.witness)
                .expect("recursive witness requirements");
        } else {
            expected_root_level_commits
                .add_terminal(0, &schedule.terminal.params.witness)
                .expect("terminal witness requirements");
        }
        let actual_root_level_commits = requirements
            .entries()
            .iter()
            .filter(|entry| entry.fold_level == 0 && entry.cluster == NttOperationCluster::Commit)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            actual_root_level_commits, expected_root_level_commits.entries,
            "prove planning must not charge the already-completed root commitment"
        );

        assert!(!requirements.entries().is_empty());
        assert!(requirements.entries().iter().all(|entry| !matches!(
            entry.cluster,
            NttOperationCluster::Opening | NttOperationCluster::Tensor
        )));
        assert!(requirements.entries().iter().any(|entry| {
            entry.fold_level == 0
                && entry.cluster == NttOperationCluster::RingSwitch
                && entry.key.ring_d == schedule.root.params.open_commit_matrix.ring_dimension()
                && entry.key.domain == NttTransformDomain::Cyclic
        }));
        assert!(requirements.entries().iter().any(|entry| {
            entry.fold_level == schedule.recursive_folds.len()
                && entry.cluster == NttOperationCluster::Commit
                && entry.key.ring_d == schedule.terminal.params.witness.d_a()
                && entry.key.domain == NttTransformDomain::Negacyclic
        }));
    }
}
