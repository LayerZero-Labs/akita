//! Declarative NTT requirements for one resolved prover execution.

use akita_field::AkitaError;
use akita_types::{
    centered_quotient_requires_i16_tail, CommittedGroupParams, FoldSchedule, NttCacheKey,
    NttTransformDomain, PrecommittedLevelParams, SetupPrefixSlotId, SisModulusProfileId,
    TerminalCommittedGroupParams,
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
    /// Compile the complete root-commit plus prove call layout used by profile
    /// execution and other callers that own both phases.
    pub fn from_commit_and_prove_schedule(schedule: &FoldSchedule) -> Result<Self, AkitaError> {
        let mut requirements = Self::from_prove_schedule(schedule)?;
        let root = &schedule.root.params;
        requirements.add_group_commit(0, &root.final_group.commitment)?;
        for precommitted in &root.precommitted_groups {
            requirements.add_precommitted_commit(0, &precommitted.commitment)?;
        }
        Ok(requirements)
    }

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

    /// Max-joined requirements in deterministic routing order.
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
        let inner_key = NttCacheKey::from_matrix_shape(
            params.inner_commit_matrix.ring_dimension(),
            params.inner_commit_matrix.output_rank(),
            params.inner_commit_matrix.input_width(),
            signed_commit_domain(
                params.inner_commit_matrix.input_width(),
                params.log_basis_inner,
            )?,
        )?;
        self.add_matrix(
            fold_level,
            NttOperationCluster::Commit,
            inner_key,
            matrix_extent(
                params.inner_commit_matrix.output_rank(),
                params.inner_commit_matrix.input_width(),
            )?,
        )?;
        let outer_key = NttCacheKey::from_matrix_shape(
            params.outer_commit_matrix.ring_dimension(),
            params.outer_commit_matrix.output_rank(),
            params.outer_commit_matrix.input_width(),
            NttTransformDomain::Negacyclic,
        )?;
        self.add_matrix(
            fold_level,
            NttOperationCluster::Commit,
            outer_key,
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
        let inner_key = NttCacheKey::from_matrix_shape(
            params.inner_commit_matrix.ring_dimension(),
            params.inner_commit_matrix.output_rank(),
            params.inner_commit_matrix.input_width(),
            signed_commit_domain(
                params.inner_commit_matrix.input_width(),
                params.log_basis_inner,
            )?,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            inner_key,
            matrix_extent(
                params.inner_commit_matrix.output_rank(),
                params.inner_commit_matrix.input_width(),
            )?,
        )?;
        let outer_key = NttCacheKey::from_matrix_shape(
            params.outer_commit_matrix.ring_dimension(),
            params.outer_commit_matrix.output_rank(),
            params.outer_commit_matrix.input_width(),
            NttTransformDomain::Negacyclic,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            outer_key,
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
            params.log_basis_open,
            params.num_digits_fold,
            params.inner_commit_matrix.sis_modulus_profile(),
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
            params.log_basis_open,
            params.num_digits_fold,
            params.layout.inner_commit_matrix.sis_modulus_profile(),
        )
    }

    fn add_precommitted_commit(
        &mut self,
        level: usize,
        params: &PrecommittedLevelParams,
    ) -> Result<(), AkitaError> {
        let layout = &params.layout;
        let inner_key = NttCacheKey::from_matrix_shape(
            layout.inner_commit_matrix.ring_dimension(),
            layout.inner_commit_matrix.output_rank(),
            layout.inner_commit_matrix.input_width(),
            signed_commit_domain(
                layout.inner_commit_matrix.input_width(),
                layout.log_basis_inner,
            )?,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            inner_key,
            matrix_extent(
                layout.inner_commit_matrix.output_rank(),
                layout.inner_commit_matrix.input_width(),
            )?,
        )?;
        let outer_key = NttCacheKey::from_matrix_shape(
            layout.outer_commit_matrix.ring_dimension(),
            layout.outer_commit_matrix.output_rank(),
            layout.outer_commit_matrix.input_width(),
            NttTransformDomain::Negacyclic,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            outer_key,
            matrix_extent(
                layout.outer_commit_matrix.output_rank(),
                layout.outer_commit_matrix.input_width(),
            )?,
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
        log_basis_open: u32,
        num_digits_fold: usize,
        modulus_profile: SisModulusProfileId,
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
        )?;
        let (negative, positive) = akita_types::sis::fold_witness_representable_linf_bounds(
            log_basis_open,
            num_digits_fold,
        );
        let rhs_abs_bound = u64::try_from(negative.max(positive)).map_err(|_| {
            AkitaError::InvalidSetup("folded-witness bound exceeds NTT capacity model".into())
        })?;
        if centered_quotient_requires_i16_tail(modulus_profile, d_a, rhs_abs_bound)? {
            self.add_matrix(
                level,
                NttOperationCluster::RingSwitch,
                NttCacheKey::from_matrix_shape(
                    d_a,
                    n_a,
                    width_a,
                    NttTransformDomain::I16TailBothTransforms,
                )?,
                a_extent,
            )?;
        }
        Ok(())
    }

    fn add_terminal(
        &mut self,
        level: usize,
        params: &TerminalCommittedGroupParams,
    ) -> Result<(), AkitaError> {
        let key = NttCacheKey::from_matrix_shape(
            params.inner_commit_matrix.ring_dimension(),
            params.inner_commit_matrix.output_rank(),
            params.inner_commit_matrix.input_width(),
            signed_commit_domain(
                params.inner_commit_matrix.input_width(),
                params.log_basis_inner,
            )?,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            key,
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

/// Transform domain required to commit balanced digits at one basis.
fn signed_commit_domain(width: usize, log_basis: u32) -> Result<NttTransformDomain, AkitaError> {
    match crate::validation::signed_digit_kernel_for_setup(log_basis, "for NTT cache planning")? {
        akita_types::SignedDigitKernel::I8 => Ok(NttTransformDomain::Negacyclic),
        akita_types::SignedDigitKernel::I16 => Ok(NttTransformDomain::ExactNegacyclicI16 {
            width,
            rhs_abs_bound: akita_types::balanced_signed_digit_abs_bound(log_basis)
                .ok_or_else(|| AkitaError::InvalidSetup("invalid signed digit basis".into()))?,
        }),
    }
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
        NttTransformDomain::I16TailBothTransforms => 2,
        NttTransformDomain::ExactNegacyclicI16 { .. } => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "schedules-default")]
    use akita_config::proof_optimized::{fp128, fp32, fp64};
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
            .add_relation_ab(
                0,
                64,
                2,
                3,
                128,
                5,
                7,
                1,
                1,
                SisModulusProfileId::Q128OffsetA7F7,
            )
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
        let schedule = fp128::OneHot::select_schedule_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::unit_one_hot(32, 1, 256),
        ))
        .expect("generated schedule")
        .into_schedule();
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

    #[test]
    #[cfg(feature = "schedules-default")]
    fn complete_execution_includes_the_root_commitment() {
        let schedule = fp128::OneHot::select_schedule_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::unit_one_hot(32, 1, 256),
        ))
        .expect("generated schedule")
        .into_schedule();
        let prove = NttExecutionRequirements::from_prove_schedule(&schedule).unwrap();
        let complete = NttExecutionRequirements::from_commit_and_prove_schedule(&schedule).unwrap();
        let root = &schedule.root.params.final_group.commitment;
        assert!(complete.entries().iter().any(|entry| {
            entry.fold_level == 0
                && entry.cluster == NttOperationCluster::Commit
                && entry.key.ring_d == root.inner_commit_matrix.ring_dimension()
        }));
        assert!(complete.entries().len() >= prove.entries().len());
    }

    #[test]
    #[cfg(feature = "schedules-default")]
    fn fp128_dense_prewarms_centered_quotient_tail() {
        let schedule = fp128::Dense::select_schedule_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(26),
        ))
        .expect("generated dense schedule")
        .into_schedule();
        let requirements =
            NttExecutionRequirements::from_prove_schedule(&schedule).expect("compile requirements");
        assert!(requirements.entries().iter().any(|entry| {
            entry.fold_level == 0
                && entry.cluster == NttOperationCluster::RingSwitch
                && entry.key.ring_d
                    == schedule
                        .root
                        .params
                        .final_group
                        .commitment
                        .role_dims()
                        .d_a()
                && entry.key.domain == NttTransformDomain::I16TailBothTransforms
        }));
    }

    #[test]
    #[cfg(feature = "schedules-default")]
    fn dense_small_field_nv26_cache_plan_matches_adaptive_geometry() {
        for (
            schedule,
            expected_root_d,
            expected_root_basis,
            expected_root_cache_len,
            expects_i16_tail,
        ) in [
            (
                fp32::Dense::select_schedule_for_key(&AkitaScheduleLookupKey::single(
                    PolynomialGroupLayout::singleton(26),
                ))
                .expect("generated fp32 dense schedule")
                .into_schedule(),
                512,
                5,
                5376,
                false,
            ),
            (
                fp64::Dense::select_schedule_for_key(&AkitaScheduleLookupKey::single(
                    PolynomialGroupLayout::singleton(26),
                ))
                .expect("generated fp64 dense schedule")
                .into_schedule(),
                256,
                8,
                16_384,
                false,
            ),
        ] {
            let root = &schedule.root.params.final_group.commitment;
            assert_eq!(root.role_dims().d_a(), expected_root_d);
            assert_eq!(root.log_basis_inner, expected_root_basis);

            let requirements = NttExecutionRequirements::from_commit_and_prove_schedule(&schedule)
                .expect("compile complete small-field NTT requirements");
            let has_i16_tail = requirements.entries().iter().any(|entry| {
                matches!(
                    entry.key.domain,
                    NttTransformDomain::I16TailBothTransforms
                        | NttTransformDomain::ExactNegacyclicI16 { .. }
                )
            });
            assert_eq!(has_i16_tail, expects_i16_tail);
            assert!(requirements.entries().iter().any(|entry| {
                let expected_commit_domain = match crate::validation::signed_digit_kernel_for_setup(
                    expected_root_basis,
                    "for adaptive cache-plan test",
                )
                .expect("supported generated root basis")
                {
                    akita_types::SignedDigitKernel::I8 => {
                        entry.key.domain == NttTransformDomain::Negacyclic
                    }
                    akita_types::SignedDigitKernel::I16 => matches!(
                        entry.key.domain,
                        NttTransformDomain::ExactNegacyclicI16 { .. }
                    ),
                };
                entry.fold_level == 0
                    && entry.cluster == NttOperationCluster::Commit
                    && entry.key.ring_d == expected_root_d
                    && entry.key.num_ring_elements == expected_root_cache_len
                    && expected_commit_domain
            }));
        }
    }
}
