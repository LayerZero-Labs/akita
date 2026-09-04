//! Declarative NTT requirements for one resolved prover execution.

use akita_error::AkitaError;
use akita_types::{
    centered_quotient_requires_i16_tail, CommittedGroupParams, FoldSchedule, GroupOpenPhaseParams,
    NttCacheKey, NttTransformDomain, RingRelationMode, SetupPrefixSlotId, SisModulusProfileId,
    TerminalFoldParams,
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
        requirements.add_group_commit(0, root, SignedCommitSource::Dense)?;
        for precommitted in root.precommitted_groups() {
            requirements.add_precommitted_commit(0, precommitted)?;
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
        let root_num_chunks = root.witness_chunk.num_chunks;
        requirements.add_group_relation(0, root, root_num_chunks)?;
        for precommitted in root.precommitted_groups() {
            requirements.add_precommitted_relation(0, precommitted, root_num_chunks)?;
        }
        requirements.add_opening_relation(0, root)?;

        for (index, step) in schedule.recursive_folds.iter().enumerate() {
            let predecessor_level = index;
            let level = index + 1;
            let num_chunks = step.params.witness_chunk.num_chunks;
            requirements.add_group_commit(
                predecessor_level,
                &step.params,
                SignedCommitSource::RecursiveWitness,
            )?;
            requirements.add_group_relation(level, &step.params, num_chunks)?;
            if let Some(prefix) = &step.params.setup_prefix() {
                requirements.add_setup_prefix_commitment(
                    level,
                    &prefix.slot_id().expect("setup prefix group"),
                )?;
                requirements.add_precommitted_relation(level, prefix, num_chunks)?;
            }
            requirements.add_opening_relation(level, &step.params)?;
        }

        requirements.add_terminal(schedule.recursive_folds.len(), &schedule.terminal)?;
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
        let params = &slot.commitment_profile;
        let inner_key = NttCacheKey::from_matrix_shape(
            params.inner.matrix.ring_dimension(),
            params.inner.matrix.output_rank(),
            params.inner.matrix.input_width(),
            signed_commit_domain(
                params.inner.matrix.sis_modulus_profile(),
                params.inner.matrix.ring_dimension(),
                params.inner.matrix.input_width(),
                params.inner.digits.log_basis,
                SignedCommitSource::Dense,
            )?,
        )?;
        self.add_matrix(
            fold_level,
            NttOperationCluster::Commit,
            inner_key,
            matrix_extent(
                params.inner.matrix.output_rank(),
                params.inner.matrix.input_width(),
            )?,
        )?;
        let outer_key = NttCacheKey::from_matrix_shape(
            params.outer.matrix.ring_dimension(),
            params.outer.matrix.output_rank(),
            params.outer.matrix.input_width(),
            NttTransformDomain::Negacyclic,
        )?;
        self.add_matrix(
            fold_level,
            NttOperationCluster::Commit,
            outer_key,
            matrix_extent(
                params.outer.matrix.output_rank(),
                params.outer.matrix.input_width(),
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
        source: SignedCommitSource,
    ) -> Result<(), AkitaError> {
        let inner_key = NttCacheKey::from_matrix_shape(
            params.inner().matrix.ring_dimension(),
            params.inner().matrix.output_rank(),
            params.inner().matrix.input_width(),
            signed_commit_domain(
                params.inner().matrix.sis_modulus_profile(),
                params.inner().matrix.ring_dimension(),
                params.inner().matrix.input_width(),
                params.inner().digits.log_basis,
                source,
            )?,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            inner_key,
            matrix_extent(
                params.inner().matrix.output_rank(),
                params.inner().matrix.input_width(),
            )?,
        )?;
        let outer_key = NttCacheKey::from_matrix_shape(
            params.outer().matrix.ring_dimension(),
            params.outer().matrix.output_rank(),
            params.outer().matrix.input_width(),
            NttTransformDomain::Negacyclic,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            outer_key,
            matrix_extent(
                params.outer().matrix.output_rank(),
                params.outer().matrix.input_width(),
            )?,
        )?;
        Ok(())
    }

    fn add_group_relation(
        &mut self,
        level: usize,
        params: &CommittedGroupParams,
        num_chunks: usize,
    ) -> Result<(), AkitaError> {
        match params.ring_relation_mode {
            RingRelationMode::QuotientLift => {
                self.add_relation_ab(
                    level,
                    params.inner().matrix.ring_dimension(),
                    params.inner().matrix.output_rank(),
                    params.inner().matrix.input_width(),
                    params.outer().matrix.ring_dimension(),
                    params.outer().matrix.output_rank(),
                    params.outer().matrix.input_width(),
                    params.open().digits.log_basis,
                    params.num_digits_fold(),
                    num_chunks,
                    params.inner().matrix.sis_modulus_profile(),
                )?;
                for precommitted in params.precommitted_groups() {
                    self.add_precommitted_relation(level, precommitted, num_chunks)?;
                }
            }
            RingRelationMode::ReducedEvaluation => {}
        }
        Ok(())
    }

    fn add_opening_relation(
        &mut self,
        level: usize,
        params: &CommittedGroupParams,
    ) -> Result<(), AkitaError> {
        let open = &params.open().matrix;
        let extent = matrix_extent(open.output_rank(), open.input_width())?;
        self.add_matrix(
            level,
            NttOperationCluster::RingSwitch,
            NttCacheKey::from_matrix_shape(
                open.ring_dimension(),
                open.output_rank(),
                open.input_width(),
                NttTransformDomain::Negacyclic,
            )?,
            extent,
        )?;
        match params.ring_relation_mode {
            RingRelationMode::QuotientLift => self.add_matrix(
                level,
                NttOperationCluster::RingSwitch,
                NttCacheKey::from_matrix_shape(
                    open.ring_dimension(),
                    open.output_rank(),
                    open.input_width(),
                    NttTransformDomain::Cyclic,
                )?,
                extent,
            ),
            RingRelationMode::ReducedEvaluation => Ok(()),
        }
    }

    fn add_precommitted_relation(
        &mut self,
        level: usize,
        params: &GroupOpenPhaseParams,
        num_chunks: usize,
    ) -> Result<(), AkitaError> {
        self.add_relation_ab(
            level,
            params.profile.inner.matrix.ring_dimension(),
            params.profile.inner.matrix.output_rank(),
            params.profile.inner.matrix.input_width(),
            params.profile.outer.matrix.ring_dimension(),
            params.profile.outer.matrix.output_rank(),
            params.profile.outer.matrix.input_width(),
            params.opening.log_basis_open,
            params.opening.num_digits_fold,
            num_chunks,
            params.profile.inner.matrix.sis_modulus_profile(),
        )
    }

    fn add_precommitted_commit(
        &mut self,
        level: usize,
        params: &GroupOpenPhaseParams,
    ) -> Result<(), AkitaError> {
        let layout = &params.profile;
        let inner_key = NttCacheKey::from_matrix_shape(
            layout.inner.matrix.ring_dimension(),
            layout.inner.matrix.output_rank(),
            layout.inner.matrix.input_width(),
            signed_commit_domain(
                layout.inner.matrix.sis_modulus_profile(),
                layout.inner.matrix.ring_dimension(),
                layout.inner.matrix.input_width(),
                layout.inner.digits.log_basis,
                SignedCommitSource::Dense,
            )?,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            inner_key,
            matrix_extent(
                layout.inner.matrix.output_rank(),
                layout.inner.matrix.input_width(),
            )?,
        )?;
        let outer_key = NttCacheKey::from_matrix_shape(
            layout.outer.matrix.ring_dimension(),
            layout.outer.matrix.output_rank(),
            layout.outer.matrix.input_width(),
            NttTransformDomain::Negacyclic,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            outer_key,
            matrix_extent(
                layout.outer.matrix.output_rank(),
                layout.outer.matrix.input_width(),
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
        num_chunks: usize,
        modulus_profile: SisModulusProfileId,
    ) -> Result<(), AkitaError> {
        if num_chunks == 0 {
            return Err(AkitaError::InvalidSetup(
                "ring-switch relation must retain at least one fold chunk".into(),
            ));
        }
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
        let (negative, positive) =
            akita_types::sis::balanced_digit_representable_bounds(log_basis_open, num_digits_fold);
        let rhs_abs_bound = negative
            .max(positive)
            .checked_mul(num_chunks as u128)
            .and_then(|bound| u64::try_from(bound).ok())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "aggregated folded-witness bound exceeds NTT capacity model".into(),
                )
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
        params: &TerminalFoldParams,
    ) -> Result<(), AkitaError> {
        let key = NttCacheKey::from_matrix_shape(
            params.inner.matrix.ring_dimension(),
            params.inner.matrix.output_rank(),
            params.inner.matrix.input_width(),
            signed_commit_domain(
                params.inner.matrix.sis_modulus_profile(),
                params.inner.matrix.ring_dimension(),
                params.inner.matrix.input_width(),
                params.inner.digits.log_basis,
                SignedCommitSource::RecursiveWitness,
            )?,
        )?;
        self.add_matrix(
            level,
            NttOperationCluster::Commit,
            key,
            matrix_extent(
                params.inner.matrix.output_rank(),
                params.inner.matrix.input_width(),
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SignedCommitSource {
    Dense,
    RecursiveWitness,
}

fn signed_commit_domain(
    modulus_profile: SisModulusProfileId,
    ring_dimension: usize,
    width: usize,
    log_basis: u32,
    source: SignedCommitSource,
) -> Result<NttTransformDomain, AkitaError> {
    let rhs_abs_bound = akita_types::balanced_signed_digit_abs_bound(log_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid signed digit basis".into()))?;
    match crate::validation::signed_digit_kernel_for_setup(log_basis, "for NTT cache planning")? {
        akita_types::SignedDigitKernel::I8
            if source == SignedCommitSource::RecursiveWitness
                || !akita_types::dense_i8_commit_prefers_exact_ifma52(
                    modulus_profile.modulus(),
                    ring_dimension,
                    width,
                    rhs_abs_bound,
                ) =>
        {
            Ok(NttTransformDomain::Negacyclic)
        }
        akita_types::SignedDigitKernel::I8 | akita_types::SignedDigitKernel::I16 => {
            Ok(NttTransformDomain::ExactNegacyclicI16 {
                width,
                rhs_abs_bound,
            })
        }
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
    fn recursive_signed_commit_domains_match_runtime_kernels() {
        let i8_domain = signed_commit_domain(
            SisModulusProfileId::Q128OffsetA7F7,
            512,
            19_456,
            7,
            SignedCommitSource::RecursiveWitness,
        )
        .expect("recursive i8 domain");
        assert_eq!(i8_domain, NttTransformDomain::Negacyclic);

        let i16_domain = signed_commit_domain(
            SisModulusProfileId::Q128OffsetA7F7,
            512,
            19_456,
            9,
            SignedCommitSource::RecursiveWitness,
        )
        .expect("recursive i16 domain");
        assert_eq!(
            i16_domain,
            NttTransformDomain::ExactNegacyclicI16 {
                width: 19_456,
                rhs_abs_bound: 256,
            }
        );
    }

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
    fn distributed_fold_aggregation_selects_the_q128_d64_tail() {
        let mut single = NttExecutionRequirements::default();
        single
            .add_relation_ab(
                1,
                64,
                6,
                4_096,
                64,
                1,
                1,
                4,
                11,
                1,
                SisModulusProfileId::Q128OffsetA7F7,
            )
            .unwrap();
        assert!(!single
            .entries()
            .iter()
            .any(|entry| { entry.key.domain == NttTransformDomain::I16TailBothTransforms }));

        let mut distributed = NttExecutionRequirements::default();
        distributed
            .add_relation_ab(
                1,
                64,
                6,
                4_096,
                64,
                1,
                1,
                4,
                11,
                8,
                SisModulusProfileId::Q128OffsetA7F7,
            )
            .unwrap();
        assert!(distributed.entries().iter().any(|entry| {
            entry.fold_level == 1
                && entry.cluster == NttOperationCluster::RingSwitch
                && entry.key
                    == NttCacheKey::from_matrix_shape(
                        64,
                        6,
                        4_096,
                        NttTransformDomain::I16TailBothTransforms,
                    )
                    .unwrap()
        }));
    }

    #[test]
    fn distributed_fold_bound_overflow_rejects() {
        let mut requirements = NttExecutionRequirements::default();
        assert!(matches!(
            requirements.add_relation_ab(
                0,
                64,
                1,
                1,
                64,
                1,
                1,
                4,
                4,
                usize::MAX,
                SisModulusProfileId::Q128OffsetA7F7,
            ),
            Err(AkitaError::InvalidSetup(_))
        ));
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
        let schedule = fp128::OneHot::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::new(32, 1),
        ))
        .expect("generated schedule")
        .into_schedule();
        let requirements =
            NttExecutionRequirements::from_prove_schedule(&schedule).expect("compile requirements");
        let mut expected_root_level_commits = NttExecutionRequirements::default();
        if let Some(first_recursive) = schedule.recursive_folds.first() {
            expected_root_level_commits
                .add_group_commit(
                    0,
                    &first_recursive.params,
                    SignedCommitSource::RecursiveWitness,
                )
                .expect("recursive witness requirements");
        } else {
            expected_root_level_commits
                .add_terminal(0, &schedule.terminal)
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
                && entry.key.ring_d == schedule.root.params.open().matrix.ring_dimension()
                && entry.key.domain == NttTransformDomain::Cyclic
        }));
        assert!(requirements.entries().iter().any(|entry| {
            entry.fold_level == schedule.recursive_folds.len()
                && entry.cluster == NttOperationCluster::Commit
                && entry.key.ring_d == schedule.terminal.d_a()
                && entry.key.domain == NttTransformDomain::Negacyclic
        }));
    }

    #[test]
    #[cfg(feature = "schedules-default")]
    fn complete_execution_includes_the_root_commitment() {
        let schedule = fp128::OneHot::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::new(32, 1),
        ))
        .expect("generated schedule")
        .into_schedule();
        let prove = NttExecutionRequirements::from_prove_schedule(&schedule).unwrap();
        let complete = NttExecutionRequirements::from_commit_and_prove_schedule(&schedule).unwrap();
        let root = &schedule.root.params;
        assert!(complete.entries().iter().any(|entry| {
            entry.fold_level == 0
                && entry.cluster == NttOperationCluster::Commit
                && entry.key.ring_d == root.inner().matrix.ring_dimension()
        }));
        assert!(complete.entries().len() >= prove.entries().len());
    }

    #[test]
    #[cfg(feature = "schedules-default")]
    fn reduced_relation_requirements_have_no_quotient_only_transforms() {
        let schedule = fp128::OneHot::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::new(32, 1),
        ))
        .expect("generated schedule")
        .into_schedule();
        let mut params = schedule.root.params.clone();
        params.ring_relation_mode = RingRelationMode::ReducedEvaluation;
        let mut requirements = NttExecutionRequirements::default();

        requirements
            .add_group_relation(2, &params, params.witness_chunk.num_chunks)
            .unwrap();
        requirements.add_opening_relation(2, &params).unwrap();

        assert_eq!(requirements.entries().len(), 1);
        assert!(requirements.entries().iter().all(|entry| {
            entry.fold_level == 2
                && entry.cluster == NttOperationCluster::RingSwitch
                && entry.key.domain == NttTransformDomain::Negacyclic
        }));
        assert_eq!(
            requirements.entries()[0].key.ring_d,
            params.open().matrix.ring_dimension()
        );
    }

    #[test]
    #[cfg(feature = "schedules-default")]
    fn fp128_dense_prewarms_the_selected_centered_quotient_profile() {
        let schedule = fp128::Dense::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(26),
        ))
        .expect("generated dense schedule")
        .into_schedule();
        let requirements =
            NttExecutionRequirements::from_prove_schedule(&schedule).expect("compile requirements");
        let root = &schedule.root.params;
        let (negative, positive) = akita_types::sis::balanced_digit_representable_bounds(
            root.open().digits.log_basis,
            root.num_digits_fold(),
        );
        let rhs_abs_bound = u64::try_from(
            negative
                .max(positive)
                .checked_mul(root.witness_chunk.num_chunks as u128)
                .expect("generated root bound"),
        )
        .expect("generated root bound fits u64");
        let expects_tail = centered_quotient_requires_i16_tail(
            root.inner().matrix.sis_modulus_profile(),
            root.role_dims().d_a(),
            rhs_abs_bound,
        )
        .expect("generated root tail decision");
        let has_tail = requirements.entries().iter().any(|entry| {
            entry.fold_level == 0
                && entry.cluster == NttOperationCluster::RingSwitch
                && entry.key.ring_d == root.role_dims().d_a()
                && entry.key.domain == NttTransformDomain::I16TailBothTransforms
        });
        assert_eq!(has_tail, expects_tail);
    }

    #[test]
    #[cfg(feature = "schedules-default")]
    fn fp128_dense_root_commit_prewarms_selected_i8_accumulation() {
        for num_vars in [26, 28, 30] {
            let schedule = fp128::Dense::resolve_catalog_row_for_key(
                &AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars)),
            )
            .expect("generated dense schedule")
            .into_schedule();
            let root = &schedule.root.params;
            let width = root.inner().matrix.input_width();
            let domain = signed_commit_domain(
                root.inner().matrix.sis_modulus_profile(),
                root.inner().matrix.ring_dimension(),
                width,
                root.inner().digits.log_basis,
                SignedCommitSource::Dense,
            )
            .expect("selected root commit domain");
            let expected = NttCacheKey::from_matrix_shape(
                root.inner().matrix.ring_dimension(),
                root.inner().matrix.output_rank(),
                width,
                domain,
            )
            .expect("root commit key");
            let requirements = NttExecutionRequirements::from_commit_and_prove_schedule(&schedule)
                .expect("compile complete NTT requirements");

            assert!(requirements.entries().iter().any(|entry| {
                entry.fold_level == 0
                    && entry.cluster == NttOperationCluster::Commit
                    && entry.key == expected
            }));
        }
    }

    #[test]
    #[cfg(feature = "schedules-default")]
    fn dense_small_field_nv26_cache_plan_matches_selected_geometry() {
        for schedule in [
            fp32::Dense::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
                PolynomialGroupLayout::singleton(26),
            ))
            .expect("generated fp32 dense schedule")
            .into_schedule(),
            fp64::Dense::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
                PolynomialGroupLayout::singleton(26),
            ))
            .expect("generated fp64 dense schedule")
            .into_schedule(),
        ] {
            let root = &schedule.root.params;
            assert!(matches!(
                root.opening_method(),
                akita_types::OpeningMethod::SubringCoefficientPacking { .. }
            ));
            assert_eq!(
                root.source_encoding,
                akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
            );

            let expected_commit_domain = signed_commit_domain(
                root.inner().matrix.sis_modulus_profile(),
                root.inner().matrix.ring_dimension(),
                root.inner().matrix.input_width(),
                root.inner().digits.log_basis,
                SignedCommitSource::Dense,
            )
            .expect("selected commit domain");
            let expected_commit_key = NttCacheKey::from_matrix_shape(
                root.inner().matrix.ring_dimension(),
                root.inner().matrix.output_rank(),
                root.inner().matrix.input_width(),
                expected_commit_domain,
            )
            .expect("selected root commit key");

            let requirements = NttExecutionRequirements::from_commit_and_prove_schedule(&schedule)
                .expect("compile complete small-field NTT requirements");
            assert!(requirements.entries().iter().any(|entry| {
                entry.fold_level == 0
                    && entry.cluster == NttOperationCluster::Commit
                    && entry.key == expected_commit_key
            }));
        }
    }
}
