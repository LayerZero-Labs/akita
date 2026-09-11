use super::{
    CommitmentExecutionOutput, CommitmentExecutionPlan, CommitmentExecutor, CommitmentSource,
    CommitmentStateOutput, CommitmentStatePolicy,
};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};
use std::ops::Range;

/// Which inner/outer operation shape a round requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerOuterRouteKind {
    /// One operation performs A, decomposition, slicing, and B.
    Fused,
    /// Separate operations perform inner A and outer B.
    Split,
    /// Only inner A is executed, as in the terminal fold.
    InnerOnly,
}

#[derive(Clone)]
struct InnerOuterSelection {
    name: String,
    kind: InnerOuterRouteKind,
}

struct RegisteredExecutor<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    inner_outer: String,
    compression: String,
    executor: &'a CommitmentExecutor<'a, F, SP>,
}

/// One inspectable commitment route selected for a protocol round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentRoundStep {
    round: usize,
    inner_outer: String,
    inner_outer_kind: InnerOuterRouteKind,
    compression: String,
}

impl CommitmentRoundStep {
    pub const fn round(&self) -> usize {
        self.round
    }

    pub fn inner_outer(&self) -> &str {
        &self.inner_outer
    }

    pub const fn inner_outer_kind(&self) -> InnerOuterRouteKind {
        self.inner_outer_kind
    }

    pub fn compression(&self) -> &str {
        &self.compression
    }
}

struct CompiledRound<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    step: CommitmentRoundStep,
    executor: &'a CommitmentExecutor<'a, F, SP>,
}

/// Immutable, fully covered commitment routing plan.
///
/// The caller can inspect every selected round before arithmetic starts. The
/// plan never chooses another backend after a stage begins.
pub struct CommitmentExecutionSchedule<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    rounds: Vec<CompiledRound<'a, F, SP>>,
}

impl<'a, F, SP> CommitmentExecutionSchedule<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    fn validate_round_mode(
        &self,
        round: usize,
        plan: &CommitmentExecutionPlan,
    ) -> Result<(), AkitaError> {
        let kind = self
            .rounds
            .get(round)
            .ok_or_else(|| AkitaError::InvalidInput("commitment round is outside the plan".into()))?
            .step
            .inner_outer_kind;
        let matches = match plan.mode() {
            super::CommitmentExecutionMode::InnerOnly => kind == InnerOuterRouteKind::InnerOnly,
            super::CommitmentExecutionMode::Full | super::CommitmentExecutionMode::Uncompressed => {
                matches!(
                    kind,
                    InnerOuterRouteKind::Fused | InnerOuterRouteKind::Split
                )
            }
        };
        if !matches {
            return Err(AkitaError::InvalidInput(
                "commitment schedule route kind does not match the round execution mode".into(),
            ));
        }
        Ok(())
    }

    /// Ordered, read-only routing decisions for all rounds.
    pub fn steps(&self) -> impl ExactSizeIterator<Item = &CommitmentRoundStep> {
        self.rounds.iter().map(|round| &round.step)
    }

    /// Selected executor for one round.
    pub fn executor(&self, round: usize) -> Result<&'a CommitmentExecutor<'a, F, SP>, AkitaError> {
        self.rounds
            .get(round)
            .map(|entry| entry.executor)
            .ok_or_else(|| AkitaError::InvalidInput("commitment round is outside the plan".into()))
    }

    /// Validate a round's arithmetic plan and source admission without
    /// invoking a backend operation.
    pub fn preflight(
        &self,
        round: usize,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<(), AkitaError> {
        self.validate_round_mode(round, plan)?;
        self.executor(round)?.preflight_request(plan, sources)
    }

    /// Execute a full A/B/compression round through its preselected route.
    pub fn execute_full(
        &self,
        round: usize,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<CommitmentExecutionOutput<F, SP::State>, AkitaError> {
        self.validate_round_mode(round, plan)?;
        self.executor(round)?.execute_full(plan, sources)
    }

    /// Execute an uncompressed A/B round through its preselected route.
    pub fn execute_uncompressed(
        &self,
        round: usize,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<CommitmentExecutionOutput<F, SP::State>, AkitaError> {
        self.validate_round_mode(round, plan)?;
        self.executor(round)?.execute_uncompressed(plan, sources)
    }

    /// Execute an inner-only round through its preselected route.
    pub fn execute_inner(
        &self,
        round: usize,
        plan: &CommitmentExecutionPlan,
        sources: &[&dyn CommitmentSource<F>],
    ) -> Result<CommitmentStateOutput<SP::State>, AkitaError> {
        self.validate_round_mode(round, plan)?;
        self.executor(round)?.execute_inner(plan, sources)
    }
}

/// Builder for independent A/B and compression assignments.
///
/// Register one already prepared executor for each A/B-compression pair that
/// the selected ranges can produce. Compilation rejects incomplete ranges and
/// missing pairs before any backend call.
pub struct CommitmentExecutionScheduleBuilder<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    round_count: usize,
    inner_outer: Vec<Option<InnerOuterSelection>>,
    compression: Vec<Option<String>>,
    executors: Vec<RegisteredExecutor<'a, F, SP>>,
}

impl<'a, F, SP> CommitmentExecutionScheduleBuilder<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    pub fn new(round_count: usize) -> Result<Self, AkitaError> {
        if round_count == 0 {
            return Err(AkitaError::InvalidInput(
                "commitment execution schedule requires at least one round".into(),
            ));
        }
        Ok(Self {
            round_count,
            inner_outer: vec![None; round_count],
            compression: vec![None; round_count],
            executors: Vec::new(),
        })
    }

    /// Register the executor implementing one named A/B-compression pair.
    pub fn register_executor(
        &mut self,
        inner_outer: impl Into<String>,
        compression: impl Into<String>,
        executor: &'a CommitmentExecutor<'a, F, SP>,
    ) -> Result<&mut Self, AkitaError> {
        let inner_outer = inner_outer.into();
        let compression = compression.into();
        if inner_outer.is_empty() || compression.is_empty() {
            return Err(AkitaError::InvalidInput(
                "commitment route names must be nonempty".into(),
            ));
        }
        if self
            .executors
            .first()
            .is_some_and(|registered| registered.executor.setup != executor.setup)
        {
            return Err(AkitaError::InvalidSetup(
                "commitment schedule executors use different setup descriptors".into(),
            ));
        }
        if self.executors.iter().any(|registered| {
            registered.inner_outer == inner_outer && registered.compression == compression
        }) {
            return Err(AkitaError::InvalidInput(
                "commitment executor pair is already registered".into(),
            ));
        }
        self.executors.push(RegisteredExecutor {
            inner_outer,
            compression,
            executor,
        });
        Ok(self)
    }

    /// Assign one named A/B operation to a half-open range of rounds.
    pub fn inner_outer(
        &mut self,
        range: Range<usize>,
        name: impl Into<String>,
        kind: InnerOuterRouteKind,
    ) -> Result<&mut Self, AkitaError> {
        let name = name.into();
        Self::validate_range(self.round_count, &range, &name)?;
        if range.clone().any(|round| self.inner_outer[round].is_some()) {
            return Err(AkitaError::InvalidInput(
                "commitment A/B round is assigned more than once".into(),
            ));
        }
        for round in range {
            self.inner_outer[round] = Some(InnerOuterSelection {
                name: name.clone(),
                kind,
            });
        }
        Ok(self)
    }

    /// Assign one named compression operation to a half-open range of rounds.
    pub fn compression(
        &mut self,
        range: Range<usize>,
        name: impl Into<String>,
    ) -> Result<&mut Self, AkitaError> {
        let name = name.into();
        Self::validate_range(self.round_count, &range, &name)?;
        if range.clone().any(|round| self.compression[round].is_some()) {
            return Err(AkitaError::InvalidInput(
                "commitment compression round is assigned more than once".into(),
            ));
        }
        for round in range {
            self.compression[round] = Some(name.clone());
        }
        Ok(self)
    }

    fn validate_range(
        round_count: usize,
        range: &Range<usize>,
        name: &str,
    ) -> Result<(), AkitaError> {
        if name.is_empty() || range.start > range.end || range.end > round_count {
            return Err(AkitaError::InvalidInput(
                "commitment route has an invalid name or round range".into(),
            ));
        }
        Ok(())
    }

    /// Resolve every independently assigned pair into an immutable plan.
    pub fn compile(self) -> Result<CommitmentExecutionSchedule<'a, F, SP>, AkitaError> {
        let mut rounds = Vec::with_capacity(self.round_count);
        for round in 0..self.round_count {
            let inner_outer = self.inner_outer[round].as_ref().ok_or_else(|| {
                AkitaError::InvalidInput("commitment schedule has an unassigned A/B round".into())
            })?;
            let compression = self.compression[round].as_ref().ok_or_else(|| {
                AkitaError::InvalidInput(
                    "commitment schedule has an unassigned compression round".into(),
                )
            })?;
            let executor = self
                .executors
                .iter()
                .find(|registered| {
                    registered.inner_outer == inner_outer.name
                        && registered.compression == *compression
                })
                .ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "commitment schedule has no executor for a selected route pair".into(),
                    )
                })?
                .executor;
            if !executor.matches_inner_outer_kind(inner_outer.kind) {
                return Err(AkitaError::InvalidInput(
                    "registered executor does not implement the selected fused or split route"
                        .into(),
                ));
            }
            rounds.push(CompiledRound {
                step: CommitmentRoundStep {
                    round,
                    inner_outer: inner_outer.name.clone(),
                    inner_outer_kind: inner_outer.kind,
                    compression: compression.clone(),
                },
                executor,
            });
        }
        Ok(CommitmentExecutionSchedule { rounds })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::{
        BackendKindId, CommitmentExecutorBuilder, CommitmentRequestCapabilities,
        CommitmentStateBinding, CompressionOperationCapabilities, DenseType,
        FusedInnerOuterOperation, InnerImage, PolynomialType, PortableStatePolicy,
        PreparedCommitmentResources, PreparedCompression, PreparedFusedCommitment,
        PreparedInnerCommitment, ResolvedCommitSource, StageDimensionCapabilities, StageResources,
        StateOwnerCapability, UncompressedCommitPlan, UncompressedCommitmentOutput,
    };
    use crate::compute::{CpuBackend, CpuCompressionOperation, CpuInnerCommitOperation};
    use crate::AkitaProverSetup;
    use akita_types::{RingVec, SetupMatrixCapacity};
    use jolt_field::Prime64Offset59;
    use std::sync::Arc;

    type F = Prime64Offset59;

    struct ZeroFused {
        owner: StateOwnerCapability<InnerImage>,
    }

    impl FusedInnerOuterOperation<F> for ZeroFused {
        fn commit_inner_outer(
            &self,
            binding: &CommitmentStateBinding,
            plan: &UncompressedCommitPlan,
            _sources: &[ResolvedCommitSource<'_, F>],
        ) -> Result<UncompressedCommitmentOutput<F>, AkitaError> {
            let image = self.owner.bind(binding.clone(), 0, ());
            let u = RingVec::from_coeffs_with_ring_dim(
                vec![F::default(); plan.outer().output_coefficient_len()?],
                plan.outer().ring_dimension(),
            )?;
            UncompressedCommitmentOutput::new(image, u, plan)
        }
    }

    fn fused_executor<'a>(
        backend: &'a CpuBackend,
        prepared: &'a crate::compute::CpuPreparedSetup<F>,
        expanded: &'a akita_types::AkitaExpandedSetup<F>,
    ) -> CommitmentExecutor<'a, F, PortableStatePolicy> {
        struct FusedContext;
        struct FusedCommand;
        struct FusedBackend;

        let mut builder = CommitmentExecutorBuilder::new(expanded, PortableStatePolicy);
        let instance = builder.issue_backend_instance();
        let resources = StageResources::controlled(
            PreparedCommitmentResources::new(backend, prepared, expanded).unwrap(),
        );
        let fused_context = builder
            .operation_context(instance, "zero-fused", resources.clone())
            .unwrap();
        let inner_context = builder
            .operation_context(instance, "cpu-inner", resources.clone())
            .unwrap();
        let compression_context = builder
            .operation_context(instance, "cpu-compression", resources)
            .unwrap();
        let fused_owner = StateOwnerCapability::new();
        builder
            .register_fused(
                PreparedFusedCommitment::new(
                    Arc::new(ZeroFused {
                        owner: fused_owner.clone(),
                    }),
                    fused_owner,
                    fused_context,
                    CommitmentRequestCapabilities::fused::<FusedContext, FusedCommand>(
                        BackendKindId::of::<FusedBackend>("zero-fused").unwrap(),
                        vec![PolynomialType::Dense(DenseType::Coefficients)],
                    ),
                    StageDimensionCapabilities::new(vec![64]).unwrap(),
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let inner = Arc::new(CpuInnerCommitOperation::new(backend, prepared));
        builder
            .register_inner(
                PreparedInnerCommitment::new(
                    inner.clone(),
                    inner.owner().clone(),
                    inner_context,
                    CommitmentRequestCapabilities::split::<FusedContext>(
                        BackendKindId::of::<FusedBackend>("cpu-inner").unwrap(),
                        vec![PolynomialType::Dense(DenseType::Coefficients)],
                    ),
                    StageDimensionCapabilities::new(vec![64]).unwrap(),
                    Some(inner.portable_exporter()),
                )
                .unwrap(),
            )
            .unwrap();
        let compression =
            Arc::new(CpuCompressionOperation::new(backend, prepared, expanded).unwrap());
        builder
            .register_compression(PreparedCompression::new(
                compression.clone(),
                compression.owner().clone(),
                compression_context,
                CompressionOperationCapabilities::cpu::<F>(),
                None,
            ))
            .unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn independent_boundaries_compile_to_an_inspectable_sequence() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 128 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared =
            crate::compute::ComputeBackendSetup::prepare_setup(&backend, &setup).unwrap();
        let fused_metal = fused_executor(&backend, &prepared, setup.expanded.as_ref());
        let fused_cpu = fused_executor(&backend, &prepared, setup.expanded.as_ref());
        let split_cpu = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            PortableStatePolicy,
        )
        .unwrap();
        let split_metal = CommitmentExecutor::cpu(
            &backend,
            &prepared,
            setup.expanded.as_ref(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
            PortableStatePolicy,
        )
        .unwrap();
        let mut builder = CommitmentExecutionScheduleBuilder::new(4).unwrap();
        for (inner_outer, compression, executor) in [
            ("fused", "metal", &fused_metal),
            ("fused", "cpu", &fused_cpu),
            ("cpu-split", "cpu", &split_cpu),
        ] {
            builder
                .register_executor(inner_outer, compression, executor)
                .unwrap();
        }
        builder
            .inner_outer(0..2, "fused", InnerOuterRouteKind::Fused)
            .unwrap()
            .inner_outer(2..4, "cpu-split", InnerOuterRouteKind::Split)
            .unwrap()
            .compression(0..1, "metal")
            .unwrap()
            .compression(1..4, "cpu")
            .unwrap();
        let schedule = builder.compile().unwrap();
        let steps = schedule.steps().cloned().collect::<Vec<_>>();
        assert_eq!(steps.len(), 4);
        assert_eq!(steps[0].inner_outer(), "fused");
        assert_eq!(steps[0].compression(), "metal");
        assert_eq!(steps[1].inner_outer(), "fused");
        assert_eq!(steps[1].compression(), "cpu");
        assert_eq!(steps[2].inner_outer_kind(), InnerOuterRouteKind::Split);
        assert_eq!(steps[3].inner_outer(), "cpu-split");
        assert!(std::ptr::eq(schedule.executor(0).unwrap(), &fused_metal));
        assert!(std::ptr::eq(schedule.executor(1).unwrap(), &fused_cpu));
        assert!(std::ptr::eq(schedule.executor(2).unwrap(), &split_cpu));

        let mut mismatch = CommitmentExecutionScheduleBuilder::new(1).unwrap();
        mismatch
            .register_executor("declared-fused", "cpu", &split_cpu)
            .unwrap()
            .inner_outer(0..1, "declared-fused", InnerOuterRouteKind::Fused)
            .unwrap()
            .compression(0..1, "cpu")
            .unwrap();
        assert!(mismatch.compile().is_err());

        for (i, j) in [(0, 0), (4, 4), (1, 3), (3, 1), (2, 2)] {
            let mut boundaries = CommitmentExecutionScheduleBuilder::new(4).unwrap();
            for (inner_outer, compression, executor) in [
                ("fused", "metal", &fused_metal),
                ("fused", "cpu", &fused_cpu),
                ("split", "metal", &split_metal),
                ("split", "cpu", &split_cpu),
            ] {
                boundaries
                    .register_executor(inner_outer, compression, executor)
                    .unwrap();
            }
            boundaries
                .inner_outer(0..i, "fused", InnerOuterRouteKind::Fused)
                .unwrap()
                .inner_outer(i..4, "split", InnerOuterRouteKind::Split)
                .unwrap()
                .compression(0..j, "metal")
                .unwrap()
                .compression(j..4, "cpu")
                .unwrap();
            assert_eq!(boundaries.compile().unwrap().steps().len(), 4);
        }
    }

    #[test]
    fn mixed_executor_routes_fused_then_inner_only_without_an_outer_operation() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 128 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared =
            crate::compute::ComputeBackendSetup::prepare_setup(&backend, &setup).unwrap();
        let executor = fused_executor(&backend, &prepared, setup.expanded.as_ref());
        let mut builder = CommitmentExecutionScheduleBuilder::new(2).unwrap();
        builder
            .register_executor("hybrid", "cpu", &executor)
            .unwrap()
            .inner_outer(0..1, "hybrid", InnerOuterRouteKind::Fused)
            .unwrap()
            .inner_outer(1..2, "hybrid", InnerOuterRouteKind::InnerOnly)
            .unwrap()
            .compression(0..2, "cpu")
            .unwrap();

        let schedule = builder.compile().unwrap();
        let steps = schedule.steps().collect::<Vec<_>>();
        assert_eq!(steps[0].inner_outer_kind(), InnerOuterRouteKind::Fused);
        assert_eq!(steps[1].inner_outer_kind(), InnerOuterRouteKind::InnerOnly);
    }

    #[test]
    fn gaps_overlaps_and_missing_pairs_fail_before_execution() {
        let mut gaps =
            CommitmentExecutionScheduleBuilder::<F, PortableStatePolicy>::new(2).unwrap();
        gaps.inner_outer(0..1, "fused", InnerOuterRouteKind::Fused)
            .unwrap();
        gaps.compression(0..2, "cpu").unwrap();
        assert!(gaps.compile().is_err());

        let mut overlap =
            CommitmentExecutionScheduleBuilder::<F, PortableStatePolicy>::new(2).unwrap();
        overlap
            .inner_outer(0..2, "fused", InnerOuterRouteKind::Fused)
            .unwrap();
        assert!(overlap
            .inner_outer(1..2, "cpu", InnerOuterRouteKind::Split)
            .is_err());
    }
}
