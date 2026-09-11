use super::{CommitmentExecutionPlan, CommitmentExecutor, CommitmentStatePolicy, PolynomialType};
use crate::compute::requirements::{signed_commit_domain, SignedCommitSource};
use crate::compute::{
    CompressionComputeBackend, NttCacheOwnerId, NttOperationCluster, RoutedNttRequirement,
};
use akita_error::{checked, AkitaError};
use akita_types::{AkitaSetupDescriptor, NttCacheKey, NttTransformDomain};
use jolt_field::{CanonicalEncoding, Field};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_BACKEND_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_OPERATION_ID: AtomicU64 = AtomicU64::new(1);

/// Builder-issued identity for one physical backend instance.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendInstanceId(u64);

impl std::fmt::Debug for BackendInstanceId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("BackendInstanceId(..)")
    }
}

impl BackendInstanceId {
    pub(crate) fn issue() -> Self {
        Self(NEXT_BACKEND_INSTANCE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Builder-issued identity for one registered stage operation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommitmentOperationId(u64);

impl std::fmt::Debug for CommitmentOperationId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CommitmentOperationId(..)")
    }
}

impl CommitmentOperationId {
    pub(crate) fn issue() -> Self {
        Self(NEXT_OPERATION_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Commitment stage that owns one exact NTT request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitmentNttStage {
    /// Inner A commitment.
    Inner,
    /// Outer B commitment.
    Outer,
}

/// Execution route that owns a proof-wide commitment cache request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitmentNttRoute {
    /// Terminal A-only commitment.
    InnerOnly,
    /// A/B commitment, executed by the fused registration when one exists.
    InnerOuter,
}

/// Exact cache request routed to one registered commitment stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentNttRequirement {
    stage: CommitmentNttStage,
    key: NttCacheKey,
    routing_extent: usize,
}

impl CommitmentNttRequirement {
    /// Construct after validating the operation-level routing extent.
    pub fn new(
        stage: CommitmentNttStage,
        key: NttCacheKey,
        routing_extent: usize,
    ) -> Result<Self, AkitaError> {
        if routing_extent < key.num_ring_elements {
            return Err(AkitaError::InvalidSetup(
                "commitment NTT routing extent is smaller than its cache prefix".into(),
            ));
        }
        Ok(Self {
            stage,
            key,
            routing_extent,
        })
    }

    /// Owning commitment stage.
    pub const fn stage(&self) -> CommitmentNttStage {
        self.stage
    }

    /// Exact backend cache key.
    pub const fn key(&self) -> NttCacheKey {
        self.key
    }

    /// Full operation extent used by cached-versus-streamed policy.
    pub const fn routing_extent(&self) -> usize {
        self.routing_extent
    }

    fn routed(&self) -> RoutedNttRequirement {
        RoutedNttRequirement {
            fold_level: 0,
            cluster: NttOperationCluster::Commit,
            commitment_stage: Some(self.stage),
            commitment_route: Some(CommitmentNttRoute::InnerOuter),
            key: self.key,
            routing_extent: self.routing_extent,
        }
    }
}

impl CommitmentExecutionPlan {
    /// Exact A-matrix cache request for a selected standard source kernel.
    ///
    /// One-hot commitment reads matrix columns directly and therefore has no
    /// inner NTT requirement.
    pub fn inner_ntt_requirement(
        &self,
        selected: PolynomialType,
    ) -> Result<Option<CommitmentNttRequirement>, AkitaError> {
        let source = match selected {
            PolynomialType::Dense(_) => SignedCommitSource::Dense,
            PolynomialType::ShortNorm(_) => SignedCommitSource::RecursiveWitness,
            PolynomialType::OneHot(_) => return Ok(None),
        };
        let plan = self.inner();
        let width = checked::product([plan.num_positions_per_block, plan.num_digits_inner])
            .ok_or_else(|| AkitaError::InvalidSetup("commitment A width overflow".into()))?;
        let domain = signed_commit_domain(
            self.inner_modulus_profile(),
            plan.ring_dimension,
            width,
            plan.log_basis_inner,
            source,
        )?;
        let key = NttCacheKey::from_matrix_shape(plan.ring_dimension, plan.n_a, width, domain)?;
        let routing_extent = checked::product([plan.n_a, width])
            .ok_or_else(|| AkitaError::InvalidSetup("commitment A extent overflow".into()))?;
        CommitmentNttRequirement::new(CommitmentNttStage::Inner, key, routing_extent).map(Some)
    }

    /// Exact B-matrix cache request when this route contains an outer stage.
    pub fn outer_ntt_requirement(&self) -> Result<Option<CommitmentNttRequirement>, AkitaError> {
        let Some(plan) = self.uncompressed().map(|plan| plan.outer()) else {
            return Ok(None);
        };
        let width = plan.geometry().physical_input_width();
        let key = NttCacheKey::from_matrix_shape(
            plan.ring_dimension(),
            plan.n_b(),
            width,
            NttTransformDomain::Negacyclic,
        )?;
        let routing_extent = checked::product([plan.n_b(), width])
            .ok_or_else(|| AkitaError::InvalidSetup("commitment B extent overflow".into()))?;
        CommitmentNttRequirement::new(CommitmentNttStage::Outer, key, routing_extent).map(Some)
    }
}

/// Object-safe lifecycle boundary for backend resources used by a stage.
pub trait CommitmentResourceControl<F>: Send + Sync
where
    F: Field + CanonicalEncoding,
{
    /// Setup descriptor from which the captured prepared state was built.
    fn setup_descriptor(&self) -> &AkitaSetupDescriptor;

    /// Ensure one exact NTT cache slot.
    fn ensure_ntt_slot(&self, requirement: CommitmentNttRequirement) -> Result<(), AkitaError>;

    /// Whether this request remains cached rather than streamed.
    fn requirement_is_cached(
        &self,
        requirement: CommitmentNttRequirement,
    ) -> Result<bool, AkitaError>;

    /// Physical cache-owner identity used for deduplication.
    fn cache_owner_id(&self) -> NttCacheOwnerId;

    /// Release backend-designated cache slots.
    fn release_built_ntt_slots(&self) -> Result<usize, AkitaError>;

    /// Current compression cache bytes when exposed by the backend.
    fn compression_cache_bytes(&self) -> Option<usize> {
        None
    }
}

/// Associated-type-erased resource adapter over one prepared backend.
pub struct PreparedCommitmentResources<'a, F, B>
where
    F: Field + CanonicalEncoding,
    B: CompressionComputeBackend<F>,
{
    backend: &'a B,
    prepared: &'a B::PreparedSetup,
    setup: AkitaSetupDescriptor,
    marker: PhantomData<fn() -> F>,
}

impl<'a, F, B> PreparedCommitmentResources<'a, F, B>
where
    F: Field + CanonicalEncoding,
    B: CompressionComputeBackend<F>,
{
    /// Capture a prepared backend after checking its expanded setup.
    pub fn new(
        backend: &'a B,
        prepared: &'a B::PreparedSetup,
        setup: &akita_types::AkitaExpandedSetup<F>,
    ) -> Result<Self, AkitaError> {
        backend.validate_prepared_setup(prepared, setup)?;
        Ok(Self {
            backend,
            prepared,
            setup: setup.descriptor().clone(),
            marker: PhantomData,
        })
    }
}

impl<F, B> CommitmentResourceControl<F> for PreparedCommitmentResources<'_, F, B>
where
    F: Field + CanonicalEncoding,
    B: CompressionComputeBackend<F>,
{
    fn setup_descriptor(&self) -> &AkitaSetupDescriptor {
        &self.setup
    }

    fn ensure_ntt_slot(&self, requirement: CommitmentNttRequirement) -> Result<(), AkitaError> {
        self.backend
            .ensure_ntt_slot(self.prepared, requirement.key())
    }

    fn requirement_is_cached(
        &self,
        requirement: CommitmentNttRequirement,
    ) -> Result<bool, AkitaError> {
        self.backend
            .ntt_requirement_is_cached(self.prepared, requirement.routed())
    }

    fn cache_owner_id(&self) -> NttCacheOwnerId {
        self.backend.ntt_cache_owner_id(self.prepared)
    }

    fn release_built_ntt_slots(&self) -> Result<usize, AkitaError> {
        self.backend.release_built_ntt_slots(self.prepared)
    }

    fn compression_cache_bytes(&self) -> Option<usize> {
        self.backend.compression_cache_bytes(self.prepared)
    }
}

/// Explicit resource declaration attached to one stage registration.
pub enum StageResources<'a, F>
where
    F: Field + CanonicalEncoding,
{
    /// This operation owns no prepared matrix or cache resources.
    None,
    /// This operation uses the captured resource controller.
    Controlled(Arc<dyn CommitmentResourceControl<F> + 'a>),
}

/// Builder-validated backend identity, diagnostic name, and resource declaration.
pub struct CommitmentOperationContext<'a, F>
where
    F: Field + CanonicalEncoding,
{
    pub(crate) setup: AkitaSetupDescriptor,
    pub(crate) backend_instance: BackendInstanceId,
    pub(crate) name: &'static str,
    pub(crate) resources: StageResources<'a, F>,
}

impl<F> Clone for StageResources<'_, F>
where
    F: Field + CanonicalEncoding,
{
    fn clone(&self) -> Self {
        match self {
            Self::None => Self::None,
            Self::Controlled(control) => Self::Controlled(control.clone()),
        }
    }
}

impl<'a, F> StageResources<'a, F>
where
    F: Field + CanonicalEncoding,
{
    /// Explicit no-resources declaration.
    pub const fn none() -> Self {
        Self::None
    }

    /// Capture a resource controller.
    pub fn controlled(control: impl CommitmentResourceControl<F> + 'a) -> Self {
        Self::Controlled(Arc::new(control))
    }

    pub(crate) fn validate_setup(&self, setup: &AkitaSetupDescriptor) -> Result<(), AkitaError> {
        if let Self::Controlled(control) = self {
            if control.setup_descriptor() != setup {
                return Err(AkitaError::InvalidSetup(
                    "commitment stage resources were prepared for a different setup".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) const fn is_controlled(&self) -> bool {
        matches!(self, Self::Controlled(_))
    }

    pub(crate) fn cache_owner_id(&self) -> Option<NttCacheOwnerId> {
        match self {
            Self::None => None,
            Self::Controlled(control) => Some(control.cache_owner_id()),
        }
    }

    pub(crate) fn ensure_ntt_slot(
        &self,
        requirement: CommitmentNttRequirement,
    ) -> Result<(), AkitaError> {
        match self {
            Self::None => Err(AkitaError::InvalidSetup(
                "commitment stage has no resources for its NTT requirement".into(),
            )),
            Self::Controlled(control) => control.ensure_ntt_slot(requirement),
        }
    }

    pub(crate) fn requirement_is_cached(
        &self,
        requirement: CommitmentNttRequirement,
    ) -> Result<bool, AkitaError> {
        match self {
            Self::None => Ok(false),
            Self::Controlled(control) => control.requirement_is_cached(requirement),
        }
    }

    pub(crate) fn release_built_ntt_slots(&self) -> Result<usize, AkitaError> {
        match self {
            Self::None => Ok(0),
            Self::Controlled(control) => control.release_built_ntt_slots(),
        }
    }
}

impl<'a, F, SP> CommitmentExecutor<'a, F, SP>
where
    F: Field + CanonicalEncoding,
    SP: CommitmentStatePolicy<F>,
{
    fn routed_requirement(
        requirement: RoutedNttRequirement,
    ) -> Result<CommitmentNttRequirement, AkitaError> {
        let stage = requirement.commitment_stage.ok_or_else(|| {
            AkitaError::InvalidSetup("commitment NTT requirement has no stage discriminator".into())
        })?;
        CommitmentNttRequirement::new(stage, requirement.key, requirement.routing_extent)
    }

    fn routed_resources(
        &self,
        route: CommitmentNttRoute,
        stage: CommitmentNttStage,
    ) -> Result<&StageResources<'a, F>, AkitaError> {
        if route == CommitmentNttRoute::InnerOuter {
            if let Some(fused) = self.fused() {
                return Ok(&fused.stage.resources);
            }
        }
        if self.inner().is_none() && self.outer().is_none() {
            return self
                .fused()
                .map(|fused| &fused.stage.resources)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("commitment route has no A/B resources".into())
                });
        }
        match stage {
            CommitmentNttStage::Inner => self
                .inner()
                .map(|inner| &inner.stage.resources)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("commitment route has no inner resources".into())
                }),
            CommitmentNttStage::Outer => self
                .outer()
                .map(|outer| &outer.stage.resources)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("commitment route has no outer resources".into())
                }),
        }
    }

    pub(crate) fn prewarm_routed_requirement(
        &self,
        requirement: RoutedNttRequirement,
    ) -> Result<(), AkitaError> {
        let route = requirement.commitment_route.ok_or_else(|| {
            AkitaError::InvalidSetup("commitment NTT requirement has no route discriminator".into())
        })?;
        let requirement = Self::routed_requirement(requirement)?;
        self.routed_resources(route, requirement.stage())?
            .ensure_ntt_slot(requirement)
    }

    pub(crate) fn retained_routed_requirement(
        &self,
        requirement: RoutedNttRequirement,
    ) -> Result<Option<NttCacheOwnerId>, AkitaError> {
        let route = requirement.commitment_route.ok_or_else(|| {
            AkitaError::InvalidSetup("commitment NTT requirement has no route discriminator".into())
        })?;
        let requirement = Self::routed_requirement(requirement)?;
        let resources = self.routed_resources(route, requirement.stage())?;
        if !resources.requirement_is_cached(requirement)? {
            return Ok(None);
        }
        Ok(resources.cache_owner_id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_types::{CommittedGroupParams, SisModulusProfileId};

    fn full_plan() -> CommitmentExecutionPlan {
        let params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q64Offset59,
            64,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 8, 1, 2, 2)
        .unwrap();
        CommitmentExecutionPlan::for_root(&params.own_group().profile).unwrap()
    }

    #[test]
    fn commitment_requirement_rejects_short_routing_extent() {
        let key = NttCacheKey::from_matrix_shape(64, 2, 8, NttTransformDomain::Negacyclic).unwrap();
        assert!(CommitmentNttRequirement::new(CommitmentNttStage::Inner, key, 15).is_err());
        assert!(CommitmentNttRequirement::new(CommitmentNttStage::Inner, key, 16).is_ok());
    }

    #[test]
    fn selected_kernel_determines_stage_specific_requirements() {
        let plan = full_plan();
        let dense = plan
            .inner_ntt_requirement(PolynomialType::Dense(super::super::DenseType::Coefficients))
            .unwrap()
            .unwrap();
        assert_eq!(dense.stage(), CommitmentNttStage::Inner);
        assert_eq!(dense.key().ring_d, plan.inner().ring_dimension);

        let onehot = super::super::OneHotType::new(8, super::super::OneHotIndexWidth::U8).unwrap();
        assert!(plan
            .inner_ntt_requirement(PolynomialType::OneHot(onehot))
            .unwrap()
            .is_none());

        let outer = plan.outer_ntt_requirement().unwrap().unwrap();
        assert_eq!(outer.stage(), CommitmentNttStage::Outer);
        assert_eq!(outer.key().domain, NttTransformDomain::Negacyclic);
        assert_eq!(
            outer.key().num_ring_elements,
            plan.uncompressed()
                .unwrap()
                .outer()
                .geometry()
                .physical_input_width()
        );
    }
}
