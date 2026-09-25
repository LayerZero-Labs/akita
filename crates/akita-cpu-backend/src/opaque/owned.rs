//! Reusable source and commitment ownership. No proof state lives in these objects.
use super::source::{
    prepare_and_evaluate_opening_group, scalar_opening_from_folded_ring,
    PreparedExtensionOpeningGroup,
};
use crate::arithmetic::extension_opening_reduction::{
    tensor_column_partials_from_base_evals, tensor_packed_witness_evals,
};
use crate::commitment::CommitmentSource;
use crate::opaque::CpuPreparedOpeningHandle;
use crate::opaque::*;
use crate::sources::poly::SourceCoefficients;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, Ring, Unreduced};
use std::sync::Arc;

/// An immutable imported source owned by one backend.
pub struct SourceHandle<
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
> {
    pub(super) owner: u64,
    pub(super) storage: Arc<dyn PrivateSource<F, E, Cfg>>,
    pub(super) metadata: SourceMetadata,
}
impl<F, E, Cfg> Clone for SourceHandle<F, E, Cfg>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
{
    fn clone(&self) -> Self {
        Self {
            owner: self.owner,
            storage: self.storage.clone(),
            metadata: self.metadata,
        }
    }
}
/// A reusable commitment retaining the exact source and parameters it committed.
pub struct CommitmentHandle<
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
> {
    pub(super) owner: u64,
    pub(super) committed: Arc<CommittedSource<F, E, Cfg>>,
}
impl<F, E, Cfg> Clone for CommitmentHandle<F, E, Cfg>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
{
    fn clone(&self) -> Self {
        Self {
            owner: self.owner,
            committed: self.committed.clone(),
        }
    }
}
pub(super) struct CommittedSource<
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
> {
    pub(super) source: Arc<dyn PrivateSource<F, E, Cfg>>,
    pub(super) metadata: SourceMetadata,
    pub(super) commitment_id: u128,
    pub(super) parameters: GroupCommitPhaseParams,
    pub(super) public: Commitment<F>,
    pub(super) retained: crate::commitment::PortableCommitmentHandle<F>,
}
/// Public commitment and the reusable private handle used for proving.
pub struct CommitOutput<
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
> {
    pub committed_group: CommittedGroup<F>,
    pub private_handle: CommitmentHandle<F, E, Cfg>,
}
impl<F, E, Cfg> CommitmentHandleMetadata for CommitmentHandle<F, E, Cfg>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
{
    fn metadata(&self) -> SourceMetadata {
        self.committed.metadata
    }
}

// This object-safe interface never leaves the CPU crate. It retains one whole
// homogeneous group while allowing different representations in one proof.
pub(super) trait PrivateSource<
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
>: Send + Sync
{
    fn commitment_sources(&self) -> Vec<&dyn CommitmentSource<F>>;
    fn dense_polynomials(&self) -> Result<Vec<crate::DensePoly<F>>, AkitaError>;
    #[cfg(feature = "response-model-diagnostics")]
    fn source_l2_sq(&self) -> Option<u128>;
    fn opening(
        &self,
        backend: &CpuBackend<Cfg>,
        binding: &OperationBinding,
        plan: &ValidatedRecursiveGroupOpeningPlan<'_, E>,
        source: crate::opaque::openings::PreparedOpeningSource<F, E, Cfg>,
    ) -> Result<PreparedGroupOpening<E, CpuPreparedOpeningHandle<F, E, Cfg>>, AkitaError>;
    fn probe(
        &self,
        backend: &CpuBackend<Cfg>,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<super::CpuAcceptedFoldHandle<F>>, AkitaError>;
    fn extension(
        &self,
        backend: &CpuBackend<Cfg>,
        ring_dimension: usize,
        point: &[E],
    ) -> Result<PreparedExtensionOpeningGroup<E>, AkitaError>;
    #[allow(clippy::too_many_arguments)]
    fn begin_eor(
        &self,
        backend: &CpuBackend<Cfg>,
        ring_dimension: usize,
        coefficients: &[E],
        tail: &[E],
        eta: &[E],
        extra: Vec<E>,
        claim: E,
    ) -> Result<Box<dyn crate::opaque::eor::ExtensionOpeningSession<E>>, AkitaError>;
}
pub(super) struct OwnedPolynomials<P> {
    pub(super) polynomials: Vec<P>,
}

impl<F, E, Cfg, P> PrivateSource<F, E, Cfg> for OwnedPolynomials<P>
where
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
    F: Field + CanonicalEncoding + AkitaSerialize + Ring + Unreduced + 'static,
    F::Wide: From<F> + jolt_field::AdditiveGroup,
    E: ExtField<F>
        + FpExtEncoding<F>
        + MulBaseUnreduced<F>
        + Unreduced
        + Fold
        + AkitaSerialize
        + 'static,
    P: SourceCoefficients<F>
        + RuntimeRootProvePoly<F>
        + CommitmentSource<F>
        + Send
        + Sync
        + 'static,
    CpuBackend<Cfg>: ComputeBackendSetup<F, PreparedSetup = CpuPreparedSetup<F>>
        + FoldHandleBackend<F, AcceptedFold = super::CpuAcceptedFoldHandle<F>>
        + RuntimeOpeningProveBackendFor<F, P>
        + RuntimeCoefficientPackingBackendFor<F, P, E>,
{
    #[cfg(feature = "response-model-diagnostics")]
    fn source_l2_sq(&self) -> Option<u128> {
        self.polynomials.iter().try_fold(0u128, |sum, p| {
            sum.checked_add(RootPolyMeta::<F>::exact_integer_coeff_l2_sq(p)?)
        })
    }
    fn commitment_sources(&self) -> Vec<&dyn CommitmentSource<F>> {
        self.polynomials
            .iter()
            .map(|p| p as &dyn CommitmentSource<F>)
            .collect()
    }
    fn dense_polynomials(&self) -> Result<Vec<crate::DensePoly<F>>, AkitaError> {
        self.polynomials
            .iter()
            .map(|polynomial| {
                crate::DensePoly::from_field_evals(
                    RootPolyMeta::num_vars(polynomial),
                    polynomial.source_coefficients()?.into_owned(),
                )
            })
            .collect()
    }
    fn opening(
        &self,
        backend: &CpuBackend<Cfg>,
        binding: &OperationBinding,
        plan: &ValidatedRecursiveGroupOpeningPlan<'_, E>,
        source: crate::opaque::openings::PreparedOpeningSource<F, E, Cfg>,
    ) -> Result<PreparedGroupOpening<E, CpuPreparedOpeningHandle<F, E, Cfg>>, AkitaError> {
        let polys = self.polynomials.iter().collect::<Vec<_>>();
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            plan.ring_dimension(),
            |D| {
                let prepared = Some(backend.prepared()?);
                if let OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension,
                } = plan.opening_method()
                {
                    let geometry = SubringCoefficientPackingGeometry::try_new(
                        E::DEGREE,
                        D,
                        challenge_subring_dimension,
                    )?;
                    let first = polys.first().ok_or(AkitaError::InvalidProof)?;
                    let live = <P as RootPolyShape<F, D>>::num_live_ring_elems(*first);
                    let vars = RootPolyMeta::<F>::num_vars(*first);
                    if polys.iter().any(|p| {
                        <P as RootPolyShape<F, D>>::num_live_ring_elems(*p) != live
                            || RootPolyMeta::<F>::num_vars(*p) != vars
                    }) || live.div_ceil(plan.positions_per_block()) != plan.live_blocks()
                    {
                        return Err(AkitaError::InvalidInput(
                            "source and opening geometry disagree".into(),
                        ));
                    }
                    let point = PreparedSubringCoefficientPackingPoint::new(
                        geometry,
                        plan.basis(),
                        live,
                        plan.positions_per_block(),
                        vars,
                        plan.point(),
                    )?;
                    let batch = <P as RootOpeningSource<F, D>>::opening_batch(&polys)?;
                    let partials =
                        SubringCoefficientPackingBatchKernel::coefficient_packing_partials_batch(
                            backend,
                            prepared,
                            batch,
                            SubringCoefficientPackingPlan { point: &point },
                        )?;
                    let openings = partials
                        .iter()
                        .map(|p| {
                            crate::arithmetic::coefficient_packing_fold::coefficient_packing_scalar_opening::<F, E>(
                                geometry,
                                point.num_live_blocks(),
                                std::slice::from_ref(p),
                                &[E::one()],
                                point.live_block_weights(),
                                point.tail_weights(),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(crate::opaque::prepared_opening::coefficient_packing(
                        binding.clone(),
                        source,
                        point,
                        partials,
                        openings,
                    ));
                }
                let (point, (folded, by_claim)) =
                    prepare_and_evaluate_opening_group::<F, E, P, CpuBackend<Cfg>, D>(
                        backend,
                        prepared,
                        &polys,
                        plan.point(),
                        plan.basis(),
                        plan.positions_per_block(),
                        plan.live_blocks(),
                        plan.alpha_bits(),
                    )?;
                let inner = &plan.point()[..plan.point().len().min(plan.alpha_bits())];
                let openings = folded
                    .iter()
                    .map(|row| {
                        scalar_opening_from_folded_ring::<F, E, D>(row, &point, inner, plan.basis())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(crate::opaque::prepared_opening::evaluation_trace(
                    binding.clone(),
                    source,
                    point,
                    by_claim
                        .iter()
                        .map(|row| RingVec::from_ring_elems(row).into_compact())
                        .collect(),
                    openings,
                ))
            }
        )
    }
    fn probe(
        &self,
        backend: &CpuBackend<Cfg>,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<super::CpuAcceptedFoldHandle<F>>, AkitaError> {
        let polys = self.polynomials.iter().collect::<Vec<_>>();
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            plan.ring_dimension(),
            |D| {
                let batch = <P as RootOpeningSource<F, D>>::opening_batch(&polys)?;
                FoldResponseKernel::probe(backend, Some(backend.prepared()?), batch, plan)
            }
        )
    }
    fn extension(
        &self,
        _backend: &CpuBackend<Cfg>,
        _ring_dimension: usize,
        point: &[E],
    ) -> Result<PreparedExtensionOpeningGroup<E>, AkitaError> {
        let mut openings = Vec::with_capacity(self.polynomials.len());
        let mut proof_partials = Vec::new();
        let mut rows = Vec::with_capacity(self.polynomials.len());
        for poly in &self.polynomials {
            let coefficients = poly.source_coefficients()?;
            let partials = tensor_column_partials_from_base_evals::<F, E>(
                RootPolyMeta::<F>::num_vars(poly),
                &coefficients,
                point,
            )?;
            openings.push(derive_tensor_extension_opening_claim_from_partials::<F, E>(
                point, &partials,
            )?);
            rows.push(tensor_row_partials_from_columns::<F, E>(&partials)?);
            proof_partials.extend(partials);
        }
        Ok(PreparedExtensionOpeningGroup {
            openings,
            proof_partials,
            row_partials_by_claim: rows,
        })
    }
    fn begin_eor(
        &self,
        _backend: &CpuBackend<Cfg>,
        _ring_dimension: usize,
        coefficients: &[E],
        tail: &[E],
        eta: &[E],
        extra: Vec<E>,
        claim: E,
    ) -> Result<Box<dyn crate::opaque::eor::ExtensionOpeningSession<E>>, AkitaError> {
        let witnesses = self
            .polynomials
            .iter()
            .map(|poly| {
                tensor_packed_witness_evals::<F, E>(
                    RootPolyMeta::<F>::num_vars(poly),
                    &poly.source_coefficients()?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        crate::opaque::cpu_extension_opening_session_from_witnesses::<F, E>(
            witnesses,
            coefficients,
            tail,
            eta,
            extra,
            claim,
        )
    }
}

/// CPU-supported owned source representation. Import consumes the entire group.
#[allow(private_bounds)]
pub trait CpuSource<
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
>: SourceImport<F, E, Cfg>
{
}
trait SourceImport<
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
>: CommitmentSource<F> + Sized + 'static
{
    fn retain_owned(polynomials: Vec<Self>) -> Arc<dyn PrivateSource<F, E, Cfg>>;
}
impl<F, E, Cfg, P> SourceImport<F, E, Cfg> for P
where
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
    F: Field + CanonicalEncoding + AkitaSerialize + Ring + Unreduced + 'static,
    F::Wide: From<F> + jolt_field::AdditiveGroup,
    E: ExtField<F>
        + FpExtEncoding<F>
        + MulBaseUnreduced<F>
        + Unreduced
        + Fold
        + AkitaSerialize
        + 'static,
    P: SourceCoefficients<F>
        + RuntimeRootProvePoly<F>
        + CommitmentSource<F>
        + Send
        + Sync
        + 'static,
    CpuBackend<Cfg>: ComputeBackendSetup<F, PreparedSetup = CpuPreparedSetup<F>>
        + FoldHandleBackend<F, AcceptedFold = super::CpuAcceptedFoldHandle<F>>
        + RuntimeOpeningProveBackendFor<F, P>
        + RuntimeCoefficientPackingBackendFor<F, P, E>,
{
    fn retain_owned(polynomials: Vec<Self>) -> Arc<dyn PrivateSource<F, E, Cfg>> {
        Arc::new(OwnedPolynomials { polynomials })
    }
}
impl<F, E, Cfg, P> CpuSource<F, E, Cfg> for P
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
    P: SourceImport<F, E, Cfg>,
{
}

impl<Cfg: akita_config::CommitmentConfig> CpuBackend<Cfg> {
    /// Consume a homogeneous source group without exposing its storage to proving.
    pub fn import_source<P>(
        &self,
        polynomials: Vec<P>,
    ) -> Result<SourceHandle<Cfg::Field, Cfg::ExtField, Cfg>, AkitaError>
    where
        Cfg::Field: Field + CanonicalEncoding + AkitaSerialize + Ring + Unreduced + 'static,
        <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + jolt_field::AdditiveGroup,
        Cfg::ExtField: FpExtEncoding<Cfg::Field>
            + MulBaseUnreduced<Cfg::Field>
            + Unreduced
            + Fold
            + AkitaSerialize
            + 'static,
        P: CpuSource<Cfg::Field, Cfg::ExtField, Cfg>,
    {
        let prepared = self.prepared()?;
        let layout =
            crate::commitment::resolve_polynomial_group_layout(&polynomials, &prepared.expanded)?;
        let metadata = SourceMetadata::try_new(layout.num_polynomials(), layout.num_vars())?;
        Ok(SourceHandle {
            owner: self.owner().backend_id(),
            storage: P::retain_owned(polynomials),
            metadata,
        })
    }
}

impl<F, E, Cfg> core::fmt::Debug for SourceHandle<F, E, Cfg>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SourceHandle")
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}
impl<F, E, Cfg> core::fmt::Debug for CommitmentHandle<F, E, Cfg>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CommitmentHandle")
            .field("metadata", &self.committed.metadata)
            .finish_non_exhaustive()
    }
}
impl<F, E, Cfg> core::fmt::Debug for CommitOutput<F, E, Cfg>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F, ExtField = E>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CommitOutput")
            .field("private_handle", &self.private_handle)
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "response-model-diagnostics"))]
impl<Cfg: akita_config::CommitmentConfig> CpuBackend<Cfg> {
    /// Source energy of a witness, checked against the witness's proof binding.
    pub(crate) fn witness_source_l2_sq<F: Field>(
        &self,
        witness: &crate::opaque::CpuWitnessHandle,
    ) -> Result<Option<u128>, AkitaError> {
        self.validate_binding(&witness.operation_binding())?;
        Ok(witness.source_l2_sq::<F>())
    }
}

impl<F: Field + CanonicalEncoding> SourceCoefficients<F> for crate::DensePoly<F> {
    fn source_coefficients(&self) -> Result<std::borrow::Cow<'_, [F]>, AkitaError> {
        let len = akita_error::checked::pow2(RootPolyMeta::<F>::num_vars(self))
            .ok_or(AkitaError::InvalidProof)?;
        Ok(std::borrow::Cow::Borrowed(
            self.field_coeffs()
                .get(..len)
                .ok_or(AkitaError::InvalidProof)?,
        ))
    }
}
impl<F: Field + CanonicalEncoding, I: crate::OneHotIndex> SourceCoefficients<F>
    for crate::OneHotPoly<F, I>
{
    fn source_coefficients(&self) -> Result<std::borrow::Cow<'_, [F]>, AkitaError> {
        let len = akita_error::checked::product([self.onehot_k(), self.indices().len()])
            .ok_or(AkitaError::InvalidProof)?;
        let mut values = vec![F::zero(); len];
        for (chunk, index) in self.indices().iter().enumerate() {
            if let Some(index) = index {
                values[chunk * self.onehot_k() + index.as_usize()] = F::one();
            }
        }
        Ok(std::borrow::Cow::Owned(values))
    }
}
