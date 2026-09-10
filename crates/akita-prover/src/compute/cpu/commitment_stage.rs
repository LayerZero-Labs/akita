use super::{CpuBackend, CpuPreparedSetup};
use crate::compute::commitment::{
    BackendStateRef, CommitmentStateBinding, InnerCommitOperation, InnerCommitOutput, InnerImage,
    InnerImageExportOperation, InnerImageInput, OuterCommitOperation, ResolvedCommitSource,
    StateOwnerCapability,
};
use crate::compute::{CommitInnerPlan, OuterCommitPlan};
use crate::CommitInnerWitness;
use akita_error::{checked, AkitaError};
use akita_types::{dispatch_for_field, RingVec};
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::mem::size_of;
use std::sync::Arc;

struct CpuInnerImageStore<F: Field> {
    owner: StateOwnerCapability<InnerImage>,
    marker: std::marker::PhantomData<fn() -> F>,
}

impl<F: Field> Clone for CpuInnerImageStore<F> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<F: Field + 'static> CpuInnerImageStore<F> {
    fn with_witnesses<R>(
        &self,
        image: &BackendStateRef<InnerImage>,
        consume: impl FnOnce(&[CommitInnerWitness<F>]) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        consume(self.owner.value::<Vec<CommitInnerWitness<F>>>(image)?)
    }

    fn export_rows(
        &self,
        plan: &CommitInnerPlan,
        image: &BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        if image.binding().inner_plan() != plan {
            return Err(AkitaError::InvalidInput(
                "CPU inner export plan disagrees with resident state".into(),
            ));
        }
        let expected_coefficients =
            checked::product([plan.num_live_blocks, plan.n_a, plan.ring_dimension]).ok_or_else(
                || AkitaError::InvalidInput("CPU inner export extent overflow".into()),
            )?;
        self.with_witnesses(image, |witnesses| {
            if witnesses.len() != image.binding().source_count() {
                return Err(AkitaError::InvalidInput(
                    "CPU inner resident source count is invalid".into(),
                ));
            }
            witnesses
                .iter()
                .map(|witness| {
                    if witness.inner_rows.ring_dim() != plan.ring_dimension
                        || witness.inner_rows.coeff_len() != expected_coefficients
                    {
                        return Err(AkitaError::InvalidInput(
                            "CPU inner resident row shape is invalid".into(),
                        ));
                    }
                    Ok(witness.inner_rows.clone())
                })
                .collect()
        })
    }

    fn consume_rows(
        &self,
        plan: &CommitInnerPlan,
        image: BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        if image.binding().inner_plan() != plan {
            return Err(AkitaError::InvalidInput(
                "CPU inner export plan disagrees with resident state".into(),
            ));
        }
        let source_count = image.binding().source_count();
        let expected_coefficients =
            checked::product([plan.num_live_blocks, plan.n_a, plan.ring_dimension]).ok_or_else(
                || AkitaError::InvalidInput("CPU inner export extent overflow".into()),
            )?;
        match self.owner.try_unwrap::<Vec<CommitInnerWitness<F>>>(image)? {
            Ok(witnesses) => {
                if witnesses.len() != source_count {
                    return Err(AkitaError::InvalidInput(
                        "CPU inner resident source count is invalid".into(),
                    ));
                }
                witnesses
                    .into_iter()
                    .map(|witness| {
                        if witness.inner_rows.ring_dim() != plan.ring_dimension
                            || witness.inner_rows.coeff_len() != expected_coefficients
                        {
                            return Err(AkitaError::InvalidInput(
                                "CPU inner resident row shape is invalid".into(),
                            ));
                        }
                        Ok(witness.inner_rows)
                    })
                    .collect()
            }
            Err(shared) => self.export_rows(plan, &shared),
        }
    }
}

struct CpuInnerImageExporter<F: Field> {
    storage: CpuInnerImageStore<F>,
}

impl<F: Field + 'static> InnerImageExportOperation<F> for CpuInnerImageExporter<F> {
    fn export_inner_rows(
        &self,
        plan: &CommitInnerPlan,
        image: &BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        self.storage.export_rows(plan, image)
    }

    fn consume_inner_rows(
        &self,
        plan: &CommitInnerPlan,
        image: BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        self.storage.consume_rows(plan, image)
    }
}

/// CPU inner-stage operation whose returned state directly owns its witnesses.
pub struct CpuInnerCommitOperation<'a, F: Field> {
    backend: &'a CpuBackend,
    prepared: &'a CpuPreparedSetup<F>,
    storage: CpuInnerImageStore<F>,
}

impl<'a, F: Field> CpuInnerCommitOperation<'a, F> {
    /// Construct a CPU inner operation for a custom commitment executor.
    pub fn new(backend: &'a CpuBackend, prepared: &'a CpuPreparedSetup<F>) -> Self {
        Self {
            backend,
            prepared,
            storage: CpuInnerImageStore {
                owner: StateOwnerCapability::new(),
                marker: std::marker::PhantomData,
            },
        }
    }

    /// State owner used by this operation and compatible outer stages.
    pub fn owner(&self) -> &StateOwnerCapability<InnerImage> {
        &self.storage.owner
    }

    /// Exporter for crossing from this resident CPU image to another stage owner.
    pub fn portable_exporter(&self) -> Arc<dyn InnerImageExportOperation<F>>
    where
        F: 'static,
    {
        Arc::new(CpuInnerImageExporter {
            storage: self.storage.clone(),
        })
    }
}

impl<F> InnerCommitOperation<F> for CpuInnerCommitOperation<'_, F>
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator + 'static,
{
    fn commit_inner(
        &self,
        binding: &CommitmentStateBinding,
        plan: &CommitInnerPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<InnerCommitOutput, AkitaError> {
        if binding.inner_plan() != plan || binding.source_count() != sources.len() {
            return Err(AkitaError::InvalidInput(
                "CPU inner stage request disagrees with its registration binding".into(),
            ));
        }
        let images = if sources
            .first()
            .and_then(ResolvedCommitSource::external)
            .is_some()
        {
            if sources.iter().any(|source| source.external().is_none()) {
                return Err(AkitaError::InvalidInput(
                    "CPU inner stage cannot mix external and standard source paths".into(),
                ));
            }
            let first = sources[0].external().ok_or_else(|| {
                AkitaError::InvalidInput("external CPU source group is empty".into())
            })?;
            let identity = first.operation().identity();
            let capability = first.input().capability();
            let mut inputs = Vec::with_capacity(sources.len());
            for source in sources {
                source.validate_plan(plan)?;
                let external = source.external().ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "CPU inner stage cannot mix external and standard source paths".into(),
                    )
                })?;
                if external.operation().identity() != identity
                    || external.input().capability() != capability
                {
                    return Err(AkitaError::InvalidInput(
                        "external CPU source group is not operation-homogeneous".into(),
                    ));
                }
                inputs.push(*external.input());
            }
            let images = first
                .operation()
                .commit_group(plan, &inputs, self.prepared)?;
            if images.len() != sources.len() {
                return Err(AkitaError::InvalidInput(
                    "external CPU commitment returned the wrong source count".into(),
                ));
            }
            images
        } else {
            if sources.iter().any(|source| source.external().is_some()) {
                return Err(AkitaError::InvalidInput(
                    "CPU inner stage cannot mix standard and external source paths".into(),
                ));
            }
            dispatch_for_field!(
                akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
                F,
                plan.ring_dimension,
                |D| self
                    .backend
                    .commit_resolved_inner_host::<F, D>(self.prepared, sources, *plan,)
            )?
        };
        let retained_coefficients = images.iter().try_fold(0usize, |total, image| {
            total
                .checked_add(image.inner_rows.coeff_len())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("CPU inner retained extent overflow".into())
                })
        })?;
        let retained_bytes = checked::product([retained_coefficients, size_of::<F>()])
            .ok_or_else(|| AkitaError::InvalidInput("CPU inner retained bytes overflow".into()))?;
        Ok(InnerCommitOutput::new(self.storage.owner.bind(
            binding.clone(),
            retained_bytes,
            images,
        )))
    }
}

impl<F> InnerImageExportOperation<F> for CpuInnerCommitOperation<'_, F>
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator + 'static,
{
    fn export_inner_rows(
        &self,
        plan: &CommitInnerPlan,
        image: &crate::compute::commitment::BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        self.storage.export_rows(plan, image)
    }

    fn consume_inner_rows(
        &self,
        plan: &CommitInnerPlan,
        image: BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        self.storage.consume_rows(plan, image)
    }
}

/// CPU outer-stage operation over resident or explicitly exported inner rows.
pub struct CpuOuterCommitOperation<'a, F: Field> {
    backend: &'a CpuBackend,
    prepared: &'a CpuPreparedSetup<F>,
    inner_storage: CpuInnerImageStore<F>,
}

impl<'a, F: Field> CpuOuterCommitOperation<'a, F> {
    /// Construct a CPU outer operation for a custom commitment executor.
    pub fn new(
        backend: &'a CpuBackend,
        prepared: &'a CpuPreparedSetup<F>,
        inner: &CpuInnerCommitOperation<'a, F>,
    ) -> Self {
        Self {
            backend,
            prepared,
            inner_storage: inner.storage.clone(),
        }
    }

    fn commit_rows(
        &self,
        inner_plan: &CommitInnerPlan,
        outer_plan: &OuterCommitPlan,
        rows: &[&RingVec<F>],
    ) -> Result<RingVec<F>, AkitaError>
    where
        F: CanonicalEncoding,
    {
        dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            inner_plan.ring_dimension,
            |D_A| dispatch_for_field!(
                akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Outer),
                F,
                outer_plan.ring_dimension(),
                |D_B| crate::api::commitment::compute_outer_commitment_from_rows::<
                    F,
                    CpuBackend,
                    D_A,
                    D_B,
                >(self.backend, self.prepared, rows, inner_plan, outer_plan,)
            )
        )
    }
}

impl<F> OuterCommitOperation<F> for CpuOuterCommitOperation<'_, F>
where
    F: Field + CanonicalEncoding + 'static,
{
    fn commit_outer(
        &self,
        plan: &crate::compute::UncompressedCommitPlan,
        inner: InnerImageInput<'_, F>,
    ) -> Result<RingVec<F>, AkitaError> {
        let inner_plan = *plan.inner();
        let outer_plan = plan.outer();
        match inner {
            InnerImageInput::Owned(image) => {
                if image.binding().inner_plan() != plan.inner()
                    || image.binding().source_count() != outer_plan.geometry().num_polynomials()
                {
                    return Err(AkitaError::InvalidInput(
                        "CPU outer stage received an inner image from a different request".into(),
                    ));
                }
                self.inner_storage.with_witnesses(image, |witnesses| {
                    let rows = witnesses
                        .iter()
                        .map(|witness| &witness.inner_rows)
                        .collect::<Vec<_>>();
                    self.commit_rows(&inner_plan, outer_plan, &rows)
                })
            }
            InnerImageInput::HostRows(rows) => {
                let rows = rows.iter().collect::<Vec<_>>();
                self.commit_rows(&inner_plan, outer_plan, &rows)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{
        compile_commitment_request, BackendKindId, CommitmentRequestCapabilities, CommitmentSource,
        ComputeBackendSetup, DenseType, PolynomialType,
    };
    use crate::{AkitaProverSetup, DensePoly};
    use akita_types::SetupMatrixCapacity;
    use jolt_field::{Prime128Offset275, Ring};

    type F = Prime128Offset275;

    struct CpuKind;

    #[test]
    fn object_safe_cpu_inner_stage_retains_and_exports_exact_rows() {
        let plan = CommitInnerPlan {
            ring_dimension: 64,
            num_live_blocks: 1,
            n_a: 2,
            num_positions_per_block: 8,
            num_digits_inner: 1,
            log_basis_inner: 1,
        };
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            9,
            1,
            SetupMatrixCapacity {
                num_field_elements: 2 * 8 * 64,
            },
        )
        .unwrap();
        let backend = CpuBackend::DEFAULT;
        let prepared = backend.prepare_setup(&setup).unwrap();
        let poly = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
        let sources: [&dyn CommitmentSource<F>; 1] = [&poly];
        let resolved = compile_commitment_request(
            &plan,
            &sources,
            &CommitmentRequestCapabilities::split::<()>(
                BackendKindId::of::<CpuKind>("cpu").unwrap(),
                vec![PolynomialType::Dense(DenseType::Coefficients)],
            ),
        )
        .unwrap()
        .materialize()
        .unwrap();
        let operation = CpuInnerCommitOperation::new(&backend, &prepared);
        let binding = crate::compute::commitment::CommitmentStateBinding::new(
            setup.expanded.descriptor.clone(),
            plan,
            1,
            None,
        )
        .unwrap();
        let trait_object: &dyn InnerCommitOperation<F> = &operation;
        let output = trait_object
            .commit_inner(&binding, &plan, &resolved)
            .unwrap();
        let rows = operation.export_inner_rows(&plan, output.image()).unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ring_dim(), 64);
        assert_eq!(rows[0].coeff_len(), 2 * 64);
        let resident_pointer = operation
            .owner()
            .value::<Vec<CommitInnerWitness<F>>>(output.image())
            .unwrap()[0]
            .inner_rows
            .coeffs()
            .as_ptr();
        let moved = operation
            .consume_inner_rows(&plan, output.into_image())
            .unwrap();
        assert_eq!(moved[0].coeffs().as_ptr(), resident_pointer);
    }
}
