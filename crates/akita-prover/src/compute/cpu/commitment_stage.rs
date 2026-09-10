use super::{CpuBackend, CpuPreparedSetup};
use crate::commitment::{
    for_each_outer_slice_input, BackendStateRef, CommitmentStateBinding, InnerCommitOperation,
    InnerCommitOutput, InnerImage, InnerImageExportOperation, InnerImageInput,
    OuterCommitOperation, OuterCommitPlan, ResolvedCommitSource, StateOwnerCapability,
};
use crate::compute::{CommitInnerPlan, DigitRowsComputeBackend};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::CommitInnerWitness;
use akita_algebra::ring::CyclotomicRing;
use akita_error::{checked, AkitaError};
use akita_types::{dispatch_for_field, DigitBlocks, RingVec};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field, Unreduced, WithCommitAccumulator};
use std::mem::size_of;
use std::sync::Arc;

fn compute_outer_commitment_from_rows<F, B, const D_A: usize, const D_B: usize>(
    backend: &B,
    prepared: &B::PreparedSetup,
    inner_rows: &[&RingVec<F>],
    inner_plan: &CommitInnerPlan,
    outer_plan: &OuterCommitPlan,
) -> Result<RingVec<F>, AkitaError>
where
    F: Field + CanonicalEncoding,
    B: DigitRowsComputeBackend<F>,
{
    if inner_plan.ring_dimension != D_A || outer_plan.ring_dimension() != D_B {
        return Err(AkitaError::InvalidSetup(
            "commitment stage plan ring dimensions disagree with dispatch".into(),
        ));
    }
    if outer_plan.geometry().num_polynomials() != inner_rows.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} inner commitments for {} sources",
            inner_rows.len(),
            outer_plan.geometry().num_polynomials()
        )));
    }
    let expected_rows = inner_plan
        .num_live_blocks
        .checked_mul(inner_plan.n_a)
        .ok_or_else(|| AkitaError::InvalidSetup("inner commitment row count overflow".into()))?;
    let prepared_polynomials = cfg_into_iter!(inner_rows)
        .map(|rows| -> Result<DigitBlocks, AkitaError> {
            if rows.ring_dim() != D_A || rows.count() != expected_rows {
                return Err(AkitaError::InvalidSetup(
                    "resident inner commitment row shape is invalid".into(),
                ));
            }
            let typed = rows.as_ring_slice::<D_A>().map_err(|_| {
                AkitaError::InvalidSetup("resident inner commitment ring storage is invalid".into())
            })?;
            let blocks = typed.chunks_exact(inner_plan.n_a).collect::<Vec<_>>();
            if blocks.len() != inner_plan.num_live_blocks {
                return Err(AkitaError::InvalidSetup(
                    "resident inner commitment block geometry is invalid".into(),
                ));
            }
            decompose_commit_blocks_into::<F, D_A, D_B>(
                &blocks,
                outer_plan.num_digits_outer(),
                outer_plan.log_basis_outer(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let typed_u = commit_outer_slices::<F, _, D_B>(
        backend,
        prepared,
        outer_plan.n_b(),
        prepared_polynomials.iter(),
        outer_plan.geometry(),
        outer_plan.log_basis_outer(),
    )?;
    let u = RingVec::from_ring_elems(&typed_u);
    let expected_coefficients = outer_plan.output_coefficient_len()?;
    if u.coeff_len() != expected_coefficients {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {} outer commitment coefficients, expected {expected_coefficients}",
            u.coeff_len()
        )));
    }
    Ok(u)
}

fn commit_outer_slices<'a, F, B, const D_B: usize>(
    backend: &B,
    prepared: &B::PreparedSetup,
    n_b: usize,
    polynomial_digits: impl IntoIterator<Item = &'a DigitBlocks>,
    geometry: &akita_types::CommitmentSliceGeometry,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D_B>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    B: DigitRowsComputeBackend<F>,
{
    let per_block = geometry.ring_elements_per_block_per_polynomial();
    let num_live_blocks = geometry
        .block_ranges()
        .last()
        .map(|range| range.end)
        .ok_or_else(|| AkitaError::InvalidSetup("B commitment has no slices".into()))?;
    let polynomial_planes = polynomial_digits
        .into_iter()
        .map(|digits| {
            if digits.block_count() != num_live_blocks
                || digits.block_sizes().iter().any(|&size| size != per_block)
            {
                return Err(AkitaError::InvalidSetup(
                    "B slice input does not match the frozen block geometry".into(),
                ));
            }
            digits.typed_planes::<D_B>()
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let mut inputs = Vec::with_capacity(geometry.slice_count().get());
    for_each_outer_slice_input::<D_B>(polynomial_planes, geometry, |input| {
        inputs.push(input.to_vec());
        Ok(())
    })?;
    let input_refs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let row_batches = backend.digit_rows::<D_B>(prepared, n_b, &input_refs, log_basis)?;
    if row_batches.len() != input_refs.len() || row_batches.iter().any(|rows| rows.len() != n_b) {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned B commitment row shape {:?}, expected {} batches of {n_b} rows",
            row_batches.iter().map(Vec::len).collect::<Vec<_>>(),
            input_refs.len(),
        )));
    }
    let mut stacked = Vec::with_capacity(geometry.logical_output_rows(n_b)?);
    stacked.extend(row_batches.into_iter().flatten());
    Ok(stacked)
}

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
        image: &crate::commitment::BackendStateRef<InnerImage>,
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
                |D_B| compute_outer_commitment_from_rows::<F, CpuBackend, D_A, D_B>(
                    self.backend,
                    self.prepared,
                    rows,
                    inner_plan,
                    outer_plan,
                )
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
        plan: &crate::commitment::UncompressedCommitPlan,
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
    use crate::commitment::{
        compile_commitment_request, BackendKindId, CommitmentRequestCapabilities, CommitmentSource,
        DenseType, PolynomialType,
    };
    use crate::compute::ComputeBackendSetup;
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
        let binding = crate::commitment::CommitmentStateBinding::new(
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
