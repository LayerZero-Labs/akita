//! Contract test for downstream-style custom root commit sources.
//!
//! Proves that the unified explicit-parameter `commit` accepts a polynomial
//! type that is not one of Akita's built-in root representations, with a
//! downstream-owned backend implementing the root commit capability
//! for local views (orphan-rule-safe: the backend type is local to this test
//! crate).

#![allow(missing_docs)]

use akita_algebra::CyclotomicRing;
use akita_config::proof_optimized::fp64;
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_prover::backend::DenseView;
use akita_prover::compute::{
    CommitInnerPlan, CompressionComputeBackend, CompressionRowsProducts, ComputeBackendSetup,
    DigitRowsComputeBackend, RootCommitKernel, RootCommitSource, RootPolyShape,
};
use akita_prover::{
    AkitaProverSetup, CpuBackend, CpuPreparedSetup, DensePoly, GroupContext, UniformProverStack,
};
use akita_types::{CommittedSourceEncoding, NttCacheKey, OpeningClaimsLayout};
use jolt_field::Unreduced;
use jolt_field::{CanonicalEncoding, Field, Ring};
use std::sync::atomic::{AtomicUsize, Ordering};

type Cfg = fp64::Dense;
type F = <Cfg as CommitmentConfig>::Field;
// The folded-only protocol requires at least two folds. `nv=8` was a
// root-direct fixture; `nv=14` is the first supported adaptive fp64 singleton.
const CONTRACT_NUM_VARS: usize = 14;
static COMMIT_KERNEL_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Downstream-like root polynomial: not `DensePoly`, `OneHotPoly`, etc.
///
/// D-free storage; the commit source impls are generic over every runtime
/// ring dimension, matching the `Runtime*` capability bounds on the D-free
/// commit entry points.
#[derive(Debug, Clone)]
struct ContractRootPoly {
    num_vars: usize,
    dense: DensePoly<F>,
}

impl ContractRootPoly {
    fn from_field_evals(num_vars: usize, evals: &[F]) -> Result<Self, AkitaError> {
        Ok(Self {
            num_vars,
            dense: DensePoly::<F>::from_field_evals(num_vars, evals)?,
        })
    }
}

/// Local commit view owned by the downstream test crate.
#[derive(Debug, Clone, Copy)]
struct ContractCommitView<'a> {
    poly: &'a ContractRootPoly,
}

impl<const DD: usize> RootPolyShape<F, DD> for ContractRootPoly {
    fn num_ring_elems(&self) -> usize {
        RootPolyShape::<F, DD>::num_ring_elems(&self.dense)
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl akita_prover::RootPolyMeta<F> for ContractRootPoly {
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<const DD: usize> RootCommitSource<F, DD> for ContractRootPoly {
    type CommitView<'a>
        = ContractCommitView<'a>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        Ok(ContractCommitView { poly: self })
    }

    /// A downstream source must answer the bounded-commitment question too; this
    /// one wraps a dense poly, so it delegates to the dense scan.
    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError> {
        RootCommitSource::<F, DD>::committed_centered_reach(
            &self.dense,
            modulus,
            centering_threshold,
        )
    }
}

/// Downstream-owned backend: delegates row work to [`CpuBackend`] but carries
/// the [`RootCommitKernel`] impl for [`ContractCommitView`] in this crate.
#[derive(Debug, Default, Clone, Copy)]
struct ContractCommitBackend;

impl<F> ComputeBackendSetup<F> for ContractCommitBackend
where
    F: Field + CanonicalEncoding,
{
    type PreparedSetup = CpuPreparedSetup<F>;

    fn prepare_expanded(
        &self,
        expanded: std::sync::Arc<akita_types::AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        CpuBackend::DEFAULT.prepare_expanded(expanded)
    }

    fn ensure_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError> {
        CpuBackend::DEFAULT.ensure_ntt_slot(prepared, key)
    }

    fn prepared_expanded_setup<'a>(
        &self,
        prepared: &'a Self::PreparedSetup,
    ) -> &'a akita_types::AkitaExpandedSetup<F> {
        CpuBackend::DEFAULT.prepared_expanded_setup(prepared)
    }
}

impl<F> DigitRowsComputeBackend<F> for ContractCommitBackend
where
    F: Field + CanonicalEncoding,
{
    fn digit_rows<const RING_D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digit_vectors: &[&[[i8; RING_D]]],
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, RING_D>>>, AkitaError> {
        CpuBackend::DEFAULT.digit_rows(prepared, row_len, digit_vectors, log_basis)
    }
}

impl<F> CompressionComputeBackend<F> for ContractCommitBackend
where
    F: Field + CanonicalEncoding,
{
    fn compression_cache_bytes(&self, prepared: &Self::PreparedSetup) -> Option<usize> {
        CpuBackend::DEFAULT.compression_cache_bytes(prepared)
    }

    fn compression_rows_products<const RING_D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; RING_D]]],
    ) -> Result<Vec<CompressionRowsProducts<F, RING_D>>, AkitaError> {
        CpuBackend::DEFAULT.compression_rows_products(prepared, digit_vectors)
    }

    fn compression_negacyclic_rows<const RING_D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        digit_vectors: &[&[[i8; RING_D]]],
    ) -> Result<Vec<Vec<CyclotomicRing<F, RING_D>>>, AkitaError> {
        CpuBackend::DEFAULT.compression_negacyclic_rows(prepared, digit_vectors)
    }
}

impl<const DD: usize> RootCommitKernel<ContractCommitView<'_>, F, DD> for ContractCommitBackend
where
    F: Field + CanonicalEncoding + Ring + Unreduced,
    <F as Unreduced>::Wide: From<F>,
{
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<ContractCommitView<'_>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<akita_prover::CommitInnerWitness<F>>, AkitaError> {
        COMMIT_KERNEL_CALLS.fetch_add(1, Ordering::Relaxed);
        let dense_sources = sources
            .into_iter()
            .map(|source| RootCommitSource::<F, DD>::commit_view(&source.poly.dense))
            .collect::<Result<Vec<_>, _>>()?;
        <CpuBackend as RootCommitKernel<DenseView<'_, F, DD>, F, DD>>::commit_inner_group(
            &CpuBackend::DEFAULT,
            prepared,
            dense_sources,
            plan,
        )
    }
}

#[test]
fn custom_commit_source_runs_unified_explicit_commit() {
    COMMIT_KERNEL_CALLS.store(0, Ordering::Relaxed);
    let len = 1usize << CONTRACT_NUM_VARS;
    let evals: Vec<F> = (0..len).map(|idx| F::from_u64((idx as u64) + 1)).collect();
    let contract =
        ContractRootPoly::from_field_evals(CONTRACT_NUM_VARS, &evals).expect("contract poly");
    let schedules = akita_config::test_support::workspace_schedule_catalog::<Cfg>()
        .expect("workspace schedule catalog");
    let dense = DensePoly::<F>::from_field_evals(CONTRACT_NUM_VARS, &evals).expect("dense oracle");
    let opening_batch = OpeningClaimsLayout::new(CONTRACT_NUM_VARS, 1).expect("opening batch");
    let key = akita_types::AkitaScheduleLookupKey::single(
        opening_batch
            .root_final_group_layout()
            .expect("root group layout"),
    );
    let params = schedules
        .resolve_key(&key)
        .map(|row| row.schedule().root.params.clone())
        .expect("layout");
    assert_eq!(
        params.source_encoding,
        CommittedSourceEncoding::CanonicalCoefficientTable,
        "the selected packing root must exercise the canonical commit capability"
    );

    let setup_envelope =
        akita_config::SetupRequirements::from_catalog::<Cfg>(&schedules, CONTRACT_NUM_VARS, 1)
            .map(|requirements| requirements.matrix_capacity)
            .expect("envelope");
    let setup = AkitaProverSetup::<F>::generate_with_capacity(CONTRACT_NUM_VARS, 1, setup_envelope)
        .expect("setup");
    let contract_backend = ContractCommitBackend;
    let prepared = contract_backend.prepare_setup(&setup).expect("prepared");
    let expanded = setup.expanded.as_ref();
    let contract_stack = UniformProverStack::uniform(&contract_backend, &prepared, expanded)
        .expect("contract stack");

    let contract_output = akita_prover::commit::<Cfg, ContractRootPoly, _>(
        std::slice::from_ref(&contract),
        expanded,
        &schedules,
        &contract_stack,
        GroupContext::explicit(&params.own_group().profile),
    )
    .expect("contract commit");

    let cpu_prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("cpu prepared");
    let cpu_stack = UniformProverStack::uniform(&CpuBackend::DEFAULT, &cpu_prepared, expanded)
        .expect("cpu stack");
    let dense_output = akita_prover::commit::<Cfg, DensePoly<F>, CpuBackend>(
        std::slice::from_ref(&dense),
        expanded,
        &schedules,
        &cpu_stack,
        GroupContext::explicit(&params.own_group().profile),
    )
    .expect("dense oracle commit");

    assert_eq!(
        contract_output.committed_group,
        dense_output.committed_group
    );
    assert_eq!(contract_output.hint, dense_output.hint);
    assert_eq!(COMMIT_KERNEL_CALLS.load(Ordering::Relaxed), 1);

    let mut malformed_profile = params.own_group().profile;
    malformed_profile.inner.digits.num_digits += 1;
    let error = akita_prover::commit::<Cfg, ContractRootPoly, _>(
        std::slice::from_ref(&contract),
        expanded,
        &schedules,
        &contract_stack,
        GroupContext::explicit(&malformed_profile),
    )
    .expect_err("malformed explicit profile must reject before arithmetic");
    assert!(matches!(error, AkitaError::InvalidSetup(_)));
    assert_eq!(COMMIT_KERNEL_CALLS.load(Ordering::Relaxed), 1);
}
