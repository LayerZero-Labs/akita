//! Contract test for downstream-style custom root commit sources.
//!
//! Proves that `batched_commit_with_params` accepts a polynomial type that is
//! not one of Akita's built-in root representations, with only
//! [`RootCommitSource`] on `P` and a downstream-owned backend implementing
//! [`RootCommitKernel`] for a local commit view (orphan-rule-safe: the backend
//! type is local to this test crate).

#![cfg(feature = "schedules-default")]
#![allow(missing_docs)]

use akita_algebra::CyclotomicRing;
use akita_config::proof_optimized::fp64;
use akita_config::CommitmentConfig;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_prover::backend::DenseView;
use akita_prover::compute::{
    CommitInnerPlan, ComputeBackendSetup, DigitRowsComputeBackend, OperationCtx, RootCommitKernel,
    RootCommitSource, RootPolyShape,
};
use akita_prover::{
    batched_commit_with_params, commit_with_params, AkitaProverSetup, CpuBackend, CpuPreparedSetup,
    DensePoly,
};
use akita_types::{NttCacheKey, OpeningClaimsLayout};

type Cfg = fp64::D128Dense;
type F = <Cfg as CommitmentConfig>::Field;
const D: usize = Cfg::D;
// The folded-only protocol requires at least two folds. `nv=8` was a
// root-direct fixture; `nv=14` is the first supported fp64 D128 singleton.
const CONTRACT_NUM_VARS: usize = 14;

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
            dense: DensePoly::<F>::from_field_evals(num_vars, D, evals)?,
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
    fn num_ring_elems(&self) -> usize {
        akita_prover::RootPolyMeta::num_ring_elems(&self.dense)
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        None
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
}

/// Downstream-owned backend: delegates row work to [`CpuBackend`] but carries
/// the [`RootCommitKernel`] impl for [`ContractCommitView`] in this crate.
#[derive(Debug, Default, Clone, Copy)]
struct ContractCommitBackend;

impl<F> ComputeBackendSetup<F> for ContractCommitBackend
where
    F: FieldCore + CanonicalField,
{
    type PreparedSetup = CpuPreparedSetup<F>;

    fn prepare_expanded<const RING_D: usize>(
        &self,
        expanded: std::sync::Arc<akita_types::AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup, AkitaError> {
        CpuBackend.prepare_expanded::<RING_D>(expanded)
    }

    fn ensure_ntt_slot(
        &self,
        prepared: &Self::PreparedSetup,
        key: NttCacheKey,
    ) -> Result<(), AkitaError> {
        CpuBackend.ensure_ntt_slot(prepared, key)
    }

    fn prepared_expanded_setup<'a>(
        &self,
        prepared: &'a Self::PreparedSetup,
    ) -> &'a akita_types::AkitaExpandedSetup<F> {
        CpuBackend.prepared_expanded_setup(prepared)
    }
}

impl<F> DigitRowsComputeBackend<F> for ContractCommitBackend
where
    F: FieldCore + CanonicalField,
{
    fn digit_rows<const RING_D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        row_len: usize,
        digits: &[[i8; RING_D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, RING_D>>, AkitaError> {
        CpuBackend.digit_rows(prepared, row_len, digits, log_basis)
    }
}

impl<const DD: usize> RootCommitKernel<ContractCommitView<'_>, F, DD> for ContractCommitBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup,
        source: ContractCommitView<'_>,
        plan: CommitInnerPlan,
    ) -> Result<akita_prover::CommitInnerWitness<F>, AkitaError> {
        RootCommitKernel::<DenseView<'_, F, DD>, F, DD>::commit_inner(
            &CpuBackend,
            prepared,
            RootCommitSource::<F, DD>::commit_view(&source.poly.dense)?,
            plan,
        )
    }
}

fn assert_commit_source_only<P>(_poly: &P)
where
    P: RootCommitSource<F, D>,
{
}

#[test]
fn custom_commit_source_runs_commit_with_params() {
    let len = 1usize << CONTRACT_NUM_VARS;
    let evals: Vec<F> = (0..len).map(|idx| F::from_u64((idx as u64) + 1)).collect();
    let contract =
        ContractRootPoly::from_field_evals(CONTRACT_NUM_VARS, &evals).expect("contract poly");
    assert_commit_source_only(&contract);

    let dense =
        DensePoly::<F>::from_field_evals(CONTRACT_NUM_VARS, D, &evals).expect("dense oracle");
    let opening_batch = OpeningClaimsLayout::new(CONTRACT_NUM_VARS, 1).expect("opening batch");
    let params = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");

    let setup_envelope = Cfg::max_setup_matrix_size(CONTRACT_NUM_VARS, 1).expect("envelope");
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(CONTRACT_NUM_VARS, 1, D, setup_envelope)
            .expect("setup");
    let prepared = ContractCommitBackend
        .prepare_setup(&setup)
        .expect("prepared");
    let expanded = setup.expanded.as_ref();
    let contract_ctx =
        OperationCtx::new(&ContractCommitBackend, &prepared, expanded).expect("contract ctx");

    let (contract_commitment, contract_hint) = commit_with_params::<F, ContractRootPoly, _>(
        std::slice::from_ref(&contract),
        expanded,
        &contract_ctx,
        &params,
    )
    .expect("contract commit");

    let cpu_prepared = CpuBackend.prepare_setup(&setup).expect("cpu prepared");
    let cpu_ctx = OperationCtx::new(&CpuBackend, &cpu_prepared, expanded).expect("cpu ctx");
    let (dense_commitment, dense_hint) = commit_with_params::<F, DensePoly<F>, CpuBackend>(
        std::slice::from_ref(&dense),
        expanded,
        &cpu_ctx,
        &params,
    )
    .expect("dense oracle commit");

    assert_eq!(
        contract_commitment.rows().count(),
        dense_commitment.rows().count()
    );
    assert_eq!(
        contract_hint.decomposed_inner_rows,
        dense_hint.decomposed_inner_rows
    );
}

#[test]
fn custom_commit_source_runs_batched_commit_with_params() {
    let len = 1usize << CONTRACT_NUM_VARS;
    let evals: Vec<F> = (0..len).map(|idx| F::from_u64((idx as u64) + 1)).collect();
    let contract =
        ContractRootPoly::from_field_evals(CONTRACT_NUM_VARS, &evals).expect("contract poly");
    assert_commit_source_only(&contract);
    let dense =
        DensePoly::<F>::from_field_evals(CONTRACT_NUM_VARS, D, &evals).expect("dense oracle");
    let opening_batch = OpeningClaimsLayout::new(CONTRACT_NUM_VARS, 1).expect("opening batch");
    let params = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");

    let setup_envelope = Cfg::max_setup_matrix_size(CONTRACT_NUM_VARS, 1).expect("envelope");
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(CONTRACT_NUM_VARS, 1, D, setup_envelope)
            .expect("setup");
    let prepared = ContractCommitBackend
        .prepare_setup(&setup)
        .expect("prepared");
    let expanded = setup.expanded.as_ref();
    let contract_ctx =
        OperationCtx::new(&ContractCommitBackend, &prepared, expanded).expect("contract ctx");

    let (contract_commitment, contract_hint) =
        batched_commit_with_params::<F, ContractRootPoly, ContractCommitBackend>(
            std::slice::from_ref(&contract),
            expanded,
            &contract_ctx,
            &params,
        )
        .expect("contract batched commit");

    let cpu_prepared = CpuBackend.prepare_setup(&setup).expect("cpu prepared");
    let cpu_ctx = OperationCtx::new(&CpuBackend, &cpu_prepared, expanded).expect("cpu ctx");
    let (dense_commitment, dense_hint) = batched_commit_with_params::<F, DensePoly<F>, CpuBackend>(
        std::slice::from_ref(&dense),
        expanded,
        &cpu_ctx,
        &params,
    )
    .expect("dense batched commit");

    assert_eq!(
        contract_commitment.rows().count(),
        dense_commitment.rows().count()
    );
    assert_eq!(
        contract_hint.decomposed_inner_rows,
        dense_hint.decomposed_inner_rows
    );
}
