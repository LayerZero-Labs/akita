//! Contract test for downstream-style custom root commit sources.
//!
//! Proves that `commit_with_params` accepts a polynomial type that is not one
//! of Akita's built-in root representations, with only [`RootCommitSource`] on
//! `P` and a downstream-owned backend implementing [`RootCommitKernel`] for a
//! local commit view (orphan-rule-safe: the backend type is local to this test
//! crate).

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
use akita_types::OpeningBatchShape;

type Cfg = fp64::D32Full;
type F = <Cfg as CommitmentConfig>::Field;
const D: usize = Cfg::D;

/// Downstream-like root polynomial: not `DensePoly`, `OneHotPoly`, etc.
#[derive(Debug, Clone)]
struct ContractRootPoly {
    num_vars: usize,
    coeffs: Vec<CyclotomicRing<F, D>>,
}

impl ContractRootPoly {
    fn from_field_evals(num_vars: usize, evals: &[F]) -> Result<Self, AkitaError> {
        Ok(Self {
            num_vars,
            coeffs: DensePoly::<F, D>::from_field_evals(num_vars, evals)?.coeffs,
        })
    }
}

/// Local commit view owned by the downstream test crate.
#[derive(Debug, Clone, Copy)]
struct ContractCommitView<'a> {
    poly: &'a ContractRootPoly,
}

impl RootPolyShape<F, D> for ContractRootPoly {
    fn num_ring_elems(&self) -> usize {
        self.coeffs.len()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl RootCommitSource<F, D> for ContractRootPoly {
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
    type PreparedSetup<const RING_D: usize> = CpuPreparedSetup<F, RING_D>;

    fn prepare_expanded<const RING_D: usize>(
        &self,
        expanded: std::sync::Arc<akita_types::AkitaExpandedSetup<F>>,
    ) -> Result<Self::PreparedSetup<RING_D>, AkitaError> {
        CpuBackend.prepare_expanded(expanded)
    }

    fn prepared_expanded_setup<'a, const RING_D: usize>(
        &self,
        prepared: &'a Self::PreparedSetup<RING_D>,
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
        prepared: &Self::PreparedSetup<RING_D>,
        row_len: usize,
        digits: &[[i8; RING_D]],
        log_basis: u32,
    ) -> Result<Vec<CyclotomicRing<F, RING_D>>, AkitaError> {
        CpuBackend.digit_rows(prepared, row_len, digits, log_basis)
    }
}

impl RootCommitKernel<ContractCommitView<'_>, F, D> for ContractCommitBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup<D>,
        source: ContractCommitView<'_>,
        plan: CommitInnerPlan,
    ) -> Result<akita_prover::CommitInnerWitness<F, D>, AkitaError> {
        let dense = DensePoly::<F, D>::from_ring_coeffs(source.poly.coeffs.clone());
        RootCommitKernel::<DenseView<'_, F, D>, F, D>::commit_inner(
            &CpuBackend,
            prepared,
            dense.commit_view()?,
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
    const NUM_VARS: usize = 8;
    let len = 1usize << NUM_VARS;
    let evals: Vec<F> = (0..len).map(|idx| F::from_u64((idx as u64) + 1)).collect();
    let contract = ContractRootPoly::from_field_evals(NUM_VARS, &evals).expect("contract poly");
    assert_commit_source_only(&contract);

    let dense = DensePoly::<F, D>::from_field_evals(NUM_VARS, &evals).expect("dense oracle");
    let opening_batch = OpeningBatchShape::new(NUM_VARS, 1).expect("opening batch");
    let params = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");

    let setup_envelope = Cfg::max_setup_matrix_size(NUM_VARS, 1).expect("envelope");
    let setup = AkitaProverSetup::<F, D>::generate_with_capacity(NUM_VARS, 1, setup_envelope)
        .expect("setup");
    let prepared = ContractCommitBackend
        .prepare_setup(&setup)
        .expect("prepared");
    let expanded = setup.expanded.as_ref();
    let contract_ctx =
        OperationCtx::new(&ContractCommitBackend, &prepared, expanded).expect("contract ctx");

    let (contract_commitment, contract_hint) = commit_with_params::<F, D, ContractRootPoly, _>(
        std::slice::from_ref(&contract),
        expanded,
        &contract_ctx,
        &params,
    )
    .expect("contract commit");

    let cpu_prepared = CpuBackend.prepare_setup(&setup).expect("cpu prepared");
    let cpu_ctx = OperationCtx::new(&CpuBackend, &cpu_prepared, expanded).expect("cpu ctx");
    let (dense_commitment, dense_hint) = commit_with_params::<F, D, DensePoly<F, D>, CpuBackend>(
        std::slice::from_ref(&dense),
        expanded,
        &cpu_ctx,
        &params,
    )
    .expect("dense oracle commit");

    assert_eq!(contract_commitment, dense_commitment);
    assert_eq!(
        contract_hint.decomposed_inner_rows,
        dense_hint.decomposed_inner_rows
    );
}

#[test]
fn custom_commit_source_runs_batched_commit_with_params() {
    const NUM_VARS: usize = 8;
    let len = 1usize << NUM_VARS;
    let evals: Vec<F> = (0..len).map(|idx| F::from_u64((idx as u64) + 1)).collect();
    let contract = ContractRootPoly::from_field_evals(NUM_VARS, &evals).expect("contract poly");
    let dense = DensePoly::<F, D>::from_field_evals(NUM_VARS, &evals).expect("dense oracle");
    let opening_batch = OpeningBatchShape::new(NUM_VARS, 1).expect("opening batch");
    let params = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");

    let setup_envelope = Cfg::max_setup_matrix_size(NUM_VARS, 1).expect("envelope");
    let setup = AkitaProverSetup::<F, D>::generate_with_capacity(NUM_VARS, 1, setup_envelope)
        .expect("setup");
    let prepared = ContractCommitBackend
        .prepare_setup(&setup)
        .expect("prepared");
    let expanded = setup.expanded.as_ref();
    let contract_ctx =
        OperationCtx::new(&ContractCommitBackend, &prepared, expanded).expect("contract ctx");

    let (contract_commitment, contract_hint) =
        batched_commit_with_params::<F, D, ContractRootPoly, ContractCommitBackend>(
            std::slice::from_ref(&contract),
            expanded,
            &contract_ctx,
            &params,
        )
        .expect("contract batched commit");

    let cpu_prepared = CpuBackend.prepare_setup(&setup).expect("cpu prepared");
    let cpu_ctx = OperationCtx::new(&CpuBackend, &cpu_prepared, expanded).expect("cpu ctx");
    let (dense_commitment, dense_hint) =
        batched_commit_with_params::<F, D, DensePoly<F, D>, CpuBackend>(
            std::slice::from_ref(&dense),
            expanded,
            &cpu_ctx,
            &params,
        )
        .expect("dense batched commit");

    assert_eq!(contract_commitment, dense_commitment);
    assert_eq!(
        contract_hint.decomposed_inner_rows,
        dense_hint.decomposed_inner_rows
    );
}
