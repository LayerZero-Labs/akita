//! Parity and fallback tests for runtime-installed commitment stage providers.
//!
//! The fake provider re-wraps the CPU reference inner and outer operations of
//! a second backend under fresh owners and a distinct backend kind, so every
//! accelerated route crosses the same boundaries a device provider would.

use super::{CommitOutput, CpuBackend, CpuInnerCommitOperation, CpuOuterCommitOperation};
use crate::commitment::{
    AvailablePolynomialTypes, BackendInstanceId, BackendKindId, CommitSourceDescriptor,
    CommitmentExecutionPlan, CommitmentOperationContext, CommitmentRequestCapabilities,
    CommitmentSource, CommitmentStageProvider, CommitmentStages, CommitmentStateBinding,
    GroupContext, InnerCommitOperation, InnerCommitOutput, InnerImageInput, OneHotIndexWidth,
    OneHotType, OuterCommitOperation, PolynomialRepresentation, PolynomialType,
    PolynomialTypeSelection, PreparedCommitmentResources, PreparedInnerCommitment,
    PreparedOuterCommitment, ResolvedCommitSource, StageDimensionCapabilities, StageResources,
    StateOwnerCapability, UncompressedCommitPlan,
};
use crate::opaque::CommitInnerPlan;
use crate::{AkitaProverSetup, DensePoly, OneHotPoly};
use akita_config::proof_optimized::fp128;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_prover::SelectedProverOpeningData;
use akita_types::{
    AkitaExpandedSetup, AkitaScheduleLookupKey, BasisMode, GroupCommitPhaseParams, OpeningClaims,
    PolynomialGroupClaims, PolynomialGroupLayout, RingRole, RingVec,
};
use jolt_field::Ring;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type F = fp128::Field;
type E = <fp128::Dense as CommitmentConfig>::ExtField;

const NUM_VARS: usize = 16;

struct DelegatingKind;
struct DelegatingContext;

/// How the fake provider declines a request, if at all.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Decline {
    Never,
    InnerPlan,
    OuterPlan,
    SourceType,
}

struct DelegatingInner<'a> {
    cpu: CpuInnerCommitOperation<'a, F, E>,
    accepts: bool,
    calls: &'a AtomicUsize,
}

impl InnerCommitOperation<F> for DelegatingInner<'_> {
    fn supports_plan(&self, _plan: &CommitInnerPlan) -> bool {
        self.accepts
    }

    fn commit_inner(
        &self,
        binding: &CommitmentStateBinding,
        plan: &CommitInnerPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<InnerCommitOutput, AkitaError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.cpu.commit_inner(binding, plan, sources)
    }
}

struct DelegatingOuter<'a> {
    cpu: CpuOuterCommitOperation<'a, F, E>,
    accepts: bool,
    calls: &'a AtomicUsize,
}

impl OuterCommitOperation<F> for DelegatingOuter<'_> {
    fn supports_plan(&self, _plan: &UncompressedCommitPlan) -> bool {
        self.accepts
    }

    fn commit_outer(
        &self,
        plan: &UncompressedCommitPlan,
        inner: InnerImageInput<'_, F>,
    ) -> Result<RingVec<F>, AkitaError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.cpu.commit_outer(plan, inner)
    }
}

/// Provider whose stages delegate to a second CPU backend over the same setup.
struct DelegatingProvider {
    backend: CpuBackend<F, E>,
    inner: bool,
    outer: bool,
    decline: Decline,
    inner_calls: AtomicUsize,
    outer_calls: AtomicUsize,
}

impl DelegatingProvider {
    fn new(
        expanded: Arc<AkitaExpandedSetup<F>>,
        inner: bool,
        outer: bool,
        decline: Decline,
    ) -> Self {
        Self {
            backend: CpuBackend::new(expanded).unwrap(),
            inner,
            outer,
            decline,
            inner_calls: AtomicUsize::new(0),
            outer_calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> (usize, usize) {
        (
            self.inner_calls.load(Ordering::SeqCst),
            self.outer_calls.load(Ordering::SeqCst),
        )
    }
}

impl CommitmentStageProvider<F> for DelegatingProvider {
    fn stages<'a>(
        &'a self,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<CommitmentStages<'a, F>, AkitaError> {
        let prepared = self.backend.prepared()?;
        let cpu_inner = CpuInnerCommitOperation::new(&self.backend, prepared);
        let owner = cpu_inner.owner().clone();
        let exporter = cpu_inner.portable_exporter();
        let instance = BackendInstanceId::issue();
        let resources = StageResources::controlled(PreparedCommitmentResources::new(
            &self.backend,
            prepared,
            expanded,
        )?);
        let outer = if self.outer {
            // Alone, the outer stage must receive exported host rows.
            let outer_owner = if self.inner {
                owner.clone()
            } else {
                StateOwnerCapability::new()
            };
            Some(PreparedOuterCommitment::new(
                Arc::new(DelegatingOuter {
                    cpu: CpuOuterCommitOperation::new(&self.backend, prepared, &cpu_inner),
                    accepts: self.decline != Decline::OuterPlan,
                    calls: &self.outer_calls,
                }),
                outer_owner,
                CommitmentOperationContext::new(
                    expanded.descriptor(),
                    instance,
                    "delegating-outer",
                    resources.clone(),
                )?,
                StageDimensionCapabilities::cpu_role::<F>(RingRole::Outer),
            ))
        } else {
            None
        };
        let inner = if self.inner {
            let mut capabilities = if self.decline == Decline::SourceType {
                CommitmentRequestCapabilities::split::<DelegatingContext>(
                    BackendKindId::of::<DelegatingKind>("delegating")?,
                    vec![PolynomialType::OneHot(OneHotType::new(
                        3,
                        OneHotIndexWidth::U16,
                    )?)],
                )
            } else {
                CommitmentRequestCapabilities::split::<DelegatingContext>(
                    BackendKindId::of::<DelegatingKind>("delegating")?,
                    Vec::new(),
                )
            };
            if self.decline != Decline::SourceType {
                capabilities.accept_any_standard_type();
            }
            Some(PreparedInnerCommitment::new(
                Arc::new(DelegatingInner {
                    cpu: cpu_inner,
                    accepts: self.decline != Decline::InnerPlan,
                    calls: &self.inner_calls,
                }),
                owner,
                CommitmentOperationContext::new(
                    expanded.descriptor(),
                    instance,
                    "delegating-inner",
                    resources,
                )?,
                capabilities,
                StageDimensionCapabilities::cpu_role::<F>(RingRole::Inner),
                Some(exporter),
            )?)
        } else {
            None
        };
        Ok(CommitmentStages { inner, outer })
    }
}

/// Dense source that counts representation materializations.
struct CountingSource<'a> {
    poly: &'a DensePoly<F>,
    materializations: AtomicUsize,
}

impl CommitmentSource<F> for CountingSource<'_> {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        self.poly.descriptor()
    }

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError> {
        self.poly
            .committed_centered_reach(modulus, centering_threshold)
    }

    fn available_polynomial_types(
        &self,
        plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        self.poly.available_polynomial_types(plan)
    }

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        self.materializations.fetch_add(1, Ordering::SeqCst);
        self.poly.represent_as(selected, plan)
    }
}

fn catalog<Cfg>() -> TrustedScheduleCatalog<Cfg>
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap()
}

fn setup<Cfg>(catalog: &TrustedScheduleCatalog<Cfg>) -> AkitaProverSetup<F>
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
{
    let capacity = akita_config::SetupRequirements::from_catalog::<Cfg>(catalog, NUM_VARS, 1)
        .unwrap()
        .matrix_capacity();
    AkitaProverSetup::<F>::generate_with_capacity(NUM_VARS, 1, capacity).unwrap()
}

fn with_provider(
    setup: &AkitaProverSetup<F>,
    provider: &Arc<DelegatingProvider>,
) -> CpuBackend<F, E> {
    let provider: Arc<dyn CommitmentStageProvider<F>> = provider.clone();
    CpuBackend::new(setup.expanded.clone())
        .unwrap()
        .with_commitment_stage_provider(provider)
}

fn dense_poly() -> DensePoly<F> {
    let evals = (0..1usize << NUM_VARS)
        .map(|index| F::from_u64(index as u64 * 7 + 3))
        .collect::<Vec<_>>();
    DensePoly::from_field_evals(NUM_VARS, &evals).unwrap()
}

fn onehot_poly() -> OneHotPoly<F, u8> {
    let chunk = akita_config::unit_onehot_source_chunk_size::<fp128::OneHot>().unwrap();
    let chunks = (1usize << NUM_VARS) / chunk;
    let indices = (0..chunks)
        .map(|index| (index % 5 != 0).then(|| ((index * 31 + 7) % chunk) as u8))
        .collect();
    OneHotPoly::new(chunk, indices).unwrap()
}

fn commit<Cfg, P>(
    backend: &CpuBackend<F, E>,
    catalog: &TrustedScheduleCatalog<Cfg>,
    poly: &P,
) -> CommitOutput<F, E>
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    P: crate::CpuSource<F, E> + Clone,
{
    let source = backend.import_source(vec![poly.clone()]).unwrap();
    backend
        .commit(
            catalog,
            &source,
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap()
}

/// Commit, export and import the required setup prefixes, and prove.
///
/// Returns the public commitment, retained state, exported setup-prefix
/// artifacts, and proof bytes.
/// Lagrange-basis root opening of a source at its committed profile.
macro_rules! root_opening {
    ($poly:expr, $point:expr, $profile:expr) => {
        akita_types::dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            $profile.inner.matrix.ring_dimension(),
            |D| crate::evaluate_root_polynomial::<F, _, D>(
                $poly,
                $point,
                $profile.blocks.positions_per_block,
                $profile.blocks.live_blocks,
                BasisMode::Lagrange,
            )
        )
        .unwrap()
    };
}

fn dense_opening(poly: &DensePoly<F>, point: &[F], profile: &GroupCommitPhaseParams) -> F {
    root_opening!(poly, point, profile)
}

fn onehot_opening(poly: &OneHotPoly<F, u8>, point: &[F], profile: &GroupCommitPhaseParams) -> F {
    root_opening!(poly, point, profile)
}

type Opening<P> = fn(&P, &[F], &GroupCommitPhaseParams) -> F;

fn commit_and_prove<Cfg, P>(
    backend: &CpuBackend<F, E>,
    catalog: &TrustedScheduleCatalog<Cfg>,
    setup: &AkitaProverSetup<F>,
    poly: &P,
    opening: Opening<P>,
) -> (
    CommitOutput<F, E>,
    crate::commitment::SetupPrefixProverRegistry<F>,
    Vec<u8>,
)
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    P: crate::CpuSource<F, E> + Clone,
{
    let output = commit(backend, catalog, poly);
    let profile = *output.committed_group.profile();
    let point = (0..NUM_VARS)
        .map(|index| F::from_u64(index as u64 * 13 + 5))
        .collect::<Vec<_>>();
    let evaluation = opening(poly, &point, &profile);
    let group = PolynomialGroupClaims::new(point, vec![evaluation], output.committed_group.clone())
        .unwrap();
    let opening = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        OpeningClaims::from_groups(vec![group]).unwrap(),
        vec![output.private_handle.clone()],
        catalog,
    )
    .unwrap();
    let resolved = catalog.resolve_selection(opening.selection()).unwrap();
    let required = akita_config::required_setup_prefix_slot_ids_for_schedule(
        resolved.schedule(),
        opening.opening_layout(),
    )
    .unwrap();
    let exported = backend.export_setup_prefixes(&required).unwrap();
    let prefix_slots = backend.import_setup_prefixes(&exported, &required).unwrap();
    let proof = akita_prover::batched_prove::<Cfg, CpuBackend<F, E>>(
        setup.expanded.descriptor(),
        &prefix_slots,
        catalog,
        backend,
        opening,
        b"test/commitment-stage-provider",
        BasisMode::Lagrange,
    )
    .unwrap();
    (output, exported, proof)
}

fn assert_provider_route_matches_cpu<Cfg, P>(poly: P, opening: Opening<P>)
where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    P: crate::CpuSource<F, E> + Clone,
{
    let catalog = catalog::<Cfg>();
    let setup = setup(&catalog);
    let cpu = CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap();
    let (expected, expected_prefixes, expected_proof) =
        commit_and_prove(&cpu, &catalog, &setup, &poly, opening);

    let provider = Arc::new(DelegatingProvider::new(
        setup.expanded.clone(),
        true,
        true,
        Decline::Never,
    ));
    let accelerated = with_provider(&setup, &provider);
    let (actual, actual_prefixes, actual_proof) =
        commit_and_prove(&accelerated, &catalog, &setup, &poly, opening);
    let (inner_calls, outer_calls) = provider.calls();
    // The root and at least one recursive witness commitment took the route.
    assert!(inner_calls > 1 && outer_calls > 0);
    assert_eq!(actual.committed_group, expected.committed_group);
    assert!(actual.private_handle.committed.retained == expected.private_handle.committed.retained);
    assert!(actual_prefixes == expected_prefixes);
    assert!(!expected_proof.is_empty());
    assert_eq!(actual_proof, expected_proof);

    // A single accelerated stage crosses owners through exported host rows.
    for (inner, outer) in [(true, false), (false, true)] {
        let provider = Arc::new(DelegatingProvider::new(
            setup.expanded.clone(),
            inner,
            outer,
            Decline::Never,
        ));
        let actual = commit(&with_provider(&setup, &provider), &catalog, &poly);
        let (inner_calls, outer_calls) = provider.calls();
        assert_eq!((inner_calls > 0, outer_calls > 0), (inner, outer));
        assert_eq!(actual.committed_group, expected.committed_group);
        assert!(
            actual.private_handle.committed.retained == expected.private_handle.committed.retained
        );
    }
}

fn run_with_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(test)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn dense_provider_route_matches_cpu_bytes() {
    run_with_stack(|| {
        assert_provider_route_matches_cpu::<fp128::Dense, _>(dense_poly(), dense_opening)
    });
}

#[test]
fn onehot_provider_route_matches_cpu_bytes() {
    run_with_stack(|| {
        assert_provider_route_matches_cpu::<fp128::OneHot, _>(onehot_poly(), onehot_opening)
    });
}

#[test]
fn setup_prefix_provider_route_matches_cpu_artifacts() {
    run_with_stack(|| {
        let catalog = catalog::<fp128::Dense>();
        let setup = setup(&catalog);
        let row = catalog
            .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                NUM_VARS, 1,
            )))
            .unwrap();
        let params = &row.schedule().root.params;
        let n_prefix = (params.d_a() * params.outer_slice_count().get()).next_power_of_two();
        let prefix = akita_types::setup_prefix_precommitted_params(params, n_prefix).unwrap();
        let id = akita_types::scheduled_setup_prefix(n_prefix, prefix)
            .slot_id()
            .unwrap();
        let cpu = CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap();
        let expected = cpu
            .export_setup_prefixes(std::slice::from_ref(&id))
            .unwrap();

        let provider = Arc::new(DelegatingProvider::new(
            setup.expanded.clone(),
            true,
            true,
            Decline::Never,
        ));
        let accelerated = with_provider(&setup, &provider);
        let actual = accelerated
            .export_setup_prefixes(std::slice::from_ref(&id))
            .unwrap();
        assert_eq!(provider.calls(), (1, 1));
        assert!(actual == expected);
    });
}

#[test]
fn declined_requests_fall_back_before_materialization() {
    run_with_stack(|| {
        let catalog = catalog::<fp128::Dense>();
        let setup = setup(&catalog);
        let poly = dense_poly();
        let profile = catalog
            .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                NUM_VARS, 1,
            )))
            .unwrap()
            .profiles()
            .final_group;
        let plan = CommitmentExecutionPlan::for_root(&profile).unwrap();
        let cpu = CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap();
        let sources: [&dyn CommitmentSource<F>; 1] = [&poly];
        let expected = cpu
            .commitment_executor(Vec::new(), &plan, &sources)
            .unwrap()
            .execute_full_via_outer_image(&cpu, profile, &sources)
            .unwrap();

        for decline in [Decline::InnerPlan, Decline::OuterPlan, Decline::SourceType] {
            let provider = Arc::new(DelegatingProvider::new(
                setup.expanded.clone(),
                true,
                true,
                decline,
            ));
            let backend = with_provider(&setup, &provider);
            let source = CountingSource {
                poly: &poly,
                materializations: AtomicUsize::new(0),
            };
            let sources: [&dyn CommitmentSource<F>; 1] = [&source];
            let executor = backend
                .commitment_executor(Vec::new(), &plan, &sources)
                .unwrap();
            assert_eq!(source.materializations.load(Ordering::SeqCst), 0);
            let actual = executor
                .execute_full_via_outer_image(&backend, profile, &sources)
                .unwrap();
            assert_eq!(source.materializations.load(Ordering::SeqCst), 1);
            assert_eq!(provider.calls(), (0, 0));
            assert!(actual == expected);
        }
    });
}
