//! Heterogeneous [`ProverComputeStack`] contract: distinct backend types per cluster.
//!
//! Proving still uses the uniform degenerate stack today (`B,B,B,B`), but the
//! stack type already supports mixing backends. This test locks that contract:
//! construction validates every cluster, and mismatched prepared setup is rejected.

#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_config::proof_optimized::fp64;
use akita_config::CommitmentConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_prover::compute::{ComputeBackendSetup, ProverComputeStack};
use akita_prover::{AkitaProverSetup, CpuBackend, CpuPreparedSetup};
use akita_types::SetupMatrixEnvelope;
use std::any::TypeId;

type Cfg = fp64::D32Full;
type F = <Cfg as CommitmentConfig>::Field;
const D: usize = Cfg::D;

fn test_envelope(max_setup_len: usize) -> SetupMatrixEnvelope {
    SetupMatrixEnvelope {
        max_setup_len,
        #[cfg(feature = "zk")]
        max_zk_b_len: 0,
        #[cfg(feature = "zk")]
        max_zk_d_len: 0,
    }
}

/// Distinct opening-cluster backend type that delegates setup prep to [`CpuBackend`].
#[derive(Debug, Clone, Copy, Default)]
struct DummyOpeningBackend;

/// Distinct ring-switch-cluster backend type that delegates setup prep to [`CpuBackend`].
#[derive(Debug, Clone, Copy, Default)]
struct DummyRingSwitchBackend;

impl<F> ComputeBackendSetup<F> for DummyOpeningBackend
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

impl<F> ComputeBackendSetup<F> for DummyRingSwitchBackend
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

type HeterogeneousStack<'a> = ProverComputeStack<
    'a,
    F,
    D,
    CpuBackend,
    DummyOpeningBackend,
    CpuBackend,
    DummyRingSwitchBackend,
>;

#[test]
fn heterogeneous_stack_has_distinct_cluster_backend_types() {
    let setup = AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, test_envelope(4096))
        .expect("setup");
    let cpu_prepared = CpuBackend.prepare_setup(&setup).expect("cpu prepared");
    let opening_prepared = DummyOpeningBackend
        .prepare_setup(&setup)
        .expect("opening prepared");
    let ring_prepared = DummyRingSwitchBackend
        .prepare_setup(&setup)
        .expect("ring prepared");

    let stack: HeterogeneousStack<'_> = ProverComputeStack::new(
        (&CpuBackend, &cpu_prepared),
        (&DummyOpeningBackend, &opening_prepared),
        (&CpuBackend, &cpu_prepared),
        (&DummyRingSwitchBackend, &ring_prepared),
        setup.expanded.as_ref(),
    )
    .expect("heterogeneous stack");

    assert_ne!(
        TypeId::of::<DummyOpeningBackend>(),
        TypeId::of::<CpuBackend>()
    );
    assert_ne!(
        TypeId::of::<DummyRingSwitchBackend>(),
        TypeId::of::<CpuBackend>()
    );
    assert_ne!(
        TypeId::of::<DummyOpeningBackend>(),
        TypeId::of::<DummyRingSwitchBackend>()
    );

    assert!(std::ptr::eq(stack.commit().backend(), &CpuBackend));
    assert!(std::ptr::eq(
        stack.opening().backend(),
        &DummyOpeningBackend
    ));
    assert!(std::ptr::eq(stack.tensor().backend(), &CpuBackend));
    assert!(std::ptr::eq(
        stack.ring_switch().backend(),
        &DummyRingSwitchBackend
    ));
}

#[test]
fn heterogeneous_stack_rejects_mismatched_opening_prepared() {
    let setup_a = AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, test_envelope(4096))
        .expect("setup a");
    let setup_b = AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, test_envelope(8192))
        .expect("setup b");
    assert_ne!(setup_a.expanded.seed(), setup_b.expanded.seed());

    let cpu_prepared = CpuBackend.prepare_setup(&setup_a).expect("cpu prepared");
    let wrong_opening = DummyOpeningBackend
        .prepare_setup(&setup_b)
        .expect("opening prepared for setup b");
    let ring_prepared = DummyRingSwitchBackend
        .prepare_setup(&setup_a)
        .expect("ring prepared");

    assert!(matches!(
        ProverComputeStack::<
            F,
            D,
            CpuBackend,
            DummyOpeningBackend,
            CpuBackend,
            DummyRingSwitchBackend,
        >::new(
            (&CpuBackend, &cpu_prepared),
            (&DummyOpeningBackend, &wrong_opening),
            (&CpuBackend, &cpu_prepared),
            (&DummyRingSwitchBackend, &ring_prepared),
            setup_a.expanded.as_ref(),
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}
