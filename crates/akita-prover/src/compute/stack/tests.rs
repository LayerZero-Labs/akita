use super::*;
use crate::compute::CompressionComputeBackend;
use crate::AkitaProverSetup;
use crate::CpuBackend;
use akita_field::{AkitaError, Fp64};
use akita_types::SetupMatrixCapacity;

type F = Fp64<4294967197>;
fn test_envelope(num_field_elements: usize) -> SetupMatrixCapacity {
    SetupMatrixCapacity { num_field_elements }
}

#[test]
fn operation_ctx_rejects_mismatched_expanded_setup() {
    let setup_a =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup a");
    let setup_b =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(8192)).expect("setup b");
    assert_ne!(setup_a.expanded.seed(), setup_b.expanded.seed());

    let prepared_a = CpuBackend::DEFAULT
        .prepare_setup(&setup_a)
        .expect("prepared a");
    assert!(matches!(
        OperationCtx::new(&CpuBackend::DEFAULT, &prepared_a, setup_b.expanded.as_ref()),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn operation_ctx_accepts_matching_expanded_setup() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    OperationCtx::new(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
        .expect("matching expanded metadata should validate");
}

use crate::compute::{CommitCluster, RingSwitchCluster};

fn assert_distinct_backend_types<C: 'static, R: 'static>() {
    fn type_id<T: 'static>() -> std::any::TypeId {
        std::any::TypeId::of::<T>()
    }
    assert_ne!(type_id::<C>(), type_id::<R>());
}

type TestUniformStack<'a> = UniformProverStack<'a, F, CpuBackend>;
type TestHeterogeneousStack<'a> =
    ProverComputeStack<'a, F, CommitCluster, CpuBackend, CpuBackend, RingSwitchCluster>;

fn all_cluster_requirements() -> NttExecutionRequirements {
    let mut requirements = NttExecutionRequirements::default();
    for (cluster, domain, width) in [
        (
            NttOperationCluster::Commit,
            akita_types::NttTransformDomain::Negacyclic,
            3,
        ),
        (
            NttOperationCluster::Opening,
            akita_types::NttTransformDomain::Negacyclic,
            4,
        ),
        (
            NttOperationCluster::Tensor,
            akita_types::NttTransformDomain::Cyclic,
            5,
        ),
        (
            NttOperationCluster::RingSwitch,
            akita_types::NttTransformDomain::Cyclic,
            6,
        ),
    ] {
        requirements
            .add_matrix(
                0,
                cluster,
                akita_types::NttCacheKey::from_matrix_shape(64, 1, width, domain).unwrap(),
                width,
            )
            .unwrap();
    }
    requirements
}

#[test]
fn heterogeneous_stack_accepts_distinct_operation_clusters() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let commit_backend = CommitCluster;
    let ring_backend = RingSwitchCluster;
    let stack: TestHeterogeneousStack<'_> = ProverComputeStack::new(
        (&commit_backend, &prepared),
        (&CpuBackend::DEFAULT, &prepared),
        (&CpuBackend::DEFAULT, &prepared),
        (&ring_backend, &prepared),
        setup.expanded.as_ref(),
    )
    .expect("heterogeneous stack");
    assert_distinct_backend_types::<CommitCluster, RingSwitchCluster>();
    assert_eq!(
        stack.commit().backend() as *const _,
        &commit_backend as *const _
    );
    assert_eq!(
        stack.ring_switch().backend() as *const _,
        &ring_backend as *const _
    );
}

#[test]
fn heterogeneous_stack_implements_level_prove_stacks() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let commit_backend = CommitCluster;
    let ring_backend = RingSwitchCluster;
    let stack: TestHeterogeneousStack<'_> = ProverComputeStack::new(
        (&commit_backend, &prepared),
        (&CpuBackend::DEFAULT, &prepared),
        (&CpuBackend::DEFAULT, &prepared),
        (&ring_backend, &prepared),
        setup.expanded.as_ref(),
    )
    .expect("heterogeneous stack");
    let selected: &TestHeterogeneousStack<'_> = LevelProveStacks::prove_stack_at_level(&stack, 0);
    assert_eq!(
        selected.commit().backend() as *const _,
        stack.commit().backend() as *const _
    );
}

#[test]
fn prewarm_routes_only_to_declared_physical_cluster_owner() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let commit_prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("commit prepared");
    let ring_prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("ring prepared");
    let commit_backend = CommitCluster;
    let ring_backend = RingSwitchCluster;
    let stack: TestHeterogeneousStack<'_> = ProverComputeStack::new(
        (&commit_backend, &commit_prepared),
        (&CpuBackend::DEFAULT, &commit_prepared),
        (&CpuBackend::DEFAULT, &commit_prepared),
        (&ring_backend, &ring_prepared),
        setup.expanded.as_ref(),
    )
    .expect("heterogeneous stack");
    let mut requirements = NttExecutionRequirements::default();
    requirements
        .add_matrix(
            0,
            NttOperationCluster::Commit,
            akita_types::NttCacheKey::from_matrix_shape(
                64,
                2,
                3,
                akita_types::NttTransformDomain::Negacyclic,
            )
            .unwrap(),
            6,
        )
        .unwrap();
    requirements
        .add_matrix(
            0,
            NttOperationCluster::RingSwitch,
            akita_types::NttCacheKey::from_matrix_shape(
                64,
                1,
                5,
                akita_types::NttTransformDomain::Cyclic,
            )
            .unwrap(),
            5,
        )
        .unwrap();

    prewarm_ntt_requirements::<F, _>(&stack, &requirements).expect("prewarm plan");

    assert_eq!(commit_prepared.shared_ntt_cache_metrics().unwrap().len(), 1);
    assert_eq!(ring_prepared.shared_ntt_cache_metrics().unwrap().len(), 1);
    assert_eq!(
        commit_prepared.shared_ntt_cache_metrics().unwrap()[0]
            .key
            .domain,
        akita_types::NttTransformDomain::Negacyclic
    );
    assert_eq!(
        ring_prepared.shared_ntt_cache_metrics().unwrap()[0]
            .key
            .domain,
        akita_types::NttTransformDomain::Cyclic
    );
    let metrics = planned_ntt_cache_metrics::<F, _>(&stack, &requirements).unwrap();
    assert_eq!(metrics.len(), 2);
    assert_eq!(
        metrics
            .iter()
            .map(|metric| metric.cache_bytes)
            .sum::<usize>(),
        commit_prepared.shared_ntt_cache_bytes() + ring_prepared.shared_ntt_cache_bytes()
    );
}

#[test]
fn prewarm_and_metrics_skip_streamed_cpu_ring_switch_slots() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let stack = TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
        .expect("uniform stack");
    let mut requirements = NttExecutionRequirements::default();
    for domain in [
        akita_types::NttTransformDomain::Negacyclic,
        akita_types::NttTransformDomain::Cyclic,
    ] {
        requirements
            .add_matrix(
                0,
                NttOperationCluster::RingSwitch,
                akita_types::NttCacheKey::from_matrix_shape(
                    64,
                    1,
                    CpuBackend::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS + 1,
                    domain,
                )
                .unwrap(),
                CpuBackend::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS + 1,
            )
            .unwrap();
    }

    prewarm_ntt_requirements::<F, _>(&stack, &requirements).expect("prewarm streamed plan");

    assert!(prepared.shared_ntt_cache_metrics().unwrap().is_empty());
    assert!(planned_ntt_cache_metrics::<F, _>(&stack, &requirements)
        .unwrap()
        .is_empty());
}

#[test]
fn configured_ring_switch_limit_drives_prewarm_and_metrics_boundary() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let backend =
        CpuBackend::with_resource_limits(5, CpuBackend::DEFAULT_ONEHOT_SCRATCH_BYTES_PER_WORKER)
            .unwrap();
    let prepared = backend.prepare_setup(&setup).expect("prepared");
    let stack = TestUniformStack::uniform(&backend, &prepared, setup.expanded.as_ref())
        .expect("uniform stack");
    let mut requirements = NttExecutionRequirements::default();
    for extent in [5, 6] {
        requirements
            .add_matrix(
                0,
                NttOperationCluster::RingSwitch,
                akita_types::NttCacheKey::from_matrix_shape(
                    64,
                    1,
                    extent,
                    akita_types::NttTransformDomain::Cyclic,
                )
                .unwrap(),
                extent,
            )
            .unwrap();
    }

    prewarm_ntt_requirements::<F, _>(&stack, &requirements).expect("prewarm configured plan");

    let resident = prepared.shared_ntt_cache_metrics().unwrap();
    assert_eq!(resident.len(), 1);
    assert_eq!(resident[0].key.num_ring_elements, 5);
    let planned = planned_ntt_cache_metrics::<F, _>(&stack, &requirements).unwrap();
    assert_eq!(planned.len(), 1);
    assert_eq!(planned[0].keys.len(), 1);
    assert_eq!(planned[0].keys[0].num_ring_elements, 5);
}

#[test]
fn prewarm_preserves_cached_operation_sharing_a_route_with_streamed_operation() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let stack = TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
        .expect("uniform stack");
    let mut requirements = NttExecutionRequirements::default();
    requirements
        .add_matrix(
            0,
            NttOperationCluster::RingSwitch,
            akita_types::NttCacheKey::from_matrix_shape(
                64,
                1,
                5,
                akita_types::NttTransformDomain::Cyclic,
            )
            .unwrap(),
            5,
        )
        .unwrap();
    requirements
        .add_matrix(
            0,
            NttOperationCluster::RingSwitch,
            akita_types::NttCacheKey::from_matrix_shape(
                64,
                1,
                CpuBackend::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS + 1,
                akita_types::NttTransformDomain::Cyclic,
            )
            .unwrap(),
            CpuBackend::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS + 1,
        )
        .unwrap();

    prewarm_ntt_requirements::<F, _>(&stack, &requirements).expect("prewarm mixed plan");

    let resident = prepared.shared_ntt_cache_metrics().unwrap();
    assert_eq!(resident.len(), 1);
    assert_eq!(resident[0].key.num_ring_elements, 5);
    let planned = planned_ntt_cache_metrics::<F, _>(&stack, &requirements).unwrap();
    assert_eq!(planned.len(), 1);
    assert_eq!(planned[0].keys.len(), 1);
    assert_eq!(planned[0].keys[0].num_ring_elements, 5);
}

#[test]
fn prewarm_max_joins_retained_requests_by_physical_owner_before_building() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let stack = TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
        .expect("uniform stack");
    let mut requirements = NttExecutionRequirements::default();
    requirements
        .add_matrix(
            0,
            NttOperationCluster::Commit,
            akita_types::NttCacheKey::from_matrix_shape(
                64,
                1,
                5,
                akita_types::NttTransformDomain::Cyclic,
            )
            .unwrap(),
            5,
        )
        .unwrap();
    requirements
        .add_matrix(
            0,
            NttOperationCluster::RingSwitch,
            akita_types::NttCacheKey::from_matrix_shape(
                64,
                1,
                7,
                akita_types::NttTransformDomain::Cyclic,
            )
            .unwrap(),
            7,
        )
        .unwrap();

    prewarm_ntt_requirements::<F, _>(&stack, &requirements).expect("prewarm retained plan");

    assert_eq!(prepared.ntt_slot_build_count(), 1);
    let resident = prepared.shared_ntt_cache_metrics().unwrap();
    assert_eq!(resident.len(), 1);
    assert_eq!(resident[0].key.num_ring_elements, 7);
}

#[test]
fn fused_operation_extent_routes_all_domains_together() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let stack = TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
        .expect("uniform stack");
    let mut requirements = NttExecutionRequirements::default();
    for domain in [
        akita_types::NttTransformDomain::Negacyclic,
        akita_types::NttTransformDomain::Cyclic,
    ] {
        requirements
            .add_matrix(
                0,
                NttOperationCluster::RingSwitch,
                akita_types::NttCacheKey::from_matrix_shape(64, 1, 5, domain).unwrap(),
                CpuBackend::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS + 1,
            )
            .unwrap();
    }

    prewarm_ntt_requirements::<F, _>(&stack, &requirements).expect("prewarm fused streamed plan");

    assert!(prepared.shared_ntt_cache_metrics().unwrap().is_empty());
    assert!(planned_ntt_cache_metrics::<F, _>(&stack, &requirements)
        .unwrap()
        .is_empty());
}

#[test]
fn planned_metrics_deduplicate_all_shared_clusters() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let stack = TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
        .expect("uniform stack");
    let requirements = all_cluster_requirements();

    prewarm_ntt_requirements::<F, _>(&stack, &requirements).unwrap();
    let metrics = planned_ntt_cache_metrics::<F, _>(&stack, &requirements).unwrap();

    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].cache_bytes, prepared.shared_ntt_cache_bytes());
}

#[test]
fn root_lifecycle_retains_by_default_and_explicit_release_deduplicates_owner() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let stack = TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
        .expect("uniform stack");
    let requirements = all_cluster_requirements();

    prewarm_ntt_requirements::<F, _>(&stack, &requirements).unwrap();
    let compression_digits = vec![[0i8; 64]; 3];
    CpuBackend::DEFAULT
        .compression_rows_products::<64>(&prepared, &[compression_digits.as_slice()])
        .expect("warm compression NTT");
    let resident = prepared.ntt_cache_bytes().unwrap();
    assert!(resident > 0);

    LevelProveStacks::after_root_fold(&stack).unwrap();
    assert_eq!(prepared.ntt_cache_bytes().unwrap(), resident);

    let releasing = ReleaseRootNttAfterFold::new(&stack);
    LevelProveStacks::after_root_fold(&releasing).unwrap();
    assert_eq!(prepared.ntt_cache_bytes().unwrap(), 0);

    prewarm_ntt_requirements::<F, _>(&stack, &requirements).unwrap();
    CpuBackend::DEFAULT
        .compression_rows_products::<64>(&prepared, &[compression_digits.as_slice()])
        .expect("rebuild compression NTT");
    assert_eq!(stack.release_built_ntt_slots().unwrap(), resident);
}

#[test]
fn planned_metrics_keep_four_independent_clusters_separate() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let commit = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("commit prepared");
    let opening = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("opening prepared");
    let tensor = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("tensor prepared");
    let ring = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("ring prepared");
    let stack = ProverComputeStack::new(
        (&CpuBackend::DEFAULT, &commit),
        (&CpuBackend::DEFAULT, &opening),
        (&CpuBackend::DEFAULT, &tensor),
        (&CpuBackend::DEFAULT, &ring),
        setup.expanded.as_ref(),
    )
    .expect("independent stack");
    let requirements = all_cluster_requirements();

    prewarm_ntt_requirements::<F, _>(&stack, &requirements).unwrap();
    let metrics = planned_ntt_cache_metrics::<F, _>(&stack, &requirements).unwrap();

    assert_eq!(metrics.len(), 4);
    assert_eq!(
        metrics
            .iter()
            .map(|metric| metric.cache_bytes)
            .sum::<usize>(),
        commit.shared_ntt_cache_bytes()
            + opening.shared_ntt_cache_bytes()
            + tensor.shared_ntt_cache_bytes()
            + ring.shared_ntt_cache_bytes()
    );
}

#[test]
fn tiered_prove_stacks_rejects_empty_table() {
    let result =
        TieredProveStacks::<F, CpuBackend, CpuBackend, CpuBackend, CpuBackend>::new(&[], &[]);
    assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
}

#[test]
fn tiered_prove_stacks_rejects_length_mismatch() {
    let setup =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let stack = TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
        .expect("stack");
    let stacks = [stack];
    let result = TieredProveStacks::new(&stacks, &[1, 2]);
    assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
}

#[test]
fn tiered_prove_stacks_rejects_non_increasing_bounds() {
    let setup_a =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup a");
    let setup_b =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(8192)).expect("setup b");
    let prepared_a = CpuBackend::DEFAULT
        .prepare_setup(&setup_a)
        .expect("prepared a");
    let prepared_b = CpuBackend::DEFAULT
        .prepare_setup(&setup_b)
        .expect("prepared b");
    let stack_a =
        TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared_a, setup_a.expanded.as_ref())
            .expect("stack a");
    let stack_b =
        TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared_b, setup_b.expanded.as_ref())
            .expect("stack b");
    let stacks = [stack_a, stack_b];
    let result = TieredProveStacks::new(&stacks, &[2, 1]);
    assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
}

#[test]
fn tiered_prove_stacks_selects_by_fold_level() {
    let setup_a =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(4096)).expect("setup a");
    let setup_b =
        AkitaProverSetup::<F>::generate_with_capacity(8, 1, test_envelope(8192)).expect("setup b");
    let prepared_a = CpuBackend::DEFAULT
        .prepare_setup(&setup_a)
        .expect("prepared a");
    let prepared_b = CpuBackend::DEFAULT
        .prepare_setup(&setup_b)
        .expect("prepared b");
    let stack_a =
        TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared_a, setup_a.expanded.as_ref())
            .expect("stack a");
    let stack_b =
        TestUniformStack::uniform(&CpuBackend::DEFAULT, &prepared_b, setup_b.expanded.as_ref())
            .expect("stack b");
    let stacks = [stack_a, stack_b];
    let tiered = TieredProveStacks::new(&stacks, &[1, usize::MAX]).expect("tiered");
    assert!(std::ptr::eq(
        tiered.prove_stack_at_level(0),
        tiered.prove_stack_at_level(1),
    ));
    assert!(!std::ptr::eq(
        tiered.prove_stack_at_level(0),
        tiered.prove_stack_at_level(2),
    ));
}
