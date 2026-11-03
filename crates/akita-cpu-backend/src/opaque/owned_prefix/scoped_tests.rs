//! Supplied public prefixes retain their own source without aliasing ordinary slots.

use super::*;
use akita_config::proof_optimized::fp128;
use jolt_field::{One, Zero};

#[test]
fn scoped_prefix_retains_source_without_replacing_ordinary_cache() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            type Cfg = fp128::Dense;
            type F = fp128::Field;
            const NV: usize = 14;
            let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>(
                akita_params::ChunkedWitnessCfg::default_non_chunked(),
            )
            .unwrap();
            let capacity = akita_config::SetupRequirements::from_catalog::<Cfg>(&catalog, NV, 1)
                .unwrap()
                .matrix_capacity();
            let setup =
                crate::AkitaProverSetup::<F>::generate_with_capacity(NV, 1, capacity).unwrap();
            let params = &catalog
                .resolve_key(&akita_params::ScheduleLookupKey::single(
                    akita_params::PolynomialGroupLayout::new(NV, 1),
                ))
                .unwrap()
                .schedule()
                .root
                .params;
            let n_prefix = (params.d_a() * params.outer_slice_count().get()).next_power_of_two();
            let profile = akita_params::setup_prefix_precommitted_params(params, n_prefix).unwrap();
            let id = akita_params::scheduled_setup_prefix(n_prefix, profile)
                .slot_id()
                .unwrap();
            let backend = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
            let ordinary = backend.prepare_setup_prefix(&id).unwrap();
            assert_eq!(backend.setup_prefix_cache_len().unwrap(), 1);
            for len in [0, n_prefix - 1, n_prefix + 1] {
                assert!(matches!(
                    backend.prepare_setup_prefix_from_coefficients(&id, vec![F::zero(); len]),
                    Err(AkitaError::InvalidInput(_))
                ));
            }
            let mut coefficients = vec![F::zero(); n_prefix];
            coefficients[0] = F::one();
            let scoped = backend
                .prepare_setup_prefix_from_coefficients(&id, coefficients.clone())
                .unwrap();
            assert_eq!(backend.setup_prefix_cache_len().unwrap(), 1);
            let retained = scoped
                .commitment_handle
                .committed
                .source
                .dense_polynomials()
                .unwrap();
            assert_eq!(retained.len(), 1);
            assert_eq!(&retained[0].field_coeffs()[..n_prefix], coefficients);
            let again = backend.prepare_setup_prefix(&id).unwrap();
            assert_eq!(ordinary.public, again.public);
            assert!(Arc::ptr_eq(
                &ordinary.commitment_handle.committed.source,
                &again.commitment_handle.committed.source,
            ));
            assert!(!Arc::ptr_eq(
                &ordinary.commitment_handle.committed.source,
                &scoped.commitment_handle.committed.source,
            ));
        })
        .unwrap()
        .join()
        .unwrap();
}
