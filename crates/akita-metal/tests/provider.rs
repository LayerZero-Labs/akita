//! The Metal commitment stage provider installed in the CPU backend: every
//! committed group, exported setup prefix and proof byte equals the plain CPU
//! backend's, and the device stages actually ran.

#![cfg(target_os = "macos")]

mod support;

mod fp128_provider {
    use std::sync::Arc;

    use akita_config::proof_optimized::fp128;
    use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
    use akita_cpu_backend::commitment_backend::CommitmentStageProvider;
    use akita_cpu_backend::{
        AkitaProverSetup, CommitOutput, CpuBackend, CpuSource, DensePoly, GroupContext, OneHotPoly,
        SetupPrefixProverRegistry,
    };
    use akita_metal::provider::MetalCommitmentProvider;
    use akita_prover::SelectedProverOpeningData;
    use akita_types::{BasisMode, GroupCommitPhaseParams, OpeningClaims, PolynomialGroupClaims};
    use jolt_field::Ring;

    use crate::support::gpu;

    type F = fp128::Field;
    type E = <fp128::Dense as CommitmentConfig>::ExtField;

    const NUM_VARS: usize = 16;

    fn setup<Cfg>(catalog: &TrustedScheduleCatalog<Cfg>) -> AkitaProverSetup<F>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
    {
        let capacity = akita_config::SetupRequirements::from_catalog::<Cfg>(catalog, NUM_VARS, 1)
            .unwrap()
            .matrix_capacity();
        AkitaProverSetup::<F>::generate_with_capacity(NUM_VARS, 1, capacity).unwrap()
    }

    fn dense_poly() -> DensePoly<F> {
        // Large coefficients reach every digit of the inner decomposition.
        let evals = (0..1u64 << NUM_VARS)
            .map(|index| {
                F::from_u64(index.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5bd1_e995)
                    * F::from_u64(index + 3)
                    - F::from_u64(index * 7)
            })
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

    type Opening<P> = fn(&P, &[F], &GroupCommitPhaseParams) -> F;

    macro_rules! root_opening {
        ($poly:expr, $point:expr, $profile:expr) => {
            akita_types::dispatch_for_field!(
                akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
                F,
                $profile.inner.matrix.ring_dimension(),
                |D| akita_cpu_backend::evaluate_root_polynomial::<F, _, D>(
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

    fn onehot_opening(
        poly: &OneHotPoly<F, u8>,
        point: &[F],
        profile: &GroupCommitPhaseParams,
    ) -> F {
        root_opening!(poly, point, profile)
    }

    /// Commit, export and import the setup prefixes the opening needs,
    /// and prove.
    fn commit_and_prove<Cfg, P>(
        backend: &CpuBackend<F, E>,
        catalog: &TrustedScheduleCatalog<Cfg>,
        setup: &AkitaProverSetup<F>,
        poly: &P,
        opening: Opening<P>,
    ) -> (CommitOutput<F, E>, SetupPrefixProverRegistry<F>, Vec<u8>)
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
        P: CpuSource<F, E> + Clone,
    {
        let source = backend.import_source(vec![poly.clone()]).unwrap();
        let output = backend
            .commit(
                catalog,
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        let profile = *output.committed_group.profile();
        let point = (0..NUM_VARS)
            .map(|index| F::from_u64(index as u64 * 13 + 5))
            .collect::<Vec<_>>();
        let evaluation = opening(poly, &point, &profile);
        let group =
            PolynomialGroupClaims::new(point, vec![evaluation], output.committed_group.clone())
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
            b"test/metal-commitment-provider",
            BasisMode::Lagrange,
        )
        .unwrap();
        (output, exported, proof)
    }

    fn assert_matches_cpu<Cfg, P>(poly: P, opening: Opening<P>)
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
        P: CpuSource<F, E> + Clone,
    {
        let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
        let setup = setup(&catalog);
        let test = gpu();
        let cpu = CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap();
        let expected = commit_and_prove(&cpu, &catalog, &setup, &poly, opening);

        let provider = Arc::new(MetalCommitmentProvider::with_device(
            test.metal,
            setup.expanded.clone(),
        ));
        let installed: Arc<dyn CommitmentStageProvider<F>> = provider.clone();
        let metal = CpuBackend::<F, E>::new(setup.expanded.clone())
            .unwrap()
            .with_commitment_stage_provider(installed);
        let actual = commit_and_prove(&metal, &catalog, &setup, &poly, opening);
        let calls = provider.stage_calls();
        // The root commitment ran on the device; the recursive
        // short-norm witnesses have no device stage and were declined.
        assert!(calls.inner >= 1 && calls.outer >= 1, "{calls:?}");
        assert_eq!(calls.onehot_host, 0);
        assert!(provider.prepared_matrices() > 0);
        assert_eq!(actual.0.committed_group, expected.0.committed_group);
        assert!(actual.1 == expected.1);
        assert_eq!(actual.2, expected.2);
    }

    /// Two polynomials in one group: the outer stage interleaves their blocks
    /// into each slice on the host before the B matvec.
    #[test]
    fn two_polynomial_group_matches_cpu_commitment() {
        run(|| {
            let catalog =
                akita_config::test_support::workspace_schedule_catalog::<fp128::Dense>().unwrap();
            let capacity = akita_config::SetupRequirements::from_catalog::<fp128::Dense>(
                &catalog, NUM_VARS, 2,
            )
            .unwrap()
            .matrix_capacity();
            let setup =
                AkitaProverSetup::<F>::generate_with_capacity(NUM_VARS, 2, capacity).unwrap();
            let second = {
                let evals = (0..1u64 << NUM_VARS)
                    .map(|index| -F::from_u64(index * index + 11))
                    .collect::<Vec<_>>();
                DensePoly::from_field_evals(NUM_VARS, &evals).unwrap()
            };
            let polys = vec![dense_poly(), second];
            let test = gpu();
            let commit = |backend: &CpuBackend<F, E>| {
                let source = backend.import_source(polys.clone()).unwrap();
                backend
                    .commit(
                        &catalog,
                        &source,
                        GroupContext::scheduler_without_precommitted_groups(),
                    )
                    .unwrap()
                    .committed_group
            };
            let expected = commit(&CpuBackend::new(setup.expanded.clone()).unwrap());
            let provider = Arc::new(MetalCommitmentProvider::with_device(
                test.metal,
                setup.expanded.clone(),
            ));
            let installed: Arc<dyn CommitmentStageProvider<F>> = provider.clone();
            let metal = CpuBackend::<F, E>::new(setup.expanded.clone())
                .unwrap()
                .with_commitment_stage_provider(installed);
            assert_eq!(commit(&metal), expected);
            assert_eq!(provider.stage_calls().outer, 1);
        });
    }

    fn run(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(test)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn dense_matches_cpu_bytes() {
        run(|| assert_matches_cpu::<fp128::Dense, _>(dense_poly(), dense_opening));
    }

    #[test]
    fn onehot_matches_cpu_bytes() {
        run(|| assert_matches_cpu::<fp128::OneHot, _>(onehot_poly(), onehot_opening));
    }
}

/// fp64 commitments. Its presets open over an extension field, so these
/// compare the committed groups; the fp128 suite covers proof bytes.
mod fp64_provider {
    use std::sync::Arc;

    use akita_config::proof_optimized::fp64;
    use akita_config::CommitmentConfig;
    use akita_cpu_backend::commitment_backend::CommitmentStageProvider;
    use akita_cpu_backend::{
        AkitaProverSetup, CpuBackend, CpuSource, DensePoly, GroupContext, OneHotPoly,
    };
    use akita_metal::provider::MetalCommitmentProvider;
    use akita_types::CommittedGroup;
    use jolt_field::Ring;

    use crate::support::gpu;

    type F = fp64::Field;
    type E = <fp64::Dense as CommitmentConfig>::ExtField;

    /// The smallest arity with an fp64 schedule of `Cfg`.
    fn num_vars<Cfg: CommitmentConfig<Field = F, ExtField = E>>() -> usize {
        let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
        (16..=30)
            .find(|&num_vars| {
                akita_config::SetupRequirements::from_catalog::<Cfg>(&catalog, num_vars, 1).is_ok()
            })
            .unwrap()
    }

    fn assert_matches_cpu<Cfg, P>(num_vars: usize, poly: P)
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
        P: CpuSource<F, E> + Clone,
    {
        let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
        let capacity = akita_config::SetupRequirements::from_catalog::<Cfg>(&catalog, num_vars, 1)
            .unwrap()
            .matrix_capacity();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(num_vars, 1, capacity).unwrap();
        let test = gpu();
        let commit = |backend: &CpuBackend<F, E>| -> CommittedGroup<F> {
            let source = backend.import_source(vec![poly.clone()]).unwrap();
            backend
                .commit(
                    &catalog,
                    &source,
                    GroupContext::scheduler_without_precommitted_groups(),
                )
                .unwrap()
                .committed_group
        };
        let expected = commit(&CpuBackend::new(setup.expanded.clone()).unwrap());
        let provider = Arc::new(MetalCommitmentProvider::with_device(
            test.metal,
            setup.expanded.clone(),
        ));
        let installed: Arc<dyn CommitmentStageProvider<F>> = provider.clone();
        let metal = CpuBackend::<F, E>::new(setup.expanded.clone())
            .unwrap()
            .with_commitment_stage_provider(installed);
        assert_eq!(commit(&metal), expected);
        let calls = provider.stage_calls();
        assert_eq!((calls.inner, calls.outer, calls.onehot_host), (1, 1, 0));
    }

    fn run(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(test)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn dense_matches_cpu_commitment() {
        run(|| {
            let num_vars = num_vars::<fp64::Dense>();
            let evals = (0..1u64 << num_vars)
                .map(|index| F::from_u64(index.wrapping_mul(0x9e37_79b9_7f4a_7c15)))
                .collect::<Vec<_>>();
            assert_matches_cpu::<fp64::Dense, _>(
                num_vars,
                DensePoly::<F>::from_field_evals(num_vars, &evals).unwrap(),
            );
        });
    }

    #[test]
    fn onehot_matches_cpu_commitment() {
        run(|| {
            let num_vars = num_vars::<fp64::OneHot>();
            let chunk = akita_config::unit_onehot_source_chunk_size::<fp64::OneHot>().unwrap();
            let indices = (0..(1usize << num_vars) / chunk)
                .map(|index| (index % 3 != 0).then(|| ((index * 29 + 3) % chunk) as u8))
                .collect();
            assert_matches_cpu::<fp64::OneHot, _>(
                num_vars,
                OneHotPoly::<F, u8>::new(chunk, indices).unwrap(),
            );
        });
    }
}
