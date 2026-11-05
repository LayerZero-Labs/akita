use super::*;
use crate::commitment::{CommitmentExecutor, PortableStatePolicy};
use crate::opaque::owned::CommittedSource;
use crate::{AkitaProverSetup, DensePoly};
use akita_params::sis::{
    sis_table_key_for_linf_bound, CommittedSourceClass, CommittedSourceContract,
    InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole, SisTableDigest,
    DEFAULT_SIS_SECURITY_POLICY,
};
use akita_params::{
    BlockGeometry, CommitmentSliceCount, CommitmentSliceGeometry, DecompositionParams,
    GadgetDigits, GroupCommitPhaseParams, PolynomialGroupLayout, RoleParams, SetupMatrixCapacity,
    SisModulusProfileId,
};
use akita_types::binary_source::OrdinaryNativeRootPlan;
use akita_types::root_repacking::RootCoefficientRepacking;
use jolt_algebra::{One, Prime128Offset275, Ring, Zero};
use std::sync::Arc;

type F = Prime128Offset275;

fn profile(slices: CommitmentSliceCount) -> GroupCommitPhaseParams {
    // Resolve real audited table cells and their minimum ranks; zero-bound
    // params-only shells are not valid frozen commitment identities.
    let key = |role, degree| {
        sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            SisModulusProfileId::Q128Offset275,
            role,
            degree,
            8,
        )
        .unwrap()
    };
    let inner =
        InnerCommitMatrixParams::try_new_with_min_rank(key(SisMatrixRole::Inner, 128), 2).unwrap();
    let geometry =
        CommitmentSliceGeometry::try_new(slices, 4, 1, inner.output_rank(), 43, 128, 64).unwrap();
    let outer = OuterCommitMatrixParams::try_new_with_min_rank(
        key(SisMatrixRole::Outer, 64),
        geometry.physical_input_width(),
    )
    .unwrap();
    let profile = GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        randomness_columns: 0,
        group: PolynomialGroupLayout::singleton(10),
        blocks: BlockGeometry::new(8, 2, 4),
        outer_slice_count: slices,
        inner: RoleParams::new(GadgetDigits::new(3, 1), inner),
        outer: RoleParams::new(GadgetDigits::new(3, 43), outer),
    };
    profile.validate_frozen_precommit(128).unwrap();
    profile
}

#[test]
fn mixed_preimage_uses_canonical_multi_block_b_slices() {
    let profiles = [
        profile(CommitmentSliceCount::ONE),
        profile(CommitmentSliceCount::TWO),
    ];
    let capacity = profiles
        .iter()
        .map(|profile| {
            [
                profile.inner.matrix.output_rank() * profile.inner.matrix.input_width() * 128,
                profile.outer.matrix.output_rank() * profile.outer.matrix.input_width() * 64,
                profile
                    .outer_compression_plan()
                    .unwrap()
                    .max_setup_field_elements()
                    .unwrap(),
            ]
            .into_iter()
            .max()
            .unwrap()
        })
        .max()
        .unwrap()
        .next_power_of_two();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        10,
        1,
        SetupMatrixCapacity {
            num_field_elements: capacity,
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
    let coefficients = (0..1024)
        .map(|i| F::from_i64((i % 8) as i64 - 4))
        .collect::<Vec<_>>();
    let source = backend
        .import_source(vec![
            DensePoly::from_field_evals(10, coefficients.clone()).unwrap()
        ])
        .unwrap();
    let prepared = backend.prepared().unwrap();
    let executor = CommitmentExecutor::cpu(
        &backend,
        prepared,
        setup.expanded.as_ref(),
        Vec::new(),
        PortableStatePolicy,
    )
    .unwrap();
    let mut previous_t = None;
    for profile in profiles {
        let slices = profile.outer_slice_count;
        let geometry = profile.derive_slice_geometry().unwrap();
        assert_eq!(profile.blocks.live_blocks, 4);
        assert_eq!(geometry.max_blocks_per_slice(), 4 / slices.get());
        assert!(geometry.max_blocks_per_slice() > 1);
        let (committed, retained) = executor
            .execute_full_via_outer_image(
                &backend,
                profile,
                None,
                &source.storage.commitment_sources(),
                None,
            )
            .unwrap();
        let handle = CommitmentHandle {
            owner: source.owner,
            committed: Arc::new(CommittedSource {
                source: source.storage.clone(),
                metadata: source.metadata,
                commitment_id: backend.owner().next_operation_id().unwrap(),
                parameters: profile,
                producer_contract: CommittedSourceContract::try_new(
                    CommittedSourceClass::BalancedSignedDigit,
                    DecompositionParams {
                        log_basis: 3,
                        log_commit_bound: 3,
                        log_open_bound: Some(128),
                    },
                )
                .unwrap(),
                public: committed.commitment().clone(),
                retained: Some(retained),
            }),
        };
        let plan = OrdinaryNativeRootPlan::new(profile, 0).unwrap();
        let segments = backend.ordinary_root_preimage(&handle, &plan).unwrap();
        assert_eq!(segments.len(), 4);
        assert_eq!(segments[0].len(), coefficients.len());
        assert_eq!(segments[1].len(), geometry.logical_input_width() * 64);
        if let Some(previous) = &previous_t {
            // Global T ordering is independent of how B slices group blocks.
            assert_eq!(&segments[1], previous);
        }
        previous_t = Some(segments[1].clone());
        let image = &handle
            .committed
            .ordinary_retained()
            .unwrap()
            .inner_rows()
            .unwrap()[0];
        for (image_element, row) in image.coeffs().chunks_exact(128).enumerate() {
            for (coefficient, &expected) in row.iter().enumerate() {
                let mut actual = F::zero();
                let mut power = F::one();
                for digit in 0..43 {
                    actual += power
                        * F::from_i64(i64::from(
                            segments[1][((image_element * 2 + coefficient / 64) * 43 + digit) * 64
                                + coefficient % 64],
                        ));
                    power *= F::from_u64(8);
                }
                assert_eq!(actual, expected);
            }
        }
        let repacking = RootCoefficientRepacking::new(plan.segments().to_vec(), 64).unwrap();
        let natural = segments.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let digits = repacking.repack(&natural).unwrap();
        let mut table = digits
            .into_iter()
            .map(|digit| F::from_i64(i64::from(digit)))
            .collect::<Vec<_>>();
        let point = (0..10).map(|i| F::from_u64(i + 3)).collect::<Vec<_>>();
        let evaluation = akita_algebra::poly::multilinear_eval(&coefficients, &point).unwrap();
        let batches = (0..plan.linear_batch_count())
            .map(|i| F::from_u64(i as u64 + 11))
            .collect::<Vec<_>>();
        let weights = plan
            .linear_weights(
                &repacking,
                setup.expanded.descriptor(),
                &point,
                &[evaluation],
                committed.rows().coeffs(),
                F::from_u64(17),
                &batches,
            )
            .unwrap();
        weights.check_table(&table).unwrap();
        // Pin the authenticated T ordering across a boundary inside a B slice.
        let changed = repacking
            .coefficient_address(1, profile.inner.matrix.output_rank() * 2, 0, 3)
            .unwrap();
        table[changed] += F::one();
        assert!(weights.check_table(&table).is_err());

        let mut malformed = profile;
        malformed.outer_slice_count = CommitmentSliceCount::FOUR;
        assert!(malformed.validate_frozen_precommit(128).is_err());
        assert!(OrdinaryNativeRootPlan::new(malformed, 0).is_err());
    }
}
