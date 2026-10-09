//! The dynamic one-hot example, checked as a bounded integration test.

mod common;
#[path = "dynamic_backends/portable.rs"]
mod portable;
#[path = "dynamic_backends/private_cpu.rs"]
mod private_cpu;

use akita_config::{proof_optimized::fp128, unit_onehot_source_chunk_size};
use akita_cpu_backend::{CpuBackend, OneHotPoly};
use akita_error::AkitaError;
use akita_params::{lagrange_weights, BasisMode};
use akita_pcs::{
    batched_prove, BackendId, BackendRegistry, FixedFoldRoute, FoldExecutionPolicy,
    SelectedProverOpeningData,
};
use akita_prover::backend::FoldExecutionRequirements;
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::CanonicalEncoding;

type Cfg = fp128::OneHot;
type F = fp128::Field;
type E = F;

use private_cpu::{Fault, PrivateCpu};

const NUM_VARS: usize = 15;
const NUM_POLYS: usize = 4;
const DOMAIN: &[u8] = b"akita/test/dynamic-onehot/v1";

struct RecordingRoute<'a> {
    route: FixedFoldRoute,
    source: &'a PrivateCpu,
    owners: Vec<BackendId>,
}

impl FoldExecutionPolicy for RecordingRoute<'_> {
    fn choose_backend(
        &mut self,
        current: Option<BackendId>,
        requirements: &FoldExecutionRequirements<'_>,
    ) -> Result<BackendId, AkitaError> {
        assert_eq!(requirements.level(), self.owners.len());
        assert_eq!(current, self.owners.last().copied());
        assert_eq!(self.source.admitted.get(), 0);
        let owner = self.route.choose_backend(current, requirements)?;
        self.owners.push(owner);
        Ok(owner)
    }
}

#[test]
fn dynamic_onehot_matches_homogeneous_proof() {
    common::init_rayon_pool();
    common::run_on_large_stack(|| run(false).unwrap());
}

#[test]
fn dynamic_backend_failures_cleanup_and_retry() {
    common::init_rayon_pool();
    common::run_on_large_stack(|| run(true).unwrap());
}

fn run(check_failures: bool) -> Result<(), AkitaError> {
    let scheme = common::load_workspace_scheme::<Cfg>()?;
    let chunk_size = unit_onehot_source_chunk_size::<Cfg>()?;
    let chunks_per_poly = (1usize << NUM_VARS) / chunk_size;
    let point: Vec<F> = (0..NUM_VARS)
        .map(|i| F::from_u128_reduced(i as u128 + 2))
        .collect();

    // Polynomial p has its one at position p in every compact chunk.
    // Its opening is the low-coordinate Lagrange weight for p: the identical
    // chunks are independent of the high coordinates, whose weights sum to one.
    let low_vars = chunk_size.trailing_zeros() as usize;
    let low_weights = lagrange_weights(&point[..low_vars])?;
    let evaluations = low_weights[..NUM_POLYS].to_vec();
    let polys = (0..NUM_POLYS)
        .map(|p| OneHotPoly::<F, u8>::new(chunk_size, vec![Some(p as u8); chunks_per_poly]))
        .collect::<Result<Vec<_>, _>>()?;

    let setup = scheme.setup_prover(NUM_VARS, NUM_POLYS)?;
    let private = PrivateCpu::new(CpuBackend::new(setup.expanded.clone())?);
    let cpu = CpuBackend::<F, E>::new(setup.expanded.clone())?;

    let (committed_group, private_handle) = private.commit_onehot(scheme.schedules(), polys)?;
    let opening = || {
        SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                evaluations.clone(),
                committed_group.clone(),
            )?])?,
            vec![private_handle.clone()],
            scheme.schedules(),
        )
    };
    let input = opening()?;
    let selection = input.selection();
    let resolved = scheme.schedules().resolve_selection(selection)?;
    let prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
        resolved.schedule(),
        input.opening_layout(),
    )?;
    let prefixes_private = private.prefixes(&setup, &prefix_ids);
    let prefixes_cpu = cpu.import_setup_prefixes(&setup.prefix_slots, &prefix_ids)?;
    let mut registry = BackendRegistry::<Cfg>::new()?;
    let private_id = registry.register(&private, &prefixes_private)?;
    let cpu_id = registry.register(&cpu, &prefixes_cpu)?;
    let levels = resolved.schedule().num_fold_levels();
    assert!(levels >= 3, "the route must exercise both directed bridges");

    if check_failures {
        let mut foreign = BackendRegistry::<Cfg>::new()?;
        let foreign_id = foreign.register(&cpu, &prefixes_cpu)?;
        for (owners, expected) in [
            (
                vec![private_id; levels - 1],
                "fixed backend route is shorter than the schedule",
            ),
            (
                {
                    let mut route = vec![private_id; levels];
                    route[levels - 1] = foreign_id;
                    route
                },
                "backend ID belongs to another registry",
            ),
            (
                {
                    let mut route = vec![private_id; levels];
                    route[levels - 1] = cpu_id;
                    route
                },
                "no bridge registered for selected backend types",
            ),
        ] {
            let error = batched_prove(
                setup.expanded.descriptor(),
                scheme.schedules(),
                &registry,
                opening()?,
                DOMAIN,
                BasisMode::Lagrange,
                &mut FixedFoldRoute::new(owners),
            )
            .unwrap_err();
            assert!(matches!(error, AkitaError::InvalidInput(message) if message == expected));
            assert_eq!(private.admitted.get(), 0);
            assert_eq!(private.commitments.get(), 0);
        }
    }
    registry.register_bridge::<PrivateCpu, CpuBackend<F, E>>()?;
    registry.register_bridge::<CpuBackend<F, E>, PrivateCpu>()?;

    // Alternate owners: PrivateCpu on even folds, stock CPU on odd folds.
    let mut route = RecordingRoute {
        route: FixedFoldRoute::new(
            (0..levels)
                .map(|level| if level % 2 == 0 { private_id } else { cpu_id })
                .collect(),
        ),
        source: &private,
        owners: Vec::new(),
    };
    let proof = batched_prove(
        setup.expanded.descriptor(),
        scheme.schedules(),
        &registry,
        input,
        DOMAIN,
        BasisMode::Lagrange,
        &mut route,
    )?;
    assert_eq!(route.owners.len(), levels);

    let exports = private.exports.get();
    let imports = private.imports.get();
    let reference = batched_prove(
        setup.expanded.descriptor(),
        scheme.schedules(),
        &registry,
        opening()?,
        DOMAIN,
        BasisMode::Lagrange,
        &mut FixedFoldRoute::new(vec![private_id; levels]),
    )?;
    assert_eq!(proof, reference);
    assert_eq!(private.exports.get(), exports);
    assert_eq!(private.imports.get(), imports);

    let verifier = scheme.verifier(scheme.setup_verifier(&setup)?)?;
    let statement = GroupBatchStatement::new(
        selection,
        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            evaluations.clone(),
            &committed_group,
        )?])?,
    )?;
    verifier.batched_verify(&proof, DOMAIN, statement, BasisMode::Lagrange)?;
    if check_failures {
        let destination = PrivateCpu::new(CpuBackend::new(setup.expanded.clone())?);
        let unused = PrivateCpu::new(CpuBackend::new(setup.expanded.clone())?);
        let prefixes_destination = destination.prefixes(&setup, &prefix_ids);
        let prefixes_unused = unused.prefixes(&setup, &prefix_ids);
        let mut registry = BackendRegistry::<Cfg>::new()?;
        let source_id = registry.register(&private, &prefixes_private)?;
        let destination_id = registry.register(&destination, &prefixes_destination)?;
        let unused_id = registry.register(&unused, &prefixes_unused)?;
        registry.register_bridge_with::<PrivateCpu, PrivateCpu>(|packet, _| Ok(packet))?;
        let prove = |include_unused| {
            let mut owners = vec![destination_id; levels];
            owners[0] = source_id;
            if include_unused {
                owners[levels - 1] = unused_id;
            }
            batched_prove(
                setup.expanded.descriptor(),
                scheme.schedules(),
                &registry,
                opening()?,
                DOMAIN,
                BasisMode::Lagrange,
                &mut FixedFoldRoute::new(owners),
            )
        };
        let backends = [&private, &destination, &unused];
        // Unselected executors must not be admitted, even if they would fail admission.
        unused.fault.set(Fault::Preparation);
        assert_eq!(prove(false)?, reference);
        assert_eq!(unused.admitted.get(), 0);
        unused.fault.set(Fault::None);
        // Finishing stops at the first error and aborts every remaining selected scope.
        for (fault, target) in [
            (Fault::Preparation, &destination),
            (Fault::BeginFold, &destination),
            (Fault::Export, &private),
            (Fault::Import, &destination),
            (Fault::SourceStage1, &private),
            (Fault::DestinationOpening, &destination),
            (Fault::Finish, &destination),
        ] {
            let before = backends.map(|b| (b.admitted.get(), b.finished.get(), b.aborted.get()));
            let folds = private.folds.get();
            let imports = destination.imports.get();
            let commitments = private.commitments.get();
            target.fault.set(fault);
            let error = prove(fault == Fault::Finish).unwrap_err();
            assert!(matches!(error, AkitaError::InvalidInput(message)
                if message == format!("injected {fault:?} failure")));
            for (index, backend) in backends.iter().enumerate() {
                let prepared = usize::from(
                    index == 0
                        || (index == 1 && fault != Fault::Preparation)
                        || (index == 2 && fault == Fault::Finish),
                );
                let finished = usize::from(fault == Fault::Finish && index == 0);
                assert_eq!(backend.admitted.get() - before[index].0, prepared);
                assert_eq!(backend.finished.get() - before[index].1, finished);
                assert_eq!(backend.aborted.get() - before[index].2, prepared - finished);
            }
            if fault == Fault::Preparation {
                assert_eq!(private.folds.get(), folds);
            }
            if matches!(fault, Fault::Preparation | Fault::BeginFold) {
                assert_eq!(private.commitments.get(), commitments);
            }
            if matches!(fault, Fault::SourceStage1 | Fault::DestinationOpening) {
                assert_eq!(destination.imports.get(), imports + 1);
            }
            target.fault.set(Fault::None);
            let finished = backends.map(|b| b.finished.get());
            let aborted = backends.map(|b| b.aborted.get());
            assert_eq!(prove(fault == Fault::Finish)?, reference);
            for (index, backend) in backends.iter().enumerate() {
                assert_eq!(
                    backend.finished.get(),
                    finished[index] + usize::from(index < 2 || fault == Fault::Finish)
                );
                assert_eq!(backend.aborted.get(), aborted[index]);
                assert_eq!(
                    backend.admitted.get(),
                    backend.finished.get() + backend.aborted.get()
                );
            }
        }
    }
    Ok(())
}
