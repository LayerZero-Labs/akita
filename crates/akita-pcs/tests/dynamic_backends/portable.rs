//! Portable extension-field imports and executor-local setup-prefix ownership.
use super::common;
use akita_config::{proof_optimized::fp32, CommitmentConfig, RecursiveCommitmentConfig};
use akita_cpu_backend::{
    CpuBackend, CpuImportPacket, CpuSource, DensePoly, GroupContext, OneHotPoly, SuccessorSection,
};
use akita_params::{BasisMode, CommittedSourceEncoding};
use akita_pcs::{batched_prove, BackendRegistry, FixedFoldRoute, SelectedProverOpeningData};
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::Ring;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

type F = fp32::Field;
type E = fp32::ExtensionField;
type Cpu = CpuBackend<F, E>;
const DOMAIN: &[u8] = b"akita/test/portable-extension-import";

#[test]
fn portable_tensor_import_matches_homogeneous_proof() {
    common::init_rayon_pool();
    common::run_on_large_stack(|| {
        let chunk_size = akita_config::unit_onehot_source_chunk_size::<fp32::OneHot>().unwrap();
        let poly =
            OneHotPoly::<F, u8>::new(chunk_size, vec![Some(0); (1 << 16) / chunk_size]).unwrap();
        let expected =
            akita_params::lagrange_weights(&vec![E::from_u64(2); chunk_size.ilog2() as usize])
                .unwrap()[0];
        run::<fp32::OneHot, _>(16, vec![poly; 2], expected);
    });
}

#[test]
fn offloaded_route_uses_only_local_prefix_handles() {
    common::init_rayon_pool();
    common::run_on_large_stack(|| {
        // The smallest shipped fp32 recursive row with an offloaded setup prefix.
        let poly = DensePoly::from_field_evals(24, vec![F::from_u64(1); 1 << 24]).unwrap();
        run::<RecursiveCommitmentConfig<fp32::Dense>, _>(24, vec![poly], E::from_u64(1));
    });
}

fn run<Cfg, P>(nv: usize, polys: Vec<P>, expected: E)
where
    Cfg: CommitmentConfig<Field = F, ExtField = E> + 'static,
    P: CpuSource<F, E>,
{
    let scheme = common::load_workspace_scheme::<Cfg>().unwrap();
    let setup = scheme.setup_prover(nv, polys.len()).unwrap();
    let producer = Cpu::new(setup.expanded.clone()).unwrap();
    let consumer = Cpu::new(setup.expanded.clone()).unwrap();
    let num_polys = polys.len();
    let committed = producer
        .commit(
            scheme.schedules(),
            &producer.import_source(polys).unwrap(),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let opening = || {
        SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                vec![E::from_u64(2); nv],
                vec![expected; num_polys],
                committed.committed_group.clone(),
            )
            .unwrap()])
            .unwrap(),
            vec![committed.private_handle.clone()],
            scheme.schedules(),
        )
        .unwrap()
    };
    let input = opening();
    let selection = input.selection();
    let resolved = scheme.schedules().resolve_selection(selection).unwrap();
    let schedule = resolved.schedule();
    let levels = schedule.num_fold_levels();
    let all_ids =
        akita_config::required_setup_prefix_slot_ids_for_schedule(schedule, input.opening_layout())
            .unwrap();
    assert_eq!(!all_ids.is_empty(), Cfg::recursive_setup_planning());
    assert!(schedule.recursive_folds.iter().any(|fold| matches!(
        fold.params.source_encoding,
        CommittedSourceEncoding::TensorSubfieldProjection { .. }
    )));

    // Transfer after the last offloaded fold. Its producer also needs the
    // successor's prefix, so place every prefix-consuming fold on the producer.
    let first_tensor = schedule
        .recursive_folds
        .iter()
        .position(|fold| {
            matches!(
                fold.params.source_encoding,
                CommittedSourceEncoding::TensorSubfieldProjection { .. }
            )
        })
        .unwrap()
        + 1;
    let switch = std::iter::once(&schedule.root.params)
        .chain(schedule.recursive_folds.iter().map(|fold| &fold.params))
        .enumerate()
        .filter(|(_, params)| params.setup_prefix().is_some())
        .map(|(level, _)| level + 1)
        .max()
        .unwrap_or(0)
        .max(first_tensor);
    assert!(switch < levels - 1);
    let prefixes_a = producer
        .import_setup_prefixes(&setup.prefix_slots, &all_ids)
        .unwrap();
    let prefixes_b = consumer
        .import_setup_prefixes(&setup.prefix_slots, &[])
        .unwrap();
    let mut registry = BackendRegistry::<Cfg>::new().unwrap();
    let a = registry.register(&producer, &prefixes_a).unwrap();
    let b = registry.register(&consumer, &prefixes_b).unwrap();
    let converted = Arc::new(AtomicUsize::new(0));
    let tensor = Arc::new(AtomicUsize::new(0));
    let tamper = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&converted);
    let observed_tensor = Arc::clone(&tensor);
    let packet_fault = Arc::clone(&tamper);
    registry
        .register_bridge_with::<Cpu, Cpu>(move |packet, plan| {
            observed.fetch_add(1, Ordering::Relaxed);
            if matches!(
                plan.commitment().source_encoding(),
                Some(CommittedSourceEncoding::TensorSubfieldProjection { .. })
            ) {
                observed_tensor.fetch_add(1, Ordering::Relaxed);
            }
            let (mut descriptor, mut sections) = packet.into_sections()?;
            match packet_fault.load(Ordering::Relaxed) {
                1 => descriptor.metadata.handoff ^= 1,
                2 => sections
                    .iter_mut()
                    .find(|(section, _)| *section == SuccessorSection::InnerRows)
                    .unwrap()
                    .1
                    .fill(0xff),
                _ => {}
            }
            CpuImportPacket::new(descriptor, sections)
        })
        .unwrap();
    let prove = |owners| {
        batched_prove(
            setup.expanded.descriptor(),
            scheme.schedules(),
            &registry,
            opening(),
            DOMAIN,
            BasisMode::Lagrange,
            &mut FixedFoldRoute::new(owners),
        )
    };
    if !all_ids.is_empty() {
        // Public slots exist in the union, but the selected executor lacks the
        // local handle required by the first offloaded fold.
        let error = prove({
            let mut owners = vec![b; levels];
            owners[0] = a;
            owners
        })
        .unwrap_err();
        assert!(
            matches!(error, akita_error::AkitaError::InvalidSetup(message) if message == "executor lacks its setup-prefix handle")
        );
        assert_eq!(converted.load(Ordering::Relaxed), 0);
    }
    let mut owners = vec![b; levels];
    owners[..switch].fill(a);
    if all_ids.is_empty() {
        for (fault, expected_error) in [
            (1, "successor export differs from its handoff plan"),
            (2, "noncanonical successor field coefficient"),
        ] {
            tamper.store(fault, Ordering::Relaxed);
            assert!(matches!(
                prove(owners.clone()).unwrap_err(),
                akita_error::AkitaError::InvalidInput(message) if message == expected_error
            ));
        }
        tamper.store(0, Ordering::Relaxed);
    }
    let transfers_before = converted.load(Ordering::Relaxed);
    let tensors_before = tensor.load(Ordering::Relaxed);
    let proof = prove(owners).unwrap();
    assert_eq!(converted.load(Ordering::Relaxed), transfers_before + 1);
    assert_eq!(tensor.load(Ordering::Relaxed), tensors_before + 1);
    assert_eq!(proof, prove(vec![a; levels]).unwrap());
    assert_eq!(
        converted.load(Ordering::Relaxed),
        transfers_before + 1,
        "same-ID route must not convert"
    );
    let verifier = scheme
        .verifier(scheme.setup_verifier(&setup).unwrap())
        .unwrap();
    verifier
        .batched_verify(
            &proof,
            DOMAIN,
            GroupBatchStatement::new(
                selection,
                OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                    vec![E::from_u64(2); nv],
                    vec![expected; num_polys],
                    &committed.committed_group,
                )
                .unwrap()])
                .unwrap(),
            )
            .unwrap(),
            BasisMode::Lagrange,
        )
        .unwrap();
}
