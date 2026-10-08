//! Dynamic execution preserves transcript bytes and typed kernel ownership.

mod common;
#[path = "dynamic_backends/private_cpu.rs"]
mod private_cpu;

use akita_config::proof_optimized::fp128;
use akita_cpu_backend::{CpuBackend, CpuImportPacket, DensePoly, GroupContext};
use akita_params::{BasisMode, PolynomialGroupLayout, ScheduleLookupKey};
use akita_prover::{batched_prove, BackendRegistry, FixedFoldRoute, SelectedProverOpeningData};
use akita_types::{CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::{One, Ring};
use private_cpu::{Fault, PrivateCpu};

type Cfg = fp128::Dense;
type F = fp128::Field;
type E = F;
const NV: usize = 14;
const DOMAIN: &[u8] = b"akita/dynamic-backends";

fn claims<H: akita_prover::CommitmentHandleMetadata>(
    group: &CommittedGroup<F>,
    handle: H,
    schedules: &akita_config::TrustedScheduleCatalog<Cfg>,
) -> SelectedProverOpeningData<'static, F, H, F> {
    SelectedProverOpeningData::from_committed_claims::<Cfg>(
        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            vec![F::from_u64(2); NV],
            vec![F::one()],
            group.clone(),
        )
        .unwrap()])
        .unwrap(),
        vec![handle],
        schedules,
    )
    .unwrap()
}

#[test]
fn heterogeneous_routes_and_failure_cleanup() {
    common::run_on_large_stack(|| {
        let scheme = common::load_workspace_scheme::<Cfg>().unwrap();
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let row = scheme
            .schedules()
            .resolve_key(&ScheduleLookupKey::single(PolynomialGroupLayout::new(
                NV, 1,
            )))
            .unwrap();
        let ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            row.schedule(),
            &ScheduleLookupKey::single(PolynomialGroupLayout::new(NV, 1))
                .opening_layout()
                .unwrap(),
        )
        .unwrap();
        let a = PrivateCpu::new(CpuBackend::new(setup.expanded.clone()).unwrap());
        let b = PrivateCpu::new(CpuBackend::new(setup.expanded.clone()).unwrap());
        let cpu = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
        let cpu_second = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
        let pa = a.prefixes(&setup, &ids);
        let pb = b.prefixes(&setup, &ids);
        let pc = cpu
            .import_setup_prefixes(&setup.prefix_slots, &ids)
            .unwrap();
        let pc_second = cpu_second
            .import_setup_prefixes(&setup.prefix_slots, &ids)
            .unwrap();
        let (group, handle) = a.commit_root(scheme.schedules());
        let source = cpu
            .import_source(vec![DensePoly::from_field_evals(
                NV,
                vec![F::one(); 1 << NV],
            )
            .unwrap()])
            .unwrap();
        let ordinary = cpu
            .commit(
                scheme.schedules(),
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        assert_eq!(
            ordinary.committed_group.rows().coeffs(),
            group.rows().coeffs()
        );
        let reference = scheme
            .batched_prove(
                &setup,
                claims(
                    &ordinary.committed_group,
                    ordinary.private_handle,
                    scheme.schedules(),
                ),
                &cpu,
                DOMAIN,
                BasisMode::Lagrange,
            )
            .unwrap();
        let mut registry = BackendRegistry::<Cfg>::new().unwrap();
        let ra = registry.register(&a, &pa).unwrap();
        let rb = registry.register(&b, &pb).unwrap();
        let rc = registry.register(&cpu, &pc).unwrap();
        let rc_second = registry.register(&cpu_second, &pc_second).unwrap();
        // One owner needs no edge; a switch must fail before exporting if no edge exists.
        let levels = row.schedule().num_fold_levels();
        let no_edge = |route| {
            batched_prove(
                setup.expanded.descriptor(),
                scheme.schedules(),
                &registry,
                claims(&group, handle.clone(), scheme.schedules()),
                DOMAIN,
                BasisMode::Lagrange,
                &mut FixedFoldRoute::new(route),
            )
        };
        assert_eq!(no_edge(vec![ra; levels]).unwrap(), reference);
        a.events.borrow_mut().clear();
        let mut switch = vec![ra; levels];
        switch[1] = rb;
        let a_abort = a.aborted.get();
        let b_abort = b.aborted.get();
        assert!(
            matches!(no_edge(switch), Err(akita_error::AkitaError::InvalidInput(message)) if message.contains("no bridge"))
        );
        assert!(!a.events.borrow().contains(&"export"));
        assert_eq!(a.aborted.get(), a_abort + 1);
        assert_eq!(b.aborted.get(), b_abort + 1);
        registry
            .register_bridge::<PrivateCpu, PrivateCpu>()
            .unwrap();
        registry
            .register_bridge::<PrivateCpu, CpuBackend<F, E>>()
            .unwrap();
        registry
            .register_bridge::<CpuBackend<F, E>, PrivateCpu>()
            .unwrap();
        let conversions = std::rc::Rc::new(std::cell::Cell::new(0));
        let observed = conversions.clone();
        // Both endpoint types are foreign to this application. A custom converter
        // needs no local backend wrapper or foreign trait implementation.
        registry
            .register_bridge_with::<CpuBackend<F, E>, CpuBackend<F, E>>(move |packet, _| {
                observed.set(observed.get() + 1);
                let (descriptor, sections) = packet.into_sections()?;
                CpuImportPacket::new(descriptor, sections)
            })
            .unwrap();
        assert!(registry
            .register_bridge::<CpuBackend<F, E>, CpuBackend<F, E>>()
            .is_err());
        assert!(registry
            .register_bridge_with::<PrivateCpu, PrivateCpu>(|_, _| {
                Err(akita_error::AkitaError::InvalidInput(
                    "unused converter".into(),
                ))
            })
            .is_err());
        assert!(registry
            .register_bridge::<PrivateCpu, PrivateCpu>()
            .is_err());
        assert!(registry.register(&a, &pa).is_err());
        let count = row.schedule().num_fold_levels() - 1;
        assert!(
            count >= 2,
            "fixture must contain a recursive and terminal successor"
        );
        let verify = |proof: &[u8]| {
            let opening = claims(&group, handle.clone(), scheme.schedules());
            scheme
                .verifier(scheme.setup_verifier(&setup).unwrap())
                .unwrap()
                .batched_verify(
                    proof,
                    DOMAIN,
                    GroupBatchStatement::new(
                        opening.selection(),
                        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                            vec![F::from_u64(2); NV],
                            vec![F::one()],
                            &group,
                        )
                        .unwrap()])
                        .unwrap(),
                    )
                    .unwrap(),
                    BasisMode::Lagrange,
                )
                .unwrap();
        };
        let prove = |successors: Vec<_>| {
            let mut route = vec![ra];
            route.extend(successors);
            batched_prove(
                setup.expanded.descriptor(),
                scheme.schedules(),
                &registry,
                claims(&group, handle.clone(), scheme.schedules()),
                DOMAIN,
                BasisMode::Lagrange,
                &mut FixedFoldRoute::new(route),
            )
        };
        for portable in [false, true] {
            a.portable.set(portable);
            b.portable.set(portable);
            for route in [
                vec![ra; count],
                vec![rb; count],
                (0..count)
                    .map(|i| if i + 1 == count { rc_second } else { rc })
                    .collect(),
                (0..count)
                    .map(|i| if i % 2 == 0 { rb } else { ra })
                    .collect(),
                (0..count)
                    .map(|i| if i + 1 == count { rc } else { rb })
                    .collect(),
            ] {
                a.events.borrow_mut().clear();
                b.events.borrow_mut().clear();
                let before = a.commitments.get() + b.commitments.get();
                let all_native = route.iter().all(|id| *id == ra);
                let includes_cpu = route.contains(&rc);
                let b_admitted = b.admitted.get();
                let b_finished = b.finished.get();
                let proof = prove(route).unwrap();
                assert_eq!(b.admitted.get(), b_admitted + 1);
                assert_eq!(b.finished.get(), b_finished + 1);
                assert_eq!(proof, reference);
                verify(&proof);
                if !includes_cpu {
                    assert_eq!(a.commitments.get() + b.commitments.get() - before, count);
                }
                if all_native {
                    assert_eq!(
                        a.events
                            .borrow()
                            .iter()
                            .filter(|event| **event == "begin_fold")
                            .count(),
                        row.schedule().num_fold_levels(),
                    );
                    assert!(!b.events.borrow().contains(&"begin_fold"));
                    assert!(!a
                        .events
                        .borrow()
                        .iter()
                        .any(|event| matches!(*event, "export" | "import")));
                } else {
                    let events = a.events.borrow();
                    let export = events.iter().position(|e| *e == "export").unwrap();
                    let stage = events.iter().position(|e| *e == "stage1").unwrap();
                    assert!(export < stage);
                }
            }
        }
        assert_eq!(conversions.get(), 2);
        a.portable.set(false);
        b.portable.set(false);
        // The route's root need not be the first registered instance.
        let (group_b, handle_b) = b.commit_root(scheme.schedules());
        let mut root_b_route = FixedFoldRoute::new(
            std::iter::once(rb)
                .chain(std::iter::repeat_n(ra, count))
                .collect(),
        );
        let proof = batched_prove(
            setup.expanded.descriptor(),
            scheme.schedules(),
            &registry,
            claims(&group_b, handle_b, scheme.schedules()),
            DOMAIN,
            BasisMode::Lagrange,
            &mut root_b_route,
        )
        .unwrap();
        assert_eq!(proof, reference);
        verify(&proof);
        // A route cannot send root handles to a different instance.
        assert!(batched_prove(
            setup.expanded.descriptor(),
            scheme.schedules(),
            &registry,
            claims(&group, handle.clone(), scheme.schedules()),
            DOMAIN,
            BasisMode::Lagrange,
            &mut root_b_route,
        )
        .is_err());
        // A different handle family is rejected at registry root dispatch.
        let error = batched_prove(
            setup.expanded.descriptor(),
            scheme.schedules(),
            &registry,
            claims(&group, handle.clone(), scheme.schedules()),
            DOMAIN,
            BasisMode::Lagrange,
            &mut FixedFoldRoute::new(vec![rc; count + 1]),
        )
        .unwrap_err();
        assert!(
            matches!(error, akita_error::AkitaError::InvalidInput(message) if message.contains("handle family"))
        );
        // Every session is prepared before root work; failed preparation aborts
        // earlier sessions even when the failing backend is not on the route.
        a.events.borrow_mut().clear();
        let a_abort = a.aborted.get();
        let commitments = a.commitments.get();
        b.fault.set(Fault::Preparation);
        assert!(prove(vec![ra; count]).is_err());
        assert_eq!(*a.events.borrow(), vec!["prepare_executor"]);
        assert_eq!(a.commitments.get(), commitments);
        assert_eq!(a.aborted.get(), a_abort + 1);
        b.fault.set(Fault::None);
        a.fault.set(Fault::SignedDigits);
        assert_eq!(prove(vec![rb; count]).unwrap(), reference);
        a.fault.set(Fault::None);
        for (source_fault, destination_fault) in [
            (Fault::Export, Fault::None),
            (Fault::None, Fault::Import),
            (Fault::SourceStage1, Fault::None),
            (Fault::None, Fault::DestinationOpening),
            (Fault::Descriptor, Fault::None),
            (Fault::Packet, Fault::None),
            (Fault::InvalidDigits, Fault::None),
            (Fault::InvalidFields, Fault::None),
        ] {
            a.fault.set(source_fault);
            b.fault.set(destination_fault);
            let a_abort = a.aborted.get();
            let b_abort = b.aborted.get();
            assert!(prove(vec![rb; count]).is_err());
            assert_eq!(a.aborted.get(), a_abort + 1);
            assert_eq!(b.aborted.get(), b_abort + 1);
            a.fault.set(Fault::None);
            b.fault.set(Fault::None);
            assert_eq!(prove(vec![rb; count]).unwrap(), reference);
        }
        // Admission is local: B only supports level 1; A resumes the remainder.
        *b.supported.borrow_mut() = vec![1];
        let mut route = vec![ra; count];
        route[0] = rb;
        assert_eq!(prove(route).unwrap(), reference);
        assert!(prove(vec![rb; count]).is_err());
        b.supported.borrow_mut().clear();
        let mut other = BackendRegistry::<Cfg>::new().unwrap();
        let foreign = other.register(&a, &pa).unwrap();
        assert!(prove(vec![foreign; count]).is_err());
        let admitted = a.admitted.get() + b.admitted.get();
        assert!(batched_prove(
            setup.expanded.descriptor(),
            scheme.schedules(),
            &registry,
            claims(&group, handle.clone(), scheme.schedules()),
            DOMAIN,
            BasisMode::Lagrange,
            &mut FixedFoldRoute::new(vec![foreign; count + 1]),
        )
        .is_err());
        assert_eq!(a.admitted.get() + b.admitted.get(), admitted);
        // An adaptive policy receives public requirements for every level.
        struct Adaptive {
            choices: Vec<akita_prover::BackendId>,
            calls: usize,
        }
        impl akita_prover::FoldExecutionPolicy for Adaptive {
            fn choose_backend(
                &mut self,
                current: Option<akita_prover::BackendId>,
                req: &akita_prover::backend::FoldExecutionRequirements<'_>,
            ) -> Result<akita_prover::BackendId, akita_error::AkitaError> {
                assert_eq!(req.level(), self.calls);
                assert_eq!(
                    current,
                    req.level()
                        .checked_sub(1)
                        .map(|level| self.choices[level % self.choices.len()])
                );
                self.calls += 1;
                let chosen = self.choices[req.level() % self.choices.len()];
                Ok(chosen)
            }
        }
        let mut policy = Adaptive {
            choices: vec![ra, rb],
            calls: 0,
        };
        assert_eq!(
            batched_prove(
                setup.expanded.descriptor(),
                scheme.schedules(),
                &registry,
                claims(&group, handle, scheme.schedules()),
                DOMAIN,
                BasisMode::Lagrange,
                &mut policy
            )
            .unwrap(),
            reference
        );
        assert_eq!(policy.calls, count + 1);
        assert_eq!(a.admitted.get(), a.finished.get() + a.aborted.get());
        assert_eq!(b.admitted.get(), b.finished.get() + b.aborted.get());
    });
}

// Instantiate the same private backend under extension fields and chunked profiles.
macro_rules! dynamic_family {
    ($module:ident, $cfg:ty, $nv:expr) => {
        mod $module {
            use super::common;
            type Cfg = $cfg;
            type F = <Cfg as akita_config::CommitmentConfig>::Field;
            type E = <Cfg as akita_config::CommitmentConfig>::ExtField;
            const NV: usize = $nv;
            mod private_cpu {
                include!("dynamic_backends/private_cpu.rs");
            }
            #[test]
            fn portable_route_matches_homogeneous() {
                common::run_on_large_stack(|| {
                    use akita_cpu_backend::CpuBackend;
                    use akita_params::{BasisMode, PolynomialGroupLayout, ScheduleLookupKey};
                    use akita_prover::{
                        BackendRegistry, FixedFoldRoute, SelectedProverOpeningData,
                    };
                    use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
                    use jolt_field::{One, Zero};
                    let scheme = common::load_workspace_scheme::<Cfg>().unwrap();
                    let setup = scheme.setup_prover(NV, 1).unwrap();
                    let key = ScheduleLookupKey::single(PolynomialGroupLayout::new(NV, 1));
                    let schedule = scheme.schedules().resolve_key(&key).unwrap().schedule();
                    let ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
                        schedule,
                        &key.opening_layout().unwrap(),
                    )
                    .unwrap();
                    let a = private_cpu::PrivateCpu::new(
                        CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap(),
                    );
                    let b = private_cpu::PrivateCpu::new(
                        CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap(),
                    );
                    let pa = a.prefixes(&setup, &ids);
                    let pb = b.prefixes(&setup, &ids);
                    let (group, handle) = a.commit_root(scheme.schedules());
                    let public = || {
                        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                            vec![E::zero(); NV],
                            vec![E::one()],
                            group.clone(),
                        )
                        .unwrap()])
                        .unwrap()
                    };
                    let opening = || {
                        SelectedProverOpeningData::from_committed_claims::<Cfg>(
                            public(),
                            vec![handle.clone()],
                            scheme.schedules(),
                        )
                        .unwrap()
                    };
                    let mut registry = BackendRegistry::<Cfg>::new().unwrap();
                    let ra = registry.register(&a, &pa).unwrap();
                    let rb = registry.register(&b, &pb).unwrap();
                    registry
                        .register_bridge::<private_cpu::PrivateCpu, private_cpu::PrivateCpu>()
                        .unwrap();
                    let reference = akita_prover::batched_prove::<Cfg, _>(
                        setup.expanded.descriptor(),
                        scheme.schedules(),
                        &registry,
                        opening(),
                        super::DOMAIN,
                        BasisMode::Lagrange,
                        &mut FixedFoldRoute::new(vec![ra; schedule.num_fold_levels()]),
                    )
                    .unwrap();
                    if !ids.is_empty() {
                        // Root owns only the prefixes used by its own fold and
                        // successor commitment; B owns all later fold resources.
                        let root_ids: Vec<_> = schedule
                            .root
                            .params
                            .setup_prefix()
                            .into_iter()
                            .chain(schedule.recursive_folds.first().and_then(|fold| {
                                fold.params.setup_prefix()
                            }))
                            .filter_map(|prefix| prefix.slot_id())
                            .collect();
                        assert!(root_ids.len() < ids.len());
                        let root_prefixes = a.prefixes(&setup, &root_ids);
                        let mut conflicting = akita_prover::SetupPrefixProverRegistry::default();
                        for public in pb.public_slots() {
                            let mut slot = pb.get(&public.id).unwrap().clone();
                            let mut coefficients = slot.public.commitment.rows[0].coeffs().to_vec();
                            coefficients[0] += F::one();
                            slot.public.commitment.rows[0] =
                                akita_types::RingVec::from_coeffs(coefficients);
                            conflicting.insert(slot).unwrap();
                        }
                        let mut partial = BackendRegistry::<Cfg>::new().unwrap();
                        let root = partial.register(&a, &root_prefixes).unwrap();
                        assert!(matches!(
                            partial.register(&b, &conflicting),
                            Err(akita_error::AkitaError::InvalidSetup(message))
                                if message.contains("disagree")
                        ));
                        // A rejected registration must not poison a subsequent
                        // valid registration of the same instance and shared slots.
                        let later = partial.register(&b, &pb).unwrap();
                        partial
                            .register_bridge::<private_cpu::PrivateCpu, private_cpu::PrivateCpu>()
                            .unwrap();
                        let prove_partial = |route| {
                            akita_prover::batched_prove::<Cfg, _>(
                                setup.expanded.descriptor(),
                                scheme.schedules(),
                                &partial,
                                opening(),
                                super::DOMAIN,
                                BasisMode::Lagrange,
                                &mut FixedFoldRoute::new(route),
                            )
                        };
                        // Public union membership does not grant a local handle.
                        assert!(matches!(
                            prove_partial(vec![root; schedule.num_fold_levels()]),
                            Err(akita_error::AkitaError::InvalidSetup(message))
                                if message.contains("executor lacks")
                        ));
                        let route = std::iter::once(root)
                            .chain(std::iter::repeat_n(later, schedule.num_fold_levels() - 1))
                            .collect();
                        assert_eq!(prove_partial(route).unwrap(), reference);
                    }
                    for portable in [false, true] {
                        a.portable.set(portable);
                        b.portable.set(portable);
                        let route = (0..schedule.num_fold_levels())
                            .map(|level| if level % 2 == 1 { rb } else { ra })
                            .collect();
                        let proof = akita_prover::batched_prove(
                            setup.expanded.descriptor(),
                            scheme.schedules(),
                            &registry,
                            opening(),
                            super::DOMAIN,
                            BasisMode::Lagrange,
                            &mut FixedFoldRoute::new(route),
                        )
                        .unwrap();
                        assert_eq!(proof, reference);
                        scheme
                            .verifier(scheme.setup_verifier(&setup).unwrap())
                            .unwrap()
                            .batched_verify(
                                &proof,
                                super::DOMAIN,
                                GroupBatchStatement::new(
                                    opening().selection(),
                                    OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                                        vec![E::zero(); NV],
                                        vec![E::one()],
                                        &group,
                                    )
                                    .unwrap()])
                                    .unwrap(),
                                )
                                .unwrap(),
                                BasisMode::Lagrange,
                            )
                            .unwrap();
                    }
                    if <E as jolt_field::ExtField<F>>::DEGREE != 1 {
                        // Tensor import validates logical digits before deriving
                        // the transformed source, with either public encoding.
                        let route = || {
                            std::iter::once(ra)
                                .chain(std::iter::repeat_n(rb, schedule.num_fold_levels() - 1))
                                .collect()
                        };
                        a.fault.set(private_cpu::Fault::InvalidDigits);
                        assert!(akita_prover::batched_prove::<Cfg, _>(
                            setup.expanded.descriptor(),
                            scheme.schedules(),
                            &registry,
                            opening(),
                            super::DOMAIN,
                            BasisMode::Lagrange,
                            &mut FixedFoldRoute::new(route()),
                        )
                        .is_err());
                        a.fault.set(private_cpu::Fault::SignedDigits);
                        assert_eq!(
                            akita_prover::batched_prove::<Cfg, _>(
                                setup.expanded.descriptor(),
                                scheme.schedules(),
                                &registry,
                                opening(),
                                super::DOMAIN,
                                BasisMode::Lagrange,
                                &mut FixedFoldRoute::new(route()),
                            )
                            .unwrap(),
                            reference,
                        );
                        a.fault.set(private_cpu::Fault::None);
                    }
                });
            }
        }
    };
}
dynamic_family!(
    extension_field,
    akita_config::proof_optimized::fp32::OneHot,
    14
);
dynamic_family!(
    multiple_chunks,
    akita_config::proof_optimized::fp128::DenseMultiChunk,
    16
);

dynamic_family!(
    recursive_setup,
    akita_config::RecursiveCommitmentConfig<akita_config::proof_optimized::fp128::Dense>,
    20
);

#[test]
fn ordered_root_groups_survive_portable_handoff() {
    common::run_on_large_stack(|| {
        use akita_params::PrecommittedGroupProfiles;
        let scheme = common::load_workspace_scheme::<Cfg>().unwrap();
        let setup = scheme.setup_prover(16, 2).unwrap();
        let cpu = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
        let commit_source = |nv| {
            cpu.import_source(vec![DensePoly::from_field_evals(
                nv,
                vec![F::one(); 1 << nv],
            )
            .unwrap()])
                .unwrap()
        };
        let pre = cpu
            .commit(
                scheme.schedules(),
                &commit_source(14),
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        let profiles =
            PrecommittedGroupProfiles::from_profiles(vec![*pre.committed_group.profile()]).unwrap();
        let final_group = cpu
            .commit(
                scheme.schedules(),
                &commit_source(16),
                GroupContext::scheduler_with_precommitted_groups(&profiles),
            )
            .unwrap();
        let public = || {
            OpeningClaims::from_groups(vec![
                PolynomialGroupClaims::new(
                    vec![F::from_u64(2); 14],
                    vec![F::one()],
                    pre.committed_group.clone(),
                )
                .unwrap(),
                PolynomialGroupClaims::new(
                    vec![F::from_u64(2); 16],
                    vec![F::one()],
                    final_group.committed_group.clone(),
                )
                .unwrap(),
            ])
            .unwrap()
        };
        let opening = || {
            SelectedProverOpeningData::from_committed_claims::<Cfg>(
                public(),
                vec![
                    pre.private_handle.clone(),
                    final_group.private_handle.clone(),
                ],
                scheme.schedules(),
            )
            .unwrap()
        };
        let row = scheme
            .schedules()
            .resolve_selection(opening().selection())
            .unwrap();
        let ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            row.schedule(),
            opening().opening_layout(),
        )
        .unwrap();
        let pc = cpu
            .import_setup_prefixes(&setup.prefix_slots, &ids)
            .unwrap();
        let destination = PrivateCpu::new(CpuBackend::new(setup.expanded.clone()).unwrap());
        destination.portable.set(true);
        let pd = destination.prefixes(&setup, &ids);
        let reference = scheme
            .batched_prove(&setup, opening(), &cpu, DOMAIN, BasisMode::Lagrange)
            .unwrap();
        let mut registry = BackendRegistry::<Cfg>::new().unwrap();
        let root = registry.register(&cpu, &pc).unwrap();
        let next = registry.register(&destination, &pd).unwrap();
        registry
            .register_bridge::<CpuBackend<F, E>, PrivateCpu>()
            .unwrap();
        let proof = batched_prove(
            setup.expanded.descriptor(),
            scheme.schedules(),
            &registry,
            opening(),
            DOMAIN,
            BasisMode::Lagrange,
            &mut FixedFoldRoute::new(
                std::iter::once(root)
                    .chain(std::iter::repeat_n(
                        next,
                        row.schedule().num_fold_levels() - 1,
                    ))
                    .collect(),
            ),
        )
        .unwrap();
        assert_eq!(proof, reference);
        let verifier_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(
                vec![F::from_u64(2); 14],
                vec![F::one()],
                &pre.committed_group,
            )
            .unwrap(),
            PolynomialGroupClaims::new(
                vec![F::from_u64(2); 16],
                vec![F::one()],
                &final_group.committed_group,
            )
            .unwrap(),
        ])
        .unwrap();
        scheme
            .verifier(scheme.setup_verifier(&setup).unwrap())
            .unwrap()
            .batched_verify(
                &proof,
                DOMAIN,
                GroupBatchStatement::new(opening().selection(), verifier_claims).unwrap(),
                BasisMode::Lagrange,
            )
            .unwrap();
    });
}
