//! Reusable backend ownership, statement binding, and independent proof lifetimes.
#![allow(missing_docs)]

mod common;

use akita_config::proof_optimized::fp128;
use akita_cpu_backend::{CommitmentHandle, CpuBackend, DensePoly, GroupContext};
use akita_prover::{
    ProofAdmission, ProofContext, ProofScope, ProofScopeConsumer, SelectedProverOpeningData,
};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_types::{
    BasisMode, Commitment, CommittedGroup, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims, PolynomialGroupLayout, RingVec,
};
use jolt_field::{One, Ring};
use std::sync::Arc;

type Cfg = fp128::Dense;
type F = fp128::Field;
const NV: usize = 14;
const DOMAIN: &[u8] = b"akita/owning-backend-contract";

fn claims(
    commitment: &CommittedGroup<F>,
    handle: CommitmentHandle<F, F, Cfg>,
    schedules: &akita_config::TrustedScheduleCatalog<Cfg>,
) -> SelectedProverOpeningData<'static, F, CommitmentHandle<F, F, Cfg>, F> {
    SelectedProverOpeningData::from_committed_claims::<Cfg>(
        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            vec![F::from_u64(2); NV],
            vec![F::one()],
            commitment.clone(),
        )
        .unwrap()])
        .unwrap(),
        vec![handle],
        schedules,
    )
    .unwrap()
}

#[test]
fn shared_commitment_supports_concurrent_deterministic_proofs_after_rejected_requests() {
    common::run_on_large_stack(|| {
        let scheme = common::load_workspace_scheme::<Cfg>().unwrap();
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let backend =
            Arc::new(CpuBackend::<Cfg>::new(setup.expanded.clone(), scheme.schedules()).unwrap());
        let source = backend
            .import_source(vec![DensePoly::from_field_evals(
                NV,
                vec![F::one(); 1 << NV],
            )
            .unwrap()])
            .unwrap();
        let output = backend
            .commit(
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        drop(source);

        let foreign = CpuBackend::<Cfg>::new(setup.expanded.clone(), scheme.schedules()).unwrap();
        assert!(scheme
            .batched_prove(
                &setup,
                claims(
                    &output.committed_group,
                    output.private_handle.clone(),
                    scheme.schedules()
                ),
                &foreign,
                DOMAIN,
                BasisMode::Lagrange,
            )
            .is_err());

        let mut fields = output.committed_group.rows().coeffs().to_vec();
        fields[0] += F::one();
        let changed = CommittedGroup::new(
            *output.committed_group.profile(),
            Commitment::new(RingVec::from_coeffs(fields)),
        );
        assert!(scheme
            .batched_prove(
                &setup,
                claims(&changed, output.private_handle.clone(), scheme.schedules()),
                &backend,
                DOMAIN,
                BasisMode::Lagrange,
            )
            .is_err());

        // This request reaches arithmetic with a valid owner and commitment,
        // then fails because the claimed scalar differs from the retained source.
        let false_claim = SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                vec![F::from_u64(2); NV],
                vec![F::from_u64(2)],
                output.committed_group.clone(),
            )
            .unwrap()])
            .unwrap(),
            vec![output.private_handle.clone()],
            scheme.schedules(),
        )
        .unwrap();
        assert!(scheme
            .batched_prove(&setup, false_claim, &backend, DOMAIN, BasisMode::Lagrange,)
            .is_err());

        // A public statement can arrive over the wire without carrying the
        // CPU's in-memory RingVec layout metadata.
        let mut public_bytes = Vec::new();
        output
            .committed_group
            .serialize_compressed(&mut public_bytes)
            .unwrap();
        let decoded = CommittedGroup::<F>::deserialize_compressed(&public_bytes[..], &()).unwrap();
        assert_eq!(decoded.rows().ring_dim(), 0);
        assert_eq!(
            decoded.rows().coeffs(),
            output.committed_group.rows().coeffs()
        );

        let prove = || {
            let opening = claims(&decoded, output.private_handle.clone(), scheme.schedules());
            let selection = opening.selection();
            let proof = scheme
                .batched_prove(&setup, opening, &backend, DOMAIN, BasisMode::Lagrange)
                .unwrap();
            let public = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                vec![F::from_u64(2); NV],
                vec![F::one()],
                &decoded,
            )
            .unwrap()])
            .unwrap();
            scheme
                .batched_verify(
                    &proof,
                    &scheme.setup_verifier(&setup).unwrap(),
                    DOMAIN,
                    GroupBatchStatement::new(selection, public).unwrap(),
                    BasisMode::Lagrange,
                )
                .unwrap();
            proof
        };
        let (first, second) = std::thread::scope(|scope| {
            let a = std::thread::Builder::new()
                .stack_size(common::STACK_SIZE)
                .spawn_scoped(scope, prove)
                .unwrap();
            let b = std::thread::Builder::new()
                .stack_size(common::STACK_SIZE)
                .spawn_scoped(scope, prove)
                .unwrap();
            (a.join().unwrap(), b.join().unwrap())
        });
        assert_eq!(first, second);
        assert_eq!(first, prove());
    });
}

#[test]
fn admission_rejects_wrong_context_and_scope_cleanup_preserves_other_proofs() {
    common::run_on_large_stack(|| {
        let scheme = common::load_workspace_scheme::<Cfg>().unwrap();
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let backend = CpuBackend::<Cfg>::new(setup.expanded.clone(), scheme.schedules()).unwrap();
        let source = backend
            .import_source(vec![DensePoly::from_field_evals(
                NV,
                vec![F::one(); 1 << NV],
            )
            .unwrap()])
            .unwrap();
        let output = backend
            .commit(
                &source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        let key = akita_types::AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(NV, 1));
        let schedule = scheme.schedules().resolve_key(&key).unwrap().schedule();
        let layout = key.opening_layout().unwrap();
        let mut absent_schedule = schedule.clone();
        absent_schedule.root.input_witness_len += 1;
        absent_schedule
            .validate_structure()
            .expect("the altered public plan is structurally valid");
        assert!(matches!(
            <CpuBackend<Cfg> as ProofAdmission<F, F>>::begin_proof(
                &backend,
                setup.expanded.descriptor(),
                &absent_schedule,
                &layout,
            ),
            Err(akita_pcs::AkitaError::UnsupportedSchedule(_)),
        ));
        let a = <CpuBackend<Cfg> as ProofAdmission<F, F>>::begin_proof(
            &backend,
            setup.expanded.descriptor(),
            schedule,
            &layout,
        )
        .unwrap();
        let b = <CpuBackend<Cfg> as ProofAdmission<F, F>>::begin_proof(
            &backend,
            setup.expanded.descriptor(),
            schedule,
            &layout,
        )
        .unwrap();
        let guard_a = ProofScope::admitted(&backend, a);
        let guard_b = ProofScope::admitted(&backend, b);
        let context = <CpuBackend<Cfg> as ProofAdmission<F, F>>::proof_context(
            &backend,
            guard_a.session(),
            0,
        )
        .unwrap()
        .for_group(0);
        assert!(<CpuBackend<Cfg> as ProofAdmission<F, F>>::proof_context(
            &backend,
            guard_a.session(),
            u32::MAX
        )
        .is_err());
        let wrong_setup =
            ProofContext::new(context.backend_id(), [0; 32], context.scope_id(), 0).for_group(0);
        for bad in [wrong_setup, context.for_group(1)] {
            assert!(
                <CpuBackend<Cfg> as ProofAdmission<F, F>>::validate_commitment(
                    &backend,
                    guard_a.session(),
                    &bad,
                    &output.private_handle,
                    output.committed_group.profile(),
                    output.committed_group.commitment(),
                )
                .is_err()
            );
        }
        let mut changed_profile = *output.committed_group.profile();
        changed_profile.group = PolynomialGroupLayout::new(NV, 2);
        assert!(
            <CpuBackend<Cfg> as ProofAdmission<F, F>>::validate_commitment(
                &backend,
                guard_a.session(),
                &context,
                &output.private_handle,
                &changed_profile,
                output.committed_group.commitment(),
            )
            .is_err()
        );
        <CpuBackend<Cfg> as ProofScopeConsumer>::finish_scope(&backend, guard_a.session()).unwrap();
        assert!(
            <CpuBackend<Cfg> as ProofAdmission<F, F>>::validate_commitment(
                &backend,
                guard_a.session(),
                &context,
                &output.private_handle,
                output.committed_group.profile(),
                output.committed_group.commitment(),
            )
            .is_err()
        );
        let context_b = <CpuBackend<Cfg> as ProofAdmission<F, F>>::proof_context(
            &backend,
            guard_b.session(),
            0,
        )
        .unwrap()
        .for_group(0);
        <CpuBackend<Cfg> as ProofScopeConsumer>::finish_scope(&backend, guard_b.session()).unwrap();
        assert!(
            <CpuBackend<Cfg> as ProofAdmission<F, F>>::validate_commitment(
                &backend,
                guard_b.session(),
                &context_b,
                &output.private_handle,
                output.committed_group.profile(),
                output.committed_group.commitment(),
            )
            .is_err()
        );
        let mut changed_setup = setup.expanded.descriptor().clone();
        changed_setup.setup_seed = [9; 32].into();
        assert!(<CpuBackend<Cfg> as ProofAdmission<F, F>>::begin_proof(
            &backend,
            &changed_setup,
            schedule,
            &layout
        )
        .is_err());
    });
}

#[test]
fn admission_uses_the_extension_field_owned_by_the_configuration() {
    common::run_on_large_stack(|| {
        type SmallCfg = akita_config::proof_optimized::fp32::OneHot;
        type SmallF = akita_config::proof_optimized::fp32::Field;
        type SmallE = <SmallCfg as akita_config::CommitmentConfig>::ExtField;
        let scheme = common::load_workspace_scheme::<SmallCfg>().unwrap();
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let backend =
            CpuBackend::<SmallCfg>::new(setup.expanded.clone(), scheme.schedules()).unwrap();
        let key = akita_types::AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(NV, 1));
        let row = scheme.schedules().resolve_key(&key).unwrap();
        let session = <CpuBackend<SmallCfg> as ProofAdmission<SmallF, SmallE>>::begin_proof(
            &backend,
            setup.expanded.descriptor(),
            row.schedule(),
            &key.opening_layout().unwrap(),
        )
        .unwrap();
        <CpuBackend<SmallCfg> as ProofScopeConsumer>::finish_scope(&backend, &session).unwrap();
    });
}
