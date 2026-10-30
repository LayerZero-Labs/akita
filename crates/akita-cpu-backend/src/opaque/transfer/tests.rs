use super::*;
use crate::{AkitaProverSetup, DensePoly, GroupContext};
use akita_config::{
    proof_optimized::fp128, test_support::workspace_schedule_catalog, SetupRequirements,
};
use akita_params::{BasisMode, PolynomialGroupLayout, ScheduleLookupKey};
use akita_prover::{
    BackendRegistry, FixedFoldRoute, SelectedProverOpeningData, SetupPrefixProverRegistry,
};
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::Ring;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

type F = fp128::Field;
type Cpu = CpuBackend<F, F>;
type Cfg = fp128::Dense;
const DOMAIN: &[u8] = b"portable-cpu-import";

#[test]
fn signed_portable_handoff_preserves_verified_proof() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| run(true))
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn packed_portable_handoff_preserves_verified_proof() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| run(false))
        .unwrap()
        .join()
        .unwrap();
}

fn signed_packet(packet: CpuExportPacket<F>) -> CpuImportPacket<F> {
    let (mut descriptor, mut sections) = packet.into_sections().unwrap();
    let digits = descriptor
        .sections
        .iter_mut()
        .find(|s| s.section == SuccessorSection::LogicalDigits)
        .unwrap();
    let SuccessorEncoding::PackedSigned { bit_width } = digits.encoding else {
        panic!("CPU packed digits")
    };
    let bytes = &mut sections
        .iter_mut()
        .find(|(s, _)| *s == SuccessorSection::LogicalDigits)
        .unwrap()
        .1;
    let packed =
        PackedSignedDigits::import_encoded(digits.coefficients, bit_width, std::mem::take(bytes))
            .unwrap();
    *bytes = packed
        .decode()
        .into_iter()
        .map(|digit| digit as u8)
        .collect();
    digits.encoding = SuccessorEncoding::SignedI8;
    digits.bytes = bytes.len();
    CpuImportPacket::new(descriptor, sections).unwrap()
}

fn run(signed: bool) {
    let nv = 14;
    let catalog =
        workspace_schedule_catalog::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked())
            .unwrap();
    let row = catalog
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(nv),
        ))
        .unwrap();
    let capacity = SetupRequirements::from_catalog(&catalog, nv, 1)
        .unwrap()
        .matrix_capacity();
    let capacity = akita_params::SetupMatrixCapacity {
        num_field_elements: capacity.num_field_elements.next_power_of_two(),
    };
    let setup = AkitaProverSetup::<F>::generate_with_capacity(nv, 1, capacity).unwrap();
    let producer = Cpu::new(setup.expanded.clone()).unwrap();
    let consumer = Cpu::new(setup.expanded.clone()).unwrap();
    let source = producer
        .import_source(vec![DensePoly::from_field_evals(
            nv,
            vec![F::from_u64(1); 1 << nv],
        )
        .unwrap()])
        .unwrap();
    let committed = producer
        .commit(
            &catalog,
            &source,
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let opening = || {
        SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                vec![F::from_u64(2); nv],
                vec![F::from_u64(1)],
                committed.committed_group.clone(),
            )
            .unwrap()])
            .unwrap(),
            vec![committed.private_handle.clone()],
            &catalog,
        )
        .unwrap()
    };
    let selection = opening().selection();
    let ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
        row.schedule(),
        opening().opening_layout(),
    )
    .unwrap();
    let mut prefixes_a = SetupPrefixProverRegistry::default();
    let mut prefixes_b = SetupPrefixProverRegistry::default();
    for id in ids {
        prefixes_a
            .insert(producer.prepare_setup_prefix(&id).unwrap())
            .unwrap();
        prefixes_b
            .insert(consumer.prepare_setup_prefix(&id).unwrap())
            .unwrap();
    }
    let mut registry = BackendRegistry::<Cfg>::new().unwrap();
    let a = registry.register(&producer, &prefixes_a).unwrap();
    let b = registry.register(&consumer, &prefixes_b).unwrap();
    let converted = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&converted);
    registry
        .register_bridge_with::<Cpu, Cpu>(move |packet, plan| {
            observed.fetch_add(1, Ordering::Relaxed);
            let packet = if signed {
                signed_packet(packet)
            } else {
                let (descriptor, sections) = packet.into_sections()?;
                CpuImportPacket::new(descriptor, sections)?
            };
            check_malformed_digits(plan, &packet);
            Ok(packet)
        })
        .unwrap();
    let levels = row.schedule().num_fold_levels();
    let prove = |owners| {
        akita_prover::batched_prove::<Cfg, _>(
            setup.expanded.descriptor(),
            &catalog,
            &registry,
            opening(),
            DOMAIN,
            BasisMode::Lagrange,
            &mut FixedFoldRoute::new(owners),
        )
        .unwrap()
    };
    let mut owners = vec![b; levels];
    owners[0] = a;
    let proof = prove(owners);
    assert_eq!(converted.load(Ordering::Relaxed), 1);
    assert_eq!(proof, prove(vec![a; levels]));
    let verifier = akita_verifier::AkitaVerifier::<Cfg>::for_selection(
        setup.to_verifier_setup(capacity).unwrap(),
        catalog,
        selection,
        None,
    )
    .unwrap();
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        vec![F::from_u64(2); nv],
        vec![F::from_u64(1)],
        &committed.committed_group,
    )
    .unwrap()])
    .unwrap();
    verifier
        .batched_verify(
            &proof,
            DOMAIN,
            GroupBatchStatement::new(selection, claims).unwrap(),
            BasisMode::Lagrange,
            None,
            akita_verifier::SetupMode::Check,
        )
        .unwrap();
    {
        use akita_prover::backend::SetupBatchBackend;
        use akita_types::setup_claims::{
            SetupBatchMemory, SetupBatchStatement, SetupPolynomialId, SetupQuery,
        };
        struct Advice<'a> {
            backend: &'a Cpu,
            domain: SetupPolynomialId,
            prepared: Vec<<Cpu as SetupBatchBackend<F, F>>::PreparedQuery>,
        }
        impl akita_verifier::SetupClaimAdvice<F> for Advice<'_> {
            fn answer(&mut self, query: &SetupQuery<F>) -> Result<F, AkitaError> {
                let (value, state) = self.backend.prepare_setup_query(
                    &self.domain,
                    query,
                    32,
                    SetupBatchMemory::Retain,
                )?;
                self.prepared.push(state);
                Ok(value)
            }
        }
        // Provisioning need not be a power of two; select the largest actual prefix.
        let domain = SetupPolynomialId::new::<F, F>(
            &setup.expanded,
            1usize << capacity.num_field_elements.ilog2(),
        )
        .unwrap();
        let mut advice = Advice {
            backend: &producer,
            domain: domain.clone(),
            prepared: Vec::new(),
        };
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            vec![F::from_u64(2); nv],
            vec![F::from_u64(1)],
            &committed.committed_group,
        )
        .unwrap()])
        .unwrap();
        let mut deferred =
            akita_verifier::SetupClaimAccumulator::new(domain.clone()).with_advice(&mut advice);
        verifier
            .batched_verify(
                &proof,
                DOMAIN,
                GroupBatchStatement::new(selection, claims).unwrap(),
                BasisMode::Lagrange,
                Some(&mut deferred),
                akita_verifier::SetupMode::Defer,
            )
            .unwrap();
        let pending = deferred.into_pending();
        assert!(!pending.is_empty());
        let statement = SetupBatchStatement::new(domain.clone(), 32, pending).unwrap();
        let batch = akita_prover::prove_setup_batch::<F, F, _>(
            &producer,
            &statement,
            b"portable-setup-batch",
            advice.prepared,
        )
        .unwrap();
        let accumulator = akita_verifier::verify_setup_batch_deferred::<F, F>(
            &domain,
            &statement,
            &batch,
            b"portable-setup-batch",
        )
        .unwrap();
        akita_verifier::decide_setup_accumulator(&setup.expanded, &accumulator).unwrap();
    }
}

fn check_malformed_digits(
    plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    packet: &CpuImportPacket<F>,
) {
    let logical = packet
        .descriptor
        .sections
        .iter()
        .find(|s| s.section == SuccessorSection::LogicalDigits)
        .unwrap();
    if logical.encoding != SuccessorEncoding::SignedI8 {
        return;
    }
    let assert_rejected = |offset: usize, digit: i8| {
        let mut sections = packet.sections.clone();
        sections
            .iter_mut()
            .find(|(s, _)| *s == SuccessorSection::LogicalDigits)
            .unwrap()
            .1[offset] = digit as u8;
        let mut malformed = CpuImportPacket::new(packet.descriptor.clone(), sections).unwrap();
        assert!(read_portable::<F>(plan, &mut malformed).is_err());
    };
    for unit in plan.witness_layout().units() {
        let group = &plan.producer_parameters().groups()[unit.group_index()];
        for (range, log_basis) in [
            (unit.z_range(), group.log_basis_open()),
            (unit.e_range(), group.log_basis_open()),
            (unit.t_range(), group.log_basis_outer()),
        ] {
            if range.is_empty() || log_basis >= 8 {
                continue;
            }
            let half = (1i16 << (log_basis - 1)) as i8;
            assert_rejected(range.start, half);
            assert_rejected(range.end - 1, -half - 1);
        }
    }
    for range in plan.witness_layout().negative_binary_support_intervals() {
        if !range.is_empty() {
            assert_rejected(range.start, 1);
            assert_rejected(range.end - 1, -2);
        }
    }
}
