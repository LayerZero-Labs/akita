#![allow(missing_docs)]

mod common;

use akita_prover::{ComputeBackendSetup, CpuBackend};
#[cfg(feature = "logging-transcript")]
use common::native_mutations::{
    assert_native_ranges_match_context, representative_native_mutation_ranges,
    selected_sumcheck_protocols,
};
use common::*;
use jolt_field::One;

const NUM_VARS: usize = 14;
const LABEL: &[u8] = b"hardening/onehot/native";

#[test]
fn native_stream_binds_session_statement_basis_and_eof() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = load_workspace_scheme::<OneHotCfg>().expect("workspace schedule catalog");
        let layout = scheme
            .schedules()
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                akita_types::PolynomialGroupLayout::singleton(NUM_VARS),
            ))
            .expect("layout")
            .schedule()
            .root
            .params
            .final_group();
        let poly = make_onehot_poly::<OneHotCfg>(NUM_VARS, 0x5151);
        let point = random_point(NUM_VARS, 0x6161);
        let opening = opening_from_poly_for_layout(&poly, &point, &layout, BasisMode::Lagrange);
        let setup = scheme.setup_prover(NUM_VARS, 1).expect("setup");
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&setup)
            .expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            prover_state: hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&poly),
                stack.commitment(),
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        #[cfg(feature = "logging-transcript")]
        akita_transcript::clear_thread_events();
        let proof = scheme
            .batched_prove(
                &setup,
                prove_input::<OneHotCfg, _>(
                    &point,
                    &[&poly],
                    &commitment,
                    hint,
                    scheme.schedules(),
                ),
                &stack,
                LABEL,
                BasisMode::Lagrange,
            )
            .expect("prove");
        #[cfg(feature = "logging-transcript")]
        let prover_events = akita_transcript::thread_events();
        #[cfg(feature = "logging-transcript")]
        let prover_ranges = akita_transcript::thread_proof_ranges();
        let verify = |candidate: &[u8], session: &[u8], claimed: F, basis| {
            scheme.batched_verify(
                candidate,
                &verifier_setup,
                session,
                verify_input::<OneHotCfg>(&point, &[claimed], &commitment, scheme.schedules()),
                basis,
            )
        };

        #[cfg(feature = "logging-transcript")]
        akita_transcript::clear_thread_events();
        verify(&proof, LABEL, opening, BasisMode::Lagrange).expect("honest proof");
        #[cfg(feature = "logging-transcript")]
        {
            let verifier_events = akita_transcript::thread_events();
            assert!(!prover_events.is_empty());
            assert_eq!(verifier_events, prover_events);
            assert_native_ranges_match_context(&prover_ranges);

            let mut ordered_ranges = prover_ranges.clone();
            ordered_ranges.sort_unstable_by_key(|range| range.start);
            let mut cursor = 0usize;
            for range in &ordered_ranges {
                assert_eq!(range.start, cursor, "native proof ranges must be gap-free");
                cursor = range.start.checked_add(range.len).expect("range end");
            }
            assert_eq!(
                cursor,
                proof.len(),
                "native proof ranges must cover the proof"
            );

            let role_ranges = representative_native_mutation_ranges(prover_ranges.clone());
            for family in [
                akita_transcript::SITE_FAMILY_SUMCHECK,
                akita_transcript::SITE_FAMILY_OPENING_PAYLOAD,
                akita_transcript::SITE_FAMILY_STAGE1,
                akita_transcript::SITE_FAMILY_STAGE2,
                akita_transcript::SITE_FAMILY_NEXT_WITNESS,
                akita_transcript::SITE_FAMILY_TERMINAL,
            ] {
                assert!(
                    role_ranges
                        .iter()
                        .any(|(bucket, _)| bucket.family == family),
                    "workload must exercise native proof-message family {family}"
                );
            }
            let sumcheck_protocols = selected_sumcheck_protocols(&role_ranges);
            for protocol in [
                akita_types::SumcheckProtocol::Stage1,
                akita_types::SumcheckProtocol::Stage2,
            ] {
                assert!(
                    sumcheck_protocols.contains(&protocol),
                    "workload must exercise native {protocol:?} sumcheck messages"
                );
            }
            for (family, protocol) in [
                (
                    akita_transcript::SITE_FAMILY_PHYSICAL_L2,
                    akita_types::SumcheckProtocol::PhysicalL2,
                ),
                (
                    akita_transcript::SITE_FAMILY_STAGE3,
                    akita_types::SumcheckProtocol::Stage3,
                ),
            ] {
                if role_ranges
                    .iter()
                    .any(|(bucket, _)| bucket.family == family)
                {
                    assert!(
                        sumcheck_protocols.contains(&protocol),
                        "fixture with {protocol:?} messages must mutate its sumcheck"
                    );
                }
            }
            assert!(
                role_ranges.iter().any(|(bucket, range)| {
                    bucket.family == akita_transcript::SITE_FAMILY_SUMCHECK
                        && akita_transcript::ProtocolSiteId::from_bytes(range.context.site_id).round
                            > 0
                }),
                "workload must exercise and mutate a later sumcheck round"
            );
            for (bucket, range) in role_ranges {
                let end = range.start.checked_add(range.len).expect("range end");
                assert!(end <= proof.len(), "recorded proof range must be in bounds");
                let mut mutated = proof.clone();
                mutated[range.start] ^= 1;
                assert!(
                    verify(&mutated, LABEL, opening, BasisMode::Lagrange).is_err(),
                    "fixed-shape mutation in native bucket={bucket:?} must reject",
                );
            }
        }
        assert!(verify(
            &proof,
            b"hardening/onehot/different-session",
            opening,
            BasisMode::Lagrange,
        )
        .is_err());
        assert!(verify(&proof, LABEL, opening, BasisMode::Monomial).is_err());
        assert!(verify(&proof, LABEL, opening + F::one(), BasisMode::Lagrange).is_err());
        assert!(verify(
            &proof[..proof.len() - 1],
            LABEL,
            opening,
            BasisMode::Lagrange,
        )
        .is_err());
        let mut trailing = proof.clone();
        trailing.push(0);
        assert!(verify(&trailing, LABEL, opening, BasisMode::Lagrange).is_err());
    });
}

#[test]
fn native_stream_mutations_reject_without_panicking() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = load_workspace_scheme::<DenseCfg>().expect("workspace schedule catalog");
        let poly = make_dense_poly(NUM_VARS, 0x7171);
        let point = random_point(NUM_VARS, 0x8181);
        let row = scheme
            .schedules()
            .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                akita_types::PolynomialGroupLayout::singleton(NUM_VARS),
            ))
            .expect("layout");
        let opening = opening_from_poly_for_layout(
            &poly,
            &point,
            &row.schedule().root.params.final_group(),
            BasisMode::Lagrange,
        );
        let setup = scheme.setup_prover(NUM_VARS, 1).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            prover_state: hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&poly),
                stack.commitment(),
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        let proof = scheme
            .batched_prove(
                &setup,
                prove_input::<DenseCfg, _>(&point, &[&poly], &commitment, hint, scheme.schedules()),
                &stack,
                LABEL,
                BasisMode::Lagrange,
            )
            .expect("prove");
        for offset in [
            0,
            proof.len() / 4,
            proof.len() / 2,
            proof.len() * 3 / 4,
            proof.len() - 1,
        ] {
            let mut mutated = proof.clone();
            mutated[offset] ^= 1;
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scheme.batched_verify(
                    &mutated,
                    &verifier_setup,
                    LABEL,
                    verify_input::<DenseCfg>(&point, &[opening], &commitment, scheme.schedules()),
                    BasisMode::Lagrange,
                )
            }));
            assert!(matches!(outcome, Ok(Err(_))));
        }
    });
}
