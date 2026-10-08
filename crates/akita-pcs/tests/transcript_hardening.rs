#![allow(missing_docs)]

#[cfg(feature = "logging")]
#[path = "transcript_hardening/catalog_events.rs"]
mod catalog_events;
mod common;

use akita_cpu_backend::CpuBackend;
#[cfg(feature = "logging")]
use akita_params::transcript_site::{
    SITE_FAMILY_NEXT_WITNESS, SITE_FAMILY_OPENING_PAYLOAD, SITE_FAMILY_PHYSICAL_L2,
    SITE_FAMILY_STAGE1, SITE_FAMILY_STAGE2, SITE_FAMILY_STAGE3, SITE_FAMILY_SUMCHECK,
    SITE_FAMILY_TERMINAL,
};
use akita_pcs::{AkitaSponge, PROOF_STREAM_PROTOCOL};
#[cfg(feature = "logging")]
use common::mutations::{
    assert_messages_cover, message_ranges, representative_mutation_ranges,
    selected_sumcheck_protocols,
};
use common::*;
use jolt_field::One;
use jolt_transcript::{Channel, ProtocolId, ProverTranscript, VerifierTranscript};

const NUM_VARS: usize = 14;
const LABEL: &[u8] = b"hardening/onehot/native";

#[test]
fn stream_binds_session_statement_basis_and_eof() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = load_workspace_scheme::<OneHotCfg>().expect("workspace schedule catalog");
        let layout = scheme
            .schedules()
            .resolve_key(&akita_params::ScheduleLookupKey::single(
                akita_params::PolynomialGroupLayout::singleton(NUM_VARS),
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
        let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_cpu_backend::CommitOutput {
            committed_group: commitment,
            private_handle: hint,
        } = stack
            .commit(
                scheme.schedules(),
                &stack.import_source(vec![poly.clone()]).expect("source"),
                akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        let mut prover = ProverTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, LABEL);
        scheme
            .batched_prove(
                &setup,
                prove_input::<OneHotCfg>(&point, &[opening], &commitment, hint, scheme.schedules()),
                &stack,
                &mut prover,
                BasisMode::Lagrange,
            )
            .expect("prove");
        #[cfg(feature = "logging")]
        let prover_events = prover.events().to_vec();
        let proof = prover.finish();
        let verify = |candidate: &[u8], session: &[u8], claimed: F, basis| {
            scheme
                .verifier(verifier_setup.clone())
                .and_then(|verifier| {
                    verifier.verify_standalone(
                        candidate,
                        session,
                        verify_input::<OneHotCfg>(
                            &point,
                            &[claimed],
                            &commitment,
                            scheme.schedules(),
                        ),
                        basis,
                    )
                })
        };

        verify(&proof, LABEL, opening, BasisMode::Lagrange).expect("honest proof");
        let mut verifier =
            VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, LABEL, &proof);
        scheme
            .verifier(verifier_setup.clone())
            .and_then(|akita| {
                akita.batched_verify(
                    &mut verifier,
                    verify_input::<OneHotCfg>(&point, &[opening], &commitment, scheme.schedules()),
                    BasisMode::Lagrange,
                )
            })
            .expect("honest proof on the caller's transcript");
        #[cfg(feature = "logging")]
        let verifier_events = verifier.events().to_vec();
        verifier.finish().expect("honest proof is consumed exactly");

        // The caller's transcript is bound: the same proof under another
        // protocol id, or after a caller prefix the prover never absorbed,
        // rejects.
        const OTHER_PROTOCOL: ProtocolId = ProtocolId::new::<AkitaSponge>("akita-pcs/other-caller");
        for (protocol, prefix) in [
            (&OTHER_PROTOCOL, None),
            (&PROOF_STREAM_PROTOCOL, Some(&b"caller prefix"[..])),
        ] {
            let mut verifier = VerifierTranscript::<AkitaSponge>::new(protocol, LABEL, &proof);
            if let Some(prefix) = prefix {
                verifier.public_bytes(prefix);
            }
            let verified = scheme.verifier(verifier_setup.clone()).and_then(|akita| {
                akita.batched_verify(
                    &mut verifier,
                    verify_input::<OneHotCfg>(&point, &[opening], &commitment, scheme.schedules()),
                    BasisMode::Lagrange,
                )
            });
            assert!(
                verified.is_err() || verifier.finish().is_err(),
                "a proof must not verify on a different caller transcript"
            );
        }
        #[cfg(feature = "logging")]
        {
            assert!(!prover_events.is_empty());
            assert_eq!(verifier_events, prover_events);

            let messages = message_ranges(&prover_events);
            assert_messages_cover(&messages, proof.len());

            let role_ranges = representative_mutation_ranges(messages);
            for family in [
                SITE_FAMILY_SUMCHECK,
                SITE_FAMILY_OPENING_PAYLOAD,
                SITE_FAMILY_STAGE1,
                SITE_FAMILY_STAGE2,
                SITE_FAMILY_NEXT_WITNESS,
                SITE_FAMILY_TERMINAL,
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
                akita_params::SumcheckProtocol::Stage1,
                akita_params::SumcheckProtocol::Stage2,
            ] {
                assert!(
                    sumcheck_protocols.contains(&protocol),
                    "workload must exercise native {protocol:?} sumcheck messages"
                );
            }
            for (family, protocol) in [
                (
                    SITE_FAMILY_PHYSICAL_L2,
                    akita_params::SumcheckProtocol::PhysicalL2,
                ),
                (SITE_FAMILY_STAGE3, akita_params::SumcheckProtocol::Stage3),
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
                role_ranges.iter().any(|(bucket, message)| {
                    bucket.family == SITE_FAMILY_SUMCHECK && message.site.round > 0
                }),
                "workload must exercise and mutate a later sumcheck round"
            );
            for (bucket, message) in role_ranges {
                assert!(
                    message.range.end <= proof.len(),
                    "recorded proof range must be in bounds"
                );
                let mut mutated = proof.clone();
                mutated[message.range.start] ^= 1;
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
fn stream_mutations_reject_without_panicking() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = load_workspace_scheme::<DenseCfg>().expect("workspace schedule catalog");
        let poly = make_dense_poly(NUM_VARS, 0x7171);
        let point = random_point(NUM_VARS, 0x8181);
        let row = scheme
            .schedules()
            .resolve_key(&akita_params::ScheduleLookupKey::single(
                akita_params::PolynomialGroupLayout::singleton(NUM_VARS),
            ))
            .expect("layout");
        let opening = opening_from_poly_for_layout(
            &poly,
            &point,
            &row.schedule().root.params.final_group(),
            BasisMode::Lagrange,
        );
        let setup = scheme.setup_prover(NUM_VARS, 1).expect("setup");
        let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
        let akita_cpu_backend::CommitOutput {
            committed_group: commitment,
            private_handle: hint,
        } = stack
            .commit(
                scheme.schedules(),
                &stack.import_source(vec![poly.clone()]).expect("source"),
                akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        let proof = scheme
            .prove_standalone(
                &setup,
                prove_input::<DenseCfg>(&point, &[opening], &commitment, hint, scheme.schedules()),
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
                scheme
                    .verifier(verifier_setup.clone())
                    .and_then(|verifier| {
                        verifier.verify_standalone(
                            &mutated,
                            LABEL,
                            verify_input::<DenseCfg>(
                                &point,
                                &[opening],
                                &commitment,
                                scheme.schedules(),
                            ),
                            BasisMode::Lagrange,
                        )
                    })
            }));
            assert!(matches!(outcome, Ok(Err(_))));
        }
    });
}
