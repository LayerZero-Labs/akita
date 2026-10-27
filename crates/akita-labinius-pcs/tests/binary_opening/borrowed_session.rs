//! Complete ordinary PCS and LaBinius openings on one native session.

use super::common::{Case, H};
use akita_config::{proof_optimized::fp128, CommitmentConfig};
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_labinius_pcs::family::Family64;
use akita_labinius_verifier::RootOpeningMode;
use akita_params::{lagrange_weights, BasisMode};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_transcript::{new_prover_channel, new_verifier_channel};
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_field::{Ring, Zero};

type Cfg = fp128::Dense;
type F = <Cfg as CommitmentConfig>::Field;

#[test]
fn ordinary_pcs_and_complete_root_openings_share_one_session() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(|| composition(false))
        .unwrap()
        .join()
        .unwrap();
}

#[cfg(feature = "logging-transcript")]
#[test]
fn observed_gkr_frame_preserves_complete_session_events() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(|| composition(true))
        .unwrap()
        .join()
        .unwrap();
}

fn composition(interleave_gkr: bool) {
    const NV: usize = 14;
    let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap();
    let scheme = AkitaCommitmentScheme::<Cfg>::new(catalog);
    let setup = scheme.setup_prover(NV, 1).unwrap();
    let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
    let evals: Vec<F> = (0..(1 << NV))
        .map(|index| F::from_u64(index as u64))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(NV, &evals).unwrap();
    let output = backend
        .commit(
            scheme.schedules(),
            &backend.import_source(vec![poly]).unwrap(),
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let point: Vec<F> = (0..NV).map(|index| F::from_u64(index as u64 + 2)).collect();
    let value = evals
        .iter()
        .zip(lagrange_weights(&point).unwrap())
        .fold(F::zero(), |sum, (&v, w)| sum + v * w);
    let opening = || {
        SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![value],
                output.committed_group.clone(),
            )
            .unwrap()])
            .unwrap(),
            vec![output.private_handle.clone()],
            scheme.schedules(),
        )
        .unwrap()
    };
    let selection = opening().selection();
    let statement = || {
        GroupBatchStatement::new(
            selection,
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![value],
                &output.committed_group,
            )
            .unwrap()])
            .unwrap(),
        )
        .unwrap()
    };
    let verifier = scheme
        .verifier(scheme.setup_verifier(&setup).unwrap())
        .unwrap();
    let root = Case::<Family64>::new(16, RootOpeningMode::Both);
    let committed = root.prover.commit::<H>(&root.source).unwrap();
    let standalone = root
        .prover
        .open::<H>(&root.source, &committed, &root.statement())
        .unwrap();
    root.verifier
        .verify::<H>(&committed.commitment, &root.statement(), &standalone)
        .unwrap();
    let prove = |prefix: &[u8], reverse: bool, suffix: &[u8]| {
        #[cfg(feature = "logging-transcript")]
        akita_transcript::clear_thread_events();
        let mut session =
            new_prover_channel(b"aggregate-test/v1", b"application-instance").unwrap();
        akita_transcript::prover_context(&mut session, application_event(0));
        session.public_message(prefix);
        let mut challenges = Vec::new();
        for is_root in if reverse {
            [true, false]
        } else {
            [false, true]
        } {
            #[cfg(feature = "logging-transcript")]
            let earlier = akita_transcript::thread_events();
            if is_root {
                root.prover
                    .open_in_session::<H>(&root.source, &committed, &root.statement(), &mut session)
                    .unwrap();
            } else {
                scheme
                    .batched_prove_in_session(
                        &setup,
                        opening(),
                        &backend,
                        &mut session,
                        BasisMode::Lagrange,
                    )
                    .unwrap();
            }
            #[cfg(feature = "logging-transcript")]
            assert_events_retained(&earlier);
            if interleave_gkr && !is_root {
                // An application GKR-style round on the very same channel.
                akita_transcript::prover_context(&mut session, application_event(1));
                session.prover_message(&[3_u8; 32]);
                akita_transcript::prover_context(&mut session, application_event(2));
                let challenge: [u8; 32] = session.verifier_message();
                akita_transcript::prover_context(&mut session, application_event(3));
                session.prover_message(&challenge);
            }
            challenges.push(session.verifier_message::<[u8; 32]>());
        }
        let opening_end = session.narg_string().len();
        akita_transcript::prover_context(&mut session, application_event(4));
        session.public_message(suffix);
        let final_challenge: [u8; 32] = session.verifier_message();
        session.prover_message(&final_challenge);
        (
            session.narg_string().to_vec(),
            challenges,
            final_challenge,
            opening_end,
        )
    };
    let (proof, challenges, final_challenge, opening_end) = prove(b"prefix", false, b"suffix");
    #[cfg(feature = "logging-transcript")]
    let prover_events = akita_transcript::thread_events();
    let (_, changed_prefix, _, _) = prove(b"changed-prefix", false, b"suffix");
    assert_ne!(challenges, changed_prefix);
    let (_, changed_order, _, _) = prove(b"prefix", true, b"suffix");
    assert_ne!(challenges, changed_order);
    let (_, same_openings, changed_suffix, _) = prove(b"prefix", false, b"changed-suffix");
    assert_eq!(
        challenges, same_openings,
        "future data cannot change past challenges"
    );
    assert_ne!(final_challenge, changed_suffix);
    let replay = |proof: &[u8],
                  domain: &[u8],
                  instance: &[u8],
                  prefix: &[u8],
                  reverse: bool,
                  suffix: &[u8]| {
        #[cfg(feature = "logging-transcript")]
        akita_transcript::clear_thread_events();
        let mut session = new_verifier_channel(domain, instance, proof)?;
        akita_transcript::verifier_context(&mut session, application_event(0));
        session.public_message(prefix);
        for is_root in if reverse {
            [true, false]
        } else {
            [false, true]
        } {
            #[cfg(feature = "logging-transcript")]
            let earlier = akita_transcript::thread_events();
            if is_root {
                root.verifier.verify_in_session::<H>(
                    &committed.commitment,
                    &root.statement(),
                    &mut session,
                )?;
            } else {
                verifier.batched_verify_in_session(
                    &mut session,
                    statement(),
                    BasisMode::Lagrange,
                )?;
            }
            #[cfg(feature = "logging-transcript")]
            assert_events_retained(&earlier);
            if interleave_gkr && !is_root {
                akita_transcript::verifier_context(&mut session, application_event(1));
                assert_eq!(session.prover_message::<[u8; 32]>()?, [3_u8; 32]);
                akita_transcript::verifier_context(&mut session, application_event(2));
                let challenge: [u8; 32] = session.verifier_message()?;
                akita_transcript::verifier_context(&mut session, application_event(3));
                assert_eq!(session.prover_message::<[u8; 32]>()?, challenge);
            }
            let _: [u8; 32] = session.verifier_message()?;
        }
        akita_transcript::verifier_context(&mut session, application_event(4));
        session.public_message(suffix);
        let expected: [u8; 32] = session.verifier_message()?;
        let received: [u8; 32] = session.prover_message()?;
        if expected != received {
            return Err(AkitaError::InvalidProof);
        }
        session.check_eof()
    };
    replay(
        &proof,
        b"aggregate-test/v1",
        b"application-instance",
        b"prefix",
        false,
        b"suffix",
    )
    .unwrap();
    #[cfg(feature = "logging-transcript")]
    {
        assert!(!prover_events.is_empty());
        assert_eq!(akita_transcript::thread_events(), prover_events);
        assert_eq!(
            prover_events.first(),
            Some(&akita_transcript::TranscriptEvent::Context(
                application_event(0)
            ))
        );
        assert_eq!(
            prover_events.last(),
            Some(&akita_transcript::TranscriptEvent::Context(
                application_event(4)
            ))
        );
        if interleave_gkr {
            for stage in 1..=3 {
                assert!(
                    prover_events.contains(&akita_transcript::TranscriptEvent::Context(
                        application_event(stage)
                    ))
                );
            }
        }
    }
    // The second constructor argument binds application instance context, not the hash suite.
    for (domain, instance, prefix, reverse, suffix) in [
        (
            b"wrong-domain".as_slice(),
            b"application-instance".as_slice(),
            b"prefix".as_slice(),
            false,
            b"suffix".as_slice(),
        ),
        (
            b"aggregate-test/v1",
            b"wrong-instance",
            b"prefix",
            false,
            b"suffix",
        ),
        (
            b"aggregate-test/v1",
            b"application-instance",
            b"wrong-prefix",
            false,
            b"suffix",
        ),
        (
            b"aggregate-test/v1",
            b"application-instance",
            b"prefix",
            true,
            b"suffix",
        ),
        (
            b"aggregate-test/v1",
            b"application-instance",
            b"prefix",
            false,
            b"wrong-suffix",
        ),
    ] {
        assert!(replay(&proof, domain, instance, prefix, reverse, suffix).is_err());
    }
    let mut corrupted = proof.clone();
    corrupted[opening_end / 2] ^= 1;
    for bad in [&proof[..opening_end - 1], corrupted.as_slice()] {
        assert!(replay(
            bad,
            b"aggregate-test/v1",
            b"application-instance",
            b"prefix",
            false,
            b"suffix"
        )
        .is_err());
    }
    let mut trailing = proof;
    trailing.push(0);
    assert!(replay(
        &trailing,
        b"aggregate-test/v1",
        b"application-instance",
        b"prefix",
        false,
        b"suffix"
    )
    .is_err());
}

fn application_event(stage: u8) -> akita_transcript::ProtocolContextRecord {
    use akita_transcript::{ProtocolContextRecord, ProtocolMessageKind};
    let kind = match stage {
        1 | 3 => ProtocolMessageKind::ProofAtoms,
        2 => ProtocolMessageKind::Challenge,
        _ => ProtocolMessageKind::PublicValue,
    };
    let (encoded_bytes, challenge_bytes) = match stage {
        1 | 3 => (32, 0),
        2 => (0, 32),
        _ => (6, 0),
    };
    ProtocolContextRecord::new(
        [0xa0 + stage; 32],
        kind as u32,
        1,
        encoded_bytes,
        challenge_bytes,
    )
}

#[cfg(feature = "logging-transcript")]
fn assert_events_retained(earlier: &[akita_transcript::TranscriptEvent]) {
    let events = akita_transcript::thread_events();
    assert!(!earlier.is_empty());
    assert!(events.len() > earlier.len(), "opening must record events");
    assert!(
        events.starts_with(earlier),
        "opening erased earlier application/opening events"
    );
}
