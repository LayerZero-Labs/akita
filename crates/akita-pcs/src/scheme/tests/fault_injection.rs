//! The verifier must reject transcript-consistent proofs from a prover that
//! violates one verifier-enforced condition.

use super::*;
use akita_prover::fault_injection::{with_fault, Fault, FaultReport, OverBound, WitnessSegment};

const NUM_VARS: usize = 16;

/// Faulty proving outcome: the prover's clean error, or the verifier's result.
enum FaultOutcome {
    ProverError(AkitaError),
    Verified(Result<(), AkitaError>),
}

fn prove_and_verify_with_fault(fault: Fault) -> (FaultOutcome, FaultReport) {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || {
            let scheme = workspace_scheme::<Cfg>().expect("workspace schedule artifact");
            let alpha = D.trailing_zeros() as usize;
            let layout = singleton_layout(&scheme, NUM_VARS);
            let num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;
            let (poly, evals) = make_dense_poly(num_vars);
            let setup = scheme.setup_prover(num_vars, 1).unwrap();
            let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
            let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
            let akita_cpu_backend::CommitOutput {
                committed_group: commitment,
                private_handle,
            } = stack
                .commit(
                    scheme.schedules(),
                    &stack.import_source(vec![poly]).expect("source"),
                    akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
                )
                .unwrap();
            let point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
            let opening = evals
                .iter()
                .zip(lagrange_weights(&point).unwrap().iter())
                .fold(F::zero(), |sum, (&c, &w)| sum + c * w);

            let (proof, report) = with_fault(fault, || {
                scheme.batched_prove(
                    &setup,
                    prover_claims(&scheme, &point, &[opening], &commitment, private_handle),
                    &stack,
                    b"test/fault-injection",
                    BasisMode::Lagrange,
                )
            });
            let outcome = match proof {
                Err(error) => FaultOutcome::ProverError(error),
                Ok(proof) => FaultOutcome::Verified(
                    scheme
                        .verifier(verifier_setup)
                        .and_then(|verifier| {
                            verifier.batched_verify(
                                &proof,
                                b"test/fault-injection",
                                verifier_claims(&scheme, &point, &[opening], &commitment),
                                BasisMode::Lagrange,
                            )
                        })
                        .map(|_| ()),
                ),
            };
            (outcome, report)
        })
        .expect("fault-injection test thread")
        .join()
        .expect("fault-injection test thread panicked")
}

fn schedule() -> akita_types::FoldSchedule {
    let scheme = workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    let key = akita_types::ScheduleLookupKey::single(akita_types::PolynomialGroupLayout::new(
        NUM_VARS, 1,
    ));
    let selection = scheme.schedules().resolve_key(&key).expect("schedule row");
    selection.schedule().clone()
}

fn terminal_level() -> u32 {
    u32::try_from(schedule().recursive_folds.len() + 1).unwrap()
}

/// First non-terminal fold level on the L2 security route, if any.
fn l2_route_level() -> Option<u32> {
    let schedule = schedule();
    std::iter::once(&schedule.root.params)
        .chain(schedule.recursive_folds.iter().map(|fold| &fold.params))
        .position(|params| {
            matches!(
                params.inner().matrix.security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            )
        })
        .map(|level| u32::try_from(level).unwrap())
}

/// An applied fault must never yield an accepted proof.
fn assert_rejected(fault: Fault) {
    let (outcome, report) = prove_and_verify_with_fault(fault);
    assert!(report.applied > 0, "{fault:?} was not applied");
    match outcome {
        FaultOutcome::ProverError(error) => {
            panic!("{fault:?} should still produce a proof, but proving failed: {error:?}")
        }
        FaultOutcome::Verified(result) => {
            assert!(result.is_err(), "verifier accepted a proof with {fault:?}");
        }
    }
}

#[test]
fn unapplied_fault_keeps_proof_valid() {
    let fault = Fault::PerturbWitnessDigit {
        level: u32::MAX,
        segment: WitnessSegment::Z,
        index: 0,
        delta: 1,
    };
    let (outcome, report) = prove_and_verify_with_fault(fault);
    assert_eq!(report.applied, 0);
    match outcome {
        FaultOutcome::Verified(result) => result.expect("honest proof must verify"),
        FaultOutcome::ProverError(error) => panic!("honest proving failed: {error:?}"),
    }
}

#[test]
fn perturbed_terminal_response_is_rejected() {
    assert_rejected(Fault::PerturbTerminalResponse { index: 0, delta: 1 });
}

#[test]
fn perturbed_root_response_digit_is_rejected() {
    assert_rejected(Fault::PerturbWitnessDigit {
        level: 0,
        segment: WitnessSegment::Z,
        index: 0,
        delta: 1,
    });
}

#[test]
fn perturbed_relation_quotient_digit_is_rejected() {
    assert_rejected(Fault::PerturbWitnessDigit {
        level: 0,
        segment: WitnessSegment::R,
        index: 0,
        delta: 1,
    });
}

#[test]
fn perturbed_norm_claim_is_rejected() {
    let Some(level) = l2_route_level() else {
        eprintln!("fixture schedule has no L2-route fold level; skipping");
        return;
    };
    assert_rejected(Fault::PerturbNormClaim { level, delta: -1 });
}

#[test]
fn accepted_rejected_nonces_are_rejected() {
    // Honest responses are rarely rejected, so only some levels find a
    // rejected nonce within the attempt budget; the others fall back to the
    // honest nonce and must still verify.
    let mut applied_levels = 0;
    for level in 0..=terminal_level() {
        let fault = Fault::AcceptRejectedNonce {
            level: Some(level),
            group: None,
        };
        let (outcome, report) = prove_and_verify_with_fault(fault);
        match outcome {
            FaultOutcome::ProverError(error) => panic!("{fault:?}: proving failed: {error:?}"),
            FaultOutcome::Verified(result) => assert_eq!(
                result.is_err(),
                report.applied > 0,
                "{fault:?}: {report:?}, verifier returned {result:?}"
            ),
        }
        applied_levels += usize::from(report.applied > 0);
    }
    assert!(applied_levels > 0, "no level produced a rejected nonce");
}

#[test]
fn responses_pushed_over_bound_are_rejected() {
    // The push applies at the first accepted nonce. The L2 route applies only
    // at L2-route levels, and a terminal push applies only if it still fits the
    // payload budget; unapplied pushes must leave the proof valid.
    let mut applied_levels = 0;
    for level in 0..=terminal_level() {
        for route in [
            OverBound::Linf {
                excess: 1,
                negative: false,
            },
            OverBound::Linf {
                excess: 1,
                negative: true,
            },
            OverBound::L2,
        ] {
            let fault = Fault::PushResponseOverBound {
                level: Some(level),
                group: None,
                index: 0,
                route,
            };
            let (outcome, report) = prove_and_verify_with_fault(fault);
            match outcome {
                FaultOutcome::ProverError(error) => panic!("{fault:?}: proving failed: {error:?}"),
                FaultOutcome::Verified(result) => assert_eq!(
                    result.is_err(),
                    report.applied > 0,
                    "{fault:?}: {report:?}, verifier returned {result:?}"
                ),
            }
            applied_levels += usize::from(report.applied > 0);
        }
    }
    assert!(
        applied_levels > 0,
        "no level pushed its response over bound"
    );
}
