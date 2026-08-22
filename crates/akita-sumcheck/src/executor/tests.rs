use super::shared::invalid;
use super::*;
use crate::{advance_eq_factored_claim, EqFactoredSumcheckProof, SumcheckProof};
use akita_field::{Fp64, HalvingField};
use akita_serialization::AkitaSerialize;
use akita_transcript::{labels, AkitaTranscript, Transcript};
use std::sync::{Arc, Mutex};

mod dense_alignment;

type F = Fp64<4294967197>;

#[derive(Clone, Debug, Eq, PartialEq)]
enum Event {
    AbsorbClaim,
    AbsorbRound,
    Squeeze(F),
    Start {
        group: usize,
        master_round: usize,
        local_round: usize,
        previous_challenge: Option<F>,
        factor: Option<(F, F)>,
        master_lift_prefix: Option<F>,
    },
    Finish {
        group: usize,
        master_round: usize,
    },
    Bind {
        group: usize,
        final_challenge: Option<F>,
    },
}

struct RecordingTranscript {
    events: Arc<Mutex<Vec<Event>>>,
    challenges: Vec<F>,
    next_challenge: usize,
}

impl RecordingTranscript {
    fn with_challenges(events: Arc<Mutex<Vec<Event>>>, challenges: Vec<F>) -> Self {
        Self {
            events,
            challenges,
            next_challenge: 0,
        }
    }

    fn record(&self, event: Event) {
        self.events.lock().expect("event log mutex").push(event);
    }
}

impl Transcript<F> for RecordingTranscript {
    fn new(_domain_label: &[u8]) -> Self {
        Self::with_challenges(Arc::new(Mutex::new(Vec::new())), vec![F::one(); 64])
    }

    fn bind_instance_bytes(&mut self, _instance_bytes: &[u8]) {}

    fn append_bytes(&mut self, _label: &[u8], _bytes: &[u8]) {}

    fn append_field(&mut self, _label: &[u8], _x: &F) {}

    fn append_serde<S: AkitaSerialize>(&mut self, label: &[u8], _s: &S) {
        if label == labels::ABSORB_SUMCHECK_CLAIM {
            self.record(Event::AbsorbClaim);
        } else if label == labels::ABSORB_SUMCHECK_ROUND {
            self.record(Event::AbsorbRound);
        }
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> F {
        let challenge = self.challenges[self.next_challenge];
        self.next_challenge += 1;
        self.record(Event::Squeeze(challenge));
        challenge
    }

    fn challenge_bytes(&mut self, _label: &[u8], len: usize) -> Vec<u8> {
        vec![0; len]
    }
}

enum FakeMessages {
    Standard,
    StandardFixed(Vec<UniPoly<F>>),
    EqFactored(Vec<EqFactoredUniPoly<F>>),
}

struct DelayedExecutor {
    group: usize,
    events: Arc<Mutex<Vec<Event>>>,
    messages: FakeMessages,
    terminals: Vec<F>,
    pending: Option<(usize, F)>,
    completed_rounds: usize,
}

impl DelayedExecutor {
    fn standard(group: usize, events: Arc<Mutex<Vec<Event>>>, terminals: Vec<F>) -> Self {
        Self {
            group,
            events,
            messages: FakeMessages::Standard,
            terminals,
            pending: None,
            completed_rounds: 0,
        }
    }

    fn eq_factored(
        group: usize,
        events: Arc<Mutex<Vec<Event>>>,
        messages: Vec<EqFactoredUniPoly<F>>,
        terminals: Vec<F>,
    ) -> Self {
        Self {
            group,
            events,
            messages: FakeMessages::EqFactored(messages),
            terminals,
            pending: None,
            completed_rounds: 0,
        }
    }

    fn standard_fixed(
        group: usize,
        events: Arc<Mutex<Vec<Event>>>,
        messages: Vec<UniPoly<F>>,
        terminals: Vec<F>,
    ) -> Self {
        Self {
            group,
            events,
            messages: FakeMessages::StandardFixed(messages),
            terminals,
            pending: None,
            completed_rounds: 0,
        }
    }

    fn record(&self, event: Event) {
        self.events.lock().expect("event log mutex").push(event);
    }
}

impl SumcheckRoundExecutor<F> for DelayedExecutor {
    fn start_round(&mut self, request: CheckedRoundRequest<'_, F>) -> Result<(), AkitaError> {
        if self.pending.is_some() {
            return Err(invalid("test executor already has a pending round"));
        }
        if request.round.local_round != self.completed_rounds {
            return Err(invalid(
                "test executor received an out-of-order local round",
            ));
        }
        let (claim, factor, master_lift_prefix) = match (&self.messages, request.context) {
            (
                FakeMessages::Standard | FakeMessages::StandardFixed(_),
                CheckedRoundContext::Standard {
                    previous_claim,
                    batching_coefficients,
                },
            ) => {
                if batching_coefficients.is_empty() {
                    return Err(invalid("test executor received no batching coefficients"));
                }
                (previous_claim, None, None)
            }
            (
                FakeMessages::EqFactored(_),
                CheckedRoundContext::EqFactored {
                    scaled_claim,
                    factor_at_zero,
                    factor_at_one,
                    master_lift_prefix,
                    batching_coefficients,
                    ..
                },
            ) => {
                if batching_coefficients.is_empty() {
                    return Err(invalid("test executor received no batching coefficients"));
                }
                (
                    scaled_claim,
                    Some((factor_at_zero, factor_at_one)),
                    Some(master_lift_prefix),
                )
            }
            _ => return Err(invalid("test executor received the wrong protocol format")),
        };
        self.record(Event::Start {
            group: self.group,
            master_round: request.round.master_round,
            local_round: request.round.local_round,
            previous_challenge: request.previous_challenge,
            factor,
            master_lift_prefix,
        });
        self.pending = Some((request.round.master_round, claim));
        Ok(())
    }

    fn finish_round(&mut self) -> Result<GroupRoundMessage<F>, AkitaError> {
        let (master_round, claim) = self
            .pending
            .take()
            .ok_or_else(|| invalid("test executor has no pending round"))?;
        self.record(Event::Finish {
            group: self.group,
            master_round,
        });
        let message = match &self.messages {
            FakeMessages::Standard => {
                GroupRoundMessage::Standard(UniPoly::from_coeffs(vec![claim.half()]))
            }
            FakeMessages::StandardFixed(messages) => GroupRoundMessage::Standard(
                messages
                    .get(self.completed_rounds)
                    .cloned()
                    .ok_or_else(|| invalid("test executor standard message is missing"))?,
            ),
            FakeMessages::EqFactored(messages) => GroupRoundMessage::EqFactored(
                messages
                    .get(self.completed_rounds)
                    .cloned()
                    .ok_or_else(|| invalid("test executor eq message is missing"))?,
            ),
        };
        self.completed_rounds += 1;
        Ok(message)
    }

    fn finish_binding(
        &mut self,
        final_challenge: Option<F>,
    ) -> Result<GroupTerminalClaims<F>, AkitaError> {
        if self.pending.is_some() {
            return Err(invalid("test executor cannot bind with pending work"));
        }
        self.record(Event::Bind {
            group: self.group,
            final_challenge,
        });
        Ok(GroupTerminalClaims {
            claims: self.terminals.clone(),
        })
    }
}

fn f(value: u64) -> F {
    F::from_u64(value)
}

fn serialized_bytes<S: AkitaSerialize>(value: &S) -> Vec<u8> {
    let mut bytes = Vec::new();
    value
        .serialize_uncompressed(&mut bytes)
        .expect("test proof serializes");
    bytes
}

fn standard_plan() -> CheckedStandardBatch {
    CheckedStandardBatch::new(
        vec![
            SumcheckMemberShape::new(3, 0),
            SumcheckMemberShape::new(1, 0),
            SumcheckMemberShape::new(3, 0),
            SumcheckMemberShape::new(0, 0),
            SumcheckMemberShape::new(1, 0),
        ],
        vec![
            SumcheckGroupSpec::new(vec![0, 2]),
            SumcheckGroupSpec::new(vec![1, 4]),
            SumcheckGroupSpec::new(vec![3]),
        ],
    )
    .expect("valid standard plan")
}

struct ConstantInstance {
    rounds: usize,
    input_claim: F,
}

impl crate::SumcheckInstanceProver<F> for ConstantInstance {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        0
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        UniPoly::from_coeffs(vec![previous_claim.half()])
    }

    fn ingest_challenge(&mut self, _round: usize, _r_round: F) {}
}

#[test]
fn standard_executor_matches_current_front_loaded_proof_and_challenges() {
    let claims = [f(8), f(2), f(8), F::one(), f(2)];
    let rounds = [3, 1, 3, 0, 1];
    let mut scalar_instances: Vec<_> = rounds
        .iter()
        .zip(claims)
        .map(|(&rounds, input_claim)| ConstantInstance {
            rounds,
            input_claim,
        })
        .collect();
    let scalar_refs: Vec<&mut (dyn crate::SumcheckInstanceProver<F> + Send)> = scalar_instances
        .iter_mut()
        .map(|instance| instance as &mut (dyn crate::SumcheckInstanceProver<F> + Send))
        .collect();
    let mut scalar_transcript = AkitaTranscript::<F>::new(b"test/executor-byte-equality");
    let (scalar_proof, scalar_point) =
        crate::prove_batched_sumcheck(scalar_refs, &mut scalar_transcript, |transcript| {
            transcript.challenge_scalar(b"test/challenge")
        })
        .expect("current scalar driver succeeds");

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut executors: Vec<Box<dyn SumcheckRoundExecutor<F>>> = vec![
        Box::new(DelayedExecutor::standard(
            0,
            Arc::clone(&events),
            vec![F::one(), F::one()],
        )),
        Box::new(DelayedExecutor::standard(
            1,
            Arc::clone(&events),
            vec![F::one(), F::one()],
        )),
        Box::new(DelayedExecutor::standard(2, events, vec![F::one()])),
    ];
    let mut executor_transcript = AkitaTranscript::<F>::new(b"test/executor-byte-equality");
    let output = prove_standard_executor_batch::<F, _, _, _>(
        &standard_plan(),
        &claims,
        &mut executors,
        &mut executor_transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect("executor shell succeeds");

    assert_eq!(
        serialized_bytes(&output.proof),
        serialized_bytes(&scalar_proof)
    );
    assert_eq!(output.master_point, scalar_point);
}

#[test]
fn standard_executor_preserves_submission_order_recurrence_and_suffixes() {
    let plan = standard_plan();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut transcript = RecordingTranscript::with_challenges(
        Arc::clone(&events),
        vec![f(2), f(3), f(5), f(7), f(11), f(13), f(17), f(19)],
    );
    let mut executors: Vec<Box<dyn SumcheckRoundExecutor<F>>> = vec![
        Box::new(DelayedExecutor::standard(
            0,
            Arc::clone(&events),
            vec![F::one(), F::one()],
        )),
        Box::new(DelayedExecutor::standard(
            1,
            Arc::clone(&events),
            vec![F::one(), F::one()],
        )),
        Box::new(DelayedExecutor::standard(
            2,
            Arc::clone(&events),
            vec![F::one()],
        )),
    ];
    let output = prove_standard_executor_batch::<F, _, _, _>(
        &plan,
        &[f(8), f(2), f(8), F::one(), f(2)],
        &mut executors,
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect("standard shell succeeds");

    assert_eq!(
        output.batching_coefficients,
        vec![f(2), f(3), f(5), f(7), f(11)]
    );
    assert_eq!(output.master_point, vec![f(13), f(17), f(19)]);
    assert_eq!(output.terminal_claims, vec![F::one(); 5]);
    assert_eq!(
        plan.member_point(&output.master_point, 0).unwrap(),
        output.master_point
    );
    assert_eq!(
        plan.member_point(&output.master_point, 1).unwrap(),
        &[f(19)]
    );
    assert_eq!(plan.member_point(&output.master_point, 3).unwrap(), &[]);

    let mut claim = f(8) * (f(2) + f(3) + f(5) + f(7) + f(11));
    for (round, poly) in output.proof.round_polys.iter().enumerate() {
        let decompressed = poly.decompress(&claim);
        assert_eq!(
            decompressed.evaluate(&F::zero()) + decompressed.evaluate(&F::one()),
            claim
        );
        claim = poly.eval_from_hint(&claim, &output.master_point[round]);
    }
    assert_eq!(claim, f(2) + f(3) + f(5) + f(7) + f(11));

    let log = events.lock().expect("event log mutex");
    let start_0 = log
        .iter()
        .position(|event| {
            *event
                == Event::Start {
                    group: 0,
                    master_round: 2,
                    local_round: 2,
                    previous_challenge: Some(f(17)),
                    factor: None,
                    master_lift_prefix: None,
                }
        })
        .expect("group 0 starts final round");
    let start_1 = log
        .iter()
        .position(|event| {
            *event
                == Event::Start {
                    group: 1,
                    master_round: 2,
                    local_round: 0,
                    previous_challenge: None,
                    factor: None,
                    master_lift_prefix: None,
                }
        })
        .expect("group 1 starts final round");
    let finish_0 = log
        .iter()
        .position(|event| {
            *event
                == Event::Finish {
                    group: 0,
                    master_round: 2,
                }
        })
        .unwrap();
    let finish_1 = log
        .iter()
        .position(|event| {
            *event
                == Event::Finish {
                    group: 1,
                    master_round: 2,
                }
        })
        .unwrap();
    assert!(start_0 < finish_0 && start_1 < finish_0);
    assert!(start_0 < finish_1 && start_1 < finish_1);
    let absorb = log
        .iter()
        .enumerate()
        .skip(finish_1)
        .find(|(_, event)| **event == Event::AbsorbRound)
        .map(|(index, _)| index)
        .unwrap();
    assert!(finish_0 < absorb && finish_1 < absorb);
    assert!(log.contains(&Event::Bind {
        group: 0,
        final_challenge: Some(f(19))
    }));
    assert!(log.contains(&Event::Bind {
        group: 1,
        final_challenge: Some(f(19))
    }));
    assert!(log.contains(&Event::Bind {
        group: 2,
        final_challenge: None
    }));
    assert!(log.contains(&Event::Start {
        group: 0,
        master_round: 1,
        local_round: 1,
        previous_challenge: Some(f(13)),
        factor: None,
        master_lift_prefix: None,
    }));
}

fn eq_terminal(
    input_claim: F,
    coefficient: F,
    equality_point: &[F],
    messages: &[EqFactoredUniPoly<F>],
    challenges: &[F],
) -> F {
    let mut scaled_claim = coefficient * input_claim;
    let mut claim_scale = F::one();
    let mut equality_scalar = F::one();
    for ((tau, message), challenge) in equality_point.iter().zip(messages).zip(challenges) {
        let factor_at_one = equality_scalar * *tau;
        let factor_at_zero = equality_scalar - factor_at_one;
        (scaled_claim, claim_scale) = advance_eq_factored_claim(
            scaled_claim,
            claim_scale,
            factor_at_zero,
            factor_at_one,
            message,
            *challenge,
        );
        equality_scalar *= *tau * *challenge + (F::one() - *tau) * (F::one() - *challenge);
    }
    scaled_claim
        * (claim_scale * coefficient)
            .inverse()
            .expect("nonzero deterministic terminal scale")
}

fn eq_eval(point: &[F], evaluation_point: &[F]) -> F {
    point
        .iter()
        .zip(evaluation_point)
        .fold(F::one(), |value, (&tau, &challenge)| {
            value * (tau * challenge + (F::one() - tau) * (F::one() - challenge))
        })
}

fn dense_q_at(table: &[F], point: &[F]) -> F {
    crate::multilinear_eval(table, point).expect("valid dense multilinear shape")
}

fn dense_source_messages(
    table: &[F],
    equality_point: &[F],
    coefficient: F,
    challenges: &[F],
) -> Vec<EqFactoredUniPoly<F>> {
    (0..equality_point.len())
        .map(|local_round| {
            let mut q_at_zero_point = challenges[..local_round].to_vec();
            q_at_zero_point.push(F::zero());
            q_at_zero_point.extend_from_slice(&equality_point[local_round + 1..]);
            EqFactoredUniPoly::from_q_coeffs(vec![
                coefficient * dense_q_at(table, &q_at_zero_point),
                F::zero(),
            ])
        })
        .collect()
}

#[test]
fn eq_executor_combines_same_factor_groups_before_transcript_absorb() {
    let equality_point = vec![f(23), f(29)];
    let plan = CheckedEqFactoredBatch::new(
        vec![
            SumcheckMemberShape::new(2, 0),
            SumcheckMemberShape::new(2, 0),
        ],
        vec![
            SumcheckGroupSpec::new(vec![0]),
            SumcheckGroupSpec::new(vec![1]),
        ],
        equality_point.clone(),
    )
    .expect("valid eq plan");
    let group_0_messages = vec![
        EqFactoredUniPoly::from_q_coeffs(vec![f(13)]),
        EqFactoredUniPoly::from_q_coeffs(vec![f(17)]),
    ];
    let group_1_messages = vec![
        EqFactoredUniPoly::from_q_coeffs(vec![f(19)]),
        EqFactoredUniPoly::from_q_coeffs(vec![f(31)]),
    ];
    let group_0_terminal = eq_terminal(
        f(31),
        f(2),
        &equality_point,
        &group_0_messages,
        &[f(5), f(7)],
    );
    let group_1_terminal = eq_terminal(
        f(37),
        f(3),
        &equality_point,
        &group_1_messages,
        &[f(5), f(7)],
    );

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut transcript =
        RecordingTranscript::with_challenges(Arc::clone(&events), vec![f(2), f(3), f(5), f(7)]);
    let mut executors: Vec<Box<dyn SumcheckRoundExecutor<F>>> = vec![
        Box::new(DelayedExecutor::eq_factored(
            0,
            Arc::clone(&events),
            group_0_messages,
            vec![group_0_terminal],
        )),
        Box::new(DelayedExecutor::eq_factored(
            1,
            Arc::clone(&events),
            group_1_messages,
            vec![group_1_terminal],
        )),
    ];
    let output = prove_eq_factored_executor_batch::<F, _, _, _>(
        &plan,
        &[f(31), f(37)],
        &mut executors,
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect("eq shell succeeds");

    assert_eq!(output.master_point, vec![f(5), f(7)]);
    assert_eq!(
        output.terminal_claims,
        vec![group_0_terminal, group_1_terminal]
    );
    assert_eq!(
        output.proof.round_polys,
        vec![
            EqFactoredUniPoly::from_q_coeffs(vec![f(32)]),
            EqFactoredUniPoly::from_q_coeffs(vec![f(48)]),
        ]
    );
    assert_eq!(plan.member_equality_point(1).unwrap(), equality_point);
    assert_eq!(
        plan.member_point(&output.master_point, 1).unwrap(),
        &[f(5), f(7)]
    );

    let log = events.lock().expect("event log mutex");
    let starts: Vec<_> = log
        .iter()
        .filter_map(|event| match event {
            Event::Start {
                group,
                master_round: 0,
                factor,
                ..
            } => Some((*group, *factor)),
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 2);
    assert_eq!(starts[0].1, starts[1].1);
    let start_1 = log
        .iter()
        .position(|event| {
            matches!(
                event,
                Event::Start {
                    group: 1,
                    master_round: 1,
                    previous_challenge: Some(challenge),
                    ..
                } if *challenge == f(5)
            )
        })
        .unwrap();
    let first_finish = log
        .iter()
        .position(|event| {
            matches!(
                event,
                Event::Finish {
                    master_round: 1,
                    ..
                }
            )
        })
        .unwrap();
    assert!(start_1 < first_finish);
    assert!(log.contains(&Event::Bind {
        group: 0,
        final_challenge: Some(f(7))
    }));
    assert!(log.contains(&Event::Bind {
        group: 1,
        final_challenge: Some(f(7))
    }));
    assert_eq!(
        log.iter()
            .filter(|event| **event == Event::AbsorbRound)
            .count(),
        2,
        "one existing proof message is absorbed per round"
    );
    drop(log);

    let verifier_events = Arc::new(Mutex::new(Vec::new()));
    let mut verifier_transcript =
        RecordingTranscript::with_challenges(verifier_events, vec![f(2), f(3), f(5), f(7)]);
    let verification = verify_eq_factored_executor_batch_rounds::<F, _, _, _>(
        &plan,
        &[f(31), f(37)],
        &output.proof,
        &mut verifier_transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect("same-round verifier replay succeeds");
    assert_eq!(verification.master_point, output.master_point);
    verification
        .check_terminal_claims(&output.terminal_claims)
        .expect("same-round terminals close the verifier recurrence");
}

#[test]
fn unequal_round_eq_master_lift_matches_dense_reference_and_verifier() {
    let equality_point = vec![f(23), f(29), f(31)];
    let coefficients = [f(2), f(3), f(5), f(7)];
    let challenges = [f(11), f(13), f(17)];
    let q_tables = [
        vec![f(2), f(3), f(5), f(7), f(11), f(13), f(17), f(19)],
        vec![f(23), f(29), f(31), f(37)],
        vec![f(41), f(43)],
        vec![f(47)],
    ];
    let local_rounds = [3, 2, 1, 0];
    let claims: Vec<_> = q_tables
        .iter()
        .zip(local_rounds)
        .map(|(table, rounds)| dense_q_at(table, &equality_point[3 - rounds..]))
        .collect();
    let plan = CheckedEqFactoredBatch::new(
        local_rounds
            .iter()
            .map(|&rounds| SumcheckMemberShape::new(rounds, 1))
            .collect(),
        vec![
            SumcheckGroupSpec::new(vec![0]),
            SumcheckGroupSpec::new(vec![1]),
            SumcheckGroupSpec::new(vec![2]),
            SumcheckGroupSpec::new(vec![3]),
        ],
        equality_point.clone(),
    )
    .expect("three unequal suffix offsets are valid");

    let events = Arc::new(Mutex::new(Vec::new()));
    let transcript_challenges: Vec<_> = coefficients.iter().chain(&challenges).copied().collect();
    let mut transcript =
        RecordingTranscript::with_challenges(Arc::clone(&events), transcript_challenges.clone());
    let terminals: Vec<_> = q_tables
        .iter()
        .zip(local_rounds)
        .map(|(table, rounds)| {
            let tau = &equality_point[3 - rounds..];
            let point = &challenges[3 - rounds..];
            eq_eval(tau, point) * dense_q_at(table, point)
        })
        .collect();
    let mut executors: Vec<Box<dyn SumcheckRoundExecutor<F>>> = q_tables
        .iter()
        .zip(local_rounds)
        .zip(coefficients)
        .zip(&terminals)
        .enumerate()
        .map(|(group, (((table, rounds), coefficient), terminal))| {
            let offset = 3 - rounds;
            Box::new(DelayedExecutor::eq_factored(
                group,
                Arc::clone(&events),
                dense_source_messages(
                    table,
                    &equality_point[offset..],
                    coefficient,
                    &challenges[offset..],
                ),
                vec![*terminal],
            )) as Box<dyn SumcheckRoundExecutor<F>>
        })
        .collect();
    let output = prove_eq_factored_executor_batch::<F, _, _, _>(
        &plan,
        &claims,
        &mut executors,
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect("master-lifted unequal-round batch succeeds");

    let expected_rounds: Vec<_> = (0..3)
        .map(|master_round| {
            let q_at_zero = q_tables
                .iter()
                .zip(local_rounds)
                .zip(coefficients)
                .zip(&claims)
                .fold(F::zero(), |sum, (((table, rounds), coefficient), claim)| {
                    let offset = 3 - rounds;
                    if master_round < offset {
                        sum + coefficient * *claim
                    } else {
                        let mut point = challenges[offset..master_round].to_vec();
                        point.push(F::zero());
                        point.extend_from_slice(&equality_point[master_round + 1..]);
                        assert_eq!(point.len(), rounds);
                        sum + coefficient * dense_q_at(table, &point)
                    }
                });
            EqFactoredUniPoly::from_q_coeffs(vec![q_at_zero, F::zero()])
        })
        .collect();
    let expected_proof = EqFactoredSumcheckProof {
        round_polys: expected_rounds,
    };
    assert_eq!(
        serialized_bytes(&output.proof),
        serialized_bytes(&expected_proof)
    );
    assert_eq!(output.master_point, challenges);
    assert_eq!(output.terminal_claims, terminals);
    for member_index in 0..4 {
        let offset = member_index;
        assert_eq!(
            plan.member_equality_point(member_index).unwrap(),
            &equality_point[offset..]
        );
        assert_eq!(
            plan.member_point(&output.master_point, member_index)
                .unwrap(),
            &challenges[offset..]
        );
    }

    let log = events.lock().expect("event log mutex");
    for (group, expected_starts) in [3, 2, 1, 0].into_iter().enumerate() {
        assert_eq!(
            log.iter()
                .filter(
                    |event| matches!(event, Event::Start { group: found, .. } if *found == group)
                )
                .count(),
            expected_starts,
            "virtual prefix rounds must not start or fold the source"
        );
    }
    let first_finish_round_two = log
        .iter()
        .position(|event| {
            matches!(
                event,
                Event::Finish {
                    master_round: 2,
                    ..
                }
            )
        })
        .unwrap();
    for group in 0..3 {
        let start = log
            .iter()
            .position(|event| {
                matches!(event, Event::Start { group: found, master_round: 2, .. } if *found == group)
            })
            .unwrap();
        assert!(start < first_finish_round_two);
    }
    assert!(log.iter().any(|event| {
        matches!(
            event,
            Event::Start {
                group: 2,
                master_round: 2,
                local_round: 0,
                previous_challenge: None,
                master_lift_prefix: Some(prefix),
                ..
            } if *prefix == eq_eval(&equality_point[..2], &challenges[..2])
        )
    }));
    drop(log);

    let verifier_events = Arc::new(Mutex::new(Vec::new()));
    let mut verifier_transcript =
        RecordingTranscript::with_challenges(Arc::clone(&verifier_events), transcript_challenges);
    let verification = verify_eq_factored_executor_batch_rounds::<F, _, _, _>(
        &plan,
        &claims,
        &output.proof,
        &mut verifier_transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect("protocol verifier replays the batch");
    assert_eq!(verification.master_point, challenges);
    assert_eq!(verification.batching_coefficients, coefficients);
    verification
        .check_terminal_claims(&terminals)
        .expect("local terminal obligations close the recurrence");
    for (member_index, obligation) in verification.terminal_obligations.iter().enumerate() {
        assert_eq!(obligation.member_index, member_index);
        assert_eq!(
            obligation.point(&verification.master_point).unwrap(),
            &challenges[member_index..]
        );
        assert_eq!(
            obligation.prefix_equality_scalar,
            eq_eval(&equality_point[..member_index], &challenges[..member_index])
        );
    }
}

#[test]
fn two_round_virtual_prefix_keeps_accumulated_scalar_in_engine_factor() {
    let equality_point = [f(5), f(7), f(11)];
    let anchor_coefficient = f(3);
    let coefficient = f(13);
    let challenges = [f(17), f(19), f(23)];
    let q_table = [f(29), f(31)];
    let input_claim = dense_q_at(&q_table, &equality_point[2..]);
    let local_terminal =
        eq_eval(&equality_point[2..], &challenges[2..]) * dense_q_at(&q_table, &challenges[2..]);
    let plan = CheckedEqFactoredBatch::new(
        vec![
            SumcheckMemberShape::new(3, 1),
            SumcheckMemberShape::new(1, 1),
        ],
        vec![
            SumcheckGroupSpec::new(vec![0]),
            SumcheckGroupSpec::new(vec![1]),
        ],
        equality_point.to_vec(),
    )
    .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut transcript = RecordingTranscript::with_challenges(
        Arc::clone(&events),
        [anchor_coefficient, coefficient]
            .into_iter()
            .chain(challenges)
            .collect(),
    );
    let mut executors: Vec<Box<dyn SumcheckRoundExecutor<F>>> = vec![
        Box::new(DelayedExecutor::eq_factored(
            0,
            Arc::clone(&events),
            vec![EqFactoredUniPoly::from_q_coeffs(vec![F::zero(); 2]); 3],
            vec![F::zero()],
        )),
        Box::new(DelayedExecutor::eq_factored(
            1,
            Arc::clone(&events),
            dense_source_messages(
                &q_table,
                &equality_point[2..],
                coefficient,
                &challenges[2..],
            ),
            vec![local_terminal],
        )),
    ];
    let output = prove_eq_factored_executor_batch::<F, _, _, _>(
        &plan,
        &[F::zero(), input_claim],
        &mut executors,
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect("two-round virtual master lift succeeds");

    let mut claim_scale = F::one();
    let mut scaled_claim = coefficient * input_claim;
    for round in 0..3 {
        let bound = &challenges[..round];
        let equality_scalar = eq_eval(&equality_point[..round], bound);
        let linear_at = |x: F| {
            equality_scalar
                * (equality_point[round] * x + (F::one() - equality_point[round]) * (F::one() - x))
        };
        let l_at_zero = linear_at(F::zero());
        let l_at_one = linear_at(F::one());
        let s_at_zero =
            dense_alignment::round_sum(&equality_point, &q_table, 2, coefficient, bound, F::zero());
        let s_at_one =
            dense_alignment::round_sum(&equality_point, &q_table, 2, coefficient, bound, F::one());
        let current_claim = s_at_zero + s_at_one;
        assert_eq!(scaled_claim, claim_scale * current_claim);

        let q_at_zero = output.proof.round_polys[round].constant_term();
        let q_linear = (current_claim - (l_at_zero + l_at_one) * q_at_zero)
            * l_at_one.inverse().expect("chosen linear factor is nonzero");
        for x in [F::zero(), F::one(), f(37)] {
            let restored_q = q_at_zero + q_linear * x;
            let direct_q = if round < 2 {
                coefficient * input_claim
            } else {
                coefficient * dense_q_at(&q_table, &[x])
            };
            assert_eq!(restored_q, direct_q, "restored q differs in round {round}");
            assert_eq!(
                linear_at(x) * restored_q,
                dense_alignment::round_sum(&equality_point, &q_table, 2, coefficient, bound, x,),
                "restored s differs in round {round}"
            );
        }

        let challenge = challenges[round];
        let l_at_challenge = linear_at(challenge);
        let scaled_linear = scaled_claim - claim_scale * (l_at_zero + l_at_one) * q_at_zero;
        let next_scale = claim_scale * l_at_one;
        let denominator_free_next =
            next_scale * l_at_challenge * q_at_zero + l_at_challenge * challenge * scaled_linear;
        let direct_next_claim =
            dense_alignment::round_sum(&equality_point, &q_table, 2, coefficient, bound, challenge);
        assert_eq!(denominator_free_next, next_scale * direct_next_claim);
        claim_scale = next_scale;
        scaled_claim = denominator_free_next;
    }

    let prefix_scalar = eq_eval(&equality_point[..2], &challenges[..2]);
    assert_eq!(
        output.proof.round_polys[2].constant_term(),
        coefficient * q_table[0],
        "the accumulated prefix scalar stays in engine-owned L, not transmitted q"
    );
    assert_ne!(prefix_scalar, F::one());
    assert_eq!(
        scaled_claim,
        claim_scale * coefficient * prefix_scalar * local_terminal
    );
    assert_eq!(output.terminal_claims, vec![F::zero(), local_terminal]);

    let log = events.lock().expect("event log mutex");
    assert_eq!(
        log.iter()
            .filter(|event| matches!(event, Event::Start { group: 1, .. }))
            .count(),
        1,
        "both virtual prefix rounds avoid the source"
    );
}

#[test]
fn all_terminal_eq_batch_uses_an_empty_existing_proof() {
    let plan = CheckedEqFactoredBatch::new(
        vec![
            SumcheckMemberShape::new(0, 0),
            SumcheckMemberShape::new(0, 0),
        ],
        vec![
            SumcheckGroupSpec::new(vec![0]),
            SumcheckGroupSpec::new(vec![1]),
        ],
        Vec::new(),
    )
    .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut transcript =
        RecordingTranscript::with_challenges(Arc::clone(&events), vec![f(2), f(3)]);
    let mut executors: Vec<Box<dyn SumcheckRoundExecutor<F>>> = vec![
        Box::new(DelayedExecutor::eq_factored(
            0,
            Arc::clone(&events),
            Vec::new(),
            vec![f(11)],
        )),
        Box::new(DelayedExecutor::eq_factored(
            1,
            Arc::clone(&events),
            Vec::new(),
            vec![f(13)],
        )),
    ];
    let output = prove_eq_factored_executor_batch::<F, _, _, _>(
        &plan,
        &[f(11), f(13)],
        &mut executors,
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .unwrap();
    assert!(output.proof.round_polys.is_empty());
    assert!(output.master_point.is_empty());
    assert_eq!(output.terminal_claims, vec![f(11), f(13)]);
    assert!(events
        .lock()
        .expect("event log mutex")
        .contains(&Event::Bind {
            group: 0,
            final_challenge: None,
        }));
}

#[test]
fn eq_verifier_rejects_malformed_and_tampered_inputs_without_panicking() {
    let tau = f(7);
    let challenge = f(3);
    let coefficient = f(2);
    let q_table = [f(3), f(5)];
    let claim = dense_q_at(&q_table, &[tau]);
    let terminal = eq_eval(&[tau], &[challenge]) * dense_q_at(&q_table, &[challenge]);
    let plan = CheckedEqFactoredBatch::new(
        vec![SumcheckMemberShape::new(1, 1)],
        vec![SumcheckGroupSpec::new(vec![0])],
        vec![tau],
    )
    .unwrap();
    let proof = EqFactoredSumcheckProof {
        round_polys: vec![EqFactoredUniPoly::from_q_coeffs(vec![
            coefficient * q_table[0],
            F::zero(),
        ])],
    };

    let shape_events = Arc::new(Mutex::new(Vec::new()));
    let mut shape_transcript = RecordingTranscript::with_challenges(
        Arc::clone(&shape_events),
        vec![coefficient, challenge],
    );
    assert_eq!(
        verify_eq_factored_executor_batch_rounds::<F, _, _, _>(
            &plan,
            &[],
            &proof,
            &mut shape_transcript,
            |transcript| transcript.challenge_scalar(b"test/challenge"),
        ),
        Err(AkitaError::InvalidSize {
            expected: 1,
            actual: 0,
        })
    );
    assert!(shape_events.lock().expect("event log mutex").is_empty());

    for malformed in [
        EqFactoredSumcheckProof {
            round_polys: Vec::new(),
        },
        EqFactoredSumcheckProof {
            round_polys: vec![EqFactoredUniPoly {
                coeffs_except_linear_term: vec![F::one(), F::one()],
            }],
        },
    ] {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut transcript =
            RecordingTranscript::with_challenges(Arc::clone(&events), vec![coefficient, challenge]);
        assert!(verify_eq_factored_executor_batch_rounds::<F, _, _, _>(
            &plan,
            &[claim],
            &malformed,
            &mut transcript,
            |transcript| transcript.challenge_scalar(b"test/challenge"),
        )
        .is_err());
        assert!(events.lock().expect("event log mutex").is_empty());
    }

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut wrong_claim_transcript =
        RecordingTranscript::with_challenges(Arc::clone(&events), vec![coefficient, challenge]);
    let wrong_claim = verify_eq_factored_executor_batch_rounds::<F, _, _, _>(
        &plan,
        &[claim + F::one()],
        &proof,
        &mut wrong_claim_transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .unwrap();
    assert_eq!(
        wrong_claim.check_terminal_claims(&[terminal]),
        Err(AkitaError::InvalidProof)
    );

    let mut tampered_proof = proof.clone();
    tampered_proof.round_polys[0].coeffs_except_linear_term[0] += F::one();
    let mut tampered_transcript =
        RecordingTranscript::with_challenges(Arc::clone(&events), vec![coefficient, challenge]);
    let tampered = verify_eq_factored_executor_batch_rounds::<F, _, _, _>(
        &plan,
        &[claim],
        &tampered_proof,
        &mut tampered_transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .unwrap();
    assert_eq!(
        tampered.check_terminal_claims(&[terminal]),
        Err(AkitaError::InvalidProof)
    );

    let mut valid_transcript =
        RecordingTranscript::with_challenges(events, vec![coefficient, challenge]);
    let valid = verify_eq_factored_executor_batch_rounds::<F, _, _, _>(
        &plan,
        &[claim],
        &proof,
        &mut valid_transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .unwrap();
    assert_eq!(
        valid.check_terminal_claims(&[]),
        Err(AkitaError::InvalidSize {
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(
        valid.check_terminal_claims(&[terminal + F::one()]),
        Err(AkitaError::InvalidProof)
    );
    valid.check_terminal_claims(&[terminal]).unwrap();
}

#[test]
fn plans_reject_empty_inconsistent_duplicate_missing_and_overflow_shapes() {
    assert!(CheckedStandardBatch::new(Vec::new(), Vec::new()).is_err());
    assert!(CheckedStandardBatch::new(
        vec![SumcheckMemberShape::new(1, 2)],
        vec![SumcheckGroupSpec::new(Vec::new())],
    )
    .is_err());
    assert!(CheckedStandardBatch::new(
        vec![
            SumcheckMemberShape::new(1, 2),
            SumcheckMemberShape::new(2, 2)
        ],
        vec![SumcheckGroupSpec::new(vec![0, 1])],
    )
    .is_err());
    assert!(CheckedStandardBatch::new(
        vec![
            SumcheckMemberShape::new(1, 2),
            SumcheckMemberShape::new(1, 2),
        ],
        vec![
            SumcheckGroupSpec::new(vec![1]),
            SumcheckGroupSpec::new(vec![0]),
        ],
    )
    .is_err());
    assert!(CheckedStandardBatch::new(
        vec![SumcheckMemberShape::new(1, 2)],
        vec![
            SumcheckGroupSpec::new(vec![0]),
            SumcheckGroupSpec::new(vec![0])
        ],
    )
    .is_err());
    assert!(CheckedStandardBatch::new(
        vec![
            SumcheckMemberShape::new(1, 2),
            SumcheckMemberShape::new(1, 2)
        ],
        vec![SumcheckGroupSpec::new(vec![0])],
    )
    .is_err());
    assert!(CheckedStandardBatch::new(
        vec![SumcheckMemberShape::new(usize::MAX, 2)],
        vec![SumcheckGroupSpec::new(vec![0])],
    )
    .is_err());
    assert!(CheckedStandardBatch::new(
        vec![SumcheckMemberShape::new(1, usize::MAX)],
        vec![SumcheckGroupSpec::new(vec![0])],
    )
    .is_err());
    assert!(CheckedEqFactoredBatch::new(
        vec![SumcheckMemberShape::new(2, 2)],
        vec![SumcheckGroupSpec::new(vec![0])],
        vec![F::one()],
    )
    .is_err());
}

#[test]
fn checked_plan_has_no_logical_batch_maximum_or_power_of_two_rule() {
    let member_count = 257;
    let plan = CheckedStandardBatch::new(
        vec![SumcheckMemberShape::new(0, 0); member_count],
        vec![SumcheckGroupSpec::new((0..member_count).collect())],
    )
    .expect("a large non-power-of-two logical batch is valid");
    assert_eq!(plan.groups()[0].member_indices().len(), member_count);
    assert_eq!(plan.master_rounds(), 0);
}

#[test]
fn suffixes_are_derived_and_wrong_master_dimensions_are_typed() {
    let plan = standard_plan();
    assert_eq!(plan.groups()[1].suffix_offset(), 2);
    assert_eq!(
        plan.member_point(&[f(2), f(3)], 1),
        Err(AkitaError::InvalidPointDimension {
            expected: 3,
            actual: 2,
        })
    );
    assert!(matches!(
        plan.member_point(&[f(2), f(3), f(5)], 99),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn proof_shape_validation_rejects_missing_empty_and_round_specific_degree() {
    let standard = CheckedStandardBatch::new(
        vec![SumcheckMemberShape::new(1, 1)],
        vec![SumcheckGroupSpec::new(vec![0])],
    )
    .unwrap();
    assert!(standard
        .validate_proof(&SumcheckProof::<F> {
            round_polys: Vec::new()
        })
        .is_err());
    assert!(standard
        .validate_proof(&SumcheckProof {
            round_polys: vec![crate::CompressedUniPoly {
                coeffs_except_linear_term: Vec::<F>::new()
            }],
        })
        .is_err());
    assert!(standard
        .validate_proof(&SumcheckProof {
            round_polys: vec![crate::CompressedUniPoly {
                coeffs_except_linear_term: vec![F::one(), F::one()]
            }],
        })
        .is_err());

    let round_specific = CheckedStandardBatch::new(
        vec![
            SumcheckMemberShape::new(2, 1),
            SumcheckMemberShape::new(1, 3),
        ],
        vec![
            SumcheckGroupSpec::new(vec![0]),
            SumcheckGroupSpec::new(vec![1]),
        ],
    )
    .unwrap();
    let malformed = SumcheckProof {
        round_polys: vec![
            crate::CompressedUniPoly {
                coeffs_except_linear_term: vec![F::one(), F::one()],
            },
            crate::CompressedUniPoly {
                coeffs_except_linear_term: vec![F::one()],
            },
        ],
    };
    assert!(matches!(
        round_specific.validate_proof(&malformed),
        Err(AkitaError::InvalidInput(_))
    ));

    let eq = CheckedEqFactoredBatch::new(
        vec![SumcheckMemberShape::new(1, 0)],
        vec![SumcheckGroupSpec::new(vec![0])],
        vec![f(9)],
    )
    .unwrap();
    assert!(matches!(
        eq.validate_proof(&EqFactoredSumcheckProof {
            round_polys: Vec::new(),
        }),
        Err(AkitaError::InvalidSize {
            expected: 1,
            actual: 0
        })
    ));
    assert!(eq
        .validate_proof(&EqFactoredSumcheckProof {
            round_polys: vec![EqFactoredUniPoly {
                coeffs_except_linear_term: Vec::new(),
            }],
        })
        .is_err());
}

#[test]
fn execution_shape_and_recurrence_errors_are_typed() {
    let plan = CheckedStandardBatch::new(
        vec![SumcheckMemberShape::new(1, 1)],
        vec![SumcheckGroupSpec::new(vec![0])],
    )
    .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut transcript =
        RecordingTranscript::with_challenges(Arc::clone(&events), vec![f(2), f(3)]);
    let input_error = prove_standard_executor_batch::<F, _, _, _>(
        &plan,
        &[],
        &mut [],
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect_err("wrong input and executor shapes are rejected");
    assert_eq!(
        input_error,
        AkitaError::InvalidSize {
            expected: 1,
            actual: 0,
        }
    );
    assert!(events.lock().expect("event log mutex").is_empty());

    let mut executors: Vec<Box<dyn SumcheckRoundExecutor<F>>> =
        vec![Box::new(DelayedExecutor::standard_fixed(
            0,
            Arc::clone(&events),
            vec![UniPoly::from_coeffs(vec![F::one()])],
            vec![F::one()],
        ))];
    let recurrence_error = prove_standard_executor_batch::<F, _, _, _>(
        &plan,
        &[f(8)],
        &mut executors,
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect_err("a malformed round recurrence is rejected");
    assert!(matches!(recurrence_error, AkitaError::InvalidInput(_)));

    let degree_zero_plan = CheckedStandardBatch::new(
        vec![SumcheckMemberShape::new(1, 0)],
        vec![SumcheckGroupSpec::new(vec![0])],
    )
    .unwrap();
    let mut degree_executors: Vec<Box<dyn SumcheckRoundExecutor<F>>> =
        vec![Box::new(DelayedExecutor::standard_fixed(
            0,
            Arc::clone(&events),
            vec![UniPoly::from_coeffs(vec![F::one(), f(14)])],
            vec![F::one()],
        ))];
    let degree_error = prove_standard_executor_batch::<F, _, _, _>(
        &degree_zero_plan,
        &[f(8)],
        &mut degree_executors,
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect_err("the executor's uncompressed degree is checked");
    assert!(matches!(degree_error, AkitaError::InvalidInput(_)));
}

#[test]
fn terminal_claim_count_and_value_are_checked() {
    let plan = CheckedStandardBatch::new(
        vec![SumcheckMemberShape::new(0, 0)],
        vec![SumcheckGroupSpec::new(vec![0])],
    )
    .unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut transcript =
        RecordingTranscript::with_challenges(Arc::clone(&events), vec![f(2), f(3)]);
    let mut missing: Vec<Box<dyn SumcheckRoundExecutor<F>>> = vec![Box::new(
        DelayedExecutor::standard(0, Arc::clone(&events), Vec::new()),
    )];
    let error = prove_standard_executor_batch::<F, _, _, _>(
        &plan,
        &[f(11)],
        &mut missing,
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect_err("missing terminal claims are rejected");
    assert_eq!(
        error,
        AkitaError::InvalidSize {
            expected: 1,
            actual: 0,
        }
    );

    let mut wrong: Vec<Box<dyn SumcheckRoundExecutor<F>>> = vec![Box::new(
        DelayedExecutor::standard(0, Arc::clone(&events), vec![f(12)]),
    )];
    let error = prove_standard_executor_batch::<F, _, _, _>(
        &plan,
        &[f(11)],
        &mut wrong,
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect_err("wrong terminal claims are rejected");
    assert!(matches!(error, AkitaError::InvalidInput(_)));
}
