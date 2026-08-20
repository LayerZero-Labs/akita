use super::shared::invalid;
use super::*;
use crate::{advance_eq_factored_claim, EqFactoredSumcheckProof, SumcheckProof};
use akita_field::{Fp64, HalvingField};
use akita_serialization::AkitaSerialize;
use akita_transcript::{labels, AkitaTranscript, Transcript};
use std::sync::{Arc, Mutex};

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
        let (claim, factor) = match (&self.messages, request.context) {
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
                (previous_claim, None)
            }
            (
                FakeMessages::EqFactored(_),
                CheckedRoundContext::EqFactored {
                    scaled_claim,
                    factor_at_zero,
                    factor_at_one,
                    batching_coefficients,
                    ..
                },
            ) => {
                if batching_coefficients.is_empty() {
                    return Err(invalid("test executor received no batching coefficients"));
                }
                (scaled_claim, Some((factor_at_zero, factor_at_one)))
            }
            _ => return Err(invalid("test executor received the wrong protocol format")),
        };
        self.record(Event::Start {
            group: self.group,
            master_round: request.round.master_round,
            local_round: request.round.local_round,
            previous_challenge: request.previous_challenge,
            factor,
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
}

#[test]
fn unequal_round_eq_execution_is_rejected_before_transcript_or_executor_work() {
    let plan = CheckedEqFactoredBatch::new(
        vec![
            SumcheckMemberShape::new(2, 0),
            SumcheckMemberShape::new(1, 0),
            SumcheckMemberShape::new(0, 0),
        ],
        vec![
            SumcheckGroupSpec::new(vec![0]),
            SumcheckGroupSpec::new(vec![1]),
            SumcheckGroupSpec::new(vec![2]),
        ],
        vec![f(23), f(29)],
    )
    .expect("unequal-round eq geometry is valid");
    assert_eq!(plan.member_equality_point(1).unwrap(), &[f(29)]);
    assert_eq!(plan.member_equality_point(2).unwrap(), &[]);

    let events = Arc::new(Mutex::new(Vec::new()));
    let mut transcript =
        RecordingTranscript::with_challenges(Arc::clone(&events), vec![F::one(); 8]);
    let mut executors: Vec<Box<dyn SumcheckRoundExecutor<F>>> = (0..3)
        .map(|group| {
            Box::new(DelayedExecutor::eq_factored(
                group,
                Arc::clone(&events),
                Vec::new(),
                vec![F::one()],
            )) as Box<dyn SumcheckRoundExecutor<F>>
        })
        .collect();
    let error = prove_eq_factored_executor_batch::<F, _, _, _>(
        &plan,
        &[F::one(); 3],
        &mut executors,
        &mut transcript,
        |transcript| transcript.challenge_scalar(b"test/challenge"),
    )
    .expect_err("the current proof type cannot encode unequal factors");
    assert!(matches!(error, AkitaError::UnsupportedSchedule(_)));
    assert!(events.lock().expect("event log mutex").is_empty());
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
