#![allow(missing_docs)]
//! Akita running inside a caller's Spongefish transcript.

mod common;

use akita_cpu_backend::{AkitaProverSetup, CommitOutput, CpuBackend};
use akita_pcs::AkitaCommitmentScheme;
use akita_transcript::{
    new_prover_channel, new_verifier_channel, public_bytes, receive_byte_group, send_byte_group,
    ProtocolSiteId, VerifierChannel,
};
use akita_types::{AkitaVerifierSetup, PendingEof};
use common::*;
use jolt_field::One;

const NUM_VARS: usize = 14;
const SESSION: &[u8] = b"embedded/onehot";
const CALLER_STATEMENT: &[u8] = b"caller statement";
const CALLER_BEFORE: &[u8] = b"caller message before Akita";
const CALLER_AFTER: &[u8] = b"caller message after Akita";

type Cfg = OneHotCfg;
/// Session, opening point, claimed value and commitment of one proof.
type RootStatement<'a> = (&'a [u8], &'a [F], F, &'a CommittedGroup<F>);

struct Fixture {
    scheme: AkitaCommitmentScheme<Cfg>,
    setup: AkitaProverSetup<F>,
    stack: CpuBackend<F, F>,
    verifier_setup: AkitaVerifierSetup<F>,
    commitment: CommittedGroup<F>,
    hint: CommitmentHandle<F, F>,
    point: Vec<F>,
    opening: F,
}

impl Fixture {
    fn new() -> Self {
        let scheme =
            load_workspace_scheme::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked())
                .expect("workspace schedule catalog");
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
        let poly = make_onehot_poly::<Cfg>(NUM_VARS, 0x7171);
        let point = random_point(NUM_VARS, 0x8181);
        let opening = opening_from_poly_for_layout(&poly, &point, &layout, BasisMode::Lagrange);
        let setup = scheme.setup_prover(NUM_VARS, 1).expect("setup");
        let stack = CpuBackend::new(setup.expanded().clone()).expect("backend");
        let verifier_setup = unsafe { scheme.setup_verifier(&setup) }.expect("verifier setup");
        let CommitOutput {
            committed_group: commitment,
            private_handle: hint,
        } = stack
            .commit(
                scheme.schedules(),
                &stack.import_source(vec![poly]).expect("source"),
                akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("commit");
        Self {
            scheme,
            setup,
            stack,
            verifier_setup,
            commitment,
            hint,
            point,
            opening,
        }
    }

    fn prove_standalone(&self) -> Vec<u8> {
        self.scheme
            .batched_prove(
                &self.setup,
                prove_input::<Cfg>(
                    &self.point,
                    &[self.opening],
                    &self.commitment,
                    self.hint.clone(),
                    self.scheme.schedules(),
                ),
                &self.stack,
                SESSION,
                BasisMode::Lagrange,
            )
            .expect("standalone prove")
    }

    /// Prove inside a caller transcript: a caller public value and message,
    /// then Akita, then a second caller message. Returns the argument string
    /// and the caller's challenge after its last message.
    fn prove_embedded(&self) -> (Vec<u8>, [u8; 32]) {
        let mut state = new_prover_channel(SESSION);
        public_bytes(&mut state, caller_site(0), CALLER_STATEMENT).unwrap();
        send_byte_group(&mut state, caller_site(1), CALLER_BEFORE).unwrap();
        self.scheme
            .batched_prove_embedded(
                &self.setup,
                prove_input::<Cfg>(
                    &self.point,
                    &[self.opening],
                    &self.commitment,
                    self.hint.clone(),
                    self.scheme.schedules(),
                ),
                &self.stack,
                &mut state,
                BasisMode::Lagrange,
            )
            .expect("embedded prove");
        send_byte_group(&mut state, caller_site(2), CALLER_AFTER).unwrap();
        let challenge = state.verifier_message::<[u8; 32]>();
        (state.narg_string().to_vec(), challenge)
    }

    fn verify_standalone(
        &self,
        proof: &[u8],
        session: &[u8],
        point: &[F],
        opening: F,
        commitment: &CommittedGroup<F>,
    ) -> Result<(), akita_error::AkitaError> {
        unsafe { self.scheme.verifier(self.verifier_setup.clone()) }?.batched_verify(
            proof,
            session,
            verify_input::<Cfg>(point, &[opening], commitment, self.scheme.schedules()),
            BasisMode::Lagrange,
        )
    }

    /// Run the caller's verifier up to Akita's end; the caller then finishes.
    fn verify_embedded_prefix(
        &self,
        state: &mut VerifierChannel<'_>,
        point: &[F],
        opening: F,
        commitment: &CommittedGroup<F>,
    ) -> Result<PendingEof, akita_error::AkitaError> {
        public_bytes(state, caller_site(0), CALLER_STATEMENT)?;
        let before = receive_byte_group(state, caller_site(1), CALLER_BEFORE.len())?;
        assert_eq!(before, CALLER_BEFORE);
        unsafe { self.scheme.verifier(self.verifier_setup.clone()) }?.batched_verify_embedded(
            state,
            verify_input::<Cfg>(point, &[opening], commitment, self.scheme.schedules()),
            BasisMode::Lagrange,
        )
    }

    /// A different valid commitment with the same profile.
    fn other_commitment(&self) -> CommittedGroup<F> {
        let mut rows = self.commitment.rows().coeffs().to_vec();
        rows[0] += F::one();
        CommittedGroup::new(
            *self.commitment.profile(),
            akita_types::Commitment::new(akita_types::RingVec::from_coeffs(rows)),
        )
        .expect("changed commitment is well formed")
    }
}

fn caller_site(stage: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family: u32::MAX,
        stage,
        ..ProtocolSiteId::default()
    }
}

#[test]
fn standalone_is_embedded_on_a_fresh_session() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = Fixture::new();
        let standalone = fixture.prove_standalone();
        let mut state = new_prover_channel(SESSION);
        fixture
            .scheme
            .batched_prove_embedded(
                &fixture.setup,
                prove_input::<Cfg>(
                    &fixture.point,
                    &[fixture.opening],
                    &fixture.commitment,
                    fixture.hint.clone(),
                    fixture.scheme.schedules(),
                ),
                &fixture.stack,
                &mut state,
                BasisMode::Lagrange,
            )
            .expect("embedded prove");
        assert_eq!(state.narg_string(), standalone.as_slice());

        let mut verifier = new_verifier_channel(SESSION, &standalone);
        let pending = unsafe { fixture.scheme.verifier(fixture.verifier_setup.clone()) }
            .and_then(|verifier_scheme| {
                verifier_scheme.batched_verify_embedded(
                    &mut verifier,
                    verify_input::<Cfg>(
                        &fixture.point,
                        &[fixture.opening],
                        &fixture.commitment,
                        fixture.scheme.schedules(),
                    ),
                    BasisMode::Lagrange,
                )
            })
            .expect("embedded verify");
        pending.check_eof(verifier).expect("eof");
    });
}

/// Caller messages before and after Akita share Akita's sponge, and the
/// caller's final challenge agrees on both sides.
#[test]
fn embedded_proof_round_trips_between_caller_messages() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = Fixture::new();
        let (proof, prover_challenge) = fixture.prove_embedded();
        let mut state = new_verifier_channel(SESSION, &proof);
        let pending = fixture
            .verify_embedded_prefix(
                &mut state,
                &fixture.point,
                fixture.opening,
                &fixture.commitment,
            )
            .expect("embedded verify");
        let after = receive_byte_group(&mut state, caller_site(2), CALLER_AFTER.len()).unwrap();
        assert_eq!(after, CALLER_AFTER);
        assert_eq!(state.verifier_message::<[u8; 32]>(), prover_challenge);
        pending.check_eof(state).expect("eof");
    });
}

#[test]
fn embedded_rejection_poisons_the_caller_state() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = Fixture::new();
        let (mut proof, _) = fixture.prove_embedded();
        // Flip one byte inside Akita's part of the argument string.
        let akita_start = CALLER_BEFORE.len();
        proof[akita_start + 40] ^= 1;
        let mut state = new_verifier_channel(SESSION, &proof);
        assert!(fixture
            .verify_embedded_prefix(
                &mut state,
                &fixture.point,
                fixture.opening,
                &fixture.commitment,
            )
            .is_err());
        assert!(receive_byte_group(&mut state, caller_site(2), CALLER_AFTER.len()).is_err());
        assert!(state.check_eof().is_err());
    });
}

#[test]
fn standalone_rejects_trailing_bytes() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = Fixture::new();
        let mut proof = fixture.prove_standalone();
        fixture
            .verify_standalone(
                &proof,
                SESSION,
                &fixture.point,
                fixture.opening,
                &fixture.commitment,
            )
            .expect("honest proof");
        proof.push(0);
        assert!(fixture
            .verify_standalone(
                &proof,
                SESSION,
                &fixture.point,
                fixture.opening,
                &fixture.commitment,
            )
            .is_err());

        let (mut proof, _) = fixture.prove_embedded();
        proof.push(0);
        let mut state = new_verifier_channel(SESSION, &proof);
        let pending = fixture
            .verify_embedded_prefix(
                &mut state,
                &fixture.point,
                fixture.opening,
                &fixture.commitment,
            )
            .expect("Akita's part is honest");
        receive_byte_group(&mut state, caller_site(2), CALLER_AFTER.len()).unwrap();
        assert!(pending.check_eof(state).is_err());
    });
}

/// Every root statement item is bound: changing any one of them rejects the
/// honest proof, in both modes.
#[test]
fn root_statement_binds_every_item() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = Fixture::new();
        let proof = fixture.prove_standalone();
        let (embedded, _) = fixture.prove_embedded();
        let other = fixture.other_commitment();
        let mut point = fixture.point.clone();
        point[0] += F::one();
        let opening = fixture.opening + F::one();
        let cases: [RootStatement<'_>; 4] = [
            (
                b"embedded/other-session",
                &fixture.point,
                fixture.opening,
                &fixture.commitment,
            ),
            (SESSION, &point, fixture.opening, &fixture.commitment),
            (SESSION, &fixture.point, opening, &fixture.commitment),
            (SESSION, &fixture.point, fixture.opening, &other),
        ];
        for (session, point, opening, commitment) in cases {
            assert!(fixture
                .verify_standalone(&proof, session, point, opening, commitment)
                .is_err());
            let mut state = new_verifier_channel(session, &embedded);
            let accepted = fixture
                .verify_embedded_prefix(&mut state, point, opening, commitment)
                .and_then(|pending| {
                    receive_byte_group(&mut state, caller_site(2), CALLER_AFTER.len())?;
                    pending.check_eof(state)
                });
            assert!(accepted.is_err());
        }
    });
}

/// The root statement is absorbed before the first challenge: the instance
/// binding, the 32-byte digest per group, the points and the evaluations.
/// No values the prover sent are absorbed a second time.
#[cfg(feature = "logging-transcript")]
#[test]
fn sent_witness_values_are_absorbed_once() {
    use akita_transcript::{ProtocolMessageKind, TranscriptEvent};
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = Fixture::new();
        akita_transcript::clear_thread_events();
        fixture.prove_standalone();
        let records = akita_transcript::thread_events()
            .into_iter()
            .map(|TranscriptEvent::Context(record)| record)
            .collect::<Vec<_>>();
        let site = |record: &akita_transcript::ProtocolContextRecord| {
            ProtocolSiteId::from_bytes(record.site_id)
        };
        let public = ProtocolMessageKind::PublicValue as u32;
        let first_challenge = records
            .iter()
            .position(|record| record.kind == ProtocolMessageKind::Challenge as u32)
            .expect("the proof squeezes challenges");
        let prefix = &records[..first_challenge];
        assert_eq!(
            site(&prefix[0]).family,
            akita_transcript::SITE_FAMILY_INSTANCE
        );
        let root_commitments = prefix
            .iter()
            .filter(|record| {
                let site = site(record);
                site.family == akita_transcript::SITE_FAMILY_ROOT_STATEMENT && site.stage == 1
            })
            .collect::<Vec<_>>();
        assert_eq!(root_commitments.len(), 1);
        assert_eq!(root_commitments[0].kind, public);
        assert_eq!(
            root_commitments[0].encoded_bytes,
            akita_transcript::COMMITMENT_DIGEST_LEN as u64
        );
        for record in &records {
            let site = site(record);
            let reabsorbed_payload =
                site.family == akita_transcript::SITE_FAMILY_FOLD_BINDING && site.stage == 3;
            let reabsorbed_t =
                site.family == akita_transcript::SITE_FAMILY_TERMINAL && site.stage == 2;
            assert!(
                !(record.kind == public && (reabsorbed_payload || reabsorbed_t)),
                "a sent witness value is absorbed again at {site:?}"
            );
            if site.family == akita_transcript::SITE_FAMILY_NEXT_WITNESS {
                assert_ne!(record.kind, public, "next witness must be a prover message");
            }
        }
        assert!(records
            .iter()
            .any(|record| site(record).family == akita_transcript::SITE_FAMILY_NEXT_WITNESS));
    });
}
