#![allow(missing_docs)]
//! Akita running inside a caller's Spongefish transcript.

mod common;

use akita_cpu_backend::{AkitaProverSetup, CommitOutput, CpuBackend};
use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_transcript::{
    new_prover_channel, new_verifier_channel, ProverChannel, ProverTranscript, VerifierChannel,
};
use akita_types::{AkitaVerifierSetup, PendingEof};
use akita_verifier::SetupPolicy;
use common::*;
use jolt_field::One;
use jolt_transcript::BytesMsg;

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
        Self::with_num_vars(NUM_VARS)
    }

    fn with_num_vars(num_vars: usize) -> Self {
        let scheme =
            load_workspace_scheme::<Cfg>(akita_params::ChunkedWitnessCfg::default_non_chunked())
                .expect("workspace schedule catalog");
        let layout = scheme
            .schedules()
            .resolve_key(&akita_params::ScheduleLookupKey::single(
                akita_params::PolynomialGroupLayout::singleton(num_vars),
            ))
            .expect("layout")
            .schedule()
            .root
            .params
            .final_group();
        let poly = make_onehot_poly::<Cfg>(num_vars, 0x7171);
        let point = random_point(num_vars, 0x8181);
        let opening = opening_from_poly_for_layout(&poly, &point, &layout, BasisMode::Lagrange);
        let setup = scheme.setup_prover(num_vars, 1).expect("setup");
        let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
        let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
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
        state.public_message(&BytesMsg(CALLER_STATEMENT));
        send_caller_message(&mut state, CALLER_BEFORE);
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
        send_caller_message(&mut state, CALLER_AFTER);
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
    ) -> Result<(), AkitaError> {
        self.scheme
            .verifier(self.verifier_setup.clone())?
            .batched_verify(
                proof,
                session,
                verify_input::<Cfg>(point, &[opening], commitment, self.scheme.schedules()),
                BasisMode::Lagrange,
                SetupPolicy::Check(None),
            )
    }

    /// Run the caller's verifier up to Akita's end; the caller then finishes.
    fn verify_embedded_prefix(
        &self,
        state: &mut VerifierChannel<'_>,
        point: &[F],
        opening: F,
        commitment: &CommittedGroup<F>,
    ) -> Result<PendingEof, AkitaError> {
        state.public_message(&BytesMsg(CALLER_STATEMENT));
        let before = receive_caller_message(state)?;
        assert_eq!(before, CALLER_BEFORE);
        self.scheme
            .verifier(self.verifier_setup.clone())?
            .batched_verify_embedded(
                state,
                verify_input::<Cfg>(point, &[opening], commitment, self.scheme.schedules()),
                BasisMode::Lagrange,
                SetupPolicy::Check(None),
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

/// Caller messages are length-prefixed `BytesMsg`s, as a real caller frames
/// variable-length data; Akita's byte groups bind no length.
fn send_caller_message(state: &mut ProverChannel, message: &[u8]) {
    ProverTranscript::send(state, &BytesMsg(message));
}

fn receive_caller_message(state: &mut VerifierChannel<'_>) -> Result<Vec<u8>, AkitaError> {
    state
        .prover_message::<BytesMsg>()
        .map(|message| message.0)
        .map_err(|_| AkitaError::InvalidProof)
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
        let pending = fixture
            .scheme
            .verifier(fixture.verifier_setup.clone())
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
                    SetupPolicy::Check(None),
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
        let after = receive_caller_message(&mut state).unwrap();
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
        // The caller message before Akita is an 8-byte length and its body.
        let akita_start = 8 + CALLER_BEFORE.len();
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
        assert!(receive_caller_message(&mut state).is_err());
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
        receive_caller_message(&mut state).unwrap();
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
                    receive_caller_message(&mut state)?;
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
    use akita_transcript::{ProtocolMessageKind, ProtocolSiteId, TranscriptEvent};
    init_rayon_pool();
    run_on_large_stack(|| {
        let fixture = Fixture::new();
        akita_transcript::clear_thread_events();
        let proof = fixture.prove_standalone();
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
        // Each fold level's next-witness messages (a terminal `t` prefix is
        // followed by its last-block digest), with the atom counts the
        // verifier's level grammar expects, sent before that level's
        // ring-switch challenge.
        let next_witness = |records: &[akita_transcript::ProtocolContextRecord]| {
            records
                .iter()
                .enumerate()
                .filter(|(_, record)| {
                    site(record).family == akita_transcript::SITE_FAMILY_NEXT_WITNESS
                })
                .map(|(index, record)| (index, site(record).level, record.atom_count))
                .collect::<Vec<_>>()
        };
        let sent = next_witness(&records);
        assert!(!sent.is_empty(), "the schedule folds at least once");
        akita_transcript::clear_thread_events();
        fixture
            .verify_standalone(
                &proof,
                SESSION,
                &fixture.point,
                fixture.opening,
                &fixture.commitment,
            )
            .expect("honest proof verifies");
        let verifier_records = akita_transcript::thread_events()
            .into_iter()
            .map(|TranscriptEvent::Context(record)| record)
            .collect::<Vec<_>>();
        let expected = next_witness(&verifier_records);
        assert_eq!(
            sent.iter()
                .map(|&(_, level, atoms)| (level, atoms))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|&(_, level, atoms)| (level, atoms))
                .collect::<Vec<_>>(),
            "the prover sends each level's next witness once, at the expected size"
        );
        let challenge = ProtocolMessageKind::Challenge as u32;
        for &(index, level, _) in &sent {
            let ring_switch = akita_params::GrindingSite::RingSwitchAlpha { level }.site_id(0);
            let ring_switch_challenge = records
                .iter()
                .position(|record| {
                    let site = site(record);
                    record.kind == challenge
                        && site.family == ring_switch.family
                        && site.level == level
                })
                .expect("every folded level draws a ring-switch challenge");
            assert!(
                index < ring_switch_challenge,
                "level {level}'s next witness is sent after its ring-switch challenge"
            );
        }
    });
}

/// Emit the same fixture artifacts from the baseline and candidate builds.
/// This opt-in test never chooses its expected bytes from the implementation.
#[test]
fn verifier_feature_artifacts() {
    let Ok(directory) = std::env::var("AKITA_FIXTURE_OUTPUT") else {
        return;
    };
    init_rayon_pool();
    run_on_large_stack(move || {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        for num_vars in [12, 20] {
            let fixture = Fixture::with_num_vars(num_vars);
            let proof = fixture.prove_standalone();
            assert_eq!(proof, fixture.prove_standalone(), "repeat proof identity");
            fixture.verify_standalone(
                &proof, SESSION, &fixture.point, fixture.opening, &fixture.commitment,
            ).unwrap();
            std::fs::write(directory.join(format!("{num_vars}.proof")), &proof).unwrap();
            #[cfg(feature = "logging-transcript")]
            akita_transcript::clear_thread_events();
            let (embedded, challenge) = fixture.prove_embedded();
            #[cfg(feature = "logging-transcript")]
            std::fs::write(directory.join(format!("{num_vars}.transcript")),
                format!("{:?}\n", akita_transcript::thread_events())).unwrap();
            assert_eq!((embedded.clone(), challenge), fixture.prove_embedded());
            let mut state = new_verifier_channel(SESSION, &embedded);
            let pending = fixture.verify_embedded_prefix(
                &mut state, &fixture.point, fixture.opening, &fixture.commitment,
            ).unwrap();
            assert_eq!(receive_caller_message(&mut state).unwrap(), CALLER_AFTER);
            assert_eq!(state.verifier_message::<[u8; 32]>(), challenge);
            pending.check_eof(state).unwrap();
            std::fs::write(directory.join(format!("{num_vars}.embedded")), &embedded).unwrap();
            std::fs::write(directory.join(format!("{num_vars}.challenge")), challenge).unwrap();
            let key = akita_params::ScheduleLookupKey::single(
                akita_params::PolynomialGroupLayout::singleton(num_vars),
            );
            let row = fixture.scheme.schedules().resolve_key(&key).unwrap();
            let catalog = fixture.scheme.schedules().to_verifier_view(&[row.selection()]).unwrap();
            std::fs::write(directory.join(format!("{num_vars}.catalog")), catalog).unwrap();
            let terminal = &row.schedule().terminal;
            let d = terminal.d_a();
            let rows = terminal.inner.matrix.output_rank();
            let field_bytes = akita_params::field_bytes(
                <F as jolt_field::CanonicalEncoding>::MODULUS_BITS,
            );
            let certificate_bytes = rows * (2 * d - 1) * field_bytes;
            std::fs::write(directory.join(format!("{num_vars}.geometry")),
                format!("{d} {rows} {field_bytes} {certificate_bytes}\n")).unwrap();
            let mut mutations = vec![proof[..proof.len() - 1].to_vec()];
            let mut trailing = proof.clone();
            trailing.push(0);
            mutations.push(trailing);
            if std::env::var("AKITA_EXPECT_CERTIFICATE").as_deref() == Ok("1") {
                let start = proof.len().checked_sub(certificate_bytes).unwrap();
                // Replace each canonical first coefficient by a different
                // canonical scalar, avoiding a decoder-only rejection.
                for offset in [start, start + rows * d * field_bytes] {
                    let mut changed = proof.clone();
                    let was_zero = changed[offset..offset + field_bytes].iter().all(|&b| b == 0);
                    changed[offset..offset + field_bytes].fill(0);
                    changed[offset] = u8::from(was_zero);
                    mutations.push(changed);
                }
            }
            let mutation_count = mutations.len();
            if let Ok(other) = std::env::var("AKITA_CROSS_PROOF") {
                let incompatible = std::fs::read(
                    std::path::Path::new(&other).join(format!("{num_vars}.proof")),
                ).unwrap();
                let rejection = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    fixture.verify_standalone(
                        &incompatible, SESSION, &fixture.point, fixture.opening, &fixture.commitment,
                    )
                }));
                assert!(rejection.is_ok());
                assert!(rejection.unwrap().is_err());
            }
            for changed in mutations {
                let rejection = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    fixture.verify_standalone(
                        &changed, SESSION, &fixture.point, fixture.opening, &fixture.commitment,
                    )
                }));
                assert!(rejection.is_ok(), "malformed proof panicked");
                assert!(rejection.unwrap().is_err(), "malformed proof accepted");
            }
            std::fs::write(directory.join(format!("{num_vars}.checks")),
                format!("accept repeat embedded transcript PASS; rejected mutations={mutation_count}\n")).unwrap();
        }
    });
}
