//! Zero-knowledge schedules from the planner, end to end (Whiteout W1 and W2).
//!
//! The planner's test-only zero-knowledge search masks levels `0..=1`: their
//! B slices and D images carry hiding randomness, their fold digits are sized
//! for the one-shot target, and their fold responses are one-shot. A
//! test-only catalog admits the row after
//! `akita_params::zk::ZkOpeningSchedule::admit`; a trusted catalog rejects it.

#![allow(missing_docs)]

mod common;
#[path = "dynamic_backends/private_cpu.rs"]
mod private_cpu;

use akita_config::{
    derive_transcript_grinding_plan, policy_of, proof_optimized::fp128, CommitmentConfig,
    TrustedScheduleCatalog, ValidatedScheduleCatalog,
};
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::zk::{ZkOpeningSchedule, ZK_CUT_LEVEL};
use akita_params::{
    lagrange_weights, suffix_opening_layout, BasisMode, ChunkedWitnessCfg,
    CommittedGroupBatchProfile, FoldResponseSearch, FoldSchedule, GrindingQueryKind, GrindingSite,
    GroupCommitPhaseParams, InnerCommitSecurityRoute, PolynomialGroupLayout, RelationQuotientPlan,
    RelationWitnessGeometry, ScheduleLookupKey, WitnessLayout, FOLD_RESPONSE_NONCE_BITS,
};
use akita_pcs::{AkitaCommitmentScheme, BackendRegistry, FixedFoldRoute};
use akita_prover::SelectedProverOpeningData;
use akita_types::{
    AkitaSponge, FoldDrawRecord, GrindingReplay, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims, ProverGrinding, VerifierGrinding, PROOF_STREAM_PROTOCOL,
};
use jolt_algebra::{One, Ring, Zero};
use jolt_transcript::{ProverTranscript, VerifierTranscript};
use private_cpu::PrivateCpu;
use rand::rngs::StdRng;
use rand::SeedableRng;

type Cfg = fp128::Dense;
type F = <Cfg as CommitmentConfig>::Field;
type E = F;

const NV: usize = 14;
const SESSION: &[u8] = b"zero-knowledge-e2e";

fn zk() -> ZkOpeningSchedule {
    ZkOpeningSchedule::new(ZK_CUT_LEVEL).unwrap()
}

fn key() -> ScheduleLookupKey {
    ScheduleLookupKey::single(PolynomialGroupLayout::singleton(NV))
}

fn policy() -> akita_schedules::PlannerPolicy {
    policy_of::<Cfg>(ChunkedWitnessCfg::default_non_chunked())
}

/// The planner's zero-knowledge row for one dense `NV`-variable polynomial.
fn zk_row() -> (CommittedGroupBatchProfile, FoldSchedule) {
    let key = key();
    let planned = akita_planner::find_schedule_for_test_zk(
        &key,
        Cfg::committed_source_contract().unwrap(),
        &policy(),
        Cfg::ring_challenge_config,
        zk(),
    )
    .unwrap();
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &planned.schedule.root.params,
        )
        .unwrap(),
        precommitteds: Vec::new(),
    };
    (profiles, planned.schedule)
}

fn admit(schedule: &FoldSchedule) -> Result<ZkOpeningSchedule, AkitaError> {
    ZkOpeningSchedule::admit(schedule, Cfg::committed_source_contract().unwrap())
}

fn zk_catalog() -> TrustedScheduleCatalog<Cfg> {
    let validated = ValidatedScheduleCatalog::try_new_with_test_zk(
        Cfg::schedule_family_name(),
        [zk_row()],
        &policy(),
        Cfg::ring_challenge_config,
        Cfg::committed_source_contract().unwrap(),
    )
    .unwrap();
    TrustedScheduleCatalog::new(validated, ChunkedWitnessCfg::default_non_chunked()).unwrap()
}

/// The parameters of every nonterminal fold, in level order.
fn levels(schedule: &FoldSchedule) -> Vec<&akita_params::CommittedGroupParams> {
    std::iter::once(&schedule.root.params)
        .chain(schedule.recursive_folds.iter().map(|fold| &fold.params))
        .collect()
}

fn levels_mut(schedule: &mut FoldSchedule) -> Vec<&mut akita_params::CommittedGroupParams> {
    std::iter::once(&mut schedule.root.params)
        .chain(
            schedule
                .recursive_folds
                .iter_mut()
                .map(|fold| &mut fold.params),
        )
        .collect()
}

fn on_large_stack(test: fn()) {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(test)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn planner_masks_every_level_up_to_the_cut() {
    on_large_stack(|| {
        let (profiles, schedule) = zk_row();
        assert_eq!(schedule.zk, Some(zk()));
        assert_eq!(admit(&schedule), Ok(zk()));
        let levels = levels(&schedule);
        assert!(levels.len() > ZK_CUT_LEVEL as usize + 1);
        for (level, params) in levels.iter().enumerate() {
            let masked = zk().masks(level);
            assert_eq!(
                params.own_group().profile.randomness_columns != 0,
                masked,
                "level {level} B randomness"
            );
            assert_eq!(
                params.open_randomness_columns != 0,
                masked,
                "level {level} D randomness"
            );
            if masked {
                assert!(matches!(
                    params.inner().matrix.security_route(),
                    InnerCommitSecurityRoute::Linf(_)
                ));
            }
        }
        assert_eq!(
            profiles.final_group.randomness_columns,
            schedule.root.params.own_group().profile.randomness_columns
        );

        // W2-a: the plan is the clear plan of the same folds except that the
        // masked fold responses are one-shot, and the descriptor binds the cut.
        let layout = key().opening_layout().unwrap();
        let mut clear = schedule.clone();
        clear.zk = None;
        let clear_plan = derive_transcript_grinding_plan::<Cfg>(&clear, &layout).unwrap();
        let zk_plan = derive_transcript_grinding_plan::<Cfg>(&schedule, &layout).unwrap();
        assert_eq!(clear_plan.runs().len(), zk_plan.runs().len());
        for (clear_run, zk_run) in clear_plan.runs().iter().zip(zk_plan.runs()) {
            assert_eq!(clear_run.site(), zk_run.site());
            match zk_run.site() {
                GrindingSite::FoldResponse { level } => {
                    assert_eq!(clear_run.nonce_bits(), FOLD_RESPONSE_NONCE_BITS);
                    let expected = if zk().masks(level as usize) {
                        FoldResponseSearch::OneShot
                    } else {
                        FoldResponseSearch::Grind
                    };
                    assert_eq!(zk_run.fold_response_search(), Some(expected));
                }
                _ => assert_eq!(clear_run, zk_run),
            }
        }
        assert_ne!(clear_plan.digest().unwrap(), zk_plan.digest().unwrap());
        assert_ne!(
            clear.canonical_descriptor_bytes(),
            schedule.canonical_descriptor_bytes()
        );
    });
}

#[test]
fn trusted_catalogs_reject_zero_knowledge_rows() {
    on_large_stack(|| {
        let error = ValidatedScheduleCatalog::try_new(
            Cfg::schedule_family_name(),
            [zk_row()],
            &policy(),
            Cfg::ring_challenge_config,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not selectable"), "{error}");
    });
}

#[test]
fn admission_recomputes_the_masked_levels() {
    on_large_stack(|| {
        let (_, schedule) = zk_row();
        assert!(ZkOpeningSchedule::new(ZK_CUT_LEVEL + 1).is_err());
        let mut clear = schedule.clone();
        clear.zk = None;
        assert!(admit(&clear).is_err());

        let rejects = |mutate: &dyn Fn(&mut FoldSchedule), what: &str| {
            let mut mutated = schedule.clone();
            mutate(&mut mutated);
            let error = admit(&mutated).unwrap_err();
            assert!(error.to_string().contains(what), "{what}: {error}");
        };
        for level in 0..=ZK_CUT_LEVEL as usize {
            rejects(
                &|s| {
                    levels_mut(s)[level]
                        .own_group_mut()
                        .profile
                        .randomness_columns -= 1
                },
                "wrong B hiding randomness",
            );
            rejects(
                &|s| levels_mut(s)[level].open_randomness_columns -= 1,
                "wrong D hiding randomness",
            );
            rejects(
                &|s| levels_mut(s)[level].own_group_mut().opening.num_digits_fold -= 1,
                "fold digits",
            );
        }
        // The first unmasked level carries no randomness.
        rejects(
            &|s| levels_mut(s)[ZK_CUT_LEVEL as usize + 1].open_randomness_columns += 1,
            "wrong D hiding randomness",
        );
        // An ℓ2-route level at the cut, taken from a clear catalog row.
        let clear_catalog =
            common::load_workspace_scheme::<Cfg>(ChunkedWitnessCfg::default_non_chunked()).unwrap();
        let clear_row = clear_catalog.schedules().resolve_key(&key()).unwrap();
        let l2_level = clear_row
            .schedule()
            .recursive_folds
            .iter()
            .find(|fold| {
                matches!(
                    fold.params.inner().matrix.security_route(),
                    InnerCommitSecurityRoute::L2 { .. }
                )
            })
            .expect("the clear row has an ℓ2 level")
            .clone();
        rejects(
            &|s| s.recursive_folds[ZK_CUT_LEVEL as usize - 1] = l2_level.clone(),
            "L2 route",
        );
    });
}

/// Every randomness span lies in the live witness prefix, which the next
/// level's ring switch requires its digit-range domain to cover exactly, so
/// the range check proves every randomness digit.
#[test]
fn randomness_sits_inside_the_range_checked_witness() {
    on_large_stack(|| {
        let (_, schedule) = zk_row();
        let levels = std::iter::once((&schedule.root, key().opening_layout().unwrap())).chain(
            schedule.recursive_folds.iter().map(|fold| {
                (
                    fold,
                    suffix_opening_layout(fold.input_witness_len, None).unwrap(),
                )
            }),
        );
        for (level, (fold, batch)) in levels.take(ZK_CUT_LEVEL as usize + 1).enumerate() {
            let params = &fold.params;
            let geometry = RelationWitnessGeometry::for_level(params, &batch, 1).unwrap();
            let layout = WitnessLayout::new(
                params,
                &batch,
                &geometry,
                1,
                RelationQuotientPlan::for_field_bits(params, 128).unwrap(),
            )
            .unwrap();
            let randomness = layout.randomness();
            let outer = randomness.outer(batch.root_final_group_index().unwrap());
            let opening = randomness.opening();
            let d_b = params.role_dims().d_b();
            let geometry = params.own_group().profile.derive_slice_geometry().unwrap();
            assert_eq!(
                outer.len(),
                geometry.total_randomness_columns().unwrap() * d_b,
                "level {level}"
            );
            assert_eq!(
                opening.len(),
                params.open_randomness_columns * params.role_dims().d_d()
            );
            assert!(outer.start >= layout.tail_range().start);
            assert!(outer.end <= opening.start);
            assert!(opening.end <= layout.live_coeff_len());
            assert_eq!(
                layout.live_coeff_len(),
                fold.output_witness_len,
                "level {level}"
            );
            layout.validate_internal_ranges().unwrap();
        }
    });
}

fn polynomial() -> (Vec<F>, DensePoly<F>) {
    let evals: Vec<F> = (0..(1 << NV))
        .map(|index| F::from_u64(3 * index as u64 + 1))
        .collect();
    let poly = DensePoly::<F>::from_field_evals(NV, &evals).unwrap();
    (evals, poly)
}

fn point_and_opening(evals: &[F]) -> (Vec<F>, F) {
    let point: Vec<F> = (0..NV)
        .map(|index| F::from_u64((5 * index + 2) as u64))
        .collect();
    let opening = evals
        .iter()
        .zip(lagrange_weights(&point).unwrap())
        .fold(F::zero(), |sum, (&value, weight)| sum + value * weight);
    (point, opening)
}

/// A seeded hiding source, so every proof here is reproducible.
fn backend(setup: &akita_pcs::AkitaProverSetup<F>, seed: u64) -> CpuBackend<F, F> {
    CpuBackend::new(setup.expanded.clone())
        .unwrap()
        .with_hiding_rng(StdRng::seed_from_u64(seed))
}

#[test]
fn hiding_commitments_to_one_polynomial_differ() {
    on_large_stack(|| {
        let scheme = AkitaCommitmentScheme::<Cfg>::new(zk_catalog());
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let backend = backend(&setup, 1);
        let commit = || {
            backend
                .commit(
                    scheme.schedules(),
                    &backend.import_source(vec![polynomial().1]).unwrap(),
                    GroupContext::scheduler_without_precommitted_groups(),
                )
                .unwrap()
                .committed_group
        };
        let first = commit();
        let second = commit();
        assert_eq!(first.profile(), second.profile());
        assert_ne!(first.commitment(), second.commitment());
    });
}

/// One proof exercises both: hiding randomness at the masked levels and
/// one-shot fold responses there.
#[test]
fn zero_knowledge_proof_verifies_and_rejects_a_wrong_opening() {
    on_large_stack(|| {
        let scheme = AkitaCommitmentScheme::<Cfg>::new(zk_catalog());
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let backend = backend(&setup, 2);
        let (evals, poly) = polynomial();
        let output = backend
            .commit(
                scheme.schedules(),
                &backend.import_source(vec![poly]).unwrap(),
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        let commitment = output.committed_group;
        let (point, opening) = point_and_opening(&evals);
        let group =
            PolynomialGroupClaims::new(point.clone(), vec![opening], commitment.clone()).unwrap();
        let prover_data = SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![group]).unwrap(),
            vec![output.private_handle],
            scheme.schedules(),
        )
        .unwrap();
        let selection = prover_data.selection();
        let proof = scheme
            .batched_prove(&setup, prover_data, &backend, SESSION, BasisMode::Lagrange)
            .unwrap();

        let verifier = scheme
            .verifier(scheme.setup_verifier(&setup).unwrap())
            .unwrap();
        let verify = |claimed: F| {
            let group =
                PolynomialGroupClaims::new(point.clone(), vec![claimed], &commitment).unwrap();
            let claims = OpeningClaims::from_groups(vec![group]).unwrap();
            let statement = GroupBatchStatement::new(selection, claims).unwrap();
            verifier.verify_standalone(&proof, SESSION, statement, BasisMode::Lagrange)
        };
        verify(opening).unwrap();
        assert!(verify(opening + F::one()).is_err());
    });
}

/// W2-b: a fold response the backend rejects at a masked level aborts the
/// proof after that one probe.
#[test]
fn rejected_masked_fold_response_overflows_after_one_probe() {
    on_large_stack(|| run_overflow().unwrap());
}

fn run_overflow() -> Result<(), AkitaError> {
    let scheme = AkitaCommitmentScheme::<Cfg>::new(zk_catalog());
    let setup = scheme.setup_prover(NV, 1)?;
    let private = PrivateCpu::new(backend(&setup, 3));
    let (evals, poly) = polynomial();
    let (committed_group, handle) = private.commit(scheme.schedules(), vec![poly])?;
    let (point, opening) = point_and_opening(&evals);
    let input = || {
        SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![opening],
                committed_group.clone(),
            )?])?,
            vec![handle.clone()],
            scheme.schedules(),
        )
    };
    let selection = input()?.selection();
    let resolved = scheme.schedules().resolve_selection(selection)?;
    let levels = resolved.schedule().num_fold_levels();
    let prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
        resolved.schedule(),
        input()?.opening_layout(),
    )?;
    let prefixes = private.prefixes(&setup, &prefix_ids);
    let mut registry = BackendRegistry::<Cfg>::new()?;
    let id = registry.register(&private, &prefixes)?;
    let prove = || {
        common::prove_on_session(SESSION, |transcript| {
            akita_pcs::batched_prove(
                setup.expanded.descriptor(),
                scheme.schedules(),
                &registry,
                input()?,
                transcript,
                BasisMode::Lagrange,
                &mut FixedFoldRoute::new(vec![id; levels]),
            )
        })
    };

    let proof = prove()?;
    let verifier = scheme.verifier(scheme.setup_verifier(&setup)?)?;
    verifier.verify_standalone(
        &proof,
        SESSION,
        GroupBatchStatement::new(
            selection,
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![opening],
                &committed_group,
            )?])?,
        )?,
        BasisMode::Lagrange,
    )?;

    for level in 0..=ZK_CUT_LEVEL {
        let start = private.fold_probes.get();
        private.reject_fold_probe.set(Some(start + level as usize));
        assert_eq!(prove(), Err(AkitaError::FoldOverflow { level }));
        assert_eq!(private.fold_probes.get(), start + level as usize + 1);
    }
    Ok(())
}

/// A one-shot fold response sends no nonce and forks at counter 0, and the
/// verifier rejects a nonce in its place.
#[test]
fn one_shot_fold_response_replays_without_a_nonce() {
    on_large_stack(|| {
        let (_, schedule) = zk_row();
        let layout = key().opening_layout().unwrap();
        let plan = derive_transcript_grinding_plan::<Cfg>(&schedule, &layout).unwrap();
        let site = GrindingSite::FoldResponse { level: 0 };
        let start = plan
            .runs()
            .iter()
            .position(|run| run.site() == site)
            .expect("level-0 fold response");
        // Every run before the response is a proof-of-work query.
        let earlier = &plan.runs()[..start];
        assert!(earlier
            .iter()
            .all(|run| run.kind() == GrindingQueryKind::ProofOfWork));
        let [group, next, ..] = &plan.runs()[start + 1..] else {
            panic!("plan too short");
        };
        let coordinates = usize::try_from(group.fold_coordinate_count().unwrap()).unwrap();

        let mut transcript = ProverTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION);
        let mut prover = ProverGrinding::<AkitaSponge>::new(&mut transcript, &plan);
        for run in earlier {
            prover.grinded_ext_challenge::<F, E>(run.site()).unwrap();
        }
        assert_eq!(
            prover.begin_fold_response(site).unwrap(),
            FoldResponseSearch::OneShot
        );
        assert!(prover.fold_response_fork(site, 1).is_err());
        let at_response = prover.state_mut().narg().len();
        let mut prover_fork = prover.commit_fold_response(site, 0).unwrap();
        assert_eq!(
            prover.state_mut().narg().len(),
            at_response,
            "no nonce is sent"
        );
        prover
            .record_fold_challenges(0, 0, coordinates, &FoldDrawRecord::default())
            .unwrap();
        prover.grinded_ext_challenge::<F, E>(next.site()).unwrap();
        let proof = transcript.finish();

        let verify = |proof: &[u8]| -> Result<[u8; 32], AkitaError> {
            let mut transcript =
                VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION, proof);
            let mut verifier = VerifierGrinding::<AkitaSponge>::new(&mut transcript, &plan);
            for run in earlier {
                verifier.grinded_ext_challenge::<F, E>(run.site())?;
            }
            let mut fork = verifier.read_fold_response(site)?;
            verifier.record_fold_challenges(0, 0, coordinates, &FoldDrawRecord::default())?;
            verifier.grinded_ext_challenge::<F, E>(next.site())?;
            transcript.finish()?;
            Ok(fork.squeeze::<32>())
        };
        assert_eq!(verify(&proof), Ok(prover_fork.squeeze::<32>()));
        // A clear prover's nonce for counter 0 where the response sits.
        let mut with_nonce = proof;
        with_nonce.insert(at_response, 0);
        assert!(verify(&with_nonce).is_err());

        let mut transcript = ProverTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION);
        let mut prover = ProverGrinding::<AkitaSponge>::new(&mut transcript, &plan);
        for run in earlier {
            prover.grinded_ext_challenge::<F, E>(run.site()).unwrap();
        }
        prover.begin_fold_response(site).unwrap();
        assert!(matches!(
            prover.commit_fold_response(site, 1),
            Err(AkitaError::InvalidInput(_))
        ));
    });
}
