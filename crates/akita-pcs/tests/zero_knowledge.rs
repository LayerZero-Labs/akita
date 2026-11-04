//! Zero-knowledge schedules from the planner, end to end (Whiteout W1 and W2).
//!
//! The planner's test-only zero-knowledge search masks levels `0..=1`: their
//! B slices and D images carry hiding randomness, their fold digits are sized
//! for the one-shot target, and their fold responses are one-shot. A
//! test-only catalog admits the row after
//! `akita_params::zk::ZkOpeningSchedule::admit`; a trusted catalog rejects it.

#![allow(missing_docs)]

mod common;

use akita_config::{
    derive_transcript_grinding_plan, policy_of, proof_optimized::fp128, CommitmentConfig,
    TrustedScheduleCatalog, ValidatedScheduleCatalog, ZkCommitmentConfig,
};
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext, HidingRandomness};
use akita_error::AkitaError;
use akita_params::zk::{ZkOpeningSchedule, ZK_CUT_LEVEL};
use akita_params::{
    lagrange_weights, suffix_opening_layout, BasisMode, ChunkedWitnessCfg,
    CommittedGroupBatchProfile, FoldResponseSearch, FoldSchedule, GrindingQueryKind, GrindingSite,
    GroupCommitPhaseParams, InnerCommitSecurityRoute, PolynomialGroupLayout, RelationQuotientPlan,
    RelationWitnessGeometry, RingRelationMode, ScheduleLookupKey, WitnessLayout,
    FOLD_RESPONSE_NONCE_BITS,
};
use akita_pcs::{AkitaCommitmentScheme, BackendRegistry, FixedFoldRoute};
use akita_prover::SelectedProverOpeningData;
use akita_transcript::{new_prover_channel, new_verifier_channel};
use akita_types::{
    Commitment, FoldResponseReplay, GrindingReplay, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims, ProverGrinding, VerifierGrinding,
};
use jolt_algebra::{Ring, Zero};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// The clear preset whose zero-knowledge family the tests plan in.
type Clear = fp128::Dense;
type Cfg = ZkCommitmentConfig<Clear>;
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
    let planned = akita_planner::find_zk_schedule(
        &key,
        Cfg::committed_source_contract().unwrap(),
        &[],
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
    ZkOpeningSchedule::admit(
        schedule,
        Cfg::committed_source_contract().unwrap(),
        Cfg::EXT_DEGREE,
    )
}

fn zk_catalog() -> TrustedScheduleCatalog<Cfg> {
    TrustedScheduleCatalog::try_from_rows([zk_row()], ChunkedWitnessCfg::default_non_chunked())
        .unwrap()
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
            let residual = level == ZK_CUT_LEVEL as usize + 1;
            assert_eq!(
                params.own_group().profile.randomness_columns != 0,
                masked || residual,
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
        let clear_plan = derive_transcript_grinding_plan::<Cfg>(&clear, &layout, None).unwrap();
        let zk_plan = derive_transcript_grinding_plan::<Cfg>(&schedule, &layout, None).unwrap();
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
fn clear_catalogs_reject_zero_knowledge_rows_and_the_reverse() {
    on_large_stack(|| {
        let trusted = |row| {
            ValidatedScheduleCatalog::try_new(
                Clear::schedule_family_name(),
                [row],
                &policy(),
                Cfg::ring_challenge_config,
            )
            .unwrap_err()
            .to_string()
        };
        let (profiles, schedule) = zk_row();
        let error = trusted((profiles.clone(), schedule.clone()));
        assert!(
            error.contains("zero-knowledge cut is not selectable"),
            "{error}"
        );
        // The same folds without the cut still carry hiding randomness.
        let mut no_cut = schedule;
        no_cut.zk = None;
        let error = trusted((profiles.clone(), no_cut.clone()));
        assert!(
            error.contains("hiding randomness is not selectable"),
            "{error}"
        );
        // A zero-knowledge catalog admits no clear row.
        let error = TrustedScheduleCatalog::<Cfg>::try_from_rows(
            [(profiles, no_cut)],
            ChunkedWitnessCfg::default_non_chunked(),
        )
        .err()
        .unwrap()
        .to_string();
        assert!(
            error.contains("admits only zero-knowledge schedules"),
            "{error}"
        );
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
        // The level after the cut opens the residual commitment: exactly
        // r_a's columns, at base 16, in one slice, with no D randomness.
        let after_cut = ZK_CUT_LEVEL as usize + 1;
        for delta in [-1isize, 1] {
            rejects(
                &|s| {
                    let mut levels = levels_mut(s);
                    let group = levels[after_cut].own_group_mut();
                    group.profile.randomness_columns = group
                        .profile
                        .randomness_columns
                        .checked_add_signed(delta)
                        .unwrap();
                },
                "wrong residual randomness r_a",
            );
        }
        rejects(
            &|s| levels_mut(s)[after_cut].open_randomness_columns += 1,
            "D hiding randomness after the cut",
        );
        rejects(
            &|s| {
                levels_mut(s)[after_cut]
                    .own_group_mut()
                    .opening
                    .log_basis_open -= 1
            },
            "basis other than 16",
        );
        rejects(
            &|s| {
                levels_mut(s)[after_cut]
                    .own_group_mut()
                    .profile
                    .outer_slice_count = akita_params::CommitmentSliceCount::TWO
            },
            "more than one outer slice",
        );
        rejects(
            &|s| {
                levels_mut(s)[after_cut].payload_mode =
                    akita_params::CommitmentPayloadMode::Compressed
            },
            "residual commitment compressed",
        );
        // Every later level carries no randomness.
        if let Some(later) = levels(&schedule)
            .len()
            .checked_sub(1)
            .filter(|&last| last > after_cut)
        {
            rejects(
                &|s| {
                    levels_mut(s)[later]
                        .own_group_mut()
                        .profile
                        .randomness_columns += 1
                },
                "wrong B hiding randomness",
            );
            rejects(
                &|s| levels_mut(s)[later].open_randomness_columns += 1,
                "wrong D hiding randomness",
            );
        }
        let masked = ZK_CUT_LEVEL as usize;
        rejects(
            &|s| levels_mut(s)[masked].ring_relation_mode = RingRelationMode::ReducedEvaluation,
            "reduced evaluation",
        );
        rejects(
            &|s| levels_mut(s)[masked].own_group_mut().opening.log_basis_open += 1,
            "open basis other than 8",
        );
        rejects(
            &|s| levels_mut(s)[masked].witness_chunk.num_chunks = 2,
            "several witness chunks",
        );
        rejects(
            &|s| s.recursive_folds.truncate(ZK_CUT_LEVEL as usize),
            "reaches the terminal witness",
        );
        // A setup prefix after the cut would give the cut level a Stage 3.
        let prefix = |s: &FoldSchedule| {
            let mut group = *s.recursive_folds[after_cut].params.own_group();
            group.profile.randomness_columns = 0;
            group.setup_natural_len = Some(1);
            group
        };
        rejects(
            &|s| {
                let group = prefix(s);
                levels_mut(s)[after_cut]
                    .set_setup_prefix(Some(group))
                    .unwrap();
            },
            "Stage 3",
        );
        // Hiding randomness on a preceding group of an unmasked level.
        rejects(
            &|s| {
                let mut group = prefix(s);
                group.profile.randomness_columns = 1;
                let last = s.recursive_folds.len();
                levels_mut(s)[last].set_setup_prefix(Some(group)).unwrap();
            },
            "gives a preceding group the wrong B hiding randomness",
        );
        let error =
            ZkOpeningSchedule::admit(&schedule, Cfg::committed_source_contract().unwrap(), 2)
                .unwrap_err();
        assert!(error.to_string().contains("base field"), "{error}");
        // An ℓ2-route level at the cut, taken from a clear catalog row.
        let clear_catalog =
            common::load_workspace_scheme::<Clear>(ChunkedWitnessCfg::default_non_chunked())
                .unwrap();
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
        // A residual commitment with d_A ≠ d_B after the cut, from any clear
        // fp128 dense level with unequal A and B rings.
        let unequal = std::iter::once(&clear_row.schedule().root)
            .chain(&clear_row.schedule().recursive_folds)
            .find(|fold| fold.params.role_dims().d_a() != fold.params.role_dims().d_b())
            .expect("a clear level with d_A != d_B")
            .clone();
        rejects(
            &|s| s.recursive_folds[ZK_CUT_LEVEL as usize] = unequal.clone(),
            "d_A = d_B",
        );
    });
}

/// W5e (A41): the Σ-protocol opens the residual commitment, so admission
/// prices A_a and [B_a | B′_a] at the Σ collision radius. The W5d rows'
/// shape (rank-1 `[B_a | B′_a]` over `R_64`, its r_a resized to match) is
/// rejected naming the matrix, and so is A_a one rank below its binding
/// rank; the planned row itself sits at both binding ranks.
#[test]
fn admission_prices_the_residual_at_the_sigma_radius() {
    on_large_stack(|| {
        use akita_params::sis::sigma_binding::sigma_binding_min_rank;
        use akita_params::{InnerCommitMatrixParams, OuterCommitMatrixParams};

        let (_, schedule) = zk_row();
        let after_cut = ZK_CUT_LEVEL as usize + 1;
        let own = *levels(&schedule)[after_cut].own_group();
        let (inner, outer) = (own.profile.inner.matrix, own.profile.outer.matrix);
        let binding = |rank, width, d, modulus| {
            assert_eq!(sigma_binding_min_rank(modulus, d, width), Some(rank));
        };
        binding(
            inner.output_rank(),
            inner.input_width(),
            inner.ring_dimension(),
            inner.sis_modulus_profile(),
        );
        binding(
            outer.output_rank(),
            outer.input_width(),
            outer.ring_dimension(),
            outer.sis_modulus_profile(),
        );

        let rejects = |mutate: &dyn Fn(&mut akita_params::GroupOpenPhaseParams), what: &str| {
            let mut mutated = schedule.clone();
            mutate(levels_mut(&mut mutated)[after_cut].own_group_mut());
            let error = admit(&mutated).unwrap_err();
            assert!(
                matches!(&error, AkitaError::InvalidSetup(message) if message.contains(what)),
                "{what}: {error}"
            );
        };
        rejects(
            &|group| {
                let key = group.profile.outer.matrix.sis_table_key();
                let t_hat =
                    group.profile.outer.matrix.input_width() - group.profile.randomness_columns;
                let rank_one = |width| {
                    OuterCommitMatrixParams::try_new(
                        key.policy,
                        key.table_digest,
                        key.modulus_profile,
                        1,
                        width,
                        key.coeff_linf_bound,
                        key.ring_dimension as usize,
                    )
                    .unwrap()
                };
                group.profile.outer.matrix = rank_one(t_hat + 1);
                group.profile.randomness_columns =
                    akita_params::zk::b_randomness_columns(group).unwrap();
                group.profile.outer.matrix = rank_one(t_hat + group.profile.randomness_columns);
            },
            "[B_a | B′_a] of rank 1, below its Σ-binding rank",
        );
        rejects(
            &|group| {
                let InnerCommitSecurityRoute::Linf(key) =
                    group.profile.inner.matrix.security_route()
                else {
                    panic!("the residual's A is on the ℓ∞ route");
                };
                group.profile.inner.matrix = InnerCommitMatrixParams::try_new(
                    key.policy,
                    key.table_digest,
                    key.modulus_profile,
                    group.profile.inner.matrix.output_rank() - 1,
                    group.profile.inner.matrix.input_width(),
                    key.coeff_linf_bound,
                    key.ring_dimension as usize,
                )
                .unwrap();
            },
            "A_a of rank",
        );
        rejects(
            &|group| group.profile.inner.digits.num_digits = 2,
            "more than one inner digit plane",
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

/// A precommitted group is committed before the opening under a frozen
/// hiding profile, and every such commit draws fresh randomness.
#[test]
fn hiding_precommitted_commitments_to_one_polynomial_differ() {
    on_large_stack(|| {
        let (profiles, _) = zk_row();
        let profile = profiles.final_group;
        assert_ne!(profile.randomness_columns, 0);
        let scheme = AkitaCommitmentScheme::<Cfg>::new(zk_catalog());
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let backend = backend(&setup, 4);
        let commit = || {
            backend
                .commit(
                    scheme.schedules(),
                    &backend.import_source(vec![polynomial().1]).unwrap(),
                    GroupContext::explicit(&profile),
                )
                .unwrap()
                .committed_group
        };
        let first = commit();
        let second = commit();
        assert_eq!(*first.profile(), profile);
        assert_eq!(first.profile(), second.profile());
        assert_ne!(first.commitment(), second.commitment());
    });
}

/// A precommitted group's hiding randomness leaves the committing process
/// as bytes: another backend rebuilds the same commitment and a handle
/// carrying the same randomness from (source, bytes), and refuses other
/// data, altered randomness, a malformed encoding and a clear profile.
#[test]
fn exported_hiding_randomness_rebuilds_exactly_its_commitment() {
    on_large_stack(|| {
        use akita_serialization::{AkitaDeserialize, AkitaSerialize};

        let (profiles, _) = zk_row();
        let profile = profiles.final_group;
        let scheme = AkitaCommitmentScheme::<Cfg>::new(zk_catalog());
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let (committed, bytes) = {
            let committer = backend(&setup, 5);
            let output = committer
                .commit(
                    scheme.schedules(),
                    &committer.import_source(vec![polynomial().1]).unwrap(),
                    GroupContext::explicit(&profile),
                )
                .unwrap();
            let randomness = output.private_handle.hiding_randomness().unwrap();
            assert!(
                !format!("{randomness:?}").contains('['),
                "Debug prints no digits"
            );
            (output.committed_group, randomness.to_bytes())
        };
        // The commitment ships on the wire, like the randomness.
        let mut wire = Vec::new();
        committed
            .commitment()
            .serialize_compressed(&mut wire)
            .unwrap();
        let shipped = Commitment::<F>::deserialize_compressed(
            wire.as_slice(),
            &committed.commitment().rows().coeff_len(),
        )
        .unwrap();
        let randomness = HidingRandomness::from_bytes(&bytes).unwrap();
        // The rebuilding backend has no hiding source of its own.
        let rebuilder = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
        let recommit = |poly: DensePoly<F>, randomness: &HidingRandomness| {
            rebuilder.recommit_hiding(
                scheme.schedules(),
                &rebuilder.import_source(vec![poly]).unwrap(),
                GroupContext::explicit(&profile),
                randomness,
                &shipped,
            )
        };
        let rebuilt = recommit(polynomial().1, &randomness).unwrap();
        assert_eq!(rebuilt.committed_group, committed);
        assert_eq!(
            *rebuilt
                .private_handle
                .hiding_randomness()
                .unwrap()
                .to_bytes(),
            *bytes
        );

        let rejects = |result: Result<(), AkitaError>, what: &str| match result {
            Err(AkitaError::InvalidInput(_)) => {}
            Err(error) => panic!("{what} rejected for another reason: {error}"),
            Ok(()) => panic!("{what} accepted"),
        };
        let rebuilds = |poly: DensePoly<F>, bytes: &[u8]| {
            recommit(poly, &HidingRandomness::from_bytes(bytes)?).map(|_| ())
        };
        let mut evals = polynomial().0;
        evals[3] += F::from_u64(1);
        rejects(
            rebuilds(
                DensePoly::<F>::from_field_evals(NV, &evals).unwrap(),
                &bytes,
            ),
            "other data",
        );
        // The wire form: 4 magic bytes, a version byte, a u64 count, digits.
        const DIGITS: usize = 13;
        let mut altered = bytes.to_vec();
        altered[DIGITS] = if altered[DIGITS] == 0 { 1 } else { 0 };
        rejects(rebuilds(polynomial().1, &altered), "an altered digit");
        let mut outside = bytes.to_vec();
        outside[DIGITS] = 100;
        rejects(
            rebuilds(polynomial().1, &outside),
            "a digit outside the alphabet",
        );
        let mut shorter = bytes[..bytes.len() - 1].to_vec();
        shorter[5..DIGITS].copy_from_slice(&((bytes.len() - DIGITS - 1) as u64).to_le_bytes());
        rejects(
            rebuilds(polynomial().1, &shorter),
            "randomness of another length",
        );
        rejects(
            rebuilds(polynomial().1, &bytes[..bytes.len() - 1]),
            "a truncated encoding",
        );
        let mut version = bytes.to_vec();
        version[4] = 2;
        rejects(rebuilds(polynomial().1, &version), "another version");

        let clear_scheme =
            common::load_workspace_scheme::<Clear>(ChunkedWitnessCfg::default_non_chunked())
                .unwrap();
        let clear_setup = clear_scheme.setup_prover(NV, 1).unwrap();
        let clear_backend = CpuBackend::<F, F>::new(clear_setup.expanded.clone()).unwrap();
        let clear_source = clear_backend.import_source(vec![polynomial().1]).unwrap();
        let clear = clear_backend
            .commit(
                clear_scheme.schedules(),
                &clear_source,
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        rejects(
            clear.private_handle.hiding_randomness().map(|_| ()),
            "exporting a clear commitment",
        );
        rejects(
            clear_backend
                .recommit_hiding(
                    clear_scheme.schedules(),
                    &clear_source,
                    GroupContext::scheduler_without_precommitted_groups(),
                    &randomness,
                    clear.committed_group.commitment(),
                )
                .map(|_| ()),
            "a clear profile",
        );
    });
}

/// No unmasked proof under a zero-knowledge configuration: the clear
/// entry points of both roles (`AkitaCommitmentScheme::batched_prove`,
/// `akita_pcs::batched_prove`, `AkitaVerifier::verify_standalone`) refuse
/// before any transcript operation. The masked path is `masked_prefix.rs`
/// and `zk_end_to_end.rs`.
#[test]
fn clear_entry_points_refuse_a_zero_knowledge_configuration() {
    on_large_stack(|| {
        fn refused<T>(result: Result<T, AkitaError>) {
            match result {
                Err(AkitaError::UnsupportedSchedule(message)) => {
                    assert!(message.contains("masked prefix"), "{message}")
                }
                Err(error) => panic!("refused for another reason: {error}"),
                Ok(_) => panic!("a zero-knowledge configuration opened in the clear"),
            }
        }
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
        let prover_data = || {
            SelectedProverOpeningData::from_committed_claims::<Cfg>(
                OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                    point.clone(),
                    vec![opening],
                    commitment.clone(),
                )
                .unwrap()])
                .unwrap(),
                vec![output.private_handle.clone()],
                scheme.schedules(),
            )
            .unwrap()
        };
        let selection = prover_data().selection();
        refused(scheme.batched_prove(
            &setup,
            prover_data(),
            &backend,
            SESSION,
            BasisMode::Lagrange,
        ));

        let resolved = scheme.schedules().resolve_selection(selection).unwrap();
        let ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            prover_data().opening_layout(),
        )
        .unwrap();
        let prefixes = backend
            .import_setup_prefixes(&setup.prefix_slots, &ids)
            .unwrap();
        let mut registry = BackendRegistry::<Cfg>::new().unwrap();
        let id = registry.register(&backend, &prefixes).unwrap();
        let mut transcript = new_prover_channel(SESSION);
        refused(akita_pcs::batched_prove_embedded(
            setup.expanded.descriptor(),
            scheme.schedules(),
            &registry,
            prover_data(),
            &mut transcript,
            BasisMode::Lagrange,
            &mut FixedFoldRoute::new(vec![id; resolved.schedule().num_fold_levels()]),
        ));
        assert!(
            transcript.narg_string().is_empty(),
            "the prover wrote before refusing"
        );

        let verifier = scheme
            .verifier(scheme.setup_verifier(&setup).unwrap())
            .unwrap();
        let statement = GroupBatchStatement::new(
            selection,
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![opening],
                &commitment,
            )
            .unwrap()])
            .unwrap(),
        )
        .unwrap();
        refused(verifier.batched_verify(
            &[],
            SESSION,
            statement,
            BasisMode::Lagrange,
            akita_verifier::SetupPolicy::Check(None),
        ));
    });
}

/// S9: a zero-knowledge configuration's descriptor binds `zk = true`, so it
/// decodes only against that feature set, and a clear configuration's only
/// against `zk = false`, for the same row and setup.
#[test]
fn the_descriptor_binds_the_configuration_mode() {
    on_large_stack(|| {
        use akita_serialization::AkitaDeserialize;
        use akita_types::{AkitaInstanceDescriptor, ProtocolFeatureSet};

        let scheme = AkitaCommitmentScheme::<Cfg>::new(zk_catalog());
        let setup = scheme.setup_prover(NV, 1).unwrap();
        let row = scheme.schedules().resolve_key(&key()).unwrap();
        let layout = key().opening_layout().unwrap();
        let descriptor = |zk: bool| {
            let bytes = if zk {
                akita_config::transcript_instance_descriptor::<F, Cfg>(
                    setup.expanded.descriptor(),
                    &layout,
                    row.selection(),
                    row.schedule(),
                    BasisMode::Lagrange,
                    None,
                )
            } else {
                akita_config::transcript_instance_descriptor::<F, Clear>(
                    setup.expanded.descriptor(),
                    &layout,
                    row.selection(),
                    row.schedule(),
                    BasisMode::Lagrange,
                    None,
                )
            };
            bytes.unwrap().1
        };
        for (bound, other) in [(true, false), (false, true)] {
            let bytes = descriptor(bound);
            let decode = |zk| {
                AkitaInstanceDescriptor::deserialize_uncompressed_exact(
                    &bytes,
                    &ProtocolFeatureSet { zk },
                )
            };
            assert_eq!(decode(bound).unwrap().setup.protocol_features.zk, bound);
            assert!(
                decode(other).is_err(),
                "a zk = {bound} descriptor decodes as {other}"
            );
        }
    });
}

/// A one-shot fold response sends no nonce and previews only counter 0, and
/// the verifier rejects a nonce in its place.
#[test]
fn one_shot_fold_response_replays_without_a_nonce() {
    on_large_stack(|| {
        let (_, schedule) = zk_row();
        let layout = key().opening_layout().unwrap();
        let plan = derive_transcript_grinding_plan::<Cfg>(&schedule, &layout, None).unwrap();
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
        let [work, group, next, ..] = &plan.runs()[start + 1..] else {
            panic!("plan too short");
        };
        assert_eq!(work.site(), GrindingSite::FoldChallengeWork { level: 0 });
        let coordinates = usize::try_from(group.fold_coordinate_count().unwrap()).unwrap();

        let mut transcript = new_prover_channel(SESSION);
        let mut prover = ProverGrinding::new(&mut transcript, &plan);
        for run in earlier {
            prover.grinded_ext_challenge::<E>(run.site()).unwrap();
        }
        assert_eq!(
            prover.fold_response_search(site).unwrap(),
            FoldResponseSearch::OneShot
        );
        assert!(matches!(
            prover.preview_fold_response(site, 1),
            Err(AkitaError::InvalidInput(_))
        ));
        let at_response = prover.state_mut().narg_string().len();
        let mut preview = prover.preview_fold_response(site, 0).unwrap();
        prover
            .commit_fold_response(site, preview.take_work().unwrap())
            .unwrap();
        prover.record_fold_challenges(0, 0, coordinates).unwrap();
        let challenge = prover.grinded_ext_challenge::<E>(next.site()).unwrap();
        prover.finish().unwrap();
        let proof = transcript.narg_string().to_vec();

        let verify = |proof: &[u8]| -> Result<(u32, E), AkitaError> {
            let mut state = new_verifier_channel(SESSION, proof);
            let mut verifier = VerifierGrinding::new(&mut state, &plan);
            for run in earlier {
                verifier.grinded_ext_challenge::<E>(run.site())?;
            }
            let counter = verifier.read_fold_response(site)?;
            verifier.record_fold_challenges(0, 0, coordinates)?;
            let challenge = verifier.grinded_ext_challenge::<E>(next.site())?;
            verifier.finish()?.check_eof(state)?;
            Ok((counter, challenge))
        };
        assert_eq!(verify(&proof), Ok((0, challenge)));
        // A clear prover's nonce for counter 0 where the response sits.
        let mut with_nonce = proof;
        with_nonce.insert(at_response, 0);
        assert!(verify(&with_nonce).is_err());
    });
}

/// W2b: groups at masked levels. The production search never offloads into
/// the cut level (`specs/whiteout-akita.md` §5.5.3), so these planning tests
/// take the row that offloads level 0's setup to a level-1 setup prefix
/// (`find_zk_schedule_offloading_at_cut_for_test`), as the proving tests do
/// at 31 variables. The root here also folds one precommitted dense group,
/// the shape of Jolt's untrusted advice.
type RecCfg = ZkCommitmentConfig<akita_config::RecursiveCommitmentConfig<fp128::OneHot>>;
const MULTI_NV: usize = 37;

/// The planner's row, offloading at the cut, for one `MULTI_NV`-variable
/// one-hot polynomial beside one precommitted dense polynomial committed
/// under [`zk_row`]'s hiding root profile.
fn multi_group_zk_row() -> (CommittedGroupBatchProfile, FoldSchedule) {
    let precommitted = zk_row().0.final_group;
    let key = ScheduleLookupKey {
        root_accumulated_setup_prefix_len: 0,
        final_group: PolynomialGroupLayout::singleton(MULTI_NV),
        precommitteds: vec![precommitted],
    };
    let planned = akita_planner::find_zk_schedule_offloading_at_cut_for_test(
        &key,
        RecCfg::committed_source_contract().unwrap(),
        &[Cfg::committed_source_contract().unwrap()],
        &policy_of::<RecCfg>(ChunkedWitnessCfg::default_non_chunked()),
        RecCfg::ring_challenge_config,
        zk(),
    )
    .unwrap();
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &planned.schedule.root.params,
        )
        .unwrap(),
        precommitteds: vec![precommitted],
    };
    (profiles, planned.schedule)
}

fn admit_multi(schedule: &FoldSchedule) -> Result<ZkOpeningSchedule, AkitaError> {
    ZkOpeningSchedule::admit(
        schedule,
        RecCfg::committed_source_contract().unwrap(),
        RecCfg::EXT_DEGREE,
    )
}

#[test]
fn masked_levels_admit_a_hiding_precommitted_group_and_a_public_setup_prefix() {
    on_large_stack(|| {
        let (_, schedule) = multi_group_zk_row();
        assert_eq!(admit_multi(&schedule), Ok(zk()));
        let root = &schedule.root.params;
        let [root_precommitted] = root.precommitted_groups() else {
            panic!("one precommitted root group");
        };
        assert_eq!(
            root_precommitted.profile.randomness_columns,
            akita_params::zk::b_randomness_columns(root_precommitted).unwrap()
        );
        let level1 = &schedule.recursive_folds[0].params;
        let level1_prefix = level1.setup_prefix().expect("a level-1 setup prefix");
        assert_eq!(level1_prefix.profile.randomness_columns, 0);
        assert_eq!(akita_params::zk::witness_group_count(root), 2);
        assert_eq!(akita_params::zk::witness_group_count(level1), 1);

        let rejects = |mutate: &dyn Fn(&mut FoldSchedule), what: &str| {
            let mut mutated = schedule.clone();
            mutate(&mut mutated);
            let error = admit_multi(&mutated).unwrap_err();
            assert!(error.to_string().contains(what), "{what}: {error}");
        };
        fn precommitted(s: &mut FoldSchedule) -> &mut akita_params::GroupOpenPhaseParams {
            s.root.params.preceding_group_mut_for_test(0).unwrap()
        }
        fn prefix(s: &mut FoldSchedule) -> &mut akita_params::GroupOpenPhaseParams {
            s.recursive_folds[0]
                .params
                .preceding_group_mut_for_test(0)
                .unwrap()
        }
        let hiding = "level 0 gives a preceding group the wrong B hiding randomness";
        rejects(&|s| precommitted(s).profile.randomness_columns = 0, hiding);
        rejects(&|s| precommitted(s).profile.randomness_columns -= 1, hiding);
        rejects(
            &|s| prefix(s).profile.randomness_columns = 1,
            "level 1 gives a preceding group the wrong B hiding randomness",
        );
        rejects(
            &|s| precommitted(s).opening.num_digits_fold -= 1,
            "level 0 has",
        );
        rejects(&|s| prefix(s).opening.num_digits_fold -= 1, "level 1 has");
        rejects(
            &|s| precommitted(s).opening.log_basis_open += 1,
            "level 0 is masked with an open basis other than 8",
        );
    });
}

/// Jolt's one-hot root: the production zero-knowledge search keeps the
/// edge into the cut level direct at every size, the 36:1 headline and
/// 37 variables included, where the setup alone would favour offloading
/// the root's setup at the cut (`specs/whiteout-akita.md` §5.5.3).
#[test]
fn the_priced_search_never_offloads_into_the_cut_level() {
    on_large_stack(|| {
        for num_vars in [36, 37] {
            let planned = akita_planner::find_zk_schedule(
                &ScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars)),
                RecCfg::committed_source_contract().unwrap(),
                &[],
                &policy_of::<RecCfg>(ChunkedWitnessCfg::default_non_chunked()),
                RecCfg::ring_challenge_config,
                zk(),
            )
            .unwrap();
            assert_eq!(admit_multi(&planned.schedule), Ok(zk()));
            assert!(
                planned.schedule.recursive_folds[ZK_CUT_LEVEL as usize - 1]
                    .params
                    .setup_prefix()
                    .is_none(),
                "{num_vars} variables"
            );
        }
    });
}

/// The root's two witness groups each get a quarter of the 2^-136 budget
/// (two masked levels, two groups); level 1's setup prefix is public, so its
/// own group keeps half.
#[test]
fn witness_groups_split_the_one_shot_budget_per_level() {
    on_large_stack(|| {
        let (_, schedule) = multi_group_zk_row();
        let target = |params: &akita_params::CommittedGroupParams| {
            zk().fold_failure_target(akita_params::zk::witness_group_count(params))
                .unwrap()
        };
        let one_shot = |masked_folds| akita_params::sis::FoldFailureTarget::OneShot {
            security_bits: akita_params::zk::ZK_FOLD_SECURITY_BITS,
            masked_folds,
        };
        assert_eq!(target(&schedule.root.params), one_shot(4));
        assert_eq!(target(&schedule.recursive_folds[0].params), one_shot(2));
    });
}
