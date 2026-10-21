//! Clear-protocol proofs over schedules whose first levels carry hiding
//! randomness columns (Whiteout W1).
//!
//! The randomness widths come from the planner's test-only hiding search, so
//! every level up to the cut appends fresh randomness to its B slices and its
//! D image. The clear verifier must accept such proofs unchanged.

#![allow(missing_docs)]

use akita_config::{
    policy_of, proof_optimized::fp128, CommitmentConfig, TrustedScheduleCatalog,
    ValidatedScheduleCatalog,
};
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_params::{
    lagrange_weights, suffix_opening_layout, BasisMode, ChunkedWitnessCfg,
    CommittedGroupBatchProfile, FoldSchedule, GroupCommitPhaseParams, PolynomialGroupLayout,
    RelationQuotientPlan, RelationWitnessGeometry, ScheduleLookupKey, WitnessLayout,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_algebra::{One, Ring, Zero};
use rand::rngs::StdRng;
use rand::SeedableRng;

type Cfg = fp128::Dense;
type F = <Cfg as CommitmentConfig>::Field;

const NV: usize = 14;
const CUT_LEVEL: usize = 1;
const SESSION: &[u8] = b"hiding-randomness-e2e";

fn hiding_schedule() -> (CommittedGroupBatchProfile, FoldSchedule) {
    hiding_schedule_to(CUT_LEVEL)
}

fn hiding_schedule_to(cut_level: usize) -> (CommittedGroupBatchProfile, FoldSchedule) {
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(NV));
    let policy = policy_of::<Cfg>(ChunkedWitnessCfg::default_non_chunked());
    let planned = akita_planner::find_schedule_for_test_hiding(
        &key,
        Cfg::committed_source_contract().unwrap(),
        &policy,
        Cfg::ring_challenge_config,
        cut_level,
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

fn hiding_catalog(cut_level: usize) -> TrustedScheduleCatalog<Cfg> {
    let (profiles, schedule) = hiding_schedule_to(cut_level);
    let policy = policy_of::<Cfg>(ChunkedWitnessCfg::default_non_chunked());
    let validated = ValidatedScheduleCatalog::try_new_with_test_hiding(
        Cfg::schedule_family_name(),
        [(profiles, schedule)],
        &policy,
        Cfg::ring_challenge_config,
        cut_level,
    )
    .unwrap();
    TrustedScheduleCatalog::new(validated, ChunkedWitnessCfg::default_non_chunked()).unwrap()
}

/// A trusted catalog rejects hiding randomness until zero knowledge is
/// selectable.
#[test]
fn trusted_catalogs_reject_hiding_randomness() {
    on_large_stack(|| {
        let (profiles, schedule) = hiding_schedule();
        let policy = policy_of::<Cfg>(ChunkedWitnessCfg::default_non_chunked());
        let error = ValidatedScheduleCatalog::try_new(
            Cfg::schedule_family_name(),
            [(profiles, schedule)],
            &policy,
            Cfg::ring_challenge_config,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not selectable"), "{error}");
    });
}

/// A seeded hiding source, so every hiding proof here is reproducible.
fn backend(setup: &akita_pcs::AkitaProverSetup<F>, seed: u64) -> CpuBackend<F, F> {
    CpuBackend::new(setup.expanded.clone())
        .unwrap()
        .with_hiding_rng(StdRng::seed_from_u64(seed))
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
fn planner_sizes_randomness_at_every_level_up_to_the_cut() {
    on_large_stack(|| {
        let (profiles, schedule) = hiding_schedule();
        let levels = std::iter::once(&schedule.root.params)
            .chain(schedule.recursive_folds.iter().map(|fold| &fold.params))
            .collect::<Vec<_>>();
        assert!(
            levels.len() > CUT_LEVEL,
            "the cut must precede the terminal"
        );
        for (level, params) in levels.iter().enumerate() {
            let hides = level <= CUT_LEVEL;
            assert_eq!(
                params.own_group().profile.randomness_columns != 0,
                hides,
                "level {level} B randomness"
            );
            assert_eq!(
                params.open_randomness_columns != 0,
                hides,
                "level {level} D randomness"
            );
        }
        assert_eq!(
            profiles.final_group.randomness_columns,
            schedule.root.params.own_group().profile.randomness_columns
        );
        akita_params::zk::validate_hiding_randomness(&schedule, CUT_LEVEL).unwrap();
    });
}

#[test]
fn validation_recomputes_widths_and_rejects_a_narrow_schedule() {
    on_large_stack(|| {
        let (_, schedule) = hiding_schedule();
        for level in 0..=CUT_LEVEL {
            let mut narrow_b = schedule.clone();
            let mut narrow_d = schedule.clone();
            let (b_params, d_params) = if level == 0 {
                (&mut narrow_b.root.params, &mut narrow_d.root.params)
            } else {
                (
                    &mut narrow_b.recursive_folds[level - 1].params,
                    &mut narrow_d.recursive_folds[level - 1].params,
                )
            };
            b_params.own_group_mut().profile.randomness_columns -= 1;
            d_params.open_randomness_columns -= 1;
            for narrow in [&narrow_b, &narrow_d] {
                assert!(
                    akita_params::zk::validate_hiding_randomness(narrow, CUT_LEVEL).is_err(),
                    "level {level}"
                );
            }
        }
        // A cut one level earlier leaves level 1's randomness unaccounted for.
        assert!(akita_params::zk::validate_hiding_randomness(&schedule, CUT_LEVEL - 1).is_err());
        // The deepest admissible cut is the fold before the last nonterminal
        // one; a cut at the last nonterminal fold would put its randomness in
        // the terminal witness, which is sent in the clear.
        let deepest = schedule.recursive_folds.len() - 1;
        let (_, deepest_schedule) = hiding_schedule_to(deepest);
        akita_params::zk::validate_hiding_randomness(&deepest_schedule, deepest).unwrap();
        let error = akita_params::zk::validate_hiding_randomness(&deepest_schedule, deepest + 1)
            .unwrap_err();
        assert!(error.to_string().contains("terminal witness"), "{error}");
    });
}

/// Every randomness span lies in the live witness prefix, which the next
/// level's ring switch requires its digit-range domain to cover exactly, so
/// the range check proves every randomness digit.
#[test]
fn randomness_sits_inside_the_range_checked_witness() {
    on_large_stack(|| {
        let (_, schedule) = hiding_schedule();
        let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(NV));
        let levels = std::iter::once((&schedule.root, key.opening_layout().unwrap())).chain(
            schedule.recursive_folds.iter().map(|fold| {
                (
                    fold,
                    suffix_opening_layout(fold.input_witness_len, None).unwrap(),
                )
            }),
        );
        for (level, (fold, batch)) in levels.take(CUT_LEVEL + 1).enumerate() {
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

#[test]
fn hiding_commitments_to_one_polynomial_differ() {
    on_large_stack(|| {
        let scheme = AkitaCommitmentScheme::<Cfg>::new(hiding_catalog(CUT_LEVEL));
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

/// Commit, prove an opening, and verify it. Returns whether the verifier
/// accepted the claimed opening; a wrong opening is always rejected.
fn prove_and_verify(cut_level: usize) -> bool {
    let scheme = AkitaCommitmentScheme::<Cfg>::new(hiding_catalog(cut_level));
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
    let point: Vec<F> = (0..NV)
        .map(|index| F::from_u64((5 * index + 2) as u64))
        .collect();
    let opening = evals
        .iter()
        .zip(lagrange_weights(&point).unwrap())
        .fold(F::zero(), |sum, (&value, weight)| sum + value * weight);
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
        let group = PolynomialGroupClaims::new(point.clone(), vec![claimed], &commitment).unwrap();
        let claims = OpeningClaims::from_groups(vec![group]).unwrap();
        let statement = GroupBatchStatement::new(selection, claims).unwrap();
        verifier.verify_standalone(&proof, SESSION, statement, BasisMode::Lagrange)
    };
    assert!(verify(opening + F::one()).is_err());
    verify(opening).is_ok()
}

#[test]
fn clear_protocol_proof_with_hiding_randomness_verifies() {
    on_large_stack(|| assert!(prove_and_verify(CUT_LEVEL)));
}

/// Hiding one level past the headline cut reaches a reduced-evaluation level,
/// whose quotient-free relation weights and setup scan carry the randomness
/// too.
#[test]
fn reduced_relation_level_with_randomness_verifies() {
    on_large_stack(|| {
        let (_, schedule) = hiding_schedule_to(CUT_LEVEL + 1);
        assert!(
            schedule.recursive_folds[CUT_LEVEL]
                .params
                .ring_relation_mode
                .is_reduced_evaluation(),
            "the extra hiding level must use the reduced relation"
        );
        assert!(prove_and_verify(CUT_LEVEL + 1));
    });
}
