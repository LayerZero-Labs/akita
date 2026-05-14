//! Slice D acceptance test: multi-group batched Hachi commit kernel.
//!
//! Verifies the multi-group commit primitive
//! ([`batched_commit_with_params`] under `LevelParams.groups == Some(vec)`)
//! produces per-group commitments that exactly match the per-group result
//! of `commit_with_params(group_polys, setup, &spec.lower_into_outer(&lp))`
//! at mismatched `(m_g, r_g, B_g, δ_open_g)`.
//!
//! Also covers the heterogeneous M-eval entry point: when
//! `LevelParams.groups == Some(vec)` contains heterogeneous specs,
//! [`prepare_m_eval`] accepts the per-group row layout instead of
//! silently collapsing to the outer single-LP shape.
//!
//! Per `specs/phase-d-full-design.md` §6 Slice D acceptance:
//!  - Multi-group commit at root with two polys at mismatched `(m, r)`
//!    produces per-group `u_g` matching the per-group single-LP result.
//!  - The `groups == None` path stays bit-equivalent (verified by the
//!    rest of the workspace tests passing).
//!  - `prepare_m_eval` accepts heterogeneous multi-group LP row layouts.

use akita_algebra::CyclotomicRing;
use akita_challenges::{sample_stage1_challenges, SparseChallengeConfig, Stage1ChallengeShape};
use akita_field::Prime128OffsetA7F7;
use akita_prover::{batched_commit_with_params, commit_with_params, AkitaProverSetup, DensePoly};
use akita_transcript::{labels::CHALLENGE_RING_SWITCH, Blake2bTranscript, Transcript};
use akita_types::{AjtaiKeyParams, GroupSpec, LevelParams, RingOpeningPoint};
use akita_verifier::prepare_m_eval;

type F = Prime128OffsetA7F7;
const D_TEST: usize = 32;

/// Build an outer `LevelParams` whose direct (outer) fields describe
/// group 1's shape. The `groups` field is `None` initially; tests
/// override it to inject heterogeneous specs.
fn outer_level_params(stage1_config: SparseChallengeConfig) -> LevelParams {
    LevelParams {
        ring_dimension: D_TEST,
        log_basis: 2,
        a_key: AjtaiKeyParams::new_unchecked(1, 8, 0, D_TEST),
        b_key: AjtaiKeyParams::new_unchecked(2, 4, 0, D_TEST),
        d_key: AjtaiKeyParams::new_unchecked(1, 2, 0, D_TEST),
        num_blocks: 4,
        block_len: 2,
        m_vars: 1,
        r_vars: 2,
        stage1_config,
        stage1_challenge_shape: Stage1ChallengeShape::Flat,
        use_setup_claim_reduction: false,
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold: 1,
        groups: None,
    }
}

fn sample_stage1_config() -> SparseChallengeConfig {
    SparseChallengeConfig::Uniform {
        weight: 1,
        nonzero_coeffs: vec![-1, 1],
    }
}

fn make_setup() -> AkitaProverSetup<F, D_TEST> {
    // `max_rows = 4` covers `b_key.row_len = 3` from group 2 in the
    // mismatched-(m, r) acceptance test. `max_stride = 8` covers the
    // largest per-group B-matrix column width
    // (`num_blocks * a_key.row_len * num_digits_open`) used by either
    // group, including the outer LP's homogeneous fall-back.
    AkitaProverSetup::<F, D_TEST>::generate_with_capacity(8, 4, 2, 4, 8)
        .expect("test setup must generate")
}

fn dense_poly(num_vars: usize, seed: u64) -> DensePoly<F, D_TEST> {
    let mut buf = [F::zero(); D_TEST];
    let coeffs = (0..(1 << num_vars))
        .map(|i| {
            buf[0] = F::from_u64(seed + i as u64);
            CyclotomicRing::<F, D_TEST>::from_coefficients(buf)
        })
        .collect::<Vec<_>>();
    DensePoly::<F, D_TEST>::from_ring_coeffs(coeffs)
}

/// Slice D acceptance: with `LevelParams.groups == Some([spec_g1,
/// spec_g2])` at mismatched `(m, r, B, δ_open)`, the per-group output
/// of `batched_commit_with_params` matches the independent per-group
/// `commit_with_params` invoked with the lowered single-group LP.
#[test]
fn batched_commit_with_heterogeneous_groups_matches_per_group_commit() {
    let setup = make_setup();
    let outer_lp = outer_level_params(sample_stage1_config());

    // Two groups with different (m_g, r_g, B_g, δ_open_g). Polynomials
    // in both groups have `num_vars == 3` (eight ring elements) so the
    // setup can host them both, but the per-group block-shape differs:
    //
    //  - g1: (m=1, r=2) -> num_blocks=4, block_len=2, b_key.row_len=2
    //  - g2: (m=2, r=1) -> num_blocks=2, block_len=4, b_key.row_len=3
    let spec_g1 = GroupSpec {
        m_vars: 1,
        r_vars: 2,
        num_blocks: 4,
        block_len: 2,
        b_key: AjtaiKeyParams::new_unchecked(2, 4, 0, D_TEST),
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold: 1,
        tier: None,
    };
    let spec_g2 = GroupSpec {
        m_vars: 2,
        r_vars: 1,
        num_blocks: 2,
        block_len: 4,
        b_key: AjtaiKeyParams::new_unchecked(3, 4, 0, D_TEST),
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold: 1,
        tier: None,
    };
    assert_ne!(spec_g1, spec_g2);

    let multi_lp = LevelParams {
        groups: Some(vec![spec_g1.clone(), spec_g2.clone()]),
        ..outer_lp.clone()
    };
    assert!(
        !multi_lp.groups_are_homogeneous(),
        "test LP should expose the heterogeneous multi-group case"
    );

    let poly_g1 = dense_poly(3, 100);
    let poly_g2 = dense_poly(3, 200);

    let multi_groups: &[&[DensePoly<F, D_TEST>]] = &[
        std::slice::from_ref(&poly_g1),
        std::slice::from_ref(&poly_g2),
    ];
    let (multi_commits, multi_hints) =
        batched_commit_with_params::<F, D_TEST, DensePoly<F, D_TEST>>(
            multi_groups,
            &setup,
            &multi_lp,
        )
        .expect("multi-group commit must succeed at mismatched (m, r)");
    assert_eq!(multi_commits.len(), 2);
    assert_eq!(multi_hints.len(), 2);

    let lp_g1 = spec_g1.lower_into_outer(&outer_lp);
    let lp_g2 = spec_g2.lower_into_outer(&outer_lp);

    let (per_group_g1, _hint_g1) =
        commit_with_params(std::slice::from_ref(&poly_g1), &setup, &lp_g1)
            .expect("per-group g1 commit");
    let (per_group_g2, _hint_g2) =
        commit_with_params(std::slice::from_ref(&poly_g2), &setup, &lp_g2)
            .expect("per-group g2 commit");

    assert_eq!(
        multi_commits[0], per_group_g1,
        "multi-group g1 commitment must match the per-group g1 commit_with_params output"
    );
    assert_eq!(
        multi_commits[1], per_group_g2,
        "multi-group g2 commitment must match the per-group g2 commit_with_params output"
    );

    assert_eq!(multi_commits[0].u.len(), spec_g1.b_key.row_len());
    assert_eq!(multi_commits[1].u.len(), spec_g2.b_key.row_len());
}

/// Regression: `groups == None` and `groups == Some(vec![outer; n])`
/// where every spec equals `GroupSpec::from_outer(&outer_lp)` must
/// produce bit-equivalent commitments.
#[test]
fn batched_commit_with_homogeneous_groups_matches_none() {
    let setup = make_setup();
    let outer_lp = outer_level_params(sample_stage1_config());
    let outer_spec = GroupSpec::from_outer(&outer_lp);

    let poly_a = dense_poly(3, 11);
    let poly_b = dense_poly(3, 22);

    let group_a: &[&[DensePoly<F, D_TEST>]] =
        &[std::slice::from_ref(&poly_a), std::slice::from_ref(&poly_b)];

    let (commits_none, _hints_none) =
        batched_commit_with_params::<F, D_TEST, DensePoly<F, D_TEST>>(group_a, &setup, &outer_lp)
            .expect("baseline groups=None must commit");

    let outer_with_homogeneous = LevelParams {
        groups: Some(vec![outer_spec.clone(), outer_spec.clone()]),
        ..outer_lp.clone()
    };
    assert!(
        outer_with_homogeneous.groups_are_homogeneous(),
        "outer-spec replication must collapse to the homogeneous case"
    );

    let (commits_homo, _hints_homo) =
        batched_commit_with_params::<F, D_TEST, DensePoly<F, D_TEST>>(
            group_a,
            &setup,
            &outer_with_homogeneous,
        )
        .expect("homogeneous groups=Some(vec![outer;n]) must commit identically");

    assert_eq!(commits_none, commits_homo);
}

/// `prepare_m_eval` must accept `LevelParams.groups == Some(vec)` with
/// heterogeneous per-group specs and size the row layout from the per-group
/// B ranks.
#[test]
fn prepare_m_eval_accepts_heterogeneous_groups() {
    let setup = make_setup();
    let outer_lp = outer_level_params(sample_stage1_config());

    let mut transcript = Blake2bTranscript::<F>::new(b"multi_group_commit/prepare_m_eval");
    let challenges = sample_stage1_challenges::<F, _, D_TEST>(
        &mut transcript,
        6,
        1,
        &outer_lp.stage1_config,
        &outer_lp.stage1_challenge_shape,
    )
    .expect("stage1 challenges");

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let spec_g1 = GroupSpec {
        m_vars: 1,
        r_vars: 2,
        num_blocks: 4,
        block_len: 2,
        b_key: AjtaiKeyParams::new_unchecked(2, 4, 0, D_TEST),
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold: 1,
        tier: None,
    };
    let spec_g2 = GroupSpec {
        m_vars: 2,
        r_vars: 1,
        num_blocks: 2,
        block_len: 4,
        b_key: AjtaiKeyParams::new_unchecked(3, 4, 0, D_TEST),
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold: 1,
        tier: None,
    };
    let expected_rows = outer_lp.d_key.row_len()
        + spec_g1.b_key.row_len()
        + spec_g2.b_key.row_len()
        + 2
        + 1
        + outer_lp.a_key.row_len();
    let heterogeneous_lp = LevelParams {
        groups: Some(vec![spec_g1, spec_g2]),
        ..outer_lp.clone()
    };
    assert!(!heterogeneous_lp.groups_are_homogeneous());

    let tau1_len = heterogeneous_lp
        .m_row_count(2, 2)
        .next_power_of_two()
        .trailing_zeros() as usize;
    let tau1 = vec![F::one(); tau1_len];
    let gamma = vec![F::one(); 2];

    let prepared = prepare_m_eval::<F, D_TEST>(
        &challenges,
        alpha,
        &heterogeneous_lp,
        &tau1,
        &[1, 1],
        &gamma,
        2,
        2,
        &[0, 1],
    )
    .expect("prepare_m_eval must accept heterogeneous multi-group LP");

    assert_eq!(heterogeneous_lp.m_row_count(2, 2), expected_rows);

    let opening_points = vec![
        RingOpeningPoint {
            a: vec![
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(43),
                F::from_u64(47),
            ],
            b: vec![
                F::from_u64(5),
                F::from_u64(7),
                F::from_u64(11),
                F::from_u64(13),
            ],
        },
        RingOpeningPoint {
            a: vec![
                F::from_u64(17),
                F::from_u64(19),
                F::from_u64(23),
                F::from_u64(29),
            ],
            b: vec![
                F::from_u64(31),
                F::from_u64(37),
                F::from_u64(53),
                F::from_u64(59),
            ],
        },
    ];
    let x_challenges = vec![F::from_u64(41); 6];
    let split = prepared
        .eval_split_at_point::<D_TEST>(&x_challenges, &setup.expanded, &opening_points, alpha)
        .expect("heterogeneous prepared M-eval should evaluate");
    let weights = prepared
        .setup_weight_table_at_point::<D_TEST>(&x_challenges, &setup.expanded, alpha)
        .expect("heterogeneous setup weights");
    let row_count = prepared.setup_polynomial_row_count();
    let setup_table = setup
        .expanded
        .shared_matrix
        .setup_polynomial_view::<D_TEST>(row_count, setup.expanded.seed.max_stride)
        .materialize_table();
    let materialized_setup: F = weights
        .iter()
        .zip(setup_table.iter())
        .map(|(w, s)| *w * *s)
        .sum();
    assert_eq!(split.setup, materialized_setup);
}
