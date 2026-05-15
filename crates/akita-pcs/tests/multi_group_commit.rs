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
use akita_prover::{
    batched_commit_with_params, commit_with_params, AkitaPolyOps, AkitaProverSetup, DensePoly,
};
use akita_transcript::{labels::CHALLENGE_RING_SWITCH, Blake2bTranscript, Transcript};
use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AjtaiKeyParams,
    BasisMode, BlockOrder, GroupSpec, LevelParams, RingOpeningPoint, TieredSetupParams,
};
use akita_verifier::{__test_dense_ring_opening_at_point, prepare_m_eval};

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
    // mismatched-(m, r) acceptance test. `max_stride = 64` covers the
    // largest per-group column width across:
    //  - heterogeneous untiered: 2-group layout
    //  - tiered: 3-group layout (W + k chunks + meta) where each
    //    group's z_base region is `num_eval_rows * block_len *
    //    num_digits_commit` and they concatenate (book §5.4
    //    multi-group commit at level L+1)
    AkitaProverSetup::<F, D_TEST>::generate_with_capacity(8, 4, 2, 4, 64)
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

/// Task A invariant (book §5.4 line 752): `setup_weight_table_at_point`
/// × `setup_matrix` must equal `eval_split_at_point.setup` for a
/// tier-marked group too. This is the same structural-vs-materialized
/// consistency the un-tiered heterogeneous test asserts, applied to
/// the block-diagonal `D_chunk` / `B_chunk` collapse path.
///
/// The layout here mirrors the production tiered routing at level L+1:
/// `groups = [W, chunks (tiered, k=4), meta]` under the inferred
/// 1-claim-per-point rule.
#[test]
fn tiered_prepare_m_eval_setup_weight_matches_eval_split() {
    let setup = make_setup();
    let outer_lp = outer_level_params(sample_stage1_config());

    // f = 2 → k = 4 chunks.
    let tier = TieredSetupParams::new(2).expect("f=2 tier");
    // Chunks at PRODUCTION shape: chunk_lp shrinks r_S and m_S each by
    // log_2(f) = 1, so chunks have HALF the num_blocks and HALF the
    // block_len of the outer LP (book §5.4 "1/f the column width").
    // This shape difference is what the verifier-side block-diagonal
    // collapse (Task A) handles; mirroring it in the unit test makes
    // the invariant catch failures the same-shape case missed.
    let mut chunks_spec = GroupSpec {
        m_vars: outer_lp.m_vars.saturating_sub(1),
        r_vars: outer_lp.r_vars.saturating_sub(1),
        num_blocks: outer_lp.num_blocks / 2,
        block_len: outer_lp.block_len.max(2) / 2,
        b_key: outer_lp.b_key.clone(),
        num_digits_commit: outer_lp.num_digits_commit,
        num_digits_open: outer_lp.num_digits_open,
        num_digits_fold: outer_lp.num_digits_fold,
        tier: Some(tier),
    };
    // Force block_len to be at least 1 even when outer was already 1.
    if chunks_spec.block_len == 0 {
        chunks_spec.block_len = 1;
    }

    // Meta is a regular group (no tier marker) — book line 695 "standard
    // Akita commitment".
    let meta_spec = GroupSpec::from_outer(&outer_lp);
    let w_spec = GroupSpec::from_outer(&outer_lp);

    let tiered_lp = LevelParams {
        groups: Some(vec![w_spec, chunks_spec.clone(), meta_spec]),
        ..outer_lp.clone()
    };
    assert!(!tiered_lp.groups_are_homogeneous());

    // Mirror the post-fix verifier: num_eval_rows = num_groups (one
    // y_ring per group, book §5.4 line 949 "share folding challenges"),
    // and claim_to_point maps each claim to its enclosing group index
    // rather than each claim to a distinct point.
    let claim_group_sizes = [1usize, tier.num_chunks, 1usize];
    let num_claims = claim_group_sizes.iter().sum::<usize>();
    let num_eval_rows = claim_group_sizes.len();

    let m_rows = tiered_lp.m_row_count(claim_group_sizes.len(), num_eval_rows);
    let total_b_rows = tiered_lp.total_b_row_count(claim_group_sizes.len());
    // Tier-aware total: W.b + tier.num_chunks * chunks.b + meta.b
    assert_eq!(
        total_b_rows,
        outer_lp.b_key.row_len()
            + tier.num_chunks * chunks_spec.b_key.row_len()
            + outer_lp.b_key.row_len()
    );

    let mut transcript = Blake2bTranscript::<F>::new(b"multi_group_commit/tiered_invariant");
    // Stage-1 needs `sum_g claim_g * num_blocks_g` blocks worth of
    // challenges; the tensor stage-1 in prepare_m_eval rounds up to
    // the next power of two as needed.
    let total_blocks: usize = claim_group_sizes
        .iter()
        .zip([outer_lp.num_blocks, chunks_spec.num_blocks, outer_lp.num_blocks])
        .map(|(c, nb)| c * nb)
        .sum();
    let challenges = sample_stage1_challenges::<F, _, D_TEST>(
        &mut transcript,
        total_blocks,
        1,
        &outer_lp.stage1_config,
        &outer_lp.stage1_challenge_shape,
    )
    .expect("stage1 challenges");

    let alpha: F = transcript.challenge_scalar(CHALLENGE_RING_SWITCH);

    let tau1_len = m_rows.next_power_of_two().trailing_zeros() as usize;
    let tau1: Vec<F> = (0..tau1_len).map(|i| F::from_u64(73 + i as u64)).collect();
    let gamma: Vec<F> = (0..num_claims)
        .map(|i| F::from_u64(101 + i as u64))
        .collect();

    let mut claim_to_point: Vec<usize> = Vec::with_capacity(num_claims);
    for (group_idx, &group_size) in claim_group_sizes.iter().enumerate() {
        for _ in 0..group_size {
            claim_to_point.push(group_idx);
        }
    }
    let prepared = prepare_m_eval::<F, D_TEST>(
        &challenges,
        alpha,
        &tiered_lp,
        &tau1,
        &claim_group_sizes,
        &gamma,
        num_eval_rows,
        num_eval_rows,
        &claim_to_point,
    )
    .expect("tiered prepare_m_eval");

    let opening_points: Vec<RingOpeningPoint<F>> = (0..num_eval_rows)
        .map(|p| RingOpeningPoint {
            a: (0..outer_lp.block_len.max(chunks_spec.block_len))
                .map(|i| F::from_u64(2 + (p * 13 + i) as u64))
                .collect(),
            b: (0..outer_lp.num_blocks.max(chunks_spec.num_blocks))
                .map(|i| F::from_u64(31 + (p * 17 + i) as u64))
                .collect(),
        })
        .collect();

    // x_challenges length must cover the M-table's padded column count.
    let x_bits = {
        let weights = prepared
            .setup_weight_table_at_point::<D_TEST>(
                &vec![F::zero(); 64],
                &setup.expanded,
                alpha,
            )
            .or_else(|_| {
                // Fall back to compute via prepared dims directly when
                // the placeholder x is too long.
                prepared.setup_weight_table_at_point::<D_TEST>(
                    &vec![F::zero(); 32],
                    &setup.expanded,
                    alpha,
                )
            })
            .map(|_| ())
            .ok();
        // Conservative upper bound on x_bits: enough for the largest
        // padded column count of the M-table.
        let _ = weights;
        16usize
    };
    let x_challenges: Vec<F> = (0..x_bits).map(|i| F::from_u64(3 + i as u64)).collect();

    let split = prepared
        .eval_split_at_point::<D_TEST>(&x_challenges, &setup.expanded, &opening_points, alpha)
        .expect("tiered eval_split");
    let weights = prepared
        .setup_weight_table_at_point::<D_TEST>(&x_challenges, &setup.expanded, alpha)
        .expect("tiered setup weights");
    let row_count = prepared.setup_polynomial_row_count();
    // The verifier's weight table is sized for the padded
    // `(row_bits, col_bits, coeff_bits)`; the materialized setup view
    // uses `max_stride` for its column dimension, which may be larger.
    // Take a stride-aware view of exactly the padded col_count.
    let (_, col_bits, _) =
        prepared.setup_polynomial_padded_dims(setup.expanded.seed.max_stride);
    let col_count = 1usize << col_bits;
    let setup_table = setup
        .expanded
        .shared_matrix
        .setup_polynomial_view_with_stride::<D_TEST>(
            row_count,
            col_count,
            setup.expanded.seed.max_stride.max(col_count),
        )
        .materialize_table();
    assert_eq!(
        weights.len(),
        setup_table.len(),
        "weight table length must match the padded setup view"
    );
    let materialized_setup: F = weights
        .iter()
        .zip(setup_table.iter())
        .map(|(w, s)| *w * *s)
        .sum();
    assert_eq!(
        split.setup, materialized_setup,
        "tiered structured setup eval must match the weight-table sum (book §5.4 line 752)"
    );

    // Algebraic-only fast path must equal `split.algebraic`. The
    // claim-reduction verifier reconstructs `m_val = m_alg +
    // m_setup_eval`. If the algebraic-only call diverges from the
    // split's algebraic component, the main stage-2 sumcheck closing
    // (`expected_main == final_running_claim`) fails even when the
    // setup half is consistent.
    let algebraic_only = prepared
        .eval_algebraic_at_point::<D_TEST>(&x_challenges, &opening_points, alpha)
        .expect("tiered algebraic-only eval");
    assert_eq!(
        algebraic_only, split.algebraic,
        "eval_algebraic_at_point must match the algebraic component of \
         eval_split_at_point for tier-marked groups (otherwise the \
         claim-reduction main-sumcheck closing rejects)"
    );

    // CRITICAL: `eval_at_point` MUST be multilinear in `x_challenges`
    // for the prover's `compute_m_evals_x` materialization to be
    // consistent with the verifier's `m_eval(bound_x)`. The prover
    // materializes `m_evals_x[idx] = eval_at_point(boolean(idx))` for
    // idx in 0..2^x_bits, then the stage-2 sumcheck folds via random
    // challenges. After binding, the prover's claim component equals
    // `multilinear_eval(m_evals_x, bound_x)`. The verifier reconstructs
    // `eval_at_point(bound_x)`. For these to match, eval_at_point must
    // BE the multilinear extension over its boolean evaluations.
    let x_bits = x_challenges.len();
    let x_len = 1usize << x_bits;
    let materialized_m: Vec<F> = (0..x_len)
        .map(|idx| {
            let point: Vec<F> = (0..x_bits)
                .map(|bit| {
                    if (idx >> bit) & 1 == 1 {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
                .collect();
            prepared
                .eval_at_point::<D_TEST>(&point, &setup.expanded, &opening_points, alpha)
                .expect("eval_at_point at boolean point")
        })
        .collect();
    let mle_at_bound = akita_sumcheck::multilinear_eval(&materialized_m, &x_challenges)
        .expect("multilinear_eval");
    let combined_at_bound = split.combined();
    assert_eq!(
        mle_at_bound, combined_at_bound,
        "eval_at_point must be the MLE of its boolean evaluations \
         (prover's m_evals_x materialization assumes this; if violated, \
         the stage-2 main sumcheck closing rejects)"
    );
}

/// Cross-check: the verifier's `dense_ring_opening_at_point` (used by
/// `expand_tiered_setup_claims` to compute per-chunk and per-meta
/// openings of the routed setup material) must produce the SAME scalar
/// as the prover's `DensePoly::evaluate_and_fold` path on the same
/// `(coeffs, opening_point, claim_lp)`. The previous implementation
/// reused `y_setup` for every routed claim, causing transcript
/// divergence at the per-claim openings absorption inside
/// `verify_one_level` (book §5.4 line 949 "share folding challenges").
///
/// This test exercises the new MLE reconstruction at small dimensions
/// (runs in milliseconds) and short-circuits the slow tiered E2E if
/// the formula is wrong.
#[test]
fn verifier_dense_ring_opening_matches_prover_evaluate_and_fold() {
    // Tiny dimensions exercise the same algebra without spinning up
    // a full tiered setup. `D = 8` keeps alpha_bits = 3; the poly is
    // 4 ring elements (so target_num_vars = 2 + 0 + 3 = 5).
    const D_LOCAL: usize = 8;
    type FL = Prime128OffsetA7F7;
    let mut buf = [FL::zero(); D_LOCAL];
    let coeffs: Vec<CyclotomicRing<FL, D_LOCAL>> = (0..4)
        .map(|ring_idx| {
            for (k, b) in buf.iter_mut().enumerate() {
                *b = FL::from_u64(((ring_idx * 13 + k * 7 + 1) as u64).wrapping_mul(101));
            }
            CyclotomicRing::<FL, D_LOCAL>::from_coefficients(buf)
        })
        .collect();

    let alpha_bits = D_LOCAL.trailing_zeros() as usize; // 3
    let claim_lp = LevelParams {
        ring_dimension: D_LOCAL,
        log_basis: 1,
        a_key: AjtaiKeyParams::new_unchecked(1, 1, 0, D_LOCAL),
        b_key: AjtaiKeyParams::new_unchecked(1, 1, 0, D_LOCAL),
        d_key: AjtaiKeyParams::new_unchecked(1, 1, 0, D_LOCAL),
        num_blocks: 4,
        block_len: 1,
        m_vars: 0,
        r_vars: 2,
        stage1_config: sample_stage1_config(),
        stage1_challenge_shape: Stage1ChallengeShape::Flat,
        use_setup_claim_reduction: false,
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold: 1,
        groups: None,
    };
    // target_num_vars = m_vars + r_vars + alpha_bits = 0 + 2 + 3 = 5.
    // 2^5 = 32 = 4 ring elements * D_LOCAL = 32 ✓.
    let target_num_vars = claim_lp.m_vars + claim_lp.r_vars + alpha_bits;
    assert_eq!(target_num_vars, 5);

    // A non-trivial opening point. The function pads/truncates to
    // target_num_vars so any length works; use exactly target_num_vars
    // here to match the un-padded case.
    let opening_point: Vec<FL> = (0..target_num_vars)
        .map(|i| FL::from_u64(17 + (i as u64) * 23))
        .collect();

    // Verifier's path: the new helper that
    // `expand_tiered_setup_claims` will call for chunks and meta.
    let verifier_opening = __test_dense_ring_opening_at_point::<FL, D_LOCAL>(
        &coeffs,
        &opening_point,
        &claim_lp,
        alpha_bits,
    )
    .expect("verifier opening");

    // Prover's reference path: exactly mirror
    // `prove_recursive_multi_fold_with_params` lines 1505-1583 for a
    // single dense claim with this claim_lp.
    let inner_point = &opening_point[..alpha_bits];
    let reduced_point = &opening_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field::<FL>(
        reduced_point,
        claim_lp.r_vars,
        claim_lp.m_vars,
        BasisMode::Lagrange,
        BlockOrder::ColumnMajor,
    )
    .expect("ring opening point");
    let inner_reduction =
        reduce_inner_opening_to_ring_element::<FL, D_LOCAL>(inner_point, BasisMode::Lagrange)
            .expect("inner reduction");
    let dense_poly = DensePoly::<FL, D_LOCAL>::from_ring_coeffs(coeffs.clone());
    let (y_ring, _folded) = AkitaPolyOps::<FL, D_LOCAL>::evaluate_and_fold(
        &dense_poly,
        &ring_opening_point.b,
        &ring_opening_point.a,
        claim_lp.block_len,
    );
    let prover_opening = (y_ring * inner_reduction.sigma_m1()).coefficients()[0];

    assert_eq!(
        verifier_opening, prover_opening,
        "verifier dense_ring_opening_at_point disagrees with prover's \
         DensePoly::evaluate_and_fold path; transcript would diverge in tiered routing"
    );
}
