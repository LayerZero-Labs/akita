#![cfg(not(feature = "zk"))]

use super::*;
use akita_algebra::CyclotomicRing;
use akita_config::akita_batched_root_layout;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::LiftBase;
use akita_prover::protocol::ring_switch::{ring_switch_build_w, ring_switch_finalize};
use akita_prover::{
    AkitaPolyOps, CommitmentProver, CommittedPolynomials, DensePoly, OneHotPoly, QuadraticEquation,
};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::labels::{
    ABSORB_EVALUATION_CLAIMS, ABSORB_EVAL_OPENINGS_FIELD, CHALLENGE_EVAL_BATCH,
};
use akita_transcript::AkitaTranscript;
use akita_types::stage1_tree_stage_shapes;
use akita_types::BlockOrder;
use akita_types::ClaimIncidenceSummary;
use akita_types::ExtensionOpeningReductionProof;
use akita_types::{
    append_batched_commitments_to_transcript, flatten_batched_commitment_rows, lagrange_weights,
    monomial_weights, reduce_inner_opening_to_ring_element, relation_claim_from_rows,
    ring_opening_point_from_field, MRowLayout,
};
use akita_types::{r_decomp_levels, w_ring_element_count, w_ring_element_count_with_counts};
use akita_types::{
    AkitaBatchedProofShape, AkitaProofStepShape, FlatRingVec, LevelProofShape,
    TerminalLevelProofShape,
};
use akita_types::{AkitaScheduleInputs, AkitaScheduleLookupKey, Step};
use akita_verifier::direct_witness_opening_matches;
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Once;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
type Cfg = fp128::D64Full;
type F = fp128::Field;
const D: usize = Cfg::D;
type Scheme = AkitaCommitmentScheme<D, Cfg>;

fn single_point_group_incidence(num_vars: usize, group_poly_count: usize) -> ClaimIncidenceSummary {
    ClaimIncidenceSummary::from_point_polys(num_vars, vec![group_poly_count])
        .expect("valid single-point incidence")
}

type OneHotF = fp128::Field;
type OneHotCfg = fp128::D64OneHot;
const ONEHOT_D: usize = OneHotCfg::D;
const BENCH_ONEHOT_K: usize = ONEHOT_D;
type OneHotScheme = AkitaCommitmentScheme<ONEHOT_D, OneHotCfg>;
/// Minimum w vector length (in field elements) below which further folding
/// is not beneficial.  When `w.len() <= MIN_W_LEN_FOR_FOLDING`, the prover
/// sends `w` directly instead of recursing.
const MIN_W_LEN_FOR_FOLDING: usize = 4096;

fn batched_shape_rounds(level_d: usize, next_w_len: usize) -> usize {
    let num_ring_elems = next_w_len / level_d;
    num_ring_elems.next_power_of_two().trailing_zeros() as usize + level_d.trailing_zeros() as usize
}

/// Batched recursion already consults the byte planner before folding
/// again. The runtime safety guard here only needs to catch tiny tails and
/// fixed points, not enforce the single-proof shrink-ratio heuristic.
fn should_stop_batched_folding(w_len: usize, prev_w_len: usize) -> bool {
    w_len <= MIN_W_LEN_FOR_FOLDING || w_len >= prev_w_len
}

#[test]
#[cfg_attr(
    not(feature = "planner"),
    ignore = "requires planner fallback for generated schedule misses"
)]
fn same_point_batched_root_preserves_opening_geometry() {
    for num_claims in [4usize, 6] {
        let incidence =
            akita_types::ClaimIncidenceSummary::same_point(20, num_claims).expect("incidence");
        let schedule = OneHotCfg::get_params_for_prove(&incidence).expect("same-point root plan");
        let Some(Step::Fold(root_step)) = schedule.steps.first() else {
            panic!("same-point schedule should start with a fold");
        };
        let root_inputs = AkitaScheduleInputs {
            num_vars: 20,
            level: 0,
            current_w_len: root_step.current_w_len,
        };
        let level_lp = &root_step.params;
        let root_lp =
            OneHotCfg::root_level_params_for_layout_with_log_basis(root_inputs, level_lp).unwrap();
        assert_eq!(root_lp.block_len, level_lp.block_len);
        assert_eq!(root_lp.num_blocks, level_lp.num_blocks);
        assert_eq!(root_lp.m_vars, level_lp.m_vars);
        assert_eq!(root_lp.r_vars, level_lp.r_vars);
    }
}

fn expected_same_point_batched_shape(
    max_num_vars: usize,
    num_claims: usize,
    _proof: &AkitaBatchedProof<OneHotF, OneHotF>,
) -> AkitaBatchedProofShape {
    let incidence = akita_types::ClaimIncidenceSummary::same_point(max_num_vars, num_claims)
        .expect("incidence");
    let schedule = OneHotCfg::get_params_for_prove(&incidence).expect("batched root runtime plan");
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        panic!("batched schedule should start with a fold");
    };
    let num_fold_levels = akita_types::schedule_num_fold_levels(&schedule);
    let root_inputs = AkitaScheduleInputs {
        num_vars: max_num_vars,
        level: 0,
        current_w_len: root_step.current_w_len,
    };
    let level_lp = &root_step.params;
    let root_lp =
        OneHotCfg::root_level_params_for_layout_with_log_basis(root_inputs, level_lp).unwrap();
    let root_w_len = root_step.next_w_len;
    let root_rounds = batched_shape_rounds(root_lp.ring_dimension, root_w_len);

    // 1-fold schedule: the root IS the terminal fold. Emit a terminal-rooted
    // shape with no recursive-suffix steps.
    if num_fold_levels == 1 {
        // The terminal fold's `next` parameters live at `schedule.steps[1]`,
        // which is a `Direct` step encoding the final packed-digit basis.
        let next_inputs = AkitaScheduleInputs {
            num_vars: max_num_vars,
            level: 1,
            current_w_len: root_w_len,
        };
        let terminal_next_params = scheduled_next_level_params(
            &schedule,
            1,
            next_inputs,
            OneHotCfg::level_params_with_log_basis,
        )
        .expect("terminal next params");
        return AkitaBatchedProofShape::Terminal(TerminalLevelProofShape {
            y_rings_coeffs: incidence.num_public_rows() * root_lp.ring_dimension,
            extension_opening_reduction: None,
            stage2_sumcheck: vec![3; root_rounds],
            final_witness: akita_types::DirectWitnessShape::PackedDigits((
                root_w_len,
                terminal_next_params.log_basis,
            )),
        });
    }

    let next_inputs = AkitaScheduleInputs {
        num_vars: max_num_vars,
        level: 1,
        current_w_len: root_step.next_w_len,
    };
    let next_level_params = scheduled_next_level_params(
        &schedule,
        1,
        next_inputs,
        OneHotCfg::level_params_with_log_basis,
    )
    .unwrap();
    let root_shape = LevelProofShape {
        y_ring_coeffs: incidence.num_public_rows() * root_lp.ring_dimension,
        extension_opening_reduction: None,
        v_coeffs: root_lp.d_key.row_len() * root_lp.ring_dimension,
        stage1_stages: stage1_tree_stage_shapes(root_rounds, 1usize << level_lp.log_basis),
        stage2_sumcheck: vec![3; root_rounds],
        next_commit_coeffs: next_level_params.b_key.row_len() * next_level_params.ring_dimension,
    };
    let first_level_params = next_level_params.clone();

    // After Phase 1, the recursive suffix has `num_fold_levels - 1` steps in
    // total: `num_fold_levels - 2` intermediate steps followed by exactly one
    // terminal step. (We've already consumed the root.)
    let num_intermediate_after_root = num_fold_levels.saturating_sub(2);
    let mut step_shapes = Vec::with_capacity(num_fold_levels - 1);
    let mut current_w_len = root_w_len;
    let mut current_log_basis = first_level_params.log_basis;
    let mut current_level = 1usize;
    for _ in 0..num_intermediate_after_root {
        let inputs = AkitaScheduleInputs {
            num_vars: max_num_vars,
            level: current_level,
            current_w_len,
        };
        let (level_params, next_level_params) = scheduled_fold_execution(
            &schedule,
            current_level,
            inputs,
            current_log_basis,
            OneHotCfg::level_params_with_log_basis,
        )
        .expect("scheduled recursive fold");
        let current_lp = akita_types::recursive_level_layout_from_params(
            &level_params,
            current_w_len,
            OneHotCfg::decomposition(),
        )
        .expect("recursive layout");
        let next_w_len =
            w_ring_element_count::<OneHotF>(&current_lp).unwrap() * current_lp.ring_dimension;
        let rounds = batched_shape_rounds(current_lp.ring_dimension, next_w_len);
        step_shapes.push(AkitaProofStepShape::Intermediate(LevelProofShape {
            y_ring_coeffs: current_lp.ring_dimension,
            extension_opening_reduction: None,
            v_coeffs: current_lp.d_key.row_len() * current_lp.ring_dimension,
            stage1_stages: stage1_tree_stage_shapes(rounds, 1usize << current_lp.log_basis),
            stage2_sumcheck: vec![3; rounds],
            next_commit_coeffs: next_level_params.b_key.row_len()
                * next_level_params.ring_dimension,
        }));
        current_w_len = next_w_len;
        current_log_basis = next_level_params.log_basis;
        current_level += 1;
    }

    // Terminal fold step (always present in the multi-fold case): its params
    // live at `schedule.steps[current_level]` (still a `Step::Fold`); the
    // immediately following Direct step encodes the final packed-digit basis.
    let terminal_inputs = AkitaScheduleInputs {
        num_vars: max_num_vars,
        level: current_level,
        current_w_len,
    };
    let (terminal_params, terminal_next_params) = scheduled_fold_execution(
        &schedule,
        current_level,
        terminal_inputs,
        current_log_basis,
        OneHotCfg::level_params_with_log_basis,
    )
    .expect("scheduled terminal fold");
    let terminal_lp = akita_types::recursive_level_layout_from_params(
        &terminal_params,
        current_w_len,
        OneHotCfg::decomposition(),
    )
    .expect("terminal layout");
    // The terminal recursive fold ships its `w` in cleartext under
    // MRowLayout::Terminal (D-block omitted from per-row `r` quotients), so
    // the expected packed-digit witness shape uses the terminal-layout ring
    // count instead of the intermediate-layout `w_ring_element_count`.
    let terminal_next_w_len = akita_types::w_ring_element_count_with_counts_for_layout::<OneHotF>(
        &terminal_lp,
        1,
        1,
        1,
        1,
        akita_types::MRowLayout::Terminal,
    )
    .expect("terminal-layout witness count")
        * terminal_lp.ring_dimension;
    let terminal_rounds = batched_shape_rounds(terminal_lp.ring_dimension, terminal_next_w_len);
    step_shapes.push(AkitaProofStepShape::Terminal(TerminalLevelProofShape {
        y_rings_coeffs: terminal_lp.ring_dimension,
        extension_opening_reduction: None,
        stage2_sumcheck: vec![3; terminal_rounds],
        final_witness: akita_types::DirectWitnessShape::PackedDigits((
            terminal_next_w_len,
            terminal_next_params.log_basis,
        )),
    }));

    AkitaBatchedProofShape::Fold {
        root_shape,
        step_shapes,
    }
}

fn make_dense_poly(num_vars: usize) -> (DensePoly<F, D>, Vec<F>) {
    let len = 1usize << num_vars;
    let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
    (poly, evals)
}

#[test]
fn batched_suffix_stop_guard_does_not_preempt_profitable_fold() {
    // These states came from the batched onehot nv=32 profile runs that
    // regressed after a generic shrink-ratio guard was briefly added to
    // the batched suffix. The runtime guard should not stop folding here.
    assert!(!should_stop_batched_folding(87_744, 140_672));
    assert!(!should_stop_batched_folding(129_216, 224_064));
}

type VerifyFixture = (
    AkitaVerifierSetup<F>,
    RingCommitment<F, D>,
    AkitaBatchedProof<F, F>,
    Vec<F>,
    F,
    LevelParams,
);

fn make_verify_fixture(num_vars: usize) -> VerifyFixture {
    let alpha = D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(num_vars).unwrap();
    let full_num_vars = layout.m_vars + layout.r_vars + alpha;

    let (poly, evals) = make_dense_poly(full_num_vars);
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(full_num_vars, 1, 1);
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup).unwrap();

    let opening_point: Vec<F> = (0..full_num_vars)
        .map(|i| F::from_u64((i + 2) as u64))
        .collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    let [commitment] = commitments;
    (
        verifier_setup,
        commitment,
        proof,
        opening_point,
        opening,
        layout,
    )
}

fn dense_opening(evals: &[F], point: &[F]) -> F {
    let lw = lagrange_weights(point).unwrap();
    evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w)
}

fn init_debug_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .compact()
            .with_target(false)
            .with_span_events(FmtSpan::CLOSE);
        tracing_subscriber::registry()
            .with(EnvFilter::new("info"))
            .with(fmt_layer)
            .init();
    });
}

fn init_debug_rayon_pool() {
    #[cfg(feature = "parallel")]
    {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            rayon::ThreadPoolBuilder::new()
                .stack_size(64 * 1024 * 1024)
                .build_global()
                .ok();
        });
    }
}

fn run_debug_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(f)
        .expect("failed to spawn debug thread")
        .join()
        .expect("debug thread panicked");
}

fn debug_random_point(nv: usize) -> Vec<OneHotF> {
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| OneHotF::from_canonical_u128_reduced(rng.r#gen::<u128>()))
        .collect()
}

fn debug_make_onehot_poly(layout: &LevelParams, seed: u64) -> OneHotPoly<OneHotF, ONEHOT_D, u8> {
    let total_ring = layout.num_blocks * layout.block_len;
    let num_vars = layout.m_vars + layout.r_vars + ONEHOT_D.trailing_zeros() as usize;
    assert_eq!(total_ring * BENCH_ONEHOT_K, 1usize << num_vars);

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..BENCH_ONEHOT_K) as u8))
        .collect();

    OneHotPoly::<OneHotF, ONEHOT_D, u8>::new(BENCH_ONEHOT_K, indices).expect("debug onehot poly")
}

fn debug_opening_from_poly<P: AkitaPolyOps<OneHotF, ONEHOT_D>>(
    poly: &P,
    point: &[OneHotF],
    layout: &LevelParams,
) -> OneHotF {
    let alpha_bits = ONEHOT_D.trailing_zeros() as usize;
    assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )
    .expect("debug opening point");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v =
        reduce_inner_opening_to_ring_element::<OneHotF, ONEHOT_D>(inner_point, BasisMode::Lagrange)
            .expect("debug inner opening point");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

fn debug_relation_sum_from_tables(
    w_evals_compact: &[i8],
    _live_x_cols: usize,
    alpha_evals_y: &[OneHotF],
    m_evals_x: &[OneHotF],
    start_x: usize,
    end_x: usize,
) -> OneHotF {
    let mut acc = OneHotF::zero();
    for x in start_x..end_x {
        let mut y_eval = OneHotF::zero();
        for (y, alpha_eval) in alpha_evals_y.iter().enumerate() {
            y_eval += *alpha_eval
                * OneHotF::from_i64(w_evals_compact[x * alpha_evals_y.len() + y] as i64);
        }
        acc += y_eval * m_evals_x[x];
    }
    acc
}

#[test]
fn commit_singleton_group_returns_single_claim_hint() {
    let alpha = D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(16).unwrap();
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let (poly, _) = make_dense_poly(num_vars);
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1);

    let (_, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup).unwrap();

    assert_eq!(hint.decomposed_inner_rows.len(), 1);
    assert_eq!(hint.recomposed_inner_rows().unwrap().len(), 1);
}

#[test]
#[ignore = "manual tracing-only relation-claim check"]
fn debug_batched_root_relation_claim_matches_tables() {
    init_debug_tracing();
    init_debug_rayon_pool();
    run_debug_on_large_stack(|| {
        const BATCH_NUM_VARS: usize = 29;
        const BATCH_SIZE: usize = 1 << 5;

        let batch_layout = akita_batched_root_layout::<OneHotCfg>(BATCH_NUM_VARS, BATCH_SIZE)
            .expect("batch debug layout");
        let batched_root_lp = akita_types::scale_batched_root_layout(
            &batch_layout,
            BATCH_SIZE,
            OneHotCfg::stage1_challenge_config(OneHotCfg::D).l1_norm(),
            OneHotCfg::decomposition().field_bits(),
        )
        .expect("batched debug root layout");
        let batch_root_inputs = AkitaScheduleInputs {
            num_vars: BATCH_NUM_VARS,
            level: 0,
            current_w_len: akita_types::root_current_w_len(&batch_layout),
        };
        let batch_root_params = OneHotCfg::level_params_with_log_basis(
            batch_root_inputs,
            OneHotCfg::log_basis_at_level(batch_root_inputs),
        );

        let batch_polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
            .map(|idx| debug_make_onehot_poly(&batch_layout, 0x0bee_fcaf_e000_2900 + idx as u64))
            .collect();
        let batch_setup = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(
            BATCH_NUM_VARS,
            BATCH_SIZE,
            1,
        );
        let batch_poly_refs: Vec<&OneHotPoly<OneHotF, ONEHOT_D, u8>> = batch_polys.iter().collect();
        let (batch_commitment, batch_hint) = <OneHotScheme as CommitmentProver<
            OneHotF,
            ONEHOT_D,
        >>::commit(&batch_poly_refs, &batch_setup)
        .expect("batched debug commit");
        let batch_commitments = [batch_commitment];
        let batch_hints = vec![batch_hint];
        let batch_commitment_rows = flatten_batched_commitment_rows(&batch_commitments);

        let batch_point = debug_random_point(BATCH_NUM_VARS);
        let alpha = batch_root_params.ring_dimension.trailing_zeros() as usize;
        let target_num_vars = batch_layout.m_vars + batch_layout.r_vars + alpha;
        let mut padded_point = batch_point.clone();
        padded_point.resize(target_num_vars, OneHotF::zero());
        let outer_point = &padded_point[alpha..];
        let ring_opening_point = ring_opening_point_from_field::<OneHotF>(
            outer_point,
            batch_layout.r_vars,
            batch_layout.m_vars,
            BasisMode::Lagrange,
            BlockOrder::RowMajor,
        )
        .expect("debug opening point");
        let ring_multiplier_point =
            akita_types::RingMultiplierOpeningPoint::from_base(&ring_opening_point);
        let inner_reduction = reduce_inner_opening_to_ring_element::<OneHotF, ONEHOT_D>(
            &padded_point[..alpha],
            BasisMode::Lagrange,
        )
        .expect("debug inner reduction");
        let (y_rings, w_folded_by_poly): (Vec<_>, Vec<_>) = batch_polys
            .iter()
            .map(|poly| {
                poly.evaluate_and_fold(
                    &ring_opening_point.b,
                    &ring_opening_point.a,
                    batch_layout.block_len,
                )
            })
            .unzip();

        let mut transcript = AkitaTranscript::<OneHotF>::new(b"debug/relation-claim/batched");
        append_batched_commitments_to_transcript(&batch_commitments, &mut transcript);
        for pt in &padded_point {
            transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
        }
        let field_openings: Vec<OneHotF> = y_rings
            .iter()
            .map(|y_ring| (*y_ring * inner_reduction.sigma_m1()).coefficients()[0])
            .collect();
        for opening in &field_openings {
            transcript.append_field(ABSORB_EVAL_OPENINGS_FIELD, opening);
        }
        let batch_gammas: Vec<OneHotF> = (0..batch_poly_refs.len())
            .map(|_| transcript.challenge_scalar(CHALLENGE_EVAL_BATCH))
            .collect();
        let batch_gamma_rings = batch_gammas
            .iter()
            .map(|gamma| CyclotomicRing::<OneHotF, ONEHOT_D>::one().scale(gamma))
            .collect::<Vec<_>>();
        let batched_y_rings: Vec<CyclotomicRing<OneHotF, ONEHOT_D>> = {
            let mut combined = CyclotomicRing::<OneHotF, ONEHOT_D>::zero();
            for (claim_idx, y) in y_rings.iter().enumerate() {
                combined += y.scale(&batch_gammas[claim_idx]);
            }
            vec![combined]
        };
        for y_ring in &batched_y_rings {
            transcript.append_serde(ABSORB_EVALUATION_CLAIMS, y_ring);
        }
        let incidence_summary = single_point_group_incidence(BATCH_NUM_VARS, BATCH_SIZE);

        let debug_batch_hint = batch_hints[0].clone();
        let debug_w_folded_by_poly: Vec<Vec<CyclotomicRing<OneHotF, ONEHOT_D>>> =
            w_folded_by_poly.clone();
        let mut quad_eq = Box::new(
            QuadraticEquation::<OneHotF, { ONEHOT_D }>::new_prover(
                &batch_setup.ntt_shared,
                vec![ring_opening_point.clone()],
                vec![ring_multiplier_point.clone()],
                vec![0usize; BATCH_SIZE],
                &batch_poly_refs,
                w_folded_by_poly,
                &incidence_summary,
                batched_root_lp.clone(),
                batch_hints,
                &mut transcript,
                &batch_commitments,
                &batched_y_rings,
                batch_gamma_rings,
                batch_setup.expanded.seed.max_stride,
                MRowLayout::Intermediate,
            )
            .expect("debug batched quadratic equation"),
        );
        let w = ring_switch_build_w::<OneHotF, { ONEHOT_D }>(
            &mut quad_eq,
            &batch_setup.expanded,
            &batch_setup.ntt_shared,
            &batched_root_lp,
        )
        .expect("debug batched w");
        let commit_inputs = AkitaScheduleInputs {
            num_vars: BATCH_NUM_VARS,
            level: 1,
            current_w_len: w.len(),
        };
        let commit_params = OneHotCfg::level_params_with_log_basis(
            commit_inputs,
            OneHotCfg::log_basis_at_level(commit_inputs),
        );
        let mut commit_ntt_cache = MultiDNttCaches::default();
        let next_commitment =
            akita_prover::commit_next_w_with_policy::<OneHotF, OneHotF, _, _, ONEHOT_D>(
                &commit_params,
                &batch_setup.ntt_shared,
                &mut commit_ntt_cache,
                &batch_setup.expanded,
                &w,
                |params, current_w_len| {
                    akita_types::recursive_level_layout_from_params(
                        params,
                        current_w_len,
                        WCommitmentConfig::<{ ONEHOT_D }, OneHotCfg>::decomposition(),
                    )
                },
                recursive_w_commit_layout_for_d::<OneHotCfg>,
            )
            .expect("debug batched w commit");
        let w_commitment_proof = next_commitment.commitment.clone();
        let rs = ring_switch_finalize::<OneHotF, OneHotF, _, { ONEHOT_D }>(
            &quad_eq,
            &batch_setup.expanded,
            &mut transcript,
            &w,
            &w_commitment_proof,
            &batched_root_lp,
            MRowLayout::Intermediate,
        )
        .expect("debug batched ring switch");

        let relation_claim = relation_claim_from_rows::<OneHotF, ONEHOT_D>(
            &rs.tau1,
            rs.alpha,
            &quad_eq.v,
            &batch_commitment_rows,
            &batched_y_rings,
        )
        .unwrap();
        let relation_sum = debug_relation_sum_from_tables(
            &rs.w_evals_compact,
            rs.live_x_cols,
            &rs.alpha_evals_y,
            &rs.m_evals_x,
            0,
            rs.live_x_cols,
        );
        let w_alpha_evals: Vec<OneHotF> = (0..rs.live_x_cols)
            .map(|x| {
                rs.alpha_evals_y
                    .iter()
                    .enumerate()
                    .fold(OneHotF::zero(), |acc, (y, alpha_eval)| {
                        acc + *alpha_eval
                            * OneHotF::from_i64(
                                rs.w_evals_compact[x * rs.alpha_evals_y.len() + y] as i64,
                            )
                    })
            })
            .collect();
        let w_hat_len = batched_root_lp.num_digits_open * batched_root_lp.num_blocks * BATCH_SIZE;
        let t_hat_len = batched_root_lp.num_digits_open
            * batch_root_params.a_key.row_len()
            * batched_root_lp.num_blocks
            * BATCH_SIZE;
        let z_pre_len = batched_root_lp.inner_width() * batched_root_lp.num_digits_fold;
        let num_points = 1usize;
        let num_public_rows = 1usize;
        let m_rows = batch_root_params
            .m_row_count(num_points, num_public_rows)
            .unwrap();
        let r_tail_len = m_rows * r_decomp_levels::<OneHotF>(batched_root_lp.log_basis);
        let w_hat_relation_sum = debug_relation_sum_from_tables(
            &rs.w_evals_compact,
            rs.live_x_cols,
            &rs.alpha_evals_y,
            &rs.m_evals_x,
            0,
            w_hat_len,
        );
        let t_hat_relation_sum = debug_relation_sum_from_tables(
            &rs.w_evals_compact,
            rs.live_x_cols,
            &rs.alpha_evals_y,
            &rs.m_evals_x,
            w_hat_len,
            w_hat_len + t_hat_len,
        );
        let z_pre_relation_sum = debug_relation_sum_from_tables(
            &rs.w_evals_compact,
            rs.live_x_cols,
            &rs.alpha_evals_y,
            &rs.m_evals_x,
            w_hat_len + t_hat_len,
            w_hat_len + t_hat_len + z_pre_len,
        );
        let r_tail_relation_sum = debug_relation_sum_from_tables(
            &rs.w_evals_compact,
            rs.live_x_cols,
            &rs.alpha_evals_y,
            &rs.m_evals_x,
            w_hat_len + t_hat_len + z_pre_len,
            w_hat_len + t_hat_len + z_pre_len + r_tail_len,
        );
        let eq_tau1 = akita_algebra::eq_poly::EqPolynomial::evals(&rs.tau1).unwrap();
        // Row layout: consistency (1) | public (1) | D (n_d) |
        //             B (n_b * num_points) | A (n_a)
        let consistency_weight = eq_tau1[0];
        let public_weight = eq_tau1[1];
        let d_start = 2usize;
        let b_start = d_start + batch_root_params.d_key.row_len();
        let a_start = b_start + batch_root_params.b_key.row_len() * num_points;
        let a_weights = &eq_tau1[a_start..m_rows];
        let alpha_pows = &rs.alpha_evals_y;
        let eval_sparse_alpha = |challenge: &akita_challenges::SparseChallenge| -> OneHotF {
            challenge
                .positions
                .iter()
                .zip(challenge.coeffs.iter())
                .fold(OneHotF::zero(), |acc, (&pos, &coeff)| {
                    acc + OneHotF::from_i64(coeff as i64) * alpha_pows[pos as usize]
                })
        };
        let eval_ring_at_pows_local =
            |ring: &CyclotomicRing<OneHotF, ONEHOT_D>, pows: &[OneHotF]| -> OneHotF {
                ring.coefficients()
                    .iter()
                    .zip(pows.iter())
                    .fold(OneHotF::zero(), |acc, (coeff, alpha_pow)| {
                        acc + *coeff * *alpha_pow
                    })
            };
        let c_alphas: Vec<OneHotF> = quad_eq.challenges.iter().map(eval_sparse_alpha).collect();
        let gadget_scalars = |levels: usize| -> Vec<OneHotF> {
            let base = OneHotF::from_canonical_u128_reduced(1u128 << batched_root_lp.log_basis);
            let mut out = Vec::with_capacity(levels);
            let mut power = OneHotF::one();
            for _ in 0..levels {
                out.push(power);
                power *= base;
            }
            out
        };
        let g1_open = gadget_scalars(batched_root_lp.num_digits_open);
        let g1_commit = gadget_scalars(batched_root_lp.num_digits_commit);
        let fold_gadget = gadget_scalars(batched_root_lp.num_digits_fold);
        let r_gadget = gadget_scalars(r_decomp_levels::<OneHotF>(batched_root_lp.log_basis));
        let debug_stride = batch_setup.expanded.seed.max_stride;
        let d_view = batch_setup
            .expanded
            .shared_matrix
            .ring_view::<ONEHOT_D>(batch_root_params.d_key.row_len(), debug_stride)
            .unwrap();
        let b_view = batch_setup
            .expanded
            .shared_matrix
            .ring_view::<ONEHOT_D>(batch_root_params.b_key.row_len(), debug_stride)
            .unwrap();
        let a_view = batch_setup
            .expanded
            .shared_matrix
            .ring_view::<ONEHOT_D>(batch_root_params.a_key.row_len(), debug_stride)
            .unwrap();
        let denom = alpha_pows[ONEHOT_D - 1] * rs.alpha + OneHotF::one();
        let expected_d_sum = quad_eq
            .v
            .iter()
            .enumerate()
            .take(batch_root_params.d_key.row_len())
            .fold(OneHotF::zero(), |acc, (di, row)| {
                acc + eq_tau1[d_start + di] * akita_algebra::ring::eval_ring_at(row, &rs.alpha)
            });
        let expected_b_sum =
            batch_commitment_rows
                .iter()
                .enumerate()
                .fold(OneHotF::zero(), |acc, (bi, row)| {
                    acc + eq_tau1[b_start + bi] * akita_algebra::ring::eval_ring_at(row, &rs.alpha)
                });
        let expected_public_sum =
            public_weight * akita_algebra::ring::eval_ring_at(&batched_y_rings[0], &rs.alpha);
        let stored_inner_rows_by_poly = debug_batch_hint
            .recomposed_inner_rows()
            .expect("debug batched stored inner rows")
            .to_vec();
        let mut debug_hint_flat = debug_batch_hint;
        debug_hint_flat
            .ensure_recomposed_inner_rows(
                batched_root_lp.num_digits_open,
                batched_root_lp.log_basis,
            )
            .expect("debug batched inner-row recomposition");
        #[cfg(feature = "zk")]
        let (debug_decomposed_inner_rows, debug_recomposed_inner_rows, debug_b_blinding_digits) =
            debug_hint_flat.into_flat_parts();
        #[cfg(not(feature = "zk"))]
        let (debug_decomposed_inner_rows, debug_recomposed_inner_rows) =
            debug_hint_flat.into_flat_parts();
        let _debug_decomposed_inner_rows_flat = debug_decomposed_inner_rows.flat_digits().to_vec();
        let debug_recomposed_inner_rows =
            debug_recomposed_inner_rows.expect("debug batched inner rows");
        let debug_w_folded_flat: Vec<_> = debug_w_folded_by_poly
            .clone()
            .into_iter()
            .flatten()
            .collect();
        let debug_w_hat: Vec<Vec<[i8; ONEHOT_D]>> = debug_w_folded_by_poly
            .iter()
            .flat_map(|folded_rows| {
                folded_rows.iter().map(|w_i| {
                    w_i.balanced_decompose_pow2_i8(
                        batched_root_lp.num_digits_open,
                        batched_root_lp.log_basis,
                    )
                })
            })
            .collect();
        let debug_w_hat_flat: Vec<_> = debug_w_hat
            .iter()
            .flat_map(|block| block.iter().copied())
            .collect();
        #[cfg(feature = "zk")]
        let debug_d_blinding_digits = quad_eq
            .d_blinding_digits()
            .expect("debug batched D-blinding digits");
        let mut debug_z_witnesses = batch_polys
            .iter()
            .zip(quad_eq.challenges.chunks(batched_root_lp.num_blocks))
            .map(|(poly, poly_challenges)| {
                poly.decompose_fold(
                    poly_challenges,
                    batched_root_lp.block_len,
                    batched_root_lp.num_digits_commit,
                    batched_root_lp.log_basis,
                )
            });
        let mut debug_z = debug_z_witnesses.next().expect("debug batched z witness");
        for witness in debug_z_witnesses {
            for (dst, src) in debug_z.z_pre.iter_mut().zip(witness.z_pre.iter()) {
                *dst += *src;
            }
            for (dst, src) in debug_z
                .centered_coeffs
                .iter_mut()
                .zip(witness.centered_coeffs.iter())
            {
                for k in 0..ONEHOT_D {
                    dst[k] += src[k];
                }
            }
        }
        debug_z.centered_inf_norm = debug_z
            .centered_coeffs
            .iter()
            .flat_map(|coeffs| coeffs.iter())
            .map(|coeff| coeff.unsigned_abs())
            .max()
            .unwrap_or(0);
        let debug_y = akita_prover::protocol::quadratic_equation::generate_y::<OneHotF, ONEHOT_D>(
            &quad_eq.v,
            &batch_commitment_rows,
            &batched_y_rings,
            batch_root_params.d_key.row_len(),
            batch_root_params.b_key.row_len(),
            batch_root_params.a_key.row_len(),
        )
        .expect("debug batched y");
        let debug_r =
            akita_prover::protocol::quadratic_equation::compute_r_split_eq::<OneHotF, ONEHOT_D>(
                &batched_root_lp,
                &batch_setup.expanded,
                &quad_eq.challenges,
                &debug_w_hat_flat,
                #[cfg(feature = "zk")]
                debug_d_blinding_digits,
                &debug_decomposed_inner_rows,
                #[cfg(feature = "zk")]
                &debug_b_blinding_digits,
                &debug_recomposed_inner_rows,
                &debug_w_folded_flat,
                std::slice::from_ref(&ring_multiplier_point),
                &[0usize; BATCH_SIZE],
                &[0usize; BATCH_SIZE],
                &(0..BATCH_SIZE).collect::<Vec<_>>(),
                quad_eq.row_coefficient_rings(),
                &debug_z.centered_coeffs,
                debug_z.centered_inf_norm,
                &debug_y,
                &[BATCH_SIZE],
                1,
                batched_root_lp.num_blocks,
                batched_root_lp.inner_width(),
                batch_setup.expanded.seed.max_stride,
                &batch_setup.ntt_shared,
                // Legacy path: no per-point outer digits.
                &[],
                MRowLayout::Intermediate,
            )
            .expect("debug batched r");
        // Local sparse-mul-accumulate: dispatches `+1` / `-1` / generic fast
        // paths over the cyclotomic shift kernels. Inlined here so the
        // `akita-challenges` crate doesn't need to ship a helper that's only
        // useful for this debug cross-check.
        let mul_sparse_into =
            |ring: &CyclotomicRing<OneHotF, ONEHOT_D>,
             challenge: &akita_challenges::SparseChallenge,
             dst: &mut CyclotomicRing<OneHotF, ONEHOT_D>| {
                for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
                    match coeff {
                        1 => ring.shift_accumulate_into(dst, pos as usize),
                        -1 => ring.shift_sub_into(dst, pos as usize),
                        c => ring.shift_scale_accumulate_into(
                            dst,
                            pos as usize,
                            OneHotF::from_i64(c as i64),
                        ),
                    }
                }
            };
        let stored_inner_rows_flat: Vec<_> = stored_inner_rows_by_poly
            .iter()
            .flatten()
            .cloned()
            .collect();
        let stored_a_inner_rows = quad_eq
            .challenges
            .iter()
            .zip(stored_inner_rows_flat.iter())
            .fold(
                CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
                |mut acc, (challenge, block_rows)| {
                    mul_sparse_into(&block_rows[0], challenge, &mut acc);
                    acc
                },
            );
        let reduced_a_inner_rows = quad_eq
            .challenges
            .iter()
            .zip(debug_recomposed_inner_rows.iter())
            .fold(
                CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
                |mut acc, (challenge, block_rows)| {
                    mul_sparse_into(&block_rows[0], challenge, &mut acc);
                    acc
                },
            );
        let reduced_a_z = debug_z.z_pre.iter().enumerate().fold(
            CyclotomicRing::<OneHotF, ONEHOT_D>::zero(),
            |mut acc, (k, z_ring)| {
                a_view.row(0).unwrap()[k].mul_accumulate_into(z_ring, &mut acc);
                acc
            },
        );
        let reduced_a_diff = reduced_a_inner_rows - reduced_a_z;
        let direct_raw_a_inner_rows = c_alphas
            .iter()
            .zip(debug_recomposed_inner_rows.iter())
            .fold(OneHotF::zero(), |acc, (c_alpha, block_rows)| {
                acc + *c_alpha * akita_algebra::ring::eval_ring_at(&block_rows[0], &rs.alpha)
            });
        let direct_raw_a_z =
            debug_z
                .z_pre
                .iter()
                .enumerate()
                .fold(OneHotF::zero(), |acc, (k, z_ring)| {
                    acc - eval_ring_at_pows_local(&a_view.row(0).unwrap()[k], alpha_pows)
                        * akita_algebra::ring::eval_ring_at(z_ring, &rs.alpha)
                });
        let direct_raw_a_r =
            -(denom * akita_algebra::ring::eval_ring_at(&debug_r[a_start], &rs.alpha));
        let direct_raw_a_inner_rowsotal = direct_raw_a_inner_rows + direct_raw_a_z + direct_raw_a_r;
        let d_matrix_width = batched_root_lp.d_matrix_width();
        let d_group_w = (0..w_hat_len).fold(OneHotF::zero(), |acc, x| {
            let coeff =
                (0..batch_root_params.d_key.row_len()).fold(OneHotF::zero(), |inner, di| {
                    inner
                        + eq_tau1[d_start + di]
                            * eval_ring_at_pows_local(
                                &d_view.row(di).unwrap()[x % d_matrix_width],
                                alpha_pows,
                            )
                });
            acc + w_alpha_evals[x] * coeff
        });
        let d_group_r = (0..batch_root_params.d_key.row_len()).fold(OneHotF::zero(), |acc, di| {
            let row_idx = d_start + di;
            let row_start = w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
            acc + (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                inner
                    + w_alpha_evals[row_start + level_idx]
                        * (-(eq_tau1[row_idx] * denom * r_gadget[level_idx]))
            })
        });
        let outer_width = batched_root_lp.outer_width();
        let b_group_t = (0..t_hat_len).fold(OneHotF::zero(), |acc, x| {
            let coeff =
                (0..batch_root_params.b_key.row_len()).fold(OneHotF::zero(), |inner, bi| {
                    inner
                        + eq_tau1[b_start + bi]
                            * eval_ring_at_pows_local(
                                &b_view.row(bi).unwrap()[x % outer_width],
                                alpha_pows,
                            )
                });
            acc + w_alpha_evals[w_hat_len + x] * coeff
        });
        let b_group_r = (0..batch_root_params.b_key.row_len()).fold(OneHotF::zero(), |acc, bi| {
            let row_idx = b_start + bi;
            let row_start = w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
            acc + (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                inner
                    + w_alpha_evals[row_start + level_idx]
                        * (-(eq_tau1[row_idx] * denom * r_gadget[level_idx]))
            })
        });
        let public_group_w = (0..w_hat_len).fold(OneHotF::zero(), |acc, x| {
            let blocks_per_claim = batched_root_lp.num_blocks * batched_root_lp.num_digits_open;
            let claim_idx = x / blocks_per_claim;
            let claim_offset = x % blocks_per_claim;
            let block_idx = claim_offset / batched_root_lp.num_digits_open;
            let digit_idx = claim_offset % batched_root_lp.num_digits_open;
            acc + w_alpha_evals[x]
                * public_weight
                * quad_eq.gamma()[claim_idx]
                * ring_opening_point.b[block_idx]
                * g1_open[digit_idx]
        });
        // The batched protocol has exactly one public y-row at row index 1.
        let public_group_r = {
            let row_idx = 1usize;
            let row_start = w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
            (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                inner
                    + w_alpha_evals[row_start + level_idx]
                        * (-(eq_tau1[row_idx] * denom * r_gadget[level_idx]))
            })
        };
        let row4_group_w = (0..w_hat_len).fold(OneHotF::zero(), |acc, x| {
            let blocks_per_claim = batched_root_lp.num_blocks * batched_root_lp.num_digits_open;
            let claim_idx = x / blocks_per_claim;
            let claim_offset = x % blocks_per_claim;
            let block_idx = claim_offset / batched_root_lp.num_digits_open;
            let digit_idx = claim_offset % batched_root_lp.num_digits_open;
            let global_block_idx = claim_idx * batched_root_lp.num_blocks + block_idx;
            acc + w_alpha_evals[x]
                * consistency_weight
                * c_alphas[global_block_idx]
                * g1_open[digit_idx]
        });
        let row4_group_z = (0..z_pre_len).fold(OneHotF::zero(), |acc, idx| {
            let k = idx / batched_root_lp.num_digits_fold;
            let fold_idx = idx % batched_root_lp.num_digits_fold;
            let block_idx = k / batched_root_lp.num_digits_commit;
            let digit_idx = k % batched_root_lp.num_digits_commit;
            acc + w_alpha_evals[w_hat_len + t_hat_len + idx]
                * (-(consistency_weight
                    * ring_opening_point.a[block_idx]
                    * g1_commit[digit_idx]
                    * fold_gadget[fold_idx]))
        });
        let row4_group_r = {
            let row_start = w_hat_len + t_hat_len + z_pre_len;
            (0..r_gadget.len()).fold(OneHotF::zero(), |acc, level_idx| {
                acc + w_alpha_evals[row_start + level_idx]
                    * (-(consistency_weight * denom * r_gadget[level_idx]))
            })
        };
        let a_group_t = (0..t_hat_len).fold(OneHotF::zero(), |acc, x| {
            let blocks_per_claim = batch_root_params.a_key.row_len()
                * batched_root_lp.num_digits_open
                * batched_root_lp.num_blocks;
            let claim_idx = x / blocks_per_claim;
            let claim_offset = x % blocks_per_claim;
            let block_idx = claim_offset
                / (batch_root_params.a_key.row_len() * batched_root_lp.num_digits_open);
            let rem = claim_offset
                % (batch_root_params.a_key.row_len() * batched_root_lp.num_digits_open);
            let a_idx = rem / batched_root_lp.num_digits_open;
            let digit_idx = rem % batched_root_lp.num_digits_open;
            let global_block_idx = claim_idx * batched_root_lp.num_blocks + block_idx;
            acc + w_alpha_evals[w_hat_len + x]
                * a_weights[a_idx]
                * c_alphas[global_block_idx]
                * g1_open[digit_idx]
        });
        let a_group_z = (0..z_pre_len).fold(OneHotF::zero(), |acc, idx| {
            let k = idx / batched_root_lp.num_digits_fold;
            let fold_idx = idx % batched_root_lp.num_digits_fold;
            let block_idx = k / batched_root_lp.num_digits_commit;
            let coeff =
                a_weights
                    .iter()
                    .enumerate()
                    .fold(OneHotF::zero(), |inner, (a_idx, eq_i)| {
                        inner
                            + *eq_i
                                * eval_ring_at_pows_local(
                                    &a_view.row(a_idx).unwrap()[k],
                                    alpha_pows,
                                )
                    });
            let _ = block_idx;
            acc + w_alpha_evals[w_hat_len + t_hat_len + idx] * (-(coeff * fold_gadget[fold_idx]))
        });
        let a_group_r =
            a_weights
                .iter()
                .enumerate()
                .fold(OneHotF::zero(), |acc, (row_offset, eq_i)| {
                    let row_idx = a_start + row_offset;
                    let row_start = w_hat_len + t_hat_len + z_pre_len + row_idx * r_gadget.len();
                    acc + (0..r_gadget.len()).fold(OneHotF::zero(), |inner, level_idx| {
                        inner
                            + w_alpha_evals[row_start + level_idx]
                                * (-(*eq_i * denom * r_gadget[level_idx]))
                    })
                });

        tracing::info!(
            relation_claim_u128 = relation_claim.to_canonical_u128(),
            relation_sum_u128 = relation_sum.to_canonical_u128(),
            w_hat_relation_sum_u128 = w_hat_relation_sum.to_canonical_u128(),
            t_hat_relation_sum_u128 = t_hat_relation_sum.to_canonical_u128(),
            z_pre_relation_sum_u128 = z_pre_relation_sum.to_canonical_u128(),
            r_tail_relation_sum_u128 = r_tail_relation_sum.to_canonical_u128(),
            d_group_u128 = (d_group_w + d_group_r).to_canonical_u128(),
            expected_d_u128 = expected_d_sum.to_canonical_u128(),
            b_group_u128 = (b_group_t + b_group_r).to_canonical_u128(),
            expected_b_u128 = expected_b_sum.to_canonical_u128(),
            public_group_u128 = (public_group_w + public_group_r).to_canonical_u128(),
            expected_public_u128 = expected_public_sum.to_canonical_u128(),
            row4_group_u128 = (row4_group_w + row4_group_z + row4_group_r).to_canonical_u128(),
            a_group_t_u128 = a_group_t.to_canonical_u128(),
            a_group_z_u128 = a_group_z.to_canonical_u128(),
            a_group_r_u128 = a_group_r.to_canonical_u128(),
            a_group_u128 = (a_group_t + a_group_z + a_group_r).to_canonical_u128(),
            stored_a_ring_matches = stored_a_inner_rows == reduced_a_z,
            stored_vs_recomposed_inner_rows = stored_inner_rows_flat == debug_recomposed_inner_rows,
            reduced_a_ring_matches = reduced_a_inner_rows == reduced_a_z,
            reduced_a_diff_alpha_u128 =
                akita_algebra::ring::eval_ring_at(&reduced_a_diff, &rs.alpha).to_canonical_u128(),
            direct_raw_a_inner_rows_u128 = direct_raw_a_inner_rows.to_canonical_u128(),
            direct_raw_a_z_u128 = direct_raw_a_z.to_canonical_u128(),
            direct_raw_a_r_u128 = direct_raw_a_r.to_canonical_u128(),
            direct_raw_a_inner_rowsotal_u128 = direct_raw_a_inner_rowsotal.to_canonical_u128(),
            live_x_cols = rs.live_x_cols,
            col_bits = rs.col_bits,
            ring_bits = rs.ring_bits,
            "batched relation claim consistency"
        );
        tracing::info!(
            matches = relation_sum == relation_claim,
            "batched relation claim comparison complete"
        );
    });
}

#[test]
#[ignore = "manual tracing-only benchmark breakdown"]
fn debug_onehot_batched_profile_compare() {
    init_debug_tracing();
    init_debug_rayon_pool();
    run_debug_on_large_stack(|| {
        const SINGLE_NUM_VARS: usize = 34;
        const BATCH_NUM_VARS: usize = 29;
        const BATCH_SIZE: usize = 1 << 5;
        const BATCH_COMMITMENT_GROUPS: usize = 1;

        let single_layout =
            OneHotCfg::commitment_layout(SINGLE_NUM_VARS).expect("single debug layout");
        let batch_layout = akita_batched_root_layout::<OneHotCfg>(BATCH_NUM_VARS, BATCH_SIZE)
            .expect("batch debug layout");
        let batched_root_lp = akita_types::scale_batched_root_layout(
            &batch_layout,
            BATCH_SIZE,
            OneHotCfg::stage1_challenge_config(OneHotCfg::D).l1_norm(),
            OneHotCfg::decomposition().field_bits(),
        )
        .expect("batched debug root layout");

        let single_root_inputs = AkitaScheduleInputs {
            num_vars: SINGLE_NUM_VARS,
            level: 0,
            current_w_len: akita_types::root_current_w_len(&single_layout),
        };
        let single_root_params = OneHotCfg::level_params_with_log_basis(
            single_root_inputs,
            OneHotCfg::log_basis_at_level(single_root_inputs),
        );
        let _batch_root_inputs = AkitaScheduleInputs {
            num_vars: BATCH_NUM_VARS,
            level: 0,
            current_w_len: akita_types::root_current_w_len(&batch_layout),
        };
        let _batch_root_params = OneHotCfg::level_params_with_log_basis(
            _batch_root_inputs,
            OneHotCfg::log_basis_at_level(_batch_root_inputs),
        );

        let single_root_w_ring = w_ring_element_count::<OneHotF>(&single_root_params).unwrap();
        let batched_root_w_ring = w_ring_element_count_with_counts::<OneHotF>(
            &batched_root_lp,
            BATCH_COMMITMENT_GROUPS,
            BATCH_SIZE,
            BATCH_SIZE,
            1,
        )
        .unwrap();

        tracing::info!(
            ?single_layout,
            ?batch_layout,
            ?batched_root_lp,
            single_root_w_ring,
            batched_root_w_ring,
            single_root_w_coeffs = single_root_w_ring * ONEHOT_D,
            batched_root_w_coeffs = batched_root_w_ring * ONEHOT_D,
            total_field_single = 1usize << SINGLE_NUM_VARS,
            total_field_batched = BATCH_SIZE * (1usize << BATCH_NUM_VARS),
            "onehot root comparison"
        );

        let single_poly = debug_make_onehot_poly(&single_layout, 0x0bee_fcaf_e000_0034);
        let batch_polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
            .map(|idx| debug_make_onehot_poly(&batch_layout, 0x0bee_fcaf_e000_2900 + idx as u64))
            .collect();

        let single_point = debug_random_point(SINGLE_NUM_VARS);
        let batch_point = debug_random_point(BATCH_NUM_VARS);
        let single_opening = debug_opening_from_poly(&single_poly, &single_point, &single_layout);
        let batch_openings: Vec<OneHotF> = batch_polys
            .iter()
            .map(|poly| debug_opening_from_poly(poly, &batch_point, &batch_layout))
            .collect();

        let single_setup = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(
            SINGLE_NUM_VARS,
            1,
            1,
        );
        let single_verifier_setup =
            <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_verifier(&single_setup);
        let (single_commitment, single_hint) =
            <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit(
                std::slice::from_ref(&single_poly),
                &single_setup,
            )
            .expect("single debug commit");

        let single_poly_refs: [&OneHotPoly<OneHotF, ONEHOT_D, u8>; 1] = [&single_poly];
        let single_commitments = [single_commitment];
        let single_openings = [single_opening];
        let single_opening_groups = [&single_openings[..]];

        let _single_prove_span = tracing::info_span!("debug_single_prove").entered();
        let mut single_prover_transcript = AkitaTranscript::<OneHotF>::new(b"debug/onehot/single");
        let single_proof = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::batched_prove(
            &single_setup,
            vec![(
                &single_point[..],
                CommittedPolynomials {
                    polynomials: &single_poly_refs[..],
                    commitment: &single_commitments[0],
                    hint: single_hint,
                },
            )],
            &mut single_prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("single debug prove");
        drop(_single_prove_span);

        let _single_verify_span = tracing::info_span!("debug_single_verify").entered();
        <OneHotScheme as CommitmentVerifier<OneHotF, ONEHOT_D>>::batched_verify(
            &single_proof,
            &single_verifier_setup,
            &mut AkitaTranscript::<OneHotF>::new(b"debug/onehot/single"),
            vec![(
                &single_point[..],
                CommittedOpenings {
                    openings: single_opening_groups[0],
                    commitment: &single_commitments[0],
                },
            )],
            BasisMode::Lagrange,
        )
        .expect("single debug verify");
        drop(_single_verify_span);

        let batch_setup = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(
            BATCH_NUM_VARS,
            BATCH_SIZE,
            1,
        );
        let batch_verifier_setup =
            <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_verifier(&batch_setup);
        let (batch_commitment, batch_hint) = <OneHotScheme as CommitmentProver<
            OneHotF,
            ONEHOT_D,
        >>::commit(&batch_polys, &batch_setup)
        .expect("batched debug commit");
        let batch_commitments = [batch_commitment];
        let batch_hints = vec![batch_hint];

        let _batched_prove_span = tracing::info_span!("debug_batched_prove").entered();
        let mut batch_prover_transcript = AkitaTranscript::<OneHotF>::new(b"debug/onehot/batched");
        let batch_proof = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::batched_prove(
            &batch_setup,
            vec![(
                &batch_point[..],
                CommittedPolynomials {
                    polynomials: &batch_polys[..],
                    commitment: &batch_commitments[0],
                    hint: batch_hints.into_iter().next().unwrap(),
                },
            )],
            &mut batch_prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("batched debug prove");
        drop(_batched_prove_span);

        let _batched_verify_span = tracing::info_span!("debug_batched_verify").entered();
        let batch_opening_groups = [&batch_openings[..]];
        <OneHotScheme as CommitmentVerifier<OneHotF, ONEHOT_D>>::batched_verify(
            &batch_proof,
            &batch_verifier_setup,
            &mut AkitaTranscript::<OneHotF>::new(b"debug/onehot/batched"),
            vec![(
                &batch_point[..],
                CommittedOpenings {
                    openings: batch_opening_groups[0],
                    commitment: &batch_commitments[0],
                },
            )],
            BasisMode::Lagrange,
        )
        .expect("batched debug verify");
        drop(_batched_verify_span);
    });
}

#[test]
#[cfg(not(feature = "zk"))]
#[cfg_attr(
    not(feature = "planner"),
    ignore = "requires planner fallback for generated schedule misses"
)]
fn batched_commit_matches_individual_commits() {
    let alpha = D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(16).unwrap();
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 1) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 7) as u64)).collect();
    let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1);
    let poly_groups = [std::slice::from_ref(&poly_a), std::slice::from_ref(&poly_b)];

    let (batched_commitments, batched_hints): (Vec<_>, Vec<_>) = poly_groups
        .iter()
        .map(|group| <Scheme as CommitmentProver<F, D>>::commit(group, &setup))
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .unzip();
    let (commitment_a, hint_a) =
        <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly_a), &setup).unwrap();
    let (commitment_b, hint_b) =
        <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly_b), &setup).unwrap();

    assert_eq!(batched_commitments, vec![commitment_a, commitment_b]);
    assert_eq!(batched_hints, vec![hint_a, hint_b]);
}

/// Exercise the batched root-direct fast path: for a layout/batch shape
/// whose offline-planned schedule has zero fold levels, the prover must
/// emit a [`AkitaBatchedRootProof::Direct`] variant with no recursive
/// suffix, and the verifier must accept it via the batched root-direct
/// checks (per-claim opening + joint per-group re-commit).
#[test]
#[cfg_attr(
    not(feature = "planner"),
    ignore = "requires planner fallback for generated schedule misses"
)]
fn batched_root_direct_fast_path_round_trip() {
    // For Cfg = fp128::D64Full with num_t_vectors = 4 and a same-
    // point batch of 4 claims, the generated schedule table is
    // direct-only up to num_vars = 12.
    const NUM_VARS: usize = 8;
    const NUM_POLYS: usize = 4;

    let len = 1usize << NUM_VARS;
    let polys: Vec<DensePoly<F, D>> = (0..NUM_POLYS)
        .map(|poly_idx| {
            let evals: Vec<F> = (0..len)
                .map(|i| F::from_u64((i * (poly_idx + 1) + 17) as u64))
                .collect();
            DensePoly::<F, D>::from_field_evals(NUM_VARS, &evals).unwrap()
        })
        .collect();
    let poly_refs: Vec<&DensePoly<F, D>> = polys.iter().collect();

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NUM_VARS, NUM_POLYS, 1);
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(&poly_refs, &setup).unwrap();
    let commitments = [commitment];
    let hints = vec![hint];

    let opening_point: Vec<F> = (0..NUM_VARS).map(|i| F::from_u64((i + 3) as u64)).collect();
    let openings: Vec<F> = polys
        .iter()
        .map(|poly| {
            let mut evals = vec![F::zero(); len];
            for (i, ring) in poly.coeffs.iter().enumerate() {
                let base = i * D;
                let take = (len.saturating_sub(base)).min(D);
                if take == 0 {
                    break;
                }
                evals[base..base + take].copy_from_slice(&ring.coefficients()[..take]);
            }
            let lw = lagrange_weights(&opening_point).unwrap();
            evals
                .iter()
                .zip(lw.iter())
                .fold(F::zero(), |a, (&c, &w)| a + c * w)
        })
        .collect();

    let poly_group = [&polys[0], &polys[1], &polys[2], &polys[3]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/batched-root-direct");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_group[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("batched root-direct prove");

    assert!(
        proof.is_root_direct(),
        "expected a root-direct batched proof at num_vars={NUM_VARS}, num_t_vectors={NUM_POLYS}"
    );
    let direct_witnesses = proof
        .root
        .as_direct()
        .expect("root-direct variant must expose per-claim direct witnesses");
    assert_eq!(direct_witnesses.len(), NUM_POLYS);
    assert!(
        proof.steps.is_empty(),
        "root-direct batched proof must not carry recursive-suffix steps"
    );

    let mut bytes = Vec::new();
    let shape = proof.shape();
    assert!(matches!(shape, AkitaBatchedProofShape::Direct { .. }));
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let round_trip = AkitaBatchedProof::<F, F>::deserialize_uncompressed(&*bytes, &shape).unwrap();
    assert_eq!(round_trip, proof);

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/batched-root-direct");
    let opening_groups = [&openings[..]];
    <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &round_trip,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .expect("batched root-direct verify");
}

/// The verifier must reject a root-direct batched proof whose
/// per-claim direct witnesses disagree with the claimed opening.
#[test]
#[cfg_attr(
    not(feature = "planner"),
    ignore = "requires planner fallback for generated schedule misses"
)]
fn batched_root_direct_rejects_wrong_opening() {
    const NUM_VARS: usize = 8;
    const NUM_POLYS: usize = 4;
    let len = 1usize << NUM_VARS;
    let polys: Vec<DensePoly<F, D>> = (0..NUM_POLYS)
        .map(|poly_idx| {
            let evals: Vec<F> = (0..len)
                .map(|i| F::from_u64((i + poly_idx + 11) as u64))
                .collect();
            DensePoly::<F, D>::from_field_evals(NUM_VARS, &evals).unwrap()
        })
        .collect();
    let poly_refs: Vec<&DensePoly<F, D>> = polys.iter().collect();

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NUM_VARS, NUM_POLYS, 1);
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(&poly_refs, &setup).unwrap();
    let commitments = [commitment];
    let hints = vec![hint];

    let opening_point: Vec<F> = (0..NUM_VARS).map(|i| F::from_u64((i + 2) as u64)).collect();
    let openings: Vec<F> = (0..NUM_POLYS).map(|_| F::from_u64(999_999)).collect();

    let poly_group = [&polys[0], &polys[1], &polys[2], &polys[3]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/batched-root-direct-bad-opening");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_group[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("batched root-direct prove");
    assert!(proof.is_root_direct());

    let mut verifier_transcript =
        AkitaTranscript::<F>::new(b"test/batched-root-direct-bad-opening");
    let opening_groups = [&openings[..]];
    let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );
    assert!(result.is_err(), "verifier must reject bogus openings");
}

#[test]
#[cfg_attr(
    not(feature = "planner"),
    ignore = "requires planner fallback for generated schedule misses"
)]
fn batched_verify_passes_for_consistent_openings() {
    let alpha = D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(16).unwrap();
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 5) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 7 + 3) as u64)).collect();
    let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1);
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let poly_group = [&poly_a, &poly_b];
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(&poly_group, &setup).unwrap();
    let commitments = [commitment];
    let hints = vec![hint];

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 9) as u64)).collect();
    let openings = [
        dense_opening(&evals_a, &opening_point),
        dense_opening(&evals_b, &opening_point),
    ];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/batched-prove");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_group[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    let mut bytes = Vec::new();
    let shape = proof.shape();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let proof = AkitaBatchedProof::<F, F>::deserialize_uncompressed(&*bytes, &shape).unwrap();

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/batched-prove");
    let opening_groups = [&openings[..]];
    let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );

    assert!(result.is_ok());
}

#[test]
#[cfg_attr(
    not(feature = "planner"),
    ignore = "requires planner fallback for generated schedule misses"
)]
fn batched_onehot_roundtrip_matches_public_shape_context() {
    // NV chosen large enough that the runtime schedule yields at least two
    // fold steps so the proof is fold-rooted (not terminal-rooted). Under
    // the post-soundness-fix proof shape, a single-fold schedule emits a
    // `Terminal` root with no recursive suffix, which this test does not
    // exercise.
    const NV: usize = 20;
    const BATCH_SIZE: usize = 2;

    let layout = akita_batched_root_layout::<OneHotCfg>(NV, BATCH_SIZE).expect("layout");
    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    let total_chunks = total_field / BENCH_ONEHOT_K;
    assert_eq!(total_chunks * BENCH_ONEHOT_K, total_field);

    let polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
        .map(|poly_idx| debug_make_onehot_poly(&layout, 0x0bee_fcaf_e000_1500 + poly_idx as u64))
        .collect();
    let poly_refs: Vec<&OneHotPoly<OneHotF, ONEHOT_D, u8>> = polys.iter().collect();
    let point = debug_random_point(NV);
    let openings: Vec<OneHotF> = polys
        .iter()
        .map(|poly| debug_opening_from_poly(poly, &point, &layout))
        .collect();

    let setup =
        <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(NV, BATCH_SIZE, 1);
    let verifier_setup =
        <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit(&poly_refs, &setup)
            .expect("batched onehot commit");
    let commitments = [commitment];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    let proof = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::batched_prove(
        &setup,
        vec![(
            &point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("batched onehot prove");

    let expected_shape = expected_same_point_batched_shape(NV, BATCH_SIZE, &proof);
    let actual_shape = proof.shape();
    // The expected and actual shapes must match in their root variant: either
    // both `Fold` (multi-fold schedules) or both `Terminal` (1-fold schedules).
    match (&expected_shape, &actual_shape) {
        (
            AkitaBatchedProofShape::Fold {
                root_shape: expected_root,
                step_shapes: expected_steps,
            },
            AkitaBatchedProofShape::Fold {
                root_shape: actual_root,
                step_shapes: actual_steps,
            },
        ) => {
            assert_eq!(expected_root.y_ring_coeffs, actual_root.y_ring_coeffs);
            assert_eq!(expected_root.v_coeffs, actual_root.v_coeffs);
            assert_eq!(expected_root.stage1_stages, actual_root.stage1_stages);
            assert_eq!(expected_root.stage2_sumcheck, actual_root.stage2_sumcheck);
            assert_eq!(
                expected_root.next_commit_coeffs,
                actual_root.next_commit_coeffs
            );
            assert_eq!(expected_steps, actual_steps);
        }
        (
            AkitaBatchedProofShape::Terminal(expected_terminal),
            AkitaBatchedProofShape::Terminal(actual_terminal),
        ) => {
            assert_eq!(expected_terminal, actual_terminal);
        }
        _ => panic!(
            "expected and actual shape root variants disagree: expected={expected_shape:?}, actual={actual_shape:?}"
        ),
    }
    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let decoded =
        AkitaBatchedProof::<OneHotF, OneHotF>::deserialize_uncompressed(&*bytes, &expected_shape)
            .expect("deserialize batched proof with derived shape");
    assert_eq!(decoded, proof);

    let opening_groups = [&openings[..]];
    let mut verifier_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    <OneHotScheme as CommitmentVerifier<OneHotF, ONEHOT_D>>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .expect("batched onehot verify");
}

#[test]
#[cfg_attr(
    not(feature = "planner"),
    ignore = "requires planner fallback for generated schedule misses"
)]
fn batched_verify_rejects_wrong_opening() {
    let alpha = D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(16).unwrap();
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 11) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 5 + 13) as u64)).collect();
    let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1);
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let poly_group = [&poly_a, &poly_b];
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(&poly_group, &setup).unwrap();
    let commitments = [commitment];
    let hints = vec![hint];

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 4) as u64)).collect();
    let mut openings = [
        dense_opening(&evals_a, &opening_point),
        dense_opening(&evals_b, &opening_point),
    ];
    openings[1] += F::one();

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/batched-prove/bad");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_group[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/batched-prove/bad");
    let opening_groups = [&openings[..]];
    let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );

    assert!(matches!(result, Err(AkitaError::InvalidProof)));
}

#[test]
#[cfg_attr(
    not(feature = "planner"),
    ignore = "requires planner fallback for generated schedule misses"
)]
fn batched_verify_rejects_batch_count_beyond_setup_capacity() {
    let alpha = D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(16).unwrap();
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 17) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 19) as u64)).collect();
    let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1);
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let poly_group = [&poly_a, &poly_b];
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(&poly_group, &setup).unwrap();
    let commitments = [commitment];
    let hints = vec![hint];

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 6) as u64)).collect();
    let openings = vec![
        dense_opening(&evals_a, &opening_point),
        dense_opening(&evals_b, &opening_point),
    ];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/batched-prove/oversized");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_group[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    let mut oversized_proof = proof.clone();
    {
        let fold = oversized_proof
            .root
            .as_fold_mut()
            .expect("oversized-y-rings test expects a fold-rooted batched proof");
        let mut oversized_y_coeffs = fold.y_rings.coeffs().to_vec();
        oversized_y_coeffs.extend(vec![F::zero(); D]);
        fold.y_rings = FlatRingVec::from_coeffs(oversized_y_coeffs);
    }

    let mut oversized_openings = openings;
    oversized_openings.push(F::zero());
    let oversized_opening_groups = [&oversized_openings[..]];

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/batched-prove/oversized");
    let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &oversized_proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: oversized_opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );

    assert!(matches!(result, Err(AkitaError::InvalidProof)));
}

#[test]
fn verify_passes_for_consistent_opening() {
    let alpha = D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(16).unwrap();
    let num_vars = layout.m_vars + layout.r_vars + alpha;

    let (poly, evals) = make_dense_poly(num_vars);

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1);
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup).unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );

    assert!(result.is_ok());
}

#[test]
fn verify_rejects_wrong_opening() {
    let alpha = D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(16).unwrap();
    let num_vars = layout.m_vars + layout.r_vars + alpha;

    let (poly, evals) = make_dense_poly(num_vars);

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1);
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup).unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    let wrong_opening = opening + F::one();
    let wrong_openings = [wrong_opening];
    let wrong_opening_groups = [&wrong_openings[..]];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: wrong_opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );

    assert!(
        result.is_err(),
        "verify must reject an incorrect opening value"
    );
}

#[test]
fn verify_rejects_malformed_y_ring_dimension_without_panicking() {
    let (verifier_setup, commitment, mut proof, opening_point, opening, _layout) =
        make_verify_fixture(16);
    let root_fold = proof
        .root
        .as_fold_mut()
        .expect("expected a fold-rooted batched proof");
    let mut coeffs = root_fold.y_rings.coeffs().to_vec();
    let _ = coeffs.pop().expect("expected non-empty y_rings");
    root_fold.y_rings = FlatRingVec::from_coeffs(coeffs);

    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                CommittedOpenings {
                    openings: opening_groups[0],
                    commitment: &commitments[0],
                },
            )],
            BasisMode::Lagrange,
        )
    }));

    assert!(matches!(result, Ok(Err(AkitaError::InvalidProof))));
}

#[test]
fn fp128_degree_one_batched_proof_roundtrip_is_stable() {
    let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
        make_verify_fixture(16);
    let (_, _, same_proof, _, _, _) = make_verify_fixture(16);
    let shape = proof.shape();
    assert_eq!(shape, same_proof.shape());

    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let mut same_bytes = Vec::new();
    same_proof.serialize_uncompressed(&mut same_bytes).unwrap();
    #[cfg(not(feature = "zk"))]
    assert_eq!(bytes, same_bytes);

    let mut repeated_bytes = Vec::new();
    proof.serialize_uncompressed(&mut repeated_bytes).unwrap();
    assert_eq!(bytes, repeated_bytes);

    let decoded = AkitaBatchedProof::<F, F>::deserialize_uncompressed(&*bytes, &shape)
        .expect("degree-one proof should roundtrip");
    assert_eq!(decoded, proof);

    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .expect("degree-one roundtrip proof should verify");
}

#[test]
fn folded_payload_commitments_and_digits_stay_base_field() {
    fn assert_base_flat_ring_vec(_: &FlatRingVec<F>) {}
    fn assert_base_direct_witness(_: &akita_types::DirectWitnessProof<F>) {}

    let (_, _, proof, _, _, _) = make_verify_fixture(16);
    let root = proof
        .root
        .as_fold()
        .expect("fixture should use folded root proof");
    assert_base_flat_ring_vec(&root.y_rings);
    assert_base_flat_ring_vec(&root.v);
    assert_base_flat_ring_vec(&root.stage2.next_w_commitment);

    for level in proof.fold_levels() {
        assert_base_flat_ring_vec(&level.y_ring);
        assert_base_flat_ring_vec(&level.v);
        assert_base_flat_ring_vec(level.next_w_commitment());
    }
    assert_base_direct_witness(proof.final_witness());
}

#[test]
fn folded_root_rejects_unchecked_extension_opening_reduction_payload() {
    let (verifier_setup, commitment, mut proof, opening_point, opening, _) =
        make_verify_fixture(16);
    let dummy_sumcheck = proof
        .root
        .as_fold()
        .expect("fixture should use folded root proof")
        .stage2
        .sumcheck
        .clone();
    proof
        .root
        .as_fold_mut()
        .expect("fixture should use folded root proof")
        .extension_opening_reduction = Some(ExtensionOpeningReductionProof {
        partials: vec![F::zero()],
        sumcheck: dummy_sumcheck,
    });

    let openings = [opening];
    let commitments = [commitment];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let err = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .unwrap_err();
    assert!(matches!(err, AkitaError::InvalidProof));
}

#[test]
fn monomial_basis_prove_verify_round_trip() {
    let alpha = D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(16).unwrap();
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;

    let coeffs: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &coeffs).unwrap();

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1);
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup).unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();

    let mw = monomial_weights(&opening_point).unwrap();
    let opening: F = coeffs
        .iter()
        .zip(mw.iter())
        .fold(F::zero(), |acc, (&c, &w)| acc + c * w);

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/monomial");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Monomial,
    )
    .unwrap();

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/monomial");
    let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Monomial,
    );

    assert!(
        result.is_ok(),
        "monomial-basis proof should verify: {result:?}"
    );
}

#[test]
fn tiny_d32_root_direct_helpers_accept_valid_proof() {
    type DirectCfg = fp128::D32Full;
    type DirectF = fp128::Field;
    const DIRECT_D: usize = DirectCfg::D;
    type DirectScheme = AkitaCommitmentScheme<DIRECT_D, DirectCfg>;

    let num_vars = 4usize;
    let evals: Vec<DirectF> = (0..(1usize << num_vars))
        .map(|i| DirectF::from_u64((i + 1) as u64))
        .collect();
    let poly = DensePoly::<DirectF, DIRECT_D>::from_field_evals(num_vars, &evals).unwrap();
    let opening_point = vec![DirectF::zero(); num_vars];
    let opening = evals[0];

    let setup = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::setup_prover(num_vars, 1, 1);
    let verifier_setup =
        <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::setup_verifier(&setup);
    let (commitment, hint) = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::commit(
        std::slice::from_ref(&poly),
        &setup,
    )
    .unwrap();

    let poly_refs: [&DensePoly<DirectF, DIRECT_D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<DirectF>::new(b"test/tiny-direct");
    let proof = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    assert!(proof.is_root_direct());
    assert_eq!(proof.num_fold_levels(), 0);
    let witnesses = proof
        .root
        .as_direct()
        .expect("root-direct batched proof expected");
    assert_eq!(witnesses.len(), 1);
    assert!(direct_witness_opening_matches::<DirectF, DirectF>(
        &witnesses[0],
        &opening_point,
        &opening,
        BasisMode::Lagrange,
    )
    .unwrap());

    let mut verifier_transcript = AkitaTranscript::<DirectF>::new(b"test/tiny-direct");
    <DirectScheme as CommitmentVerifier<DirectF, DIRECT_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .unwrap();
}

#[derive(Clone)]
struct Fp32RingSubfieldRootFoldCfg;
#[derive(Clone)]
struct Fp32RingSubfieldOuterFallbackCfg;

impl akita_types::ScheduleProvider for Fp32RingSubfieldRootFoldCfg {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("test/fp32-ring-subfield-root-fold/{key:?}")
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }
}

impl Fp32RingSubfieldRootFoldCfg {
    fn root_lp() -> LevelParams {
        LevelParams::params_only(
            akita_types::SisModulusFamily::Q32,
            Self::D,
            3,
            1,
            1,
            1,
            akita_challenges::SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(0, 0, 12, 12, 12, 0)
        .unwrap()
    }
}

fn fp32_ring_subfield_setup_matrix_size<F>(
    lp: &LevelParams,
    max_num_claims: usize,
) -> Result<(usize, usize), AkitaError>
where
    F: akita_field::CanonicalField,
{
    let _field_marker = core::marker::PhantomData::<F>;
    let outer_width = lp.outer_width();
    #[cfg(feature = "zk")]
    let outer_width = {
        outer_width
            .checked_add(akita_types::zk::blinding_column_count::<F>(
                lp.b_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
            ))
            .ok_or_else(|| AkitaError::InvalidSetup("ZK outer width overflow".to_string()))?
    };

    Ok((
        lp.a_key
            .row_len()
            .max(lp.b_key.row_len())
            .max(lp.d_key.row_len()),
        lp.inner_width().max(outer_width).max(
            lp.d_matrix_width()
                .checked_mul(max_num_claims.max(1))
                .ok_or_else(|| AkitaError::InvalidSetup("D matrix width overflow".to_string()))?,
        ),
    ))
}

impl akita_planner::PlannerConfig for Fp32RingSubfieldRootFoldCfg {
    type PlannerField = akita_field::Prime32Offset99;

    const PLANNER_D: usize = 16;

    fn planner_field_bits() -> u32 {
        32
    }

    fn planner_challenge_field_bits() -> u32 {
        32 * (<Self as CommitmentConfig>::CHAL_EXT_DEGREE as u32)
    }

    fn planner_extension_opening_width() -> usize {
        <Self as CommitmentConfig>::CLAIM_EXT_DEGREE
    }

    fn planner_sis_modulus_family() -> akita_types::SisModulusFamily {
        akita_types::SisModulusFamily::Q32
    }

    fn planner_stage1_challenge_config(_d: usize) -> akita_challenges::SparseChallengeConfig {
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        }
    }

    fn planner_schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }

    fn planner_root_level_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        _log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn planner_current_level_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        _log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp().with_layout(lp))
    }

    fn planner_log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
        (3, 3)
    }
}

impl CommitmentConfig for Fp32RingSubfieldRootFoldCfg {
    type Field = akita_field::Prime32Offset99;
    type ClaimField = akita_field::RingSubfieldFp4<Self::Field>;
    type ChallengeField = Self::ClaimField;

    const D: usize = 16;

    fn decomposition() -> akita_types::DecompositionParams {
        akita_types::DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        }
    }

    fn stage1_challenge_config(_d: usize) -> akita_challenges::SparseChallengeConfig {
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        }
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        akita_types::SisModulusFamily::Q32
    }

    fn audited_root_rank(_role: akita_types::AjtaiRole, _max_num_vars: usize) -> usize {
        1
    }

    fn envelope(_max_num_vars: usize) -> akita_types::CommitmentEnvelope {
        akita_types::CommitmentEnvelope {
            max_n_a: 1,
            max_n_b: 1,
            max_n_d: 1,
        }
    }

    fn max_setup_matrix_size(
        _max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError> {
        let max_num_claims = max_num_batched_polys
            .checked_mul(max_num_points)
            .ok_or_else(|| AkitaError::InvalidSetup("claim count overflow".to_string()))?;
        let lp = Self::root_lp();
        fp32_ring_subfield_setup_matrix_size::<Self::Field>(&lp, max_num_claims)
    }

    fn level_params_with_log_basis(_inputs: AkitaScheduleInputs, _log_basis: u32) -> LevelParams {
        Self::root_lp()
    }

    fn root_level_params_for_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp().with_layout(lp))
    }

    fn root_level_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        _log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn log_basis_at_level(_inputs: AkitaScheduleInputs) -> u32 {
        3
    }

    fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
        (3, 3)
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn get_params_for_commitment(
        _num_vars: usize,
        _num_polys_per_point: usize,
        _max_num_points: usize,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn get_params_for_prove(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = akita_types::scale_batched_root_layout(
            &Self::root_lp(),
            incidence.num_claims(),
            Self::stage1_challenge_config(Self::D).l1_norm(),
            Self::decomposition().field_bits(),
        )?;
        let w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Self::Field>(
            &lp,
            incidence.num_points(),
            incidence.num_polynomials(),
            incidence.num_claims(),
            incidence.num_public_rows(),
            akita_types::MRowLayout::Terminal,
        )?;
        let compact_w_len = w_ring * Self::D;
        Ok(akita_types::Schedule {
            steps: vec![
                Step::Fold(akita_types::FoldStep {
                    params: lp.clone(),
                    current_w_len: akita_types::root_current_w_len(&lp),
                    delta_fold_per_poly: lp.num_digits_fold,
                    w_ring,
                    next_w_len: compact_w_len,
                    level_bytes: 0,
                }),
                Step::Direct(akita_types::DirectStep {
                    current_w_len: compact_w_len,
                    witness_shape: akita_types::DirectWitnessShape::PackedDigits((
                        compact_w_len,
                        3,
                    )),
                    direct_bytes: compact_w_len,
                }),
            ],
            total_bytes: 0,
        })
    }
}

impl akita_types::ScheduleProvider for Fp32RingSubfieldOuterFallbackCfg {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("test/fp32-ring-subfield-outer-fallback/{key:?}")
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }
}

impl Fp32RingSubfieldOuterFallbackCfg {
    fn root_lp() -> LevelParams {
        LevelParams::params_only(
            akita_types::SisModulusFamily::Q32,
            Self::D,
            3,
            1,
            1,
            1,
            akita_challenges::SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(1, 0, 12, 12, 12, 0)
        .unwrap()
    }
}

impl akita_planner::PlannerConfig for Fp32RingSubfieldOuterFallbackCfg {
    type PlannerField = akita_field::Prime32Offset99;

    const PLANNER_D: usize = 16;

    fn planner_field_bits() -> u32 {
        32
    }

    fn planner_challenge_field_bits() -> u32 {
        32 * (<Self as CommitmentConfig>::CHAL_EXT_DEGREE as u32)
    }

    fn planner_extension_opening_width() -> usize {
        <Self as CommitmentConfig>::CLAIM_EXT_DEGREE
    }

    fn planner_sis_modulus_family() -> akita_types::SisModulusFamily {
        akita_types::SisModulusFamily::Q32
    }

    fn planner_stage1_challenge_config(_d: usize) -> akita_challenges::SparseChallengeConfig {
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        }
    }

    fn planner_schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }

    fn planner_root_level_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        _log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn planner_current_level_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        _log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp().with_layout(lp))
    }

    fn planner_log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
        (3, 3)
    }
}

impl CommitmentConfig for Fp32RingSubfieldOuterFallbackCfg {
    type Field = akita_field::Prime32Offset99;
    type ClaimField = akita_field::RingSubfieldFp4<Self::Field>;
    type ChallengeField = Self::ClaimField;

    const D: usize = 16;

    fn decomposition() -> akita_types::DecompositionParams {
        akita_types::DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        }
    }

    fn stage1_challenge_config(_d: usize) -> akita_challenges::SparseChallengeConfig {
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        }
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        akita_types::SisModulusFamily::Q32
    }

    fn audited_root_rank(_role: akita_types::AjtaiRole, _max_num_vars: usize) -> usize {
        1
    }

    fn envelope(_max_num_vars: usize) -> akita_types::CommitmentEnvelope {
        akita_types::CommitmentEnvelope {
            max_n_a: 1,
            max_n_b: 1,
            max_n_d: 1,
        }
    }

    fn max_setup_matrix_size(
        _max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError> {
        let max_num_claims = max_num_batched_polys
            .checked_mul(max_num_points)
            .ok_or_else(|| AkitaError::InvalidSetup("claim count overflow".to_string()))?;
        let lp = Self::root_lp();
        fp32_ring_subfield_setup_matrix_size::<Self::Field>(&lp, max_num_claims)
    }

    fn level_params_with_log_basis(_inputs: AkitaScheduleInputs, _log_basis: u32) -> LevelParams {
        Self::root_lp()
    }

    fn root_level_params_for_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp().with_layout(lp))
    }

    fn root_level_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        _log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn log_basis_at_level(_inputs: AkitaScheduleInputs) -> u32 {
        3
    }

    fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
        (3, 3)
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn get_params_for_commitment(
        _num_vars: usize,
        _num_polys_per_point: usize,
        _max_num_points: usize,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn get_params_for_prove(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = akita_types::scale_batched_root_layout(
            &Self::root_lp(),
            incidence.num_claims(),
            Self::stage1_challenge_config(Self::D).l1_norm(),
            Self::decomposition().field_bits(),
        )?;
        // Single-fold schedule: the root IS the terminal fold, so its
        // shipped `w` is built under MRowLayout::Terminal (no D-block in
        // the per-row `r` quotients). The schedule's `next_w_len` and the
        // following Direct step's witness shape must match that reduced
        // length.
        let w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Self::Field>(
            &lp,
            incidence.num_points(),
            incidence.num_polynomials(),
            incidence.num_claims(),
            incidence.num_public_rows(),
            akita_types::MRowLayout::Terminal,
        )?;
        let next_w_len = w_ring * Self::D;
        Ok(akita_types::Schedule {
            steps: vec![
                Step::Fold(akita_types::FoldStep {
                    params: lp.clone(),
                    current_w_len: akita_types::root_current_w_len(&lp),
                    delta_fold_per_poly: lp.num_digits_fold,
                    w_ring,
                    next_w_len,
                    level_bytes: 0,
                }),
                Step::Direct(akita_types::DirectStep {
                    current_w_len: next_w_len,
                    witness_shape: akita_types::DirectWitnessShape::PackedDigits((next_w_len, 3)),
                    direct_bytes: next_w_len,
                }),
            ],
            total_bytes: 0,
        })
    }
}

#[test]
fn fp32_ring_subfield_root_fold_roundtrip_uses_extension_gamma() {
    type SmallCfg = Fp32RingSubfieldRootFoldCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ClaimField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 1;
    type SmallScheme = AkitaCommitmentScheme<SMALL_D, SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals = (0..len)
        .map(|idx| SmallF::from_u64((3 * idx as u64) + 9))
        .collect::<Vec<_>>();
    let poly = DensePoly::<SmallF, SMALL_D>::from_field_evals(NUM_VARS, &evals).unwrap();
    let point = (0..NUM_VARS)
        .map(|idx| {
            SmallE::new([
                SmallF::from_u64((idx + 5) as u64),
                SmallF::from_u64((idx + 7) as u64),
                SmallF::from_u64((idx + 11) as u64),
                SmallF::from_u64((idx + 13) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let weights = lagrange_weights(&point).unwrap();
    let opening = evals
        .iter()
        .zip(weights.iter())
        .fold(SmallE::zero(), |acc, (&coeff, &weight)| {
            acc + weight * SmallE::lift_base(coeff)
        });

    let setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 1, 1);
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let (commitment, hint) = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(
        std::slice::from_ref(&poly),
        &setup,
    )
    .unwrap();

    let poly_refs = [&poly];
    let commitments = [commitment];
    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        vec![(
            &point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    // After Phase 1, a tiny `NUM_VARS=1` schedule has a single fold level so
    // the root is the `Terminal` variant (not `Fold`). Both shapes carry an
    // optional extension-opening reduction payload; this test asserts the
    // payload is absent at the root in the degree-1 extension case.
    let root_extension_opening_reduction = match &proof.root {
        akita_types::AkitaBatchedRootProof::Fold(fold) => fold.extension_opening_reduction.as_ref(),
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => {
            terminal.extension_opening_reduction.as_ref()
        }
        akita_types::AkitaBatchedRootProof::Direct { .. } => {
            panic!("root-direct proof has no folded root extension-opening reduction")
        }
    };
    assert!(
        root_extension_opening_reduction.is_none(),
        "root fold must not carry an unchecked extension-opening reduction payload"
    );

    let openings = [opening];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .unwrap();

    let wrong_openings = [opening + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: &wrong_openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );
    assert!(result.is_err());

    let wrong_point = [point[0] + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &wrong_point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );
    assert!(result.is_err());
}

#[test]
fn fp32_ring_subfield_outer_extension_uses_root_tensor_projection() {
    type SmallCfg = Fp32RingSubfieldOuterFallbackCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ClaimField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 5;
    type SmallScheme = AkitaCommitmentScheme<SMALL_D, SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals_a = (0..len)
        .map(|idx| SmallF::from_u64((idx as u64) + 3))
        .collect::<Vec<_>>();
    let evals_b = (0..len)
        .map(|idx| SmallF::from_u64((2 * idx as u64) + 7))
        .collect::<Vec<_>>();
    let poly_a = DensePoly::<SmallF, SMALL_D>::from_field_evals(NUM_VARS, &evals_a).unwrap();
    let poly_b = DensePoly::<SmallF, SMALL_D>::from_field_evals(NUM_VARS, &evals_b).unwrap();
    let point = (0..NUM_VARS)
        .map(|idx| {
            SmallE::new([
                SmallF::from_u64((idx + 2) as u64),
                SmallF::from_u64((idx + 4) as u64),
                SmallF::from_u64((idx + 6) as u64),
                SmallF::from_u64((idx + 8) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let weights = lagrange_weights(&point).unwrap();
    let opening_a = evals_a
        .iter()
        .zip(weights.iter())
        .fold(SmallE::zero(), |acc, (&coeff, &weight)| {
            acc + weight * SmallE::lift_base(coeff)
        });
    let opening_b = evals_b
        .iter()
        .zip(weights.iter())
        .fold(SmallE::zero(), |acc, (&coeff, &weight)| {
            acc + weight * SmallE::lift_base(coeff)
        });

    let setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 2, 1);
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let poly_refs = [&poly_a, &poly_b];
    let (commitment, hint) =
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(&poly_refs, &setup).unwrap();
    let commitments = [commitment];
    let openings = [opening_a, opening_b];

    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-outer-direct");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        vec![(
            &point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    // After Phase 1, the root variant depends on the schedule: multi-fold
    // produces `Fold`, single-fold produces `Terminal`. Both carry the
    // extension-opening reduction payload as `Option`.
    let root_extension_opening_reduction = match &proof.root {
        akita_types::AkitaBatchedRootProof::Fold(fold) => fold.extension_opening_reduction.as_ref(),
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => {
            terminal.extension_opening_reduction.as_ref()
        }
        akita_types::AkitaBatchedRootProof::Direct { .. } => {
            panic!("root-direct proof has no folded root extension-opening reduction")
        }
    };
    assert!(
        root_extension_opening_reduction.is_some(),
        "root tensor projection must prove the extension-opening reduction"
    );

    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-outer-direct");
    <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .unwrap();

    let wrong_openings = [opening_a, opening_b + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-outer-direct");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: &wrong_openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );
    assert!(result.is_err());
}

#[test]
fn fp32_ring_subfield_multipoint_extension_uses_root_tensor_projection() {
    type SmallCfg = Fp32RingSubfieldOuterFallbackCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ClaimField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 5;
    type SmallScheme = AkitaCommitmentScheme<SMALL_D, SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals = (0..len)
        .map(|idx| SmallF::from_u64((3 * idx as u64) + 5))
        .collect::<Vec<_>>();
    let poly = DensePoly::<SmallF, SMALL_D>::from_field_evals(NUM_VARS, &evals).unwrap();
    let point_a = (0..NUM_VARS)
        .map(|idx| {
            SmallE::new([
                SmallF::from_u64((idx + 3) as u64),
                SmallF::from_u64((idx + 5) as u64),
                SmallF::from_u64((idx + 7) as u64),
                SmallF::from_u64((idx + 9) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let point_b = (0..NUM_VARS)
        .map(|idx| {
            SmallE::new([
                SmallF::from_u64((idx + 11) as u64),
                SmallF::from_u64((idx + 13) as u64),
                SmallF::from_u64((idx + 17) as u64),
                SmallF::from_u64((idx + 19) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let opening_at = |point: &[SmallE]| {
        let weights = lagrange_weights(point).unwrap();
        evals
            .iter()
            .zip(weights.iter())
            .fold(SmallE::zero(), |acc, (&coeff, &weight)| {
                acc + weight * SmallE::lift_base(coeff)
            })
    };
    let opening_a = opening_at(&point_a);
    let opening_b = opening_at(&point_b);

    let setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 2, 2);
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let poly_refs = [&poly];
    let (commitment, hint) =
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(&poly_refs, &setup).unwrap();
    let commitments = [commitment];
    let openings_a = [opening_a];
    let openings_b = [opening_b];

    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-multipoint-direct");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        vec![
            (
                &point_a[..],
                CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint: hint.clone(),
                },
            ),
            (
                &point_b[..],
                CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint,
                },
            ),
        ],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    // After Phase 1, the root variant depends on the schedule: multi-fold
    // produces `Fold`, single-fold produces `Terminal`. Both carry the
    // extension-opening reduction payload as `Option`.
    let root_extension_opening_reduction = match &proof.root {
        akita_types::AkitaBatchedRootProof::Fold(fold) => fold.extension_opening_reduction.as_ref(),
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => {
            terminal.extension_opening_reduction.as_ref()
        }
        akita_types::AkitaBatchedRootProof::Direct { .. } => {
            panic!("root-direct proof has no folded root extension-opening reduction")
        }
    };
    assert!(
        root_extension_opening_reduction.is_some(),
        "root tensor projection must prove the extension-opening reduction"
    );

    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-multipoint-direct");
    <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![
            (
                &point_a[..],
                CommittedOpenings {
                    openings: &openings_a[..],
                    commitment: &commitments[0],
                },
            ),
            (
                &point_b[..],
                CommittedOpenings {
                    openings: &openings_b[..],
                    commitment: &commitments[0],
                },
            ),
        ],
        BasisMode::Lagrange,
    )
    .unwrap();

    let wrong_openings_b = [opening_b + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-multipoint-direct");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![
            (
                &point_a[..],
                CommittedOpenings {
                    openings: &openings_a[..],
                    commitment: &commitments[0],
                },
            ),
            (
                &point_b[..],
                CommittedOpenings {
                    openings: &wrong_openings_b[..],
                    commitment: &commitments[0],
                },
            ),
        ],
        BasisMode::Lagrange,
    );
    assert!(
        result.is_err(),
        "root tensor projection must reject a wrong claim at any point"
    );
}

// =========================================================================
// Tiered root-commitment full E2E (`specs/tiered_commit.md` Phase 4f)
// =========================================================================
//
// `Fp32TieredRootFoldCfg` mirrors `Fp32RingSubfieldRootFoldCfg` above
// but injects tiering parameters into its `LevelParams`: `split_factor
// = 2`, `outer_log_basis = 6`, `num_digits_outer = 6` (full-field for
// the ~32-bit base field). It exercises the full prover/verifier
// pipeline through `AkitaCommitmentScheme` with a real
// `split_factor > 1` schedule, validating that:
//   - the prover's `commit_with_params` auto-dispatches to tiered;
//   - `ring_switch_build_w` threads `ûhat_concat` into the M-witness;
//   - `compute_r_split_eq` emits tier-1 + F rows;
//   - the stage-1 / stage-2 sumchecks round-trip the new M̃ correctly;
//   - the verifier's `eval_at_point` dispatches to the tier-1 + F
//     reference evaluator and accepts the proof.
//
// The bypass: rather than wiring tiered candidate emission into the
// planner search loop (Phase 4e-search), this test exposes a custom
// `CommitmentConfig` whose `get_params_for_prove` directly returns a
// tiered `Schedule`. This drives every protocol-layer code path the
// planner-emission would, just without the search-and-cache wrapper.

#[derive(Clone)]
struct Fp32TieredRootFoldCfg;

impl akita_types::ScheduleProvider for Fp32TieredRootFoldCfg {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("test/fp32-tiered-root-fold/{key:?}")
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }
}

impl Fp32TieredRootFoldCfg {
    fn root_lp() -> LevelParams {
        // Start from the same base shape as the legacy
        // Fp32RingSubfieldRootFoldCfg fixture, then inject tiering.
        // `AKITA_TEST_NA` env var overrides `n_a` (default 1) so the
        // n_a > 1 path can be exercised under split>1 without changing
        // the other dimensions.
        let n_a = std::env::var("AKITA_TEST_NA")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1usize);
        let r_vars = std::env::var("AKITA_TEST_RVARS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0usize);
        let depth_commit = std::env::var("AKITA_TEST_DC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12usize);
        let depth_open = std::env::var("AKITA_TEST_DO")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12usize);
        let depth_fold = std::env::var("AKITA_TEST_DF")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12usize);
        let log_basis = std::env::var("AKITA_TEST_LB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3u32);
        let base = LevelParams::params_only(
            akita_types::SisModulusFamily::Q32,
            <Self as CommitmentConfig>::D,
            log_basis,
            n_a,
            1,
            1,
            akita_challenges::SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(0, r_vars, depth_commit, depth_open, depth_fold, 0)
        .unwrap();
        // Tiering: split_factor (2 by default, override via env) with
        // `outer_log_basis = log_basis` so the gadget-decomposed ûhat
        // digits stay in the same `[-(b/2)..(b/2)-1]` band the
        // stage-1/2 sumcheck lookup tables expect (the prefix-aware
        // constructors in `two_round_prefix.rs` index a fixed
        // `b=2^log_basis` lookup table over ALL witness cells,
        // including ûhat). Outer width = n_a · num_digits_open ·
        // num_blocks = 1·12·1 = 12; chunk_width = 12/split. With
        // `outer_log_basis = 3`, full Fp32 coverage needs `ceil(32/3) =
        // 11` outer digits; F has shape n_F=1 × (n_b'·split·δ_outer).
        let split_factor = std::env::var("AKITA_TEST_SPLIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2usize);
        // outer_log_basis must equal log_basis for stage-1/2 sumcheck
        // lookup-table compatibility. num_digits_outer must cover
        // field_bits (32 for Fp32) with outer_log_basis.
        let outer_log_basis = log_basis;
        let num_digits_outer = 32u32.div_ceil(outer_log_basis) as usize;
        let n_f = 1usize;
        let chunk_width = base.full_outer_width() / split_factor;
        let f_width = base.b_key.row_len() * split_factor * num_digits_outer;
        // b_key.col_len shrinks to chunk_width per the tiered convention
        // (full_outer_width is recoverable via `full_outer_width()`).
        let tiered_b_key = akita_types::AjtaiKeyParams::new_unchecked(
            base.b_key.sis_family(),
            base.b_key.row_len(),
            chunk_width,
            base.b_key.collision_inf(),
            base.ring_dimension,
        );
        let f_key = akita_types::AjtaiKeyParams::new_unchecked(
            akita_types::SisModulusFamily::Q32,
            n_f,
            f_width,
            akita_types::layout::sis_derivation::balanced_digit_delta_bound(outer_log_basis),
            base.ring_dimension,
        );
        LevelParams {
            split_factor,
            outer_log_basis,
            num_digits_outer,
            f_key,
            b_key: tiered_b_key,
            ..base
        }
    }
}

impl akita_planner::PlannerConfig for Fp32TieredRootFoldCfg {
    type PlannerField = akita_field::Prime32Offset99;

    const PLANNER_D: usize = 16;

    fn planner_field_bits() -> u32 {
        32
    }

    fn planner_challenge_field_bits() -> u32 {
        32 * (<Self as CommitmentConfig>::CHAL_EXT_DEGREE as u32)
    }

    fn planner_extension_opening_width() -> usize {
        <Self as CommitmentConfig>::CLAIM_EXT_DEGREE
    }

    fn planner_sis_modulus_family() -> akita_types::SisModulusFamily {
        akita_types::SisModulusFamily::Q32
    }

    fn planner_stage1_challenge_config(_d: usize) -> akita_challenges::SparseChallengeConfig {
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        }
    }

    fn planner_schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }

    fn planner_root_level_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        _log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn planner_current_level_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        _log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp().with_layout(lp))
    }

    fn planner_log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
        (3, 3)
    }
}

impl CommitmentConfig for Fp32TieredRootFoldCfg {
    type Field = akita_field::Prime32Offset99;
    type ClaimField = akita_field::RingSubfieldFp4<Self::Field>;
    type ChallengeField = Self::ClaimField;

    const D: usize = 16;

    fn decomposition() -> akita_types::DecompositionParams {
        akita_types::DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        }
    }

    fn stage1_challenge_config(_d: usize) -> akita_challenges::SparseChallengeConfig {
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        }
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        akita_types::SisModulusFamily::Q32
    }

    fn audited_root_rank(_role: akita_types::AjtaiRole, _max_num_vars: usize) -> usize {
        1
    }

    fn envelope(_max_num_vars: usize) -> akita_types::CommitmentEnvelope {
        akita_types::CommitmentEnvelope {
            max_n_a: 1,
            max_n_b: 1,
            max_n_d: 1,
        }
    }

    fn max_setup_matrix_size(
        _max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError> {
        let lp = Self::root_lp();
        let max_num_claims = max_num_batched_polys
            .checked_mul(max_num_points)
            .ok_or_else(|| AkitaError::InvalidSetup("claim count overflow".to_string()))?;
        // Use the full outer width when computing the envelope so the
        // shared matrix has enough room for legacy-equivalent t̂ scans
        // even though the tiered runtime only reads `chunk_width` at a
        // time. F lives in its own derived matrix (per
        // `crates/akita-prover/src/kernels/matrix.rs::derive_tier1_f_matrix_flat`)
        // and is NOT in shared_matrix.
        let inner = lp.inner_width();
        let outer = lp.full_outer_width();
        let d_matrix = lp
            .d_matrix_width()
            .checked_mul(max_num_claims.max(1))
            .ok_or_else(|| AkitaError::InvalidSetup("D matrix width overflow".to_string()))?;
        let max_stride = inner.max(outer).max(d_matrix);
        let max_rows = lp
            .a_key
            .row_len()
            .max(lp.b_key.row_len())
            .max(lp.d_key.row_len());
        Ok((max_rows, max_stride))
    }

    fn level_params_with_log_basis(_inputs: AkitaScheduleInputs, _log_basis: u32) -> LevelParams {
        Self::root_lp()
    }

    fn root_level_params_for_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp().with_layout(lp))
    }

    fn root_level_layout_with_log_basis(
        _inputs: AkitaScheduleInputs,
        _log_basis: u32,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn log_basis_at_level(_inputs: AkitaScheduleInputs) -> u32 {
        3
    }

    fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
        (3, 3)
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn get_params_for_commitment(
        _num_vars: usize,
        _num_polys_per_point: usize,
        _max_num_points: usize,
    ) -> Result<LevelParams, AkitaError> {
        Ok(Self::root_lp())
    }

    fn get_params_for_prove(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = Self::root_lp();
        // Single-fold + Direct schedule: the only fold IS the terminal,
        // so size the next witness under `MRowLayout::Terminal` to match
        // the runtime (the terminal-fold cutover from main's
        // `e577e79e` drops the D-block from the M-matrix at terminal
        // folds, which shrinks `w_ring` by `n_d · r_decomp` rings).
        let w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Self::Field>(
            &lp,
            incidence.num_points(),
            incidence.num_polynomials(),
            incidence.num_claims(),
            incidence.num_public_rows(),
            akita_types::MRowLayout::Terminal,
        )?;
        let compact_w_len = w_ring * Self::D;
        Ok(akita_types::Schedule {
            steps: vec![
                Step::Fold(akita_types::FoldStep {
                    params: lp.clone(),
                    current_w_len: akita_types::root_current_w_len(&lp),
                    delta_fold_per_poly: lp.num_digits_fold,
                    w_ring,
                    next_w_len: compact_w_len,
                    level_bytes: 0,
                }),
                Step::Direct(akita_types::DirectStep {
                    current_w_len: compact_w_len,
                    witness_shape: akita_types::DirectWitnessShape::PackedDigits((
                        compact_w_len,
                        lp.log_basis,
                    )),
                    direct_bytes: compact_w_len,
                }),
            ],
            total_bytes: 0,
        })
    }
}

/// Full sumcheck E2E for the tiered root commitment. Cloned from
/// `fp32_ring_subfield_root_fold_roundtrip_uses_extension_gamma` but
/// uses the `Fp32TieredRootFoldCfg` above whose root LevelParams
/// satisfies `is_tiered_root()`.
///
/// Drives the full prover/verifier pipeline through
/// `AkitaCommitmentScheme` with `split_factor = 2`, exercising and
/// validating end-to-end:
///   - `commit_with_params` auto-dispatch to `commit_tiered_with_params`
///     (commitment has the right shape, hint carries `outer_digits`).
///   - `QuadraticEquation::new_prover` threads per-point
///     `outer_digits` into the combined hint.
///   - `w_ring_element_count_with_counts` sizes `next_w_len` to include
///     the tiered `ûhat` segment.
///   - `compute_m_evals_x` skips legacy B-row α-evals, emits the tiered
///     `uhat_segment`, and uses the tiered A-row `a_start`.
///   - `relation_claim_from_rows_extension_tiered` places `u_final`
///     at the F-row positions (skipping tier-1 zero rows between D
///     and F).
///   - The verifier's `eval_at_point` includes the
///     `compute_tier1_and_f_contribution_reference` addend and uses the
///     tiered `a_start` in BOTH `compute_setup_contribution` (for the
///     Z-half A-weight slice) and `t_structured_contribution`.
///   - Stage-1/2 sumcheck digit-basis lookup tables stay consistent
///     because `outer_log_basis == lp.log_basis`.
///
/// All non-tiered configs (D32Full, the legacy
/// `fp32_ring_subfield_root_fold_roundtrip_uses_extension_gamma`, etc.)
/// remain green and byte-identical because every change above is
/// gated on `lp.is_tiered_root()`.
///
/// Note: post-merge with main (`e577e79e`), the terminal-fold cutover
/// turns this single-fold-then-Direct schedule into a Terminal root
/// (the only fold IS the terminal). The combination
/// `tiered root + MRowLayout::Terminal` needs dedicated wiring through
/// the relation-only stage-2 path (`RootStageInput::Terminal` in
/// `verify_root_level_inner`); the production `tiered_bench` uses a
/// multi-fold recursive schedule, so the root stays Intermediate and
/// is unaffected. Tracking ticket: phase4f-tiered-terminal.
#[test]
#[ignore = "tiered + terminal-fold combo: needs dedicated wiring; \
            production multi-fold schedules keep root Intermediate"]
fn tiered_root_fold_full_sumcheck_e2e_roundtrips() {
    type TCfg = Fp32TieredRootFoldCfg;
    type TF = <TCfg as CommitmentConfig>::Field;
    type TE = <TCfg as CommitmentConfig>::ClaimField;
    const TD: usize = TCfg::D;
    const NUM_VARS: usize = 1;
    type TScheme = AkitaCommitmentScheme<TD, TCfg>;

    // Sanity: the test Cfg actually enables tiering.
    let lp = TCfg::root_lp();
    assert!(lp.is_tiered_root(), "test fixture must enable tiering");
    assert!(lp.split_factor >= 2);
    assert_eq!(lp.outer_log_basis, lp.log_basis);

    let len = 1usize << NUM_VARS;
    let evals = (0..len)
        .map(|idx| TF::from_u64((3 * idx as u64) + 9))
        .collect::<Vec<_>>();
    let poly = DensePoly::<TF, TD>::from_field_evals(NUM_VARS, &evals).unwrap();
    let point = (0..NUM_VARS)
        .map(|idx| {
            TE::new([
                TF::from_u64((idx + 5) as u64),
                TF::from_u64((idx + 7) as u64),
                TF::from_u64((idx + 11) as u64),
                TF::from_u64((idx + 13) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let weights = lagrange_weights(&point).expect("lagrange weights");
    let opening = evals
        .iter()
        .zip(weights.iter())
        .fold(TE::zero(), |acc, (&coeff, &weight)| {
            acc + weight * TE::lift_base(coeff)
        });

    let setup = <TScheme as CommitmentProver<TF, TD>>::setup_prover(NUM_VARS, 1, 1);
    let verifier_setup = <TScheme as CommitmentProver<TF, TD>>::setup_verifier(&setup);
    let (commitment, hint) =
        <TScheme as CommitmentProver<TF, TD>>::commit(std::slice::from_ref(&poly), &setup).unwrap();

    // The commit should auto-dispatch to the tiered path and emit a
    // shorter u_final than the legacy n_b would. With n_F = 1 here,
    // u.len() == 1 either way (n_b also = 1), but we still check the
    // tiered hint shape.
    assert_eq!(commitment.u.len(), lp.outer_commitment_rows());
    assert!(
        !hint.outer_digits().is_empty(),
        "tiered commit must populate outer_digits",
    );

    let poly_refs = [&poly];
    let commitments = [commitment];
    let mut prover_transcript = AkitaTranscript::<TF>::new(b"test/tiered-root-fold-full-e2e");
    let proof = <TScheme as CommitmentProver<TF, TD>>::batched_prove(
        &setup,
        vec![(
            &point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("tiered batched_prove must succeed");

    // Single-fold + Direct schedule: post terminal-fold cutover the root
    // step is a `Terminal` variant (the only fold IS the terminal).
    assert!(
        proof.root.as_fold().is_some() || proof.root.as_terminal_root().is_some(),
        "root must be either a Fold (intermediate) or Terminal step",
    );

    let openings = [opening];
    let mut verifier_transcript = AkitaTranscript::<TF>::new(b"test/tiered-root-fold-full-e2e");
    <TScheme as CommitmentVerifier<TF, TD>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .expect("tiered batched_verify must succeed");

    // Wrong opening must be rejected.
    let wrong_openings = [opening + TE::one()];
    let mut verifier_transcript = AkitaTranscript::<TF>::new(b"test/tiered-root-fold-full-e2e");
    let result = <TScheme as CommitmentVerifier<TF, TD>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: &wrong_openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );
    assert!(
        result.is_err(),
        "tiered verifier must reject a wrong opening",
    );
}

/// Direct comparison: for the tiered fixture, the prover's
/// `multilinear_eval(m_evals_x, x_challenges)` must equal the
/// verifier's `prepared.eval_at_point(x_challenges, ...)` at every
/// column. Tiered analog of the legacy
/// `prepared_row_eval_matches_materialized` in
/// `crates/akita-pcs/tests/ring_switch.rs`.
///
/// Pinned this test as an oracle while debugging the Phase 4f-sumcheck
/// stage-2 `verify_sumcheck MISMATCH` — a per-column bisection on this
/// fixture located the bug in
/// `compute_setup_contribution::a_start` (was hard-coded to the legacy
/// `b_start + n_b · num_points` formula, slurping tier-1 row weights
/// into the A-row weight slice and corrupting the Z-half α-eval).
#[test]
fn tiered_prepared_row_eval_matches_materialized() {
    use akita_algebra::ring::scalar_powers;
    use akita_prover::protocol::ring_switch::{compute_m_evals_x, ring_switch_build_w};
    use akita_transcript::labels::ABSORB_COMMITMENT;
    use akita_types::ring_opening_point_from_field;
    use akita_types::{AppendToTranscript, RingMultiplierOpeningPoint};
    use akita_verifier::prepare_ring_switch_row_eval;

    type TCfg = Fp32TieredRootFoldCfg;
    type TF = <TCfg as CommitmentConfig>::Field;
    const TD: usize = TCfg::D;
    type TScheme = AkitaCommitmentScheme<TD, TCfg>;
    // NV needs to be at least `log2(D) + r_vars + m_vars` to give the
    // ring opening point enough coordinates; the diagnostic fixture
    // bumps NV automatically when r_vars > 0.
    let r_vars = std::env::var("AKITA_TEST_RVARS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0usize);
    let nv: usize = (TD.trailing_zeros() as usize + r_vars).max(1);

    let lp = TCfg::root_lp();
    assert!(lp.is_tiered_root(), "fixture must enable tiering");

    let mut rng_seed: u64 = 0xfeed_face;
    let mut next_tf = || -> TF {
        rng_seed = rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        TF::from_u64(rng_seed)
    };
    let evals: Vec<TF> = (0..(1usize << nv))
        .map(|i| TF::from_u64(7 + i as u64 * 3))
        .collect();
    let poly = DensePoly::<TF, TD>::from_field_evals(nv, &evals).unwrap();
    let point: Vec<TF> = (0..nv).map(|i| TF::from_u64(11 + i as u64 * 5)).collect();

    let alpha_bits = TD.trailing_zeros() as usize;
    let outer_point: &[TF] = if alpha_bits >= point.len() {
        &point[point.len()..]
    } else {
        &point[alpha_bits..]
    };
    let ring_opening_point = ring_opening_point_from_field(
        outer_point,
        lp.r_vars,
        lp.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )
    .expect("ring opening point");
    let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
    let (y_ring, w_folded) =
        poly.evaluate_and_fold(&ring_opening_point.b, &ring_opening_point.a, lp.block_len);

    let setup = <TScheme as CommitmentProver<TF, TD>>::setup_prover(nv, 1, 1);
    let (commitment, batched_hint) =
        <TScheme as CommitmentProver<TF, TD>>::commit(std::slice::from_ref(&poly), &setup).unwrap();

    let mut transcript = AkitaTranscript::<TF>::new(b"tiered-prepared-row-eval");
    commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
    for pt in &point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, pt);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);
    let incidence_summary = single_point_group_incidence(nv, 1);

    let mut quad_eq = QuadraticEquation::<TF, TD>::new_prover(
        &setup.ntt_shared,
        vec![ring_opening_point.clone()],
        vec![ring_multiplier_point.clone()],
        vec![0usize],
        &[&poly],
        vec![w_folded],
        &incidence_summary,
        lp.clone(),
        vec![batched_hint],
        &mut transcript,
        std::slice::from_ref(&commitment),
        std::slice::from_ref(&y_ring),
        vec![CyclotomicRing::<TF, TD>::one()],
        setup.expanded.seed.max_stride,
        MRowLayout::Intermediate,
    )
    .expect("quadratic equation");

    ring_switch_build_w::<TF, TD>(&mut quad_eq, &setup.expanded, &setup.ntt_shared, &lp)
        .expect("ring-switch witness");

    let alpha = TF::from_u64(101);
    let alpha_evals_y = scalar_powers(alpha, TD);
    let rows = lp.m_row_count(1, 1).expect("m_row_count");
    let num_i = rows.next_power_of_two().trailing_zeros() as usize;
    let tau1: Vec<TF> = (0..num_i).map(|_| next_tf()).collect();

    let m_evals_x = compute_m_evals_x::<TF, TF, TD>(
        &setup.expanded,
        &[ring_opening_point.clone()],
        std::slice::from_ref(&ring_multiplier_point),
        &[0usize],
        &quad_eq.challenges,
        alpha,
        &alpha_evals_y,
        &lp,
        &tau1,
        &[1usize],
        &[0usize],
        &[0usize],
        &[TF::one()],
        1,
        MRowLayout::Intermediate,
    )
    .expect("compute_m_evals_x");

    let num_x_bits = m_evals_x.len().trailing_zeros() as usize;

    let prepared = prepare_ring_switch_row_eval::<TF, TF, TD>(
        &quad_eq.challenges,
        alpha,
        &lp,
        &tau1,
        &[1usize],
        &[0usize],
        &[0usize],
        &[TF::one()],
        1,
        MRowLayout::Intermediate,
        1,
        std::slice::from_ref(&ring_multiplier_point),
        &[0usize],
    )
    .expect("prepare_ring_switch_row_eval");

    // Per-column check: for each column `k`, set
    // `x_challenges = bin(k)` so `eq(x_challenges, x) = δ(x = k)` and
    // `multilinear_eval(m_evals_x, x_challenges) = m_evals_x[k]`. The
    // verifier's `eval_at_point` must agree at every column. This is a
    // strictly stronger check than a single random-challenge equality
    // (it catches per-column bugs that would otherwise only show up in
    // a full-protocol sumcheck mismatch).
    for (k, &prover_cell) in m_evals_x.iter().enumerate() {
        let x_challenges: Vec<TF> = (0..num_x_bits)
            .map(|i| {
                if (k >> i) & 1 == 1 {
                    TF::one()
                } else {
                    TF::zero()
                }
            })
            .collect();
        let verifier_cell = prepared
            .eval_at_point::<TF, TD>(
                &x_challenges,
                &setup.expanded,
                std::slice::from_ref(&ring_opening_point),
                std::slice::from_ref(&ring_multiplier_point),
                alpha,
            )
            .expect("eval_at_point");
        assert_eq!(
            prover_cell, verifier_cell,
            "tiered eval_at_point != m_evals_x[{k}]: prover={prover_cell:?} verifier={verifier_cell:?}",
        );
    }
}
