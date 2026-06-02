#![cfg(not(feature = "zk"))]

use super::*;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::LiftBase;
use akita_planner::test_utils::akita_batched_root_layout;
use akita_prover::{AkitaPolyOps, CommitmentProver, CommittedPolynomials, DensePoly, OneHotPoly};
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::stage1_tree_stage_shapes;
use akita_types::w_ring_element_count;
use akita_types::BlockOrder;
use akita_types::ClaimIncidenceSummary;
use akita_types::ExtensionOpeningReductionProof;
use akita_types::{
    lagrange_weights, monomial_weights, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field,
};
use akita_types::{
    AkitaBatchedProofShape, AkitaProofStepShape, FlatRingVec, LevelProofShape,
    TerminalLevelProofShape,
};
use akita_types::{AkitaScheduleInputs, AkitaScheduleLookupKey, Step};
use akita_verifier::direct_witness_opening_matches;
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
type Cfg = akita_planner::test_utils::PlannerCfg<fp128::D64Full>;
type F = fp128::Field;
const D: usize = Cfg::D;
type Scheme = AkitaCommitmentScheme<D, Cfg>;

type OneHotF = fp128::Field;
type OneHotCfg = akita_planner::test_utils::PlannerCfg<fp128::D64OneHot>;
const ONEHOT_D: usize = OneHotCfg::D;
const BENCH_ONEHOT_K: usize = ONEHOT_D;
type OneHotScheme = AkitaCommitmentScheme<ONEHOT_D, OneHotCfg>;
/// Minimum w vector length (in field elements) below which further folding
/// is not beneficial.  When `w.len() <= MIN_W_LEN_FOR_FOLDING`, the prover
/// sends `w` directly instead of recursing.
const MIN_W_LEN_FOR_FOLDING: usize = 4096;

mod batched;
mod fp32_ring_subfield;
mod layout;
mod onehot;
mod single;

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
    let root_lp = akita_derive::root_level_params_for_layout_with_log_basis(
        OneHotCfg::sis_modulus_family(),
        OneHotCfg::D,
        OneHotCfg::decomposition(),
        OneHotCfg::stage1_challenge_config(OneHotCfg::D).unwrap(),
        OneHotCfg::ring_subfield_embedding_norm_bound(),
        root_inputs,
        level_lp,
    )
    .unwrap();
    let root_w_len = root_step.next_w_len;
    let root_rounds = batched_shape_rounds(root_lp.ring_dimension, root_w_len);

    // 1-fold schedule: the root IS the terminal fold. Emit a terminal-rooted
    // shape with no recursive-suffix steps.
    if num_fold_levels == 1 {
        // The terminal fold's `next` parameters live at `schedule.steps[1]`,
        // which is a `Direct` step encoding the final packed-digit basis.
        let terminal_next_params =
            scheduled_next_level_params(&schedule, 1).expect("terminal next params");
        return AkitaBatchedProofShape::Terminal(TerminalLevelProofShape {
            y_rings_coeffs: incidence.num_public_rows() * root_lp.ring_dimension,
            extension_opening_reduction: None,
            stage2_sumcheck: vec![3; root_rounds],
            stage3_sumcheck: None,
            final_witness: akita_types::DirectWitnessShape::PackedDigits((
                root_w_len,
                terminal_next_params.log_basis,
            )),
        });
    }

    let next_level_params = scheduled_next_level_params(&schedule, 1).unwrap();
    let root_shape = LevelProofShape {
        y_ring_coeffs: incidence.num_public_rows() * root_lp.ring_dimension,
        extension_opening_reduction: None,
        v_coeffs: root_lp.d_key.row_len() * root_lp.ring_dimension,
        stage1_stages: stage1_tree_stage_shapes(root_rounds, 1usize << level_lp.log_basis),
        stage2_sumcheck_proof: vec![3; root_rounds],
        stage3_sumcheck: None,
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
        let (level_params, next_level_params) =
            scheduled_fold_execution(&schedule, current_level, inputs, current_log_basis)
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
            stage2_sumcheck_proof: vec![3; rounds],
            stage3_sumcheck: None,
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
    let (terminal_params, terminal_next_params) =
        scheduled_fold_execution(&schedule, current_level, terminal_inputs, current_log_basis)
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
        stage3_sumcheck: None,
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

fn singleton_layout<C: CommitmentConfig>(num_vars: usize) -> LevelParams {
    let incidence = ClaimIncidenceSummary::same_point(num_vars, 1).expect("singleton incidence");
    C::get_params_for_batched_commitment(&incidence).expect("singleton commitment layout")
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
    let layout = singleton_layout::<Cfg>(num_vars);
    let full_num_vars = layout.m_vars + layout.r_vars + alpha;

    let (poly, evals) = make_dense_poly(full_num_vars);
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(full_num_vars, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .unwrap();

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
        &CpuBackend,
        &prepared,
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
        akita_types::SetupContributionMode::Direct,
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
