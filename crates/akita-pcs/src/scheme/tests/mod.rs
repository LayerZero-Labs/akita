#![cfg(not(feature = "zk"))]

use super::*;
use akita_config::proof_optimized::fp128;
use akita_config::test_support::akita_batched_root_layout;
use akita_config::CommitmentConfig;
use akita_field::LiftBase;
use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource, RootPolyShape};
use akita_prover::{CommitmentProver, CommittedPolynomials, DensePoly, OneHotPoly};
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::stage1_tree_stage_shapes;
use akita_types::w_ring_element_count;
use akita_types::BlockOrder;
use akita_types::ExtensionOpeningReductionProof;
use akita_types::OpeningBatch;
use akita_types::Step;
use akita_types::{
    lagrange_weights, monomial_weights, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field,
};
use akita_types::{scheduled_next_level_params, LevelParams};
use akita_types::{
    AkitaBatchedProofShape, AkitaProofStepShape, FlatRingVec, LevelProofShape,
    TerminalLevelProofShape,
};
use akita_verifier::cleartext_witness_opening_matches;
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
type Cfg = fp128::D64Full;
type F = fp128::Field;
const D: usize = Cfg::D;
type Scheme = AkitaCommitmentScheme<D, Cfg>;

type OneHotF = fp128::Field;
type OneHotCfg = fp128::D64OneHot;
const ONEHOT_D: usize = OneHotCfg::D;
// `fp128::D64OneHot` requires K=256 one-hot schedules (must match
// `OneHotCfg::onehot_chunk_size()`); chunks span `K/D = 4` ring elements.
const BENCH_ONEHOT_K: usize = 256;
type OneHotScheme = AkitaCommitmentScheme<ONEHOT_D, OneHotCfg>;
/// Minimum w vector length (in field elements) below which further folding
/// is not beneficial.  When `w.len() <= MIN_W_LEN_FOR_FOLDING`, the prover
/// sends `w` directly instead of recursing.
const MIN_W_LEN_FOR_FOLDING: usize = 4096;

mod batched;
mod fp32_ext4;
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

/// Terminal stage-2 sumcheck round degrees can depend on Fiat-Shamir challenges
/// (e.g. structurally zero leading cubic on the first round). Copy them from
/// the proved shape so deserialization uses the on-wire widths.
fn sync_terminal_stage2_sumcheck_from_proof(
    expected: &mut AkitaBatchedProofShape,
    proof: &AkitaBatchedProof<OneHotF, OneHotF>,
) {
    let actual = proof.shape();
    match (expected, actual) {
        (
            AkitaBatchedProofShape::Fold { step_shapes, .. },
            AkitaBatchedProofShape::Fold {
                step_shapes: actual_steps,
                ..
            },
        ) => {
            let Some(AkitaProofStepShape::Terminal(terminal)) = step_shapes.last_mut() else {
                return;
            };
            let Some(AkitaProofStepShape::Terminal(actual_terminal)) = actual_steps.last() else {
                return;
            };
            terminal.stage2_sumcheck = actual_terminal.stage2_sumcheck.clone();
        }
        (
            AkitaBatchedProofShape::Terminal(terminal),
            AkitaBatchedProofShape::Terminal(actual_terminal),
        ) => {
            terminal.stage2_sumcheck = actual_terminal.stage2_sumcheck.clone();
        }
        _ => {}
    }
}

fn expected_same_point_batched_shape(
    max_num_vars: usize,
    num_claims: usize,
    proof: &AkitaBatchedProof<OneHotF, OneHotF>,
) -> AkitaBatchedProofShape {
    let opening_batch =
        akita_types::OpeningBatch::same_point(max_num_vars, num_claims).expect("opening_batch");
    let schedule =
        OneHotCfg::get_params_for_prove(&opening_batch).expect("batched root runtime plan");
    let Some(Step::Fold(root_step)) = schedule.steps.first() else {
        panic!("batched schedule should start with a fold");
    };
    let num_fold_levels = akita_types::schedule_num_fold_levels(&schedule);
    let root_rounds = batched_shape_rounds(root_step.params.ring_dimension, root_step.next_w_len);

    // 1-fold schedule: the root IS the terminal fold. Emit a terminal-rooted
    // shape with no recursive-suffix steps.
    if num_fold_levels == 1 {
        let mut stage2_sumcheck = vec![3; root_rounds];
        let fold_basis = 1usize << root_step.params.log_basis;
        let ring_bits = root_step.params.ring_dimension.trailing_zeros() as usize;
        if root_rounds >= 2 && ring_bits >= 2 && matches!(fold_basis, 4 | 8) {
            stage2_sumcheck[0] = 2;
        }
        let mut shape = AkitaBatchedProofShape::Terminal(TerminalLevelProofShape {
            extension_opening_reduction: None,
            stage2_sumcheck,
            final_witness: akita_types::schedule_terminal_direct_witness_shape(&schedule)
                .expect("1-fold schedule should end in a direct step")
                .clone(),
        });
        sync_terminal_stage2_sumcheck_from_proof(&mut shape, proof);
        return shape;
    }

    let next_level_params = scheduled_next_level_params(&schedule, 1).unwrap();
    let root_shape = LevelProofShape {
        extension_opening_reduction: None,
        v_coeffs: root_step.params.d_key.row_len() * root_step.params.ring_dimension,
        stage1_stages: stage1_tree_stage_shapes(root_rounds, 1usize << root_step.params.log_basis),
        stage2_sumcheck_proof: vec![3; root_rounds],
        stage3_sumcheck: None,
        next_commit_coeffs: next_level_params.b_key.row_len() * next_level_params.ring_dimension,
    };
    // After Phase 1, the recursive suffix has `num_fold_levels - 1` steps in
    // total: `num_fold_levels - 2` intermediate steps followed by exactly one
    // terminal step. (We've already consumed the root.)
    let num_intermediate_after_root = num_fold_levels.saturating_sub(2);
    let mut step_shapes = Vec::with_capacity(num_fold_levels - 1);
    let mut current_w_len = root_step.next_w_len;
    let mut current_level = 1usize;
    for _ in 0..num_intermediate_after_root {
        let scheduled = schedule
            .get_execution_schedule(current_level)
            .expect("scheduled recursive fold");
        scheduled
            .validate_current_w_len(current_w_len)
            .expect("scheduled recursive fold current witness length");
        let level_params = scheduled.params;
        let next_level_params = scheduled.next_params;
        let next_w_len =
            w_ring_element_count::<OneHotF>(&level_params).unwrap() * level_params.ring_dimension;
        let rounds = batched_shape_rounds(level_params.ring_dimension, next_w_len);
        step_shapes.push(AkitaProofStepShape::Intermediate(LevelProofShape {
            extension_opening_reduction: None,
            v_coeffs: level_params.d_key.row_len() * level_params.ring_dimension,
            stage1_stages: stage1_tree_stage_shapes(rounds, 1usize << level_params.log_basis),
            stage2_sumcheck_proof: vec![3; rounds],
            stage3_sumcheck: None,
            next_commit_coeffs: next_level_params.b_key.row_len()
                * next_level_params.ring_dimension,
        }));
        current_w_len = next_w_len;
        current_level += 1;
    }

    // Terminal fold step (always present in the multi-fold case): its params
    // live at `schedule.steps[current_level]` (still a `Step::Fold`); the
    // immediately following Direct step encodes the terminal witness shape.
    let terminal_scheduled = schedule
        .get_execution_schedule(current_level)
        .expect("scheduled terminal fold");
    terminal_scheduled
        .validate_current_w_len(current_w_len)
        .expect("scheduled terminal fold current witness length");
    let terminal_params = terminal_scheduled.params;
    // The terminal recursive fold ships its `w` in cleartext under
    // MRowLayout::Terminal (D-block omitted from per-row `r` quotients), so
    // the expected packed-digit witness shape uses the terminal-layout ring
    // count instead of the intermediate-layout `w_ring_element_count`.
    let terminal_next_w_len = akita_types::w_ring_element_count_with_counts_for_layout::<OneHotF>(
        &terminal_params,
        1,
        1,
        1,
        1,
        akita_types::MRowLayout::WithoutDBlock,
    )
    .expect("terminal-layout witness count")
        * terminal_params.ring_dimension;
    let terminal_rounds = batched_shape_rounds(terminal_params.ring_dimension, terminal_next_w_len);
    let mut terminal_stage2 = vec![3; terminal_rounds];
    let fold_basis = 1usize << terminal_params.log_basis;
    let ring_bits = terminal_params.ring_dimension.trailing_zeros() as usize;
    if terminal_rounds >= 2 && ring_bits >= 2 && matches!(fold_basis, 4 | 8) {
        terminal_stage2[0] = 2;
    }
    step_shapes.push(AkitaProofStepShape::Terminal(TerminalLevelProofShape {
        extension_opening_reduction: None,
        stage2_sumcheck: terminal_stage2,
        final_witness: akita_types::schedule_terminal_direct_witness_shape(&schedule)
            .expect("terminal direct witness shape")
            .clone(),
    }));

    let mut shape = AkitaBatchedProofShape::Fold {
        root_shape,
        step_shapes,
    };
    sync_terminal_stage2_sumcheck_from_proof(&mut shape, proof);
    shape
}

fn make_dense_poly(num_vars: usize) -> (DensePoly<F, D>, Vec<F>) {
    let len = 1usize << num_vars;
    let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).unwrap();
    (poly, evals)
}

fn singleton_layout<C: CommitmentConfig>(num_vars: usize) -> LevelParams {
    let opening_batch = OpeningBatch::same_point(num_vars, 1).expect("singleton opening batch");
    C::get_params_for_batched_commitment(&opening_batch).expect("singleton commitment layout")
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
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(full_num_vars, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        std::slice::from_ref(&poly),
        &CpuBackend,
        &prepared,
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
        (
            &opening_point[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        ),
        &CpuBackend,
        &prepared,
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
    // `total_ring` ring elements of degree D cover `2^num_vars` field elements,
    // grouped into `2^num_vars / K` one-hot chunks.
    let total_field = total_ring * ONEHOT_D;
    assert_eq!(total_field, 1usize << num_vars);
    let total_chunks = total_field / BENCH_ONEHOT_K;

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..BENCH_ONEHOT_K) as u8))
        .collect();

    OneHotPoly::<OneHotF, ONEHOT_D, u8>::new(BENCH_ONEHOT_K, indices).expect("debug onehot poly")
}

fn opening_from_poly<'a, P>(poly: &'a P, point: &[OneHotF], layout: &LevelParams) -> OneHotF
where
    P: RootOpeningSource<OneHotF, ONEHOT_D> + RootPolyShape<OneHotF, ONEHOT_D>,
    CpuBackend: OpeningFoldKernel<P::OpeningView<'a>, OneHotF, ONEHOT_D>,
{
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
    .expect("opening point shape should match layout");

    let opening = OpeningFoldKernel::<P::OpeningView<'a>, OneHotF, ONEHOT_D>::evaluate_and_fold(
        &CpuBackend,
        None,
        poly.opening_view().expect("opening view"),
        OpeningFoldPlan::Base {
            eval_outer_scalars: &ring_opening_point.b,
            fold_scalars: &ring_opening_point.a,
            block_len: layout.block_len,
        },
    )
    .expect("evaluate_and_fold");
    let folded_ring = opening.eval;
    let packed_inner =
        reduce_inner_opening_to_ring_element::<OneHotF, ONEHOT_D>(inner_point, BasisMode::Lagrange)
            .expect("inner opening point should match ring dimension");
    (folded_ring * packed_inner.sigma_m1()).coefficients()[0]
}
