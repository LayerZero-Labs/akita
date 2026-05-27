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
fn recursive_w_commit_layout_rejects_unsupported_ring_dimension() {
    let params = LevelParams::params_only(
        akita_types::SisModulusFamily::Q128,
        42,
        3,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![1],
        },
    );
    let err = recursive_w_commit_layout_for_d::<Cfg>(42, &params, 64).unwrap_err();
    assert!(
        matches!(err, AkitaError::InvalidInput(message) if message.contains("unsupported ring dimension: 42"))
    );
}

#[test]
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

#[test]
#[cfg(not(feature = "zk"))]
fn batched_commit_matches_individual_commits() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 1) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 7) as u64)).collect();
    let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let poly_groups = [std::slice::from_ref(&poly_a), std::slice::from_ref(&poly_b)];

    let (batched_commitments, batched_hints): (Vec<_>, Vec<_>) = poly_groups
        .iter()
        .map(|group| {
            <Scheme as CommitmentProver<F, D>>::commit(&setup, &CpuBackend, &prepared, group)
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .unzip();
    let (commitment_a, hint_a) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly_a),
    )
    .unwrap();
    let (commitment_b, hint_b) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly_b),
    )
    .unwrap();

    assert_eq!(batched_commitments, vec![commitment_a, commitment_b]);
    assert_eq!(batched_hints, vec![hint_a, hint_b]);
}

/// Exercise the batched root-direct fast path: for a layout/batch shape
/// whose offline-planned schedule has zero fold levels, the prover must
/// emit a [`AkitaBatchedRootProof::Direct`] variant with no recursive
/// suffix, and the verifier must accept it via the batched root-direct
/// checks (per-claim opening + joint per-group re-commit).
#[test]
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

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NUM_VARS, NUM_POLYS, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(&setup, &CpuBackend, &prepared, &poly_refs)
            .unwrap();
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
        &CpuBackend,
        &prepared,
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

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NUM_VARS, NUM_POLYS, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(&setup, &CpuBackend, &prepared, &poly_refs)
            .unwrap();
    let commitments = [commitment];
    let hints = vec![hint];

    let opening_point: Vec<F> = (0..NUM_VARS).map(|i| F::from_u64((i + 2) as u64)).collect();
    let openings: Vec<F> = (0..NUM_POLYS).map(|_| F::from_u64(999_999)).collect();

    let poly_group = [&polys[0], &polys[1], &polys[2], &polys[3]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/batched-root-direct-bad-opening");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
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
fn batched_verify_passes_for_consistent_openings() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 5) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 7 + 3) as u64)).collect();
    let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let poly_group = [&poly_a, &poly_b];
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(&setup, &CpuBackend, &prepared, &poly_group)
            .unwrap();
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
        &CpuBackend,
        &prepared,
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
        <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(NV, BATCH_SIZE, 1)
            .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup =
        <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_verifier(&setup);
    let (commitment, hint) = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        &poly_refs,
    )
    .expect("batched onehot commit");
    let commitments = [commitment];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    let proof = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
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
            assert_eq!(
                expected_root.stage2_sumcheck_proof,
                actual_root.stage2_sumcheck_proof
            );
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
fn batched_verify_rejects_wrong_opening() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 11) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 5 + 13) as u64)).collect();
    let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let poly_group = [&poly_a, &poly_b];
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(&setup, &CpuBackend, &prepared, &poly_group)
            .unwrap();
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
        &CpuBackend,
        &prepared,
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
fn batched_verify_rejects_batch_count_beyond_setup_capacity() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 17) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 19) as u64)).collect();
    let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let poly_group = [&poly_a, &poly_b];
    let (commitment, hint) =
        <Scheme as CommitmentProver<F, D>>::commit(&setup, &CpuBackend, &prepared, &poly_group)
            .unwrap();
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
        &CpuBackend,
        &prepared,
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
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;

    let (poly, evals) = make_dense_poly(num_vars);

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .unwrap();

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
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;

    let (poly, evals) = make_dense_poly(num_vars);

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .unwrap();

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
        .sumcheck_proof
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
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;

    let coeffs: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &coeffs).unwrap();

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .unwrap();

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

    let setup = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::setup_prover(num_vars, 1, 1)
        .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup =
        <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::setup_verifier(&setup);
    let (commitment, hint) = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .unwrap();

    let poly_refs: [&DensePoly<DirectF, DIRECT_D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<DirectF>::new(b"test/tiny-direct");
    let proof = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::batched_prove(
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

    fn stage1_challenge_config(
        _d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        Ok(akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        })
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        akita_types::SisModulusFamily::Q32
    }

    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
        Ok(None)
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

    fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
        (3, 3)
    }

    fn get_params_for_prove(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = akita_types::scale_batched_root_layout(
            &Self::root_lp(),
            incidence.num_claims(),
            Self::stage1_challenge_config(Self::D)
                .expect("stage1 challenge config")
                .l1_norm(),
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
                    commit_params: None,
                    // Stub fixture: terminal-direct level params equal the
                    // fold's `lp` (matches the deleted
                    // `Cfg::level_params_with_log_basis` override that
                    // returned `Self::root_lp()`).
                    level_params: Some(lp.clone()),
                }),
            ],
            total_bytes: 0,
        })
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

    fn stage1_challenge_config(
        _d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        Ok(akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        })
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        akita_types::SisModulusFamily::Q32
    }

    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
        Ok(None)
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

    fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
        (3, 3)
    }

    fn get_params_for_prove(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = akita_types::scale_batched_root_layout(
            &Self::root_lp(),
            incidence.num_claims(),
            Self::stage1_challenge_config(Self::D)
                .expect("stage1 challenge config")
                .l1_norm(),
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
                    commit_params: None,
                    // Stub fixture: terminal-direct level params equal the
                    // fold's `lp` (matches the deleted
                    // `Cfg::level_params_with_log_basis` override that
                    // returned `Self::root_lp()`).
                    level_params: Some(lp.clone()),
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

    let setup =
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let (commitment, hint) = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .unwrap();

    let poly_refs = [&poly];
    let commitments = [commitment];
    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
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

    let setup =
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 2, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let poly_refs = [&poly_a, &poly_b];
    let (commitment, hint) = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        &poly_refs,
    )
    .unwrap();
    let commitments = [commitment];
    let openings = [opening_a, opening_b];

    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-outer-direct");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
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

    let setup =
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 2, 2).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let poly_refs = [&poly];
    let (commitment, hint) = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        &poly_refs,
    )
    .unwrap();
    let commitments = [commitment];
    let openings_a = [opening_a];
    let openings_b = [opening_b];

    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-multipoint-direct");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
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

// Tier-specific Fp32TieredRootFoldCfg-driven E2E tests removed during the
// origin/main merge: they targeted the pre-#100 `CommitmentConfig` /
// `ScheduleProvider` / `PlannerConfig` trait surface. The production
// `fp128::D32OneHotFastVerify` preset retains end-to-end coverage through
// `crates/akita-pcs/examples/portable_bench.rs` and the regular cargo test
// suite. See `specs/tiered_commit.md` for the protocol description.
