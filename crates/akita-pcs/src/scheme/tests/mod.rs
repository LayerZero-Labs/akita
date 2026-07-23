use super::*;
use akita_config::proof_optimized::fp128;
use akita_config::test_support::akita_batched_root_layout;
use akita_config::{CommitmentConfig, PrecommittedCommitmentConfig};
use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource, RootPolyShape};
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_prover::{DensePoly, OneHotPoly, ProverOpeningData};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::CommittedGroupParams;
use akita_types::DigitRangePlan;
use akita_types::ExtensionOpeningReductionProof;
use akita_types::{
    lagrange_weights, monomial_weights, reduce_inner_opening_to_ring_element,
    ring_opening_point_from_field,
};
use akita_types::{
    AkitaBatchedProofShape, LevelProofShape, NextWitnessBindingShape, RingVec,
    TerminalLevelProofShape,
};
use akita_types::{
    AkitaCommitmentHint, Commitment, OpeningClaims, OpeningClaimsLayout, PointVariableSelection,
    PolynomialGroupClaims,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
type Cfg = fp128::D64Full;
type F = fp128::Field;
const D: usize = Cfg::D;
type Scheme = AkitaCommitmentScheme<Cfg>;

type OneHotF = fp128::Field;
type OneHotCfg = fp128::D64OneHot;
type PrecommittedOneHotCfg = PrecommittedCommitmentConfig<OneHotCfg>;
const ONEHOT_D: usize = OneHotCfg::D;
// `fp128::D64OneHot` requires K=256 one-hot schedules (must match
// `OneHotCfg::onehot_chunk_size()`); chunks span `K/D = 4` ring elements.
const BENCH_ONEHOT_K: usize = 256;
type OneHotScheme = AkitaCommitmentScheme<OneHotCfg>;
type PrecommittedOneHotScheme = AkitaCommitmentScheme<PrecommittedOneHotCfg>;
/// Minimum w vector length (in field elements) below which further folding
/// is not beneficial.  When `w.len() <= MIN_W_LEN_FOR_FOLDING`, the prover
/// sends `w` directly instead of recursing.
const MIN_W_LEN_FOR_FOLDING: usize = 4096;

mod batched;
mod fp32_ext4;
mod layout;
mod onehot;
mod single;

fn batched_shape_rounds(level_d: usize, output_witness_len: usize) -> usize {
    let num_ring_elems = output_witness_len / level_d;
    num_ring_elems.next_power_of_two().trailing_zeros() as usize + level_d.trailing_zeros() as usize
}

/// Batched recursion already consults the byte planner before folding
/// again. The runtime safety guard here only needs to catch tiny tails and
/// fixed points, not enforce the single-proof shrink-ratio heuristic.
fn should_stop_batched_folding(witness_len: usize, prev_w_len: usize) -> bool {
    witness_len <= MIN_W_LEN_FOR_FOLDING || witness_len >= prev_w_len
}

/// Derive the structural proof shape from the schedule. The terminal carries
/// only optional EOR, its grind nonce, and the clear terminal response.
fn expected_same_point_batched_shape(
    max_num_vars: usize,
    num_claims: usize,
    _proof: &AkitaBatchedProof<OneHotF, OneHotF>,
) -> AkitaBatchedProofShape {
    let opening_batch =
        akita_types::OpeningClaimsLayout::new(max_num_vars, num_claims).expect("opening_batch");
    let schedule =
        OneHotCfg::get_params_for_prove(&opening_batch).expect("batched root runtime plan");
    let root_step = &schedule.root;
    let root_params = &root_step.params.final_group.commitment;
    let num_fold_levels = schedule.num_fold_levels();
    let root_rounds = batched_shape_rounds(root_params.d_a(), root_step.output_witness_len);

    assert!(
        num_fold_levels >= 2,
        "folded-only schedules have a root and terminal fold"
    );

    let root_successor = schedule.recursive_folds.first();
    let root_shape = LevelProofShape {
        extension_opening_reduction: None,
        v_coeffs: root_step.params.open_commit_matrix.output_rank()
            * root_step.params.open_commit_matrix.ring_dimension(),
        stage1_stages: DigitRangePlan::new(1usize << root_params.log_basis_open)
            .expect("scheduled root range basis")
            .stage_shapes(root_rounds),
        stage2_sumcheck_proof: vec![3; root_rounds],
        stage3_sumcheck: None,
        next_witness_binding: match root_successor {
            Some(successor) => {
                let next_level_params = &successor.params.witness;
                NextWitnessBindingShape::OuterCommitment {
                    coeffs: next_level_params.outer_commit_matrix.output_rank()
                        * next_level_params.outer_commit_matrix.ring_dimension(),
                }
            }
            None => NextWitnessBindingShape::TerminalInnerState,
        },
    };
    // After Phase 1, the recursive suffix has `num_fold_levels - 1` steps in
    // total: `num_fold_levels - 2` intermediate steps followed by exactly one
    // terminal step. (We've already consumed the root.)
    let mut recursive_folds = Vec::with_capacity(schedule.recursive_folds.len());
    let mut input_witness_len = root_step.output_witness_len;
    for (index, step) in schedule.recursive_folds.iter().enumerate() {
        assert_eq!(step.input_witness_len, input_witness_len);
        let level_params = &step.params.witness;
        let output_witness_len = step.output_witness_len;
        let rounds = batched_shape_rounds(level_params.d_a(), output_witness_len);
        recursive_folds.push(LevelProofShape {
            extension_opening_reduction: None,
            v_coeffs: step.params.open_commit_matrix.output_rank()
                * step.params.open_commit_matrix.ring_dimension(),
            stage1_stages: DigitRangePlan::new(1usize << level_params.log_basis_open)
                .expect("scheduled range basis")
                .stage_shapes(rounds),
            stage2_sumcheck_proof: vec![3; rounds],
            stage3_sumcheck: None,
            next_witness_binding: match schedule.recursive_folds.get(index + 1) {
                Some(successor) => {
                    let next_level_params = &successor.params.witness;
                    NextWitnessBindingShape::OuterCommitment {
                        coeffs: next_level_params.outer_commit_matrix.output_rank()
                            * next_level_params.outer_commit_matrix.ring_dimension(),
                    }
                }
                None => NextWitnessBindingShape::TerminalInnerState,
            },
        });
        input_witness_len = output_witness_len;
    }

    // Terminal fold step (always present in the multi-fold case); the
    // structural terminal field encodes its witness shape.
    assert_eq!(schedule.terminal.input_witness_len, input_witness_len);
    let terminal = TerminalLevelProofShape {
        extension_opening_reduction: None,
        terminal_response: schedule.terminal.params.response_shape.clone(),
    };

    AkitaBatchedProofShape {
        root: root_shape,
        recursive_folds,
        terminal,
    }
}

fn prover_claims<'a, E: FieldCore, P, CommitF: FieldCore>(
    point: &'a [E],
    polynomials: &'a [&'a P],
    commitment: &'a Commitment<CommitF>,
    hint: AkitaCommitmentHint<CommitF>,
) -> ProverOpeningData<'a, E, P, CommitF> {
    let group = PolynomialGroupClaims::new(
        PointVariableSelection::prefix(point.len(), point.len()).expect("full-point prover group"),
        vec![E::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims =
        OpeningClaims::from_groups(point.to_vec(), vec![group]).expect("valid prover claims");
    ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
        .expect("valid prover opening data")
}

fn verifier_claims<'a, E: FieldCore, C>(
    point: &[E],
    openings: &[E],
    commitment: &'a C,
) -> OpeningClaims<'static, E, &'a C> {
    OpeningClaims::from_groups(
        point.to_vec(),
        vec![PolynomialGroupClaims::new(
            PointVariableSelection::prefix(point.len(), point.len()).expect("full-point group"),
            openings.to_vec(),
            commitment,
        )
        .expect("valid verifier claims group")],
    )
    .expect("valid verifier claims")
}

fn make_dense_poly(num_vars: usize) -> (DensePoly<F>, Vec<F>) {
    let len = 1usize << num_vars;
    let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, D, &evals).unwrap();
    (poly, evals)
}

fn singleton_layout<C: CommitmentConfig>(num_vars: usize) -> CommittedGroupParams {
    let opening_batch = OpeningClaimsLayout::new(num_vars, 1).expect("singleton opening batch");
    C::get_params_for_batched_commitment(&opening_batch).expect("singleton commitment layout")
}

type VerifyFixture = (
    AkitaVerifierSetup<F>,
    Commitment<F>,
    AkitaBatchedProof<F, F>,
    Vec<F>,
    F,
    CommittedGroupParams,
);

fn make_verify_fixture(num_vars: usize) -> VerifyFixture {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(num_vars);
    let full_num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;

    let (poly, evals) = make_dense_poly(full_num_vars);
    let setup = Scheme::setup_prover(full_num_vars, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
    let (commitment, hint) =
        Scheme::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack).unwrap();

    let opening_point: Vec<F> = (0..full_num_vars)
        .map(|i| F::from_u64((i + 2) as u64))
        .collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let proof = Scheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&opening_point[..], &poly_refs[..], &commitments[0], hint),
        &stack,
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

fn debug_make_onehot_poly(layout: &CommittedGroupParams, seed: u64) -> OneHotPoly<OneHotF, u8> {
    let total_ring = layout.num_live_blocks * layout.num_positions_per_block;
    let num_vars = layout.position_index_bits()
        + layout.block_index_bits()
        + ONEHOT_D.trailing_zeros() as usize;
    // `total_ring` ring elements of degree D cover `2^num_vars` field elements,
    // grouped into `2^num_vars / K` one-hot chunks.
    let total_field = total_ring * ONEHOT_D;
    assert_eq!(total_field, 1usize << num_vars);
    let total_chunks = total_field / BENCH_ONEHOT_K;

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..BENCH_ONEHOT_K) as u8))
        .collect();

    OneHotPoly::<OneHotF, u8>::new(BENCH_ONEHOT_K, ONEHOT_D, indices).expect("debug onehot poly")
}

fn opening_from_poly<'a, P>(
    poly: &'a P,
    point: &[OneHotF],
    layout: &CommittedGroupParams,
) -> OneHotF
where
    P: RootOpeningSource<OneHotF, ONEHOT_D> + RootPolyShape<OneHotF, ONEHOT_D>,
    CpuBackend: OpeningFoldKernel<P::OpeningView<'a>, OneHotF, ONEHOT_D>,
{
    let alpha_bits = ONEHOT_D.trailing_zeros() as usize;
    assert_eq!(
        point.len(),
        alpha_bits + layout.position_index_bits() + layout.block_index_bits()
    );

    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.num_positions_per_block,
        layout.num_live_blocks,
        BasisMode::Lagrange,
    )
    .expect("opening point shape should match layout");

    let opening = OpeningFoldKernel::<P::OpeningView<'a>, OneHotF, ONEHOT_D>::evaluate_and_fold(
        &CpuBackend,
        None,
        poly.opening_view().expect("opening view"),
        OpeningFoldPlan::Base {
            live_block_weights: &ring_opening_point.live_block_weights,
            position_weights: &ring_opening_point.position_weights,
            num_positions_per_block: layout.num_positions_per_block,
        },
    )
    .expect("evaluate_and_fold");
    let folded_ring = opening.eval;
    let packed_inner =
        reduce_inner_opening_to_ring_element::<OneHotF, ONEHOT_D>(inner_point, BasisMode::Lagrange)
            .expect("inner opening point should match ring dimension");
    (folded_ring * packed_inner.sigma_m1()).coefficients()[0]
}
