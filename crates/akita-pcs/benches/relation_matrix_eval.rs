#![allow(missing_docs)]

use akita_algebra::CyclotomicRing;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_pcs::{AkitaCommitmentScheme, Transcript};
use akita_prover::backend::DenseView;
use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource};
use akita_prover::{
    build_relation_weight_events, ComputeBackendSetup, CpuBackend, DensePoly, ProverOpeningData,
    RelationSetupSource, RelationWeightEventInputs, RingRelationProver,
};
use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
use akita_transcript::AkitaTranscript;
use akita_types::{
    ring_opening_point_from_field, AjtaiKeyParams, AkitaCommitmentHint, AkitaExpandedSetup,
    BasisMode, Commitment, CommitmentRingDims, LevelParams, OpeningClaims, PointVariableSelection,
    PolynomialGroupClaims, RelationMatrixRowLayout, RingMultiplierOpeningPoint,
    RingRelationInstance, RingVec,
};
use akita_verifier::{
    prepare_relation_matrix_evaluator, RelationMatrixEvaluator, RingSwitchReplay,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use std::time::Duration;

type F = fp128::Field;
type Cfg = fp128::D128Full;
const D: usize = Cfg::D;
const NV: usize = 16;

struct EvalFixture {
    setup: Arc<AkitaExpandedSetup<F>>,
    evaluator: RelationMatrixEvaluator<F>,
    point: Vec<F>,
    alpha: F,
}

impl EvalFixture {
    fn evaluate(&self) -> F {
        self.evaluator
            .eval_flat_at_point::<F, D>(&self.point, self.setup.as_ref(), self.alpha, None)
            .expect("relation evaluation")
    }
}

fn prover_block_claims<'a, P>(
    point: &'a [F],
    polynomials: &'a [&'a P],
    commitment: &'a Commitment<F>,
    hint: AkitaCommitmentHint<F>,
) -> ProverOpeningData<'a, F, P, F> {
    let group = PolynomialGroupClaims::new(
        PointVariableSelection::prefix(point.len(), point.len()).expect("full-point group"),
        vec![F::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("prover group");
    let claims = OpeningClaims::from_groups(point.to_vec(), vec![group]).expect("opening claims");
    ProverOpeningData::new(claims, vec![hint], vec![polynomials]).expect("opening data")
}

fn retarget_key(key: &AjtaiKeyParams, ring_dim: usize) -> AjtaiKeyParams {
    let column_scale = D.checked_div(ring_dim).expect("nested role dimension");
    AjtaiKeyParams::new_unchecked(
        key.security_policy(),
        key.sis_table_key().table_digest,
        key.sis_modulus_profile(),
        key.sis_table_key().role,
        key.row_len(),
        key.col_len()
            .checked_mul(column_scale)
            .expect("role column width"),
        key.coeff_linf_bound(),
        ring_dim,
    )
}

fn prepare_fixture(
    setup: &Arc<AkitaExpandedSetup<F>>,
    instance: &RingRelationInstance<F>,
    level_params: &LevelParams,
    outgoing_ring_dim: usize,
    alpha: F,
    tau1: &[F],
) -> EvalFixture {
    let witness_layout = instance
        .segment_layout(level_params, None)
        .expect("witness layout");
    let opening_source_len = witness_layout
        .total_len()
        .checked_mul(D)
        .and_then(|coefficients| coefficients.checked_div(outgoing_ring_dim))
        .expect("opening source length");
    let events = build_relation_weight_events(RelationWeightEventInputs {
        setup: RelationSetupSource::Matrix(setup),
        instance,
        alpha,
        level_params,
        relation_row_point: tau1,
        claim_coefficients: &[F::one()],
        relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
        opening_source_len,
        opening_ring_dim: outgoing_ring_dim,
    })
    .expect("relation events");
    let dense = events.materialize_dense().expect("dense relation oracle");
    let mut point_rng = StdRng::seed_from_u64(0x706f_696e_742d_6d6c);
    let point = (0..dense.len().trailing_zeros() as usize)
        .map(|_| F::from_canonical_u128_reduced(point_rng.gen::<u128>()))
        .collect::<Vec<_>>();
    let claim_coefficients = [F::one()];
    let replay = RingSwitchReplay {
        setup,
        relation: instance,
        row_coefficients: &claim_coefficients,
        lp: level_params,
        opening_source_len,
        opening_ring_dim: outgoing_ring_dim,
    };
    let evaluator = prepare_relation_matrix_evaluator::<F, F, D>(&replay, alpha, tau1, None)
        .expect("relation evaluator");
    let expected = akita_sumcheck::multilinear_eval(&dense, &point).expect("dense relation MLE");
    let fixture = EvalFixture {
        setup: setup.clone(),
        evaluator,
        point,
        alpha,
    };
    assert_eq!(fixture.evaluate(), expected, "benchmark fixture parity");
    fixture
}

fn fixtures() -> [(&'static str, EvalFixture); 3] {
    let opening_batch =
        akita_types::OpeningClaimsLayout::new(NV, 1).expect("singleton opening batch");
    let level_params =
        Cfg::get_params_for_batched_commitment(&opening_batch).expect("commitment layout");
    let mut rng = StdRng::seed_from_u64(0x7265_6c61_7469_6f6e);
    let evals = (0..(1usize << NV))
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect::<Vec<_>>();
    let polynomial = DensePoly::<F>::from_field_evals(NV, D, &evals).expect("polynomial");
    let opening_point = (0..NV)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect::<Vec<_>>();

    let prover_setup = AkitaCommitmentScheme::<Cfg>::setup_prover(NV, 1).expect("setup");
    let prepared_setup = CpuBackend
        .prepare_setup(&prover_setup)
        .expect("prepared setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend,
        &prepared_setup,
        prover_setup.expanded.as_ref(),
    )
    .expect("prover stack");
    let (commitment, hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
        &prover_setup,
        std::slice::from_ref(&polynomial),
        &stack,
    )
    .expect("commitment");
    let outer_point = &opening_point[D.trailing_zeros() as usize..];
    let ring_opening_point = ring_opening_point_from_field(
        outer_point,
        level_params.num_positions_per_block,
        level_params.num_live_blocks,
        BasisMode::Lagrange,
    )
    .expect("ring opening point");
    let multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
    let opening = OpeningFoldKernel::<DenseView<'_, F, D>, F, D>::evaluate_and_fold(
        &CpuBackend,
        None,
        polynomial.opening_view().expect("opening view"),
        OpeningFoldPlan::Base {
            live_block_weights: &ring_opening_point.live_block_weights,
            position_weights: &ring_opening_point.position_weights,
            num_positions_per_block: level_params.num_positions_per_block,
        },
    )
    .expect("opening fold");
    let mut transcript = AkitaTranscript::<F>::new(b"relation-matrix-eval-bench");
    commitment
        .append_to_transcript(ABSORB_COMMITMENT, D, &mut transcript)
        .expect("commitment transcript");
    for coordinate in &opening_point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, coordinate);
    }
    let operation = akita_prover::OperationCtx::new(
        &CpuBackend,
        &prepared_setup,
        prover_setup.expanded.as_ref(),
    )
    .expect("operation context");
    let polynomial_refs = [&polynomial];
    let block_claims = prover_block_claims(&opening_point, &polynomial_refs, &commitment, hint);
    let (uniform_instance, _witness) =
        RingRelationProver::new::<F, F, _, DensePoly<F>, CpuBackend, CpuBackend>(
            &operation,
            &operation,
            ring_opening_point,
            multiplier_point,
            block_claims,
            vec![RingVec::from_ring_elems(&opening.folded)],
            level_params.clone(),
            &mut transcript,
            RingVec::from_single(&CyclotomicRing::<F, D>::one()),
            RelationMatrixRowLayout::WithDBlock,
            None,
        )
        .expect("uniform relation fixture");

    let alpha = F::from_u64(42);
    let row_variables = level_params
        .relation_row_index_num_vars_for_layout(RelationMatrixRowLayout::WithDBlock, &opening_batch)
        .expect("relation row variables");
    let tau1 = (0..row_variables)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect::<Vec<_>>();
    let uniform = prepare_fixture(
        &prover_setup.expanded,
        &uniform_instance,
        &level_params,
        D,
        alpha,
        &tau1,
    );
    let lane_factored = prepare_fixture(
        &prover_setup.expanded,
        &uniform_instance,
        &level_params,
        32,
        alpha,
        &tau1,
    );

    let role_dims = CommitmentRingDims {
        inner: D,
        outer: 64,
        opening: 32,
    };
    let mut mixed_params = level_params.clone();
    mixed_params.b_key = retarget_key(&mixed_params.b_key, role_dims.d_b());
    mixed_params.d_key = retarget_key(&mixed_params.d_key, role_dims.d_d());
    mixed_params.stamp_role_dims_from_keys();
    let group_opening_points = (0..opening_batch.num_groups())
        .map(|group| uniform_instance.group_opening_point(group).cloned())
        .collect::<Result<Vec<_>, _>>()
        .expect("group opening points");
    let group_multiplier_points = (0..opening_batch.num_groups())
        .map(|group| uniform_instance.group_ring_multiplier_point(group).cloned())
        .collect::<Result<Vec<_>, _>>()
        .expect("group multiplier points");
    let rhs_layout = akita_types::relation_rhs_layout_for(
        &mixed_params,
        &opening_batch,
        uniform_instance.relation_matrix_row_layout(),
    )
    .expect("mixed RHS layout");
    let rhs_len =
        akita_types::relation_rhs_coeff_len(role_dims, &rhs_layout).expect("mixed RHS length");
    let mixed_instance = RingRelationInstance::new(
        uniform_instance.relation_matrix_row_layout(),
        uniform_instance.group_challenges().to_vec(),
        group_opening_points,
        group_multiplier_points,
        opening_batch,
        uniform_instance.gamma().to_vec(),
        uniform_instance.row_coefficient_rings().clone(),
        RingVec::from_coeffs(vec![F::zero(); rhs_len]),
        RingVec::from_coeffs(vec![
            F::zero();
            mixed_params.d_key.row_len() * role_dims.d_d()
        ]),
        role_dims,
    )
    .expect("mixed relation fixture");
    let mixed = prepare_fixture(
        &prover_setup.expanded,
        &mixed_instance,
        &mixed_params,
        32,
        alpha,
        &tau1,
    );

    [
        ("U_uniform_128_to_128", uniform),
        ("L_uniform_128_to_32", lane_factored),
        ("M_mixed_128_64_32_to_32", mixed),
    ]
}

fn bench_relation_matrix_eval(c: &mut Criterion) {
    let fixtures = fixtures();
    let mut group = c.benchmark_group("verifier/relation_matrix_eval");
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(15));
    for (name, fixture) in &fixtures {
        group.bench_function(*name, |b| {
            b.iter(|| black_box(fixture.evaluate()));
        });
    }
    group.finish();
}

criterion_group!(relation_matrix_eval, bench_relation_matrix_eval);
criterion_main!(relation_matrix_eval);
