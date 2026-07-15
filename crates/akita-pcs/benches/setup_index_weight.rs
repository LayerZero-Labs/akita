#![allow(missing_docs)]

use akita_algebra::eq_poly::EqPolynomial;
use akita_field::Prime128OffsetA7F7;
use akita_types::{
    gadget_row_scalars, r_decomp_levels, CommitmentRingDims, LevelParams, OpeningClaimsLayout,
    RelationMatrixRowLayout, SetupContributionGroupInputs, SetupContributionPlan,
    SetupContributionPlanInputs, SetupIndexWeightEvaluator, SisModulusProfileId, WitnessLayout,
};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion,
    SamplingMode,
};
use std::time::Duration;

type F = Prime128OffsetA7F7;
const D: usize = 64;

struct SetupIndexWeightBenchCase {
    plan: SetupContributionPlan<F>,
    evaluator: SetupIndexWeightEvaluator<F>,
    rho: Vec<F>,
    alpha: F,
}

fn test_scalar(value: u128) -> F {
    F::from_canonical_u128(value)
}

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(20);
    group.nresamples(1001);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
}

fn make_case(live_fold_count: usize, blocks_per_chunk: usize) -> SetupIndexWeightBenchCase {
    assert!(live_fold_count.is_power_of_two());
    assert!(blocks_per_chunk.is_power_of_two());
    assert!(blocks_per_chunk <= live_fold_count);
    assert_eq!(live_fold_count % blocks_per_chunk, 0);

    let num_claims = 2;
    let depth_open = 2;
    let depth_commit = 2;
    let fold_position_count = 8;
    let n_a = 2;
    let n_b = 2;
    let n_d = 2;
    let log_basis = 4;
    let mut level_params = LevelParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        log_basis,
        n_a,
        n_b,
        n_d,
        akita_challenges::SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(
        fold_position_count,
        live_fold_count * fold_position_count,
        depth_commit,
        depth_open,
    )
    .unwrap();
    level_params.shard_granule = 1;
    let depth_fold = level_params
        .num_digits_fold(num_claims, level_params.field_bits_for_cache())
        .unwrap();
    let opening_batch = OpeningClaimsLayout::new(0, num_claims).unwrap();
    let z_range = fold_position_count * depth_commit;
    let layout = WitnessLayout::new(
        &level_params,
        &opening_batch,
        live_fold_count / blocks_per_chunk,
        1 + n_a + n_b + n_d,
        r_decomp_levels::<F>(log_basis),
    )
    .unwrap();

    let rows = 1 + n_a + n_b + n_d;
    let tau1 = (0..3)
        .map(|idx| test_scalar(31 + idx as u128))
        .collect::<Vec<_>>();
    let inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
        rows,
        n_a,
        n_b,
        n_d,
        num_groups: 1,
        num_polys_per_group: vec![num_claims],
        num_t_vectors: num_claims,
        num_claims,
        live_fold_count,
        fold_position_count,
        depth_open,
        depth_commit,
        depth_fold,
        inner_width: z_range,
        eq_tau1: EqPolynomial::evals(&tau1).unwrap().into(),
    };
    let opening_source_len = layout.total_len();
    let groups = vec![SetupContributionGroupInputs {
        group_id: 0,
        e_col_offset: 0,
        num_claims,
        live_fold_count,
        fold_position_count,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        n_a,
        n_b,
        t_cols_per_vector: n_a * depth_open * live_fold_count,
        a_row_start: 1,
        b_row_start: 1 + n_a,
        layout: std::sync::Arc::new(layout),
        opening_source_len,
    }];
    let static_plan = SetupContributionPlan::prepare_static(
        &inputs,
        &groups,
        rows - n_d,
        n_d,
        num_claims * live_fold_count * depth_open,
    )
    .unwrap();
    let full_vec_randomness = (0..24)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let alpha = test_scalar(3);
    let plan = SetupContributionPlan::finish_plan::<F>(
        &static_plan,
        &full_vec_randomness,
        None,
        None,
        Some(&fold_gadget),
        &groups,
        CommitmentRingDims::uniform(D),
    )
    .unwrap();
    let evaluator = SetupIndexWeightEvaluator::new::<F>(
        &inputs,
        &plan,
        &groups,
        &tau1,
        &full_vec_randomness,
        &fold_gadget,
        alpha,
    )
    .unwrap();
    let rho_bits = evaluator.required().next_power_of_two().trailing_zeros() as usize;
    let rho = (0..rho_bits)
        .map(|idx| test_scalar(901 + idx as u128))
        .collect::<Vec<_>>();

    let packed = plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap();
    let succinct = evaluator.evaluate(&rho).unwrap();
    assert_eq!(succinct, packed);

    SetupIndexWeightBenchCase {
        plan,
        evaluator,
        rho,
        alpha,
    }
}

fn bench_setup_index_weight(c: &mut Criterion) {
    let mut group = c.benchmark_group("setup_index_weight_mle");
    configure_group(&mut group);

    for live_fold_count in [64usize, 256, 1024, 4096, 16384] {
        for (layout, blocks_per_chunk) in [
            ("single_chunk", live_fold_count),
            ("chunk64", 64usize.min(live_fold_count)),
        ] {
            let case = make_case(live_fold_count, blocks_per_chunk);
            group.bench_with_input(
                BenchmarkId::new(format!("{layout}/packed_path"), live_fold_count),
                &case,
                |b, case| {
                    b.iter(|| {
                        black_box(
                            case.plan
                                .evaluate_setup_index_weight_mle(
                                    black_box(&case.rho),
                                    black_box(case.alpha),
                                )
                                .unwrap(),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{layout}/succinct_path"), live_fold_count),
                &case,
                |b, case| {
                    b.iter(|| black_box(case.evaluator.evaluate(black_box(&case.rho)).unwrap()))
                },
            );
        }
    }

    group.finish();
}

criterion_group!(setup_index_weight, bench_setup_index_weight);
criterion_main!(setup_index_weight);
