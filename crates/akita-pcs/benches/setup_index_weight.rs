#![allow(missing_docs)]

use akita_algebra::eq_poly::EqPolynomial;
use akita_field::Prime128OffsetA7F7;
use akita_types::{
    gadget_row_scalars, CommitmentRingDims, RelationMatrixRowLayout, SetupContributionGroupInputs,
    SetupContributionPlan, SetupContributionPlanInputs, SetupIndexWeightEvaluator,
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

fn make_case(num_blocks: usize, blocks_per_chunk: usize) -> SetupIndexWeightBenchCase {
    assert!(num_blocks.is_power_of_two());
    assert!(blocks_per_chunk.is_power_of_two());
    assert!(blocks_per_chunk <= num_blocks);
    assert_eq!(num_blocks % blocks_per_chunk, 0);

    let num_claims = 2;
    let depth_open = 2;
    let depth_commit = 2;
    let depth_fold = 2;
    let block_len = 8;
    let n_a = 2;
    let n_b = 2;
    let n_d = 2;
    let log_basis = 4;
    let z_range = block_len * depth_commit;
    let e_len_per_chunk = num_claims * depth_open * blocks_per_chunk;
    let t_len_per_chunk = n_a * num_claims * depth_open * blocks_per_chunk;
    let chunk_stride = z_range + e_len_per_chunk + t_len_per_chunk + 5;
    let chunks = (0..(num_blocks / blocks_per_chunk))
        .map(|idx| {
            let base = idx * chunk_stride + idx;
            let offset_e = base + z_range + 1;
            let offset_t = offset_e + e_len_per_chunk + 2;
            akita_types::WitnessChunkLayout {
                offset_z: base,
                offset_e,
                offset_t,
                offset_r: None,
                global_block_base: idx * blocks_per_chunk,
            }
        })
        .collect::<Vec<_>>();

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
        num_blocks,
        block_len,
        depth_open,
        depth_commit,
        depth_fold,
        inner_width: z_range,
        eq_tau1: EqPolynomial::evals(&tau1).unwrap(),
    };
    let groups = vec![SetupContributionGroupInputs {
        e_col_offset: 0,
        num_claims,
        num_blocks,
        block_len,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        n_a,
        n_b,
        t_cols_per_vector: n_a * depth_open * num_blocks,
        a_row_start: 1,
        b_row_start: 1 + n_a,
        blocks_per_chunk,
        chunks,
    }];
    let static_plan = SetupContributionPlan::prepare_static(
        &inputs,
        &groups,
        rows - n_d,
        n_d,
        num_claims * num_blocks * depth_open,
    )
    .unwrap();
    let full_vec_randomness = (0..24)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let plan = SetupContributionPlan::finish_plan::<F>(
        &static_plan,
        &full_vec_randomness,
        None,
        None,
        Some(&fold_gadget),
        &groups,
    )
    .unwrap();
    let evaluator = SetupIndexWeightEvaluator::new::<F>(
        &inputs,
        &static_plan,
        &groups,
        &tau1,
        &full_vec_randomness,
        &fold_gadget,
        D,
        CommitmentRingDims::uniform(D),
        test_scalar(3),
    )
    .unwrap();
    let rho_bits = evaluator.required().next_power_of_two().trailing_zeros() as usize;
    let rho = (0..rho_bits)
        .map(|idx| test_scalar(901 + idx as u128))
        .collect::<Vec<_>>();

    let packed = plan.evaluate_setup_index_weight_mle(&rho).unwrap();
    let succinct = evaluator.evaluate(&rho).unwrap().expect("supported layout");
    assert_eq!(succinct, packed);

    SetupIndexWeightBenchCase {
        plan,
        evaluator,
        rho,
    }
}

fn bench_setup_index_weight(c: &mut Criterion) {
    let mut group = c.benchmark_group("setup_index_weight_mle");
    configure_group(&mut group);

    for num_blocks in [64usize, 256, 1024, 4096, 16384] {
        for (layout, blocks_per_chunk) in [
            ("single_chunk", num_blocks),
            ("chunk64", 64usize.min(num_blocks)),
        ] {
            let case = make_case(num_blocks, blocks_per_chunk);
            group.bench_with_input(
                BenchmarkId::new(format!("{layout}/packed_path"), num_blocks),
                &case,
                |b, case| {
                    b.iter(|| {
                        black_box(
                            case.plan
                                .evaluate_setup_index_weight_mle(black_box(&case.rho))
                                .unwrap(),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{layout}/succinct_path"), num_blocks),
                &case,
                |b, case| {
                    b.iter(|| {
                        black_box(
                            case.evaluator
                                .evaluate(black_box(&case.rho))
                                .unwrap()
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(setup_index_weight, bench_setup_index_weight);
criterion_main!(setup_index_weight);
