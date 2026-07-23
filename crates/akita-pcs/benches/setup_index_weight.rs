#![allow(missing_docs)]

use akita_algebra::eq_poly::EqPolynomial;
use akita_field::Prime128OffsetA7F7;
use akita_types::{
    r_decomp_levels, CommitmentRingDims, CommittedGroupParams, OpeningClaimsLayout,
    SetupContributionPlan, SisModulusProfileId, WitnessLayout,
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

fn make_case(num_live_blocks: usize, blocks_per_chunk: usize) -> SetupIndexWeightBenchCase {
    assert!(num_live_blocks.is_power_of_two());
    assert!(blocks_per_chunk.is_power_of_two());
    assert!(blocks_per_chunk <= num_live_blocks);
    assert_eq!(num_live_blocks % blocks_per_chunk, 0);

    let num_claims = 2;
    let depth_open = 2;
    let depth_commit = 2;
    let num_positions_per_block = 8;
    let n_a = 2;
    let n_b = 2;
    let n_d = 2;
    let log_basis = 4;
    let level_params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        log_basis,
        n_a,
        n_b,
        n_d,
        akita_challenges::SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(
        num_positions_per_block,
        num_live_blocks * num_positions_per_block,
        depth_commit,
        depth_open,
        depth_open,
    )
    .unwrap();
    let opening_batch = OpeningClaimsLayout::new(0, num_claims).unwrap();
    let layout = WitnessLayout::new(
        &level_params,
        &opening_batch,
        num_live_blocks / blocks_per_chunk,
        1 + n_a + n_b + n_d,
        r_decomp_levels::<F>(log_basis),
    )
    .unwrap();

    let tau1 = (0..3)
        .map(|idx| test_scalar(31 + idx as u128))
        .collect::<Vec<_>>();
    let eq_tau1 = EqPolynomial::evals(&tau1).unwrap().into();
    let full_vec_randomness = (0..24)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let alpha = test_scalar(3);
    let plan = SetupContributionPlan::prepare::<F>(
        &level_params,
        &opening_batch,
        eq_tau1,
        &layout,
        &full_vec_randomness,
        CommitmentRingDims::uniform(D),
        D,
    )
    .unwrap();
    let rho_bits = plan.required().next_power_of_two().trailing_zeros() as usize;
    let rho = (0..rho_bits)
        .map(|idx| test_scalar(901 + idx as u128))
        .collect::<Vec<_>>();

    let _ = plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap();

    SetupIndexWeightBenchCase { plan, rho, alpha }
}

fn bench_setup_index_weight(c: &mut Criterion) {
    let mut group = c.benchmark_group("setup_index_weight_mle");
    configure_group(&mut group);

    for num_live_blocks in [64usize, 256, 1024, 4096, 16384] {
        for (layout, blocks_per_chunk) in [
            ("single_chunk", num_live_blocks),
            ("chunk64", 64usize.min(num_live_blocks)),
        ] {
            let case = make_case(num_live_blocks, blocks_per_chunk);
            group.bench_with_input(
                BenchmarkId::new(format!("{layout}/plan_point_path"), num_live_blocks),
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
        }
    }

    group.finish();
}

criterion_group!(setup_index_weight, bench_setup_index_weight);
criterion_main!(setup_index_weight);
